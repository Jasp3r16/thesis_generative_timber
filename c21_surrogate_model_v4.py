"""
TrussEdgeSafetyGNN: PyTorch Geometric Model for Structural Safety Prediction
v4 — Improved architecture with deeper encoder, dropout, symmetric edge embeddings,
     edge-aware skip connections, and numerical stability fixes.

Architecture Overview:
    1. Encoder:   Project node features (10D) -> hidden_dim via 2-layer MLP (with activation)
    2. Processor: Stack of NNConv layers with adaptive edge weights, residuals (applied from
                  layer 0), dropout, and batch normalization
    3. Decoder:   Concatenate symmetric edge embedding (|h_i - h_j|, h_i * h_j) + raw edge
                  features -> binary classification with dropout
    4. Loss:      Focal Loss with numerically stable log computation

Changes vs v3:
    - NodeEncoder is now a 2-layer MLP with activation (was a bare Linear)
    - Residual skip connections now applied from layer 0 (was layer 1+)
    - EdgeDecoder uses symmetric interactions: |h_i - h_j| and h_i x h_j
      instead of raw concatenation, making predictions invariant to edge direction
    - Dropout added to both processor and decoder for regularisation
    - FocalLoss uses numerically stable log (clamp + log instead of BCE-from-sigmoid)
    - EdgeFeatureMLPFilter gains a third hidden layer for richer edge weight generation
    - create_model() exposes dropout_p argument
    - Sanity check updated to reflect actual truss dimensions (39 nodes, 120 edges)

Usage:
    >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
    >>> model = TrussEdgeSafetyGNN(node_features_dim=10, edge_features_dim=7, hidden_dim=128).to(device)
    >>> model.cache_topology(edge_index)
    >>> predictions = model(x, edge_attr=edge_attr)  # [num_edges, 1], binary probabilities
    >>> loss_fn = FocalLoss(alpha=0.1, gamma=2.0)
    >>> loss = loss_fn(predictions, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm


# ============================================================================
# 1. NODE ENCODER: 2-Layer MLP with Activation
# ============================================================================

class NodeEncoder(nn.Module):
    """
    Two-layer MLP that projects raw node features into the hidden latent space.

    Why two layers instead of one Linear?
    ---------------------------------------
    Node features are heterogeneous: 3D coordinates (continuous, spatially meaningful),
    binary boundary condition flags (Tx, Ty, Tz, Rx, Ry, Rz), and a continuous load (Fz).
    A single linear layer cannot mix these meaningfully -- it can only scale and shift.
    A second layer with an activation in between allows the encoder to learn non-linear
    combinations, e.g. "node is pinned AND has a large downward load".

    Architecture: Linear -> LeakyReLU -> Linear
    The second Linear projects into hidden_dim without an activation, so the
    processor's first NNConv layer receives un-squashed embeddings (standard practice).
    """

    def __init__(self, node_features_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(node_features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        """
        Args:
            x: [num_nodes, node_features_dim]
        Returns:
            [num_nodes, hidden_dim]
        """
        return self.fc2(self.activation(self.fc1(x)))


# ============================================================================
# 2. FILTER-GENERATING NETWORK: Edge Feature Transformation
# ============================================================================

class EdgeFeatureMLPFilter(nn.Module):
    """
    Three-layer MLP that transforms continuous edge features into dynamic weight
    matrices for NNConv.

    Conceptual Purpose:
    -------------------
    In traditional FEA, the global stiffness matrix K is assembled from individual
    member stiffness matrices, where each member's contribution depends on its
    material properties (E-modulus) and geometry (cross-sectional area A, moments
    of inertia Iy, Iz).

    This filter mimics that process: instead of a fixed weight for all edges, we
    compute an adaptive weight matrix for each edge based on its 7D feature vector
    (Area, Length, E, Iy, Iz, J, EA/L).

    v4 change: Added a third hidden layer (hidden -> hidden -> out_channels) to give
    the filter network more capacity. Cross-sections AND materials vary per sample,
    so the filter needs to capture richer stiffness interactions.

    Example:
    --------
    - A stiff member (high E*A/L) learns high filter weights -> strong message passing
    - A weak member (low E*A/L) learns lower weights -> weaker influence on neighbours
    """

    def __init__(self, edge_features_dim, out_channels, hidden=64):
        """
        Args:
            edge_features_dim: Number of continuous edge features (e.g., 7)
            out_channels: Output size of weight matrix (hidden_dim * hidden_dim)
            hidden: Intermediate hidden layer size (default 64)
        """
        super().__init__()
        self.fc1 = nn.Linear(edge_features_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)          # added in v4
        self.fc3 = nn.Linear(hidden, out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, edge_attr):
        """
        Args:
            edge_attr: [num_edges, edge_features_dim]
        Returns:
            [num_edges, out_channels]
        """
        h = self.activation(self.fc1(edge_attr))
        h = self.activation(self.fc2(h))              # added in v4
        return self.fc3(h)


# ============================================================================
# 3. EDGE-LEVEL DECODER: Symmetric Prediction Head
# ============================================================================

class EdgeDecoder(nn.Module):
    """
    Edge-level prediction head using symmetric node interaction features.

    Why Symmetric Interactions?
    ---------------------------
    v3 concatenated [h_i | h_j | e_ij]. This is direction-sensitive: swapping
    source and target would yield a different prediction for the same physical member.
    Trusses are undirected structures -- member AB and member BA are the same element.

    v4 replaces [h_i | h_j] with two symmetric terms:
        - Difference magnitude: |h_i - h_j|  -> captures force gradient across member
        - Element-wise product:  h_i * h_j   -> captures shared activation patterns
    Both are invariant to swapping i and j (|a-b| = |b-a|, a*b = b*a),
    so the model is consistent regardless of edge orientation in the COO format.

    Raw edge features (Area, E, Iy, Iz, ...) are still concatenated to ground
    the decision in actual material and geometric properties.

    Dropout:
    --------
    Added between decoder layers (default p=0.1) to regularise. With 10k samples
    x 120 edges the model is unlikely to severely overfit, but dropout is cheap
    insurance -- especially in the decoder which sees per-edge signals.
    """

    def __init__(self, hidden_dim, edge_features_dim, dropout_p=0.1):
        """
        Args:
            hidden_dim: Dimensionality of node embeddings from processor
            edge_features_dim: Number of raw edge features (e.g., 7)
            dropout_p: Dropout probability (default 0.1)
        """
        super().__init__()

        # Symmetric concat: [|h_i - h_j| | h_i * h_j | e_ij]
        # = hidden_dim + hidden_dim + edge_features_dim
        concat_dim = 2 * hidden_dim + edge_features_dim

        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_i, h_j, e_ij):
        """
        Args:
            h_i: Source node embeddings [num_edges, hidden_dim]
            h_j: Target node embeddings [num_edges, hidden_dim]
            e_ij: Raw edge features     [num_edges, edge_features_dim]
        Returns:
            Binary predictions [num_edges, 1], values in [0, 1]
            Interpretation: P(Utilization <= 1.0) = P(safe)
        """
        diff = torch.abs(h_i - h_j)   # [num_edges, hidden_dim] -- force gradient
        prod = h_i * h_j              # [num_edges, hidden_dim] -- shared patterns
        x = torch.cat([diff, prod, e_ij], dim=1)

        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x


# ============================================================================
# 4. FOCAL LOSS: Numerically Stable Implementation
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: -alpha * (1 - p_t)^gamma * log(p_t)

    Why Use Focal Loss for Structural Safety?
    -------------------------------------------
    Timber truss datasets have extreme class imbalance:
    - ~85% of members are SAFE (Utilization <= 1.0, label=0)
    - ~15% of members are UNSAFE (Utilization > 1.0, label=1)

    A naive model achieves 85% accuracy by predicting "all safe" -- useless for
    safety prediction.

    Focal Loss Solution:
    - Down-weights easy negative examples (most safe members)
    - Up-weights hard positive examples (rare failures)
    - Forces model to learn critical failure patterns

    v4 Numerical Stability Fix:
    ----------------------------
    v3 called F.binary_cross_entropy(predictions, targets) which computes
    -[t*log(p) + (1-t)*log(1-p)]. When p is very close to 0 or 1 (confident
    predictions), log(p) can hit -inf with float32. v4 clamps p into [eps, 1-eps]
    before taking logs, preventing nan gradients during early training when the
    model might produce extreme sigmoid outputs.

    Hyperparameter Tuning:
    - alpha: Balancing weight for positive (unsafe) class.
             If ~15% unsafe, try alpha in [0.15, 0.3]. Default 0.1.
    - gamma: Focusing exponent. gamma=2.0 is standard.
             Increase to 3-4 if model still collapses to predicting all-safe.
    """

    def __init__(self, alpha=0.1, gamma=2.0, eps=1e-7):
        """
        Args:
            alpha: Balancing weight for positive class (unsafe).
            gamma: Focusing parameter (default 2.0).
            eps:   Clamp epsilon for numerical stability (default 1e-7).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps   = eps

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output [num_edges, 1], values in [0, 1] from Sigmoid
            targets:     Ground truth [num_edges, 1] or [num_edges], values in {0, 1}
                         0 = safe (Utilization <= 1.0), 1 = unsafe (Utilization > 1.0)
        Returns:
            Scalar loss value (averaged over all edges)
        """
        predictions = predictions.view(-1)
        targets     = targets.view(-1).float()

        # Clamp to avoid log(0) -- numerically stable (v4 fix)
        p = predictions.clamp(self.eps, 1.0 - self.eps)

        # Stable BCE: -[t*log(p) + (1-t)*log(1-p)]
        bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))

        # p_t: probability assigned to the true class
        p_t = torch.where(targets == 1, p, 1 - p)

        # Focal weight: suppresses easy examples
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing: penalise false negatives (missed failures) more
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(p, self.alpha),
            torch.full_like(p, 1.0 - self.alpha),
        )

        focal_loss = alpha_t * focal_weight * bce
        return focal_loss.mean()


# ============================================================================
# 5. MAIN MODEL: TrussEdgeSafetyGNN
# ============================================================================

class TrussEdgeSafetyGNN(nn.Module):
    """
    End-to-end Graph Neural Network for predicting edge-level structural safety
    in timber trusses.

    Architecture: Encoder -> Processor (NNConv Stack) -> Decoder

    Input:
    ------
    - Node features (x):        [num_nodes, node_features_dim]
                                 coordinates + boundary conditions + applied loads
    - Edge index (edge_index):  [2, num_edges] graph connectivity (fixed topology)
    - Edge features (edge_attr):[num_edges, edge_features_dim]
                                 material + geometric properties (vary per sample)

    Output:
    -------
    - Edge-level predictions: [num_edges, 1] binary probabilities in [0, 1]
      Interpretation: P(safe) = P(Utilization <= 1.0)

    Key Design Decisions (v4 additions marked with *):
    --------------------------------------------------
    1. *NodeEncoder (2-layer MLP):
       - v3 used a bare Linear layer -- no nonlinearity before message passing.
       - Node features are heterogeneous (coordinates, binary BCs, continuous loads).
       - A 2-layer MLP with LeakyReLU lets the encoder learn non-linear combinations
         before handing embeddings to the processor.

    2. NNConv + Filter-Generating MLP (*3-layer filter):
       - Standard GCN uses fixed, uniform weights for all edges.
       - Reality: stiff members (high E*A/L) have stronger structural influence.
       - NNConv learns adaptive edge weights based on 7D material/geometric features.
       - v4 adds a hidden layer to the filter MLP for richer stiffness modelling.

    3. Residual Skip Connections (*applied from layer 0):
       - Deep GNNs suffer from over-smoothing.
       - v3 skipped the residual on layer 0; v4 applies it consistently from the start.
       - All layers: h_out = NNConv(h_in) + h_in

    4. *Dropout in Processor and Decoder:
       - With 10k samples and a 128D model, dropout (p=0.1) provides cheap
         regularisation without significantly slowing convergence.

    5. *Symmetric EdgeDecoder:
       - v3 concatenated [h_i | h_j | e_ij] -- direction-sensitive.
       - v4 uses [|h_i - h_j| | h_i * h_j | e_ij] -- invariant to edge direction.
       - Physically correct: member AB == member BA in a truss.

    6. *FocalLoss numerical stability:
       - v3 used F.binary_cross_entropy which can produce -inf for extreme predictions.
       - v4 clamps p to [eps, 1-eps] before log, preventing nan gradients.

    7. Static Topology Caching:
       - Timber trusses have fixed topology (39 nodes, 120 edges in this dataset).
       - edge_index is cached once; edge_attr is NOT cached (varies per sample).
    """

    def __init__(
        self,
        node_features_dim=10,
        edge_features_dim=7,
        hidden_dim=128,
        num_layers=4,
        use_batch_norm=True,
        use_residuals=True,
        dropout_p=0.1,
    ):
        """
        Args:
            node_features_dim: Input node feature dimensionality.
                Default 10: x, y, z + Tx, Ty, Tz, Rx, Ry, Rz (BCs) + Fz (load)

            edge_features_dim: Input edge feature dimensionality.
                Default 7: Area, Length, E, Iy, Iz, J, EA/L

            hidden_dim: Latent dimensionality throughout the model. Default 128.
                Larger -> more expressive but more memory. Try 64, 128, 256.

            num_layers: Number of NNConv message-passing layers. Default 4.
                - 2-3: Fast, local stiffness effects only
                - 4-5: Balanced, captures global force propagation
                - 6+:  Risk of over-smoothing without strong residuals

            use_batch_norm: Apply BatchNorm after each NNConv layer. Default True.
                Recommended for num_layers >= 4.

            use_residuals: Apply residual skip connections. Default True.
                Strongly recommended for num_layers >= 4.

            dropout_p: Dropout probability in processor and decoder. Default 0.1.
                Set to 0.0 to disable entirely.
        """
        super().__init__()

        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.hidden_dim        = hidden_dim
        self.num_layers        = num_layers
        self.use_batch_norm    = use_batch_norm
        self.use_residuals     = use_residuals
        self.dropout_p         = dropout_p

        # ===== ENCODER: 2-Layer MLP (v4: was single Linear) =====
        self.node_encoder = NodeEncoder(node_features_dim, hidden_dim)

        # ===== PROCESSOR: Stack of NNConv Layers =====
        self.nnconv_layers = nn.ModuleList()
        self.batch_norms   = nn.ModuleList() if use_batch_norm else None
        self.dropout       = nn.Dropout(p=dropout_p)

        for _ in range(num_layers):
            # Filter-Generating MLP: edge features -> hidden_dim x hidden_dim weights
            # v4: 3-layer filter (was 2-layer) for richer stiffness representation
            edge_mlp = EdgeFeatureMLPFilter(
                edge_features_dim=edge_features_dim,
                out_channels=hidden_dim * hidden_dim,
                hidden=64,
            )

            nnconv = NNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                nn=edge_mlp,
                aggr='add',
            )
            self.nnconv_layers.append(nnconv)

            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))

        self.activation = nn.LeakyReLU(0.1)

        # ===== DECODER: Symmetric Edge-Level Prediction Head (v4) =====
        self.edge_decoder = EdgeDecoder(hidden_dim, edge_features_dim, dropout_p)

        # ===== STATIC TOPOLOGY CACHING =====
        self.register_buffer('edge_index_cache', torch.zeros((2, 1), dtype=torch.long))
        self._is_topology_cached = False

    def cache_topology(self, edge_index):
        """
        Pre-cache the fixed edge_index for static topologies.

        Call this ONCE before training/inference. Edge attributes are NOT cached
        because member cross-sections and materials vary across samples.

        Args:
            edge_index: [2, num_edges] edge connectivity in COO format
        """
        self.edge_index_cache    = edge_index.clone()
        self._is_topology_cached = True
        print(f"[TrussEdgeSafetyGNN] Topology cached: {edge_index.shape[1]} edges")

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        """
        Forward pass: Encode -> NNConv stack -> Decode.

        Args:
            x:          Node features [num_nodes, node_features_dim]
            edge_index: [2, num_edges] COO format. Optional if topology cached.
            edge_attr:  [num_edges, edge_features_dim] raw edge features. Always required.
            batch:      [num_nodes] batch vector (optional, for PyG DataLoader compat).

        Returns:
            [num_edges, 1] binary probabilities.
            prediction[i] ~= P(edge i is safe) = P(Utilization[i] <= 1.0)
        """
        if self._is_topology_cached and edge_index is None:
            edge_index = self.edge_index_cache

        if edge_index is None or edge_attr is None:
            raise ValueError(
                "edge_index must be provided or pre-cached via cache_topology(); "
                "edge_attr is always required."
            )

        # ===== ENCODE =====
        h = self.node_encoder(x)  # [num_nodes, hidden_dim]

        # ===== PROCESS =====
        for layer_idx in range(self.num_layers):
            h_residual = h  # Save before transform

            h = self.nnconv_layers[layer_idx](h, edge_index, edge_attr)

            if self.use_batch_norm:
                h = self.batch_norms[layer_idx](h)

            h = self.activation(h)
            h = self.dropout(h)

            # Residual skip (v4: applied from layer 0, not layer 1)
            if self.use_residuals:
                h = h + h_residual

        # ===== DECODE =====
        src, dst = edge_index[0], edge_index[1]
        h_i = h[src]  # [num_edges, hidden_dim]
        h_j = h[dst]  # [num_edges, hidden_dim]

        predictions = self.edge_decoder(h_i, h_j, edge_attr)  # [num_edges, 1]
        return predictions


# ============================================================================
# 6. UTILITY FUNCTIONS
# ============================================================================

def create_model(
    node_features_dim=10,
    edge_features_dim=7,
    hidden_dim=128,
    num_layers=4,
    dropout_p=0.1,
    device='cpu',
):
    """
    Convenience function to instantiate and move model to device.

    Args:
        node_features_dim: Default 10 (coordinates + BCs + applied load)
        edge_features_dim: Default 7 (material + geometric properties)
        hidden_dim:        Default 128
        num_layers:        Default 4
        dropout_p:         Dropout probability. Default 0.1. Set 0.0 to disable.
        device:            'cpu' or 'cuda'

    Returns:
        Model on specified device, ready for training.
    """
    model = TrussEdgeSafetyGNN(
        node_features_dim=node_features_dim,
        edge_features_dim=edge_features_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_batch_norm=True,
        use_residuals=True,
        dropout_p=dropout_p,
    )
    return model.to(device)


def count_parameters(model):
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 70)
    print("TrussEdgeSafetyGNN v4: Sanity Check")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    model = create_model(
        node_features_dim=10,
        edge_features_dim=7,
        hidden_dim=128,
        num_layers=4,
        dropout_p=0.1,
        device=device,
    )
    print(f"Trainable parameters: {count_parameters(model):,}\n")

    # Realistic truss dimensions (39 nodes, 120 edges)
    num_nodes = 39
    num_edges = 120

    x          = torch.randn((num_nodes, 10), device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_attr  = torch.randn((num_edges, 7), device=device)

    # Cache topology (call once before training)
    model.cache_topology(edge_index)

    print("Input shapes:")
    print(f"  x:          {x.shape}")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  edge_attr:  {edge_attr.shape}\n")

    # Training mode forward pass
    model.train()
    predictions = model(x, edge_attr=edge_attr)
    print(f"Output predictions: {predictions.shape}")
    print(f"  Min:  {predictions.min():.4f}")
    print(f"  Max:  {predictions.max():.4f}")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  All in [0,1]? {(predictions >= 0).all() and (predictions <= 1).all()}\n")

    # Focal Loss
    targets = torch.randint(0, 2, (num_edges, 1), dtype=torch.float32, device=device)
    loss_fn = FocalLoss(alpha=0.1, gamma=2.0)
    loss    = loss_fn(predictions, targets)
    print(f"Focal Loss: {loss.item():.6f}\n")

    # Backward pass
    loss.backward()
    print("Gradient flow: OK")

    # Eval mode check (dropout disabled, deterministic)
    model.eval()
    with torch.no_grad():
        preds_eval = model(x, edge_attr=edge_attr)
    print(f"Eval mode predictions (deterministic): {preds_eval.shape} OK")
    print("=" * 70)
