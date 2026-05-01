"""
TrussEdgeSafetyGNN: PyTorch Geometric Model for Structural Safety Prediction

A clean, modular implementation using NNConv for edge-aware message passing in timber trusses.

Architecture Overview:
  1. Encoder: Project node features (12D) → hidden_dim
  2. Processor: Stack of NNConv layers with adaptive edge weights, residuals, and batch normalization
  3. Decoder: Concatenate source/target embeddings + raw edge features → binary classification
  4. Loss: Focal Loss to handle class imbalance (most members are safe)

Usage:
  >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
  >>> model = TrussEdgeSafetyGNN(node_features_dim=12, edge_features_dim=7, hidden_dim=128).to(device)
  >>> model.cache_topology(edge_index, edge_attr)  # Optimize static topology
  >>> predictions = model(x, edge_index, edge_attr)  # [num_edges, 1], binary probabilities
  >>> loss_fn = FocalLoss(alpha=0.1, gamma=2.0)
  >>> loss = loss_fn(predictions, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm


# ============================================================================
# 1. FILTER-GENERATING NETWORK: Edge Feature Transformation
# ============================================================================

class EdgeFeatureMLPFilter(nn.Module):
    """
    Small MLP that transforms continuous edge features into dynamic weight matrices for NNConv.
    
    Conceptual Purpose:
    -------------------
    In traditional FEA, the global stiffness matrix K is assembled from individual member stiffness
    matrices, where each member's contribution depends on its material properties (E-modulus) and
    geometry (cross-sectional area A, moments of inertia Iy, Iz).
    
    This filter mimics that process: instead of using a fixed weight for all edges, we compute
    an adaptive weight matrix for each edge based on its 7D feature vector (Area, Length, E, Iy, Iz, J, EA/L).
    
    Example:
    --------
    - A stiff member (high E*A/L) will learn high filter weights → strong message passing
    - A weak member (low E*A/L) will learn lower weights → weaker influence on connected nodes
    - This grounding in physical properties helps the model respect structural mechanics intuition
    """
    
    def __init__(self, edge_features_dim, out_channels, hidden=64):
        """
        Args:
            edge_features_dim: Number of continuous edge features (e.g., 7)
            out_channels: Output dimensionality of the weight matrix (typically hidden_dim * hidden_dim)
            hidden: Intermediate hidden layer size (default 64)
        """
        super().__init__()
        self.edge_features_dim = edge_features_dim
        self.out_channels = out_channels
        
        self.fc1 = nn.Linear(edge_features_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_channels)
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, edge_attr):
        """
        Args:
            edge_attr: [num_edges, edge_features_dim] continuous edge features
        
        Returns:
            [num_edges, out_channels] weight matrices (reshaped internally by NNConv)
        """
        h = self.activation(self.fc1(edge_attr))
        return self.fc2(h)


# ============================================================================
# 2. EDGE-LEVEL DECODER: Prediction Head for Binary Classification
# ============================================================================

class EdgeDecoder(nn.Module):
    """
    Edge-level prediction head: reads source/target node embeddings and concatenates
    with raw continuous edge features, then outputs binary safety classification.
    
    Why Concatenate Raw Edge Features?
    -----------------------------------
    Node embeddings alone encode aggregate force flow, but cannot directly encode member geometry.
    By concatenating raw features (Area, Length, E, Iy, Iz, J, EA/L), we ground the decision
    in actual material properties:
    
    Example Scenario:
    - Two edges carry identical internal forces (node embeddings are similar)
    - Edge A: Area=0.01 m², E=12 GPa (small timber strut) → high utilization → unsafe
    - Edge B: Area=0.1 m², E=12 GPa (large beam) → low utilization → safe
    
    Without raw edge features, the model might conflate these. With concatenation, it sees
    the geometry and can predict different outcomes.
    """
    
    def __init__(self, hidden_dim, edge_features_dim):
        """
        Args:
            hidden_dim: Dimensionality of node embeddings from processor
            edge_features_dim: Number of raw edge features (e.g., 7)
        """
        super().__init__()
        
        # Concatenation: [h_i | h_j | e_ij]
        concat_dim = 2 * hidden_dim + edge_features_dim
        
        # 3-layer MLP for final prediction
        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.activation = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, h_i, h_j, e_ij):
        """
        Args:
            h_i: Source node embeddings [num_edges, hidden_dim]
            h_j: Target node embeddings [num_edges, hidden_dim]
            e_ij: Raw edge features [num_edges, edge_features_dim]
        
        Returns:
            Binary predictions [num_edges, 1], values in [0, 1]
            Interpretation: P(Utilization ≤ 1.0) = P(safe) = prediction value
        """
        x = torch.cat([h_i, h_j, e_ij], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# ============================================================================
# 3. FOCAL LOSS: Handling Severe Class Imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: -α * (1 - p_t)^γ * log(p_t)
    
    Why Use Focal Loss for Structural Safety?
    -------------------------------------------
    Timber truss datasets have extreme class imbalance:
    - ~95% of members are SAFE (Utilization ≤ 1.0, label=0)
    - ~5% of members are UNSAFE (Utilization > 1.0, label=1)
    
    A naive model achieves 95% accuracy by predicting "all safe" — useless for safety prediction!
    
    Focal Loss Solution:
    - Down-weights easy negative examples (most safe members)
    - Up-weights hard positive examples (rare failures)
    - Forces model to learn critical failure patterns
    
    Hyperparameter Tuning:
    - alpha: Balancing weight. If class ratio is 95% safe / 5% unsafe, try alpha ∈ [0.05, 0.2]
    - gamma: Focusing exponent. γ=2.0 is standard. Increase if model still predicts too many "safe".
    """
    
    def __init__(self, alpha=0.1, gamma=2.0):
        """
        Args:
            alpha: Balancing weight for positive class (unsafe). Higher → penalize false negatives more.
                   Recommended range: [0.05, 0.25] depending on pos/neg ratio.
            gamma: Focusing parameter (default 2.0). Higher → more focus on hard examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output [num_edges, 1], values in [0, 1] from Sigmoid
            targets: Ground truth [num_edges, 1] or [num_edges], values in {0, 1}
                     0 = safe (Utilization ≤ 1.0), 1 = unsafe (Utilization > 1.0)
        
        Returns:
            Scalar loss value (averaged over all edges)
        """
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1).float()
        
        # Binary cross-entropy
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Probability of true class
        p_t = torch.where(targets == 1, predictions, 1 - predictions)
        
        # Focal weight: (1 - p_t)^gamma
        # Down-weights easy examples where p_t is close to 1
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha balancing: penalize false negatives (missed failures) more
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine
        focal_loss = alpha_weight * focal_weight * bce
        return focal_loss.mean()


# ============================================================================
# 4. MAIN MODEL: TrussEdgeSafetyGNN
# ============================================================================

class TrussEdgeSafetyGNN(nn.Module):
    """
    End-to-end Graph Neural Network for predicting edge-level structural safety in timber trusses.
    
    Architecture: Encoder → Processor (NNConv Stack) → Decoder
    
    Input:
    ------
    - Node features (x): [num_nodes, 12] coordinates + boundary conditions + applied loads
    - Edge index (edge_index): [2, num_edges] graph connectivity
    - Edge features (edge_attr): [num_edges, 7] material + geometric properties
    
    Output:
    -------
    - Edge-level predictions: [num_edges, 1] binary probabilities ∈ [0, 1]
      Interpretation: P(safe) = P(Utilization ≤ 1.0)
    
    Key Design Decisions:
    ----------------------
    1. NNConv + Filter-Generating MLP:
       - Standard GCN uses fixed, uniform weights for all edges.
       - Reality: Stiff members (high E*A/L) have stronger structural influence than weak members.
       - Solution: NNConv learns adaptive edge weights based on 7D material/geometric features.
    
    2. Residual Skip Connections:
       - Deep GNNs suffer from over-smoothing: node embeddings converge to identical values.
       - Residuals preserve local structure: h_out = h_in + GNN(h_in)
       - Allows 4+ layers without losing differentiation between distant nodes (base vs. top of truss).
    
    3. Batch Normalization:
       - Stabilizes hidden activations across layers.
       - Reduces internal covariate shift; easier optimization.
       - Recommended for deep GNNs with 4+ layers.
    
    4. Raw Edge Feature Concatenation in Decoder:
       - Node embeddings represent force flow, but not member geometry.
       - Concatenating raw Area, E, Iy, Iz ensures decoder sees material properties.
       - Prevents node-only decisions from ignoring member stiffness.
    
    5. Static Topology Caching:
       - Timber trusses have fixed topology: same nodes/edges across all samples.
       - Caching edge_index avoids redundant CSR format conversions in every forward pass.
       - Call cache_topology() once before training.
    """
    
    def __init__(
        self,
        node_features_dim=12,
        edge_features_dim=7,
        hidden_dim=128,
        num_layers=4,
        use_batch_norm=True,
        use_residuals=True,
    ):
        """
        Args:
            node_features_dim: Input node feature dimensionality.
                Default 12: x, y, z (coordinates) + Tx, Ty, Tz, Rx, Ry, Rz (BCs) + Fz (applied load)
            
            edge_features_dim: Input edge feature dimensionality.
                Default 7: Area, Length, E (Young's modulus), Iy, Iz, J, EA/L
                These are passed to EdgeFeatureMLPFilter to compute adaptive edge weights.
            
            hidden_dim: Latent dimensionality for node embeddings throughout processor.
                Default 128. Larger → more expressive but more memory; try 64, 128, 256.
            
            num_layers: Number of NNConv message-passing layers.
                Default 4 (middle ground). Range: 2–6.
                - 2–3 layers: Fast, local stiffness effects only
                - 4–5 layers: Balanced, global force propagation
                - 6+ layers: Deep, but risk of over-smoothing without strong residuals
            
            use_batch_norm: Apply BatchNorm after each NNConv layer (default True).
                Recommended for stability, especially for num_layers ≥ 4.
            
            use_residuals: Apply residual skip connections (default True).
                Strongly recommended for num_layers ≥ 4 to prevent over-smoothing.
        """
        super().__init__()
        
        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_residuals = use_residuals
        
        # ===== ENCODER =====
        # Project diverse node features (coordinates, BCs, loads) into learned latent space
        self.node_encoder = nn.Linear(node_features_dim, hidden_dim)
        
        # ===== PROCESSOR: Stack of NNConv Layers =====
        self.nnconv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for layer_idx in range(num_layers):
            # Filter-Generating MLP: transforms edge features → weight matrix for NNConv
            # Input: 7D edge features (Area, Length, E, Iy, Iz, J, EA/L)
            # Output: hidden_dim × hidden_dim weight matrix
            # Why: Each edge gets adaptive weights based on its stiffness and geometry
            edge_mlp = EdgeFeatureMLPFilter(
                edge_features_dim=edge_features_dim,
                out_channels=hidden_dim * hidden_dim,
                hidden=64
            )
            
            # NNConv: Neural Network Convolution
            # Performs message passing weighted by the learned edge-dependent weights
            nnconv = NNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                nn=edge_mlp,
                aggr='add'  # 'add' is standard; alternatives: 'mean', 'max'
            )
            self.nnconv_layers.append(nnconv)
            
            # Batch normalization (optional but recommended)
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Activation function: LeakyReLU with slope 0.1
        # Chosen over standard ReLU to avoid "dead neurons" (where gradient = 0)
        self.activation = nn.LeakyReLU(0.1)
        
        # ===== DECODER: Edge-Level Prediction Head =====
        # Takes source/target embeddings + raw edge features → binary safety prediction
        self.edge_decoder = EdgeDecoder(hidden_dim, edge_features_dim)
        
        # ===== STATIC TOPOLOGY OPTIMIZATION =====
        # Register edge_index and edge_attr as buffers (not trainable, but moved to device automatically)
        self.register_buffer('edge_index_cache', torch.zeros((2, 1), dtype=torch.long))
        self.register_buffer('edge_attr_cache', torch.zeros((1, edge_features_dim), dtype=torch.float32))
        self._is_topology_cached = False
    
    def cache_topology(self, edge_index, edge_attr):
        """
        Pre-cache the edge_index and edge_attr for static topologies.
        
        Call this ONCE before training. For fixed-topology graphs, this avoids
        redundant CSR conversions in every forward pass.
        
        Args:
            edge_index: [2, num_edges] edge connectivity (COO format)
            edge_attr: [num_edges, edge_features_dim] continuous edge features
        """
        self.edge_index_cache = edge_index.clone()
        self.edge_attr_cache = edge_attr.clone()
        self._is_topology_cached = True
        
        print(f"[TrussEdgeSafetyGNN] Topology cached: "
              f"{edge_index.shape[1]} edges, {edge_attr.shape[1]} edge features")
    
    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        """
        Forward pass: Encode node features → Process with NNConv → Decode to predictions.
        
        Args:
            x: Node features [num_nodes, node_features_dim]
            
            edge_index: [2, num_edges] edge connectivity in COO format.
                Optional if topology has been cached via cache_topology().
            
            edge_attr: [num_edges, edge_features_dim] raw continuous edge features.
                Optional if cached. These are passed to Filter-Generating MLP.
            
            batch: Batch assignment vector [num_nodes] (optional, not used in this model).
                Included for compatibility with PyG DataLoader but not needed for single-graph processing.
        
        Returns:
            Edge-level predictions [num_edges, 1], binary probabilities from Sigmoid.
            Interpretation: prediction[i] ≈ P(edge i is safe) = P(Utilization[i] ≤ 1.0)
        """
        
        # Use cached topology if available and not provided
        if self._is_topology_cached and edge_index is None:
            edge_index = self.edge_index_cache
            edge_attr = self.edge_attr_cache
        
        if edge_index is None or edge_attr is None:
            raise ValueError(
                "edge_index and edge_attr must be provided or pre-cached via cache_topology()"
            )
        
        # ===== ENCODE: Node Feature Projection =====
        # Transform node features (coordinates, loads, BCs) into hidden_dim latent space
        h = self.node_encoder(x)  # [num_nodes, hidden_dim]
        
        # ===== PROCESS: NNConv Message-Passing Layers =====
        # Stack of learnable graph convolutions with adaptive edge weights
        for layer_idx in range(self.num_layers):
            h_residual = h  # Save for residual connection
            
            # NNConv: Message passing with edge-adaptive weights
            # The Filter-Generating MLP computes weights based on edge features (Area, E, etc.)
            # Why it works: Stiff edges get high weights → stronger message passing
            #              Weak edges get low weights → weaker influence
            h = self.nnconv_layers[layer_idx](h, edge_index, edge_attr)
            
            # Batch normalization (stabilizes training)
            if self.use_batch_norm:
                h = self.batch_norms[layer_idx](h)
            
            # Non-linear activation
            h = self.activation(h)
            
            # Residual skip connection (prevents over-smoothing)
            # Applied from layer 1 onwards (layer 0 doesn't benefit from skip yet)
            if self.use_residuals and layer_idx > 0:
                h = h + h_residual
        
        # ===== DECODE: Edge-Level Safety Prediction =====
        # Extract source and target node embeddings for each edge
        source_idx, target_idx = edge_index[0], edge_index[1]
        h_i = h[source_idx]  # [num_edges, hidden_dim] source node embeddings
        h_j = h[target_idx]  # [num_edges, hidden_dim] target node embeddings
        
        # Pass concatenation of embeddings + raw edge features through final MLP
        predictions = self.edge_decoder(h_i, h_j, edge_attr)  # [num_edges, 1]
        
        return predictions


# ============================================================================
# 5. UTILITY FUNCTIONS
# ============================================================================

def create_model(
    node_features_dim=12,
    edge_features_dim=7,
    hidden_dim=128,
    num_layers=4,
    device='cpu',
):
    """
    Convenience function to instantiate model and move to device.
    
    Args:
        node_features_dim: Default 12 (coordinates + BCs + applied load)
        edge_features_dim: Default 7 (material + geometric properties)
        hidden_dim: Default 128
        num_layers: Default 4 (middle ground: 2–6 layers)
        device: 'cpu' or 'cuda'
    
    Returns:
        Model on specified device, ready for training
    """
    model = TrussEdgeSafetyGNN(
        node_features_dim=node_features_dim,
        edge_features_dim=edge_features_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_batch_norm=True,
        use_residuals=True,
    )
    return model.to(device)


def count_parameters(model):
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check: forward pass on synthetic data
    print("=" * 70)
    print("TrussEdgeSafetyGNN: Sanity Check")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Create model
    model = create_model(node_features_dim=12, edge_features_dim=7, hidden_dim=128, num_layers=4, device=device)
    print(f"Model created with {count_parameters(model):,} trainable parameters\n")
    
    # Synthetic data
    num_nodes = 10
    num_edges = 15
    
    x = torch.randn((num_nodes, 12), device=device)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)  # Random edges
    edge_attr = torch.randn((num_edges, 7), device=device)  # Edge features
    
    # Forward pass
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  edge_attr: {edge_attr.shape}\n")
    
    predictions = model(x, edge_index, edge_attr)
    print(f"Output predictions: {predictions.shape}")
    print(f"  Min: {predictions.min():.4f}, Max: {predictions.max():.4f}, Mean: {predictions.mean():.4f}")
    print(f"  All values in [0, 1]? {(predictions >= 0).all() and (predictions <= 1).all()}\n")
    
    # Test Focal Loss
    targets = torch.randint(0, 2, (num_edges, 1), dtype=torch.float32, device=device)
    loss_fn = FocalLoss(alpha=0.1, gamma=2.0)
    loss = loss_fn(predictions, targets)
    print(f"Focal Loss: {loss.item():.6f}\n")
    
    # Test backward pass
    loss.backward()
    print("Gradient flow successful ✓")
    print("=" * 70)
