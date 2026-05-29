import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm


class NodeEncoder(nn.Module):
    """Two-layer MLP projecting raw node features into hidden latent space."""

    def __init__(self, node_features_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(node_features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class EdgeFeatureMLPFilter(nn.Module):
    """Three-layer MLP mapping edge features to per-edge NNConv weight matrices."""

    def __init__(self, edge_features_dim, out_channels, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(edge_features_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, edge_attr):
        h = self.activation(self.fc1(edge_attr))
        h = self.activation(self.fc2(h))
        return self.fc3(h)


class EdgeDecoder(nn.Module):
    """Symmetric edge-level prediction head.

    Uses |h_i − h_j| and h_i ⊙ h_j instead of [h_i | h_j] so predictions
    are invariant to edge direction — correct for undirected trusses.
    """

    def __init__(self, hidden_dim, edge_features_dim, dropout_p=0.1):
        super().__init__()
        concat_dim = 2 * hidden_dim + edge_features_dim  # |diff| + prod + raw edge feats

        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_i, h_j, e_ij):
        diff = torch.abs(h_i - h_j)
        prod = h_i * h_j
        x = torch.cat([diff, prod, e_ij], dim=1)

        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x


class FocalLoss(nn.Module):
    """Focal Loss: −α (1−p_t)^γ log(p_t). Clamps p to [eps, 1−eps] for stability."""

    def __init__(self, alpha=0.1, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps   = eps

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets     = targets.view(-1).float()

        p   = predictions.clamp(self.eps, 1.0 - self.eps)
        bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(p, self.alpha),
            torch.full_like(p, 1.0 - self.alpha),
        )
        return (alpha_t * focal_weight * bce).mean()


class WeightedBCELoss(nn.Module):
    """Weighted BCE for imbalanced structural safety data.

    Upweights the unsafe class without suppressing gradients for confident
    predictions. With ~20% unsafe rate, pos_weight ≈ 4.0.
    """

    def __init__(self, pos_weight: float = 4.0, eps: float = 1e-7):
        super().__init__()
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, predictions, targets):
        p = predictions.clamp(self.eps, 1.0 - self.eps).view(-1)
        t = targets.view(-1).float()
        loss = -(self.pos_weight * t * p.log() + (1 - t) * (1 - p).log())
        return loss.mean()


class TrussEdgeSafetyGNN(nn.Module):
    """
    GNN predicting per-member structural safety (P(UC > 1.0)) for timber trusses.

    Architecture: NodeEncoder → NNConv stack with residuals → EdgeDecoder.
    Fixed topology (39 nodes, 120 edges) cached once; edge_attr varies per sample.
    """

    def __init__(
        self,
        node_features_dim=10,
        edge_features_dim=9,
        hidden_dim=128,
        num_layers=4,
        use_batch_norm=True,
        use_residuals=True,
        dropout_p=0.1,
    ):
        super().__init__()

        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim
        self.hidden_dim        = hidden_dim
        self.num_layers        = num_layers
        self.use_batch_norm    = use_batch_norm
        self.use_residuals     = use_residuals
        self.dropout_p         = dropout_p

        self.node_encoder = NodeEncoder(node_features_dim, hidden_dim)

        self.nnconv_layers = nn.ModuleList()
        self.batch_norms   = nn.ModuleList() if use_batch_norm else None
        self.dropout       = nn.Dropout(p=dropout_p)

        for _ in range(num_layers):
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

        self.activation   = nn.LeakyReLU(0.1)
        self.edge_decoder = EdgeDecoder(hidden_dim, edge_features_dim, dropout_p)

        self.register_buffer('edge_index_cache', torch.zeros((2, 1), dtype=torch.long))
        self._is_topology_cached = False

    def cache_topology(self, edge_index):
        """Cache the fixed edge_index. Call once before training/inference."""
        self.edge_index_cache    = edge_index.clone()
        self._is_topology_cached = True
        print(f"[TrussEdgeSafetyGNN] Topology cached: {edge_index.shape[1]} edges")

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        """
        Args:
            x:          [num_nodes, node_features_dim]
            edge_index: [2, num_edges] COO. Optional if topology cached.
            edge_attr:  [num_edges, edge_features_dim]. Always required.
        Returns:
            [num_edges, 1] — P(unsafe) per member.
        """
        if self._is_topology_cached and edge_index is None:
            edge_index = self.edge_index_cache

        if edge_index is None or edge_attr is None:
            raise ValueError(
                "edge_index must be provided or pre-cached via cache_topology(); "
                "edge_attr is always required."
            )

        h = self.node_encoder(x)

        for layer_idx in range(self.num_layers):
            h_residual = h
            h = self.nnconv_layers[layer_idx](h, edge_index, edge_attr)
            if self.use_batch_norm:
                h = self.batch_norms[layer_idx](h)
            h = self.activation(h)
            h = self.dropout(h)
            if self.use_residuals:
                h = h + h_residual

        src, dst = edge_index[0], edge_index[1]
        return self.edge_decoder(h[src], h[dst], edge_attr)


def create_model(
    node_features_dim=10,
    edge_features_dim=9,
    hidden_dim=128,
    num_layers=4,
    dropout_p=0.1,
    device='cpu',
):
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
        edge_features_dim=9,
        hidden_dim=128,
        num_layers=4,
        dropout_p=0.1,
        device=device,
    )
    print(f"Trainable parameters: {count_parameters(model):,}\n")

    num_nodes, num_edges = 39, 120
    x          = torch.randn((num_nodes, 10), device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_attr  = torch.randn((num_edges, 9), device=device)

    model.cache_topology(edge_index)

    model.train()
    predictions = model(x, edge_attr=edge_attr)
    print(f"Output: {predictions.shape}  min={predictions.min():.4f}  max={predictions.max():.4f}")
    print(f"All in [0,1]? {(predictions >= 0).all() and (predictions <= 1).all()}\n")

    targets = torch.randint(0, 2, (num_edges, 1), dtype=torch.float32, device=device)
    loss = FocalLoss(alpha=0.1, gamma=2.0)(predictions, targets)
    loss.backward()
    print(f"Focal Loss: {loss.item():.6f}  |  Gradient flow: OK")

    model.eval()
    with torch.no_grad():
        preds_eval = model(x, edge_attr=edge_attr)
    print(f"Eval mode: {preds_eval.shape} OK")
    print("=" * 70)
