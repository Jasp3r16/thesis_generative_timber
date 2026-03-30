import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import SAGEConv


class TrussEdgeGNN(torch.nn.Module):
    """GraphSAGE edge-regression model used for c21 training and downstream inference."""

    def __init__(self, node_in_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = SAGEConv(node_in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, hidden_dim)
        self.edge_predictor = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        h = F.relu(self.conv4(h, edge_index))
        src, dst = edge_index
        edge_features = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_predictor(edge_features)


__all__ = ["TrussEdgeGNN"]
