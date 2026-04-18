import torch
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import NNConv, global_mean_pool


class TrussEdgeNNConv(torch.nn.Module):
    """
    Baseline edge-aware surrogate model used by the c21 pipeline.

    Architecture:
    - Four NNConv message-passing layers update node features using edge features.
    - Edge attributes are encoded with a small MLP.
    - Global graph features are encoded with another MLP (or a pooled-node fallback).
    - For each edge, the model concatenates source-node context, target-node context,
      edge context, and global context, then predicts one axial-force value.

    This is the stable baseline model without residual connections, LayerNorm, or dropout.
    """

    def __init__(
        self,
        node_in_dim: int = 10,
        edge_in_dim: int = 7,
        global_in_dim: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.global_in_dim = global_in_dim
        self.hidden_dim = hidden_dim

        self.edge_nns = ModuleList([
            Sequential(
                Linear(edge_in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, node_in_dim * hidden_dim),
            ),
            Sequential(
                Linear(edge_in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim * hidden_dim),
            ),
            Sequential(
                Linear(edge_in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim * hidden_dim),
            ),
            Sequential(
                Linear(edge_in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim * hidden_dim),
            ),
        ])

        self.conv1 = NNConv(node_in_dim, hidden_dim, self.edge_nns[0], aggr="mean")
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_nns[1], aggr="mean")
        self.conv3 = NNConv(hidden_dim, hidden_dim, self.edge_nns[2], aggr="mean")
        self.conv4 = NNConv(hidden_dim, hidden_dim, self.edge_nns[3], aggr="mean")

        self.edge_attr_encoder = Sequential(
            Linear(edge_in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )

        if global_in_dim > 0:
            self.global_encoder = Sequential(
                Linear(global_in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            )
        else:
            self.global_encoder = None

        self.graph_fallback = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )

        # If no global features, edge_predictor input is hidden_dim*3
        edge_predictor_in_dim = hidden_dim * (4 if global_in_dim > 0 else 3)
        self.edge_predictor = Sequential(
            Linear(edge_predictor_in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_in_dim))

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = F.relu(self.conv3(h, edge_index, edge_attr))
        h = F.relu(self.conv4(h, edge_index, edge_attr))

        graph_embedding = global_mean_pool(h, batch)

        src, dst = edge_index
        edge_batch = batch[src]

        edge_context = self.edge_attr_encoder(edge_attr)

        if self.global_in_dim > 0 and u is not None:
            if u.dim() == 1:
                u = u.unsqueeze(0)
            graph_context = self.global_encoder(u)
            graph_context_for_edges = graph_context[edge_batch]
            node_context = torch.cat([h[src], h[dst]], dim=1)
            edge_features = torch.cat([node_context, edge_context, graph_context_for_edges], dim=1)
        else:
            # No global features: skip global context
            node_context = torch.cat([h[src], h[dst]], dim=1)
            edge_features = torch.cat([node_context, edge_context], dim=1)
        return self.edge_predictor(edge_features)


class TrussEdgeNNConvV2(torch.nn.Module):
    """
    Enhanced edge-aware surrogate model with residuals, LayerNorm, and dropout.

    Architecture:
    - Four NNConv message-passing blocks process the graph.
    - Each block uses: NNConv -> LayerNorm -> ReLU -> Dropout.
    - Residual skip connections are applied across all four blocks to stabilize training.
    - Edge and global features are encoded with MLPs that also include dropout.
    - The edge prediction head uses the same combined edge/node/global context as v1,
      but with dropout regularization.

    This model is intended as an optional v2 upgrade over the baseline.
    """

    def __init__(
        self,
        node_in_dim: int = 10,
        edge_in_dim: int = 7,
        global_in_dim: int = 3,
        hidden_dim: int = 128,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.global_in_dim = global_in_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        self.edge_nns = ModuleList([
            Sequential(
                Linear(edge_in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, node_in_dim * hidden_dim),
            ),
            Sequential(
                Linear(edge_in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim * hidden_dim),
            ),
            Sequential(
                Linear(edge_in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim * hidden_dim),
            ),
            Sequential(
                Linear(edge_in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim * hidden_dim),
            ),
        ])

        self.conv1 = NNConv(node_in_dim, hidden_dim, self.edge_nns[0], aggr="mean")
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_nns[1], aggr="mean")
        self.conv3 = NNConv(hidden_dim, hidden_dim, self.edge_nns[2], aggr="mean")
        self.conv4 = NNConv(hidden_dim, hidden_dim, self.edge_nns[3], aggr="mean")

        self.input_residual = Linear(node_in_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)
        self.norm4 = LayerNorm(hidden_dim)
        self.dropout = Dropout(p=dropout_p)

        self.edge_attr_encoder = Sequential(
            Linear(edge_in_dim, hidden_dim),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )

        if global_in_dim > 0:
            self.global_encoder = Sequential(
                Linear(global_in_dim, hidden_dim),
                ReLU(),
                Dropout(p=dropout_p),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            )
        else:
            self.global_encoder = None

        self.graph_fallback = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(p=dropout_p),
        )

        edge_predictor_in_dim = hidden_dim * (4 if global_in_dim > 0 else 3)
        self.edge_predictor = Sequential(
            Linear(edge_predictor_in_dim, hidden_dim),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(hidden_dim // 2, 1),
        )

    def _apply_block(
        self,
        conv: NNConv,
        norm: LayerNorm,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        residual = h
        h = conv(h, edge_index, edge_attr)
        h = norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h + residual

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_in_dim))

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = h + self.input_residual(x)

        h = self._apply_block(self.conv2, self.norm2, h, edge_index, edge_attr)
        h = self._apply_block(self.conv3, self.norm3, h, edge_index, edge_attr)
        h = self._apply_block(self.conv4, self.norm4, h, edge_index, edge_attr)

        graph_embedding = global_mean_pool(h, batch)

        src, dst = edge_index
        edge_batch = batch[src]

        edge_context = self.edge_attr_encoder(edge_attr)

        if self.global_in_dim > 0 and u is not None:
            if u.dim() == 1:
                u = u.unsqueeze(0)
            graph_context = self.global_encoder(u)
            graph_context_for_edges = graph_context[edge_batch]
            node_context = torch.cat([h[src], h[dst]], dim=1)
            edge_features = torch.cat([node_context, edge_context, graph_context_for_edges], dim=1)
        else:
            node_context = torch.cat([h[src], h[dst]], dim=1)
            edge_features = torch.cat([node_context, edge_context], dim=1)
        return self.edge_predictor(edge_features)


TrussEdgeGNN = TrussEdgeNNConv
TrussEdgeGNNV2 = TrussEdgeNNConvV2


__all__ = ["TrussEdgeNNConv", "TrussEdgeNNConvV2", "TrussEdgeGNN", "TrussEdgeGNNV2"]
