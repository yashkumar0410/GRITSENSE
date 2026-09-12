import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    Graph Attention Network for the volleyball interaction graph.

    Input:
        x          : [num_nodes, 49]
        edge_index : [2, num_edges]
        edge_attr  : [num_edges, 4]

    Output:
        node_embeddings : [num_nodes, out_channels]
    """

    def __init__(
        self,
        in_channels=49,
        hidden_channels=32,
        out_channels=32,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads

        # ----------------------------------------------------
        # First GAT layer
        # ----------------------------------------------------
        #
        # 49 input features
        # 32 features per attention head
        # 4 attention heads
        #
        # Output:
        # 32 * 4 = 128 features
        #
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=4,
        )

        # ----------------------------------------------------
        # Second GAT layer
        # ----------------------------------------------------
        #
        # Input = 32 * 4 = 128
        #
        # 4 attention heads are used again, but concat=False.
        #
        # Therefore:
        # output = 32
        #
        self.gat2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
            edge_dim=4,
        )

        # ----------------------------------------------------
        # Residual projection
        # ----------------------------------------------------
        #
        # Original input is 49-D.
        # First GAT output is 128-D.
        #
        # Project 49 -> 128 so that the original node
        # information can be preserved.
        #
        self.residual = nn.Linear(
            in_channels,
            hidden_channels * heads
        )

        self.activation = nn.ELU()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
    ):
        """
        Forward pass through the GAT.

        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, 49]

        edge_index : torch.Tensor
            Graph connectivity [2, num_edges]

        edge_attr : torch.Tensor
            Edge features [num_edges, 4]

        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, out_channels]
        """

        # ----------------------------------------------------
        # Validate input dimensions
        # ----------------------------------------------------

        if x.dim() != 2:
            raise ValueError(
                "x must have shape [num_nodes, num_features]"
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} node features, "
                f"got {x.size(1)}"
            )

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges]"
            )

        if edge_attr.dim() != 2:
            raise ValueError(
                "edge_attr must have shape [num_edges, num_edge_features]"
            )

        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError(
                "Number of edge attributes must match "
                "number of edges"
            )

        if edge_attr.size(1) != 4:
            raise ValueError(
                f"Expected 4 edge features, "
                f"got {edge_attr.size(1)}"
            )

        # ----------------------------------------------------
        # First GAT layer
        # ----------------------------------------------------

        residual = self.residual(x)

        x = self.gat1(
            x,
            edge_index,
            edge_attr
        )

        x = self.activation(x)

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        x = x + residual

        x = self.dropout(x)

        # ----------------------------------------------------
        # Second GAT layer
        # ----------------------------------------------------

        x = self.gat2(
            x,
            edge_index,
            edge_attr
        )

        return x