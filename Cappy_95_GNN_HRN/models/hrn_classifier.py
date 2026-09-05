import torch
import torch.nn as nn

from .hrn import HRN


class HRNClassifier(nn.Module):
    """
    HRN-based group activity classifier.

    Architecture:

        Graph
          ↓
        GAT
          ↓
        Node representations
          ↓
        Team representations
          ↓
        Global representation
          ↓
        Classification head
          ↓
        8 group activity classes
    """

    NUM_CLASSES = 8

    CLASS_NAMES = [
        "right_set",
        "right_spike",
        "right_pass",
        "right_winpoint",
        "left_winpoint",
        "left_pass",
        "left_spike",
        "left_set",
    ]

    def __init__(
        self,
        in_channels=49,
        hidden_channels=32,
        out_channels=32,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        # ----------------------------------------------------
        # GAT + HRN
        # ----------------------------------------------------

        self.hrn = HRN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                out_channels,
                out_channels,
            ),
            nn.ELU(),

            nn.Dropout(dropout),

            nn.Linear(
                out_channels,
                self.NUM_CLASSES,
            ),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_type,
        node_team,
    ):
        """
        Forward pass.

        Returns
        -------
        logits:
            [8]

        global_embedding:
            [1, 32]
        """

        # ----------------------------------------------------
        # HRN
        # ----------------------------------------------------

        hrn_output = self.hrn(
            x,
            edge_index,
            edge_attr,
            node_type,
            node_team,
        )

        global_embedding = (
            hrn_output["global_embedding"]
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            global_embedding
        )

        # Remove the batch dimension.

        logits = logits.squeeze(0)

        return {
            "logits": logits,
            "global_embedding": global_embedding,
            "node_embeddings": (
                hrn_output["node_embeddings"]
            ),
            "team_embeddings": (
                hrn_output["team_embeddings"]
            ),
        }