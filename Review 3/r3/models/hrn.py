import torch
import torch.nn as nn

from .gat import GAT


class HRN(nn.Module):
    """
    Hierarchical Relation Network (HRN).

    Architecture:

        Graph
          ↓
        GAT
          ↓
        Node-level representations
          ↓
        Team-level representations
          ↓
        Global representation

    Expected input:
        x          : [num_nodes, 49]
        edge_index : [2, num_edges]
        edge_attr  : [num_edges, 4]
        node_type  : [num_nodes]
        node_team  : [num_nodes]

    Current volleyball graph:

        6 player nodes
        1 ball node
        ----------------
        7 nodes total

    Node embeddings produced by GAT:
        [7, hidden_channels]

    Team representations:
        [2, hidden_channels]

    Global representation:
        [1, hidden_channels]
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

        # ====================================================
        # LEVEL 1 - GRAPH ATTENTION
        # ====================================================

        self.gat = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
        )

        # ====================================================
        # LEVEL 2 - TEAM RELATIONS
        # ====================================================

        self.team_projection = nn.Sequential(
            nn.Linear(
                out_channels,
                out_channels,
            ),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # ====================================================
        # LEVEL 3 - GLOBAL RELATIONS
        # ====================================================

        self.global_projection = nn.Sequential(
            nn.Linear(
                out_channels,
                out_channels,
            ),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_type=None,
        node_team=None,
    ):
        """
        Forward pass through the Hierarchical Relation Network.

        Parameters
        ----------
        x : torch.Tensor
            Node features.

            Shape:
                [num_nodes, 49]

        edge_index : torch.Tensor
            Graph connectivity.

            Shape:
                [2, num_edges]

        edge_attr : torch.Tensor
            Edge features.

            Shape:
                [num_edges, 4]

        node_type : torch.Tensor, optional
            Node type information.

            0 = player
            1 = ball

        node_team : torch.Tensor, optional
            Team ID for each node.

            0 = Team 0
            1 = Team 1
           -1 = Ball

        Returns
        -------
        dict

            node_embeddings:
                [num_nodes, out_channels]

            team_embeddings:
                [2, out_channels]

            global_embedding:
                [1, out_channels]
        """

        # ====================================================
        # LEVEL 1
        # GAT NODE REPRESENTATIONS
        # ====================================================

        node_embeddings = self.gat(
            x,
            edge_index,
            edge_attr,
        )

        # ====================================================
        # IDENTIFY PLAYERS AND BALL
        # ====================================================

        if node_type is None:

            # Fallback for the current graph structure:
            # first nodes = players
            # final node = ball

            player_mask = torch.ones(
                node_embeddings.size(0),
                dtype=torch.bool,
                device=node_embeddings.device,
            )

            player_mask[-1] = False

            ball_mask = ~player_mask

        else:

            player_mask = (
                node_type == 0
            )

            ball_mask = (
                node_type == 1
            )

        # Extract player and ball embeddings.

        player_embeddings = (
            node_embeddings[player_mask]
        )

        ball_embeddings = (
            node_embeddings[ball_mask]
        )

        # ====================================================
        # LEVEL 2
        # TEAM-LEVEL REPRESENTATIONS
        # ====================================================

        # Team information is required for proper
        # hierarchical aggregation.

        if node_team is None:

            raise ValueError(
                "node_team is required for HRN "
                "team-level aggregation."
            )

        # Only take team IDs corresponding to
        # player nodes.

        player_team = (
            node_team[player_mask]
        )

        # ----------------------------------------------------
        # Identify players belonging to each team.
        # ----------------------------------------------------

        team_0_mask = (
            player_team == 0
        )

        team_1_mask = (
            player_team == 1
        )

        team_0 = (
            player_embeddings[team_0_mask]
        )

        team_1 = (
            player_embeddings[team_1_mask]
        )

        # ----------------------------------------------------
        # Validate that both teams exist.
        # ----------------------------------------------------

        if team_0.size(0) == 0:

            raise ValueError(
                "No players found for team 0."
            )

        if team_1.size(0) == 0:

            raise ValueError(
                "No players found for team 1."
            )

        # ----------------------------------------------------
        # Mean pooling
        #
        # Multiple player representations
        # become one team representation.
        # ----------------------------------------------------

        team_0_embedding = (
            team_0.mean(
                dim=0,
                keepdim=True,
            )
        )

        team_1_embedding = (
            team_1.mean(
                dim=0,
                keepdim=True,
            )
        )

        # Combine both team representations.

        team_embeddings = torch.cat(
            [
                team_0_embedding,
                team_1_embedding,
            ],
            dim=0,
        )

        # Apply learnable team-level transformation.

        team_embeddings = (
            self.team_projection(
                team_embeddings
            )
        )

        # ====================================================
        # LEVEL 3
        # GLOBAL RELATION REPRESENTATION
        # ====================================================

        # Combine both team representations
        # into one global team representation.

        team_global = (
            team_embeddings.mean(
                dim=0,
                keepdim=True,
            )
        )

        # ----------------------------------------------------
        # Include the ball representation.
        # ----------------------------------------------------

        if ball_embeddings.numel() > 0:

            ball_global = (
                ball_embeddings.mean(
                    dim=0,
                    keepdim=True,
                )
            )

            # Combine team-level information
            # with ball-level information.

            global_embedding = (
                team_global + ball_global
            ) / 2

        else:

            global_embedding = (
                team_global
            )

        # Apply learnable global transformation.

        global_embedding = (
            self.global_projection(
                global_embedding
            )
        )

        # ====================================================
        # RETURN HIERARCHICAL REPRESENTATIONS
        # ====================================================

        return {
            "node_embeddings": node_embeddings,
            "team_embeddings": team_embeddings,
            "global_embedding": global_embedding,
        }