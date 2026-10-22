import math

import numpy as np
import torch
from torch_geometric.data import Data


# ============================================================
# FEATURE GRAPH SCHEMA
# ============================================================

NUM_POSE_KEYPOINTS = 17
POSE_FEATURE_DIM = NUM_POSE_KEYPOINTS * 2

NODE_TYPE_DIM = 2
POSITION_DIM = 2
VELOCITY_DIM = 2
TEAM_DIM = 2
POSE_AVAILABLE_DIM = 1
CONFIDENCE_DIM = 1
COURT_FEATURE_DIM = 5

NODE_FEATURE_DIM = (
    NODE_TYPE_DIM
    + POSITION_DIM
    + VELOCITY_DIM
    + TEAM_DIM
    + POSE_FEATURE_DIM
    + POSE_AVAILABLE_DIM
    + CONFIDENCE_DIM
    + COURT_FEATURE_DIM
)

# 2 + 2 + 2 + 2 + 34 + 1 + 1 + 5 = 49
assert NODE_FEATURE_DIM == 49


# ============================================================
# NODE TYPES
# ============================================================

# These are included inside the neural node feature vector.

PLAYER_NODE_TYPE = np.array(
    [1.0, 0.0],
    dtype=np.float32
)

BALL_NODE_TYPE = np.array(
    [0.0, 1.0],
    dtype=np.float32
)


# ============================================================
# EDGE TYPES
# ============================================================

PLAYER_PLAYER_EDGE_TYPE = 0
PLAYER_BALL_EDGE_TYPE = 1


# ============================================================
# METADATA NODE TYPES
# ============================================================

PLAYER_NODE_INDEX = 0
BALL_NODE_INDEX = 1


class VolleyballGraphBuilder:
    """
    Converts one volleyball frame into a PyTorch Geometric graph.

    ============================================================
    FEATURE GRAPH
    ============================================================

    Each player and the ball become a node.

    Each node contains 49 features:

        0-1     node type
        2-3     normalized position
        4-5     normalized velocity
        6-7     team one-hot
        8-41    17 pose keypoints x 2 coordinates
        42      pose availability
        43      detection confidence
        44-48   court-relative features

    ============================================================
    INTERACTION GRAPH
    ============================================================

    Directed edges are created between:

        player -> player
        player -> ball
        ball -> player

    Every edge contains:

        relative_x
        relative_y
        distance
        same_team

    ============================================================
    """

    def __init__(self):
        self.feature_dim = NODE_FEATURE_DIM

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def validate_frame_inputs(
        self,
        players,
        ball,
        court
    ):
        """
        Validate the basic information required to construct
        a graph for one frame.
        """

        if not isinstance(
            players,
            (list, tuple)
        ):
            raise ValueError(
                "players must be a list or tuple "
                "of player dictionaries."
            )

        if len(players) == 0:
            raise ValueError(
                "players must contain at least "
                "one detected player."
            )

        if not isinstance(ball, dict):
            raise ValueError(
                "ball must be a dictionary "
                "with x and y coordinates."
            )

        if not isinstance(court, dict):
            raise ValueError(
                "court must be a dictionary "
                "with width and height."
            )

        width = court.get("width")
        height = court.get("height")

        if width is None or height is None:
            raise ValueError(
                "court must contain width and height."
            )

        if float(width) <= 0 or float(height) <= 0:
            raise ValueError(
                "court width and height must be positive."
            )

        # ----------------------------------------------------
        # Validate players
        # ----------------------------------------------------

        for idx, player in enumerate(players):

            if not isinstance(player, dict):
                raise ValueError(
                    f"Player at index {idx} "
                    f"is not a dictionary."
                )

            if (
                "x" not in player
                or "y" not in player
            ):
                raise ValueError(
                    f"Player at index {idx} "
                    f"is missing x or y coordinates."
                )

            x = float(player["x"])
            y = float(player["y"])

            if (
                not np.isfinite(x)
                or not np.isfinite(y)
            ):
                raise ValueError(
                    f"Player at index {idx} "
                    f"has non-finite coordinates."
                )

        # ----------------------------------------------------
        # Validate ball
        # ----------------------------------------------------

        if (
            "x" not in ball
            or "y" not in ball
        ):
            raise ValueError(
                "Ball is missing x or y coordinates."
            )

        ball_x = float(ball["x"])
        ball_y = float(ball["y"])

        if (
            not np.isfinite(ball_x)
            or not np.isfinite(ball_y)
        ):
            raise ValueError(
                "Ball has non-finite coordinates."
            )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize_position(
        self,
        x,
        y,
        width,
        height
    ):
        """
        Convert image coordinates to [0, 1].
        """

        x_norm = np.clip(
            float(x) / float(width),
            0.0,
            1.0
        )

        y_norm = np.clip(
            float(y) / float(height),
            0.0,
            1.0
        )

        return x_norm, y_norm

    def normalize_velocity(
        self,
        vx,
        vy,
        width,
        height
    ):
        """
        Normalize image-space velocity.
        """

        vx_norm = (
            float(vx)
            / float(width)
        )

        vy_norm = (
            float(vy)
            / float(height)
        )

        return vx_norm, vy_norm

    # ========================================================
    # POSE
    # ========================================================

    def normalize_pose(
        self,
        pose,
        width,
        height
    ):
        """
        Normalize a 17-keypoint pose.

        Expected:

            [x1, y1, x2, y2, ..., x17, y17]

        Total = 34 values.

        Returns:

            normalized_pose
            pose_available
        """

        if pose is None:
            return (
                np.zeros(
                    POSE_FEATURE_DIM,
                    dtype=np.float32
                ),
                0.0
            )

        pose_array = np.asarray(
            pose,
            dtype=np.float32
        ).reshape(-1)

        if pose_array.size == 0:
            return (
                np.zeros(
                    POSE_FEATURE_DIM,
                    dtype=np.float32
                ),
                0.0
            )

        if pose_array.size != POSE_FEATURE_DIM:
            raise ValueError(
                f"Pose must contain exactly "
                f"{POSE_FEATURE_DIM} values "
                f"(17 keypoints x 2 coordinates), "
                f"but got {pose_array.size}."
            )

        if not np.isfinite(
            pose_array
        ).all():
            raise ValueError(
                "Pose contains non-finite values."
            )

        pose_xy = pose_array.reshape(
            NUM_POSE_KEYPOINTS,
            2
        ).copy()

        pose_xy[:, 0] = np.clip(
            pose_xy[:, 0]
            / float(width),
            0.0,
            1.0
        )

        pose_xy[:, 1] = np.clip(
            pose_xy[:, 1]
            / float(height),
            0.0,
            1.0
        )

        return (
            pose_xy.reshape(-1)
            .astype(np.float32),
            1.0
        )

    # ========================================================
    # TEAM
    # ========================================================

    def get_team_features(
        self,
        team
    ):
        """
        Team 0 -> [1, 0]
        Team 1 -> [0, 1]
        Unknown -> [0, 0]
        """

        if team == 0:
            return np.array(
                [1.0, 0.0],
                dtype=np.float32
            )

        if team == 1:
            return np.array(
                [0.0, 1.0],
                dtype=np.float32
            )

        return np.array(
            [0.0, 0.0],
            dtype=np.float32
        )

    # ========================================================
    # COURT FEATURES
    # ========================================================

    def get_court_features(
        self,
        x,
        y,
        court
    ):
        """
        Five normalized court-context features:

            1. distance from left boundary
            2. distance from right boundary
            3. distance from top boundary
            4. distance from bottom boundary
            5. distance from net
        """

        width = float(
            court["width"]
        )

        height = float(
            court["height"]
        )

        net_x = float(
            court.get(
                "net_x",
                width / 2.0
            )
        )

        distance_left = np.clip(
            float(x) / width,
            0.0,
            1.0
        )

        distance_right = np.clip(
            (width - float(x))
            / width,
            0.0,
            1.0
        )

        distance_top = np.clip(
            float(y) / height,
            0.0,
            1.0
        )

        distance_bottom = np.clip(
            (height - float(y))
            / height,
            0.0,
            1.0
        )

        distance_net = np.clip(
            abs(
                float(x) - net_x
            ) / width,
            0.0,
            1.0
        )

        return np.asarray(
            [
                distance_left,
                distance_right,
                distance_top,
                distance_bottom,
                distance_net,
            ],
            dtype=np.float32
        )

    # ========================================================
    # PLAYER FEATURE VECTOR
    # ========================================================

    def build_player_features(
        self,
        player,
        court
    ):
        """
        Build the complete 49-dimensional
        feature vector for one player.
        """

        width = float(
            court["width"]
        )

        height = float(
            court["height"]
        )

        x = float(
            player["x"]
        )

        y = float(
            player["y"]
        )

        # Position
        x_norm, y_norm = (
            self.normalize_position(
                x,
                y,
                width,
                height
            )
        )

        # Velocity
        vx_norm, vy_norm = (
            self.normalize_velocity(
                player.get("vx", 0.0),
                player.get("vy", 0.0),
                width,
                height
            )
        )

        # Team
        team_features = (
            self.get_team_features(
                player.get("team")
            )
        )

        # Pose
        pose, pose_available = (
            self.normalize_pose(
                player.get("pose"),
                width,
                height
            )
        )

        # Detection confidence
        confidence = float(
            np.clip(
                player.get(
                    "confidence",
                    0.0
                ),
                0.0,
                1.0
            )
        )

        # Court
        court_features = (
            self.get_court_features(
                x,
                y,
                court
            )
        )

        feature_vector = np.concatenate(
            [
                PLAYER_NODE_TYPE,

                np.asarray(
                    [x_norm, y_norm],
                    dtype=np.float32
                ),

                np.asarray(
                    [vx_norm, vy_norm],
                    dtype=np.float32
                ),

                team_features,

                pose,

                np.asarray(
                    [pose_available],
                    dtype=np.float32
                ),

                np.asarray(
                    [confidence],
                    dtype=np.float32
                ),

                court_features,
            ]
        ).astype(np.float32)

        assert (
            feature_vector.shape[0]
            == NODE_FEATURE_DIM
        )

        return feature_vector

    # ========================================================
    # BALL FEATURE VECTOR
    # ========================================================

    def build_ball_features(
        self,
        ball,
        court
    ):
        """
        Build the same 49-dimensional
        representation for the ball.

        Features that do not apply to the ball
        are represented by zeros.
        """

        width = float(
            court["width"]
        )

        height = float(
            court["height"]
        )

        x = float(
            ball["x"]
        )

        y = float(
            ball["y"]
        )

        # Position
        x_norm, y_norm = (
            self.normalize_position(
                x,
                y,
                width,
                height
            )
        )

        # Velocity
        vx_norm, vy_norm = (
            self.normalize_velocity(
                ball.get("vx", 0.0),
                ball.get("vy", 0.0),
                width,
                height
            )
        )

        # Ball has no team
        team_features = np.array(
            [0.0, 0.0],
            dtype=np.float32
        )

        # Ball has no pose
        pose = np.zeros(
            POSE_FEATURE_DIM,
            dtype=np.float32
        )

        pose_available = 0.0

        # Confidence
        confidence = float(
            np.clip(
                ball.get(
                    "confidence",
                    0.0
                ),
                0.0,
                1.0
            )
        )

        # Court
        court_features = (
            self.get_court_features(
                x,
                y,
                court
            )
        )

        feature_vector = np.concatenate(
            [
                BALL_NODE_TYPE,

                np.asarray(
                    [x_norm, y_norm],
                    dtype=np.float32
                ),

                np.asarray(
                    [vx_norm, vy_norm],
                    dtype=np.float32
                ),

                team_features,

                pose,

                np.asarray(
                    [pose_available],
                    dtype=np.float32
                ),

                np.asarray(
                    [confidence],
                    dtype=np.float32
                ),

                court_features,
            ]
        ).astype(np.float32)

        assert (
            feature_vector.shape[0]
            == NODE_FEATURE_DIM
        )

        return feature_vector

    # ========================================================
    # INTERACTION GRAPH
    # ========================================================

    def build_edges(
        self,
        players,
        ball,
        court
    ):
        """
        Build the interaction graph.

        For N players:

            Player-player:
                N * (N - 1)

            Player-ball:
                2N

        For six players:

            30 player-player
            12 player-ball
            ----------------
            42 total edges

        Each edge has four features:

            [relative_x,
             relative_y,
             distance,
             same_team]
        """

        edge_list = []
        edge_attributes = []
        edge_types = []

        width = float(
            court["width"]
        )

        height = float(
            court["height"]
        )

        # ----------------------------------------------------
        # PLAYER ↔ PLAYER
        # ----------------------------------------------------

        for i in range(
            len(players)
        ):

            for j in range(
                len(players)
            ):

                if i == j:
                    continue

                source = players[i]
                target = players[j]

                relative_x = (
                    float(target["x"])
                    - float(source["x"])
                ) / width

                relative_y = (
                    float(target["y"])
                    - float(source["y"])
                ) / height

                distance = math.sqrt(
                    relative_x ** 2
                    + relative_y ** 2
                )

                team_a = source.get(
                    "team"
                )

                team_b = target.get(
                    "team"
                )

                if (
                    team_a in (0, 1)
                    and team_b in (0, 1)
                ):
                    same_team = float(
                        team_a == team_b
                    )
                else:
                    same_team = 0.0

                edge_list.append(
                    [i, j]
                )

                edge_attributes.append(
                    [
                        relative_x,
                        relative_y,
                        distance,
                        same_team
                    ]
                )

                edge_types.append(
                    PLAYER_PLAYER_EDGE_TYPE
                )

        # ----------------------------------------------------
        # PLAYER ↔ BALL
        # ----------------------------------------------------

        ball_index = len(players)

        for i, player in enumerate(
            players
        ):

            relative_x = (
                float(ball["x"])
                - float(player["x"])
            ) / width

            relative_y = (
                float(ball["y"])
                - float(player["y"])
            ) / height

            distance = math.sqrt(
                relative_x ** 2
                + relative_y ** 2
            )

            # Player -> Ball
            edge_list.append(
                [i, ball_index]
            )

            edge_attributes.append(
                [
                    relative_x,
                    relative_y,
                    distance,
                    0.0
                ]
            )

            edge_types.append(
                PLAYER_BALL_EDGE_TYPE
            )

            # Ball -> Player
            edge_list.append(
                [ball_index, i]
            )

            edge_attributes.append(
                [
                    -relative_x,
                    -relative_y,
                    distance,
                    0.0
                ]
            )

            edge_types.append(
                PLAYER_BALL_EDGE_TYPE
            )

        edge_index = torch.tensor(
            edge_list,
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_attributes,
            dtype=torch.float32
        )

        edge_type = torch.tensor(
            edge_types,
            dtype=torch.long
        )

        return (
            edge_index,
            edge_attr,
            edge_type
        )

    # ========================================================
    # GRAPH VALIDATION
    # ========================================================

    def validate_graph(
        self,
        graph
    ):
        """
        Validate the final feature + interaction graph.
        """

        # ----------------------------------------------------
        # Node feature graph
        # ----------------------------------------------------

        assert graph.x.dim() == 2

        assert (
            graph.x.size(1)
            == NODE_FEATURE_DIM
        )

        # ----------------------------------------------------
        # Interaction graph
        # ----------------------------------------------------

        assert graph.edge_index.dim() == 2

        assert (
            graph.edge_index.size(0)
            == 2
        )

        # ----------------------------------------------------
        # Edge feature graph
        # ----------------------------------------------------

        assert graph.edge_attr.dim() == 2

        assert (
            graph.edge_attr.size(0)
            == graph.edge_index.size(1)
        )

        assert (
            graph.edge_attr.size(1)
            == 4
        )

        # ----------------------------------------------------
        # Edge types
        # ----------------------------------------------------

        assert graph.edge_type.dim() == 1

        assert (
            graph.edge_type.size(0)
            == graph.edge_index.size(1)
        )

        # ----------------------------------------------------
        # Node types
        # ----------------------------------------------------

        assert graph.node_type.dim() == 1

        assert (
            graph.node_type.size(0)
            == graph.num_nodes
        )

        # ----------------------------------------------------
        # Node teams
        # ----------------------------------------------------

        assert hasattr(
            graph,
            "node_team"
        )

        assert graph.node_team.dim() == 1

        assert (
            graph.node_team.size(0)
            == graph.num_nodes
        )

        # Valid team values:
        #
        # 0 = Team 0
        # 1 = Team 1
        # -1 = Ball / no team

        assert torch.all(
            (
                graph.node_team == 0
            )
            |
            (
                graph.node_team == 1
            )
            |
            (
                graph.node_team == -1
            )
        )

        # ----------------------------------------------------
        # Valid node types
        # ----------------------------------------------------

        assert torch.all(
            (
                graph.node_type
                == PLAYER_NODE_INDEX
            )
            |
            (
                graph.node_type
                == BALL_NODE_INDEX
            )
        )

        # ----------------------------------------------------
        # Valid edge types
        # ----------------------------------------------------

        assert torch.all(
            (
                graph.edge_type
                == PLAYER_PLAYER_EDGE_TYPE
            )
            |
            (
                graph.edge_type
                == PLAYER_BALL_EDGE_TYPE
            )
        )

        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        assert torch.isfinite(
            graph.x
        ).all()

        assert torch.isfinite(
            graph.edge_attr
        ).all()

        # ----------------------------------------------------
        # Edge indices must be valid
        # ----------------------------------------------------

        assert torch.all(
            graph.edge_index >= 0
        )

        assert torch.all(
            graph.edge_index
            < graph.num_nodes
        )

        return True

    # ========================================================
    # COMPLETE GRAPH
    # ========================================================

    def build_graph(
        self,
        players,
        ball,
        court,
        frame_id=None,
        timestamp=None
    ):
        """
        Build the complete graph for one frame.
        """

        self.validate_frame_inputs(
            players,
            ball,
            court
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        player_ids = [
            player.get(
                "id",
                idx
            )
            for idx, player
            in enumerate(players)
        ]

        # ----------------------------------------------------
        # FEATURE GRAPH / NODES
        # ----------------------------------------------------

        node_features = []
        node_types = []
        node_teams = []

        # ----------------------------------------------------
        # PLAYER NODES
        # ----------------------------------------------------

        for player in players:

            # 49-dimensional player feature vector
            node_features.append(
                self.build_player_features(
                    player,
                    court
                )
            )

            # Metadata node type
            node_types.append(
                PLAYER_NODE_INDEX
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Store the actual team ID separately.
            #
            # Team 0 -> 0
            # Team 1 -> 1
            # Unknown -> -1
            # ------------------------------------------------

            team = player.get(
                "team",
                -1
            )

            if team not in (0, 1):
                team = -1

            node_teams.append(
                int(team)
            )

        # ----------------------------------------------------
        # BALL NODE
        # ----------------------------------------------------

        node_features.append(
            self.build_ball_features(
                ball,
                court
            )
        )

        node_types.append(
            BALL_NODE_INDEX
        )

        # Ball does not belong to a team.
        node_teams.append(-1)

        # ----------------------------------------------------
        # NODE FEATURE TENSOR
        # ----------------------------------------------------

        x = torch.tensor(
            np.asarray(
                node_features,
                dtype=np.float32
            ),
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # INTERACTION GRAPH / EDGES
        # ----------------------------------------------------

        (
            edge_index,
            edge_attr,
            edge_type
        ) = self.build_edges(
            players,
            ball,
            court
        )

        # ----------------------------------------------------
        # PYTORCH GEOMETRIC DATA
        # ----------------------------------------------------

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        # Edge metadata
        graph.edge_type = edge_type

        # Node metadata
        graph.node_type = torch.tensor(
            node_types,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # THIS IS THE IMPORTANT TASK 4 FIX
        #
        # For 6 players + 1 ball:
        #
        # node_team =
        # [0, 0, 0, 1, 1, 1, -1]
        #
        # Shape:
        # [7]
        # ----------------------------------------------------

        graph.node_team = torch.tensor(
            node_teams,
            dtype=torch.long
        )

        graph.player_ids = player_ids

        graph.ball_index = len(
            players
        )

        if frame_id is not None:
            graph.frame_id = frame_id

        if timestamp is not None:
            graph.timestamp = timestamp

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        self.validate_graph(
            graph
        )

        return graph