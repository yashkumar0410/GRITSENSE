import numpy as np


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


# ============================================================
# SYNTHETIC PLAYER POSE
# ============================================================

def build_pose_for_player(
    player_id: int,
    base_x: float,
    base_y: float,
):
    """
    Build a deterministic synthetic 17-keypoint pose
    for one player.

    The pose is centered around the player's (x, y)
    position so that players scattered around the court
    also have corresponding scattered poses.
    """

    keypoints = [
        # Head
        (base_x, base_y - 120),

        # Shoulders
        (base_x - 25, base_y - 95),
        (base_x + 25, base_y - 95),

        # Elbows
        (base_x - 55, base_y - 55),
        (base_x + 55, base_y - 55),

        # Wrists
        (base_x - 70, base_y - 15),
        (base_x + 70, base_y - 15),

        # Hips
        (base_x - 25, base_y),
        (base_x + 25, base_y),

        # Knees
        (base_x - 45, base_y + 70),
        (base_x + 45, base_y + 70),

        # Ankles
        (base_x - 50, base_y + 145),
        (base_x + 50, base_y + 145),

        # Additional body points
        (base_x - 35, base_y - 155),
        (base_x + 35, base_y - 155),
        (base_x - 20, base_y + 175),
        (base_x + 20, base_y + 175),
    ]

    flat_pose = []

    for x, y in keypoints:
        flat_pose.extend([x, y])

    return np.asarray(
        flat_pose,
        dtype=np.float32,
    )


# ============================================================
# SYNTHETIC VOLLEYBALL FRAME
# ============================================================

def create_sample_data():
    """
    Create one synthetic volleyball frame.

    Structure:

        Team 0: 6 players
        Team 1: 6 players
        Ball:   1

        Total = 13 nodes

    Players are intentionally scattered around their
    respective court halves to resemble a real volleyball
    formation rather than a simple grid.
    """

    # ========================================================
    # TEAM 0
    # ========================================================

    team_0_positions = [
        # Front-left
        (380, 380),

        # Front-middle
        (720, 300),

        # Front-right
        (880, 470),

        # Back-left
        (300, 760),

        # Back-middle
        (650, 850),

        # Back-right
        (900, 700),
    ]

    # ========================================================
    # TEAM 1
    # ========================================================

    team_1_positions = [
        # Front-left
        (1040, 450),

        # Front-middle
        (1180, 300),

        # Front-right
        (1530, 400),

        # Back-left
        (1080, 780),

        # Back-middle
        (1370, 850),

        # Back-right
        (1660, 690),
    ]

    players = []

    # ========================================================
    # CREATE TEAM 0 PLAYERS
    # ========================================================

    for index, (
        x,
        y,
    ) in enumerate(
        team_0_positions
    ):

        player_id = index + 1

        players.append(
            {
                "id": player_id,

                "x": x,
                "y": y,

                "vx": (
                    1.0
                    + index * 0.25
                ),

                "vy": (
                    -0.5
                    + index * 0.15
                ),

                "team": 0,

                "confidence": (
                    0.92
                    + (index % 4) * 0.02
                ),

                "pose":
                    build_pose_for_player(
                        player_id,
                        x,
                        y,
                    ),
            }
        )

    # ========================================================
    # CREATE TEAM 1 PLAYERS
    # ========================================================

    for index, (
        x,
        y,
    ) in enumerate(
        team_1_positions
    ):

        player_id = index + 7

        players.append(
            {
                "id": player_id,

                "x": x,
                "y": y,

                "vx": (
                    -1.0
                    + index * 0.20
                ),

                "vy": (
                    0.5
                    - index * 0.12
                ),

                "team": 1,

                "confidence": (
                    0.92
                    + (index % 4) * 0.02
                ),

                "pose":
                    build_pose_for_player(
                        player_id,
                        x,
                        y,
                    ),
            }
        )

    # ========================================================
    # BALL
    # ========================================================

    # Ball is close to the net and slightly above
    # the center of the court.

    ball = {
        "x": 1010,
        "y": 420,

        "vx": 3.0,
        "vy": -2.0,

        "confidence": 0.98,
    }

    # ========================================================
    # COURT
    # ========================================================

    court = {
        "width":
            IMAGE_WIDTH,

        "height":
            IMAGE_HEIGHT,

        "net_x":
            IMAGE_WIDTH / 2,
    }

    return (
        players,
        ball,
        court,
    )