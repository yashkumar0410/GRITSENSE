
import csv
import json
from pathlib import Path

import numpy as np


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


# ============================================================
# PROJECT ROOT
# ============================================================

VIP_ROOT = Path(__file__).resolve().parent

BALL_FILE = VIP_ROOT / "ball.csv"
PLAYER_FILE = VIP_ROOT / "player_keypoints.json"
COURT_FILE = VIP_ROOT / "annotations.json"


# ============================================================
# LOAD ALL FRAME IDS FROM BALL.CSV
# ============================================================

def get_available_frames():
    """
    Read ball.csv and return every frame ID that has valid data.
    """

    if not BALL_FILE.exists():
        raise FileNotFoundError(
            f"ball.csv not found:\n{BALL_FILE}"
        )

    frames = []

    with BALL_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "ball.csv does not contain a header."
            )

        print(
            "\nBall CSV columns:",
            reader.fieldnames
        )

        # ----------------------------------------------------
        # Find frame column
        # ----------------------------------------------------

        frame_column = None

        possible_names = [
            "Frame",
            "frame",
            "frame_id",
            "FrameID",
            "frameId",
        ]

        for name in possible_names:

            if name in reader.fieldnames:
                frame_column = name
                break

        if frame_column is None:
            raise ValueError(
                "Could not find frame column in ball.csv.\n"
                f"Available columns: {reader.fieldnames}"
            )

        # ----------------------------------------------------
        # Read frame IDs
        # ----------------------------------------------------

        for row in reader:

            value = row.get(frame_column)

            if value is None or value == "":
                continue

            try:

                frame_id = int(
                    float(value)
                )

                frames.append(
                    frame_id
                )

            except (
                ValueError,
                TypeError
            ):

                continue

    frames = sorted(
        set(frames)
    )

    if not frames:

        raise ValueError(
            "No valid frame IDs were found in ball.csv."
        )

    print(
        f"\nFound {len(frames)} frames in ball.csv."
    )

    print(
        f"First frame: {frames[0]}"
    )

    print(
        f"Last frame: {frames[-1]}"
    )

    return frames


# ============================================================
# LOAD REAL DATA FOR ONE FRAME
# ============================================================

def _load_real_sample_data(frame_id):
    """
    Load players, ball and court information for one frame.
    """

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not BALL_FILE.exists():

        raise FileNotFoundError(
            f"ball.csv not found:\n{BALL_FILE}"
        )

    if not PLAYER_FILE.exists():

        raise FileNotFoundError(
            f"player_keypoints.json not found:\n{PLAYER_FILE}"
        )

    if not COURT_FILE.exists():

        raise FileNotFoundError(
            f"annotations.json not found:\n{COURT_FILE}"
        )

    # ========================================================
    # BALL
    # ========================================================

    ball = None

    with BALL_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames or []

        # ----------------------------------------------------
        # Find columns
        # ----------------------------------------------------

        frame_column = None
        x_column = None
        y_column = None
        visibility_column = None

        for name in [
            "Frame",
            "frame",
            "frame_id",
            "FrameID",
            "frameId",
        ]:

            if name in fieldnames:

                frame_column = name

                break

        for name in [
            "X",
            "x",
            "ball_x",
            "Ball_X",
            "center_x",
        ]:

            if name in fieldnames:

                x_column = name

                break

        for name in [
            "Y",
            "y",
            "ball_y",
            "Ball_Y",
            "center_y",
        ]:

            if name in fieldnames:

                y_column = name

                break

        for name in [
            "Visibility",
            "visibility",
            "visible",
            "Visible",
        ]:

            if name in fieldnames:

                visibility_column = name

                break

        if frame_column is None:

            raise ValueError(
                f"No frame column found in ball.csv.\n"
                f"Columns: {fieldnames}"
            )

        if (
            x_column is None
            or y_column is None
        ):

            raise ValueError(
                f"Could not find ball X/Y columns.\n"
                f"Columns: {fieldnames}"
            )

        # ----------------------------------------------------
        # Find requested frame
        # ----------------------------------------------------

        for row in reader:

            try:

                current_frame = int(
                    float(
                        row[frame_column]
                    )
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            if current_frame != frame_id:
                continue

            try:

                x = float(
                    row[x_column]
                )

                y = float(
                    row[y_column]
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            # ------------------------------------------------
            # Visibility
            # ------------------------------------------------

            visible = True

            if visibility_column is not None:

                try:

                    visible = (
                        int(
                            float(
                                row[
                                    visibility_column
                                ]
                            )
                        )
                        == 1
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    visible = False

            if not visible:
                continue

            if x < 0 or y < 0:
                continue

            ball = {
                "x": x,
                "y": y,
                "frame": frame_id,
            }

            break

    # ========================================================
    # PLAYER DATA
    # ========================================================

    with PLAYER_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        player_data = json.load(f)

    # --------------------------------------------------------
    # Locate frame
    # --------------------------------------------------------

    frame_key = str(
        frame_id
    )

    frame_players = None

    if isinstance(
        player_data,
        dict
    ):

        # ----------------------------------------------------
        # Your actual JSON format:
        #
        # "315": {
        #     "1": {...},
        #     "2": {...},
        #     "3": {...}
        # }
        # ----------------------------------------------------

        if frame_key in player_data:

            frame_players = (
                player_data[frame_key]
            )

        elif frame_id in player_data:

            frame_players = (
                player_data[frame_id]
            )

        elif "frames" in player_data:

            frames_data = (
                player_data["frames"]
            )

            if isinstance(
                frames_data,
                dict
            ):

                if frame_key in frames_data:

                    frame_players = (
                        frames_data[
                            frame_key
                        ]
                    )

                elif frame_id in frames_data:

                    frame_players = (
                        frames_data[
                            frame_id
                        ]
                    )

    elif isinstance(
        player_data,
        list
    ):

        for item in player_data:

            if not isinstance(
                item,
                dict
            ):
                continue

            item_frame = (
                item.get("frame")
                or item.get("frame_id")
                or item.get("Frame")
            )

            try:

                item_frame = int(
                    float(item_frame)
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            if item_frame == frame_id:

                frame_players = (
                    item.get("players")
                    or item.get("detections")
                    or item.get("data")
                )

                break

    if frame_players is None:

        raise ValueError(
            f"No player data found "
            f"for frame {frame_id}"
        )

    # ========================================================
    # CONVERT DICTIONARY -> LIST
    # ========================================================

    if isinstance(
        frame_players,
        dict
    ):

        frame_players = list(
            frame_players.values()
        )

    elif not isinstance(
        frame_players,
        list
    ):

        raise ValueError(
            f"Player data for frame "
            f"{frame_id} is neither "
            f"a dictionary nor a list."
        )

    # ========================================================
    # NORMALIZE PLAYER DATA
    # ========================================================

    players = []

    for idx, player in enumerate(
        frame_players
    ):

        if not isinstance(
            player,
            dict
        ):

            continue

        # ----------------------------------------------------
        # Position
        #
        # Your JSON contains:
        #
        # "image_x": 1415.5
        # "image_y": 849.0
        # ----------------------------------------------------

        x = player.get(
            "image_x"
        )

        y = player.get(
            "image_y"
        )

        # Generic fallback

        if x is None:

            x = player.get(
                "x"
            )

        if x is None:

            x = player.get(
                "center_x"
            )

        if y is None:

            y = player.get(
                "y"
            )

        if y is None:

            y = player.get(
                "center_y"
            )

        # ----------------------------------------------------
        # Bounding box fallback
        # ----------------------------------------------------

        if (
            (x is None or y is None)
            and "bbox" in player
        ):

            bbox = player[
                "bbox"
            ]

            if (
                isinstance(
                    bbox,
                    (list, tuple)
                )
                and len(bbox) >= 4
            ):

                try:

                    x1, y1, x2, y2 = map(
                        float,
                        bbox[:4]
                    )

                    # Bottom-center of bounding box

                    x = (
                        x1 + x2
                    ) / 2.0

                    y = y2

                except (
                    ValueError,
                    TypeError
                ):

                    pass

        if x is None or y is None:
            continue

        try:

            x = float(x)
            y = float(y)

        except (
            ValueError,
            TypeError
        ):

            continue

        # ----------------------------------------------------
        # Team
        # ----------------------------------------------------

        team = player.get(
            "team"
        )

        if team is None:

            team = player.get(
                "team_id"
            )

        if team is None:

            team = player.get(
                "class_id"
            )

        try:

            team = (
                int(team)
                if team is not None
                else -1
            )

        except (
            ValueError,
            TypeError
        ):

            team = -1

        # ----------------------------------------------------
        # Pose / keypoints
        # ----------------------------------------------------

        keypoints = (
            player.get("keypoints")
            or player.get("pose")
            or player.get("kpts")
        )

        pose_available = False

        pose = []

        if keypoints is not None:

            if isinstance(
                keypoints,
                list
            ):

                # ------------------------------------------------
                # Flatten nested keypoints
                # ------------------------------------------------

                if (
                    len(keypoints) > 0
                    and isinstance(
                        keypoints[0],
                        (list, tuple)
                    )
                ):

                    flattened = []

                    for kp in keypoints:

                        if isinstance(
                            kp,
                            (list, tuple)
                        ):

                            flattened.extend(
                                kp
                            )

                    keypoints = flattened

                try:

                    pose = [
                        float(v)
                        for v in keypoints
                    ]

                    # Original pipeline expects 34 values.

                    if len(pose) == 34:

                        pose_available = True

                except (
                    ValueError,
                    TypeError
                ):

                    pose = []

                    pose_available = False

        # ----------------------------------------------------
        # Player ID
        # ----------------------------------------------------

        player_id = player.get(
            "id"
        )

        if player_id is None:

            player_id = player.get(
                "track_id"
            )

        if player_id is None:

            player_id = idx

        # ----------------------------------------------------
        # Create normalized player
        # ----------------------------------------------------

        player_entry = {

            "id": player_id,

            "x": x,

            "y": y,

            "team": team,

            "pose": pose,

            "pose_available":
                pose_available,
        }

        players.append(
            player_entry
        )

    # ========================================================
    # TEAM FALLBACK
    # ========================================================
    #
    # Your JSON currently has:
    #
    # "team": null
    #
    # Therefore infer teams when no valid team labels exist.
    # ========================================================

    valid_team_labels = [
        p["team"]
        for p in players
        if p["team"] in (0, 1)
    ]

    if (
        len(valid_team_labels) == 0
        and len(players) >= 2
    ):

        players.sort(
            key=lambda p: p["x"]
        )

        midpoint = (
            len(players) // 2
        )

        for i, player in enumerate(
            players
        ):

            if i < midpoint:

                player["team"] = 0

            else:

                player["team"] = 1

    # ========================================================
    # COURT
    # ========================================================

    with COURT_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        court_data = json.load(f)

    # --------------------------------------------------------
    # Normalize court into dictionary
    # --------------------------------------------------------

    if isinstance(
        court_data,
        dict
    ):

        court = dict(
            court_data
        )

    else:

        court = {
            "points": court_data
        }

    # ========================================================
    # REQUIRED COURT DIMENSIONS
    # ========================================================
    #
    # graph_builder.py requires:
    #
    #     court["width"]
    #     court["height"]
    #
    # Your source annotations don't contain these, so use
    # the actual video/image dimensions.
    # ========================================================

    if "width" not in court:

        court["width"] = (
            IMAGE_WIDTH
        )

    if "height" not in court:

        court["height"] = (
            IMAGE_HEIGHT
        )

    # ========================================================
    # NET POSITION
    # ========================================================

    if "net_x" not in court:

        court["net_x"] = (
            IMAGE_WIDTH / 2
        )

    # ========================================================
    # COURT POINTS
    # ========================================================

    if "points" not in court:

        if "court_points" in court:

            court["points"] = (
                court["court_points"]
            )

        elif "keypoints" in court:

            court["points"] = (
                court["keypoints"]
            )

        else:

            court["points"] = []

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if ball is None:

        raise ValueError(
            f"No valid ball detection "
            f"for frame {frame_id}"
        )

    if len(players) == 0:

        raise ValueError(
            f"No valid players "
            f"for frame {frame_id}"
        )

    if "width" not in court:

        raise ValueError(
            "court must contain width "
            "and height."
        )

    if "height" not in court:

        raise ValueError(
            "court must contain width "
            "and height."
        )

    return players, ball, court


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def create_sample_data(frame_id):
    """
    Load real data for a particular frame.

    No synthetic fallback is used.
    """

    return _load_real_sample_data(
        frame_id
    )
