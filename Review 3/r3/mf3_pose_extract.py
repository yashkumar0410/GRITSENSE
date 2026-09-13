import json
import os

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "sample3.mp4"

MF3_JSON = "player_keypoints.json"

POSE_MODEL_PATH = r"D:\capstone\recent\GRITSENSE\pose.pt"

OUTPUT_JSON = "player_keypoints_mf3_pose.json"

POSE_CONF = 0.30


# COCO 17-keypoint skeleton
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


# ============================================================
# LOAD MF3 DATA
# ============================================================

if not os.path.exists(MF3_JSON):
    raise FileNotFoundError(
        f"MF3 JSON not found:\n{os.path.abspath(MF3_JSON)}"
    )

with open(MF3_JSON, "r", encoding="utf-8") as f:
    mf3_data = json.load(f)


# ============================================================
# LOAD POSE MODEL
# ============================================================

if not os.path.exists(POSE_MODEL_PATH):
    raise FileNotFoundError(
        f"Pose model not found:\n"
        f"{os.path.abspath(POSE_MODEL_PATH)}"
    )

print()
print("=" * 70)
print("LOADING POSE MODEL")
print("=" * 70)

pose_model = YOLO(POSE_MODEL_PATH)

print("Pose model loaded successfully.")


# ============================================================
# OPEN VIDEO
# ============================================================

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Video not found:\n"
        f"{os.path.abspath(VIDEO_PATH)}"
    )

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n"
        f"{os.path.abspath(VIDEO_PATH)}"
    )

total_video_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


print()
print("VIDEO")
print("-" * 40)
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Frames:", total_video_frames)


# ============================================================
# OUTPUT DATA
# ============================================================

output_data = {}

total_players = 0
successful_poses = 0
failed_poses = 0


# ============================================================
# PROCESS FRAMES
# ============================================================

frame_ids = sorted(
    [int(k) for k in mf3_data.keys()]
)

print()
print("=" * 70)
print("MF3 ID -> POSE EXTRACTION")
print("=" * 70)

for count, frame_id in enumerate(frame_ids, start=1):

    # --------------------------------------------------------
    # Seek directly to the MF3 frame.
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_id
    )

    ret, frame = cap.read()

    if not ret:

        print(
            f"[{count}/{len(frame_ids)}] "
            f"Frame {frame_id}: VIDEO READ FAILED"
        )

        continue


    # --------------------------------------------------------
    # MF3 players for this frame.
    # --------------------------------------------------------

    frame_players = mf3_data[
        str(frame_id)
    ]

    if not isinstance(frame_players, dict):
        continue


    output_data[str(frame_id)] = {}


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # MF3 ID is the identity source.
    #
    # We DO NOT enumerate pose detections.
    # We process each MF3 player's bbox separately.
    # --------------------------------------------------------

    for mf3_id, player in frame_players.items():

        if not isinstance(player, dict):
            continue


        total_players += 1


        # ----------------------------------------------------
        # Read MF3 bounding box.
        # ----------------------------------------------------

        bbox = player.get("bbox")

        if (
            not isinstance(bbox, list)
            or len(bbox) < 4
        ):

            failed_poses += 1

            new_player = dict(player)

            new_player["keypoints"] = []
            new_player["pose_available"] = False
            new_player["pose_confidence"] = 0.0

            output_data[str(frame_id)][
                str(mf3_id)
            ] = new_player

            continue


        try:

            x1, y1, x2, y2 = map(
                int,
                bbox[:4]
            )

        except (
            ValueError,
            TypeError
        ):

            failed_poses += 1

            new_player = dict(player)

            new_player["keypoints"] = []
            new_player["pose_available"] = False
            new_player["pose_confidence"] = 0.0

            output_data[str(frame_id)][
                str(mf3_id)
            ] = new_player

            continue


        # ----------------------------------------------------
        # Clamp bbox to image.
        # ----------------------------------------------------

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))


        if x2 <= x1 or y2 <= y1:

            failed_poses += 1

            new_player = dict(player)

            new_player["keypoints"] = []
            new_player["pose_available"] = False
            new_player["pose_confidence"] = 0.0

            output_data[str(frame_id)][
                str(mf3_id)
            ] = new_player

            continue


        # ----------------------------------------------------
        # CROP EXACT MF3 PLAYER
        # ----------------------------------------------------

        player_crop = frame[
            y1:y2,
            x1:x2
        ]


        if player_crop.size == 0:

            failed_poses += 1

            new_player = dict(player)

            new_player["keypoints"] = []
            new_player["pose_available"] = False
            new_player["pose_confidence"] = 0.0

            output_data[str(frame_id)][
                str(mf3_id)
            ] = new_player

            continue


        # ----------------------------------------------------
        # RUN POSE ON THIS MF3 PLAYER ONLY
        # ----------------------------------------------------

        try:

            pose_results = pose_model(
                player_crop,
                conf=POSE_CONF,
                verbose=False
            )

        except Exception as exc:

            print(
                f"\nPose error "
                f"frame={frame_id}, "
                f"ID={mf3_id}: {exc}"
            )

            failed_poses += 1

            new_player = dict(player)

            new_player["keypoints"] = []
            new_player["pose_available"] = False
            new_player["pose_confidence"] = 0.0

            output_data[str(frame_id)][
                str(mf3_id)
            ] = new_player

            continue


        # ----------------------------------------------------
        # CHECK KEYPOINTS
        # ----------------------------------------------------

        if (
            len(pose_results) == 0
            or pose_results[0].keypoints is None
        ):

            failed_poses += 1

            new_player = dict(player)

            new_player["keypoints"] = []
            new_player["pose_available"] = False
            new_player["pose_confidence"] = 0.0

            output_data[str(frame_id)][
                str(mf3_id)
            ] = new_player

            continue


        keypoints_xy = (
            pose_results[0]
            .keypoints
            .xy
            .cpu()
            .numpy()
        )


        if len(keypoints_xy) == 0:

            failed_poses += 1

            new_player = dict(player)

            new_player["keypoints"] = []
            new_player["pose_available"] = False
            new_player["pose_confidence"] = 0.0

            output_data[str(frame_id)][
                str(mf3_id)
            ] = new_player

            continue


        # ----------------------------------------------------
        # TAKE FIRST PERSON IN THIS CROP
        # ----------------------------------------------------

        person = keypoints_xy[0]


        if person.shape[0] < 17:

            failed_poses += 1

            new_player = dict(player)

            new_player["keypoints"] = []
            new_player["pose_available"] = False
            new_player["pose_confidence"] = 0.0

            output_data[str(frame_id)][
                str(mf3_id)
            ] = new_player

            continue


        # ----------------------------------------------------
        # CONVERT CROP COORDINATES
        # BACK TO FULL FRAME
        # ----------------------------------------------------

        full_frame_keypoints = []

        for x, y in person[:17]:

            px = float(x + x1)
            py = float(y + y1)

            full_frame_keypoints.append(
                [px, py]
            )


        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        arr = np.asarray(
            full_frame_keypoints,
            dtype=np.float32
        )


        if (
            arr.shape != (17, 2)
            or not np.isfinite(arr).all()
        ):

            failed_poses += 1

            new_player = dict(player)

            new_player["keypoints"] = []
            new_player["pose_available"] = False
            new_player["pose_confidence"] = 0.0

            output_data[str(frame_id)][
                str(mf3_id)
            ] = new_player

            continue


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        new_player = dict(player)

        # MF3 remains authoritative.
        new_player["id"] = int(
            player.get(
                "id",
                mf3_id
            )
        )

        new_player["keypoints"] = (
            arr.tolist()
        )

        new_player["pose"] = (
            arr.tolist()
        )

        new_player["pose_available"] = True

        new_player["pose_confidence"] = 1.0

        output_data[str(frame_id)][
            str(mf3_id)
        ] = new_player

        successful_poses += 1


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        count % 10 == 0
        or count == len(frame_ids)
    ):

        percentage = (
            100.0
            * successful_poses
            / total_players
            if total_players
            else 0.0
        )

        print(
            f"[{count}/{len(frame_ids)}] "
            f"Frame {frame_id} | "
            f"poses {successful_poses}/"
            f"{total_players} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output_data,
        f,
        indent=2
    )


percentage = (
    100.0
    * successful_poses
    / total_players
    if total_players
    else 0.0
)


print()
print("=" * 70)
print("POSE EXTRACTION COMPLETE ✓")
print("=" * 70)

print("MF3 player records:", total_players)
print("Successful poses:", successful_poses)
print("Failed poses:", failed_poses)

print(
    f"Pose availability: "
    f"{percentage:.2f}%"
)

print()
print("Output:")
print(
    os.path.abspath(
        OUTPUT_JSON
    )
)

print()
print(
    "IMPORTANT: MF3 IDs were preserved."
)
print(
    "Pose coordinates are stored in full-frame coordinates."
)
print(
    "Each pose contains exactly 17 keypoints."
)


