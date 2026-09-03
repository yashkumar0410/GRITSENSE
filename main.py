import cv2
import torch
from ultralytics import YOLO
import numpy as np
import json


# ============================================================
# MODELS
# ============================================================

player_model = YOLO("player_detection.pt")
pose_model = YOLO("pose.pt")
ball_model = YOLO("ball.pt")

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "sample3.mp4"

COURT_POINTS_PATH = "annotations.json"

COURT_IMAGE_PATH = "court.png"

OUTPUT_VIDEO_PATH = "output.mp4"

POSE_OUTPUT_PATH = "player_keypoints.json"

COURT_WIDTH = 300
COURT_HEIGHT = 150

ACTUAL_COURT_WIDTH = 18
ACTUAL_COURT_HEIGHT = 9

MARGIN = 20

COURT_ALPHA = 0.6

POSE_CONFIDENCE = 0.25

PLAYER_CONFIDENCE = 0.4

POSE_IOU_THRESHOLD = 0.25


# IDs that should not be considered players
EXCLUDED_IDS = {
    20,
    63,
    69
}


# BGR colors
TEAM_LEFT_COLOR = (
    255,
    0,
    0
)

TEAM_RIGHT_COLOR = (
    0,
    0,
    255
)

BALL_COLOR = (
    0,
    165,
    255
)

COURT_LINE_COLOR = (
    0,
    255,
    255
)

CORNER_COLOR = (
    0,
    0,
    0
)

POSE_COLOR = (
    0,
    255,
    0
)


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box1,
    box2
):
    """
    Calculate IoU between two bounding boxes.

    Box format:

        [x1, y1, x2, y2]
    """

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        /
        union
    )


# ============================================================
# MATCH POSE TO TRACKED PLAYERS
# ============================================================

def match_pose_to_players(
    player_boxes,
    player_ids,
    pose_result
):
    """
    Match pose detections to tracked players using IoU.

    Returns:

        {
            player_id: {
                "keypoints": [
                    x1, y1,
                    x2, y2,
                    ...
                    x17, y17
                ],
                "pose_confidence": float,
                "iou": float
            }
        }
    """

    matched = {}

    # --------------------------------------------------------
    # Check pose detections
    # --------------------------------------------------------

    if (
        pose_result.boxes is None
        or len(pose_result.boxes) == 0
    ):
        return matched

    if pose_result.keypoints is None:
        return matched

    # --------------------------------------------------------
    # Pose bounding boxes
    # --------------------------------------------------------

    pose_boxes = (
        pose_result
        .boxes
        .xyxy
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Pose keypoints
    # --------------------------------------------------------

    pose_keypoints = (
        pose_result
        .keypoints
        .xy
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Pose confidence
    # --------------------------------------------------------

    pose_confidences = None

    if (
        pose_result.keypoints.conf
        is not None
    ):

        pose_confidences = (
            pose_result
            .keypoints
            .conf
            .cpu()
            .numpy()
        )

    used_pose_indices = set()

    # ========================================================
    # MATCH EACH TRACKED PLAYER
    # ========================================================

    for player_index, player_id in enumerate(
        player_ids
    ):

        player_box = (
            player_boxes[player_index]
        )

        best_iou = 0.0

        best_pose_index = None

        # ----------------------------------------------------
        # Find best pose box
        # ----------------------------------------------------

        for pose_index, pose_box in enumerate(
            pose_boxes
        ):

            if pose_index in used_pose_indices:
                continue

            iou = calculate_iou(
                player_box,
                pose_box
            )

            if iou > best_iou:

                best_iou = iou

                best_pose_index = pose_index

        # ----------------------------------------------------
        # Accept match
        # ----------------------------------------------------

        if (
            best_pose_index is None
            or best_iou < POSE_IOU_THRESHOLD
        ):
            continue

        keypoints = (
            pose_keypoints[
                best_pose_index
            ]
        )

        # ----------------------------------------------------
        # Flatten 17 x 2 keypoints
        #
        # [x1,y1,x2,y2,...,x17,y17]
        # ----------------------------------------------------

        flat_keypoints = []

        for x, y in keypoints:

            flat_keypoints.extend(
                [
                    float(x),
                    float(y)
                ]
            )

        # ----------------------------------------------------
        # Pose confidence
        # ----------------------------------------------------

        if pose_confidences is not None:

            confidence = float(
                np.mean(
                    pose_confidences[
                        best_pose_index
                    ]
                )
            )

        else:

            confidence = 1.0

        matched[player_id] = {

            "keypoints": flat_keypoints,

            "pose_confidence": confidence,

            "iou": float(best_iou)
        }

        used_pose_indices.add(
            best_pose_index
        )

    return matched


# ============================================================
# DRAW PLAYER KEYPOINTS
# ============================================================

def draw_player_keypoints(
    frame,
    pose_matches
):
    """
    Draw the 17 pose keypoints for every
    successfully matched player.
    """

    for player_id, pose_data in pose_matches.items():

        keypoints = pose_data[
            "keypoints"
        ]

        # ----------------------------------------------------
        # 17 keypoints
        # ----------------------------------------------------

        for k in range(
            0,
            len(keypoints),
            2
        ):

            x = int(
                keypoints[k]
            )

            y = int(
                keypoints[k + 1]
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                POSE_COLOR,
                -1
            )

        # ----------------------------------------------------
        # Player ID near keypoints
        # ----------------------------------------------------

        if len(keypoints) >= 2:

            first_x = int(
                keypoints[0]
            )

            first_y = int(
                keypoints[1]
            )

            cv2.putText(
                frame,
                f"P{player_id}",
                (
                    first_x + 5,
                    first_y - 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                POSE_COLOR,
                1
            )


# ============================================================
# LOAD COURT POINTS
# ============================================================

with open(
    COURT_POINTS_PATH,
    "r",
    encoding="utf-8"
) as f:

    court_points_by_frame = json.load(
        f
    )


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if (
    fps is None
    or fps <= 1
):

    fps = 30


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = 0

    USE_HALF = True

    print(
        "Using CUDA"
    )

else:

    device = "cpu"

    USE_HALF = False

    print(
        "Using CPU"
    )


# ============================================================
# COURT IMAGE
# ============================================================

court_image = cv2.imread(
    COURT_IMAGE_PATH
)

if court_image is None:

    raise FileNotFoundError(
        f"Could not load {COURT_IMAGE_PATH}"
    )


court_image = cv2.resize(
    court_image,
    (
        COURT_WIDTH,
        COURT_HEIGHT
    )
)


base_court = court_image.copy()


# ============================================================
# COURT GEOMETRY
# ============================================================

court_pts = np.array(
    [
        [0, 0],
        [COURT_WIDTH, 0],
        [COURT_WIDTH, COURT_HEIGHT],
        [0, COURT_HEIGHT]
    ],
    dtype=np.float32
)


H = None


# ============================================================
# TRACKING STATE
# ============================================================

prev_positions = {}

prev_frame_idx = {}

player_speeds = {}

player_colors = {}


# ============================================================
# OUTPUT
# ============================================================

out = None


# ============================================================
# KEYPOINT OUTPUT
# ============================================================

all_player_keypoints = {}
all_ball_data = {}


# ============================================================
# FRAME LOOP
# ============================================================

frame_idx = 0


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------------------------
    # Fresh frame
    # --------------------------------------------------------

    annotated_frame = frame.copy()
        # ========================================================
    # BALL DETECTION
    # ========================================================

    ball_result = ball_model.predict(
        frame,
        conf=0.25,
        device=device,
        verbose=False
    )[0]

    ball_data = None

    if (
        ball_result.boxes is not None
        and len(ball_result.boxes) > 0
    ):

        # Take the highest-confidence ball detection
        best_idx = int(
            torch.argmax(
                ball_result.boxes.conf
            ).item()
        )

        ball_box = (
            ball_result.boxes.xyxy[
                best_idx
            ]
            .cpu()
            .numpy()
        )

        ball_confidence = float(
            ball_result.boxes.conf[
                best_idx
            ].item()
        )

        bx1, by1, bx2, by2 = ball_box

        ball_x = float(
            (bx1 + bx2) / 2.0
        )

        ball_y = float(
            (by1 + by2) / 2.0
        )

        ball_data = {
            "x": ball_x,
            "y": ball_y,
            "confidence": ball_confidence
        }

        # Draw ball
        cv2.circle(
            annotated_frame,
            (
                int(ball_x),
                int(ball_y)
            ),
            8,
            BALL_COLOR,
            -1
        )

        cv2.putText(
            annotated_frame,
            "BALL",
            (
                int(ball_x) + 10,
                int(ball_y) - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BALL_COLOR,
            2
        )


    # ========================================================
    # PLAYER TRACKING
    # ========================================================

    results = player_model.track(

        frame,

        persist=True,

        conf=PLAYER_CONFIDENCE,

        tracker="botsort.yaml",

        verbose=False,

        device=device,

        half=USE_HALF
    )


    # ========================================================
    # POSE DETECTION
    # ========================================================

    pose_results = pose_model.predict(

        frame,

        conf=POSE_CONFIDENCE,

        verbose=False,

        device=device,

        half=USE_HALF
    )


    # ========================================================
    # PROCESS TRACKED PLAYERS
    # ========================================================

    for r in results:

        boxes = r.boxes


        if (
            boxes.id is None
            or len(boxes.id) == 0
        ):
            continue


        # ----------------------------------------------------
        # IDs and bounding boxes
        # ----------------------------------------------------

        ids = (
            boxes
            .id
            .cpu()
            .numpy()
            .astype(int)
        )


        xyxy = (
            boxes
            .xyxy
            .cpu()
            .numpy()
            .astype(int)
        )


        # ====================================================
        # MATCH POSE TO TRACKED PLAYERS
        # ====================================================

        pose_matches = {}

        if len(pose_results) > 0:

            pose_matches = (
                match_pose_to_players(
                    xyxy,
                    ids,
                    pose_results[0]
                )
            )


        # ====================================================
        # DRAW REAL 17 KEYPOINTS
        # ====================================================

        draw_player_keypoints(
            annotated_frame,
            pose_matches
        )


        # ====================================================
        # SAVE KEYPOINT INFORMATION
        # ====================================================

        frame_player_data = {}


        for i in range(
            len(ids)
        ):

            track_id = int(
                ids[i]
            )


            if track_id in EXCLUDED_IDS:
                continue


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                xyxy[i]
            )


            # ------------------------------------------------
            # Player center / bottom point
            # ------------------------------------------------

            px = (
                x1 + x2
            ) // 2

            py = y2


            # ------------------------------------------------
            # Court coordinates
            # ------------------------------------------------

            mx = None
            my = None


            if H is not None:

                point = np.array(
                    [
                        [
                            [
                                px,
                                py
                            ]
                        ]
                    ],
                    dtype=np.float32
                )


                mapped = (
                    cv2.perspectiveTransform(
                        point,
                        H
                    )
                )


                mx, my = (
                    mapped[0][0]
                )


                mx = int(mx)
                my = int(my)


            # ------------------------------------------------
            # Team
            # ------------------------------------------------

            team = None


            if mx is not None:

                if (
                    mx
                    <
                    COURT_WIDTH // 2
                ):

                    team = 0

                else:

                    team = 1


            # ------------------------------------------------
            # Speed
            # ------------------------------------------------

            speed = 0.0


            if (
                mx is not None
                and my is not None
            ):

                if (
                    track_id
                    in prev_positions
                    and
                    track_id
                    in prev_frame_idx
                    and
                    (
                        frame_idx
                        -
                        prev_frame_idx[
                            track_id
                        ]
                    )
                    == 1
                ):

                    prev_mx, prev_my = (
                        prev_positions[
                            track_id
                        ]
                    )


                    dist_px = np.sqrt(

                        (
                            mx
                            -
                            prev_mx
                        ) ** 2

                        +

                        (
                            my
                            -
                            prev_my
                        ) ** 2
                    )


                    px_per_meter = (
                        COURT_WIDTH
                        /
                        ACTUAL_COURT_WIDTH
                    )


                    dist_m = (
                        dist_px
                        /
                        px_per_meter
                    )


                    speed = (
                        dist_m
                        *
                        fps
                    )


                prev_positions[
                    track_id
                ] = (
                    mx,
                    my
                )


                prev_frame_idx[
                    track_id
                ] = frame_idx


            player_speeds[
                track_id
            ] = speed


            # =================================================
            # TEAM COLOR
            # =================================================

            if team == 0:

                player_color = (
                    TEAM_LEFT_COLOR
                )

            elif team == 1:

                player_color = (
                    TEAM_RIGHT_COLOR
                )

            else:

                player_color = (
                    TEAM_LEFT_COLOR
                )


            player_colors[
                track_id
            ] = player_color


            # =================================================
            # SAVE PLAYER INFORMATION
            # =================================================

            player_data = {

                "id":
                    track_id,

                "team":
                    team,

                "image_x":
                    float(px),

                "image_y":
                    float(py),

                "bbox":
                    [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ],

                "court_x":
                    (
                        float(mx)
                        if mx is not None
                        else None
                    ),

                "court_y":
                    (
                        float(my)
                        if my is not None
                        else None
                    ),

                "speed":
                    float(speed),

                "pose_available":
                    track_id
                    in pose_matches,

                "pose_confidence":
                    (
                        pose_matches[
                            track_id
                        ][
                            "pose_confidence"
                        ]
                        if track_id
                        in pose_matches
                        else 0.0
                    ),

                "keypoints":
                    (
                        pose_matches[
                            track_id
                        ][
                            "keypoints"
                        ]
                        if track_id
                        in pose_matches
                        else []
                    )
            }


            frame_player_data[
                str(track_id)
            ] = player_data


        # ====================================================
        # SAVE FRAME DATA
        # ====================================================

        all_player_keypoints[
            str(frame_idx)
        ] = frame_player_data
        all_ball_data[
            str(frame_idx)
        ] = ball_data

        # ====================================================
        # BUILD MINI COURT
        # ====================================================

        if H is not None:

            left_players = []

            right_players = []

            player_points = []


            for i in range(
                len(ids)
            ):

                player_id = int(
                    ids[i]
                )


                if player_id in EXCLUDED_IDS:
                    continue


                x1, y1, x2, y2 = (
                    xyxy[i]
                )


                px = (
                    x1 + x2
                ) // 2

                py = y2


                point = np.array(
                    [
                        [
                            [
                                px,
                                py
                            ]
                        ]
                    ],
                    dtype=np.float32
                )


                mapped = (
                    cv2.perspectiveTransform(
                        point,
                        H
                    )
                )


                mx, my = (
                    mapped[0][0]
                )


                mx = int(mx)

                my = int(my)


                player_points.append(
                    (
                        mx,
                        my,
                        i
                    )
                )


            # =================================================
            # TEAM SPLIT
            # =================================================

            for mx, my, idx in (
                player_points
            ):

                if (
                    mx
                    <
                    COURT_WIDTH // 2
                ):

                    left_players.append(
                        (
                            mx,
                            my,
                            idx
                        )
                    )

                else:

                    right_players.append(
                        (
                            mx,
                            my,
                            idx
                        )
                    )


            # =================================================
            # MAX 7 PLAYERS PER SIDE
            # =================================================

            if len(left_players) > 7:

                extras = (
                    left_players[7:]
                )

                left_players = (
                    left_players[:7]
                )


                for mx, my, idx in extras:

                    mx_shifted = (
                        COURT_WIDTH
                        -
                        mx
                    )

                    right_players.append(
                        (
                            mx_shifted,
                            my,
                            idx
                        )
                    )


            if len(right_players) > 7:

                extras = (
                    right_players[7:]
                )

                right_players = (
                    right_players[:7]
                )


                for mx, my, idx in extras:

                    mx_shifted = (
                        COURT_WIDTH
                        -
                        mx
                    )

                    left_players.append(
                        (
                            mx_shifted,
                            my,
                            idx
                        )
                    )


            # =================================================
            # DRAW LEFT TEAM
            # =================================================

            for mx, my, idx in (
                left_players
            ):

                cv2.circle(
                    base_court,
                    (
                        mx,
                        my
                    ),
                    12,
                    TEAM_LEFT_COLOR,
                    -1
                )


                cv2.circle(
                    base_court,
                    (
                        mx,
                        my
                    ),
                    5,
                    TEAM_LEFT_COLOR,
                    -1
                )


                player_id = int(
                    ids[idx]
                )


                cv2.putText(
                    base_court,
                    f"{player_id}",
                    (
                        mx - 8,
                        my - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (
                        0,
                        0,
                        0
                    ),
                    2
                )


            # =================================================
            # DRAW RIGHT TEAM
            # =================================================

            for mx, my, idx in (
                right_players
            ):

                cv2.circle(
                    base_court,
                    (
                        mx,
                        my
                    ),
                    12,
                    TEAM_RIGHT_COLOR,
                    -1
                )


                cv2.circle(
                    base_court,
                    (
                        mx,
                        my
                    ),
                    5,
                    TEAM_RIGHT_COLOR,
                    -1
                )


                player_id = int(
                    ids[idx]
                )


                cv2.putText(
                    base_court,
                    f"{player_id}",
                    (
                        mx - 8,
                        my - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (
                        0,
                        0,
                        0
                    ),
                    2
                )


            # =================================================
            # DRAW NET
            # =================================================

            for y in range(
                0,
                COURT_HEIGHT,
                12
            ):

                cv2.line(
                    base_court,
                    (
                        COURT_WIDTH // 2,
                        y
                    ),
                    (
                        COURT_WIDTH // 2,
                        min(
                            y + 6,
                            COURT_HEIGHT
                        )
                    ),
                    COURT_LINE_COLOR,
                    3
                )


        # ====================================================
        # DRAW TRACKING INFORMATION ON VIDEO
        # ====================================================

        for i in range(
            len(ids)
        ):

            x1, y1, x2, y2 = (
                xyxy[i]
            )


            track_id = int(
                ids[i]
            )


            if track_id in EXCLUDED_IDS:
                continue


            cbx = (
                x1 + x2
            ) // 2


            width = (
                x2 - x1
            )


            shirt_color = (
                player_colors.get(
                    track_id,
                    TEAM_LEFT_COLOR
                )
            )


            # ------------------------------------------------
            # Player marker
            # ------------------------------------------------

            cv2.ellipse(
                annotated_frame,
                (
                    cbx,
                    y2
                ),
                (
                    int(width),
                    int(
                        0.35
                        *
                        width
                    )
                ),
                0,
                -45,
                235,
                shirt_color,
                3
            )


            # ------------------------------------------------
            # ID box
            # ------------------------------------------------

            rect_w = 40

            rect_h = 20


            x1_rect = (
                cbx
                -
                rect_w // 2
            )


            y1_rect = (
                y2
                -
                rect_h // 2
            )


            x1_text = (
                x1_rect
                +
                12
            )


            if track_id > 99:

                x1_text -= 10


            speed = player_speeds.get(
                track_id,
                0.0
            )


            # ------------------------------------------------
            # Player ID
            # ------------------------------------------------

            cv2.putText(
                annotated_frame,
                f"{track_id}",
                (
                    x1_text,
                    y1_rect + 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (
                    0,
                    0,
                    0
                ),
                2
            )


            # ------------------------------------------------
            # Speed
            # ------------------------------------------------

            cv2.putText(
                annotated_frame,
                f"{speed:.2f} m/s",
                (
                    x1_text,
                    y1_rect + 32
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (
                    0,
                    0,
                    0
                ),
                2
            )


    # ========================================================
    # COURT CORNER POINTS
    # ========================================================

    frame_corner_points = (
        court_points_by_frame.get(
            str(frame_idx),
            []
        )
    )


    if len(
        frame_corner_points
    ) >= 4:

        top_left = tuple(
            map(
                int,
                frame_corner_points[0]
            )
        )


        top_right = tuple(
            map(
                int,
                frame_corner_points[1]
            )
        )


        bottom_right = tuple(
            map(
                int,
                frame_corner_points[2]
            )
        )


        bottom_left = tuple(
            map(
                int,
                frame_corner_points[3]
            )
        )


        frame_pts = np.array(
            [
                top_left,
                top_right,
                bottom_right,
                bottom_left
            ],
            dtype=np.float32
        )


        H, _ = cv2.findHomography(
            frame_pts,
            court_pts
        )


        # ----------------------------------------------------
        # Court boundary
        # ----------------------------------------------------

        labeled_points = [

            (
                1,
                top_left
            ),

            (
                2,
                top_right
            ),

            (
                3,
                bottom_right
            ),

            (
                4,
                bottom_left
            )
        ]


        corner_path = [

            top_left,
            top_right,
            bottom_right,
            bottom_left,
            top_left
        ]


        for start_point, end_point in zip(
            corner_path,
            corner_path[1:]
        ):

            cv2.line(
                annotated_frame,
                start_point,
                end_point,
                CORNER_COLOR,
                2
            )


        for label, (
            px,
            py
        ) in labeled_points:

            cv2.circle(
                annotated_frame,
                (
                    px,
                    py
                ),
                6,
                CORNER_COLOR,
                -1
            )


            cv2.putText(
                annotated_frame,
                str(label),
                (
                    px + 6,
                    py - 6
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                CORNER_COLOR,
                2
            )


    # ========================================================
    # MINI COURT OVERLAY
    # ========================================================

    h_frame, w_frame = (
        annotated_frame.shape[:2]
    )


    y1 = MARGIN

    y2 = (
        y1
        +
        COURT_HEIGHT
    )


    offset_x = 130


    x2 = (
        w_frame
        -
        MARGIN
        -
        offset_x
    )


    x1 = (
        x2
        -
        COURT_WIDTH
    )


    if (
        y2 < h_frame
        and
        x2 < w_frame
        and
        x1 >= 0
        and
        y1 >= 0
    ):

        overlay = (
            annotated_frame[
                y1:y2,
                x1:x2
            ].copy()
        )


        cv2.addWeighted(
            base_court,
            COURT_ALPHA,
            overlay,
            1 - COURT_ALPHA,
            0,
            annotated_frame[
                y1:y2,
                x1:x2
            ]
        )


    # ========================================================
    # RESET COURT FOR NEXT FRAME
    # ========================================================

    base_court = court_image.copy()


    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    if out is None:

        h, w = (
            annotated_frame.shape[:2]
        )


        out = cv2.VideoWriter(

            OUTPUT_VIDEO_PATH,

            fourcc,

            fps,

            (
                w,
                h
            )
        )


    out.write(
        annotated_frame
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    # ========================================================
    # DISPLAY
    # ========================================================

    display_frame = annotated_frame.copy()

    display_width = 1200

    display_height = int(
        display_frame.shape[0]
        * display_width
        / display_frame.shape[1]
    )

    display_frame = cv2.resize(
        display_frame,
        (
            display_width,
            display_height
        )
    )

    cv2.imshow(
        "Volleyball Tracking + Pose",
        display_frame
    )

    if (
        cv2.waitKey(1)
        &
        0xFF
    ) == 27:

        break


    if (
        cv2.waitKey(1)
        &
        0xFF
    ) == 27:

        break


    frame_idx += 1


# ============================================================
# CLEANUP
# ============================================================

cap.release()


if out is not None:

    out.release()


cv2.destroyAllWindows()


# ============================================================
# SAVE KEYPOINT DATA
# ============================================================

with open(
    POSE_OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_player_keypoints,
        f,
        indent=2
    )
with open(
        "ball_tracking.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_ball_data,
            f,
            indent=2
        )

print(
    "\n========================================"
)

print(
    "PROCESSING COMPLETE"
)

print(
    f"Output video: {OUTPUT_VIDEO_PATH}"
)

print(
    f"Player pose data: {POSE_OUTPUT_PATH}"
)

print(
    f"Frames processed: {frame_idx}"
)

print(
    "Each matched player contains:"
)

print(
    "  - Player ID"
)

print(
    "  - Team"
)

print(
    "  - Image position"
)

print(
    "  - Mini-court position"
)

print(
    "  - Speed"
)

print(
    "  - 17 pose keypoints"
)

print(
    "  - Pose confidence"
)

print(
    "Ball data: ball_tracking.json"
)

print(
    "========================================"
)