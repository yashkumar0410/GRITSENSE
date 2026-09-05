from ultralytics import YOLO

import cv2
import numpy as np
import json
import csv

from collections import deque

from helper import create_video_writer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11s.pt"

VIDEO_PATH = "sample3.mp4"

OUTPUT_PATH = "Output_sample3.mp4"

BALL_JSON_PATH = "ball_coordinates.json"

BALL_CSV_PATH = "ball_coordinates.csv"


# ============================================================
# BALL TRACKING PARAMETERS
# ============================================================

CONFIDENCE_THRESHOLD = 0.50

MAX_DISTANCE = 300

TRAIL_LENGTH = 10


# ============================================================
# GET CENTER OF DETECTION
# ============================================================

def get_center(box):

    x1, y1, x2, y2 = (
        box.xyxy[0]
        .cpu()
        .numpy()
        .astype(int)
    )

    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )


# ============================================================
# FIND NEAREST DETECTION
# ============================================================

def nearest(
    boxes,
    previous_center
):

    centers = [
        get_center(box)
        for box in boxes
    ]

    if not centers:

        return None, None

    centers_array = np.array(
        centers,
        dtype=np.float32
    )

    previous_array = np.array(
        previous_center,
        dtype=np.float32
    )

    distances = np.linalg.norm(
        centers_array - previous_array,
        axis=1
    )

    index = int(
        np.argmin(distances)
    )

    return (
        index,
        float(distances[index])
    )


# ============================================================
# SELECT FIRST DETECTION
# ============================================================

def select_first_detection(
    boxes
):

    if len(boxes) == 0:

        return None

    confidences = (
        boxes.conf
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidences)
    )

    return boxes[best_index]


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading model..."
)

model = YOLO(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 30.0


print(
    f"Video: {VIDEO_PATH}"
)

print(
    f"Resolution: {width}x{height}"
)

print(
    f"FPS: {fps}"
)


# ============================================================
# VIDEO WRITER
# ============================================================

writer = create_video_writer(
    cap,
    OUTPUT_PATH
)


# ============================================================
# BALL TRAJECTORY
# ============================================================

trajectory = deque(
    maxlen=TRAIL_LENGTH
)


# ============================================================
# BALL DATA
# ============================================================

all_ball_data = {}


# ============================================================
# TRACKING STATE
# ============================================================

previous_center = None

previous_frame = None


# ============================================================
# STATISTICS
# ============================================================

frame_number = 0

detected_frames = 0

total_frames = 0


# ============================================================
# FRAME LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # frame_number is ZERO BASED.
    #
    # This matches GritSense's player_keypoints.json.
    # --------------------------------------------------------

    current_frame = frame_number

    total_frames += 1


    # --------------------------------------------------------
    # YOLO BALL DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    boxes = results[0].boxes


    selected_box = None

    selected_distance = None


    # ========================================================
    # FIRST BALL DETECTION
    # ========================================================

    if previous_center is None:

        selected_box = (
            select_first_detection(
                boxes
            )
        )


    # ========================================================
    # TRACK USING PREVIOUS POSITION
    # ========================================================

    else:

        if len(boxes) > 0:

            index, distance = nearest(
                boxes,
                previous_center
            )

            if (
                index is not None
                and distance <= MAX_DISTANCE
            ):

                selected_box = (
                    boxes[index]
                )

                selected_distance = (
                    distance
                )


    # ========================================================
    # BALL DATA
    # ========================================================

    ball_data = None


    if selected_box is not None:

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = (
            selected_box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )


        # ----------------------------------------------------
        # Center
        # ----------------------------------------------------

        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            selected_box.conf[0]
            .cpu()
            .item()
        )


        # ----------------------------------------------------
        # Velocity
        #
        # Units:
        # pixels per frame
        # ----------------------------------------------------

        vx = 0.0

        vy = 0.0


        if (
            previous_center is not None
            and previous_frame is not None
        ):

            frame_difference = (
                current_frame
                -
                previous_frame
            )


            if frame_difference > 0:

                vx = (
                    center[0]
                    -
                    previous_center[0]
                ) / frame_difference


                vy = (
                    center[1]
                    -
                    previous_center[1]
                ) / frame_difference


        # ----------------------------------------------------
        # Save trajectory
        # ----------------------------------------------------

        trajectory.appendleft(
            center
        )


        # ----------------------------------------------------
        # Save ball data
        # ----------------------------------------------------

        ball_data = {

            "x":
                float(center[0]),

            "y":
                float(center[1]),

            "vx":
                float(vx),

            "vy":
                float(vy),

            "confidence":
                confidence,

            "bbox":
                [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ],

            "tracking_distance":
                (
                    float(selected_distance)
                    if selected_distance is not None
                    else 0.0
                ),

            "detected":
                True
        }


        # ----------------------------------------------------
        # Update tracker state
        # ----------------------------------------------------

        previous_center = center

        previous_frame = (
            current_frame
        )

        detected_frames += 1


        # ====================================================
        # DRAW BALL
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


        cv2.circle(
            frame,
            center,
            10,
            (255, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            (
                f"BALL "
                f"{confidence:.2f}"
            ),
            (
                center[0] + 12,
                center[1] - 12
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )


    # ========================================================
    # NO BALL DETECTION
    # ========================================================

    else:

        ball_data = None


    # ========================================================
    # SAVE BALL DATA FOR THIS FRAME
    # ========================================================

    all_ball_data[
        str(current_frame)
    ] = ball_data


    # ========================================================
    # DRAW TRAJECTORY
    # ========================================================

    points = list(
        trajectory
    )


    for i in range(
        1,
        len(points)
    ):

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            (0, 0, 255),
            4
        )


    # ========================================================
    # FRAME NUMBER
    # ========================================================

    cv2.putText(
        frame,
        f"Frame: {current_frame}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DETECTION STATUS
    # ========================================================

    if ball_data is not None:

        status = (
            "BALL DETECTED"
        )

    else:

        status = (
            "NO BALL"
        )


    cv2.putText(
        frame,
        status,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WRITE VIDEO
    # ========================================================

    writer.write(
        frame
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        current_frame % 30
        == 0
    ):

        print(
            f"Processed "
            f"{current_frame} frames"
        )


    frame_number += 1


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# ============================================================
# SAVE JSON
# ============================================================

with open(
    BALL_JSON_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_ball_data,
        f,
        indent=2
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    BALL_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    csv_writer = csv.writer(
        f
    )


    csv_writer.writerow(
        [
            "frame",
            "x",
            "y",
            "vx",
            "vy",
            "confidence",
            "detected"
        ]
    )


    for frame_id, data in (
        all_ball_data.items()
    ):

        if data is None:

            csv_writer.writerow(
                [
                    frame_id,
                    "",
                    "",
                    "",
                    "",
                    "",
                    False
                ]
            )

        else:

            csv_writer.writerow(
                [
                    frame_id,
                    data["x"],
                    data["y"],
                    data["vx"],
                    data["vy"],
                    data["confidence"],
                    data["detected"]
                ]
            )


# ============================================================
# FINAL STATISTICS
# ============================================================

coverage = 0.0

if total_frames > 0:

    coverage = (
        detected_frames
        /
        total_frames
        *
        100.0
    )


print()
print(
    "========================================"
)

print(
    "BALL TRACKING COMPLETE"
)

print(
    "========================================"
)

print(
    f"Frames processed: "
    f"{total_frames}"
)

print(
    f"Frames detected: "
    f"{detected_frames}"
)

print(
    f"Detection coverage: "
    f"{coverage:.2f}%"
)

print(
    f"Output video: "
    f"{OUTPUT_PATH}"
)

print(
    f"Ball JSON: "
    f"{BALL_JSON_PATH}"
)

print(
    f"Ball CSV: "
    f"{BALL_CSV_PATH}"
)

print(
    "========================================"
)