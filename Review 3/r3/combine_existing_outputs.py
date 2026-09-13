import os
import csv
import cv2


# ============================================================
# COMBINE EXISTING OUTPUTS
# ============================================================
#
# IMPORTANT:
# This script DOES NOT rerun the player detector or pose model.
#
# It takes:
#   1. output_keypoint.mp4  -> your already-generated,
#                              accurate player keypoint video
#   2. ball.csv             -> your already-generated ball detections
#
# and draws the ball onto the existing keypoint video.
#
# Therefore the player keypoints remain EXACTLY as they were
# in output_keypoint.mp4.
# ============================================================


VIDEO_PATH = "output_keypoint.mp4"
BALL_CSV_PATH = "ball.csv"
OUTPUT_PATH = "combined_output.mp4"


# ============================================================
# LOAD BALL CSV
# ============================================================

def load_ball_csv(path):

    ball_by_frame = {}

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ball.csv not found:\n{os.path.abspath(path)}"
        )

    with open(path, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        print("CSV columns:", reader.fieldnames)

        for row in reader:

            try:
                frame_id = int(float(row["Frame"]))
                visibility = int(float(row["Visibility"]))
                x = float(row["X"])
                y = float(row["Y"])
                radius = float(row["Radius"])

            except (KeyError, TypeError, ValueError):
                continue

            if visibility == 1 and x >= 0 and y >= 0:

                ball_by_frame[frame_id] = {
                    "x": int(round(x)),
                    "y": int(round(y)),
                    "radius": max(2, int(round(radius)))
                }

    return ball_by_frame


# ============================================================
# DRAW BALL
# ============================================================

def draw_ball(frame, ball):

    x = ball["x"]
    y = ball["y"]
    radius = ball["radius"]

    # Same ball marker style used in the previous integration.
    cv2.circle(
        frame,
        (x, y),
        radius,
        (0, 255, 0),
        4
    )

    cv2.circle(
        frame,
        (x, y),
        radius + 6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "BALL",
        (x + radius + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


# ============================================================
# CHECK INPUTS
# ============================================================

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"output_keypoint.mp4 not found:\n"
        f"{os.path.abspath(VIDEO_PATH)}"
    )

ball_by_frame = load_ball_csv(BALL_CSV_PATH)

print()
print("Loaded ball detections:", len(ball_by_frame))
print("Player video:", os.path.abspath(VIDEO_PATH))


# ============================================================
# OPEN EXISTING KEYPOINT VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open:\n{os.path.abspath(VIDEO_PATH)}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print()
print("Video:")
print("  Resolution:", width, "x", height)
print("  FPS:", fps)
print("  Frames:", total_frames)


# ============================================================
# OUTPUT
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create:\n{os.path.abspath(OUTPUT_PATH)}"
    )


# ============================================================
# COMBINE
# ============================================================

frame_idx = 0

print()
print("Combining...")
print()

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # IMPORTANT:
    # The frame itself already contains the EXACT player
    # keypoints from output_keypoint.mp4.
    #
    # We ONLY add the ball here.

    ball = ball_by_frame.get(frame_idx)

    if ball is not None:
        draw_ball(frame, ball)


    out.write(frame)

    frame_idx += 1

    if frame_idx % 50 == 0 or frame_idx == total_frames:

        percent = (
            frame_idx / total_frames * 100
            if total_frames > 0
            else 0
        )

        print(
            f"Processed {frame_idx}/{total_frames} "
            f"({percent:.1f}%)"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()


print()
print("=" * 60)
print("DONE ✓")
print("=" * 60)
print()
print("Final combined video:")
print(os.path.abspath(OUTPUT_PATH))
print()
print("Player keypoints: EXACTLY from output_keypoint.mp4")
print("Ball:             from ball.csv")
