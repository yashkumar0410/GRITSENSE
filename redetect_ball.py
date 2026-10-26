"""
Re-detect the ball with several candidates per frame (needs ultralytics + ball.pt).

The original tracker keeps one detection per frame at the model's default
640 px input, where the ball is only a few pixels wide; it often locks onto
the first referee's head or the mouse cursor instead. Here ball.pt is run at
640, 1280 and 1920 px, every candidate above a low confidence is kept, and
analytics/ball_track.py then chooses the physically consistent path.

Writes ball_candidates.json (cache) and then runs clean_ball_track.py, which
produces ball_clean.csv.

Run from the project root:
    python redetect_ball.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SIZES = (640, 1280, 1920)
MIN_CONF = 0.05
MERGE_PX = 12          # candidates from different sizes closer than this are the same object


def detect(video: Path, weights: Path) -> dict[int, list[list[float]]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    candidates, frame = {}, 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        found = []
        for size in SIZES:
            result = model.predict(image, conf=MIN_CONF, imgsz=size, verbose=False)[0]
            for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x, y = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
                r = float(max(box[2] - box[0], box[3] - box[1]) / 2)
                same = next((c for c in found if abs(c[0] - x) < MERGE_PX and abs(c[1] - y) < MERGE_PX), None)
                if same is None:
                    found.append([x, y, r, float(conf)])
                elif conf > same[3]:
                    same[:] = [x, y, r, float(conf)]
        candidates[frame] = found
        frame += 1
        if frame % 50 == 0:
            print(f"  {frame} frames")
    cap.release()
    return candidates


def main(video: Path | None = None):
    candidates = detect(video or ROOT / "sample3.mp4", ROOT / "ball.pt")
    with (ROOT / "ball_candidates.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in candidates.items()}, f)
    n = sum(len(v) for v in candidates.values())
    print(f"Saved ball_candidates.json: {n} candidates over {len(candidates)} frames")
    import clean_ball_track
    clean_ball_track.main()  # noqa: picks the path from the candidates


if __name__ == "__main__":
    main()
