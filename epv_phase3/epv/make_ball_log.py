"""
epv/make_ball_log.py
---------------------
Convert a raw ball *pixel* detection log (as written by the root
ball_tracker.py / the Cappy ball-tracking output, ball_coordinates.json)
into the court-space ball log consumed by epv.utils.load_ball_log, using
the same per-frame court-corner homography as main.py (cp.json format).

Inputs
------
--ball-pixels : JSON, one of
    {"0": {"x": 401.0, "y": 626.0, "detected": true, ...},
     "1": null, ...}
    or [{"frame_idx": 0, "x": 401.0, "y": 626.0, ...}, ...]
    Pixel coordinates are the ball centre in the ORIGINAL video frame
    (the same space the court corner points live in).
--court-json : per-frame court corner points, main.py's cp.json format:
    {"17": [[tlx, tly], [trx, try_], [brx, bry], [blx, bly]], ...}

Output
------
--out : JSON list of {"frame_idx", "x_m", "y_m", "speed_mps"} records
    (frames without a ball detection or without a homography are skipped),
    directly usable as `generate_dataset(..., ball_log_path=...)`.

Usage
-----
    python -m epv.make_ball_log \
        --ball-pixels path/to/ball_coordinates.json \
        --court-json path/to/cp.json \
        --fps 30 \
        --out ball_log.json
"""

import argparse
import json

import cv2
import numpy as np

COURT_W_PX = 300   # destination court raster width used by main.py
COURT_H_PX = 150   # destination court raster height used by main.py
COURT_LENGTH_M = 18.0
COURT_WIDTH_M = 9.0
PX_PER_M = COURT_W_PX / COURT_LENGTH_M


def load_ball_pixels(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((str(r.get("frame_idx")), r) for r in raw)
    else:
        raise ValueError(f"Unsupported ball pixel log format: {type(raw)}")
    for key, rec in items:
        if rec is None:
            continue
        x, y = rec.get("x"), rec.get("y")
        if x is None or y is None:
            continue
        out[int(key)] = (float(x), float(y))
    return out


def build_homography(frame_idx, court_points_by_frame):
    pts = court_points_by_frame.get(str(frame_idx), [])
    if len(pts) < 4:
        return None
    frame_pts = np.array(pts[:4], dtype=np.float32)
    court_pts = np.array(
        [[0, 0], [COURT_W_PX, 0], [COURT_W_PX, COURT_H_PX], [0, COURT_H_PX]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(frame_pts, court_pts)
    return H


def main():
    p = argparse.ArgumentParser(description="Convert pixel ball log to court-space ball log")
    p.add_argument("--ball-pixels", required=True, help="ball_coordinates.json (pixel space)")
    p.add_argument("--court-json", required=True, help="cp.json per-frame court corner points")
    p.add_argument("--fps", type=float, default=30.0, help="video fps, for the speed calculation")
    p.add_argument("--out", default="ball_log.json", help="output court-space ball log path")
    args = p.parse_args()

    with open(args.court_json, "r", encoding="utf-8") as f:
        court_points_by_frame = json.load(f)

    ball_px = load_ball_pixels(args.ball_pixels)
    print(f"[make_ball_log] {len(ball_px)} frames with ball detections")

    out_rows = []
    prev = None  # (frame_idx, x_m, y_m)
    for frame_idx in sorted(ball_px):
        H = build_homography(frame_idx, court_points_by_frame)
        if H is None:
            continue
        px_pt = np.array([[[ball_px[frame_idx][0], ball_px[frame_idx][1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(px_pt, H)[0][0]
        x_m = round(float(mapped[0]) / PX_PER_M, 4)
        y_m = round(float(mapped[1]) / (COURT_H_PX / COURT_WIDTH_M), 4)

        speed = 0.0
        if prev is not None and frame_idx - prev[0] == 1:
            speed = float(np.hypot(x_m - prev[1], y_m - prev[2]) * args.fps)

        out_rows.append({
            "frame_idx": int(frame_idx),
            "x_m": x_m,
            "y_m": y_m,
            "speed_mps": round(speed, 4),
        })
        prev = (frame_idx, x_m, y_m)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_rows, f)
    print(f"[make_ball_log] wrote {len(out_rows)} ball-frame records to {args.out}")

    inside = sum(
        1 for r in out_rows if 0 <= r["x_m"] <= COURT_LENGTH_M and 0 <= r["y_m"] <= COURT_WIDTH_M
    )
    if out_rows and inside / len(out_rows) < 0.5:
        print(f"[make_ball_log][WARN] only {inside}/{len(out_rows)} ball points landed "
              "inside the court -- check the court corner points / homography.")


if __name__ == "__main__":
    main()
