"""
Clean the raw ball track (ball.csv) into ball_clean.csv.

Removes physically impossible detections and fills short gaps (see
analytics/ball_track.py). ball_clean.csv has the same columns as ball.csv plus
a Source column (detected / interpolated / none), and is used by the graph
pipeline, the dashboard, calibration and training whenever it exists.

Run from the project root (master_main.py runs it after detection):
    python clean_ball_track.py
"""

from __future__ import annotations

import json
from pathlib import Path

from analytics.ball_track import clean, load_raw, select_path, write_csv
from analytics.camera import Camera

ROOT = Path(__file__).resolve().parent


def main(root: Path = ROOT) -> dict:
    with (root / "annotations.json").open(encoding="utf-8") as f:
        annotations = json.load(f)
    raw = load_raw(root / "ball.csv")
    with (root / "ball.csv").open(encoding="utf-8") as f:
        n_frames = sum(1 for _ in f) - 1
    with (root / "player_keypoints_mf3_pose.json").open(encoding="utf-8") as f:
        player_data = json.load(f)
    candidates_path = root / "ball_candidates.json"
    if candidates_path.exists():
        # Several candidates per frame (redetect_ball.py): choose the physical path.
        with candidates_path.open(encoding="utf-8") as f:
            candidates = {int(k): v for k, v in json.load(f).items()}
        track, stats = select_path(candidates, Camera(annotations), player_data)
    else:
        track, stats = clean(raw, Camera(annotations), player_data)
    write_csv(track, n_frames, root / "ball_clean.csv")
    print("Ball track cleaned:", json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
