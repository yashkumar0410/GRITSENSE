"""
epv/make_synthetic_test_log.py
---------------------------------
TEST-ONLY FIXTURE GENERATOR. This is NOT part of the EPV pipeline and
produces NO real results — it exists purely so the EPV pipeline
(rally_segmentation -> state_builder -> dataset -> model -> train ->
evaluate) can be exercised end-to-end and sanity-checked, since this
repository ships no video file, no `cp.json` court-points file, and no
real player-tracking log to run `main.py --log-out` against in this
environment.

It writes a JSON file with EXACTLY the schema `main.py --log-out` produces
(see epv/utils.py), containing a few synthetic "rallies" of six players
per side drifting/oscillating with noise, separated by frame gaps (to be
picked up by rally_segmentation's gap heuristic).

Once the team has a real video + court-points file, replace this with:
    python main.py --video <clip>.mp4 --court-json cp.json --log-out real_log.json
and point epv/dataset.py at `real_log.json` instead.
"""

import argparse
import json

import numpy as np

from epv.utils import COURT_LENGTH_M, COURT_WIDTH_M


def make_synthetic_log(num_rallies=6, frames_per_rally=40, gap_frames=25, players_per_side=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    frame_idx = 0
    player_id = 1

    for r in range(num_rallies):
        left_ids = list(range(player_id, player_id + players_per_side))
        player_id += players_per_side
        right_ids = list(range(player_id, player_id + players_per_side))
        player_id += players_per_side

        # random starting positions per team, then a slow drift + noise
        left_start = rng.uniform(1, COURT_LENGTH_M / 2 - 1, size=(players_per_side, 2))
        right_start = rng.uniform(COURT_LENGTH_M / 2 + 1, COURT_LENGTH_M - 1, size=(players_per_side, 2))
        left_start[:, 1] = rng.uniform(0.5, COURT_WIDTH_M - 0.5, size=players_per_side)
        right_start[:, 1] = rng.uniform(0.5, COURT_WIDTH_M - 0.5, size=players_per_side)

        left_drift = rng.uniform(-0.05, 0.15, size=(players_per_side, 2))
        right_drift = rng.uniform(-0.15, 0.05, size=(players_per_side, 2))

        prev_left, prev_right = left_start.copy(), right_start.copy()

        for t in range(frames_per_rally):
            left_pos = np.clip(
                left_start + left_drift * t + rng.normal(0, 0.05, size=left_start.shape),
                [0, 0], [COURT_LENGTH_M / 2 - 0.1, COURT_WIDTH_M],
            )
            right_pos = np.clip(
                right_start + right_drift * t + rng.normal(0, 0.05, size=right_start.shape),
                [COURT_LENGTH_M / 2 + 0.1, 0], [COURT_LENGTH_M, COURT_WIDTH_M],
            )

            for i, pid in enumerate(left_ids):
                speed = float(np.linalg.norm(left_pos[i] - prev_left[i]) * 30.0) if t > 0 else 0.0
                rows.append({
                    "frame_idx": frame_idx, "player_id": pid, "team": "left",
                    "x_m": round(float(left_pos[i, 0]), 4), "y_m": round(float(left_pos[i, 1]), 4),
                    "speed_mps": round(speed, 4),
                })
            for i, pid in enumerate(right_ids):
                speed = float(np.linalg.norm(right_pos[i] - prev_right[i]) * 30.0) if t > 0 else 0.0
                rows.append({
                    "frame_idx": frame_idx, "player_id": pid, "team": "right",
                    "x_m": round(float(right_pos[i, 0]), 4), "y_m": round(float(right_pos[i, 1]), 4),
                    "speed_mps": round(speed, 4),
                })

            prev_left, prev_right = left_pos, right_pos
            frame_idx += 1

        frame_idx += gap_frames  # dead time between rallies -> rally boundary

    return rows


def build_argparser():
    p = argparse.ArgumentParser(description="Generate a synthetic test log (NOT real data)")
    p.add_argument("--out", default="synthetic_test_log.json")
    p.add_argument("--num-rallies", type=int, default=6)
    p.add_argument("--frames-per-rally", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    rows = make_synthetic_log(num_rallies=args.num_rallies, frames_per_rally=args.frames_per_rally, seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(rows, f)
    print(f"[synthetic] wrote {len(rows)} rows ({args.num_rallies} rallies) to {args.out}")
