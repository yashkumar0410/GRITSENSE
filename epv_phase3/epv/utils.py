"""
epv/utils.py
------------
Small shared helpers used across the EPV module. Kept deliberately tiny:
this file does NOT touch detection/tracking/pose/homography, it only knows
about the *output* schema that main.py's `--log-out` option writes.

Frame-log schema (one row per player per frame), as written by main.py:
    {
        "frame_idx": int,
        "player_id": int,
        "team": "left" | "right",
        "x_m": float,   # court-space x in metres (0..COURT_LENGTH_M)
        "y_m": float,   # court-space y in metres (0..COURT_WIDTH_M)
        "speed_mps": float,
    }
"""

import json
import os

# Real-world court dimensions, matching the constants already used in
# main.py (actual_width=18, actual_height=9) so the EPV module stays
# consistent with the perception pipeline's homography scale.
COURT_LENGTH_M = 18.0
COURT_WIDTH_M = 9.0

# Matches the 7-per-side cap already enforced in main.py's left/right
# player split logic.
MAX_PLAYERS_PER_TEAM = 7

TEAMS = ("left", "right")


def load_frame_log(path):
    """Load a per-frame player-state log written by `main.py --log-out`."""
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(
            f"Expected a list of per-player-per-frame records in {path}, "
            f"got {type(rows)}"
        )
    return rows


def rows_to_frames(rows):
    """
    Group flat (player, frame) rows into per-frame records:
        {frame_idx: [row, row, ...], ...}
    sorted by frame_idx.
    """
    frames = {}
    for row in rows:
        frames.setdefault(row["frame_idx"], []).append(row)
    return dict(sorted(frames.items()))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
