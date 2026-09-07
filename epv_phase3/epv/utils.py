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

import numpy as np

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


def load_ball_log(path):
    """
    Load a per-frame ball log (court-space, metres), as written by
    `main.py --ball-log-out` or `epv/make_ball_log.py`.

    Accepted schemas (both produced by those tools):
        - list of {"frame_idx": int, "x_m": float, "y_m": float,
                   "speed_mps": float|None}
        - dict keyed by str(frame_idx) with the same fields
    Frames where the ball was not detected must be absent (or have
    x_m/y_m = null); they become ball_present == 0 in the state vector.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ball_by_frame = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((str(r.get("frame_idx")), r) for r in raw)
    else:
        raise ValueError(f"Unsupported ball log format in {path}: {type(raw)}")

    for key, rec in items:
        if rec is None:
            continue
        x, y = rec.get("x_m"), rec.get("y_m")
        if x is None or y is None:
            continue
        ball_by_frame[int(key)] = {
            "x_m": float(x),
            "y_m": float(y),
            "speed_mps": float(rec.get("speed_mps") or 0.0),
        }
    return ball_by_frame


def log_quality_report(rows):
    """
    Quantify the positional quality of a player log BEFORE any clipping.
    This exists because the Phase 3 mid-term logs had broken homography
    output (most y-coordinates negative => most state features clipped to
    constant values). Running this on every new log is how we catch that.

    Returns a dict:
        num_frames, num_player_records, players_per_frame_mean,
        frac_x_in_court, frac_y_in_court, frac_both_in_court,
        x_range, y_range  (min/max in metres)
    """
    xs = np.array([r["x_m"] for r in rows], dtype=np.float64)
    ys = np.array([r["y_m"] for r in rows], dtype=np.float64)
    frames = {r["frame_idx"] for r in rows}
    in_x = (xs >= 0) & (xs <= COURT_LENGTH_M)
    in_y = (ys >= 0) & (ys <= COURT_WIDTH_M)
    return {
        "num_player_records": int(len(rows)),
        "num_frames": len(frames),
        "players_per_frame_mean": round(float(len(rows) / max(len(frames), 1)), 3),
        "frac_x_in_court": round(float(in_x.mean()), 4),
        "frac_y_in_court": round(float(in_y.mean()), 4),
        "frac_both_in_court": round(float((in_x & in_y).mean()), 4),
        "x_range_m": [round(float(xs.min()), 2), round(float(xs.max()), 2)],
        "y_range_m": [round(float(ys.min()), 2), round(float(ys.max()), 2)],
        "warning": (
            "Most coordinates fall OUTSIDE the court; the homography/court-point "
            "mapping in main.py is likely wrong for this clip. Fix before training, "
            "or expect a large fraction of clipped (zero-variance) state features."
        ) if (in_x & in_y).mean() < 0.5 else None,
    }


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
