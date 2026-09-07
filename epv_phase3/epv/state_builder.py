"""
epv/state_builder.py
---------------------
Task 2 of Member 3's Phase 3 work: turn a rally (list of per-frame player
records, see epv/rally_segmentation.py) into a fixed-size numerical state
vector suitable as MLP input for the EPV model.

Two representations are supported:

1. BASE (no ball log available) -- what the pipeline shipped with in Phase 3
   mid-term: only player id, team (left/right), court-space position
   (metres, from the existing homography step) and speed (m/s).
   Total dimensionality = 2 * (4 * MAX_PLAYERS_PER_TEAM + 6) + 1
                        = 2 * (28 + 6) + 1 = 69  (MAX_PLAYERS_PER_TEAM=7)

2. BALL-AUGMENTED (when a ball log is provided, see epv/utils.load_ball_log)
   -- the same 69 features PLUS 7 ball features (documented below), so the
   tracker claim "state representation from players, ball, pose and court"
   is covered for players + ball + court (pose still awaits wiring of
   pose.pt into main.py -- see TODO at the bottom of this docstring).
   Total dimensionality = 69 + BALL_FEATS = 76.

BALL FEATURES (7 dims, appended after rally_progress)
------------------------------------------------------
    ball_present          -> 1.0 if the ball was detected this frame, else 0.0
                             (all other ball dims are 0.0 when absent)
    ball_x_norm           -> ball court x / COURT_LENGTH_M, clipped to [0, 1]
    ball_y_norm           -> ball court y / COURT_WIDTH_M,  clipped to [0, 1]
    ball_speed_norm       -> ball speed (m/s) / BALL_SPEED_NORM_CAP, clipped [0, 1]
    ball_dist_left_norm   -> mean distance of left-team players to the ball,
                             divided by the court diagonal, clipped [0, 1]
                             (small = ball is among the left team -> left likely
                             in possession / receiving)
    ball_dist_right_norm  -> same for the right team
    ball_side             -> ball_x_norm - 0.5 (signed: <0 = left half, >0 = right
                             half; 0.0 when ball absent)

BASE FEATURE DOCUMENTATION (each of the 69 dims)
--------------------------------------------------
For each team in ("left", "right"), for player-slot s in [0, MAX_PLAYERS_PER_TEAM):
    present_s      -> 1.0 if a player occupies this slot this frame, else 0.0
    x_norm_s       -> player's court x-position / COURT_LENGTH_M, clipped to [0, 1]
    y_norm_s       -> player's court y-position / COURT_WIDTH_M,  clipped to [0, 1]
    speed_norm_s   -> player's speed (m/s) / SPEED_NORM_CAP, clipped to [0, 1]
    (0 for all four whenever present_s == 0, i.e. missing/padded slot)
  -> 4 * MAX_PLAYERS_PER_TEAM values per team, ordered by ascending player_id
     within that frame (so the same physical player tends to occupy the
     same slot across consecutive frames of a rally, as long as their id
     is stable, which BotSort tracking already gives us).

For each team, team-level aggregates:
    centroid_x_norm, centroid_y_norm  -> mean position of that team's tracked players
    spread_x_norm, spread_y_norm      -> std-dev of that team's player positions
                                          (a cheap proxy for "how spread out"/
                                          formation-compact the team currently is)
    mean_speed_norm                   -> mean normalized speed of that team
    num_players_norm                  -> (# tracked players) / MAX_PLAYERS_PER_TEAM

Rally-level:
    rally_progress -> (frame's position within its rally) / (rally length - 1),
                       in [0, 1]. 0 = start of rally, 1 = last logged frame.

TODO (pose): pose.pt is present in the repo root but main.py does not run it.
Once per-player pose keypoints are logged (same pattern as the ball log:
a "pose" key on each player row), add a per-player pose block here and bump
the dims -- build_state()/feature_names() are the only two places to change.
"""

import numpy as np

from epv.utils import COURT_LENGTH_M, COURT_WIDTH_M, MAX_PLAYERS_PER_TEAM, TEAMS

SPEED_NORM_CAP = 10.0  # m/s; elite volleyball movement speeds rarely exceed this
BALL_SPEED_NORM_CAP = 30.0  # m/s; volleyball spike/serve speeds are ~< 30 m/s

PER_PLAYER_FEATS = 4          # present, x_norm, y_norm, speed_norm
PER_TEAM_AGG_FEATS = 6        # centroid_x, centroid_y, spread_x, spread_y, mean_speed, num_players
BALL_FEATS = 7                # present, x, y, speed, dist_left, dist_right, side

# Backward-compatible aliases: datasets/checkpoints built before the ball
# extension used STATE_DIM == 69.
BASE_STATE_DIM = 2 * (PER_PLAYER_FEATS * MAX_PLAYERS_PER_TEAM + PER_TEAM_AGG_FEATS) + 1
STATE_DIM = BASE_STATE_DIM
STATE_DIM_WITH_BALL = BASE_STATE_DIM + BALL_FEATS

_COURT_DIAG = float(np.hypot(COURT_LENGTH_M, COURT_WIDTH_M))


def state_dim(include_ball=False):
    return STATE_DIM_WITH_BALL if include_ball else BASE_STATE_DIM


def feature_names(include_ball=False):
    """Human-readable name for every dimension of the state vector, in order."""
    names = []
    for team in TEAMS:
        for s in range(MAX_PLAYERS_PER_TEAM):
            names += [f"{team}_p{s}_present", f"{team}_p{s}_x", f"{team}_p{s}_y", f"{team}_p{s}_speed"]
        names += [
            f"{team}_centroid_x", f"{team}_centroid_y",
            f"{team}_spread_x", f"{team}_spread_y",
            f"{team}_mean_speed", f"{team}_num_players",
        ]
    names.append("rally_progress")
    if include_ball:
        names += [
            "ball_present", "ball_x", "ball_y", "ball_speed",
            "ball_dist_left", "ball_dist_right", "ball_side",
        ]
    expected = state_dim(include_ball)
    assert len(names) == expected, f"{len(names)} != {expected}"
    return names


def _team_block(players):
    """Build the per-player-slot + team-aggregate features for one team's players in one frame."""
    players = sorted(players, key=lambda r: r["player_id"])[:MAX_PLAYERS_PER_TEAM]

    slot_feats = np.zeros(PER_PLAYER_FEATS * MAX_PLAYERS_PER_TEAM, dtype=np.float32)
    xs, ys, speeds = [], [], []

    for s, row in enumerate(players):
        x_norm = float(np.clip(row["x_m"] / COURT_LENGTH_M, 0.0, 1.0))
        y_norm = float(np.clip(row["y_m"] / COURT_WIDTH_M, 0.0, 1.0))
        speed_norm = float(np.clip(row["speed_mps"] / SPEED_NORM_CAP, 0.0, 1.0))
        base = s * PER_PLAYER_FEATS
        slot_feats[base + 0] = 1.0
        slot_feats[base + 1] = x_norm
        slot_feats[base + 2] = y_norm
        slot_feats[base + 3] = speed_norm
        xs.append(x_norm)
        ys.append(y_norm)
        speeds.append(speed_norm)

    if xs:
        agg = np.array([
            float(np.mean(xs)), float(np.mean(ys)),
            float(np.std(xs)), float(np.std(ys)),
            float(np.mean(speeds)),
            len(xs) / MAX_PLAYERS_PER_TEAM,
        ], dtype=np.float32)
    else:
        agg = np.zeros(PER_TEAM_AGG_FEATS, dtype=np.float32)

    return np.concatenate([slot_feats, agg])


def _ball_block(frame_record):
    """Build the 7-dim ball feature block from frame_record["ball"] (or None)."""
    ball = frame_record.get("ball")
    block = np.zeros(BALL_FEATS, dtype=np.float32)
    if not ball:
        return block

    bx = float(np.clip(ball["x_m"] / COURT_LENGTH_M, 0.0, 1.0))
    by = float(np.clip(ball["y_m"] / COURT_WIDTH_M, 0.0, 1.0))
    bspeed = float(np.clip(ball.get("speed_mps", 0.0) / BALL_SPEED_NORM_CAP, 0.0, 1.0))

    by_team = {team: [] for team in TEAMS}
    for row in frame_record["players"]:
        if row["team"] in by_team:
            by_team[row["team"]].append(row)

    dists = {}
    for team in TEAMS:
        if by_team[team] and ball:
            d = [
                float(np.hypot(r["x_m"] - ball["x_m"], r["y_m"] - ball["y_m"])) / _COURT_DIAG
                for r in by_team[team]
            ]
            dists[team] = float(np.clip(np.mean(d), 0.0, 1.0))
        else:
            dists[team] = 0.0

    block[:] = [
        1.0, bx, by, bspeed,
        dists["left"], dists["right"],
        bx - 0.5,
    ]
    return block


def build_state(frame_record, rally_progress, include_ball=False):
    """
    Parameters
    ----------
    frame_record : dict
        {"frame_idx": int, "players": [row, ...], "ball": {...}|None} as
        produced by epv.rally_segmentation.segment_rallies (the "ball" key
        is optional and only used when include_ball=True).
    rally_progress : float in [0, 1]
        Position of this frame within its rally.
    include_ball : bool
        Append the 7 ball features (requires frame_record["ball"]).

    Returns
    -------
    np.ndarray shape (state_dim(include_ball),), dtype float32
    """
    by_team = {team: [] for team in TEAMS}
    for row in frame_record["players"]:
        if row["team"] in by_team:
            by_team[row["team"]].append(row)

    blocks = [_team_block(by_team[team]) for team in TEAMS]
    parts = blocks + [np.array([rally_progress], dtype=np.float32)]
    if include_ball:
        parts.append(_ball_block(frame_record))
    state = np.concatenate(parts)
    expected = state_dim(include_ball)
    assert state.shape[0] == expected, f"{state.shape[0]} != {expected}"
    return state.astype(np.float32)


def build_rally_states(rally, include_ball=False):
    """Build a (len(rally), state_dim) array of states for every frame in a rally."""
    n = len(rally)
    dim = state_dim(include_ball)
    states = np.zeros((n, dim), dtype=np.float32)
    for t, frame_record in enumerate(rally):
        progress = t / (n - 1) if n > 1 else 0.0
        states[t] = build_state(frame_record, progress, include_ball=include_ball)
    return states
