"""
epv/state_builder.py
---------------------
Task 2 of Member 3's Phase 3 work: turn a rally (list of per-frame player
records, see epv/rally_segmentation.py) into a fixed-size numerical state
vector suitable as MLP input for the EPV model.

Only information the current perception pipeline actually produces is
used: player id, team (left/right), court-space position (metres, from
the existing homography step) and speed (m/s, from the existing
consecutive-frame displacement calc in main.py). No pose/ball features are
included because the pipeline does not currently output them for this
repository (BlazePose/ball.pt are present as model weights but are not
wired into main.py yet) — see FEATURE DOCUMENTATION below for exactly how
to extend the vector once they are.

FEATURE DOCUMENTATION (exactly what each dimension means)
-----------------------------------------------------------
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

Total dimensionality = 2 * (4 * MAX_PLAYERS_PER_TEAM + 6) + 1
                      = 2 * (28 + 6) + 1 = 69  (with MAX_PLAYERS_PER_TEAM=7)
"""

import numpy as np

from epv.utils import COURT_LENGTH_M, COURT_WIDTH_M, MAX_PLAYERS_PER_TEAM, TEAMS

SPEED_NORM_CAP = 10.0  # m/s; elite volleyball movement speeds rarely exceed this

PER_PLAYER_FEATS = 4          # present, x_norm, y_norm, speed_norm
PER_TEAM_AGG_FEATS = 6        # centroid_x, centroid_y, spread_x, spread_y, mean_speed, num_players
STATE_DIM = 2 * (PER_PLAYER_FEATS * MAX_PLAYERS_PER_TEAM + PER_TEAM_AGG_FEATS) + 1


def feature_names():
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
    assert len(names) == STATE_DIM
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


def build_state(frame_record, rally_progress):
    """
    Parameters
    ----------
    frame_record : dict
        {"frame_idx": int, "players": [row, ...]} as produced by
        epv.rally_segmentation.segment_rallies.
    rally_progress : float in [0, 1]
        Position of this frame within its rally.

    Returns
    -------
    np.ndarray shape (STATE_DIM,), dtype float32
    """
    by_team = {team: [] for team in TEAMS}
    for row in frame_record["players"]:
        if row["team"] in by_team:
            by_team[row["team"]].append(row)

    blocks = [_team_block(by_team[team]) for team in TEAMS]
    state = np.concatenate(blocks + [np.array([rally_progress], dtype=np.float32)])
    assert state.shape[0] == STATE_DIM
    return state.astype(np.float32)


def build_rally_states(rally):
    """Build a (len(rally), STATE_DIM) array of states for every frame in a rally."""
    n = len(rally)
    states = np.zeros((n, STATE_DIM), dtype=np.float32)
    for t, frame_record in enumerate(rally):
        progress = t / (n - 1) if n > 1 else 0.0
        states[t] = build_state(frame_record, progress)
    return states
