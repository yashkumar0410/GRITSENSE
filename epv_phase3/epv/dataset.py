"""
epv/dataset.py
----------------
Task 3 of Member 3's Phase 3 work: build a (state, action, outcome)
training dataset for the EPV model out of the perception pipeline's
per-frame player log.

IMPORTANT / HONEST LIMITATION — PROXY OUTCOME LABELS
-------------------------------------------------------
The repository does not contain true rally/point outcomes (who won the
rally, side-out vs point, etc.) — main.py's output is purely
positional/tracking data, and the Phase 2 report itself states the EPV
model "requires a larger labelled dataset of rally outcomes for robust
training" that does not exist yet.

`compute_proxy_outcome` below therefore uses a clearly-labelled, coarse,
REPLACEABLE placeholder: a team's outcome for a rally is the net court
territory (in the direction of the opponent's baseline) its players
gained between the first and last logged frame of that rally, normalized
to roughly [-1, 1]. This rewards "pushed the other team back", not
"scored a point" — it is a proxy for attacking momentum, not for actually
winning the rally.

Every sample this script writes is tagged `"outcome_is_proxy": true` in
the accompanying metadata file, and `compute_proxy_outcome` is kept as a
single, isolated function specifically so it can be swapped out the
moment real rally-outcome labels (manually labelled, or derived from the
group-activity annotations used in the CVPR-2016 Volleyball Dataset that
Phase 2 references) become available — no other file needs to change.
"""

import numpy as np

from epv.action_generator import ACTIONS, infer_action_from_transition, NUM_ACTIONS
from epv.rally_segmentation import assign_game_states, segment_rallies
from epv.state_builder import build_state, feature_names, state_dim
from epv.utils import (
    COURT_LENGTH_M,
    TEAMS,
    load_ball_log,
    load_frame_log,
    log_quality_report,
    save_json,
)

# team associated with each action index, e.g. left_receive/left_set/left_attack -> "left"
ACTION_TEAM_LOOKUP = [a.split("_")[0] for a in ACTIONS]


def compute_proxy_outcome(rally):
    """
    Proxy outcome per team for one rally: normalized net advance toward
    the opponent's baseline (see module docstring). Returns a dict
    {"left": float, "right": float}, each roughly in [-1, 1].
    """
    first_by_team = {t: [] for t in TEAMS}
    last_by_team = {t: [] for t in TEAMS}

    for row in rally[0]["players"]:
        if row["team"] in TEAMS:
            first_by_team[row["team"]].append(row["x_m"])
    for row in rally[-1]["players"]:
        if row["team"] in TEAMS:
            last_by_team[row["team"]].append(row["x_m"])

    outcomes = {}
    for team in TEAMS:
        if not first_by_team[team] or not last_by_team[team]:
            outcomes[team] = 0.0
            continue
        x_first = float(np.mean(first_by_team[team]))
        x_last = float(np.mean(last_by_team[team]))
        # "left" advances by increasing x (toward the net/right side);
        # "right" advances by decreasing x (toward the net/left side).
        delta = (x_last - x_first) if team == "left" else (x_first - x_last)
        outcomes[team] = float(np.clip(delta / (COURT_LENGTH_M / 2.0), -1.0, 1.0))
    return outcomes


def generate_dataset(
    log_path,
    out_path,
    max_frame_gap=15,
    min_rally_frames=10,
    limit=None,
    ball_log_path=None,
):
    """
    Build and save the (state, action, outcome) dataset.

    Parameters
    ----------
    log_path : str
        Path to a per-frame player log written by `main.py --log-out`.
    out_path : str
        Output .npz path. Also writes `<out_path>.meta.json` alongside it.
    limit : int, optional
        Only use the first `limit` rows of the log (quick test mode, mirrors
        the perception pipeline's own `--limit` flag).
    ball_log_path : str, optional
        Path to a court-space ball log (see epv.utils.load_ball_log). When
        given, the state vectors are ball-augmented (state_dim 76 instead
        of 69) and the metadata records `"has_ball_features": true`.

    Returns
    -------
    dict with keys states, actions, outcomes, rally_ids, teams (all np.ndarray)
    """
    rows = load_frame_log(log_path)
    if limit is not None:
        rows = rows[:limit]

    quality = log_quality_report(rows)
    if quality["warning"]:
        print(f"[EPV dataset][WARN] {quality['warning']}")
        print(f"[EPV dataset] log quality: {quality}")

    ball_by_frame = None
    include_ball = ball_log_path is not None
    if include_ball:
        ball_by_frame = load_ball_log(ball_log_path)
        print(f"[EPV dataset] ball log loaded: {len(ball_by_frame)} frames with ball detections")

    rallies = segment_rallies(
        rows, max_frame_gap=max_frame_gap, min_rally_frames=min_rally_frames,
        ball_by_frame=ball_by_frame,
    )

    states, actions, outcomes, rally_ids, teams = [], [], [], [], []
    game_state_counts = {}

    for rally_id, rally in enumerate(rallies):
        proxy = compute_proxy_outcome(rally)
        n = len(rally)
        labels = assign_game_states(rally) if include_ball else None
        if labels is not None:
            for lab in labels:
                game_state_counts[lab] = game_state_counts.get(lab, 0) + 1
        for t in range(n - 1):
            progress = t / (n - 1) if n > 1 else 0.0
            state = build_state(rally[t], progress, include_ball=include_ball)
            action = infer_action_from_transition(rally[t], rally[t + 1])
            team = ACTION_TEAM_LOOKUP[action]
            outcome = proxy[team]

            states.append(state)
            actions.append(action)
            outcomes.append(outcome)
            rally_ids.append(rally_id)
            teams.append(team)

    if not states:
        raise RuntimeError(
            "No (state, action, outcome) samples were generated. Check that "
            "the frame log actually contains rallies of at least "
            f"{min_rally_frames} frames (max_frame_gap={max_frame_gap})."
        )

    data = {
        "states": np.stack(states).astype(np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "outcomes": np.array(outcomes, dtype=np.float32),
        "rally_ids": np.array(rally_ids, dtype=np.int64),
        "teams": np.array(teams),
    }

    np.savez_compressed(
        out_path,
        states=data["states"],
        actions=data["actions"],
        outcomes=data["outcomes"],
        rally_ids=data["rally_ids"],
        teams=data["teams"],
    )

    n_unique_outcomes = int(np.unique(data["outcomes"]).size)
    if n_unique_outcomes <= 2:
        print(
            f"[EPV dataset][WARN] only {n_unique_outcomes} unique outcome values "
            "in the dataset -- proxy labels are degenerate for this log; the "
            "model cannot learn more than this constant-per-team mapping."
        )

    save_json(
        {
            "num_samples": int(len(actions)),
            "num_rallies": int(len(rallies)),
            "state_dim": state_dim(include_ball),
            "has_ball_features": include_ball,
            "ball_log_path": str(ball_log_path) if include_ball else None,
            "num_actions": NUM_ACTIONS,
            "feature_names": feature_names(include_ball),
            "source_player_log": str(log_path),
            "log_quality": quality,
            "n_unique_outcomes": n_unique_outcomes,
            "game_state_counts": game_state_counts,
            "outcome_is_proxy": True,
            "proxy_outcome_definition": (
                "Per-team normalized net court-territory gain (toward the "
                "opponent baseline) between a rally's first and last logged "
                "frame. NOT a true point/rally-win label -- see "
                "epv/dataset.py module docstring."
            ),
        },
        str(out_path) + ".meta.json",
    )

    return data
