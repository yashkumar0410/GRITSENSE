"""
epv/action_generator.py
-------------------------
Task 3 (part) of Member 3's Phase 3 work: define the discrete action space
used by the EPV model, and a way to label which action was (probably)
being executed at a given transition in the perception log.

IMPORTANT / HONEST LIMITATION
------------------------------
The perception pipeline does NOT currently classify player-level actions
(setting, spiking, blocking, ...) — the Phase 2 report lists 9 such
classes as part of the target *dataset*, but main.py only outputs
position + speed per tracked player, nothing action-specific. There is
therefore no ground-truth action label available yet.

`infer_action_from_transition` below is a documented, replaceable
zone/speed heuristic: it looks at which team's player moved fastest
between two consecutive frames and which third of the court (relative to
that team's own baseline) they were in, and maps that to a coarse
"receive / set / attack" bucket for that team. This is a reasonable
stand-in given only position+speed are available, but it is a PROXY, not
a real action classifier. It is isolated in this one function specifically
so it can be swapped for a real trained action classifier later (e.g. once
the GNN/Transformer branch of the project — Phase 3 tasks for Members 1/2
— exposes per-player action logits, that should be threaded through here
instead).
"""

import numpy as np

from epv.utils import COURT_LENGTH_M, TEAMS

ACTIONS = [
    "left_receive", "left_set", "left_attack",
    "right_receive", "right_set", "right_attack",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

# Own-baseline x-coordinate for each team (net is at x = COURT_LENGTH_M / 2).
_BASELINE_X = {"left": 0.0, "right": COURT_LENGTH_M}
_NET_X = COURT_LENGTH_M / 2.0


def generate_candidate_actions(state=None):
    """
    Return every candidate action index. The EPV model/engine evaluates
    all of these for a given state and picks the highest-value one
    (see epv/evaluate.py:action ranking quality). `state` is accepted but
    unused for now — kept in the signature so a future, state-conditioned
    action shortlist (e.g. only actions valid for the team currently
    receiving serve) can be dropped in without changing callers.
    """
    return list(range(NUM_ACTIONS))


def _zone_for_player(team, x_m):
    """distance-from-own-baseline -> 'receive' (back court) / 'set' (mid) / 'attack' (near net)."""
    dist_from_net = abs(x_m - _NET_X)
    if dist_from_net < COURT_LENGTH_M * (1.0 / 6.0):       # within ~3m of net
        return "attack"
    elif dist_from_net < COURT_LENGTH_M * (1.0 / 3.0):     # within ~6m of net
        return "set"
    return "receive"


def infer_action_from_transition(frame_t, frame_t1):
    """
    Heuristic label for "what action was probably happening" between two
    consecutive frame records (see docstring above for the caveat).

    Parameters
    ----------
    frame_t, frame_t1 : dict
        {"frame_idx": int, "players": [row, ...]}

    Returns
    -------
    int action index into ACTIONS
    """
    players_t = {(r["player_id"]): r for r in frame_t["players"]}
    best_row, best_speed = None, -1.0

    for row in frame_t1["players"]:
        if row["speed_mps"] > best_speed:
            best_speed = row["speed_mps"]
            best_row = row

    if best_row is None:
        return ACTION_TO_IDX["left_receive"]  # arbitrary fallback, no players tracked

    team = best_row["team"] if best_row["team"] in TEAMS else "left"
    zone = _zone_for_player(team, best_row["x_m"])
    return ACTION_TO_IDX[f"{team}_{zone}"]
