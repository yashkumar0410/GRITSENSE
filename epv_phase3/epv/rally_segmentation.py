"""
epv/rally_segmentation.py
--------------------------
Task 1 of Member 3's Phase 3 work: segment the perception pipeline's
per-frame player log into rally-level sequences.

IMPORTANT / HONEST LIMITATION
------------------------------
The current perception pipeline (main.py) does not output a serve/rally
start-end signal, a "ball in play" flag, or point-outcome labels — it only
outputs, per frame, which players were tracked and where. Real rally
boundaries (whistle, serve, dead ball) are therefore NOT available.

Proxy rally-segmentation strategy (documented, replaceable):
    A rally boundary is declared whenever there is a gap of more than
    `max_frame_gap` frames with no tracked players at all (players lost
    from tracking / no valid homography that frame), OR when the video's
    frame index itself jumps by more than `max_frame_gap` (e.g. because
    the perception loop skipped/failed frames).

    This is a reasonable proxy given what BotSort + the existing homography
    step already expose, but it is NOT ground-truth rally segmentation. If
    the team later gets access to real serve/point annotations (e.g. by
    extending the group-activity labels already used in Phase 2's
    Volleyball Dataset), `segment_rallies` should be replaced/augmented
    with that signal — the function boundary below is kept narrow
    specifically so that swap is easy.
"""

from epv.utils import rows_to_frames

# Game-state labels (proxy definitions, see assign_game_states)
STATE_DEAD = "dead_ball"      # no ball tracked / play not visibly live
STATE_SERVE = "serve_prep"    # ball tracked but (near-)stationary -> serve prep
STATE_LIVE = "live_rally"     # ball tracked and moving -> rally in play

# A ball slower than this (m/s) is considered "not in play" by the proxy.
_BALL_LIVE_SPEED_MPS = 1.0


def segment_rallies(rows, max_frame_gap=15, min_rally_frames=10, ball_by_frame=None):
    """
    Parameters
    ----------
    rows : list[dict]
        Flat per-player-per-frame rows, as produced by
        `main.py --log-out` (see epv/utils.py for schema).
    max_frame_gap : int
        Max allowed gap (in frame indices) between two consecutive frames
        that still have tracked players before we cut a new rally.
    min_rally_frames : int
        Rallies shorter than this (in frames) are dropped as noise
        (e.g. a single player flickering in/out of detection).
    ball_by_frame : dict[int, dict], optional
        Mapping frame_idx -> {"x_m", "y_m", "speed_mps"} from
        epv.utils.load_ball_log. When given, each frame record gets a
        "ball" key so state_builder can use ball features.

    Returns
    -------
    list[list[dict]]
        Each element is a rally: an ordered list of per-frame records
        {"frame_idx": int, "players": [row, row, ...], "ball": dict|None}.
    """
    frames_dict = rows_to_frames(rows)
    frame_indices = list(frames_dict.keys())

    rallies = []
    current_rally = []
    prev_idx = None

    for idx in frame_indices:
        players = frames_dict[idx]
        if not players:
            continue

        if prev_idx is not None and (idx - prev_idx) > max_frame_gap:
            if len(current_rally) >= min_rally_frames:
                rallies.append(current_rally)
            current_rally = []

        record = {"frame_idx": idx, "players": players}
        if ball_by_frame is not None:
            record["ball"] = ball_by_frame.get(idx)
        current_rally.append(record)
        prev_idx = idx

    if len(current_rally) >= min_rally_frames:
        rallies.append(current_rally)

    return rallies


def assign_game_states(rally):
    """
    PROXY game-state labelling for one rally (documented, replaceable).

    The perception pipeline has no whistle/serve/point signal. With a ball
    log attached, the cheapest defensible proxy for "is the ball in play"
    is its speed:
        - no ball tracked that frame          -> "dead_ball"
        - ball tracked, speed < 1 m/s         -> "serve_prep" (ball being
                                                 held/tossed for a serve)
        - ball tracked, speed >= 1 m/s        -> "live_rally"

    Without a ball log every frame is labelled "dead_ball" (i.e. unknown),
    and callers should treat the labels as unavailable rather than real.

    Returns a list of labels, one per frame of the rally.
    """
    labels = []
    for frame_record in rally:
        ball = frame_record.get("ball")
        if not ball:
            labels.append(STATE_DEAD)
        elif ball.get("speed_mps", 0.0) < _BALL_LIVE_SPEED_MPS:
            labels.append(STATE_SERVE)
        else:
            labels.append(STATE_LIVE)
    return labels
