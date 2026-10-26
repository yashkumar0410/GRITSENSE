"""
Ball track cleaning and ball-contact detection.

Cleaning (the raw tracker reports one detection per frame, many of them false):

    1. Physical speed gate: the fastest published attack (spike, mean + SD
       ~31 m/s) seen from the camera's closest court distance moves at most
       MAX_STEP_PX pixels per frame. Consecutive detections further apart than
       that cannot both be the ball, so the track is cut into consistent chains.
    2. Chains supported by 3+ detections (or 1-2 confident ones) are kept; when
       two kept chains contradict each other across a short gap, the weaker
       one (fewer, less confident detections) is dropped.
    3. Gaps of up to MAX_GAP frames between kept points are filled by fitting a
       quadratic in time to the neighbouring points (a ballistic path is
       quadratic over such short spans); fills that would break the speed gate
       are not made.

Touches (detect_touches): a sharp bend in the ball's path with a player within
reach, checked in metres through the camera (height between the floor and the
highest reach, within an arm's reach sideways). A player counts as "on the ball"
from shortly before a touch to shortly after, or while holding it still.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C

MAX_GAP = 8                       # frames that may be filled by interpolation
CONFIDENT = 0.45                  # a lone detection this confident is kept
FPS = 30.0


@dataclass
class BallPoint:
    frame: int
    x: float
    y: float
    radius: float
    confidence: float
    source: str                   # "detected" or "interpolated"


def max_step_px(camera) -> float:
    """Largest possible ball movement per frame in pixels (fastest spike, closest distance)."""
    v_max = C.value("spike_speed") + C.value("spike_speed_sd")
    centre = np.median([camera.centre(f) for f in range(0, len(camera.annotations), 10)], axis=0)
    # closest point of the court floor to the camera
    nearest = np.array([np.clip(centre[0], 0, 18), np.clip(centre[1], 0, 9), 0.0])
    distance = float(np.linalg.norm(centre - nearest))
    return camera.focal * v_max / distance / FPS


def load_raw(path: Path) -> list[BallPoint]:
    points = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Visibility"] == "1":
                points.append(BallPoint(int(r["Frame"]), float(r["X"]), float(r["Y"]),
                                        float(r["Radius"]), float(r["Confidence"]), "detected"))
    return points


def _chains(points, gate):
    chains, current = [], [points[0]]
    for prev, cur in zip(points, points[1:]):
        gap = cur.frame - prev.frame
        step = np.hypot(cur.x - prev.x, cur.y - prev.y)
        if gap <= MAX_GAP and step <= gate * gap + 15:
            current.append(cur)
        else:
            chains.append(current); current = [cur]
    chains.append(current)
    return chains


def _score(chain):
    return len(chain) * float(np.mean([p.confidence for p in chain]))


STATIC_RADIUS_PX = 15             # "same place" for static-object detection


def _static_detections(points: list[BallPoint], player_data: dict | None) -> set[int]:
    """Frames whose detection is a static object (mouse cursor, antenna, logo...).

    A detection is static when the same pixel spot is detected in at least two
    separate time spans, or for 4+ frames without moving, while no player is in a
    position to be holding the ball (a server holding the ball is not removed).
    """
    static = set()
    for p in points:
        same_spot = [q for q in points if abs(q.x - p.x) < STATIC_RADIUS_PX and abs(q.y - p.y) < STATIC_RADIUS_PX]
        frames = sorted(q.frame for q in same_spot)
        spans = 1 + sum(1 for a, b in zip(frames, frames[1:]) if b - a > MAX_GAP)
        if spans < 2 and len(frames) < 4:
            continue
        held = False
        if player_data is not None:
            records = player_data.get(str(p.frame), {})
            held = any(touch_score((p.x, p.y), r["bbox"]) < 1.0 for r in records.values())
        if not held:
            static.add(p.frame)
    return static


def clean(points: list[BallPoint], camera, player_data: dict | None = None) -> tuple[list[BallPoint], dict]:
    """Remove impossible detections and fill short gaps. Returns (track, stats)."""
    if not points:
        return [], {"raw": 0}
    gate = max_step_px(camera)
    static = _static_detections(points, player_data)
    points = [p for p in points if p.frame not in static]
    chains = _chains(sorted(points, key=lambda p: p.frame), gate)
    kept = [c for c in chains if len(c) >= 3 or np.mean([p.confidence for p in c]) >= CONFIDENT]

    # Neighbouring kept chains that cannot both be right: drop the weaker one.
    changed = True
    while changed and len(kept) > 1:
        changed = False
        for a, b in zip(kept, kept[1:]):
            gap = b[0].frame - a[-1].frame
            step = np.hypot(b[0].x - a[-1].x, b[0].y - a[-1].y)
            if gap <= MAX_GAP and step > gate * gap + 15:
                kept.remove(a if _score(a) < _score(b) else b)
                changed = True
                break

    track = [p for c in kept for p in c]
    track, n_filled = _fill_gaps(track, gate)
    stats = {"raw": len(points) + len(static), "removed_static_objects": len(static),
             "kept_detections": len(track) - n_filled,
             "removed_as_impossible": len(points) - (len(track) - n_filled),
             "interpolated": n_filled, "frames_with_ball": len(track),
             "max_step_px_per_frame": round(gate, 1)}
    return track, stats


# Path selection over several candidates per frame (redetect_ball.py output).
BASE_CONF = 0.12          # a candidate adds (confidence - BASE_CONF) to the path score
MISS_PENALTY = 0.02       # per frame without the ball
MOTION_WEIGHT = 0.2       # x (step / max step)^2: prefer smooth motion
RESTART_PENALTY = 0.6     # starting a new, unconnected piece of track


def _static_candidates(candidates: dict[int, list], player_data: dict | None) -> set[tuple[int, int]]:
    """(frame, index) of candidates at spots that keep being detected without moving
    (the first referee's head, the mouse cursor, the antenna) while nobody holds the ball."""
    flat = [(f, i, c) for f, cs in candidates.items() for i, c in enumerate(cs)]
    static = set()
    for f, i, c in flat:
        frames = {g for g, _, d in flat if abs(d[0] - c[0]) < STATIC_RADIUS_PX and abs(d[1] - c[1]) < STATIC_RADIUS_PX}
        if len(frames) < 8:
            continue
        held = False
        if player_data is not None:
            held = any(touch_score((c[0], c[1]), r["bbox"]) < 1.0
                       for r in player_data.get(str(f), {}).values())
        if not held:
            static.add((f, i))
    return static


def select_path(candidates: dict[int, list], camera, player_data: dict | None = None) -> tuple[list[BallPoint], dict]:
    """Choose at most one candidate per frame forming the best physically possible path."""
    gate = max_step_px(camera)
    static = _static_candidates(candidates, player_data)
    nodes = sorted(((f, c) for f, cs in candidates.items() for i, c in enumerate(cs) if (f, i) not in static),
                   key=lambda n: n[0])
    if not nodes:
        return [], {"candidates": 0}
    dp, back = [0.0] * len(nodes), [-1] * len(nodes)
    best_upto, best_idx = [], -1                       # running best path end (for restarts)
    j0 = 0
    for j, (fj, cj) in enumerate(nodes):
        while j0 < j and nodes[j0][0] < fj:            # nodes in earlier frames are final
            if best_idx < 0 or dp[j0] > dp[best_idx]:
                best_idx = j0
            j0 += 1
        score = cj[3] - BASE_CONF
        best, arg = -RESTART_PENALTY, -1
        if best_idx >= 0 and dp[best_idx] - RESTART_PENALTY > best:
            best, arg = dp[best_idx] - RESTART_PENALTY, best_idx
        for m in range(j - 1, -1, -1):
            fm, cm = nodes[m]
            gap = fj - fm
            if gap == 0:
                continue
            if gap > MAX_GAP + 1:
                break
            step = np.hypot(cj[0] - cm[0], cj[1] - cm[1])
            if step > gate * gap + 15:
                continue
            value = dp[m] - MISS_PENALTY * (gap - 1) - MOTION_WEIGHT * (step / (gate * gap)) ** 2
            if value > best:
                best, arg = value, m
        dp[j], back[j] = score + best, arg
    end = int(np.argmax(dp))
    chosen = []
    while end >= 0:
        chosen.append(end)
        end = back[end]
    chosen = sorted(set(chosen), key=lambda k: nodes[k][0])
    track = [BallPoint(nodes[k][0], nodes[k][1][0], nodes[k][1][1], nodes[k][1][2], nodes[k][1][3], "detected")
             for k in chosen]
    # one point per frame (keep the most confident if a restart produced a duplicate)
    by_frame = {}
    for p in track:
        if p.frame not in by_frame or p.confidence > by_frame[p.frame].confidence:
            by_frame[p.frame] = p
    track = sorted(by_frame.values(), key=lambda p: p.frame)
    detected = len(track)
    track, n_filled = _fill_gaps(track, gate)
    stats = {"candidates": sum(len(v) for v in candidates.values()), "static_removed": len(static),
             "frames_detected": detected, "interpolated": n_filled, "frames_with_ball": len(track),
             "max_step_px_per_frame": round(gate, 1)}
    return track, stats


def _fill_gaps(track: list[BallPoint], gate: float) -> tuple[list[BallPoint], int]:
    """Fill gaps of up to MAX_GAP frames with a quadratic-in-time fit to nearby points."""
    filled = []
    for a, b in zip(track, track[1:]):
        gap = b.frame - a.frame
        if 1 < gap <= MAX_GAP + 1 and np.hypot(b.x - a.x, b.y - a.y) <= gate * gap + 15:
            i = track.index(a)
            near = [p for p in track[max(0, i - 2): i + 4] if abs(p.frame - a.frame) <= MAX_GAP + 3]
            t = np.array([p.frame for p in near], float)
            deg = 2 if len(near) >= 4 else 1
            fx = np.polyfit(t, [p.x for p in near], deg)
            fy = np.polyfit(t, [p.y for p in near], deg)
            radius = float(np.median([p.radius for p in near]))
            for f in range(a.frame + 1, b.frame):
                filled.append(BallPoint(f, float(np.polyval(fx, f)), float(np.polyval(fy, f)),
                                        radius, 0.0, "interpolated"))
    return sorted(track + filled, key=lambda p: p.frame), len(filled)


def write_csv(track: list[BallPoint], n_frames: int, path: Path) -> None:
    """Same columns as ball.csv (so the existing pipeline can read it) plus Source."""
    by_frame = {p.frame: p for p in track}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Visibility", "X", "Y", "Radius", "Confidence", "Velocity_X", "Velocity_Y", "Source"])
        for frame in range(n_frames):
            p = by_frame.get(frame)
            if p is None:
                w.writerow([frame, 0, -1, -1, 0, 0.0, 0.0, 0.0, "none"])
                continue
            prev, nxt = by_frame.get(frame - 1), by_frame.get(frame + 1)
            vx = (nxt.x - prev.x) / 2 if prev and nxt else 0.0
            vy = (nxt.y - prev.y) / 2 if prev and nxt else 0.0
            w.writerow([frame, 1, round(p.x, 1), round(p.y, 1), round(p.radius, 1),
                        round(p.confidence, 4), round(vx, 2), round(vy, 2), p.source])


# ----------------------------------------------------------------------------
# Contact
# ----------------------------------------------------------------------------

KINK_PX = 20.0            # change of image velocity (px/frame) that marks a touch
KINK_WINDOW = 3           # frames used for the incoming / outgoing velocity
REACH_SCORE = 1.5         # touch_score below this = a player can play the ball
NEAR_BEFORE, NEAR_AFTER = 8, 3   # "on the ball" window around a touch (frames)
HOLD_SPEED_PX = 3.0       # slower than this next to a player = ball held (e.g. before a serve)


def load_clean(path: Path) -> list[BallPoint]:
    points = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Visibility"] == "1":
                points.append(BallPoint(int(r["Frame"]), float(r["X"]), float(r["Y"]), float(r["Radius"]),
                                        float(r["Confidence"]), r.get("Source", "detected")))
    return points


MAX_TOUCH_HEIGHT_M = C.value("attack_contact_m") + 0.3   # highest reach + measurement margin
MAX_TOUCH_SIDEWAYS_M = 1.5                                # about an arm's reach from the body


def ball_offset_from_player(camera, frame: int, rec: dict, ball_xy) -> tuple[float, float]:
    """(height above the floor, sideways distance) of the ball if it is at this player's depth.

    The ball is taken to be on the vertical plane through the player's feet (facing the
    camera); the camera then gives its height and sideways offset in metres.
    """
    x1, y1, x2, y2 = rec["bbox"]
    foot = ((x1 + x2) / 2.0, float(y2))
    ground = camera.ground_point(frame, foot)
    height = camera.height_above(frame, ground, ball_xy[1])
    here = camera.project(frame, (ground[0], ground[1], max(height, 0.0)))
    beside = camera.project(frame, (ground[0] + 1.0, ground[1], max(height, 0.0)))
    px_per_m = max(1e-6, float(np.linalg.norm(beside - here)))
    return height, abs(ball_xy[0] - here[0]) / px_per_m


def detect_touches(track: list[BallPoint], player_data: dict, camera) -> tuple[dict[int, int], list[dict]]:
    """Frames where a player is on the ball -> player id, plus the list of touch events.

    A touch is a sharp bend in the ball's path (the velocity over the next few frames
    differs from the previous few by more than KINK_PX) with a player within reach:
    through the camera, the ball must be between the floor and the highest reach
    above that player, and within an arm's reach sideways. In flight the path only
    curves gently under gravity, and a ball high in the air is above anyone's reach,
    so flights don't count even when the ball passes in front of a player in the
    image. A ball held still next to a player (before a serve) also counts.
    """
    pos = {p.frame: np.array([p.x, p.y]) for p in track}

    def reacher(frame, xy):
        best_id, best = None, None
        for rec in player_data.get(str(frame), {}).values():
            if touch_score(tuple(xy), rec["bbox"]) >= REACH_SCORE:
                continue
            height, sideways = ball_offset_from_player(camera, frame, rec, xy)
            if -0.2 <= height <= MAX_TOUCH_HEIGHT_M and sideways <= MAX_TOUCH_SIDEWAYS_M:
                if best is None or sideways < best:
                    best_id, best = int(rec["id"]), sideways
        return best_id

    kinks = {}
    for f in pos:
        before = [pos[g] for g in range(f - KINK_WINDOW, f + 1) if g in pos]
        after = [pos[g] for g in range(f, f + KINK_WINDOW + 1) if g in pos]
        if len(before) == KINK_WINDOW + 1 and len(after) == KINK_WINDOW + 1:
            v_in = (before[-1] - before[0]) / KINK_WINDOW
            v_out = (after[-1] - after[0]) / KINK_WINDOW
            kinks[f] = float(np.linalg.norm(v_out - v_in))

    events = []
    for f, k in sorted(kinks.items()):
        if k < KINK_PX:
            continue
        neighbours = [kinks.get(g, 0.0) for g in range(f - 3, f + 4) if g != f]
        if k < max(neighbours, default=0.0):               # keep the strongest bend nearby
            continue
        player = reacher(f, pos[f])
        if player is not None:
            events.append({"frame": f, "player": player, "kink_px": round(k, 1)})

    on_ball: dict[int, int] = {}
    for e in events:
        for g in range(e["frame"] - NEAR_BEFORE, e["frame"] + NEAR_AFTER + 1):
            on_ball.setdefault(g, e["player"])
    for f in pos:                                           # ball held, e.g. before a serve
        window = [pos.get(g) for g in range(f - 2, f + 3)]
        if any(w is None for w in window):
            continue
        # still for 5 frames (the top of a flight only hangs for 1-2)
        if max(np.linalg.norm(b - a) for a, b in zip(window, window[1:])) < HOLD_SPEED_PX:
            player = reacher(f, pos[f])
            if player is not None:
                on_ball.setdefault(f, player)
    return on_ball, events


def touch_score(ball_xy, bbox) -> float:
    """< 1 when the ball is where this player can play it (image space)."""
    x1, y1, x2, y2 = bbox
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    top, bottom = y1 - 0.6 * h, y1 + 0.75 * h          # above the head ... waist
    dx = max(0.0, abs(ball_xy[0] - cx) - 0.9 * w) / (0.5 * w)
    dy = max(0.0, top - ball_xy[1], ball_xy[1] - bottom) / (0.3 * h)
    return float(np.hypot(dx, dy)) + abs(ball_xy[0] - cx) / (3.0 * w)
