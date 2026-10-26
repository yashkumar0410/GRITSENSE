"""
Measure the physics constants from this video instead of assuming them.

Uses the court homography and the camera recovered from it (analytics/camera.py):

    Camera           focal length and position (checked: stable over the clip)
    Player stature   median height of standing players' boxes, in metres
    Top speed        95th percentile of tracked player speed (higher percentiles
                     are tracking glitches)
    Block jump       90th percentile of peak jump heights near the net
    Ball speed       sanity check only (too few clean ball detections to fit)
    Validation       the model's mean best-spike point chance on this clip is
                     compared with published attack success (2010 Men's World
                     Championship: 72.5% outside hitters, 55.1% opposites).

Writes analytics/physics_calibration.json, which analytics/constants.py loads.

Run from the project root:
    python calibrate_physics.py
"""

from __future__ import annotations

import csv
import json

import numpy as np

from analytics import constants as C
from analytics.camera import Camera
from analytics.motion import MotionSmoother
from analytics.pose_signals import JumpDetector
from tactical_dashboard import BALL_FILE, ROOT, BallState, homography_for_frame, load_ball_rows, players_for_frame, project

FPS = 30.0
OUT = ROOT / "analytics" / "physics_calibration.json"
SPEED_RANGE = (3.0, 8.0)          # plausible volleyball top speeds; outside this, keep the fallback


def player_speeds(player_data, annotations):
    tracks, H = {}, None
    for f in range(len(player_data)):
        H = homography_for_frame(annotations, f, H)
        for p in player_data[str(f)].values():
            (x, y), = project(H, [[p["image_x"], p["image_y"]]])
            if -2 <= x <= 20 and -1.2 <= y <= 10.2:
                tracks.setdefault(int(p["id"]), {})[f] = np.array([x, y])
    speeds = []
    for track in tracks.values():
        for f in track:
            before = [track[k] for k in range(f - 5, f - 1) if k in track]
            after = [track[k] for k in range(f + 2, f + 6) if k in track]
            if len(before) == 4 and len(after) == 4:
                speeds.append(np.linalg.norm(np.mean(after, 0) - np.mean(before, 0)) / (7 / FPS))
    return np.array(speeds)


def body_measurements(player_data, camera):
    detector, standing, jumps = JumpDetector(camera), [], {}
    for f in range(len(player_data)):
        for p in player_data[str(f)].values():
            body = detector.update(f, p)
            gx, gy = body.ground_xy or camera.ground_point(f, ((p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][3]))
            if not (0 <= gx <= 18 and 0 <= gy <= 9):
                continue
            if body.jump_m == 0.0:
                standing.append(body.reach_m)
            elif body.jump_m > 0.25:
                jumps.setdefault(int(p["id"]), []).append((f, body.jump_m, body.reach_m, abs(gx - 9.0)))
    peaks = []                                       # the highest frame of each separate jump
    for track in jumps.values():
        current = [track[0]]
        for item in track[1:]:
            if item[0] - current[-1][0] <= 2:
                current.append(item)
            else:
                peaks.append(max(current, key=lambda t: t[1])); current = [item]
        peaks.append(max(current, key=lambda t: t[1]))
    return np.array(standing), np.array([(j, r, d) for _, j, r, d in peaks])


def ball_speeds(annotations):
    with (ROOT / "ball.csv").open(encoding="utf-8") as f:
        rows = {int(r["Frame"]): r for r in csv.DictReader(f) if r["Visibility"] == "1"}
    speeds, H = [], None
    for f in sorted(rows):
        H = homography_for_frame(annotations, f, H)
        nxt = rows.get(f + 1)
        if nxt is None or float(rows[f]["Confidence"]) < 0.4 or float(nxt["Confidence"]) < 0.4:
            continue
        p1, p2 = project(np.linalg.inv(H), [[8.5, 4.5], [9.5, 4.5]])   # 1 m along the court at the net
        px_per_m = float(np.linalg.norm(p2 - p1))
        speed = abs(float(nxt["X"]) - float(rows[f]["X"])) / px_per_m * FPS
        if speed < 40:                                                # reject detection jumps
            speeds.append(speed)
    return np.array(speeds)


def touch_heights(player_data, camera):
    """Ball height above the player at every detected touch (cleaned ball track)."""
    from analytics.ball_track import ball_offset_from_player, detect_touches, load_clean
    if BALL_FILE.name != "ball_clean.csv":
        return []
    track = load_clean(BALL_FILE)
    positions = {p.frame: (p.x, p.y) for p in track}
    _, events = detect_touches(track, player_data, camera)
    heights = []
    for e in events:
        rec = player_data.get(str(e["frame"]), {}).get(str(e["player"]))
        if rec is not None:
            height, _ = ball_offset_from_player(camera, e["frame"], rec, positions[e["frame"]])
            heights.append({"frame": e["frame"], "player": e["player"], "height_m": round(height, 2)})
    return heights


def attack_situations(player_data, annotations, camera):
    """(contact, defenders, defend side) for every frame with an attack, using the insight engine."""
    from analytics.insights import InsightEngine
    engine, ball_state, jumps = InsightEngine(), BallState(), JumpDetector(camera)
    motion = MotionSmoother(FPS, C.value("top_speed"))
    from analytics.ball_track import detect_touches, load_clean
    touches, _ = (detect_touches(load_clean(BALL_FILE), player_data, camera)
                  if BALL_FILE.name == "ball_clean.csv" else ({}, []))
    ball_rows = load_ball_rows(BALL_FILE)
    situations, H = [], None
    for f in range(len(player_data)):
        H = homography_for_frame(annotations, f, H)
        players = players_for_frame(player_data.get(str(f), {}), H, FPS, f, jumps, motion)
        toucher = touches.get(f)
        ins = engine.step(f, players, ball_state.update(f, ball_rows.get(f), H), toucher)
        if ins.surface is not None:
            defenders = [p for p in players if p.team == ins.defend_team]
            situations.append((ins.surface.contact, defenders, ins.defend_team))
    return situations


def validate_attacks(situations):
    """Mean point chance of the best spike target, to compare with published attack success."""
    import analytics.epv as E
    model = E.PhysicsEPV()
    spike_frames = [s for s in situations
                    if any(shot.name == "SPIKE" for shot in model.available_shots(s[0], 1 - s[2]))]
    best, average = [], []
    for c, d, side in spike_frames:
        values = model.surface(c, d, side).values
        best.append(values.max())                       # perfectly placed attack
        average.append(values[values > 0].mean())       # attack aimed anywhere it can legally land
    return float(np.mean(best)), float(np.mean(average)), len(spike_frames)


def main():
    with (ROOT / "player_keypoints_mf3_pose.json").open(encoding="utf-8") as f:
        player_data = json.load(f)
    with (ROOT / "annotations.json").open(encoding="utf-8") as f:
        annotations = json.load(f)
    camera = Camera(annotations)
    centres = np.array([camera.centre(f) for f in range(len(player_data))])

    speeds = player_speeds(player_data, annotations)
    top_speed = float(np.percentile(speeds, 95))
    standing, peaks = body_measurements(player_data, camera)
    near_net = peaks[peaks[:, 2] < 2.0] if len(peaks) else peaks
    stature = float(np.median(standing))
    block_jump = float(np.percentile(near_net[:, 0], 90)) if len(near_net) else None
    balls = ball_speeds(annotations)

    # Use the measured values while fitting the reaction time.
    import analytics.epv as E
    if SPEED_RANGE[0] <= top_speed <= SPEED_RANGE[1]:
        E.TOP_SPEED = top_speed
    E.DIVE_REACH = stature / 2
    if block_jump:
        E.BLOCK_JUMP = block_jump
        E.STANDING_BLOCK_REACH = C.value("block_reach_m") - block_jump
    touches = touch_heights(player_data, camera)
    print("Validating against published attack success (runs the insight engine over the clip)...")
    spike_best, spike_average, n_spike = validate_attacks(attack_situations(player_data, annotations, camera))

    calibration = {
        "top_speed_mps": round(top_speed, 2) if SPEED_RANGE[0] <= top_speed <= SPEED_RANGE[1] else None,
        "stature_m": round(stature, 2),
        "block_jump_m": round(block_jump, 2) if block_jump else None,
        "measured": {
            "camera_focal_px": camera.focal,
            "camera_centre_median_m": [round(float(v), 2) for v in np.median(centres, axis=0)],
            "camera_centre_std_m": [round(float(v), 2) for v in centres.std(axis=0)],
            "player_speed_samples": int(len(speeds)),
            "player_speed_median_mps": round(float(np.median(speeds)), 2),
            "player_speed_p99_mps": round(float(np.percentile(speeds, 99)), 2),
            "standing_height_samples": int(len(standing)),
            "jumps_detected": int(len(peaks)),
            "jump_peak_median_near_net_m": round(float(np.median(near_net[:, 0])), 2) if len(near_net) else None,
            "jump_peak_max_m": round(float(peaks[:, 0].max()), 2) if len(peaks) else None,
            "reach_max_near_net_m": round(float(near_net[:, 1].max()), 2) if len(near_net) else None,
            "ball_clean_pairs": int(len(balls)),
            "ball_horizontal_max_mps": round(float(balls.max()), 1) if len(balls) else None,
            "touches": touches,
            "touch_height_max_m": max((t["height_m"] for t in touches), default=None),
            "validation_spike_frames": n_spike,
            "validation_best_target_point_chance": round(spike_best, 4),
            "validation_average_target_point_chance": round(spike_average, 4),
            "validation_published_attack_success": {"outside_hitters": 0.725, "opposites": 0.551},
        },
        "notes": ("stature, top speed and block jump are measured through the camera recovered from the "
                  "court homography. Ball speeds keep published values (too few clean ball detections in "
                  "one clip). The validation compares the model's best-spike point chance with published "
                  "terminal-attack success; it is a check, not a fit."),
    }
    OUT.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(calibration, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
