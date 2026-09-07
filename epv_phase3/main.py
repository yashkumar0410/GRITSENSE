import argparse
import cv2
import os
import torch
from ultralytics import YOLO
import numpy as np
import json

# --- EPV INTEGRATION (Member 3) -------------------------------------------
# The block below is an ADDITIVE, non-invasive extension to the existing
# perception pipeline. It does NOT change detection, tracking, pose or
# homography logic. It only:
#   1. lets the video / court-points / frame-limit be passed via CLI
#      (defaults are unchanged, so `python main.py` behaves exactly as
#      before), and
#   2. optionally records the per-frame player state (id, team, real-world
#      court position in metres, speed) that is ALREADY being computed by
#      this script, so the EPV module (epv/) can consume it without
#      duplicating any detection/tracking/homography code.
parser = argparse.ArgumentParser(description="GritSense perception pipeline")
parser.add_argument("--video", default="sample3.mp4", help="input video path")
court_json_default = next(
    (p for p in ("annotations.json", "../annotations.json", "cp.json") if os.path.exists(p)), "annotations.json"
)
parser.add_argument("--court-json", default=court_json_default, help="per-frame court corner points")
parser.add_argument("--model", default="player_detection.pt", help="player detection/tracking weights")
parser.add_argument("--log-out", default=None, help="if set, dump per-frame player state log (JSON) for EPV pipeline")
parser.add_argument("--ball-log-out", default=None, help="if set, dump per-frame court-space ball log (JSON) for EPV pipeline (enables ball features in the EPV state vector)")
parser.add_argument("--ball-model", default=None, help="ball detection weights (default: ball.pt when --ball-log-out is set)")
parser.add_argument("--limit", type=int, default=None, help="process only the first N frames (quick test mode)")
parser.add_argument("--no-display", action="store_true", help="disable cv2.imshow (headless mode)")
args, _unknown = parser.parse_known_args()
# ---------------------------------------------------------------------------

player_model = YOLO(args.model)

# EPV INTEGRATION: optional ball detection model (additive; only loaded when
# --ball-log-out is requested, so default runs are unchanged). ball.pt lives
# in the repo root, not in epv_phase3/, so fall back to the parent directory.
ball_model = None
if args.ball_log_out is not None:
    ball_weights = args.ball_model
    if ball_weights is None:
        ball_weights = next(
            (p for p in ("ball.pt", "../ball.pt") if os.path.exists(p)), "ball.pt"
        )
    ball_model = YOLO(ball_weights)

with open(args.court_json, "r", encoding="utf-8") as f:
    court_points_by_frame = json.load(f)

# Auto-detect resolution of court annotations (1080p annotations vs 720p/480p video)
max_ann_x = max((max(pt[0] for pt in pts) for pts in court_points_by_frame.values() if pts), default=1280)
ann_ref_w = 1920.0 if max_ann_x > 1280 else 1280.0
ann_ref_h = 1080.0 if max_ann_x > 1280 else 720.0

cap = cv2.VideoCapture(args.video)
vid_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280.0
vid_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720.0
ann_scale_x = vid_w / ann_ref_w
ann_scale_y = vid_h / ann_ref_h

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps and fps > 1 else 30

device = 0 if torch.cuda.is_available() else "cpu"

out = None

court_image = cv2.imread("court.png")
court_w, court_h = 300, 150
court_image = cv2.resize(court_image, (court_w, court_h))
base_court = court_image.copy()
alpha = 0.6
margin = 20
actual_width = 18
actual_height = 9

court_pts = np.array([
    [0, 0],
    [court_w, 0],
    [court_w, court_h],
    [0, court_h]
], dtype=np.float32)

H = None

prev_positions = {}
prev_frame_idx = {}
frame_idx = 0
player_colors = {}
BROWN = (0, 0, 0)
EXCLUDED_IDS = {20, 63, 69}

# ---------------------------------------------------------------------------
# 2D SPATIAL TEAM TRACKER
# Completely ignores BotSort raw IDs (which jump randomly on scene cuts).
# Matches detections to existing tracks by minimizing 2D distance.
# Left Team  (China, px < 640) -> IDs 1, 2, 3, 4, 5, 6  (Blue)
# Right Team (USA,   px >= 640) -> IDs 7, 8, 9, 10, 11, 12 (Red)
# ---------------------------------------------------------------------------
class SpatialTeamTracker:
    MAX_MATCH_DIST = 120.0  # max screen pixel distance between consecutive 30fps frames
    MAX_LOST_FRAMES = 30   # hold track position for 1 second if temporarily missed

    def __init__(self):
        self.tracks = {
            "left": {i: None for i in range(1, 7)},
            "right": {i: None for i in range(7, 13)},
        }

    def update_team(self, team_name: str, detections: list, frame_idx: int) -> list:
        """
        detections: list of (px, py, mx, my, box_idx)
        Returns: list of (px, py, mx, my, box_idx, assigned_id)
        """
        pool = list(range(1, 7)) if team_name == "left" else list(range(7, 13))
        team_tracks = self.tracks[team_name]

        # Expire stale tracks
        for sid in pool:
            if team_tracks[sid] is not None:
                _, _, _, _, last_f = team_tracks[sid]
                if frame_idx - last_f > self.MAX_LOST_FRAMES:
                    team_tracks[sid] = None

        if not detections:
            return []

        active_ids = [sid for sid in pool if team_tracks[sid] is not None]
        assigned = []
        unassigned_dets = list(range(len(detections)))

        # Step 1: Spatial Nearest-Neighbor Matching
        if active_ids and unassigned_dets:
            pairs = []
            for d_idx in unassigned_dets:
                px, py, mx, my, _ = detections[d_idx]
                for sid in active_ids:
                    prev_px, prev_py, _, _, _ = team_tracks[sid]
                    dist = np.hypot(px - prev_px, py - prev_py)
                    if dist <= self.MAX_MATCH_DIST:
                        pairs.append((dist, sid, d_idx))

            pairs.sort(key=lambda x: x[0])
            used_sids = set()
            used_dets = set()

            for dist, sid, d_idx in pairs:
                if sid not in used_sids and d_idx not in used_dets:
                    used_sids.add(sid)
                    used_dets.add(d_idx)
                    px, py, mx, my, b_idx = detections[d_idx]
                    team_tracks[sid] = (px, py, mx, my, frame_idx)
                    assigned.append((px, py, mx, my, b_idx, sid))

            unassigned_dets = [d for d in unassigned_dets if d not in used_dets]

        # Step 2: Assign free pool IDs to new detections
        free_sids = [sid for sid in pool if team_tracks[sid] is None]
        for d_idx in unassigned_dets:
            if not free_sids:
                break
            sid = free_sids.pop(0)
            px, py, mx, my, b_idx = detections[d_idx]
            team_tracks[sid] = (px, py, mx, my, frame_idx)
            assigned.append((px, py, mx, my, b_idx, sid))

        return assigned

spatial_tracker = SpatialTeamTracker()

# EPV INTEGRATION: per-frame log of player state, reusing values already
# computed below (court-space position in metres + speed + team).
frame_log = [] if args.log_out else None
# EPV INTEGRATION: per-frame court-space ball log (only when --ball-log-out).
ball_frame_log = {} if args.ball_log_out else None
prev_ball = None  # (frame_idx, x_m, y_m) for the ball speed calculation
px_per_meter = court_w / actual_width  # same scale used for speed calc below

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]
    annotated_frame = frame.copy()

    # Update homography H at the START of the frame using scaled corners
    frame_corner_points = court_points_by_frame.get(str(frame_idx), [])
    if len(frame_corner_points) >= 4:
        top_left = (int(frame_corner_points[0][0] * ann_scale_x), int(frame_corner_points[0][1] * ann_scale_y))
        top_right = (int(frame_corner_points[1][0] * ann_scale_x), int(frame_corner_points[1][1] * ann_scale_y))
        bottom_right = (int(frame_corner_points[2][0] * ann_scale_x), int(frame_corner_points[2][1] * ann_scale_y))
        bottom_left = (int(frame_corner_points[3][0] * ann_scale_x), int(frame_corner_points[3][1] * ann_scale_y))

        frame_pts = np.array(
            [top_left, top_right, bottom_right, bottom_left],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(frame_pts, court_pts)

    results = player_model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="botsort.yaml",
        verbose=False,
        device=device,
        half=True,
    )

    for r in results:
        boxes = r.boxes

        if boxes.id is None or len(boxes.id) == 0:
            continue

        if H is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            
            raw_left_dets = []
            raw_right_dets = []

            for i in range(len(ids)):
                x1, y1, x2, y2 = xyxy[i]
                raw_id = ids[i]
                if raw_id in EXCLUDED_IDS:
                    continue

                px = (x1 + x2) // 2
                py = y2

                # Filter out-of-bounds non-players (crowd at top/bottom edges)
                if py > h_frame - 40 or py < 220:
                    continue

                point = np.array([[[px, py]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(point, H)
                mx, my = mapped[0][0]
                mx, my = int(mx), int(my)

                if px < w_frame // 2:
                    raw_left_dets.append((px, py, mx, my, i))
                else:
                    raw_right_dets.append((px, py, mx, my, i))

            # Perform 2D spatial tracking for both teams
            left_players = spatial_tracker.update_team("left", raw_left_dets, frame_idx)
            right_players = spatial_tracker.update_team("right", raw_right_dets, frame_idx)

            for px, py, mx, my, idx, player_id in left_players:
                overlay = base_court.copy()
                cv2.circle(overlay, (mx, my), 12, (255, 0, 0, 80), -1)
                cv2.addWeighted(overlay, 0.4, base_court, 0.6, 0, base_court)
                cv2.circle(base_court, (mx, my), 5, (255, 0, 0), -1)

                speed = 0.0

                if player_id in prev_positions and (frame_idx - prev_frame_idx[player_id]) == 1:
                    prev_mx, prev_my = prev_positions[player_id]
                    dist_px = np.sqrt((mx - prev_mx) ** 2 + (my - prev_my) ** 2)
                    px_per_meter = court_w / actual_width
                    dist_m = dist_px / px_per_meter
                    speed = dist_m * fps

                prev_positions[player_id] = (mx, my)
                prev_frame_idx[player_id] = frame_idx
                player_colors[player_id] = (255, 0, 0)

                cv2.putText(base_court, f"{player_id}", (mx-8, my-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                if 'player_speeds' not in locals():
                    player_speeds = {}
                player_speeds[player_id] = speed

                if frame_log is not None:
                    frame_log.append({
                        "frame_idx": frame_idx,
                        "player_id": int(player_id),
                        "team": "left",
                        "x_m": round(mx / px_per_meter, 4),
                        "y_m": round(my / px_per_meter, 4),
                        "speed_mps": round(float(speed), 4),
                    })

            for px, py, mx, my, idx, player_id in right_players:
                overlay = base_court.copy()
                cv2.circle(overlay, (mx, my), 12, (0, 0, 255, 80), -1)
                cv2.addWeighted(overlay, 0.4, base_court, 0.6, 0, base_court)
                cv2.circle(base_court, (mx, my), 5, (0, 0, 255), -1)
                speed = 0.0

                if player_id in prev_positions and (frame_idx - prev_frame_idx[player_id]) == 1:
                    prev_mx, prev_my = prev_positions[player_id]
                    dist_px = np.sqrt((mx - prev_mx) ** 2 + (my - prev_my) ** 2)
                    px_per_meter = court_w / actual_width
                    dist_m = dist_px / px_per_meter
                    speed = dist_m * fps

                prev_positions[player_id] = (mx, my)
                prev_frame_idx[player_id] = frame_idx
                player_colors[player_id] = (0, 0, 255)

                cv2.putText(base_court, f"{player_id}", (mx-8, my-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                if 'player_speeds' not in locals():
                    player_speeds = {}
                player_speeds[player_id] = speed

                if frame_log is not None:
                    frame_log.append({
                        "frame_idx": frame_idx,
                        "player_id": int(player_id),
                        "team": "right",
                        "x_m": round(mx / px_per_meter, 4),
                        "y_m": round(my / px_per_meter, 4),
                        "speed_mps": round(float(speed), 4),
                    })

            for y in range(0, court_h, 12):
                cv2.line(base_court, (court_w // 2, y),
                         (court_w // 2, min(y+6, court_h)), (0, 255, 255), 3)

            for px, py, mx, my, idx, track_id in left_players + right_players:
                x1, y1, x2, y2 = xyxy[idx]
                cbx = (x1 + x2) // 2
                width = x2 - x1
                shirt_color = player_colors.get(track_id, (255, 0, 0))

                cv2.ellipse(annotated_frame, (cbx, y2),
                            (int(width), int(0.35 * width)),
                            0, -45, 235, shirt_color, 3)

                rect_w, rect_h = 40, 20
                x1_rect = cbx - rect_w // 2
                y1_rect = y2 - rect_h // 2

                x1_text = x1_rect + 12

                speed = 0.0
                if 'player_speeds' in locals() and track_id in player_speeds:
                    speed = player_speeds[track_id]

                cv2.putText(annotated_frame, f"{track_id}",
                            (x1_text, y1_rect + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                cv2.putText(annotated_frame, f"{speed:.2f} m/s",
                            (x1_text, y1_rect + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    # EPV INTEGRATION: optional per-frame ball detection -> court-space ball
    # log. Uses the most confident ball detection, mapped through the same
    # homography as the players. Nothing here touches the player pipeline.
    if ball_frame_log is not None and ball_model is not None:
        ball_results = ball_model.predict(frame, conf=0.25, verbose=False, device=device)
        best = None
        for br in ball_results:
            if br.boxes is None:
                continue
            for b in br.boxes:
                conf = float(b.conf[0])
                if best is None or conf > best[0]:
                    x1b, y1b, x2b, y2b = b.xyxy[0].tolist()
                    best = (conf, (x1b + x2b) / 2.0, (y1b + y2b) / 2.0)

        if best is not None and H is not None:
            _conf, bx, by = best
            mapped = cv2.perspectiveTransform(
                np.array([[[bx, by]]], dtype=np.float32), H
            )
            bmx, bmy = float(mapped[0][0][0]), float(mapped[0][0][1])
            bx_m = bmx / px_per_meter
            by_m = bmy / (court_h / actual_height)

            ball_speed = 0.0
            if prev_ball is not None and frame_idx - prev_ball[0] == 1:
                ball_speed = float(
                    np.hypot(bx_m - prev_ball[1], by_m - prev_ball[2]) * fps
                )
            prev_ball = (frame_idx, bx_m, by_m)

            ball_frame_log[int(frame_idx)] = {
                "x_m": round(bx_m, 4),
                "y_m": round(by_m, 4),
                "speed_mps": round(ball_speed, 4),
            }

    if H is not None and len(frame_corner_points) >= 4:
        # 1=top_left, 2=top_right, 3=bottom_right, 4=bottom_left
        labeled_points = [
            (1, top_left),
            (2, top_right),
            (3, bottom_right),
            (4, bottom_left),
        ]
        corner_path = [top_left, top_right, bottom_right, bottom_left, top_left]
        for start_point, end_point in zip(corner_path, corner_path[1:]):
            cv2.line(annotated_frame, start_point, end_point, BROWN, 2)

        for label, (px, py) in labeled_points:
            cv2.circle(annotated_frame, (px, py), 6, BROWN, -1)
            cv2.putText(
                annotated_frame,
                str(label),
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BROWN,
                2,
            )

    h_frame, w_frame = annotated_frame.shape[:2]
    y1 = margin
    y2 = y1 + court_h
    offset_x = 130
    x2 = w_frame - margin - offset_x
    x1 = x2 - court_w

    if y2 < h_frame and x2 < w_frame:
        overlay = annotated_frame[y1:y2, x1:x2].copy()
        cv2.addWeighted(base_court, alpha, overlay, 1 - alpha, 0,
                        annotated_frame[y1:y2, x1:x2])

    for (x, y) in court_pts.astype(int):
        cv2.circle(annotated_frame, (x1 + x, y1 + y), 3, (0, 0, 255), -1)

    base_court = court_image.copy()

    if out is None:
        h, w = annotated_frame.shape[:2]
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

    out.write(annotated_frame)

    if not args.no_display:
        cv2.imshow("Volleyball Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    frame_idx += 1

    # EPV INTEGRATION: quick-test mode, mirrors the `--limit 100` style
    # test mode requested for the EPV pipeline so the (expensive) full
    # perception pipeline can also be smoke-tested on a handful of frames.
    if args.limit is not None and frame_idx >= args.limit:
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

# EPV INTEGRATION: persist the per-frame player-state log for the EPV
# pipeline (epv/state_builder.py, epv/rally_segmentation.py, ...). This is
# the ONLY new artifact written by this file; nothing about detection,
# tracking, pose or homography is changed.
if frame_log is not None:
    with open(args.log_out, "w", encoding="utf-8") as f:
        json.dump(frame_log, f)
    print(f"[EPV] wrote {len(frame_log)} player-frame records to {args.log_out}")

if ball_frame_log is not None:
    with open(args.ball_log_out, "w", encoding="utf-8") as f:
        json.dump(ball_frame_log, f)
    print(f"[EPV] wrote {len(ball_frame_log)} ball-frame records to {args.ball_log_out}")
