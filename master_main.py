"""
GRITSENSE master runner.

This file orchestrates the existing project stages without editing main.py,
integrate_and_rebuild.py, visualize_recent_graphs.py, or any other source file.

Pipeline:
video -> player tracking + pose inference + ball tracking -> fresh JSON/CSV
      -> existing graph/GAT/HRN/temporal pipeline -> existing visualizations

Run from the project root:
    python master_main.py --video sample3.mp4

Required project files:
    player_detection.pt, pose.pt (or yolov8n-pose.pt), ball.pt,
    annotations.json, court.png, ball_tracker.py, integrate_and_rebuild.py,
    visualize_recent_graphs.py, graph_builder.py, sample_data.py, models/,
    temporal/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _iou(box_a, box_b) -> float:
    """Intersection-over-union for xyxy boxes."""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _match_pose_to_track(track_box, pose_boxes, used_pose_indices):
    """Match pose boxes using overlap, containment, and center proximity.

    Pose-model boxes can be much tighter around a visible skeleton than the
    player-detector boxes. IoU alone rejects valid poses in that situation.
    """
    tx1, ty1, tx2, ty2 = map(float, track_box)
    tw, th = max(1.0, tx2 - tx1), max(1.0, ty2 - ty1)
    tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
    track_diag = max(1.0, (tw * tw + th * th) ** 0.5)

    best_idx, best_score = None, -1.0
    for idx, pose_box in enumerate(pose_boxes):
        if idx in used_pose_indices:
            continue
        px1, py1, px2, py2 = map(float, pose_box)
        pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0

        ix1, iy1 = max(tx1, px1), max(ty1, py1)
        ix2, iy2 = min(tx2, px2), min(ty2, py2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        pose_area, track_area = pw * ph, tw * th
        iou = inter / max(1.0, track_area + pose_area - inter)
        pose_coverage = inter / max(1.0, pose_area)

        center_distance = ((pcx - tcx) ** 2 + (pcy - tcy) ** 2) ** 0.5
        proximity = max(0.0, 1.0 - center_distance / (0.65 * track_diag))
        center_inside_expanded = (
            tx1 - 0.08 * tw <= pcx <= tx2 + 0.08 * tw
            and ty1 - 0.08 * th <= pcy <= ty2 + 0.08 * th
        )
        if not (iou >= 0.02 or pose_coverage >= 0.20 or center_inside_expanded):
            continue

        score = 0.35 * iou + 0.40 * pose_coverage + 0.25 * proximity
        if center_inside_expanded:
            score += 0.10
        if score > best_score:
            best_idx, best_score = idx, score

    # Lower than the old IoU-only cutoff so tight skeleton boxes can match.
    return (best_idx if best_score >= 0.18 else None), max(0.0, best_score)


def _clear_generated_outputs():
    """Remove only generated output_recent artifacts to prevent stale-frame mixing."""
    out_root = ROOT / "output_recent"
    for folder_name in ("graphs", "dimensions", "temporal", "graph_images", "graph_features"):
        folder = out_root / folder_name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def run_video_detections(video_path: Path) -> None:
    """
    Generate fresh project-compatible player/pose JSON and ball CSV from video.
    Original project scripts and model files are not modified.
    """
    # Imported here so --dashboard-only runs without ultralytics installed.
    from ultralytics import YOLO

    player_weights = ROOT / "player_detection.pt"
    ball_weights = ROOT / "ball.pt"
    pose_weights = _first_existing(ROOT / "pose.pt", ROOT / "yolov8n-pose.pt")
    annotations_path = ROOT / "annotations.json"

    missing = [p.name for p in (player_weights, ball_weights, annotations_path) if not p.exists()]
    if pose_weights is None:
        missing.append("pose.pt or yolov8n-pose.pt")
    if missing:
        raise FileNotFoundError("Missing required project file(s): " + ", ".join(missing))

    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    with annotations_path.open("r", encoding="utf-8") as f:
        court_points_by_frame = json.load(f)

    device = 0 if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 76)
    print("STAGE 1 — FRESH VIDEO DETECTIONS")
    print("=" * 76)
    print(f"Video: {video_path}")
    print(f"Device: {device}")
    print(f"Player model: {player_weights.name}")
    print(f"Pose model: {pose_weights.name}")
    print(f"Ball model: {ball_weights.name}")

    player_model = YOLO(str(player_weights))
    pose_model = YOLO(str(pose_weights))
    ball_model = YOLO(str(ball_weights))

    # Reuse the project's existing ball tracker implementation.
    try:
        from ball_tracker import BallTracker
    except Exception as exc:
        raise RuntimeError(
            "Could not import BallTracker from the existing ball_tracker.py. "
            "Run this script from the GRITSENSE project directory."
        ) from exc

    ball_tracker = BallTracker(ball_model, device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Mini-court used by the original main.py homography visualization.
    court_image_path = ROOT / "court.png"
    court_image = cv2.imread(str(court_image_path))
    if court_image is None:
        cap.release()
        raise FileNotFoundError(f"Could not read mini-court image: {court_image_path}")
    court_w, court_h = 300, 150
    court_image = cv2.resize(court_image, (court_w, court_h))
    court_pts = np.array(
        [[0, 0], [court_w - 1, 0], [court_w - 1, court_h - 1], [0, court_h - 1]],
        dtype=np.float32,
    )
    actual_court_width_m = 18.0
    actual_court_height_m = 9.0
    mini_court_margin = 20

    preview_path = ROOT / "master_detection_preview.mp4"
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    player_data: dict[str, dict[str, dict]] = {}
    ball_rows = []
    previous_player_positions: dict[int, tuple[float, float, int]] = {}
    frame_idx = 0
    excluded_ids = {20, 63, 69}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ball_detection_frame = frame.copy()

            # Compute this frame's image-to-court homography BEFORE mapping players.
            # annotations.json stores [top-left, top-right, bottom-right, bottom-left].
            frame_corners = court_points_by_frame.get(str(frame_idx), [])
            H = None
            if len(frame_corners) >= 4:
                image_corners = np.asarray(frame_corners[:4], dtype=np.float32)
                if image_corners.shape == (4, 2):
                    H, _ = cv2.findHomography(image_corners, court_pts)

            mini_court = court_image.copy()

            # Same player tracking model family and tracker settings as main.py.
            track_results = player_model.track(
                frame,
                persist=True,
                conf=0.4,
                tracker="botsort.yaml",
                verbose=False,
                device=device,
                quantize=16 if device != "cpu" else 32,
            )
            track_boxes = []
            track_ids = []
            track_confidences = []
            for result in track_results:
                boxes = result.boxes
                if boxes is None or boxes.id is None:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                confidences = (
                    boxes.conf.cpu().numpy()
                    if boxes.conf is not None
                    else np.zeros(len(ids), dtype=float)
                )
                for box, track_id, confidence in zip(xyxy, ids, confidences):
                    if int(track_id) in excluded_ids:
                        continue
                    track_boxes.append(box.astype(float).tolist())
                    track_ids.append(int(track_id))
                    track_confidences.append(float(confidence))

            # Pose inference is run on the same frame, then associated to player tracks by IoU.
            pose_result = pose_model.predict(
                frame, conf=0.10, device=device, verbose=False
            )[0]
            pose_boxes = []
            pose_keypoints = []
            pose_confidences = []
            if pose_result.boxes is not None and len(pose_result.boxes) > 0:
                pose_boxes = pose_result.boxes.xyxy.cpu().numpy().astype(float).tolist()
            if pose_result.keypoints is not None and len(pose_result.keypoints) > 0:
                pose_keypoints = pose_result.keypoints.xy.cpu().numpy()
                if getattr(pose_result.keypoints, "conf", None) is not None:
                    pose_confidences = pose_result.keypoints.conf.cpu().numpy()
                else:
                    pose_confidences = [None] * len(pose_keypoints)

            frame_record = {}
            used_pose = set()
            for track_box, track_id, detection_confidence in zip(
                track_boxes, track_ids, track_confidences
            ):
                x1, y1, x2, y2 = map(float, track_box)
                image_x = (x1 + x2) / 2.0
                image_y = y2
                previous_position = previous_player_positions.get(track_id)
                if previous_position is None:
                    vx = vy = 0.0
                else:
                    previous_x, previous_y, previous_frame = previous_position
                    elapsed_frames = frame_idx - previous_frame
                    if elapsed_frames > 0:
                        vx = (image_x - previous_x) / elapsed_frames
                        vy = (image_y - previous_y) / elapsed_frames
                    else:
                        vx = vy = 0.0
                previous_player_positions[track_id] = (
                    image_x,
                    image_y,
                    frame_idx,
                )

                pose_idx, match_score = _match_pose_to_track(
                    track_box, pose_boxes, used_pose
                )
                keypoints = []
                pose_confidence = 0.0
                pose_available = False

                if pose_idx is not None and pose_idx < len(pose_keypoints):
                    used_pose.add(pose_idx)
                    raw_points = np.asarray(pose_keypoints[pose_idx], dtype=float)
                    if raw_points.ndim == 2 and raw_points.shape[0] >= 17 and raw_points.shape[1] >= 2:
                        keypoints = [
                            [float(point[0]), float(point[1])]
                            for point in raw_points[:17, :2]
                        ]
                        if pose_idx < len(pose_confidences) and pose_confidences[pose_idx] is not None:
                            conf_values = np.asarray(pose_confidences[pose_idx], dtype=float)[:17]
                            pose_confidence = float(np.mean(conf_values)) if conf_values.size else 0.0
                        else:
                            pose_confidence = float(match_score)
                        pose_available = len(keypoints) == 17

                # Map the player's foot point to the top-down court.
                court_x_m = court_y_m = None
                team_id = None
                team_label = None
                if H is not None:
                    foot_point = np.array([[[image_x, image_y]]], dtype=np.float32)
                    mapped = cv2.perspectiveTransform(foot_point, H)
                    court_px, court_py = map(float, mapped[0][0])
                    if np.isfinite(court_px) and np.isfinite(court_py):
                        # Team IDs are numeric because HRN aggregates teams 0 and 1.
                        draw_x = int(np.clip(round(court_px), 0, court_w - 1))
                        draw_y = int(np.clip(round(court_py), 0, court_h - 1))
                        team_id = 0 if draw_x < court_w // 2 else 1
                        team_label = "left" if team_id == 0 else "right"
                        team_color = (255, 0, 0) if team_id == 0 else (0, 0, 255)
                        cv2.circle(mini_court, (draw_x, draw_y), 7, team_color, -1)
                        cv2.circle(mini_court, (draw_x, draw_y), 7, (255, 255, 255), 1)
                        cv2.putText(
                            mini_court, str(track_id),
                            (max(0, draw_x - 8), max(12, draw_y - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2,
                        )
                        court_x_m = float(np.clip(court_px / (court_w - 1), 0.0, 1.0) * actual_court_width_m)
                        court_y_m = float(np.clip(court_py / (court_h - 1), 0.0, 1.0) * actual_court_height_m)

                # Fallback for frames without valid court homography: split by the
                # annotated court's image-space midpoint, or frame midpoint if needed.
                # Keep `team` numeric for sample_data.py / HRN; `team_label` is display-only.
                if team_id is None:
                    if len(frame_corners) >= 4:
                        midpoint_x = float(np.mean(np.asarray(frame_corners[:4], dtype=float)[:, 0]))
                    else:
                        midpoint_x = frame.shape[1] / 2.0
                    team_id = 0 if image_x < midpoint_x else 1
                    team_label = "left" if team_id == 0 else "right"

                # Preserve the exact per-player field layout consumed by sample_data.py.
                frame_record[str(track_id)] = {
                    "id": int(track_id),
                    "team": int(team_id),
                    "team_label": team_label,
                    "image_x": float(image_x),
                    "image_y": float(image_y),
                    "vx": float(vx),
                    "vy": float(vy),
                    "confidence": detection_confidence,
                    "bbox": [int(round(v)) for v in track_box],
                    "court_x": court_x_m,
                    "court_y": court_y_m,
                    "speed": 0.0,
                    "pose_available": bool(pose_available),
                    "pose_confidence": float(pose_confidence),
                    "keypoints": keypoints if pose_available else [],
                }

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 220, 0),
                    2,
                )
                cv2.putText(
                    frame, f"ID {track_id}",
                    (int(x1), max(20, int(y1) - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2,
                )
                if pose_available:
                    for px, py in keypoints:
                        cv2.circle(frame, (int(px), int(py)), 2, (255, 180, 0), -1)

            player_data[str(frame_idx)] = frame_record

            ball_data = ball_tracker.update(ball_detection_frame, frame_idx)
            if ball_data is None:
                ball_rows.append([frame_idx, 0, -1, -1, 0, 0.0, 0.0, 0.0])
            else:
                bx, by = int(round(ball_data["x"])), int(round(ball_data["y"]))
                radius = max(
                    1,
                    int(round(max(
                        ball_data["bbox"][2] - ball_data["bbox"][0],
                        ball_data["bbox"][3] - ball_data["bbox"][1],
                    ) / 2)),
                )
                ball_rows.append([
                    frame_idx,
                    1,
                    bx,
                    by,
                    radius,
                    float(ball_data["confidence"]),
                    float(ball_data["vx"]),
                    float(ball_data["vy"]),
                ])
                ball_tracker.draw(frame, ball_data)

            # Preserve the annotations-driven court outline on the source frame.
            if len(frame_corners) >= 4:
                points = np.asarray(frame_corners[:4], dtype=np.int32)
                cv2.polylines(frame, [points], isClosed=True, color=(0, 165, 255), thickness=2)
                for corner_idx, (cx, cy) in enumerate(points, start=1):
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 165, 255), -1)
                    cv2.putText(
                        frame, str(corner_idx), (int(cx) + 5, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2,
                    )

            # Composite the mapped mini-court into the top-right corner.
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, frame_w - mini_court_margin - court_w)
            y1 = mini_court_margin
            x2 = min(frame_w, x1 + court_w)
            y2 = min(frame_h, y1 + court_h)
            if x2 > x1 and y2 > y1:
                court_crop = mini_court[:y2 - y1, :x2 - x1]
                frame_crop = frame[y1:y2, x1:x2]
                cv2.addWeighted(court_crop, 0.88, frame_crop, 0.12, 0, frame_crop)
                cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)
                cv2.putText(
                    frame, "TOP-DOWN PLAYER POSITIONS",
                    (x1, min(frame_h - 5, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                )

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0 or (total_frames and frame_idx == total_frames):
                print(f"  processed {frame_idx}/{total_frames or '?'} frames")

    finally:
        cap.release()
        writer.release()

    player_output = ROOT / "player_keypoints_mf3_pose.json"
    with player_output.open("w", encoding="utf-8") as f:
        json.dump(player_data, f, indent=2)

    ball_output = ROOT / "ball.csv"
    with ball_output.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "Frame",
            "Visibility",
            "X",
            "Y",
            "Radius",
            "Confidence",
            "Velocity_X",
            "Velocity_Y",
        ])
        csv_writer.writerows(ball_rows)

    print(f"\nFresh player/pose data: {player_output.name} ({frame_idx} frames)")
    print(f"Fresh ball detections: {ball_output.name} ({len(ball_rows)} rows)")
    print(f"Detection preview: {preview_path.name}")


def run_existing_downstream(temporal_model="gru", temporal_checkpoint=None):
    """Run graph/HRN/sequence generation and optional trained temporal inference."""
    print("\n" + "=" * 76)
    print("STAGE 2 — EXISTING GRAPH → GAT → HRN → TEMPORAL PIPELINE")
    print("=" * 76)
    import integrate_and_rebuild
    integrate_and_rebuild.main(
        temporal_model=temporal_model,
        temporal_checkpoint=temporal_checkpoint,
    )

    print("\n" + "=" * 76)
    print("STAGE 3 — EXISTING GRAPH VISUALIZATIONS")
    print("=" * 76)
    import visualize_recent_graphs
    visualize_recent_graphs.main()


def run_pretrained_group_activity(video_path: Path) -> None:
    """Run the pretrained group-activity GAT from GAT/ next to the project's own GAT/HRN."""
    print("\n" + "=" * 76)
    print("STAGE 3b — PRETRAINED GROUP-ACTIVITY GAT (GAT/ RCRG-R2-C11-conc-temporal)")
    print("=" * 76)
    try:
        import group_activity
        group_activity.run(video_path, ROOT / "player_keypoints_mf3_pose.json")
    except Exception as exc:  # optional stage: never block the rest of the pipeline
        print(f"Pretrained group-activity GAT skipped: {type(exc).__name__}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GRITSENSE video-to-temporal pipeline.")
    parser.add_argument(
        "--video",
        default="sample3.mp4",
        help="Input video path (absolute or relative to the project directory).",
    )
    parser.add_argument(
        "--keep-existing-output",
        action="store_true",
        help="Do not clear old generated files under output_recent before rebuilding.",
    )
    parser.add_argument(
        "--temporal-model", choices=("gru", "transformer"), default="gru",
        help="Temporal classifier architecture to use when a trained checkpoint is supplied.",
    )
    parser.add_argument(
        "--temporal-checkpoint", default=None,
        help="Path to a TRAINED GRU/Transformer checkpoint; enables activity prediction.",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Render the tactical dashboard (point-chance map, recommended play, commentary) at the end.",
    )
    parser.add_argument(
        "--dashboard-only", action="store_true",
        help="Skip detection and downstream stages; render the dashboard from existing outputs.",
    )
    parser.add_argument(
        "--teams", nargs=2, default=["BRA", "JPN"], metavar=("LEFT", "RIGHT"),
        help="Dashboard team names for the left and right halves.",
    )
    return parser.parse_args()


def run_dashboard(video_path: Path, team_names):
    print("\n" + "=" * 76)
    print("STAGE 4 — TACTICAL DASHBOARD (MOCK GAT / HRN / EPV + NLP)")
    print("=" * 76)
    import tactical_dashboard
    return tactical_dashboard.render_dashboard(
        video_path=video_path,
        players_path=ROOT / "player_keypoints_mf3_pose.json",
        ball_path=ROOT / "ball.csv",
        annotations_path=ROOT / "annotations.json",
        out_dir=ROOT / "output_recent" / "dashboard",
        team_names={0: team_names[0], 1: team_names[1]},
    )


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (ROOT / video_path).resolve()

    if args.dashboard_only:
        run_dashboard(video_path, args.teams)
        return

    if not args.keep_existing_output:
        _clear_generated_outputs()

    run_video_detections(video_path)

    print("\n" + "=" * 76)
    print("STAGE 1b — BALL RE-DETECTION AND CLEANING")
    print("=" * 76)
    import redetect_ball
    redetect_ball.main(video_path)          # writes ball_candidates.json and ball_clean.csv
    run_existing_downstream(
        temporal_model=args.temporal_model,
        temporal_checkpoint=args.temporal_checkpoint,
    )
    run_pretrained_group_activity(video_path)
    if args.dashboard:
        run_dashboard(video_path, args.teams)

    print("\n" + "=" * 76)
    print("GRITSENSE MASTER PIPELINE COMPLETE")
    print("=" * 76)
    print(f"Input video: {video_path}")
    print(f"Outputs: {ROOT / 'output_recent'}")
    print("Team IDs: numeric 0/1 are written for HRN aggregation; labels are stored separately.")
    if args.temporal_checkpoint:
        print("Temporal activity predictions were requested using the supplied trained checkpoint.")
    else:
        print("Temporal sequences were generated; activity classification was skipped because no trained temporal checkpoint was supplied.")
        print("A labelled training dataset is required to train a meaningful GRU/Transformer checkpoint.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nMASTER PIPELINE FAILED: {type(exc).__name__}: {exc}")
        raise
