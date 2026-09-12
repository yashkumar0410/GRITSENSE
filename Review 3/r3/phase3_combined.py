#!/usr/bin/env python3

import argparse
import sys
import os
import shutil
import traceback


def resolve_output_path(output_dir, output_file):
    if not output_file:
        return os.path.join(output_dir, "output.mp4")

    if os.path.isabs(output_file) or os.path.dirname(output_file):
        output_path = output_file
    else:
        output_path = os.path.join(output_dir, output_file)

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    return output_path


def resolve_ball_csv_path(output_dir, video_path):
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.join(output_dir, video_basename)
    candidates = [
        os.path.join(video_dir, "ball.csv"),
        os.path.join(video_dir, "ball_1.csv"),
        os.path.join(video_dir, "ball_2.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return os.path.join(video_dir, "ball.csv")


def load_ball_positions(ball_csv_path):
    if not ball_csv_path or not os.path.exists(ball_csv_path):
        return {}

    ball_by_frame = {}
    try:
        with open(ball_csv_path, "r", encoding="utf-8") as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                try:
                    frame_id = int(row.get("Frame", -1))
                except Exception:
                    continue
                try:
                    visibility = int(row.get("Visibility", 0))
                    bx = int(row.get("X", -1))
                    by = int(row.get("Y", -1))
                    br = int(row.get("Radius", 0))
                except Exception:
                    visibility = 0
                    bx = -1
                    by = -1
                    br = 0
                if visibility and bx >= 0 and by >= 0:
                    ball_by_frame[frame_id] = {
                        "x": bx,
                        "y": by,
                        "radius": br if br > 0 else 8,
                    }
    except Exception:
        return {}
    return ball_by_frame


def viptest_main(args):
    try:
        import cv2
        import torch
        from ultralytics import YOLO
        import numpy as np
        import json
        import subprocess
        import time
        import csv
    except Exception as e:
        print(f"Error importing viptest dependencies: {e}")
        return 1

    annotations_path = args.annotations or "annotations.json"
    video_path = args.video_path or "sample3.mp4"
    player_model_path = args.player_model or "player_detection.pt"
    court_image_path = args.court_image or "court.png"
    output_filename = args.output_file or "output.mp4"
    output_dir = args.output_dir or "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = resolve_output_path(output_dir, output_filename)

    try:
        player_model = YOLO(player_model_path)
    except Exception as e:
        print(f"Error loading player model '{player_model_path}': {e}")
        return 1

    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            court_points_by_frame = json.load(f)
    except Exception:
        court_points_by_frame = {}

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    predict_video_path = os.path.join(output_dir, video_basename, "predict.mp4")
    csv_path = resolve_ball_csv_path(output_dir, video_path)
    if args.model_path:
        try:
            tracker_script = os.path.join(os.path.dirname(__file__), "src", "inference_onnx_seq_gray_v2.py")
            cmd = [
                sys.executable,
                tracker_script,
                "--video_path", video_path,
                "--model_path", args.model_path,
                "--output_dir", output_dir,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            csv_path = resolve_ball_csv_path(output_dir, video_path)
            print(f"Ball CSV ready: {csv_path}")
            if os.path.exists(predict_video_path):
                print(f"Ball detection video ready: {predict_video_path}")
        except Exception as e:
            print(f"Warning: could not generate ball detection video: {e}")

    ball_positions = load_ball_positions(csv_path)

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 30

    device = 0 if torch.cuda.is_available() else "cpu"
    display_enabled = bool(args.visualize)

    out = None

    court_image = cv2.imread(court_image_path)
    if court_image is None:
        print(f"Warning: could not load court image '{court_image_path}', continuing with blank court")
        court_image = np.zeros((150, 300, 3), dtype=np.uint8)

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()
        try:
            results = player_model.track(
                frame,
                persist=True,
                conf=0.4,
                tracker="botsort.yaml",
                verbose=False,
                device=device,
                half=True,
            )
        except Exception as e:
            print(f"Error running player model: {e}")
            traceback.print_exc()
            break

        for r in results:
            boxes = r.boxes

            if boxes.id is None or len(boxes.id) == 0:
                continue

            if H is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                left_players = []
                right_players = []
                player_points = []

                for i in range(len(ids)):
                    x1, y1, x2, y2 = xyxy[i]
                    player_id = ids[i]
                    if player_id in EXCLUDED_IDS:
                        continue

                    px = (x1 + x2) // 2
                    py = y2

                    point = np.array([[[px, py]]], dtype=np.float32)
                    mapped = cv2.perspectiveTransform(point, H)
                    mx, my = mapped[0][0]
                    mx, my = int(mx), int(my)
                    player_points.append((mx, my, i))

                for mx, my, idx in player_points:
                    if mx < court_w // 2:
                        left_players.append((mx, my, idx))
                    else:
                        right_players.append((mx, my, idx))

                if len(left_players) > 7:
                    extras = left_players[7:]
                    left_players = left_players[:7]
                    for mx, my, idx in extras:
                        mx_shifted = court_w - mx
                        right_players.append((mx_shifted, my, idx))

                if len(right_players) > 7:
                    extras = right_players[7:]
                    right_players = right_players[:7]
                    for mx, my, idx in extras:
                        mx_shifted = court_w - mx
                        left_players.append((mx_shifted, my, idx))

                for mx, my, idx in left_players:
                    overlay = base_court.copy()
                    cv2.circle(overlay, (mx, my), 12, (255, 0, 0, 80), -1)
                    cv2.addWeighted(overlay, 0.4, base_court, 0.6, 0, base_court)
                    cv2.circle(base_court, (mx, my), 5, (255, 0, 0), -1)

                    player_id = ids[idx] if 'ids' in locals() else idx+1
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

                for mx, my, idx in right_players:
                    overlay = base_court.copy()
                    cv2.circle(overlay, (mx, my), 12, (0, 0, 255, 80), -1)
                    cv2.addWeighted(overlay, 0.4, base_court, 0.6, 0, base_court)
                    cv2.circle(base_court, (mx, my), 5, (0, 0, 255), -1)

                    player_id = ids[idx] if 'ids' in locals() else idx+1
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

                for y in range(0, court_h, 12):
                    cv2.line(base_court, (court_w // 2, y),
                             (court_w // 2, min(y+6, court_h)), (0, 255, 255), 3)

            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            for i in range(len(ids)):
                x1, y1, x2, y2 = xyxy[i]
                track_id = ids[i]
                if track_id in EXCLUDED_IDS:
                    continue

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
                if track_id > 99:
                    x1_text -= 10

                speed = 0.0
                if 'player_speeds' in locals() and track_id in player_speeds:
                    speed = player_speeds[track_id]

                cv2.putText(annotated_frame, f"{track_id}",
                            (x1_text, y1_rect + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                cv2.putText(annotated_frame, f"{speed:.2f} m/s",
                            (x1_text, y1_rect + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        frame_corner_points = court_points_by_frame.get(str(frame_idx), [])
        if len(frame_corner_points) >= 4:
            top_left = tuple(map(int, frame_corner_points[0]))
            top_right = tuple(map(int, frame_corner_points[1]))
            bottom_right = tuple(map(int, frame_corner_points[2]))
            bottom_left = tuple(map(int, frame_corner_points[3]))

            frame_pts = np.array(
                [top_left, top_right, bottom_right, bottom_left],
                dtype=np.float32,
            )
            H, _ = cv2.findHomography(frame_pts, court_pts)

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

        # Overlay ball position from the generated tracker CSV if available
        try:
            ball = ball_positions.get(frame_idx)
            if ball is not None:
                bx = int(ball["x"])
                by = int(ball["y"])
                draw_r = int(ball["radius"])
                cv2.circle(annotated_frame, (bx, by), max(draw_r, 8), (0, 255, 0), 4)
                cv2.circle(annotated_frame, (bx, by), max(draw_r, 8) + 5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, "BALL", (bx + draw_r + 10, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception:
            pass

        if out is None:
            h, w = annotated_frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        out.write(annotated_frame)

        if display_enabled:
            cv2.imshow("Volleyball Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()
    if display_enabled:
        cv2.destroyAllWindows()

    # Cleanup tracker subprocess if started
    try:
        if tracker_proc is not None and tracker_proc.poll() is None:
            tracker_proc.terminate()
    except Exception:
        pass

    if args.model_path and os.path.exists(predict_video_path):
        try:
            if os.path.abspath(output_path) != os.path.abspath(predict_video_path):
                shutil.copyfile(predict_video_path, output_path)
                print(f"Copied ball detection video to final output: {output_path}")
        except Exception as e:
            print(f"Warning: could not copy ball detection video to final output: {e}")

    print(f"viptest finished, output: {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference (combined)")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "viptest"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str,
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable visualization on display using cv2")

    # viptest-specific options
    parser.add_argument("--annotations", type=str, help="Annotations JSON file for viptest")
    parser.add_argument("--player_model", type=str, help="YOLO player model for viptest")
    parser.add_argument("--court_image", type=str, help="Court image for viptest overlay")
    parser.add_argument("--output_file", type=str, help="Output video filename for viptest")

    args = parser.parse_args()

    if args.mode == "track":
        if not args.model_path:
            print("Error: --model_path is required for tracking mode")
            return 1

        try:
            from src.inference_onnx_seq_gray_v2 import main as track_main

            track_args = [
                "--video_path", args.video_path or "",
                "--model_path", args.model_path,
                "--output_dir", args.output_dir,
            ]
            if args.visualize:
                track_args.append("--visualize")
            track_main(track_args)
        except ImportError as e:
            print(f"Error importing tracking module: {e}")
            return 1
        except Exception as e:
            print(f"Error during ball tracking: {e}")
            return 1

    elif args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
        try:
            from src.pose_detector import add_pose_to_track_json
            add_pose_to_track_json(
                track_file=args.track_file,
                video_path=args.video_path,
                output_dir=args.output_dir,
                visualize=args.visualize
            )
        except ImportError as e:
            print(f"Error importing pose detection module: {e}")
            return 1
        except Exception as e:
            print(f"Error during pose detection: {e}")
            return 1

    elif args.mode == "analyze":
        print("Analysis mode selected")
        print("This mode is not yet implemented")

    elif args.mode == "viptest":
        return viptest_main(args)

    else:
        print("Available modes: track, pose, analyze, viptest")

    return 0


if __name__ == "__main__":
    sys.exit(main())
