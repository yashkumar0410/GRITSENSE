#!/usr/bin/env python3
"""Combine ball detections and player keypoints into one annotated video."""

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np


COCO_PARTS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a video with ball detection + keypoints overlay")
    parser.add_argument("--video_path", type=str, default="sample3.mp4", help="Input video path")
    parser.add_argument("--ball_csv", type=str, default=None, help="Ball CSV path; auto-detects if omitted")
    parser.add_argument("--keypoints_json", type=str, default="player_keypoints.json", help="Keypoints JSON file")
    parser.add_argument("--output", type=str, default="output/xoxo_output.mp4", help="Output video path")
    parser.add_argument("--show", action="store_true", help="Show preview while processing")
    return parser.parse_args()


def resolve_ball_csv(video_path, ball_csv=None):
    if ball_csv and os.path.exists(ball_csv):
        return ball_csv

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    candidates = [
        os.path.join("output", video_name, "ball.csv"),
        os.path.join("output", video_name, "ball_1.csv"),
        os.path.join("output", video_name, "ball_2.csv"),
        os.path.join(os.getcwd(), "ball.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_ball_positions(ball_csv_path):
    positions = {}
    if not ball_csv_path or not os.path.exists(ball_csv_path):
        return positions
    try:
        with open(ball_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame_id = int(float(row.get("Frame", -1)))
                    vis = int(row.get("Visibility", 0))
                    x = int(float(row.get("X", -1)))
                    y = int(float(row.get("Y", -1)))
                    r = int(float(row.get("Radius", 0)))
                except Exception:
                    continue
                if vis == 1 and x >= 0 and y >= 0:
                    positions[frame_id] = {"x": x, "y": y, "radius": max(r, 8)}
    except Exception:
        pass
    return positions


def flatten_keypoints(raw):
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        return [float(raw)]
    flattened = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten_keypoints(item))
        elif isinstance(item, (int, float)):
            flattened.append(float(item))
    return flattened


def extract_keypoints_for_frame(raw_data, frame_idx):
    if not raw_data:
        return []

    frame_map = raw_data.get(str(frame_idx))
    if isinstance(frame_map, dict):
        persons = list(frame_map.values())
    elif isinstance(raw_data, dict):
        persons = []
        for key, val in raw_data.items():
            if str(key) == str(frame_idx):
                if isinstance(val, dict):
                    persons = list(val.values())
                break
    else:
        persons = []

    extracted = []
    for person in persons:
        if not isinstance(person, dict):
            continue
        candidate = (
            person.get("keypoints")
            or person.get("pose")
            or person.get("kpts")
            or person.get("points")
        )
        if candidate is None:
            continue
        flat = flatten_keypoints(candidate)
        if not flat:
            continue
        if len(flat) >= 17 * 3:
            points = []
            for i in range(17):
                base = i * 3
                x = flat[base]
                y = flat[base + 1]
                score = flat[base + 2] if base + 2 < len(flat) else 1.0
                if np.isfinite(x) and np.isfinite(y):
                    points.append((float(x), float(y), float(score)))
                else:
                    points.append((None, None, 0.0))
            extracted.append(points)
        elif len(flat) >= 17 * 2:
            points = []
            for i in range(17):
                base = i * 2
                x = flat[base]
                y = flat[base + 1] if base + 1 < len(flat) else 0.0
                points.append((float(x), float(y), 1.0) if np.isfinite(x) and np.isfinite(y) else (None, None, 0.0))
            extracted.append(points)
    return extracted


def load_keypoints(keypoints_json):
    if not keypoints_json or not os.path.exists(keypoints_json):
        return {}
    try:
        with open(keypoints_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    frames = defaultdict(list)
    for frame_key, value in data.items():
        try:
            frame_idx = int(float(frame_key))
        except Exception:
            continue
        if isinstance(value, dict):
            for person_key, person in value.items():
                if not isinstance(person, dict):
                    continue
                candidate = person.get("keypoints") or person.get("pose") or person.get("kpts") or person.get("points")
                if candidate is None:
                    continue
                flat = flatten_keypoints(candidate)
                if not flat:
                    continue
                if len(flat) >= 17 * 3:
                    pts = []
                    for i in range(17):
                        base = i * 3
                        x = flat[base]
                        y = flat[base + 1]
                        score = flat[base + 2] if base + 2 < len(flat) else 1.0
                        if np.isfinite(x) and np.isfinite(y):
                            pts.append((float(x), float(y), float(score)))
                        else:
                            pts.append((None, None, 0.0))
                    frames[frame_idx].append(pts)
                elif len(flat) >= 17 * 2:
                    pts = []
                    for i in range(17):
                        base = i * 2
                        x = flat[base]
                        y = flat[base + 1] if base + 1 < len(flat) else 0.0
                        pts.append((float(x), float(y), 1.0) if np.isfinite(x) and np.isfinite(y) else (None, None, 0.0))
                    frames[frame_idx].append(pts)
    return dict(frames)


def draw_skeleton(frame, points, color=(0, 255, 255)):
    if not points:
        return
    for part_a, part_b in COCO_PARTS:
        a = points[part_a]
        b = points[part_b]
        if a[0] is None or a[1] is None or b[0] is None or b[1] is None:
            continue
        if a[2] < 0.1 or b[2] < 0.1:
            continue
        pt_a = (int(round(a[0])), int(round(a[1])))
        pt_b = (int(round(b[0])), int(round(b[1])))
        cv2.line(frame, pt_a, pt_b, color, 2)

    for idx, point in enumerate(points):
        if point[0] is None or point[1] is None:
            continue
        if point[2] < 0.1:
            continue
        pt = (int(round(point[0])), int(round(point[1])))
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)


def draw_ball(frame, ball_info):
    if not ball_info:
        return
    x = int(ball_info["x"])
    y = int(ball_info["y"])
    radius = int(ball_info["radius"]) if "radius" in ball_info else 8
    cv2.circle(frame, (x, y), max(radius, 8), (0, 255, 0), 4)
    cv2.circle(frame, (x, y), max(radius, 8) + 6, (255, 255, 255), 1)
    cv2.putText(frame, "BALL", (x + radius + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def main():
    args = parse_args()
    video_path = args.video_path
    output_path = args.output
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    ball_csv = resolve_ball_csv(video_path, args.ball_csv)
    if ball_csv:
        print(f"Using ball CSV: {ball_csv}")
    else:
        print("No ball CSV found. Continuing with ball overlay disabled.")

    keypoints_json = args.keypoints_json if os.path.exists(args.keypoints_json) else "player_keypoints.json"
    if os.path.exists(keypoints_json):
        keypoints_by_frame = load_keypoints(keypoints_json)
        print(f"Loaded keypoints JSON: {keypoints_json}")
    else:
        keypoints_by_frame = {}
        print(f"Keypoints file not found: {keypoints_json}")

    ball_positions = load_ball_positions(ball_csv)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in ball_positions:
            draw_ball(frame, ball_positions[frame_idx])

        for person_points in keypoints_by_frame.get(frame_idx, []):
            draw_skeleton(frame, person_points, color=(0, 255, 255))

        writer.write(frame)

        if args.show:
            cv2.imshow("XOXO combined", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Saved combined output video: {output_path}")


if __name__ == "__main__":
    main()
