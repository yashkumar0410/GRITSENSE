import cv2
import torch
from ultralytics import YOLO
import numpy as np
import json

player_model = YOLO("player_detection.pt")

with open("cp.json", "r", encoding="utf-8") as f:
    court_points_by_frame = json.load(f)

cap = cv2.VideoCapture("sample3.mp4")

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()
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

    cv2.imshow("Volleyball Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_idx += 1

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
