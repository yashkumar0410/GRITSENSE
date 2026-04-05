import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np

player_model = YOLO("player_detection.pt")
pose_model = YOLO("pose.pt")
model = YOLO("court.pt")

cap = cv2.VideoCapture("sample.mp4")

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

frame_pts = None
H = None

prev_positions = {}
prev_frame_idx = {}
frame_idx = 0
kps = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

    results = player_model.track(
        frame,
        persist=True,
        conf=0.4,
        tracker="bytetrack.yaml",
        verbose=False,
        device=device,
        half=True,
    )

    pose_results = pose_model(frame, verbose=False)
    pose_boxes = []
    pose_keypoints = []
    if len(pose_results) > 0 and hasattr(pose_results[0], 'boxes') and hasattr(pose_results[0], 'keypoints'):
        pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy().astype(int)
        pose_keypoints = pose_results[0].keypoints.xy.cpu().numpy()

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

                best_iou = 0
                best_idx = -1
                for j, pbox in enumerate(pose_boxes):
                    xx1 = max(x1, pbox[0])
                    yy1 = max(y1, pbox[1])
                    xx2 = min(x2, pbox[2])
                    yy2 = min(y2, pbox[3])
                    inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    box_area = (x2 - x1) * (y2 - y1)
                    pbox_area = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                    union_area = box_area + pbox_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

                kps = None
                if best_idx != -1 and best_iou > 0.1:
                    kps = pose_keypoints[best_idx]

                    skeleton = [
                        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
                        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)
                    ]

                    for j1, j2 in skeleton:
                        xk1, yk1 = int(kps[j1][0]), int(kps[j1][1])
                        xk2, yk2 = int(kps[j2][0]), int(kps[j2][1])
                        cv2.line(annotated_frame, (xk1, yk1), (xk2, yk2), (0,255,255), 2)

                    for xk, yk in kps:
                        cv2.circle(annotated_frame, (int(xk), int(yk)), 3, (0,0,255), -1)

                    foot_x = int((kps[15][0] + kps[16][0]) / 2)
                    foot_y = int((kps[15][1] + kps[16][1]) / 2)
                    px, py = foot_x, foot_y
                else:
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

            cbx = (x1 + x2) // 2
            width = x2 - x1

            cv2.ellipse(annotated_frame, (cbx, y2),
                        (int(width), int(0.35 * width)),
                        0, -45, 235, (255, 0, 0), 3)

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

    if frame_pts is None:
        court_results = model.predict(frame, conf=0.25, verbose=False)
        for r in court_results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                top_left = (x1, y1)
                top_right = (x2, y1)
                bottom_right = (x2, y2)
                bottom_left = (x1, y2)

                frame_pts = np.array([top_left, top_right,
                                      bottom_right, bottom_left], dtype=np.float32)

                H, _ = cv2.findHomography(frame_pts, court_pts)

                break

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