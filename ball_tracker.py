import cv2
import numpy as np
from collections import deque


def find_ball_by_color(frame):
    """Find an orange/red volleyball candidate using HSV segmentation.

    This helper is kept for the existing GritSense unit test and as an optional
    diagnostic. The production BallTracker uses ball.pt + temporal tracking.
    Returns (x, y, confidence) or None.
    """
    small = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([5, 60, 60], dtype=np.uint8)
    upper_orange = np.array([35, 255, 255], dtype=np.uint8)
    lower_red = np.array([0, 60, 60], dtype=np.uint8)
    upper_red = np.array([8, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_orange, upper_orange),
        cv2.inRange(hsv, lower_red, upper_red)
    )

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if 0.3 <= aspect <= 3.0:
            valid.append((area, contour))

    if not valid:
        return None

    area, contour = max(valid, key=lambda item: item[0])
    x, y, w, h = cv2.boundingRect(contour)
    cx = (x + w / 2.0) / 0.6
    cy = (y + h / 2.0) / 0.6
    area_ratio = area / max(frame.shape[0] * frame.shape[1], 1)
    confidence = min(0.99, max(0.45, area_ratio * 35.0))
    return float(cx), float(cy), float(confidence)


class BallTracker:
    """YOLO ball detector with temporal candidate selection.

    This keeps the GritSense ball.pt model, but uses the useful idea from
    the GitHub volleyball tracker: when several candidates are present,
    prefer a plausible candidate near the previous ball position.

    Missing detections are returned as None. The tracker never invents a
    ball position for the analytics JSON.
    """

    def __init__(self, model, device="cpu", confidence_threshold=0.15,
                 tracking_distance=300, history_length=10):
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.tracking_distance = tracking_distance
        self.history = deque(maxlen=history_length)
        self.previous_position = None
        self.previous_frame = None

    def get_candidates(self, frame):
        try:
            result = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )[0]
        except Exception as exc:
            print(f"Ball model error: {exc}")
            return []

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        candidates = []

        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = map(float, box)
            candidates.append({
                "x": (x1 + x2) / 2.0,
                "y": (y1 + y2) / 2.0,
                "confidence": float(confidence),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        return candidates

    def select_candidate(self, candidates):
        if not candidates:
            return None

        if self.previous_position is None:
            return max(candidates, key=lambda c: c["confidence"])

        px, py = self.previous_position
        scored = []

        for candidate in candidates:
            distance = float(np.hypot(candidate["x"] - px, candidate["y"] - py))
            candidate = candidate.copy()
            candidate["distance"] = distance
            scored.append(candidate)

        nearby = [c for c in scored if c["distance"] <= self.tracking_distance]

        if nearby:
            # Confidence is weighted slightly more than distance.
            return max(
                nearby,
                key=lambda c: (
                    0.6 * c["confidence"] +
                    0.4 * (1.0 - min(c["distance"] / self.tracking_distance, 1.0))
                )
            )

        # If the ball genuinely moved a long way, do not discard every
        # candidate; fall back to the most confident detection.
        return max(scored, key=lambda c: c["confidence"])

    def update(self, frame, frame_idx):
        candidates = self.get_candidates(frame)
        candidate = self.select_candidate(candidates)

        if candidate is None:
            return None

        x = float(candidate["x"])
        y = float(candidate["y"])

        vx = 0.0
        vy = 0.0
        if self.previous_position is not None and self.previous_frame is not None:
            dt = frame_idx - self.previous_frame
            if dt > 0:
                vx = (x - self.previous_position[0]) / dt
                vy = (y - self.previous_position[1]) / dt

        distance = float(candidate.get("distance", 0.0))

        result = {
            "x": x,
            "y": y,
            "vx": float(vx),
            "vy": float(vy),
            "confidence": float(candidate["confidence"]),
            "bbox": candidate["bbox"],
            "num_candidates": len(candidates),
            "tracking_distance": distance,
            "detected": True
        }

        self.previous_position = (x, y)
        self.previous_frame = frame_idx
        self.history.append((int(x), int(y)))

        return result

    def draw(self, frame, ball_data, color=(0, 0, 255)):
        if ball_data is None:
            return frame

        x = int(ball_data["x"])
        y = int(ball_data["y"])

        cv2.circle(frame, (x, y), 10, color, 2)
        cv2.circle(frame, (x, y), 3, color, -1)
        cv2.putText(
            frame,
            f"BALL {ball_data['confidence']:.2f}",
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        points = list(self.history)
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, 2)

        return frame
