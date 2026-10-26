"""
Jump, reach and raised-arm detection from the tracked boxes and pose keypoints.

A jump is recognised when, within the last ~0.33 s, a player's feet AND head
rise together while the box keeps its size. Walking away from the camera also
moves the feet up the image, but it shrinks the box and doesn't lift the head
by the same amount, so it is not mistaken for a jump.

Heights are measured in metres with the camera recovered from the court
homography (analytics/camera.py): the take-off point is fixed on the floor, and
the current feet / box top are measured on the vertical line above it. While
the player is in the air, the take-off point is also their true court position
(the airborne feet would otherwise project too far from the camera).
"""

from __future__ import annotations

from dataclasses import dataclass

WINDOW_FRAMES = 10          # look-back for the take-off reference (~0.33 s at 30 fps)
MAX_SIZE_CHANGE = 0.10      # box height must stay within +-10% during a jump
ARMS_UP_MARGIN = 0.05       # wrist above the nose by 5% of box height
MIN_JUMP_M = 0.05           # below this, treat as standing (measurement noise)


@dataclass
class BodySignals:
    jump_m: float = 0.0                          # feet above the floor
    reach_m: float = 0.0                         # top of the box (head or raised hands) above the floor
    arms_up: bool = False
    ground_xy: tuple[float, float] | None = None # take-off point while airborne


class JumpDetector:
    """Stateful: call update() for every player, frame by frame in order."""

    def __init__(self, camera):
        self.camera = camera
        self.history: dict[int, dict[int, tuple[float, float, float, float]]] = {}

    def update(self, frame: int, record: dict) -> BodySignals:
        pid = int(record["id"])
        x1, y1, x2, y2 = record["bbox"]
        foot_u = (x1 + x2) / 2.0
        height_px = max(1.0, float(y2 - y1))
        track = self.history.setdefault(pid, {})
        track[frame] = (float(y2), float(y1), height_px, foot_u)
        for old in [f for f in track if f < frame - WINDOW_FRAMES]:
            del track[old]

        signals = BodySignals()
        past = [(f, track[f]) for f in range(frame - WINDOW_FRAMES, frame) if f in track]
        if len(past) >= WINDOW_FRAMES // 2:
            ref_frame, (foot0, top0, height0, u0) = max(past, key=lambda item: item[1][0])  # lowest feet
            foot_lift, head_lift = foot0 - y2, top0 - y1
            if foot_lift > 0 and head_lift > 0.5 * foot_lift and abs(height_px / height0 - 1.0) < MAX_SIZE_CHANGE:
                ground = self.camera.ground_point(ref_frame, (u0, foot0))
                jump = self.camera.height_above(frame, ground, y2)
                if jump > MIN_JUMP_M:
                    signals.jump_m = jump
                    signals.ground_xy = ground
                    signals.reach_m = self.camera.height_above(frame, ground, y1)
        if signals.ground_xy is None:
            ground = self.camera.ground_point(frame, (foot_u, y2))
            signals.reach_m = self.camera.height_above(frame, ground, y1)

        keypoints = record.get("keypoints") or []
        if record.get("pose_available") and len(keypoints) >= 11:
            nose_y = keypoints[0][1]
            wrist_y = min(keypoints[9][1], keypoints[10][1])
            signals.arms_up = (nose_y - wrist_y) > ARMS_UP_MARGIN * height_px
        return signals
