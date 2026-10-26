"""
Player position and velocity smoothing in court metres.

Box bottoms jitter by several pixels a frame, which becomes tens of centimetres
on the court and makes single-frame velocities noisy. Velocity is taken as the
least-squares slope of each player's court positions over the last WINDOW
frames, and position as the mean of the last POSITION_FRAMES frames.
"""

from __future__ import annotations

import numpy as np

WINDOW = 6             # frames (0.2 s at 30 fps) for the velocity fit
POSITION_FRAMES = 3    # frames averaged for the position
MAX_GAP = 3            # a player unseen for longer starts a fresh history


class MotionSmoother:
    def __init__(self, fps: float = 30.0, max_speed: float | None = None):
        self.fps = fps
        self.max_speed = max_speed
        self.history: dict[int, list[tuple[int, float, float]]] = {}

    def update(self, pid: int, frame: int, x: float, y: float):
        """Return (x, y, vx, vy) smoothed, in metres and m/s."""
        track = self.history.setdefault(pid, [])
        if track and frame - track[-1][0] > MAX_GAP:
            track.clear()
        track.append((frame, x, y))
        del track[:-WINDOW]
        recent = track[-POSITION_FRAMES:]
        sx = float(np.mean([p[1] for p in recent])); sy = float(np.mean([p[2] for p in recent]))
        if len(track) < 3:
            return sx, sy, 0.0, 0.0
        t = np.array([p[0] for p in track], float) / self.fps
        vx = float(np.polyfit(t, [p[1] for p in track], 1)[0])
        vy = float(np.polyfit(t, [p[2] for p in track], 1)[0])
        speed = np.hypot(vx, vy)
        if self.max_speed and speed > self.max_speed:
            vx, vy = vx * self.max_speed / speed, vy * self.max_speed / speed
        return sx, sy, vx, vy
