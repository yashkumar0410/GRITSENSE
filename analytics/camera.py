"""
Broadcast camera recovered from the court homography (annotations.json).

A homography between the court floor and the image fixes the camera up to its
focal length. With square pixels and the principal point at the image centre,
the first two rotation columns must be orthogonal and of equal length; one
focal length is fitted to every frame of the clip at once. The camera then
gives 3-D measurements from single images:

    project(frame, (x, y, h))          court metres + height -> pixel
    height_above(frame, (x, y), v)     height of the point at image row v that
                                       sits vertically above court point (x, y)

Checked on the clip: the camera centre stays within ~1 m over 319 panning
frames (as a fixed broadcast camera should), and the projected net and
antennas land on the real ones in the video.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COURT_CORNERS = np.array([[0, 0], [18, 0], [18, 9], [0, 9]], np.float32)


class Camera:
    def __init__(self, annotations: dict, width: int = 1920, height: int = 1080):
        self.annotations = annotations
        self.cx, self.cy = width / 2.0, height / 2.0
        self.focal = self._fit_focal()
        self.K = np.array([[self.focal, 0, self.cx], [0, self.focal, self.cy], [0, 0, 1.0]])
        self._poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    @lru_cache(maxsize=1)
    def from_project(cls) -> "Camera":
        with (ROOT / "annotations.json").open(encoding="utf-8") as f:
            return cls(json.load(f))

    def _homography(self, frame: int) -> np.ndarray | None:
        corners = self.annotations.get(str(frame))
        if not corners or len(corners) < 4:
            return None
        H, _ = cv2.findHomography(COURT_CORNERS, np.asarray(corners[:4], np.float32))
        return H

    def _fit_focal(self) -> float:
        centre = np.array([[1, 0, -self.cx], [0, 1, -self.cy], [0, 0, 1.0]])
        hs = [centre @ H for f in self.annotations if (H := self._homography(int(f))) is not None]

        def residual(focal):
            err = 0.0
            for G in hs:
                M = np.diag([1 / focal, 1 / focal, 1.0]) @ G
                a, b = M[:, 0], M[:, 1]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                err += (a @ b / (na * nb)) ** 2 + ((na - nb) / (na + nb)) ** 2
            return err

        grid = np.arange(800.0, 6000.0, 10.0)
        return float(grid[int(np.argmin([residual(f) for f in grid]))])

    def pose(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """(R, t) mapping court coordinates (x, y, z = -height) to camera coordinates."""
        if frame not in self._poses:
            H = self._homography(frame)
            if H is None:
                nearest = min((int(f) for f in self.annotations), key=lambda f: abs(f - frame))
                H = self._homography(nearest)
            M = np.linalg.inv(self.K) @ H
            lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
            if M[2, 2] * lam < 0:          # the court must be in front of the camera
                lam = -lam
            r1, r2, t = M[:, 0] * lam, M[:, 1] * lam, M[:, 2] * lam
            U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
            self._poses[frame] = (U @ Vt, t)
        return self._poses[frame]

    def centre(self, frame: int) -> np.ndarray:
        R, t = self.pose(frame)
        c = -R.T @ t
        return np.array([c[0], c[1], -c[2]])   # (x, y, height)

    def project(self, frame: int, point) -> np.ndarray:
        R, t = self.pose(frame)
        x, y, h = point
        p = self.K @ (R @ np.array([x, y, -h]) + t)
        return p[:2] / p[2]

    def ground_point(self, frame: int, pixel) -> tuple[float, float]:
        """Court point (x, y) seen at an image pixel, assuming it lies on the floor."""
        H = self._homography(frame)
        if H is None:
            H = self._homography(min((int(f) for f in self.annotations), key=lambda f: abs(f - frame)))
        p = np.linalg.inv(H) @ np.array([pixel[0], pixel[1], 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def height_above(self, frame: int, ground_xy, image_v: float) -> float:
        """Height (m) of the point on the vertical above ground_xy that appears at image row v."""
        v0 = self.project(frame, (ground_xy[0], ground_xy[1], 0.0))[1]
        v1 = self.project(frame, (ground_xy[0], ground_xy[1], 1.0))[1]
        per_metre = v0 - v1                      # image rows per metre of height there
        return float((v0 - image_v) / per_metre) if abs(per_metre) > 1e-6 else 0.0
