"""
GRITSENSE tactical dashboard.

Renders a broadcast-style dashboard video from the existing pipeline outputs:

    sample3.mp4 + player_keypoints_mf3_pose.json + ball.csv + annotations.json
        -> homography (image <-> court metres)
        -> physics EPV, matchups, team shape combined with the trained GAT / HRN (analytics/insights.py)
        -> NLP commentary                   (analytics/commentary.py)
        -> output_recent/dashboard/gritsense_dashboard.mp4

The EPV heat map and the recommended target are drawn on the top-down court
AND projected back onto the broadcast frame with the inverse homography, so
the "where to play" zone sits on the real court surface.

Run from the project root (no detection re-run needed):
    python tactical_dashboard.py
    python tactical_dashboard.py --show            # live preview window
    python tactical_dashboard.py --teams BRA JPN --max-frames 150
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analytics import (
    COURT_LENGTH,
    COURT_WIDTH,
    NET_X,
    Ball,
    CommentaryGenerator,
    InsightEngine,
    Player,
    zone_rect,
)
from analytics.insights import ZONE_DESCRIPTIONS
from analytics.ball_track import detect_touches, load_clean
from analytics.camera import Camera
from analytics.epv import TOP_SPEED
from analytics.motion import MotionSmoother
from analytics.pose_signals import JumpDetector
from analytics.nlp import DEFAULT_LOCAL_MODEL, build_backend


ROOT = Path(__file__).resolve().parent
# Cleaned ball track (clean_ball_track.py / redetect_ball.py) when available.
BALL_FILE = ROOT / "ball_clean.csv" if (ROOT / "ball_clean.csv").exists() else ROOT / "ball.csv"

# ----------------------------------------------------------------------------
# Layout (1920 x 1080) and palette (RGB)
# ----------------------------------------------------------------------------

CANVAS_W, CANVAS_H = 1920, 1080
VIDEO_CARD = (24, 80, 1264, 788)
VIDEO_BOX = (36, 92, 1216, 684)  # x, y, w, h
COURT_CARD = (1288, 80, 1896, 456)
COURT_BOX = (1306, 124, 572, 320)
PLAY_CARD = (1288, 472, 1896, 788)
TEXT_CARD = (24, 804, 1264, 1056)
SIGNAL_CARD = (1288, 804, 1896, 1056)

BG = (10, 14, 22)
CARD = (18, 25, 37)
TILE = (26, 35, 51)
BORDER = (36, 48, 67)
TEXT = (232, 237, 245)
MUTED = (134, 148, 168)
DIM = (88, 101, 122)
ACCENT = (45, 226, 190)
AMBER = (251, 191, 36)
LIVE_RED = (244, 63, 94)
TEAM_RGB = {0: (250, 204, 21), 1: (239, 68, 68)}

# Top-down court canvas geometry
COURT_SCALE = 28.0  # px per metre
COURT_MARGIN = (34.0, 34.0)
# Heat map resolution used for the projection onto the broadcast frame
WARP_SCALE = 60.0

# Players more than this far outside the court lines are bench / officials.
COURT_SLACK_X = 2.0
COURT_SLACK_Y = 1.2
MAX_PER_TEAM = 6
BALL_HOLD_FRAMES = 12


def bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _font(size: int, weight: str = "regular"):
    names = {
        "regular": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
        "semibold": ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
        "bold": ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
        "mono": ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"],
    }[weight]
    for name in names:
        for candidate in (name, f"C:/Windows/Fonts/{name}"):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


FONTS = {
    "brand": _font(28, "bold"),
    "h_sub": _font(17),
    "label": _font(14, "semibold"),
    "move": _font(34, "bold"),
    "body": _font(17),
    "body_sb": _font(17, "semibold"),
    "metric": _font(30, "bold"),
    "small": _font(13),
    "small_sb": _font(13, "semibold"),
    "comment": _font(25, "semibold"),
    "history": _font(16),
    "mono": _font(15, "mono"),
    "pill": _font(15, "bold"),
    "tag": _font(12, "bold"),
    "clock": _font(20, "semibold"),
    "court_id": _font(11, "bold"),
}


# ----------------------------------------------------------------------------
# Homography helpers
# ----------------------------------------------------------------------------

COURT_CORNERS_M = np.array(
    [[0, 0], [COURT_LENGTH, 0], [COURT_LENGTH, COURT_WIDTH], [0, COURT_WIDTH]], dtype=np.float32
)


def scale_matrix(scale: float, offset=(0.0, 0.0)) -> np.ndarray:
    return np.array([[scale, 0, offset[0]], [0, scale, offset[1]], [0, 0, 1]], dtype=np.float64)


def homography_for_frame(annotations: dict, frame_idx: int, fallback):
    """Image -> court-metres homography from annotations ([TL, TR, BR, BL])."""
    corners = annotations.get(str(frame_idx))
    if corners and len(corners) >= 4:
        img = np.asarray(corners[:4], dtype=np.float32)
        H, _ = cv2.findHomography(img, COURT_CORNERS_M)
        if H is not None:
            return H
    return fallback


def project(H: np.ndarray, pts) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H).reshape(-1, 2)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------


def load_ball_rows(path: Path) -> dict[int, tuple[float, float]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["Visibility"]) == 1:
                rows[int(row["Frame"])] = (float(row["X"]), float(row["Y"]))
    return rows


def players_for_frame(record: dict, H: np.ndarray, fps: float = 30.0,
                      frame: int | None = None, jumps: JumpDetector | None = None,
                      motion: MotionSmoother | None = None) -> list[Player]:
    """Court-space players for one frame. Pass the frame index and a JumpDetector
    (called every frame, in order) to fill in jump height, reach and raised arms;
    airborne players are then placed at their take-off point. With a MotionSmoother,
    positions and velocities are smoothed over the last few frames."""
    candidates = []
    for p in record.values():
        body = jumps.update(frame, p) if jumps is not None else None
        ix, iy = float(p["image_x"]), float(p["image_y"])
        vx, vy = float(p.get("vx", 0.0)), float(p.get("vy", 0.0))
        (cx, cy), (hx, hy) = project(H, [[ix, iy], [ix + vx * 10.0, iy + vy * 10.0]])
        if body is not None and body.ground_xy is not None:
            # Airborne feet project too far from the camera; use the take-off point.
            shift_x, shift_y = body.ground_xy[0] - cx, body.ground_xy[1] - cy
            cx, cy, hx, hy = cx + shift_x, cy + shift_y, hx + shift_x, hy + shift_y
        if not (-COURT_SLACK_X <= cx <= COURT_LENGTH + COURT_SLACK_X and -COURT_SLACK_Y <= cy <= COURT_WIDTH + COURT_SLACK_Y):
            continue
        if motion is not None and frame is not None:
            cx, cy, mvx, mvy = motion.update(int(p["id"]), frame, cx, cy)
            speed = math.hypot(mvx, mvy)
            velocity = (mvx, mvy)
            heading = (mvx / speed, mvy / speed) if speed > 0.75 else None
        else:
            dx, dy = hx - cx, hy - cy
            norm = math.hypot(dx, dy)
            heading = (dx / norm, dy / norm) if norm > 0.25 else None
            # Image velocity is px/frame over a 10-frame lookahead -> court m/s, capped at the
            # measured top speed (faster values are tracking noise).
            speed = min(norm * fps / 10.0, TOP_SPEED)
            velocity = (dx / norm * speed, dy / norm * speed) if norm > 1e-6 else (0.0, 0.0)
        team = 0 if cx < NET_X else 1
        candidates.append(Player(int(p["id"]), team, float(cx), float(cy), ix, iy, heading,
                                 float(p.get("confidence", 0.0)), velocity,
                                 body.jump_m if body else 0.0, body.arms_up if body else False))

    players = []
    for team in (0, 1):
        members = sorted((c for c in candidates if c.team == team), key=lambda c: c.confidence, reverse=True)
        players.extend(members[:MAX_PER_TEAM])
    return players


class BallState:
    """Projects the ball into court space and holds it briefly through dropouts."""

    def __init__(self):
        self.last: Ball | None = None
        self.last_frame = -10_000

    def update(self, frame_idx: int, image_pt, H: np.ndarray) -> Ball | None:
        if image_pt is not None:
            bx, by = project(H, [image_pt])[0]
            # The ball is in the air, so its ground projection overshoots; use the net's
            # image position to decide the side and clip the court position for display.
            H_inv = np.linalg.inv(H)
            net_top, net_bottom = project(H_inv, [[NET_X, 0.0], [NET_X, COURT_WIDTH]])
            net_x_img = 0.5 * (net_top[0] + net_bottom[0])
            side = 0 if image_pt[0] < net_x_img else 1
            self.last = Ball(
                float(np.clip(bx, -2.0, COURT_LENGTH + 2.0)),
                float(np.clip(by, -0.8, COURT_WIDTH + 0.8)),
                float(image_pt[0]), float(image_pt[1]), side, True,
            )
            self.last_frame = frame_idx
            return self.last
        if self.last is not None and frame_idx - self.last_frame <= BALL_HOLD_FRAMES:
            self.last.visible = False
            return self.last
        return None


# ----------------------------------------------------------------------------
# EPV heat rendering
# ----------------------------------------------------------------------------

_HEAT_STOPS = np.array([0.0, 0.35, 0.7, 1.0])
_HEAT_COLORS = np.array([(30, 58, 138), (34, 211, 238), (253, 224, 71), (244, 63, 94)], dtype=np.float32)
HEAT_LUT = np.stack(
    [np.interp(np.linspace(0, 1, 256), _HEAT_STOPS, _HEAT_COLORS[:, c]) for c in (2, 1, 0)], axis=1
).astype(np.uint8)  # BGR


def render_heat(surface, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """BGR heat image + alpha for one 9 x 9 m half at `scale` px/m."""
    # Absolute scale: colour shows the actual point probability, not a relative ranking.
    norm = np.clip(surface.values / 0.8, 0.0, 1.0)
    size = int(round(9.0 * scale))
    up = np.clip(cv2.resize(norm.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    color = HEAT_LUT[(up * 255).astype(np.uint8)]
    alpha = 0.10 + 0.62 * up ** 1.6
    return color, alpha.astype(np.float32)


def blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> None:
    a = alpha[..., None]
    dst[:] = (dst.astype(np.float32) * (1 - a) + src.astype(np.float32) * a).astype(np.uint8)


# ----------------------------------------------------------------------------
# Broadcast frame overlay
# ----------------------------------------------------------------------------


def _bezier(p0, p1, lift: float, n: int = 40) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ctrl = np.array([(p0[0] + p1[0]) / 2.0, min(p0[1], p1[1]) - lift])
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def detect_content_crop(frame, threshold: float = 18.0):
    """16:9 crop of the picture area, excluding the broadcast's black bars."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray.mean(axis=0) > threshold)[0]
    rows = np.where(gray.mean(axis=1) > threshold)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, w, h
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
    cw, ch = x1 - x0, y1 - y0
    if cw / ch > 16 / 9:
        new_w = int(round(ch * 16 / 9))
        x0, cw = x0 + (cw - new_w) // 2, new_w
    else:
        new_h = int(round(cw * 9 / 16))
        y0, ch = y0 + (ch - new_h) // 2, new_h
    return x0, y0, cw, ch


def overlay_broadcast(frame, H, ins, players, ball, t):
    """Draw heat, players, ball, target zone and trajectory on the full-res frame."""
    h, w = frame.shape[:2]
    H_inv = np.linalg.inv(H)

    if ins.surface is not None:
        color, alpha = render_heat(ins.surface, WARP_SCALE)
        full_w, full_h = int(COURT_LENGTH * WARP_SCALE), int(COURT_WIDTH * WARP_SCALE)
        big_color = np.zeros((full_h, full_w, 3), np.uint8)
        big_alpha = np.zeros((full_h, full_w), np.float32)
        x_off = 0 if ins.surface.defend_side == 0 else int(NET_X * WARP_SCALE)
        big_color[:, x_off:x_off + color.shape[1]] = color
        big_alpha[:, x_off:x_off + alpha.shape[1]] = alpha
        to_img = H_inv @ np.linalg.inv(scale_matrix(WARP_SCALE))
        warped_c = cv2.warpPerspective(big_color, to_img, (w, h), flags=cv2.INTER_LINEAR)
        warped_a = cv2.warpPerspective(big_alpha * 0.85, to_img, (w, h), flags=cv2.INTER_LINEAR)
        blend(frame, warped_c, warped_a)

        # Target zone outline on the real court surface.
        x1, y1, x2, y2 = zone_rect(ins.target_zone, ins.defend_team)
        poly = project(H_inv, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).astype(np.int32)
        cv2.polylines(frame, [poly], True, bgr(TEXT), 2, cv2.LINE_AA)

    for p in players:
        color = bgr(TEAM_RGB[p.team])
        center = (int(p.image_x), int(p.image_y))
        if p.id == ins.key_player:
            cv2.ellipse(frame, center, (46, 16), 0, 0, 360, bgr(ACCENT), 4, cv2.LINE_AA)
        cv2.ellipse(frame, center, (32, 11), 0, 0, 360, color, 3, cv2.LINE_AA)

    target_img = None
    if ins.target is not None:
        tx, ty = ins.target
        pulse = 0.85 + 0.25 * math.sin(t / 4.0)
        ring = [[tx + pulse * math.cos(a), ty + pulse * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 48)]
        core = [[tx + 0.4 * math.cos(a), ty + 0.4 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 32)]
        ring_img = project(H_inv, ring).astype(np.int32)
        core_img = project(H_inv, core).astype(np.int32)
        fill = frame.copy()
        cv2.fillPoly(fill, [ring_img], bgr(ACCENT), cv2.LINE_AA)
        cv2.addWeighted(fill, 0.35, frame, 0.65, 0, frame)
        cv2.polylines(frame, [ring_img], True, bgr(ACCENT), 4, cv2.LINE_AA)
        cv2.fillPoly(frame, [core_img], bgr(TEXT), cv2.LINE_AA)
        target_img = project(H_inv, [[tx, ty]])[0]

    if ball is not None:
        bc = (int(ball.image_x), int(ball.image_y))
        cv2.circle(frame, bc, 22, bgr(AMBER), 2 if not ball.visible else 4, cv2.LINE_AA)
        # The shot path is advice for the moment of contact: draw it only while a
        # player is on the ball, not while the ball is in the air.
        if target_img is not None and ins.ball_near:
            arc = _bezier(bc, target_img, lift=180.0)
            for i in range(0, len(arc) - 1, 2):
                a, b = arc[i].astype(int), arc[i + 1].astype(int)
                cv2.line(frame, tuple(a), tuple(b), bgr(TEXT), 4, cv2.LINE_AA)
            tip, prev = arc[-1], arc[-4]
            cv2.arrowedLine(frame, tuple(prev.astype(int)), tuple(tip.astype(int)), bgr(TEXT), 4, cv2.LINE_AA, tipLength=0.9)
    return target_img


# ----------------------------------------------------------------------------
# Top-down tactical court
# ----------------------------------------------------------------------------


def court_px(x, y):
    return int(round(COURT_MARGIN[0] + x * COURT_SCALE)), int(round(COURT_MARGIN[1] + y * COURT_SCALE))


def draw_tactical(ins, players, ball, t) -> np.ndarray:
    cw, ch = COURT_BOX[2], COURT_BOX[3]
    img = np.full((ch, cw, 3), bgr((14, 44, 52)), np.uint8)
    tl, br = court_px(0, 0), court_px(COURT_LENGTH, COURT_WIDTH)
    cv2.rectangle(img, tl, br, bgr((24, 34, 50)), -1)

    if ins.surface is not None:
        color, alpha = render_heat(ins.surface, COURT_SCALE)
        x0 = 0.0 if ins.surface.defend_side == 0 else NET_X
        ox, oy = court_px(x0, 0)
        hh, ww = color.shape[:2]
        roi = img[oy:oy + hh, ox:ox + ww]
        blend(roi, color[: roi.shape[0], : roi.shape[1]], alpha[: roi.shape[0], : roi.shape[1]] * 0.9)

        x1, y1, x2, y2 = zone_rect(ins.target_zone, ins.defend_team)
        zone_fill = img.copy()
        cv2.rectangle(zone_fill, court_px(x1, y1), court_px(x2, y2), bgr(ACCENT), -1)
        cv2.addWeighted(zone_fill, 0.16, img, 0.84, 0, img)
        cv2.rectangle(img, court_px(x1, y1), court_px(x2, y2), bgr(ACCENT), 1, cv2.LINE_AA)

    line = bgr((226, 232, 240))
    cv2.rectangle(img, tl, br, line, 2, cv2.LINE_AA)
    for ax in (NET_X - 3.0, NET_X + 3.0):
        p0, p1 = court_px(ax, 0), court_px(ax, COURT_WIDTH)
        for yy in range(p0[1], p1[1], 10):
            cv2.line(img, (p0[0], yy), (p0[0], min(yy + 5, p1[1])), line, 1, cv2.LINE_AA)
    n0, n1 = court_px(NET_X, -0.5), court_px(NET_X, COURT_WIDTH + 0.5)
    cv2.line(img, n0, n1, bgr(TEXT), 4, cv2.LINE_AA)

    by_id = {p.id: p for p in players}
    for m in ins.matchups[:3]:
        a, b = by_id.get(m.attacker), by_id.get(m.defender)
        if a is None or b is None:
            continue
        thickness = max(1, int(round(1 + m.share * 9)))
        cv2.line(img, court_px(a.x, a.y), court_px(b.x, b.y), bgr(ACCENT), thickness, cv2.LINE_AA)

    # Shot trajectory from the attacker's contact point, with the net crossing marked
    # (only while a player is on the ball).
    if ins.target is not None and ins.surface is not None and ins.ball_near:
        cx, cy = ins.surface.contact
        tx, ty = ins.target
        cp, tp = np.array(court_px(cx, cy), float), np.array(court_px(tx, ty), float)
        length = float(np.linalg.norm(tp - cp))
        n_dash = max(2, int(length / 12))
        for i in range(0, n_dash, 2):
            a = cp + (tp - cp) * (i / n_dash)
            b = cp + (tp - cp) * (min(i + 1, n_dash) / n_dash)
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), bgr(TEXT), 2, cv2.LINE_AA)
        if abs(tx - cx) > 1e-6:
            s = (NET_X - cx) / (tx - cx)
            if 0.0 <= s <= 1.0:
                npx = court_px(NET_X, cy + s * (ty - cy))
                cv2.drawMarker(img, npx, bgr(AMBER), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)

    for p in players:
        c = court_px(p.x, p.y)
        if p.heading is not None:
            hx, hy = p.heading
            cv2.line(img, c, (int(c[0] + hx * 18), int(c[1] + hy * 18)), bgr(TEAM_RGB[p.team]), 2, cv2.LINE_AA)
        if p.id == ins.key_player:
            cv2.circle(img, c, 14, bgr(ACCENT), 2, cv2.LINE_AA)
        cv2.circle(img, c, 10, bgr(TEAM_RGB[p.team]), -1, cv2.LINE_AA)
        cv2.circle(img, c, 10, bgr((15, 20, 30)), 1, cv2.LINE_AA)

    if ball is not None:
        bp = court_px(ball.x, ball.y)
        cv2.circle(img, bp, 7, bgr(TEXT), -1, cv2.LINE_AA)
        cv2.circle(img, bp, 7, bgr(AMBER), 2, cv2.LINE_AA)

    # "Hit here" crosshair, drawn last so players never cover it.
    if ins.target is not None:
        tp = court_px(*ins.target)
        pulse = int(17 + 4 * math.sin(t / 4.0))
        cv2.circle(img, tp, pulse, bgr(ACCENT), 2, cv2.LINE_AA)
        cv2.circle(img, tp, 10, bgr(TEXT), 2, cv2.LINE_AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(img, (tp[0] + dx * 6, tp[1] + dy * 6), (tp[0] + dx * 15, tp[1] + dy * 15),
                     bgr(TEXT), 2, cv2.LINE_AA)
        cv2.circle(img, tp, 4, bgr(ACCENT), -1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------
# Static base + text layer
# ----------------------------------------------------------------------------


def _card(draw, box, title=None):
    draw.rounded_rectangle(box, radius=14, fill=CARD, outline=BORDER, width=1)
    if title:
        draw.text((box[0] + 18, box[1] + 16), title, font=FONTS["label"], fill=MUTED)


def build_base(team_names) -> np.ndarray:
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 18), "GRITSENSE", font=FONTS["brand"], fill=ACCENT)
    d.text((196, 27), "Live Tactical Intelligence  ·  Volleyball", font=FONTS["h_sub"], fill=MUTED)

    cx = CANVAS_W // 2
    for name, team, anchor_x in ((team_names[0], 0, cx - 40), (team_names[1], 1, cx + 40)):
        tw = d.textlength(name, font=FONTS["pill"])
        x0 = anchor_x - tw - 28 if team == 0 else anchor_x
        d.rounded_rectangle((x0, 20, x0 + tw + 28, 50), radius=15, fill=TILE, outline=TEAM_RGB[team], width=2)
        d.text((x0 + 14, 35), name, font=FONTS["pill"], fill=TEXT, anchor="lm")
    d.text((cx, 35), "vs", font=FONTS["body"], fill=MUTED, anchor="mm")

    d.rounded_rectangle((1896 - 74, 22, 1896, 48), radius=13, fill=LIVE_RED)
    d.text((1896 - 37, 35), "LIVE", font=FONTS["tag"], fill=(255, 255, 255), anchor="mm")

    _card(d, VIDEO_CARD)
    _card(d, COURT_CARD, "COURT MAP")
    _card(d, PLAY_CARD, "RECOMMENDED PLAY")
    _card(d, TEXT_CARD)
    _card(d, SIGNAL_CARD)
    d.text((SIGNAL_CARD[0] + 18, SIGNAL_CARD[1] + 16), "KEY MATCHUPS", font=FONTS["label"], fill=MUTED)
    d.text((SIGNAL_CARD[0] + 318, SIGNAL_CARD[1] + 16), "TEAM SHAPE", font=FONTS["label"], fill=MUTED)
    return np.array(img)[:, :, ::-1].copy()


def _wrap(text, font, width, draw):
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _bar(d, x, y, w, h, frac, color):
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=TILE)
    fw = max(h, int(w * float(np.clip(frac, 0, 1))))
    d.rounded_rectangle((x, y, x + fw, y + h), radius=h // 2, fill=color)


# ----------------------------------------------------------------------------
# Pretrained group-activity GAT (group_activity.py, GAT/ checkpoint)
# ----------------------------------------------------------------------------

GROUP_ACTIVITY_FILE = ROOT / "output_recent" / "group_activity" / "group_activity.json"
ACTIVITY_RGB = {"pass": (96, 165, 250), "set": ACCENT, "spike": AMBER, "winpoint": LIVE_RED}


def split_activity(label: str) -> tuple[int, str]:
    """'l-spike' / 'r_winpoint' -> (team 0 = left / 1 = right, 'spike')."""
    return (0 if label[0] == "l" else 1), label[2:]


def load_group_activity(video_path: Path, players_path: Path, frame_count: int):
    """Window predictions for this video: reuse master_main's output, else run the model."""
    try:
        data = json.loads(GROUP_ACTIVITY_FILE.read_text(encoding="utf-8"))
        fresh = (Path(data["video"]).resolve() == video_path.resolve()
                 and abs(data["frames"] - frame_count) <= 2)
    except (OSError, ValueError, KeyError):
        fresh = False
    if not fresh:
        try:
            import group_activity
            print("Running the pretrained group-activity GAT for this video")
            data = group_activity.run(video_path, players_path, GROUP_ACTIVITY_FILE)
        except Exception as exc:
            print(f"Pretrained group-activity GAT unavailable ({type(exc).__name__}: {exc})")
            return None
    lookup = [None] * data["frames"]
    for i, w in enumerate(data["windows"]):
        for f in range(w["start_frame"], min(w["end_frame"] + 1, data["frames"])):
            lookup[f] = i
    print(f"Pretrained group-activity GAT: {len(data['windows'])} windows")
    return {**data, "lookup": lookup}


def group_activity_at(ga, frame_idx: int):
    if ga is None or frame_idx >= len(ga["lookup"]) or ga["lookup"][frame_idx] is None:
        return None
    return ga["windows"][ga["lookup"][frame_idx]]


def _dim(rgb, k=0.38):
    return tuple(int(c * k + BG[i] * (1 - k)) for i, c in enumerate(rgb))


def draw_group_activity(d, ga, frame_idx, team_names):
    """Live action card (top-right of the feed) and the rally timeline (bottom of the feed)."""
    window = group_activity_at(ga, frame_idx)
    vx, vy, vw, vh = VIDEO_BOX
    if window is None:
        return

    # Rally timeline: one segment per 9-frame window, coloured by action, team edge on top.
    tx0, tx1 = vx + 14, vx + vw - 14
    ty0, ty1 = vy + vh - 44, vy + vh - 14
    d.rounded_rectangle((tx0, ty0, tx1, ty1), radius=10, fill=(8, 12, 20), outline=BORDER, width=1)
    d.text((tx0 + 12, (ty0 + ty1) / 2), "RALLY", font=FONTS["tag"], fill=MUTED, anchor="lm")
    sx0, sx1 = tx0 + 64, tx1 - 10
    n = max(ga["frames"] - 1, 1)
    fx = lambda f: sx0 + (sx1 - sx0) * f / n
    cur = ga["lookup"][frame_idx]
    for i, w in enumerate(ga["windows"]):
        team, action = split_activity(w["label"])
        color = ACTIVITY_RGB.get(action, MUTED)
        a, b = fx(w["start_frame"]), max(fx(w["end_frame"]) - 1, fx(w["start_frame"]) + 2)
        d.rectangle((a, ty0 + 9, b, ty1 - 7), fill=color if i == cur else _dim(color))
        d.rectangle((a, ty0 + 6, b, ty0 + 8), fill=TEAM_RGB[team] if i == cur else _dim(TEAM_RGB[team], 0.6))
    px = fx(frame_idx)
    d.rectangle((px - 1, ty0 + 3, px + 1, ty1 - 3), fill=TEXT)

    # Live action card.
    team, action = split_activity(window["label"])
    color = ACTIVITY_RGB.get(action, TEXT)
    cw, ch = 300, 96
    cx1, cy0 = vx + vw - 14, vy + 14
    cx0 = cx1 - cw
    d.rounded_rectangle((cx0, cy0, cx1, cy0 + ch), radius=12, fill=(8, 12, 20), outline=color, width=2)
    d.text((cx0 + 14, cy0 + 12), "GROUP ACTIVITY", font=FONTS["tag"], fill=MUTED)
    d.text((cx1 - 14, cy0 + 12), "PRETRAINED GAT", font=FONTS["tag"], fill=DIM, anchor="ra")
    d.text((cx0 + 14, cy0 + 28), action.upper(), font=FONTS["metric"], fill=color)
    name = team_names[team]
    nw = d.textlength(name, font=FONTS["tag"]) + 22
    d.rounded_rectangle((cx1 - 14 - nw, cy0 + 36, cx1 - 14, cy0 + 60), radius=12, fill=TILE,
                        outline=TEAM_RGB[team], width=2)
    d.text((cx1 - 14 - nw / 2, cy0 + 48), name, font=FONTS["tag"], fill=TEXT, anchor="mm")
    _bar(d, cx0 + 14, cy0 + 76, cw - 80, 6, window["score"], color)
    d.text((cx1 - 14, cy0 + 79), f"{window['score']:.0%}", font=FONTS["small_sb"], fill=TEXT, anchor="rm")


def draw_text_layer(canvas_bgr, ins, commentary, epv_history, target_img, frame_idx, fps, team_names,
                    crop, players, ga=None):
    img = Image.fromarray(canvas_bgr[:, :, ::-1])
    d = ImageDraw.Draw(img)
    draw_group_activity(d, ga, frame_idx, team_names)

    # Header timecode
    secs = frame_idx / fps
    d.text((1896 - 90, 35), f"{int(secs // 60):02d}:{secs % 60:04.1f}", font=FONTS["clock"], fill=TEXT, anchor="rm")

    # --- Video overlay labels -------------------------------------------------
    fx, fy, fw, _ = crop
    sx = VIDEO_BOX[2] / fw
    if target_img is not None and ins.target_zone is not None:
        px = VIDEO_BOX[0] + (target_img[0] - fx) * sx
        py = VIDEO_BOX[1] + (target_img[1] - fy) * sx + 26
        label = f"PLAY HERE  ·  ZONE {ins.target_zone}"
        lw = d.textlength(label, font=FONTS["pill"])
        x0 = float(np.clip(px - lw / 2 - 14, VIDEO_BOX[0] + 8, VIDEO_BOX[0] + VIDEO_BOX[2] - lw - 36))
        y0 = float(np.clip(py, VIDEO_BOX[1] + 8, VIDEO_BOX[1] + VIDEO_BOX[3] - 40))
        d.rounded_rectangle((x0, y0, x0 + lw + 28, y0 + 30), radius=15, fill=(8, 12, 20), outline=ACCENT, width=2)
        d.text((x0 + 14, y0 + 15), label, font=FONTS["pill"], fill=ACCENT, anchor="lm")

    vx0, vy0 = VIDEO_BOX[0] + 14, VIDEO_BOX[1] + 14
    d.rounded_rectangle((vx0, vy0, vx0 + 112, vy0 + 28), radius=14, fill=(8, 12, 20))
    d.ellipse((vx0 + 12, vy0 + 9, vx0 + 22, vy0 + 19), fill=LIVE_RED)
    d.text((vx0 + 32, vy0 + 14), "LIVE FEED", font=FONTS["tag"], fill=TEXT, anchor="lm")
    if ins.attack_team is not None:
        state = f"#{ins.toucher}  ·  {ins.phase.upper()}" if ins.ball_near else "BALL IN FLIGHT"
        sw = d.textlength(state, font=FONTS["tag"]) + 24
        sx0 = vx0 + 122
        d.rounded_rectangle((sx0, vy0, sx0 + sw, vy0 + 28), radius=14, fill=(8, 12, 20),
                            outline=ACCENT if ins.ball_near else BORDER, width=2)
        d.text((sx0 + sw / 2, vy0 + 14), state, font=FONTS["tag"], fill=ACCENT if ins.ball_near else MUTED,
               anchor="mm")

    # --- Court card legend ------------------------------------------------------
    cx0, cy0 = COURT_CARD[0], COURT_CARD[1]
    lx = COURT_CARD[2] - 18
    for name, team in ((team_names[1], 1), (team_names[0], 0)):
        tw = d.textlength(name, font=FONTS["small_sb"])
        lx -= tw
        d.text((lx, cy0 + 24), name, font=FONTS["small_sb"], fill=TEXT, anchor="lm")
        lx -= 16
        d.ellipse((lx, cy0 + 19, lx + 10, cy0 + 29), fill=TEAM_RGB[team])
        lx -= 14

    for p in players:
        px_, py_ = court_px(p.x, p.y)
        d.text((COURT_BOX[0] + px_, COURT_BOX[1] + py_), str(p.id), font=FONTS["court_id"],
               fill=(15, 20, 30), anchor="mm")

    # "HIT HERE" label next to the crosshair on the top-down court.
    if ins.target is not None:
        tpx, tpy = court_px(*ins.target)
        gx, gy = COURT_BOX[0] + tpx, COURT_BOX[1] + tpy
        label = f"HIT HERE  ·  {ins.move.split(' · ')[0]}"
        lw = d.textlength(label, font=FONTS["tag"]) + 20
        lx = float(np.clip(gx - lw / 2, COURT_BOX[0] + 4, COURT_BOX[0] + COURT_BOX[2] - lw - 4))
        ly = gy - 50 if gy - 50 > COURT_BOX[1] + 4 else gy + 26
        d.rounded_rectangle((lx, ly, lx + lw, ly + 22), radius=11, fill=(8, 12, 20), outline=ACCENT, width=2)
        d.text((lx + lw / 2, ly + 11), label, font=FONTS["tag"], fill=ACCENT, anchor="mm")

    # --- Recommended play card ----------------------------------------------------
    x0, y0 = PLAY_CARD[0], PLAY_CARD[1]
    if ins.attack_team is not None:
        chip = f"{team_names[ins.attack_team]} ATTACKING"
        cw = d.textlength(chip, font=FONTS["tag"])
        d.rounded_rectangle((PLAY_CARD[2] - cw - 42, y0 + 12, PLAY_CARD[2] - 18, y0 + 36), radius=12,
                            fill=TILE, outline=TEAM_RGB[ins.attack_team], width=2)
        d.text((PLAY_CARD[2] - cw - 30, y0 + 24), chip, font=FONTS["tag"], fill=TEXT, anchor="lm")

    d.text((x0 + 18, y0 + 44), ins.move, font=FONTS["move"], fill=TEXT if ins.attack_chance else MUTED)
    who = f"#{ins.key_player}  ·  " if ins.key_player is not None and ins.attack_chance else ""
    d.text((x0 + 18, y0 + 92), f"{who}{ins.move_detail}", font=FONTS["body"], fill=MUTED)
    if ins.target is not None:
        zdesc = ZONE_DESCRIPTIONS.get(ins.target_zone, "")
        d.text((x0 + 18, y0 + 120), f"→  Zone {ins.target_zone}  ·  {zdesc}", font=FONTS["body_sb"], fill=ACCENT)
        if ins.hrn_prediction:
            h = ins.hrn_prediction
            d.text((PLAY_CARD[2] - 18, y0 + 122),
                   f"HRN  ·  Zone {h['zone']} {h['zone_conf']:.0%}  ·  {h['shot'].split(' ')[0].title()}",
                   font=FONTS["small_sb"], fill=MUTED, anchor="ra")

    if ins.attack_chance:
        tiles = [
            ("POINT CHANCE", f"{ins.epv:.0%}", f"{(ins.epv - ins.epv_mean) * 100:+.0f} pts vs court avg", None),
            ("BLOCK CHANCE", f"{ins.block_prob:.0%}", None, (ins.block_prob, AMBER)),
            ("OUT CHANCE", f"{ins.risk:.0%}", None, (ins.risk, AMBER)),
        ]
    else:   # no hit over the net coming yet: nothing to score
        tiles = [("POINT CHANCE", "—", None, None), ("BLOCK CHANCE", "—", None, None), ("OUT CHANCE", "—", None, None)]
    for i, (label, value, sub, bar) in enumerate(tiles):
        tx, ty = x0 + 18 + i * 194, y0 + 156
        d.rounded_rectangle((tx, ty, tx + 182, ty + 82), radius=10, fill=TILE)
        d.text((tx + 14, ty + 10), label, font=FONTS["tag"], fill=MUTED)
        d.text((tx + 14, ty + 26), value, font=FONTS["metric"], fill=TEXT)
        if sub:
            d.text((tx + 14, ty + 64), sub, font=FONTS["small"], fill=ACCENT if ins.epv >= ins.epv_mean else AMBER)
        if bar:
            _bar(d, tx + 14, ty + 66, 154, 6, bar[0], bar[1])

    # EPV sparkline
    sx0, sy0, sw, sh = x0 + 18, y0 + 258, 572, 40
    d.text((sx0, sy0 - 8), "POINT CHANCE  ·  LAST 4 S", font=FONTS["tag"], fill=DIM)
    if len(epv_history) > 1:
        vals = np.array(epv_history)
        lo, hi = 0.0, 1.0
        pts = [
            (sx0 + sw * i / (len(vals) - 1), sy0 + 8 + sh - sh * (v - lo) / (hi - lo))
            for i, v in enumerate(vals)
        ]
        d.line([(sx0, sy0 + 8 + sh), (sx0 + sw, sy0 + 8 + sh)], fill=BORDER, width=1)
        d.line(pts, fill=ACCENT, width=3, joint="curve")
        ex, ey = pts[-1]
        d.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), fill=TEXT)

    # --- Commentary card ------------------------------------------------------------
    kx, ky = TEXT_CARD[0] + 18, TEXT_CARD[1] + 16
    pulse = 4 + 2 * (0.5 + 0.5 * math.sin(frame_idx / 5.0))
    d.ellipse((kx + 5 - pulse, ky + 8 - pulse, kx + 5 + pulse, ky + 8 + pulse), fill=LIVE_RED)
    d.text((kx + 18, ky), "LIVE COMMENTARY", font=FONTS["label"], fill=MUTED)

    history = commentary.history
    if history:
        current = history[-1]
        reveal = int((frame_idx - current.frame) * 2.6) + 1
        shown = current.text[:reveal]
        lines = _wrap(shown, FONTS["comment"], TEXT_CARD[2] - TEXT_CARD[0] - 120, d)
        d.text((kx, ky + 38), current.timestamp, font=FONTS["mono"], fill=ACCENT)
        for i, ln in enumerate(lines[:3]):
            d.text((kx + 76, ky + 32 + i * 34), ln, font=FONTS["comment"], fill=TEXT)
        if reveal < len(current.text) and lines and (frame_idx // 4) % 2 == 0:
            last = lines[min(len(lines), 3) - 1]
            cx = kx + 76 + d.textlength(last, font=FONTS["comment"]) + 3
            cy = ky + 32 + (min(len(lines), 3) - 1) * 34
            d.rectangle((cx, cy + 6, cx + 3, cy + 30), fill=ACCENT)

        if len(history) > 1:
            d.line([(kx, TEXT_CARD[3] - 74), (TEXT_CARD[2] - 18, TEXT_CARD[3] - 74)], fill=BORDER, width=1)
        for j, old in enumerate(reversed(history[-3:-1])):
            oy = TEXT_CARD[3] - 36 - j * 28
            d.text((kx, oy), old.timestamp, font=FONTS["mono"], fill=DIM)
            txt = old.text
            while d.textlength(txt, font=FONTS["history"]) > TEXT_CARD[2] - TEXT_CARD[0] - 130 and len(txt) > 10:
                txt = txt[:-2]
            if txt != old.text:
                txt = txt.rstrip() + "…"
            d.text((kx + 76, oy - 1), txt, font=FONTS["history"], fill=MUTED)

    # --- Matchups / team shape card -------------------------------------------------
    gx, gy = SIGNAL_CARD[0] + 18, SIGNAL_CARD[1] + 50
    matchup_label = "PHYSICS + GAT" if ins.matchup_source == "physics+GAT" else "THREAT STOPPED"
    d.text((gx + 272, SIGNAL_CARD[1] + 18), matchup_label, font=FONTS["tag"], fill=DIM, anchor="ra")
    if ins.shape_source == "physics+HRN":
        d.text((SIGNAL_CARD[2] - 18, SIGNAL_CARD[1] + 18), "PHYSICS + HRN", font=FONTS["tag"], fill=DIM,
               anchor="ra")
    defend_color = TEAM_RGB.get(ins.defend_team, MUTED)
    for i, m in enumerate(ins.matchups[:4]):
        ry = gy + i * 46
        d.text((gx, ry), f"#{m.attacker}", font=FONTS["body_sb"], fill=ACCENT)
        d.text((gx + 46, ry), "vs", font=FONTS["body"], fill=DIM)
        d.text((gx + 72, ry), f"#{m.defender}", font=FONTS["body_sb"], fill=defend_color)
        d.text((gx + 272, ry), f"{m.share:.0%}", font=FONTS["mono"], fill=TEXT, anchor="ra")
        _bar(d, gx, ry + 26, 272, 6, m.share / max(1e-6, ins.matchups[0].share), defend_color)
    if not ins.matchups:
        d.text((gx, gy), "Waiting for possession", font=FONTS["body"], fill=DIM)

    hx, hy = SIGNAL_CARD[0] + 318, SIGNAL_CARD[1] + 50
    for k, team in enumerate((0, 1)):
        state = ins.team_shape.get(team)
        if state is None:
            continue
        by = hy + k * 98
        d.ellipse((hx, by + 4, hx + 10, by + 14), fill=TEAM_RGB[team])
        d.text((hx + 18, by), team_names[team], font=FONTS["body_sb"], fill=TEXT)
        for m, (label, val) in enumerate((("Court coverage", state.coverage), ("Balance", state.balance))):
            my = by + 28 + m * 30
            d.text((hx, my), label, font=FONTS["small"], fill=MUTED)
            d.text((hx + 272, my), f"{val:.0%}", font=FONTS["small_sb"], fill=TEXT, anchor="ra")
            _bar(d, hx, my + 18, 272, 5, val, TEAM_RGB[team])

    return np.array(img)[:, :, ::-1].copy()


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def display_view(ins):
    """What the dashboard draws. "Where to hit" (point-chance map, target, zone, shot path)
    is shown only when a hit over the net is coming: a serve, a set to a hitter, or the
    attack itself. During passes / digs and while the ball crosses the net, the card shows
    how the attack is being built instead."""
    if ins.attack_chance or ins.attack_team is None:
        return ins
    if ins.phase == "pass":
        move = f"PASS  ·  SET TO #{ins.key_player}" if ins.key_player is not None else "PASS"
        detail = f"#{ins.passer} passing  ·  attack comes after the set" if ins.passer else "building the attack"
    else:
        move, detail = "BALL IN FLIGHT", "waiting for the next touch"
    return replace(ins, surface=None, target=None, target_zone=None, move=move, move_detail=detail,
                   hrn_prediction=None)


def render_dashboard(video_path: Path, players_path: Path, ball_path: Path, annotations_path: Path,
                     out_dir: Path, team_names: dict[int, str], max_frames: int | None = None,
                     show: bool = False, snapshot_every: int = 80, nlp: str = "local",
                     local_model: str = DEFAULT_LOCAL_MODEL,
                     learned: bool = True, learned_weight: float = 0.3,
                     group_activity: bool = True) -> Path:
    with players_path.open("r", encoding="utf-8") as f:
        player_data = json.load(f)
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    ball_rows = load_ball_rows(ball_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(player_data))
    if max_frames:
        total = min(total, max_frames)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / "gritsense_dashboard.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CANVAS_W, CANVAS_H))

    learned_models = None
    if learned:
        from analytics.learned import LearnedModels
        try:
            learned_models = LearnedModels()
            print(f"Combining physics with trained GAT / HRN (learned weight {learned_weight:.0%}; "
                  f"HRN target influence scaled by test skill: zone {learned_models.zone_skill:.2f}, "
                  f"shot {learned_models.shot_skill:.2f})")
        except FileNotFoundError:
            print("No trained checkpoints (run train_gat_hrn.py) - using physics only")
    else:
        print("Physics only")
    engine = InsightEngine(learned=learned_models, learned_weight=learned_weight)
    backend = build_backend(nlp, local_model=local_model)
    commentary = CommentaryGenerator(team_names, fps, backend=backend)
    print(f"Commentary engine: {commentary.source}")
    ball_state = BallState()
    jumps = JumpDetector(Camera(annotations))
    motion = MotionSmoother(fps, TOP_SPEED)
    touches, touch_events = (detect_touches(load_clean(ball_path), player_data, Camera(annotations))
                             if ball_path.name == "ball_clean.csv" else ({}, []))
    print(f"Ball touches detected: {len(touch_events)}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(player_data))
    ga = load_group_activity(video_path, players_path, frame_count) if group_activity else None
    base = build_base(team_names)
    epv_history: list[float] = []
    insights_log = []
    H = None
    crop = None

    print(f"Rendering dashboard: {total} frames -> {out_video}")
    try:
        for frame_idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            H = homography_for_frame(annotations, frame_idx, H)
            if H is None:
                continue

            players = players_for_frame(player_data.get(str(frame_idx), {}), H, fps, frame_idx, jumps, motion)
            ball = ball_state.update(frame_idx, ball_rows.get(frame_idx), H)
            toucher = touches.get(frame_idx)
            ins = engine.step(frame_idx, players, ball, toucher)
            commentary.update(ins)

            epv_history.append(ins.epv)
            epv_history = epv_history[-int(fps * 4):]

            view = display_view(ins)
            target_img = overlay_broadcast(frame, H, view, players, ball, frame_idx)
            canvas = base.copy()
            vx, vy, vw, vh = VIDEO_BOX
            if crop is None:
                crop = detect_content_crop(frame)
            fx, fy, fw, fh = crop
            canvas[vy:vy + vh, vx:vx + vw] = cv2.resize(frame[fy:fy + fh, fx:fx + fw], (vw, vh),
                                                        interpolation=cv2.INTER_AREA)
            cx, cy, cw, ch = COURT_BOX
            canvas[cy:cy + ch, cx:cx + cw] = draw_tactical(view, players, ball, frame_idx)
            canvas = draw_text_layer(canvas, view, commentary, epv_history, target_img,
                                     frame_idx, fps, team_names, crop, players, ga)

            writer.write(canvas)
            row = ins.to_dict()
            window = group_activity_at(ga, frame_idx)
            row["group_activity"] = ({"label": window["label"], "score": window["score"]}
                                     if window else None)
            insights_log.append(row)
            if snapshot_every and frame_idx % snapshot_every == 0 and frame_idx > 0:
                cv2.imwrite(str(out_dir / f"dashboard_{frame_idx:04d}.png"), canvas)
            if show:
                cv2.imshow("GRITSENSE dashboard", cv2.resize(canvas, (1440, 810)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if (frame_idx + 1) % 50 == 0:
                print(f"  rendered {frame_idx + 1}/{total}")
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    with (out_dir / "dashboard_insights.json").open("w", encoding="utf-8") as f:
        json.dump(insights_log, f, indent=1)
    if ga is not None:
        with (out_dir / "group_activity.json").open("w", encoding="utf-8") as f:
            json.dump({k: v for k, v in ga.items() if k != "lookup"}, f, indent=1)
    with (out_dir / "commentary.txt").open("w", encoding="utf-8") as f:
        for line in commentary.history:
            f.write(f"[{line.timestamp}] ({line.kind}, {line.source}) {line.text}\n")

    print(f"Dashboard video: {out_video}")
    print(f"Commentary lines: {len(commentary.history)} -> {out_dir / 'commentary.txt'}")
    return out_video


def parse_args():
    parser = argparse.ArgumentParser(description="Render the GRITSENSE tactical dashboard.")
    parser.add_argument("--video", default="sample3.mp4")
    parser.add_argument("--players", default="player_keypoints_mf3_pose.json")
    parser.add_argument("--ball", default=BALL_FILE.name)
    parser.add_argument("--annotations", default="annotations.json")
    parser.add_argument("--out", default="output_recent/dashboard")
    parser.add_argument("--teams", nargs=2, default=["BRA", "JPN"], metavar=("LEFT", "RIGHT"),
                        help="Team names for the left (team 0) and right (team 1) halves.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="Show a live preview window (q to quit).")
    parser.add_argument("--nlp", choices=("local", "template"), default="local",
                        help="Commentary engine: offline local transformers LLM (falls back to "
                             "templates if it can't load), or templates only.")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL,
                        help="Hugging Face model id for --nlp local.")
    parser.add_argument("--physics-only", action="store_true",
                        help="Ignore the trained GAT / HRN and use the physics models alone.")
    parser.add_argument("--learned-weight", type=float, default=0.3,
                        help="Share of the trained GAT / HRN in every combined output (0-1).")
    parser.add_argument("--no-group-activity", action="store_true",
                        help="Hide the pretrained group-activity GAT (GAT/) card and rally timeline.")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def main():
    args = parse_args()
    render_dashboard(
        video_path=_resolve(args.video),
        players_path=_resolve(args.players),
        ball_path=_resolve(args.ball),
        annotations_path=_resolve(args.annotations),
        out_dir=_resolve(args.out),
        team_names={0: args.teams[0], 1: args.teams[1]},
        max_frames=args.max_frames,
        show=args.show,
        nlp=args.nlp,
        local_model=args.local_model,
        learned=not args.physics_only,
        learned_weight=args.learned_weight,
        group_activity=not args.no_group_activity,
    )


if __name__ == "__main__":
    main()
