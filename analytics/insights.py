"""
Insight engine: physics models combined with the trained GAT / HRN.

Every number comes from the REAL tracked positions (homography court
coordinates in metres); nothing is mocked. The physics models are always
computed; when trained checkpoints are loaded, each output blends in the
learned models with weight w (default 0.3):

    where to hit  - physics EPV re-ranked by the HRN's shot and zone probabilities
    key matchups  - (1 - w) * physics threat-stopped share + w * GAT attention share
    team shape    - (1 - w) * physics coverage/balance  + w * HRN prediction

Components:

    PhysicsEPV.surface(contact, defenders, side)      -> PhysicsSurface      (analytics/epv.py)
    key_matchups(epv, surface, defenders, attacker)   -> list[Matchup]       (analytics/relations.py)
    TeamShapeModel()(players_by_team)                 -> dict[int, TeamShape]


Court frame (metres): x along the 18 m length (net at x = 9), y across the
9 m width (y = 0 is the far sideline, y = 9 the camera-side sideline).
Team 0 defends the left half (x < 9), team 1 the right half.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .epv import PhysicsEPV, PhysicsSurface
from .relations import Matchup, TeamShape, TeamShapeModel, key_matchups


COURT_LENGTH = 18.0
COURT_WIDTH = 9.0
NET_X = 9.0
ATTACK_LINE = 3.0  # distance of the attack line from the net
SET_TIME_S = C.value("set_time_s")   # measured time from a set to the attack

ZONE_DESCRIPTIONS = {
    1: "deep right corner",
    2: "right front, by the antenna",
    3: "middle front, behind the block",
    4: "left front, along the line",
    5: "deep left corner",
    6: "back-middle seam",
}

# zone -> (is_front_row, lateral third counted from the defending team's left)
_ZONE_LAYOUT = {4: (True, 0), 3: (True, 1), 2: (True, 2), 5: (False, 0), 6: (False, 1), 1: (False, 2)}


@dataclass
class Player:
    id: int
    team: int
    x: float
    y: float
    image_x: float
    image_y: float
    heading: tuple[float, float] | None = None  # unit vector in court metres
    confidence: float = 0.0
    velocity: tuple[float, float] | None = None  # m/s in court metres
    jump_m: float = 0.0                          # current jump height (analytics/pose_signals.py)
    arms_up: bool = False


@dataclass
class Ball:
    x: float
    y: float
    image_x: float
    image_y: float
    side: int  # 0 = ball on left half, 1 = right half
    visible: bool = True


@dataclass
class FrameInsight:
    frame: int
    attack_team: int | None
    defend_team: int | None
    surface: PhysicsSurface | None
    target: tuple[float, float] | None
    target_zone: int | None
    epv: float
    epv_mean: float
    gap_m: float
    block_prob: float          # chance the block stops the shot (physics)
    risk: float                # chance the shot lands out (aim error, physics)
    move: str
    move_detail: str
    key_player: int | None
    matchups: list[Matchup] = field(default_factory=list)
    team_shape: dict[int, TeamShape] = field(default_factory=dict)
    possession_changed: bool = False
    move_changed: bool = False
    flight_time: float = 0.0
    defender_time: float = 0.0
    p_in: float = 0.0
    p_block: float = 0.0
    ball_near: bool = False            # a player is on the ball (contact detected)
    phase: str = "waiting"             # serve / pass / set / attack / in flight / waiting
    attack_chance: bool = False        # a hit over the net is coming: show where to hit
    passer: int | None = None          # player passing or setting when the advice is 'set to'
    toucher: int | None = None         # that player's id
    # "physics", or "physics+GAT" / "physics+HRN" when blended with the trained models
    matchup_source: str = "physics"
    shape_source: str = "physics"
    target_source: str = "physics"
    hrn_prediction: dict | None = None

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "attack_team": self.attack_team,
            "target": None if self.target is None else [round(v, 3) for v in self.target],
            "target_zone": self.target_zone,
            "epv": round(self.epv, 4),
            "epv_mean": round(self.epv_mean, 4),
            "gap_m": round(self.gap_m, 3),
            "block_prob": round(self.block_prob, 4),
            "risk": round(self.risk, 4),
            "move": self.move,
            "key_player": self.key_player,
            "flight_time_s": round(self.flight_time, 3),
            "defender_time_s": round(self.defender_time, 3),
            "p_in": round(self.p_in, 4),
            "p_block": round(self.p_block, 4),
            "matchups": [
                {"attacker": m.attacker, "defender": m.defender,
                 "impact": round(m.impact, 4), "share": round(m.share, 4)}
                for m in self.matchups
            ],
            "team_shape": {
                str(team): {"coverage": round(s.coverage, 4), "balance": round(s.balance, 4)}
                for team, s in self.team_shape.items()
            },
            "matchup_source": self.matchup_source,
            "shape_source": self.shape_source,
            "target_source": self.target_source,
            "hrn_prediction": self.hrn_prediction,
            "ball_near": self.ball_near,
            "toucher": self.toucher,
            "phase": self.phase,
            "attack_chance": self.attack_chance,
            "passer": self.passer,
        }


def zone_of(x: float, y: float, side: int) -> int:
    """Volleyball zone (1-6) of a court point, from the perspective of the team on `side`."""
    depth = abs(x - NET_X)
    lateral = y if side == 1 else COURT_WIDTH - y
    third = int(np.clip(lateral // 3.0, 0, 2))
    front = depth < ATTACK_LINE
    for zone, (is_front, zone_third) in _ZONE_LAYOUT.items():
        if is_front == front and zone_third == third:
            return zone
    return 6


def zone_rect(zone: int, side: int) -> tuple[float, float, float, float]:
    """Zone rectangle (x1, y1, x2, y2) in metres for the team defending `side`."""
    is_front, third = _ZONE_LAYOUT[zone]
    d1, d2 = (0.0, ATTACK_LINE) if is_front else (ATTACK_LINE, 9.0)
    if side == 1:
        x1, x2 = NET_X + d1, NET_X + d2
        y1, y2 = 3.0 * third, 3.0 * third + 3.0
    else:
        x1, x2 = NET_X - d2, NET_X - d1
        y1, y2 = COURT_WIDTH - 3.0 * third - 3.0, COURT_WIDTH - 3.0 * third
    return x1, y1, x2, y2


def _move_from_shot(shot_name: str, contact: tuple[float, float], target: tuple[float, float]) -> str:
    if shot_name == "SPIKE":
        return "SPIKE · CROSS-COURT" if abs(contact[1] - target[1]) > 3.0 else "SPIKE · DOWN THE LINE"
    return shot_name


def _key_attacker(players: list[Player], ball: Ball | None, attack_team: int) -> Player | None:
    """The attacker playing the ball: nearest to it, or nearest the net without a ball."""
    attackers = [p for p in players if p.team == attack_team]
    if not attackers:
        return None
    if ball is not None:
        return min(attackers, key=lambda p: (p.x - ball.x) ** 2 + (p.y - ball.y) ** 2)
    return min(attackers, key=lambda p: abs(p.x - NET_X))


class InsightEngine:
    """Per-frame orchestration of the physics models, with smoothing and debouncing."""

    def __init__(self, target_smoothing: float = 0.18, move_debounce: int = 10, possession_debounce: int = 8,
                 learned=None, learned_weight: float = 0.3):
        # learned: analytics.learned.LearnedModels, or None for physics only.
        # learned_weight: share the trained GAT / HRN get in every combined output.
        self.learned = learned
        self.learned_weight = learned_weight
        self._zone_grids: dict[int, np.ndarray] = {}
        self._last_learned: tuple[int, dict] | None = None
        self.epv = PhysicsEPV()
        self.shape_model = TeamShapeModel()
        self.alpha = target_smoothing
        self.move_debounce = move_debounce
        self.possession_debounce = possession_debounce
        self._attack_team: int | None = None
        self._side_candidate: int | None = None
        self._side_frames = 0
        self._target: np.ndarray | None = None
        self._move = "READ & REPOSITION"
        self._candidate = self._move
        self._candidate_frames = 0
        self.stat_alpha = 0.2
        self._stats: np.ndarray | None = None
        self._impact: dict[int, float] = {}
        self._shape: dict[int, np.ndarray] = {}
        self._last_toucher: tuple[int, int] | None = None
        self._prev_toucher: int | None = None          # toucher in the previous frame
        self._touch_count = 0                           # touches by the team in possession
        self._last_set_frame: int | None = None         # last frame a set was being played
        self._move_hitter: int | None = None            # hitter the shown move belongs to
        self._planned_hitter: tuple[int, int] | None = None   # (id, frame) chosen on a pass / set

    def _ema(self, previous, value):
        return value if previous is None else (1 - self.stat_alpha) * previous + self.stat_alpha * value

    def _smooth_shape(self, shapes: dict[int, TeamShape]) -> dict[int, TeamShape]:
        out = {}
        for team, s in shapes.items():
            self._shape[team] = self._ema(self._shape.get(team), np.array([s.coverage, s.balance]))
            out[team] = TeamShape(*map(float, self._shape[team]))
        return out

    def _smooth_matchups(self, matchups: list[Matchup]) -> list[Matchup]:
        # Leave-one-out impacts jump with tracking noise; smooth per defender.
        seen = {m.defender for m in matchups}
        self._impact = {k: v for k, v in self._impact.items() if k in seen}
        for m in matchups:
            m.impact = float(self._ema(self._impact.get(m.defender), m.impact))
            self._impact[m.defender] = m.impact
        total = sum(m.impact for m in matchups)
        for m in matchups:
            m.share = m.impact / total if total > 1e-9 else 0.0
        return sorted(matchups, key=lambda m: m.impact, reverse=True)

    def _learned_prediction(self, frame_idx: int) -> dict | None:
        """Trained-model output for this frame; frames without a graph (no visible
        ball) reuse the last prediction for up to half a second."""
        pred = self.learned.predict(frame_idx)
        if pred is not None:
            self._last_learned = (frame_idx, pred)
            return pred
        if self._last_learned and frame_idx - self._last_learned[0] <= 15:
            return self._last_learned[1]
        return None

    def _zone_grid(self, surface: PhysicsSurface) -> np.ndarray:
        """Zone number (1-6) of every surface cell, cached per defending side."""
        side = surface.defend_side
        if side not in self._zone_grids:
            X, Y = np.meshgrid(surface.xs, surface.ys)
            self._zone_grids[side] = np.vectorize(lambda x, y: zone_of(x, y, side))(X, Y)
        return self._zone_grids[side]

    def _combined_matchups(self, physics: PhysicsSurface, pred: dict | None, key_player: int | None,
                           defenders) -> tuple[list[Matchup], str]:
        """Blend each defender's share of the threat stopped (physics) with the share of
        the attacker's GAT attention they receive: (1 - w) * physics + w * GAT."""
        phys = {m.defender: m.share for m in key_matchups(self.epv, physics, defenders, key_player)}
        gat = {}
        if pred is not None and key_player is not None:
            weights = self.learned.attention_to(pred, key_player, [d.id for d in defenders])
            total = sum(weights.values())
            if total > 1e-9:
                gat = {d: a / total for d, a in weights.items()}
        if not gat:
            combined, source = phys, "physics"
        else:
            w = self.learned_weight
            combined = {d: (1 - w) * phys.get(d, 0.0) + w * gat.get(d, 0.0) for d in set(phys) | set(gat)}
            source = "physics+GAT"
        matchups = [Matchup(key_player, d, s, s) for d, s in combined.items()] if key_player is not None else []
        return self._smooth_matchups(matchups), source

    def step(self, frame_idx: int, players: list[Player], ball: Ball | None,
             toucher: tuple[int, str] | None = None) -> FrameInsight:
        """toucher: id of the player detected on the ball (analytics.ball_track.detect_touches)."""
        possession_changed = False
        first_possession = False
        touch_kind = toucher[1] if toucher is not None else None
        toucher = toucher[0] if toucher is not None else None
        on_ball = next((p for p in players if p.id == toucher), None) if toucher is not None else None
        new_touch = toucher is not None and toucher != self._prev_toucher
        self._prev_toucher = toucher
        if on_ball is not None:
            # A detected touch decides possession at once.
            if on_ball.team != self._attack_team:
                first_possession = self._attack_team is None
                possession_changed = not first_possession
                self._attack_team = on_ball.team
                self._target = None
                self._impact = {}
                self._touch_count = 0
                self._planned_hitter = None
                self._last_set_frame = None
            if new_touch:
                self._touch_count += 1        # 1 = pass / dig, 2 = set, 3 = attack
            self._side_candidate, self._side_frames = on_ball.team, 0
            self._last_toucher = (on_ball.id, frame_idx)
        # Otherwise possession flips only after the ball is seen on the new side for
        # several frames, so single-frame tracker jitter near the net is ignored.
        elif ball is not None and ball.visible and ball.side != self._attack_team:
            if ball.side == self._side_candidate:
                self._side_frames += 1
            else:
                self._side_candidate, self._side_frames = ball.side, 1
            if self._attack_team is None or self._side_frames >= self.possession_debounce:
                first_possession = self._attack_team is None
                possession_changed = not first_possession
                self._attack_team = ball.side
                self._target = None
                self._impact = {}
                self._touch_count = 0          # the new team starts its three touches
                self._planned_hitter = None
                self._last_set_frame = None
        elif ball is not None and ball.visible:
            self._side_frames = 0

        by_team = {0: [p for p in players if p.team == 0], 1: [p for p in players if p.team == 1]}
        pred = self._learned_prediction(frame_idx) if self.learned is not None else None
        w = self.learned_weight

        raw_shape = self.shape_model(by_team)
        shape_source = "physics"
        if pred is not None and "team_shape" in pred:
            shape_source = "physics+HRN"
            raw_shape = {
                t: TeamShape((1 - w) * raw_shape[t].coverage + w * pred["team_shape"][t][0],
                             (1 - w) * raw_shape[t].balance + w * pred["team_shape"][t][1])
                for t in (0, 1)
            }
        team_shape = self._smooth_shape(raw_shape)
        hrn_prediction = None if pred is None or "zone" not in pred else {
            k: pred[k] for k in ("zone", "zone_conf", "shot", "shot_conf", "threat")
        }

        attack = self._attack_team
        if attack is None:
            return FrameInsight(
                frame=frame_idx, attack_team=None, defend_team=None, surface=None,
                target=None, target_zone=None, epv=0.0, epv_mean=0.0, gap_m=0.0,
                block_prob=0.0, risk=0.0, move=self._move, move_detail="waiting for ball",
                key_player=None, team_shape=team_shape, shape_source=shape_source,
                hrn_prediction=hrn_prediction,
            )

        defend = 1 - attack
        defenders = by_team[defend]

        # The player on the ball is the hitter. While the ball is in the air, keep the
        # team's last toucher for up to a second; otherwise the attacker nearest the ball.
        key = on_ball if on_ball is not None and on_ball.team == attack else None
        if key is None and self._planned_hitter is not None and frame_idx - self._planned_hitter[1] <= 45:
            # ball travelling to the hitter chosen on the pass / set
            key = next((p for p in players if p.id == self._planned_hitter[0] and p.team == attack), None)
        if key is None and self._last_toucher is not None and frame_idx - self._last_toucher[1] <= 30:
            key = next((p for p in players if p.id == self._last_toucher[0] and p.team == attack), None)
        if key is None:
            key = _key_attacker(players, ball, attack)
        if key is not None:
            contact = (key.x, key.y)
        elif ball is not None:
            contact = (ball.x, ball.y)
        else:
            contact = (NET_X - 2.0 if attack == 0 else NET_X + 2.0, COURT_WIDTH / 2.0)
        key_player = key.id if key is not None else None

        # Pure physics surface: matchups and the court-average comparison use it.
        # Blockers are judged at the hit: on the 3rd touch the hit is now; on a pass or
        # set the attack comes one set later (measured set time). Serves aren't blocked.
        attacking_now = on_ball is not None and on_ball.team == attack and touch_kind == "attack"
        time_to_hit = 0.0 if attacking_now else SET_TIME_S
        physics = self.epv.surface(contact, defenders, defend, time_to_hit=time_to_hit)

        # On the 1st and 2nd touch (pass, set), or when the player on the ball has no attack
        # that can land in, the advice is who to set: the teammate whose position gives the
        # best possible attack. The 3rd touch is the attack itself.
        set_to = None
        serving = any(s.zone == "serve" for s in self.epv.available_shots(contact, attack))
        # A pass or set (or a ball held away from the baseline) is played on to a hitter.
        passing = (on_ball is not None and on_ball.team == attack and not serving
                   and touch_kind in ("pass", "set", "hold"))
        passer = on_ball.id if passing else None
        if passing or physics.values.max() <= 0:
            best = -1.0 if passing else physics.values.max()
            for mate in by_team[attack]:
                if key is not None and mate.id == key.id:
                    continue
                option = self.epv.surface((mate.x, mate.y), defenders, defend, time_to_hit=SET_TIME_S)
                if option.values.max() > best:
                    best, physics, set_to = option.values.max(), option, mate
            if set_to is not None:
                key, key_player, contact = set_to, set_to.id, (set_to.x, set_to.y)
                if on_ball is not None:
                    self._planned_hitter = (set_to.id, frame_idx)
        elif on_ball is not None:
            self._planned_hitter = None

        # Combined surface: physics EPV re-ranked by the HRN's shot and zone beliefs, each
        # weighted by the HRN's measured skill on held-out data (no skill = no influence,
        # so a model that only learned "always Zone 6" can't pin the advice there).
        surface, target_source = physics, "physics"
        w_shot = w * self.learned.shot_skill if self.learned is not None else 0.0
        w_zone = w * self.learned.zone_skill if self.learned is not None else 0.0
        if pred is not None and "shot_probs" in pred and (w_shot > 0 or w_zone > 0):
            surface = self.epv.surface(contact, defenders, defend,
                                       shot_prior=pred["shot_probs"], prior_weight=w_shot,
                                       time_to_hit=physics.time_to_hit)
            surface.apply_zone_prior(self._zone_grid(surface), pred["zone_probs"], w_zone)
            # Rescale to the physics range so the map keeps a probability-like scale.
            peak = surface.values.max()
            if peak > 1e-9:
                surface.values = surface.values * (physics.values.max() / peak)
            target_source = "physics+HRN"

        matchups, matchup_source = self._combined_matchups(physics, pred, key_player, defenders)

        raw = np.array(surface.best())
        smoothed = raw if self._target is None else (1 - self.alpha) * self._target + self.alpha * raw
        # Smoothing can drift onto cells no trajectory can reach (e.g. just behind the net);
        # snap to the best reachable cell within 1 m of the smoothed point.
        X, Y = np.meshgrid(surface.xs, surface.ys)
        near = np.hypot(X - smoothed[0], Y - smoothed[1]) <= 1.0
        if near.any() and surface.values[near].max() > 0:
            iy, ix = np.unravel_index(np.argmax(np.where(near, surface.values, -1.0)), X.shape)
            smoothed = np.array([X[iy, ix], Y[iy, ix]])
        else:
            smoothed = raw
        self._target = smoothed
        tx, ty = float(self._target[0]), float(self._target[1])

        # Sample the surface at the smoothed target so displayed numbers are consistent.
        at = surface.sample(tx, ty)
        epv = at["epv"]
        shot = at["shot"]
        risk = float(np.clip(1.0 - at["p_in"], 0.0, 1.0))
        block = float(np.clip(at["p_block"], 0.0, 1.0))
        gap = min((math.hypot(p.x - tx, p.y - ty) for p in defenders), default=3.5)

        candidate = _move_from_shot(shot.name, contact, (tx, ty))
        move_changed = False
        if candidate == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate, self._candidate_frames = candidate, 1
        # A different hitter means a different attack: show it at once, without waiting.
        hitter_changed = key_player != self._move_hitter
        self._move_hitter = key_player
        if self._candidate != self._move and (self._candidate_frames >= self.move_debounce or possession_changed
                                              or first_possession or hitter_changed):
            self._move = self._candidate
            move_changed = True

        # Tracked speeds are noisy frame to frame; smooth the displayed numbers.
        raw_stats = np.array([epv, block, risk, at["flight_t"], at["defender_t"]])
        self._stats = raw_stats if possession_changed else self._ema(self._stats, raw_stats)
        epv, block, risk, at["flight_t"], at["defender_t"] = map(float, self._stats)

        # Phase of the rally: "where to hit" is shown only when a hit over the net is coming.
        if on_ball is not None and on_ball.team == attack:
            phase = "serve" if serving else {"attack": "attack", "set": "set"}.get(touch_kind, "pass")
            if phase == "set":
                self._last_set_frame = frame_idx
        elif (self._planned_hitter is not None and self._last_set_frame is not None
              and frame_idx - self._last_set_frame <= 1.5 * SET_TIME_S * 30):
            phase = "set"          # the set is on its way to the hitter
        else:
            phase = "in flight"
        attack_chance = phase in ("serve", "set", "attack")

        detail = f"ball {at['flight_t']:.2f} s  vs  defender {at['defender_t']:.2f} s"
        if set_to is not None:
            detail = (f"set by #{passer}  ·  {detail}" if passer is not None
                      else f"set to #{set_to.id}  ·  {detail}")

        return FrameInsight(
            frame=frame_idx,
            attack_team=attack,
            defend_team=defend,
            surface=surface,
            target=(tx, ty),
            target_zone=zone_of(tx, ty, defend),
            epv=epv,
            epv_mean=float(physics.values.mean()),
            gap_m=float(gap),
            block_prob=block,
            risk=risk,
            move=self._move,
            move_detail=detail,
            key_player=key_player,
            matchups=matchups[:4],
            team_shape=team_shape,
            possession_changed=possession_changed,
            move_changed=move_changed,
            flight_time=at["flight_t"],
            defender_time=at["defender_t"],
            p_in=at["p_in"],
            p_block=at["p_block"],
            matchup_source=matchup_source,
            target_source=target_source,
            shape_source=shape_source,
            hrn_prediction=hrn_prediction,
            ball_near=on_ball is not None,
            phase=phase,
            attack_chance=attack_chance,
            passer=passer,
            toucher=on_ball.id if on_ball is not None else None,
        )
