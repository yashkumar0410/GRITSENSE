"""
Natural-language commentary for the tactical dashboard.

Rule-based event detection (possession change, new move, periodic model
signals) decides WHEN to speak; an NLP backend (analytics/nlp.py: Claude or a
local LLM) writes WHAT is said from the structured match facts. Seeded
templates are kept as the fallback when no backend is available.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .insights import ZONE_DESCRIPTIONS, FrameInsight
from .nlp import is_grounded


@dataclass
class CommentaryLine:
    frame: int
    timestamp: str
    text: str
    kind: str  # "possession" | "move" | "insight"
    source: str = "TEMPLATE"  # which engine wrote the line


_POSSESSION = [
    "Ball crosses to the {att} side. {deff} must reset their block and cover Zone {zone}.",
    "Possession switches to {att}. {deff}'s {zdesc} is the weak spot.",
    "{att} take control. {deff} are scrambling — Zone {zone} opens up ({gap:.1f} m gap).",
]

_MOVE = [
    "{att} #{key} should go {move_l} into Zone {zone}, the {zdesc}. A {epv:.0%} chance of an outright point.",
    "Recommended: {move_l} by #{key} — target the {zdesc} (Zone {zone}). {deff} left {gap:.1f} m uncovered there.",
    "Play it to Zone {zone}: {move_l} from #{key}, the {zdesc} gives a {epv:.0%} point chance.",
    "Best option for {att}: {move_l} into the {zdesc}, a {epv:.0%} point chance with {block:.0%} block risk.",
]

_INSIGHT_MATCHUP = [
    "Key matchup: #{a} against #{b}. {deff}'s #{b} alone takes {w:.0%} of the threat off the table.",
    "#{b} is {deff}'s most important defender right now. #{a} has to play away from them.",
]

_INSIGHT_SHAPE = [
    "{deff} can only reach {cov:.0%} of their half in time. The {zdesc} stays exposed.",
    "{deff}'s defence is lopsided (balance {obal:.0%}). A fast ball to Zone {zone} beats the rotation.",
    "{deff} cover {cov:.0%} of their court with a balance of {obal:.0%}. Find the gap at Zone {zone}.",
]


class CommentaryGenerator:
    def __init__(self, team_names: dict[int, str], fps: float, insight_every: int = 110, seed: int = 7,
                 backend=None):
        self.team_names = team_names
        self.backend = backend
        self.fps = fps
        self.insight_every = insight_every
        self.rng = random.Random(seed)
        self.history: list[CommentaryLine] = []
        self._last_emit = -10_000
        self._last_template: dict[str, int] = {}
        self._pending_move = False

    def _pick(self, kind: str, templates: list[str]) -> str:
        # Never repeat the same template twice in a row for a given kind.
        choices = [i for i in range(len(templates)) if i != self._last_template.get(kind)]
        idx = self.rng.choice(choices)
        self._last_template[kind] = idx
        return templates[idx]

    def _timestamp(self, frame: int) -> str:
        seconds = frame / self.fps
        return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"

    def _slots(self, ins: FrameInsight) -> dict:
        att = self.team_names.get(ins.attack_team, "Attack")
        deff = self.team_names.get(ins.defend_team, "Defence")
        zone = ins.target_zone or 6
        opp = ins.team_shape.get(ins.defend_team)
        return {
            "att": att,
            "deff": deff,
            "zone": zone,
            "zdesc": ZONE_DESCRIPTIONS.get(zone, "open court"),
            "gap": ins.gap_m,
            "epv": ins.epv,
            "lift": ins.epv - ins.epv_mean,
            "block": ins.block_prob,
            "move_l": ins.move.lower().replace(" · ", " "),
            "detail": ins.move_detail,
            "key": ins.key_player if ins.key_player is not None else "?",
            "cov": opp.coverage if opp else 0.0,
            "obal": opp.balance if opp else 0.0,
        }

    @property
    def source(self) -> str:
        return self.backend.name if self.backend is not None else "TEMPLATE"

    def _facts(self, ins: FrameInsight, slots: dict) -> dict:
        return {
            "attacking_team": slots["att"],
            "defending_team": slots["deff"],
            "recommended_move": ins.move,
            "move_detail": ins.move_detail,
            "key_player": f"#{slots['key']}",
            "target_zone": slots["zone"],
            "target_description": slots["zdesc"],
            "epv": round(ins.epv, 2),
            "epv_vs_court_average": round(slots["lift"], 2),
            "chance_ball_lands_out": f"{ins.risk:.0%}",
            "gap_to_nearest_defender_m": round(ins.gap_m, 1),
            "ball_flight_time_s": round(ins.flight_time, 2),
            "nearest_defender_time_s": round(ins.defender_time, 2),
            "chance_ball_lands_in": f"{ins.p_in:.0%}",
            "block_chance": f"{ins.p_block:.0%}",
            "key_matchup": None if not ins.matchups else
                {"attacker": f"#{slots['a']}", "defender": f"#{slots['b']}",
                 "share_of_threat_removed": f"{slots['w']:.0%}"},
            "team_shape": {
                "defending_coverage": f"{slots['cov']:.0%}",
                "defending_balance": f"{slots['obal']:.0%}",
            },
        }

    def _generate(self, kind: str, ins: FrameInsight, slots: dict) -> str | None:
        if self.backend is None:
            return None
        recent = [line.text for line in self.history[-3:]]
        facts = self._facts(ins, slots)
        try:
            text = self.backend.generate(kind, facts, recent)
            return text if text and is_grounded(text, facts) else None
        except Exception as exc:
            print(f"  [nlp] {self.backend.name} failed ({type(exc).__name__}: {str(exc)[:100]}); "
                  "switching to template commentary")
            self.backend = None
            return None

    def update(self, ins: FrameInsight) -> CommentaryLine | None:
        """Emit a new line when something worth saying happens; otherwise None."""
        if ins.attack_team is None:
            return None
        since = ins.frame - self._last_emit
        # A move change that arrives too soon after the last line is queued, not dropped,
        # so the commentary always catches up with the recommendation card.
        self._pending_move = self._pending_move or ins.move_changed
        kind = None
        if ins.possession_changed and since > 20:
            kind, template = "possession", self._pick("possession", _POSSESSION)
        elif self._pending_move and since > 30 and ins.attack_chance:
            self._pending_move = False
            kind, template = "move", self._pick("move", _MOVE)
        elif not self.history and since > 0 and ins.attack_chance:
            kind, template = "move", self._pick("move", _MOVE)
        elif since > self.insight_every:
            if ins.matchups and self.rng.random() < 0.5:
                kind, template = "insight", self._pick("matchup", _INSIGHT_MATCHUP)
            else:
                kind, template = "insight", self._pick("shape", _INSIGHT_SHAPE)
        if kind is None:
            return None

        slots = self._slots(ins)
        if ins.matchups:
            top = ins.matchups[0]
            slots.update(a=top.attacker, b=top.defender, w=top.share)
        else:
            slots.update(a="?", b="?", w=0.0)

        engine = self.source
        text = self._generate(kind, ins, slots)
        if not text:
            text, engine = template.format(**slots), "TEMPLATE"
        line = CommentaryLine(ins.frame, self._timestamp(ins.frame), text, kind, engine)
        self.history.append(line)
        self._last_emit = ins.frame
        return line
