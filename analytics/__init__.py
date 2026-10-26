"""Tactical analytics for the GRITSENSE dashboard: physics EPV, matchups, team shape and NLP commentary."""

from .commentary import CommentaryGenerator
from .insights import (
    COURT_LENGTH,
    COURT_WIDTH,
    NET_X,
    Ball,
    InsightEngine,
    Player,
    zone_rect,
)

__all__ = [
    "COURT_LENGTH",
    "COURT_WIDTH",
    "NET_X",
    "Ball",
    "CommentaryGenerator",
    "InsightEngine",
    "Player",
    "zone_rect",
]
