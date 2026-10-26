"""
Player and team relations derived from the physics model.

These stand in for the GAT (who matters to whom) and HRN (team-level state)
until those models are trained, and are computed from the same reach-time
physics as the EPV, so no training data is needed:

    Key matchups  - impact of each defender = how much the attack's threat
                    rises when that defender is removed from the EPV model.
                    threat = mean of the best 10% of target cells.
    Team shape    - coverage: average chance (same race as the EPV) that some player can
                    reach within REF_FLIGHT_S (simulated flight of an average spike
                    to the centre of the opponent's half).
                    balance: how evenly that half is split between players
                    (1 = every player owns the same area).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .epv import _TABLES, CONTROL_SIGMA, COURT_WIDTH, NET_X, PhysicsEPV, PhysicsSurface, reach_times


def _reference_flight() -> float:
    """Flight time of a spike at the published average speed, hit from 1 m behind the net
    to the centre of the opponent's half (4.5 m past the net): the simulated physics value."""
    tables = _TABLES["SPIKE"]
    T, _ = tables[len(tables) // 2].solve(np.array([5.5]), np.array([1.0]))
    return float(T[0])


REF_FLIGHT_S = _reference_flight()    # reachable within this = covered
THREAT_TOP_SHARE = 0.10


@dataclass
class Matchup:
    attacker: int
    defender: int
    impact: float      # threat removed by this defender (EPV points, 0-1 scale)
    share: float       # share of the total impact across defenders


@dataclass
class TeamShape:
    coverage: float
    balance: float


def threat(surface: PhysicsSurface) -> float:
    values = np.sort(surface.values, axis=None)
    k = max(1, int(len(values) * THREAT_TOP_SHARE))
    return float(values[-k:].mean())


def key_matchups(epv: PhysicsEPV, surface: PhysicsSurface, defenders, key_attacker: int | None) -> list[Matchup]:
    """Rank defenders by how much threat they remove (leave-one-out on the EPV)."""
    if not defenders or key_attacker is None:
        return []
    base = threat(surface)
    impacts = []
    for i, d in enumerate(defenders):
        others = defenders[:i] + defenders[i + 1:]
        without = epv.surface(surface.contact, others, surface.defend_side)
        impacts.append(max(0.0, threat(without) - base))
    total = sum(impacts)
    matchups = [
        Matchup(key_attacker, d.id, imp, imp / total if total > 1e-9 else 0.0)
        for d, imp in zip(defenders, impacts)
    ]
    matchups.sort(key=lambda m: m.impact, reverse=True)
    return matchups


class TeamShapeModel:
    def __init__(self, resolution: float = 0.25):
        self.local = np.arange(resolution / 2.0, 9.0, resolution)
        self.ys = np.arange(resolution / 2.0, COURT_WIDTH, resolution)

    def team_shape(self, players, side: int) -> TeamShape:
        if not players:
            return TeamShape(0.0, 0.0)
        x0 = 0.0 if side == 0 else NET_X
        X, Y = np.meshgrid(x0 + self.local, self.ys)
        reach = reach_times(players, X, Y)                       # [n, ny, nx]
        # Same race as the EPV: chance some player gets there before an average spike lands.
        p_reach = 1.0 / (1.0 + np.exp(-np.clip((REF_FLIGHT_S - reach) / CONTROL_SIGMA, -30, 30)))
        coverage = float((1.0 - np.prod(1.0 - p_reach, axis=0)).mean())
        if len(players) < 2:
            return TeamShape(coverage, 0.0)
        # Each cell belongs to whichever player gets there first.
        owner = reach.argmin(axis=0)
        shares = np.bincount(owner.ravel(), minlength=len(players)) / owner.size
        # Coefficient of variation is 0 for an even split and sqrt(n-1) at worst.
        cv = shares.std() / max(shares.mean(), 1e-9)
        balance = float(np.clip(1.0 - cv / np.sqrt(len(players) - 1), 0.0, 1.0))
        return TeamShape(coverage, balance)

    def __call__(self, players_by_team: dict[int, list]) -> dict[int, TeamShape]:
        # Team 0 defends the left half (side 0), team 1 the right half.
        return {team: self.team_shape(members, team) for team, members in players_by_team.items()}
