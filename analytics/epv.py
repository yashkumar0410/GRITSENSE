"""
Physics-based Expected Point Value (EPV) for volleyball.

No training data and no hand-set shot behaviour: every constant comes from
analytics/constants.py (measured from this video, published, or derived).

For every 0.25 m target cell on the defending half and every shot the attacker
can legally play from their position:

    Flight      the ball is simulated from the contact height under gravity and
                air drag (quadratic, drag coefficient by Reynolds number) for each
                launch speed in the shot's published range (mean +- SD) and every
                launch angle. A cell is reachable only if a trajectory lands there
                AND clears the net (2.43 m + ball radius) where it crosses it; the
                fastest such trajectory is used (hit as hard as still lands in).
                Tips / roll shots must also clear the block's full reach.
    Defenders   reach time = reaction (published) + time to land if airborne
                (free fall, sqrt(2h/g)) + run at the measured top speed to
                within half an arm span (dive reach) of the cell.
                P(reached by i) = sigmoid((T - t_i) / sigma), sigma from the
                published reaction-time spread ("pitch control" race).
    Block       each defender at the net blocks the ball if it crosses the net
                below their hands (published block reach, minus the part of
                the measured block jump they haven't reached yet) and within
                their hands' width.
    In / out    landing error ~ normal(0, aim_sd); aim_sd per shot is fitted
                so serves miss at the published serve-error rates.

    EPV(cell, shot) = P(in) * (1 - P(blocked)) * P(no defender reaches it)

The surface keeps the best shot per cell, so the EPV also decides the move.
Positions are court metres from the homography (net at x = 9).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C

COURT_LENGTH = 18.0
COURT_WIDTH = 9.0
NET_X = 9.0
ATTACK_LINE = 3.0          # FIVB: back-row players must hit from behind the 3 m line

G = C.value("gravity")
BALL_R = C.value("ball_radius_m")
NET_TOP = C.value("net_height_m") + BALL_R          # ball centre must pass above this
REACTION_S = C.value("reaction_s")
TOP_SPEED = C.value("top_speed")
_D = C.derived()
DIVE_REACH = _D["dive_reach_m"][0]
CONTROL_SIGMA = _D["control_sigma_s"][0]
STANDING_BLOCK_REACH = _D["standing_block_reach_m"][0]
BLOCK_REACH = C.value("block_reach_m")
BLOCK_JUMP = C.value("block_jump_m")
BLOCK_HALFWIDTH = C.value("block_hand_halfwidth")
BLOCK_HEIGHT_SD = C.value("block_height_sd")


def _phi(z):
    """Standard normal CDF (tanh approximation, vectorised)."""
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ----------------------------------------------------------------------------
# Ball flight
# ----------------------------------------------------------------------------

S_STEP = 0.05                       # m, horizontal sampling of stored trajectories
S_MAX = 32.0
ANGLES = np.radians(np.arange(-70.0, 75.01, 0.25))
_SIM_CACHE: dict = {}


def simulate(speed: float, contact_h: float, dt: float = 0.002):
    """Fly the ball for every launch angle at once (gravity + quadratic drag).

    Returns landing distance [A], flight time [A] and height on the S grid [A, S]
    (NaN after landing). Landing = ball centre at ball-radius height. Cached.
    """
    key = (round(speed, 3), round(contact_h, 3))
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]
    rho, mu = C.value("air_density"), C.value("air_viscosity")
    area = np.pi * BALL_R ** 2
    mass = C.value("ball_mass_kg")
    n = len(ANGLES)
    s = np.zeros(n); z = np.full(n, contact_h)
    vs = speed * np.cos(ANGLES); vz = speed * np.sin(ANGLES)
    path_s, path_z = [s.copy()], [z.copy()]
    land_s = np.full(n, np.nan); land_t = np.full(n, np.nan)
    flying = np.ones(n, bool)
    t = 0.0
    while flying.any() and t < 6.0:
        v = np.hypot(vs, vz)
        reynolds = rho * v * 2 * BALL_R / mu
        cd = np.where(reynolds > C.value("re_critical"), C.value("cd_supercritical"), C.value("cd_subcritical"))
        k = 0.5 * rho * cd * area / mass
        s_new = s + vs * dt; z_new = z + vz * dt
        vs = vs - k * v * vs * dt
        vz = vz - (G + k * v * vz) * dt
        landed = flying & (z_new <= BALL_R)
        if landed.any():
            frac = (z[landed] - BALL_R) / np.maximum(z[landed] - z_new[landed], 1e-9)
            land_s[landed] = s[landed] + frac * (s_new[landed] - s[landed])
            land_t[landed] = t + frac * dt
            flying &= ~landed
        flying &= s_new < S_MAX
        s, z, t = s_new, z_new, t + dt
        path_s.append(s.copy()); path_z.append(z.copy())
    path_s, path_z = np.array(path_s), np.array(path_z)
    s_grid = np.arange(0.0, S_MAX, S_STEP)
    heights = np.full((n, len(s_grid)), np.nan)
    for a in range(n):
        end = land_s[a] if np.isfinite(land_s[a]) else path_s[-1, a]
        cols = s_grid <= end
        heights[a, cols] = np.interp(s_grid[cols], path_s[:, a], path_z[:, a])
    _SIM_CACHE[key] = (land_s, land_t, heights)
    return _SIM_CACHE[key]


@dataclass
class FlightTable:
    """One branch (low = direct, high = lob) of the trajectories for one speed and contact height."""
    land_s: np.ndarray
    land_t: np.ndarray
    heights: np.ndarray
    idx: np.ndarray        # rows of `heights` belonging to this branch, in land_s order

    @classmethod
    def build(cls, speed: float, contact_h: float, branch: str = "low"):
        land_s, land_t, heights = simulate(float(speed), float(contact_h))
        ok = np.isfinite(land_s)
        apex = int(np.argmax(np.where(ok, land_s, -1.0)))      # angle of maximum range
        rows = np.arange(0, apex + 1) if branch == "low" else np.arange(apex, len(ANGLES))
        rows = rows[ok[rows]]
        order = rows[np.argsort(land_s[rows])]
        return cls(land_s[order], land_t[order], heights, order)

    def solve(self, dist: np.ndarray, net_dist: np.ndarray):
        """Flight time and height at the net for landing at `dist` (NaN if unreachable)."""
        if len(self.land_s) < 2:
            nan = np.full(dist.shape, np.nan)
            return nan, nan
        pos = np.interp(dist, self.land_s, np.arange(len(self.land_s)), left=np.nan, right=np.nan)
        valid = np.isfinite(pos)
        pos_c = np.where(valid, pos, 0.0)
        lo = np.floor(pos_c).astype(int); hi = np.minimum(lo + 1, len(self.idx) - 1); w = pos_c - lo
        T = (1 - w) * self.land_t[lo] + w * self.land_t[hi]
        col = np.clip(np.round(net_dist / S_STEP).astype(int), 0, self.heights.shape[1] - 1)
        z_net = (1 - w) * self.heights[self.idx[lo], col] + w * self.heights[self.idx[hi], col]
        return np.where(valid, T, np.nan), np.where(valid, z_net, np.nan)


# ----------------------------------------------------------------------------
# Shots
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Shot:
    name: str
    zone: str                  # "front", "back" or "serve": where it can be played from
    speeds: tuple              # launch speeds the hitter can choose from (published mean +- SD)
    contact_h: float
    aim_sd: float
    blockable: bool
    over_block: bool = False   # must clear the block's reach (tips / roll shots), any arc


def _fit_aim_sd(error_rate: float) -> float:
    """Aim SD such that aiming uniformly over the opponent's half misses at `error_rate`."""
    xs = NET_X + np.arange(0.25, 9.0, 0.5); ys = np.arange(0.25, 9.0, 0.5)
    X, Y = np.meshgrid(xs, ys)
    p_in = lambda sd: _phi(Y / sd) * _phi((COURT_WIDTH - Y) / sd) * _phi((COURT_LENGTH - X) / sd)
    lo, hi = 0.05, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if 1.0 - p_in(mid).mean() < error_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _speed_range(mean: float, sd: float) -> tuple:
    return (mean - sd, mean, mean + sd)


def build_shots():
    attack_h = C.value("attack_contact_m")
    float_h = STANDING_BLOCK_REACH                      # jump float: contact near standing reach
    spike = _speed_range(C.value("spike_speed"), C.value("spike_speed_sd"))
    jump_serve = _speed_range(C.value("jump_serve_speed"), C.value("jump_serve_speed_sd"))
    float_serve = _speed_range(C.value("float_serve_speed"), C.value("float_serve_speed_sd"))
    offspeed = tuple(float(v) for v in np.arange(5.0, C.value("offspeed_max_speed") + 0.01, 1.0))
    jump_sd = _fit_aim_sd(C.value("jump_serve_error"))
    float_sd = _fit_aim_sd(C.value("float_serve_error"))
    attack_sd = jump_sd * C.value("attack_aim_sd_ratio")
    shots = [
        Shot("SPIKE", "front", spike, attack_h, attack_sd, True),
        Shot("TIP / ROLL SHOT", "front", offspeed, attack_h, float_sd, True, over_block=True),
        Shot("BACK-ROW ATTACK", "back", spike, attack_h, attack_sd, True),
        Shot("JUMP SERVE", "serve", jump_serve, attack_h, jump_sd, False),
        Shot("FLOAT SERVE", "serve", float_serve, float_h, float_sd, False),
    ]
    tables = {}
    for shot in shots:
        branches = ("low", "high") if shot.over_block else ("low",)
        tables[shot.name] = [FlightTable.build(v, shot.contact_h, b) for v in shot.speeds for b in branches]
    return shots, tables


SHOTS, _TABLES = build_shots()


# ----------------------------------------------------------------------------
# Players
# ----------------------------------------------------------------------------


def reach_times(players, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Seconds for each player to reach each cell: [n_players, ny, nx].

    Reaction (published), then landing if airborne (free fall from the current
    jump height), then a run at the measured top speed from the position
    projected along their current movement, to within dive reach of the cell.
    """
    if not players:
        return np.full((1,) + X.shape, 9.0)
    starts, delays = [], []
    for p in players:
        vx, vy = p.velocity if p.velocity is not None else (0.0, 0.0)
        starts.append((p.x + vx * REACTION_S, p.y + vy * REACTION_S))
        delays.append(np.sqrt(2.0 * max(p.jump_m, 0.0) / G))
    sx = np.array([s[0] for s in starts])[:, None, None]
    sy = np.array([s[1] for s in starts])[:, None, None]
    delay = np.array(delays)[:, None, None]
    run = np.maximum(0.0, np.hypot(X - sx, Y - sy) - DIVE_REACH)
    return REACTION_S + delay + run / TOP_SPEED


def hand_height(player) -> float:
    """Top of a blocker's hands: standing reach plus the jump reached so far."""
    return STANDING_BLOCK_REACH + min(max(player.jump_m, 0.0), BLOCK_JUMP)


# ----------------------------------------------------------------------------
# Surface
# ----------------------------------------------------------------------------


@dataclass
class PhysicsSurface:
    xs: np.ndarray
    ys: np.ndarray
    values: np.ndarray           # [ny, nx] best score per cell (= EPV without priors)
    defend_side: int
    shot: np.ndarray             # [ny, nx] index into `shots`
    shots: list
    layers: dict                 # per-cell components of the best shot
    contact: tuple[float, float]

    def sample(self, x: float, y: float) -> dict:
        ix = int(np.clip(np.searchsorted(self.xs, x), 0, len(self.xs) - 1))
        iy = int(np.clip(np.searchsorted(self.ys, y), 0, len(self.ys) - 1))
        out = {name: float(layer[iy, ix]) for name, layer in self.layers.items()}
        out["epv"] = out["physics_epv"]            # a true probability, for display
        out["score"] = float(self.values[iy, ix])  # what the target is ranked by
        out["shot"] = self.shots[int(self.shot[iy, ix])]
        return out

    def apply_zone_prior(self, zone_grid: np.ndarray, zone_probs, weight: float) -> None:
        """Re-weight the ranking by a zone prior (e.g. the HRN's zone probabilities):
        factor = 1 - w + w * 6 * p(zone of cell); a uniform prior changes nothing."""
        probs = np.asarray(zone_probs, dtype=float)
        if weight <= 0 or probs.sum() <= 1e-9:
            return
        probs = probs / probs.sum()
        self.values = self.values * (1.0 - weight + weight * len(probs) * probs[zone_grid - 1])

    def best(self) -> tuple[float, float]:
        iy, ix = np.unravel_index(np.argmax(self.values), self.values.shape)
        return float(self.xs[ix]), float(self.ys[iy])


class PhysicsEPV:
    def __init__(self, resolution: float = 0.25):
        self.local = np.arange(resolution / 2.0, 9.0, resolution)
        self.ys = np.arange(resolution / 2.0, COURT_WIDTH, resolution)

    @staticmethod
    def available_shots(contact: tuple[float, float], attack_side: int) -> list[Shot]:
        cx = contact[0]
        # Servers hit from behind the end line; tracked feet can sit up to ~1 m inside it,
        # and nobody attacks from the last metre of the court during a rally.
        at_baseline = cx < 1.0 if attack_side == 0 else cx > COURT_LENGTH - 1.0
        if at_baseline:
            return [s for s in SHOTS if s.zone == "serve"]
        zone = "front" if abs(cx - NET_X) <= ATTACK_LINE else "back"
        return [s for s in SHOTS if s.zone == zone]

    @staticmethod
    def _flight(shot: Shot, dist, net_dist):
        """Fastest feasible flight for every cell: (flight time, ball height at the net, feasible).

        Feasible = some speed in the shot's range and some launch angle lands on the cell
        while the ball clears the net (or, for tips / roll shots, the block's full reach).
        """
        clear = BLOCK_REACH + BALL_R if shot.over_block else NET_TOP
        best_T = np.full(dist.shape, np.inf); best_z = np.full(dist.shape, np.nan)
        for table in _TABLES[shot.name]:
            T, z_net = table.solve(dist, net_dist)
            ok = np.isfinite(T) & (z_net >= clear) & (T < best_T)
            best_T = np.where(ok, T, best_T); best_z = np.where(ok, z_net, best_z)
        ok = np.isfinite(best_T)
        return np.where(ok, best_T, np.nan), best_z, ok

    def surface(self, contact: tuple[float, float], defenders, defend_side: int,
                shot_prior: dict[str, float] | None = None, prior_weight: float = 0.0) -> PhysicsSurface:
        """EPV surface. With shot_prior (e.g. the HRN's shot probabilities), each shot's
        EPV is re-weighted by (1 - w + w * n * p), where p is the prior renormalised over
        the n available shots: a uniform prior leaves the physics ranking unchanged.
        The pure physics EPV of the chosen shot is kept in layers["physics_epv"]."""
        x0 = 0.0 if defend_side == 0 else NET_X
        xs = x0 + self.local
        X, Y = np.meshgrid(xs, self.ys)
        bx, by = contact
        attack_side = 1 - defend_side
        shots = self.available_shots(contact, attack_side) or [SHOTS[0]]

        dist = np.hypot(X - bx, Y - by)
        dx = X - bx
        along = np.where(np.abs(dx) > 1e-6, (NET_X - bx) / np.where(np.abs(dx) > 1e-6, dx, 1.0), 0.0)
        along = np.clip(along, 0.0, 1.0)
        net_dist = along * dist                                   # horizontal distance to the net crossing
        y_net = by + along * (Y - by)
        far_line = 0.0 if defend_side == 0 else COURT_LENGTH
        depth_margin = np.abs(far_line - X)

        reach_t = reach_times(defenders, X, Y)                    # [n_def, ny, nx]
        defender_t = reach_t.min(axis=0)
        blockers = [d for d in defenders if abs(d.x - NET_X) <= DIVE_REACH]

        best = np.full(X.shape, -1.0)
        best_idx = np.zeros(X.shape, dtype=int)
        names = ("flight_t", "defender_t", "p_in", "p_block", "p_unreached", "physics_epv", "net_clearance")
        layers = {k: np.zeros(X.shape) for k in names}

        factors = {s.name: 1.0 for s in shots}
        if shot_prior and prior_weight > 0:
            total = sum(shot_prior.get(s.name, 0.0) for s in shots)
            if total > 1e-9:
                n = len(shots)
                factors = {s.name: 1.0 - prior_weight + prior_weight * n * shot_prior.get(s.name, 0.0) / total
                           for s in shots}

        for idx, shot in enumerate(shots):
            flight, z_net, feasible = self._flight(shot, dist, net_dist)
            flight_safe = np.where(feasible, flight, 0.0)
            p_reach = _sigmoid((flight_safe[None] - reach_t) / CONTROL_SIGMA)
            p_unreached = np.prod(1.0 - p_reach, axis=0)
            sd = shot.aim_sd
            p_in = _phi(Y / sd) * _phi((COURT_WIDTH - Y) / sd) * _phi(depth_margin / sd)

            p_block = np.zeros(X.shape)
            if blockers and shot.blockable:
                keep = np.ones(X.shape)
                z_safe = np.where(feasible, z_net, 0.0)
                for b in blockers:
                    below_hands = _phi((hand_height(b) - (z_safe - BALL_R)) / BLOCK_HEIGHT_SD)
                    within = np.exp(-((y_net - b.y) ** 2) / (2 * BLOCK_HALFWIDTH ** 2))
                    keep *= 1.0 - below_hands * within
                p_block = 1.0 - keep

            epv = np.where(feasible, p_in * (1.0 - p_block) * p_unreached, 0.0)
            score = epv * factors.get(shot.name, 1.0)
            better = score > best
            best = np.where(better, score, best)
            best_idx = np.where(better, idx, best_idx)
            for name, layer in (("flight_t", flight_safe), ("defender_t", defender_t), ("p_in", p_in),
                                ("p_block", p_block), ("p_unreached", p_unreached), ("physics_epv", epv),
                                ("net_clearance", np.where(feasible, z_net - NET_TOP, np.nan))):
                layers[name] = np.where(better, layer, layers[name])

        # Pure physics scores are probabilities in [0, 1]; prior-weighted ones may exceed 1.
        return PhysicsSurface(xs, self.ys, np.maximum(best, 0.0), defend_side,
                              best_idx, shots, layers, (float(bx), float(by)))
