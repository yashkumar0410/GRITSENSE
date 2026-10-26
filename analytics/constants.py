"""
Every physical constant used by the EPV model, with where it comes from.

    MEASURED   from this video, via the court homography / recovered camera
               (calibrate_physics.py writes them to physics_calibration.json,
               which overrides the fallback values below)
    PUBLISHED  from rules or peer-reviewed / official measurements (REFERENCES)
    DERIVED    computed from measured or published values
    ESTIMATED  no source found; kept to a minimum and listed in the viva guide

Elite men's values are used: the clip is the 2022 Men's World Championship
(Brazil v Japan).
"""

from __future__ import annotations

import json
from pathlib import Path

KMH = 1 / 3.6

REFERENCES = {
    "FIVB": "FIVB Official Volleyball Rules 2021-2024 (court 18x9 m, net 2.43 m men, "
            "antenna 0.80 m above the net, ball 65-67 cm circumference, 260-280 g)",
    "WIND_TUNNEL": "Wind-tunnel studies of FIVB volleyballs (Applied Sciences 2020, 10(9):3256 and related): "
                   "mass 0.269 kg; drag coefficient ~0.17 above the critical Reynolds number "
                   "~2.7-3.4e5",
    "SPHERE": "Standard subcritical drag coefficient of a sphere, Cd ~0.47",
    "SPIKE_RADAR": "Radar study of spike speed by competitive level (CROSBI 659042): "
                   "top group 106.3 +- 5.83 km/h",
    "SERVE_RADAR": "Olympic qualification radar study (2004): jump serve 23.03 +- 3.94 m/s, "
                   "jump float 12.05 +- 3.44 m/s, float 11.47 +- 4.22 m/s",
    "SERVE_ERRORS": "Men's college serve statistics (goldmedalsquared.com, MPSF/Big West 2023): "
                    "float 10.8% errors, topspin jump 20.9% (indicative; no elite figure found)",
    "REACTION": "Elite volleyball players' reaction time during matches (Optojump; J Hum Kinet): "
                "mean 0.571 s, range 0.264-1.055 s",
    "FIVB_REACH": "FIVB 2022 Men's World Championship team bulletin, Brazil (6 players): "
                  "block reach mean 3.18 m, highest reach mean 3.37 m (to verify against the bulletin)",
    "ARM_SPAN": "Anthropometry: arm span ~ standing height",
    "ANTICIPATION": "Volleyball anticipation (occlusion) studies (Frontiers in Sports and Active Living 2020): "
                    "defenders still anticipate attacks from video cut 240 ms before hand-ball contact; "
                    "experts' anticipation-related brain activity 100-300 ms before contact (PMC11274628)",
    "ATTACK_SUCCESS": "Millan-Sanchez, Morante Rabago & Urena Espa 2017, J Hum Sport Exerc 12(2) - "
                      "2010 Men's World Championship, 2925 terminal attacks: success 72.5% for outside "
                      "hitters, 55.1% for opposites (used to validate the model, not to fit it)",
}

CONSTANTS = {
    # name: (value, unit, source tag, reference key / note)
    "net_height_m":        (2.43, "m", "PUBLISHED", "FIVB"),
    "ball_radius_m":       (0.105, "m", "PUBLISHED", "FIVB"),
    "ball_mass_kg":        (0.269, "kg", "PUBLISHED", "WIND_TUNNEL"),
    "cd_supercritical":    (0.17, "", "PUBLISHED", "WIND_TUNNEL"),
    "cd_subcritical":      (0.47, "", "PUBLISHED", "SPHERE"),
    "re_critical":         (3.0e5, "", "PUBLISHED", "WIND_TUNNEL"),
    "air_density":         (1.2, "kg/m3", "PUBLISHED", "standard indoor air, 20 C"),
    "air_viscosity":       (1.81e-5, "Pa s", "PUBLISHED", "standard indoor air, 20 C"),
    "gravity":             (9.81, "m/s2", "PUBLISHED", "standard gravity"),

    "spike_speed":         (106.3 * KMH, "m/s", "PUBLISHED", "SPIKE_RADAR"),
    "spike_speed_sd":      (5.83 * KMH, "m/s", "PUBLISHED", "SPIKE_RADAR"),
    "jump_serve_speed":    (23.03, "m/s", "PUBLISHED", "SERVE_RADAR"),
    "jump_serve_speed_sd": (3.94, "m/s", "PUBLISHED", "SERVE_RADAR"),
    "float_serve_speed":   (12.05, "m/s", "PUBLISHED", "SERVE_RADAR (jump float, the most used float in men)"),
    "float_serve_speed_sd": (3.44, "m/s", "PUBLISHED", "SERVE_RADAR"),
    "attack_contact_m":    (3.37, "m", "PUBLISHED", "FIVB_REACH (highest reach)"),
    "block_reach_m":       (3.18, "m", "PUBLISHED", "FIVB_REACH (block reach)"),
    # Defenders start reading the attack before contact, so the effective reaction
    # after contact is the light-stimulus reaction minus the anticipation lead.
    "reaction_s":          (0.571 - 0.240, "s", "DERIVED", "REACTION 0.571 s minus ANTICIPATION 0.240 s"),
    "reaction_spread_s":   ((1.055 - 0.264) / 4, "s", "DERIVED", "REACTION range / 4 (range rule for SD)"),
    "jump_serve_error":    (0.209, "", "PUBLISHED", "SERVE_ERRORS"),
    "float_serve_error":   (0.108, "", "PUBLISHED", "SERVE_ERRORS"),

    # Measured from this video (fallbacks; physics_calibration.json overrides).
    "top_speed":           (4.75, "m/s", "MEASURED", "95th percentile of tracked player speed"),
    "stature_m":           (1.91, "m", "MEASURED", "median standing height through the camera"),
    "block_jump_m":        (0.70, "m", "MEASURED", "90th percentile of peak jumps near the net"),

    # Estimated: no published or measurable value found.
    "offspeed_max_speed":  (12.05, "m/s", "ESTIMATED", "tips / roll shots capped at float-serve hand speed"),
    "block_hand_halfwidth": (0.50, "m", "ESTIMATED", "two raised hands cover about +-0.5 m"),
    "block_height_sd":     (0.10, "m", "ESTIMATED", "spread of the blocker's top hand height"),
    "attack_aim_sd_ratio": (1.0, "", "ESTIMATED", "spike aim error taken equal to the jump serve's"),
}

_CALIBRATION = Path(__file__).with_name("physics_calibration.json")
_CALIBRATED_KEYS = {
    "top_speed_mps": ("top_speed", "MEASURED"),
    "stature_m": ("stature_m", "MEASURED"),
    "block_jump_m": ("block_jump_m", "MEASURED"),
}
if _CALIBRATION.exists():
    _measured = json.loads(_CALIBRATION.read_text(encoding="utf-8"))
    for key, (name, tag) in _CALIBRATED_KEYS.items():
        if _measured.get(key):
            _, unit, _, note = CONSTANTS[name]
            CONSTANTS[name] = (float(_measured[key]), unit, tag, note)


def value(name: str) -> float:
    return CONSTANTS[name][0]


def derived() -> dict[str, tuple[float, str, str, str]]:
    """Constants computed from the ones above."""
    stature = value("stature_m")
    return {
        "dive_reach_m": (stature / 2, "m", "DERIVED", "half the arm span (arm span ~ stature); ARM_SPAN"),
        "standing_block_reach_m": (value("block_reach_m") - value("block_jump_m"), "m", "DERIVED",
                                   "block reach minus the measured block jump"),
        "control_sigma_s": (value("reaction_spread_s") * 0.55, "s", "DERIVED",
                            "logistic scale matching the reaction-time SD (0.55 x SD)"),
    }


def table() -> list[tuple[str, float, str, str, str]]:
    rows = [(k, *v) for k, v in CONSTANTS.items()] + [(k, *v) for k, v in derived().items()]
    return sorted(rows, key=lambda r: ("MEASURED", "PUBLISHED", "DERIVED", "ESTIMATED").index(r[3]))
