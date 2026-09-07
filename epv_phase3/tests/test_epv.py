"""
tests/test_epv.py
------------------
Unit tests for the Member 3 EPV package: state building (base + ball
augmented), rally segmentation + game-state labels, ball log loading,
dataset proxy outcomes, and the model forward pass.

Run from epv_phase3/:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epv.action_generator import ACTIONS, NUM_ACTIONS, infer_action_from_transition
from epv.dataset import compute_proxy_outcome
from epv.model import EPVModel
from epv.rally_segmentation import (
    STATE_DEAD,
    STATE_LIVE,
    STATE_SERVE,
    assign_game_states,
    segment_rallies,
)
from epv.state_builder import (
    BALL_FEATS,
    BASE_STATE_DIM,
    STATE_DIM,
    STATE_DIM_WITH_BALL,
    build_rally_states,
    build_state,
    feature_names,
    state_dim,
)
from epv.utils import COURT_LENGTH_M, COURT_WIDTH_M, load_ball_log, log_quality_report


def make_player(player_id, team, x_m, y_m, speed=1.0):
    return {
        "frame_idx": 0,
        "player_id": player_id,
        "team": team,
        "x_m": x_m,
        "y_m": y_m,
        "speed_mps": speed,
    }


def make_frame(frame_idx, players, ball=None):
    record = {"frame_idx": frame_idx, "players": players}
    if ball is not None:
        record["ball"] = ball
    return record


BALL = {"x_m": 9.0, "y_m": 4.5, "speed_mps": 12.0}


class TestStateBuilder(unittest.TestCase):
    def test_base_state_dim_is_69(self):
        self.assertEqual(BASE_STATE_DIM, 69)
        self.assertEqual(STATE_DIM, 69)
        self.assertEqual(len(feature_names(include_ball=False)), 69)

    def test_ball_state_dim_is_76(self):
        self.assertEqual(STATE_DIM_WITH_BALL, 69 + BALL_FEATS)
        self.assertEqual(state_dim(True), 76)
        self.assertEqual(len(feature_names(include_ball=True)), 76)
        self.assertIn("ball_present", feature_names(True))

    def test_base_state_matches_legacy_layout(self):
        frame = make_frame(0, [make_player(1, "left", 3.0, 2.0)])
        state = build_state(frame, 0.5)
        self.assertEqual(state.shape, (69,))
        self.assertEqual(state.dtype, np.float32)
        # left p0 slot: present=1, x=3/18, y=2/9, speed=1/10
        self.assertAlmostEqual(float(state[0]), 1.0)
        self.assertAlmostEqual(float(state[1]), 3.0 / 18.0)
        self.assertAlmostEqual(float(state[2]), 2.0 / 9.0)
        self.assertAlmostEqual(float(state[3]), 0.1)
        # right team block (dims 34..67) untouched (players only on the left)
        self.assertEqual(float(state[34:68].sum()), 0.0)
        # rally_progress is the last base dim
        self.assertAlmostEqual(float(state[68]), 0.5)

    def test_ball_state_values(self):
        frame = make_frame(0, [make_player(1, "left", 9.0, 4.5)], ball=BALL)
        state = build_state(frame, 0.0, include_ball=True)
        self.assertEqual(state.shape, (76,))
        # ball block = dims 69..75
        ball_block = state[69:]
        self.assertEqual(float(ball_block[0]), 1.0)                       # present
        self.assertAlmostEqual(float(ball_block[1]), 0.5)                 # x 9/18
        self.assertAlmostEqual(float(ball_block[2]), 0.5)                 # y 4.5/9
        self.assertAlmostEqual(float(ball_block[3]), 12.0 / 30.0)         # speed
        # left player sits exactly under the ball -> dist 0; right team empty -> 0
        self.assertAlmostEqual(float(ball_block[4]), 0.0, places=5)
        self.assertAlmostEqual(float(ball_block[6]), 0.0, places=5)       # side = 0

    def test_ball_absent_block_is_zero(self):
        frame = make_frame(0, [make_player(1, "left", 1.0, 1.0)])
        state = build_state(frame, 0.0, include_ball=True)
        self.assertEqual(float(state[69:].sum()), 0.0)

    def test_ball_features_zero_variance_free(self):
        # ball position at court centre must not be clipped
        centre = {"x_m": COURT_LENGTH_M / 2, "y_m": COURT_WIDTH_M / 2, "speed_mps": 0.0}
        frame = make_frame(0, [make_player(1, "left", 1.0, 1.0)], ball=centre)
        state = build_state(frame, 0.0, include_ball=True)
        self.assertAlmostEqual(float(state[70]), 0.5)
        self.assertAlmostEqual(float(state[71]), 0.5)

    def test_build_rally_states_shapes(self):
        rally = [
            make_frame(i, [make_player(1, "left", 1.0 + i * 0.1, 2.0)], ball=BALL)
            for i in range(12)
        ]
        for include_ball in (False, True):
            states = build_rally_states(rally, include_ball=include_ball)
            self.assertEqual(states.shape, (12, state_dim(include_ball)))


class TestRallySegmentation(unittest.TestCase):
    def _rows(self, frame_indices):
        return [make_player(1, "left", 1.0, 1.0) | {"frame_idx": f} for f in frame_indices]

    def test_gap_splits_rallies(self):
        rows = self._rows(list(range(0, 20)) + list(range(100, 115)))
        rallies = segment_rallies(rows, max_frame_gap=15, min_rally_frames=10)
        self.assertEqual(len(rallies), 2)
        self.assertEqual([len(r) for r in rallies], [20, 15])

    def test_short_rallies_dropped(self):
        rows = self._rows([0, 1, 2, 50, 51, 52, 53])
        self.assertEqual(segment_rallies(rows, min_rally_frames=10), [])

    def test_ball_attached_to_frames(self):
        rows = self._rows(list(range(0, 12)))
        ball_by_frame = {i: {"x_m": 5.0, "y_m": 4.0, "speed_mps": 3.0} for i in range(0, 10)}
        rallies = segment_rallies(rows, ball_by_frame=ball_by_frame)
        self.assertEqual(len(rallies), 1)
        self.assertEqual(rallies[0][0]["ball"]["x_m"], 5.0)
        self.assertIsNone(rallies[0][11]["ball"])

    def test_game_state_labels(self):
        rally = [
            make_frame(0, [make_player(1, "left", 1.0, 1.0)]),
            make_frame(1, [make_player(1, "left", 1.0, 1.0)],
                       ball={"x_m": 2.0, "y_m": 2.0, "speed_mps": 0.2}),
            make_frame(2, [make_player(1, "left", 1.0, 1.0)],
                       ball={"x_m": 2.0, "y_m": 2.0, "speed_mps": 15.0}),
        ]
        self.assertEqual(assign_game_states(rally), [STATE_DEAD, STATE_SERVE, STATE_LIVE])


class TestBallLogLoading(unittest.TestCase):
    def test_list_format(self):
        import json
        import tempfile
        rows = [
            {"frame_idx": 0, "x_m": 1.0, "y_m": 2.0, "speed_mps": 3.0},
            {"frame_idx": 1, "x_m": None, "y_m": None, "speed_mps": 0.0},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(rows, f)
            path = f.name
        try:
            loaded = load_ball_log(path)
            self.assertEqual(set(loaded.keys()), {0})
            self.assertEqual(loaded[0]["x_m"], 1.0)
        finally:
            os.unlink(path)

    def test_dict_format_skips_nulls(self):
        import json
        import tempfile
        raw = {"0": {"x_m": 1.0, "y_m": 2.0, "speed_mps": 0.0}, "1": None}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(raw, f)
            path = f.name
        try:
            self.assertEqual(set(load_ball_log(path).keys()), {0})
        finally:
            os.unlink(path)


class TestLogQualityReport(unittest.TestCase):
    def test_flags_broken_homography(self):
        rows = [make_player(i, "left" if i % 2 else "right", -5.0, -20.0) for i in range(10)]
        report = log_quality_report(rows)
        self.assertLess(report["frac_both_in_court"], 0.5)
        self.assertIsNotNone(report["warning"])

    def test_passes_good_log(self):
        rng = np.random.default_rng(0)
        rows = [
            make_player(i, "left", float(rng.uniform(0, COURT_LENGTH_M)),
                        float(rng.uniform(0, COURT_WIDTH_M)))
            for i in range(20)
        ]
        report = log_quality_report(rows)
        self.assertEqual(report["frac_both_in_court"], 1.0)
        self.assertIsNone(report["warning"])


class TestDatasetAndModel(unittest.TestCase):
    def test_proxy_outcome_direction(self):
        # left team advanced +3m, right team retreated 1m
        first = [make_player(1, "left", 4.0, 2.0), make_player(7, "right", 14.0, 7.0)]
        last = [make_player(1, "left", 7.0, 2.0), make_player(7, "right", 13.0, 7.0)]
        rally = [make_frame(0, first), make_frame(1, last)]
        proxy = compute_proxy_outcome(rally)
        self.assertAlmostEqual(proxy["left"], (3.0 / 9.0), places=5)
        self.assertAlmostEqual(proxy["right"], (1.0 / 9.0), places=5)

    def test_action_inference_covers_all_teams(self):
        slow = make_frame(0, [make_player(1, "left", 2.0, 2.0, speed=0.1)])
        fast_left_near_net = make_frame(1, [make_player(1, "left", 11.0, 4.5, speed=8.0)])
        action = infer_action_from_transition(slow, fast_left_near_net)
        self.assertEqual(ACTIONS[action], "left_attack")
        self.assertEqual(NUM_ACTIONS, 6)

    def test_model_forward_shapes(self):
        model = EPVModel(state_dim=76, num_actions=6)
        states = torch_rand(5, 76)
        actions = torch_rand_int(5)
        out = model(states, actions)
        self.assertEqual(out.shape, (5,))


def torch_rand(*shape):
    import torch
    return torch.randn(*shape)


def torch_rand_int(n):
    import torch
    return torch.randint(0, 6, (n,))


if __name__ == "__main__":
    unittest.main()
