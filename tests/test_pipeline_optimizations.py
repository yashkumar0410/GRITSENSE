"""Regression tests for the pipeline speed-ups (CPU only, no model downloads).

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sample_data
from analytics import relations
from analytics.commentary import CommentaryGenerator
from analytics.epv import PhysicsEPV
from analytics.nlp import build_backend
from analytics.insights import FrameInsight, Player
from analytics.relations import Matchup


def _player(pid, team, x, y):
    return Player(
        id=pid, team=team, x=x, y=y, image_x=100.0 + pid, image_y=200.0 + pid,
        velocity=(0.3, -0.2), jump_m=0.05 * pid, arms_up=pid % 2 == 0,
    )


class SampleDataCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.ball = base / "ball_clean.csv"
        self.players = base / "players.json"
        self.court = base / "annotations.json"

        with self.ball.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "X", "Y", "Visibility", "Confidence"])
            writer.writerow([0, 100, 200, 1, 0.9])
            writer.writerow([1, 110, 210, 1, 0.8])
            writer.writerow([2, 120, 220, 1, 0.7])

        self.players.write_text(json.dumps({
            "0": [{"id": 1, "image_x": 10, "image_y": 20, "team": 0}],
            "1": [{"id": 2, "image_x": 30, "image_y": 40, "team": 1}],
            "2": [{"id": 3, "image_x": 50, "image_y": 60, "team": 1}],
        }), encoding="utf-8")
        self.court.write_text(json.dumps({"points": [[0, 0], [1, 0], [1, 1], [0, 1]]}), encoding="utf-8")

        self._saved = (sample_data.BALL_FILE, sample_data.PLAYER_FILE, sample_data.COURT_FILE)
        sample_data.BALL_FILE = self.ball
        sample_data.PLAYER_FILE = self.players
        sample_data.COURT_FILE = self.court
        sample_data.clear_cache()

    def tearDown(self):
        sample_data.BALL_FILE, sample_data.PLAYER_FILE, sample_data.COURT_FILE = self._saved
        sample_data.clear_cache()
        self.tmp.cleanup()

    def test_cached_reads_match_and_reload_after_change(self):
        first = sample_data.create_sample_data(1)
        second = sample_data.create_sample_data(1)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first[1]["x"], 110.0)
        self.assertEqual(len(first[0]), 1)

        # A rewritten input must invalidate the cache (mtime/size change).
        self.players.write_text(json.dumps({
            "1": [{"id": 9, "image_x": 1, "image_y": 2, "team": 1}],
        }), encoding="utf-8")
        reloaded = sample_data.create_sample_data(1)
        self.assertEqual(reloaded[0][0]["id"], 9)

    def test_court_copy_is_not_shared(self):
        _, _, court = sample_data.create_sample_data(1)
        court["points"].append(["mutated"])
        _, _, fresh = sample_data.create_sample_data(1)
        self.assertNotIn(["mutated"], fresh["points"])


class EpvReuseTests(unittest.TestCase):
    def setUp(self):
        self.defenders = [_player(i + 1, 1, 1.5 + 2.2 * i, 1.0 + 1.1 * i) for i in range(6)]

    def test_values_paths_match_full_surface(self):
        epv = PhysicsEPV()
        contact, side, tth = (7.3, 4.4), 1, 0.8
        surface = epv.surface(contact, self.defenders, side, time_to_hit=tth)

        fresh = PhysicsEPV()
        self.assertTrue(np.array_equal(
            epv.surface_values(contact, self.defenders, side, time_to_hit=tth),
            surface.values,
        ))

        loo = epv.leave_one_out_values(contact, self.defenders, side, tth)
        self.assertEqual(len(loo), len(self.defenders))
        for i, values in enumerate(loo):
            others = self.defenders[:i] + self.defenders[i + 1:]
            expected = fresh.surface(contact, others, side, time_to_hit=tth)
            self.assertTrue(np.array_equal(values, expected.values), f"defender {i}")

    def test_matchups_use_leave_one_out(self):
        epv = PhysicsEPV()
        surface = epv.surface((7.3, 4.4), self.defenders, 1, time_to_hit=0.0)
        matchups = relations.key_matchups(epv, surface, self.defenders, 7)
        self.assertEqual(len(matchups), len(self.defenders))
        self.assertAlmostEqual(sum(m.share for m in matchups), 1.0, places=9)

    def test_empty_and_single_defender(self):
        epv = PhysicsEPV()
        surface = epv.surface((7.3, 4.4), [], 1, time_to_hit=0.0)
        self.assertEqual(relations.key_matchups(epv, surface, [], 7), [])
        loo = epv.leave_one_out_values((7.3, 4.4), self.defenders[:1], 1, 0.0)
        expected = PhysicsEPV().surface((7.3, 4.4), [], 1, time_to_hit=0.0)
        self.assertTrue(np.array_equal(loo[0], expected.values))


class _FakeBackend:
    name = "FAKE"

    def generate(self, kind, facts, recent):
        player = str(facts["key_player"]).lstrip("#")
        return f"#{player} attacks Zone {facts['target_zone']} now"


def _insight(frame, *, possession=False, move_changed=False, attack=True):
    return FrameInsight(
        frame=frame, attack_team=0, defend_team=1, surface=None, target=(12.0, 4.5),
        target_zone=2, epv=0.42, epv_mean=0.21, gap_m=2.3, block_prob=0.12, risk=0.08,
        move="SPIKE · CROSS-COURT", move_detail="ball 0.9 s vs defender 1.2 s", key_player=7,
        matchups=[Matchup(7, 3, 0.4, 0.4)], team_shape={}, possession_changed=possession,
        move_changed=move_changed, flight_time=0.9, defender_time=1.2, p_in=0.92,
        p_block=0.12, attack_chance=attack,
    )


class DeferredCommentaryTests(unittest.TestCase):
    def _sequence(self):
        frames = []
        for f in range(400):
            possession = f in (5, 130)
            move_changed = f in (40, 120, 250)
            frames.append(_insight(f, possession=possession, move_changed=move_changed,
                                   attack=(f > 20)))
        return frames

    def test_deferred_matches_synchronous(self):
        sync = CommentaryGenerator({0: "BRA", 1: "JPN"}, 30.0, backend=_FakeBackend())
        deferred = CommentaryGenerator({0: "BRA", 1: "JPN"}, 30.0, backend=_FakeBackend())
        for ins in self._sequence():
            sync.update(ins)
            deferred.plan(ins)
        deferred.finish()

        self.assertEqual(len(sync.history), len(deferred.history))
        for a, b in zip(sync.history, deferred.history):
            self.assertEqual((a.frame, a.kind, a.text, a.source), (b.frame, b.kind, b.text, b.source))

    def test_template_fallback_and_no_worker(self):
        gen = CommentaryGenerator({0: "BRA", 1: "JPN"}, 30.0, backend=None)
        line = None
        for ins in self._sequence():
            line = gen.plan(ins) or line
        gen.finish()
        self.assertIsNotNone(line)
        self.assertEqual(line.source, "TEMPLATE")
        self.assertTrue(line.text)


class _RecordingBackend:
    """Deterministic backend that remembers the recent lines it was shown."""

    name = "RECORD"

    def __init__(self):
        self.calls: list[list[str]] = []

    def generate(self, kind, facts, recent):
        self.calls.append(list(recent))
        player = str(facts["key_player"]).lstrip("#")
        return f"#{player} attacks Zone {facts['target_zone']} now"


def _sequence():
    frames = []
    for f in range(400):
        possession = f in (5, 130)
        move_changed = f in (40, 120, 250)
        frames.append(_insight(f, possession=possession, move_changed=move_changed,
                               attack=(f > 20)))
    return frames


class SerializedCommentaryTests(unittest.TestCase):
    """plan(defer=True): nothing is generated until finish(), and the result
    matches the synchronous path line for line."""

    def test_deferred_generates_only_at_finish(self):
        backend = _RecordingBackend()
        gen = CommentaryGenerator({0: "BRA", 1: "JPN"}, 30.0, backend=backend)
        for ins in _sequence():
            gen.plan(ins, defer=True)
        self.assertEqual(backend.calls, [])   # no generation while analysing
        self.assertIsNone(gen._worker)        # and no background thread
        gen.finish()
        self.assertTrue(gen.history)
        self.assertTrue(all(line.text for line in gen.history))
        self.assertTrue(all(line.source == "RECORD" for line in gen.history))

    def test_deferred_matches_synchronous(self):
        sync_backend, deferred_backend = _RecordingBackend(), _RecordingBackend()
        sync = CommentaryGenerator({0: "BRA", 1: "JPN"}, 30.0, backend=sync_backend)
        deferred = CommentaryGenerator({0: "BRA", 1: "JPN"}, 30.0, backend=deferred_backend)
        for ins in _sequence():
            sync.update(ins)
            deferred.plan(ins, defer=True)
        deferred.finish()

        self.assertEqual(len(sync.history), len(deferred.history))
        for a, b in zip(sync.history, deferred.history):
            self.assertEqual((a.frame, a.kind, a.text, a.source, a.timestamp),
                             (b.frame, b.kind, b.text, b.source, b.timestamp))
        # Same prompts too: the recent lines a line sees are the lines before it.
        self.assertEqual(sync_backend.calls, deferred_backend.calls)

    def test_deferred_falls_back_to_template(self):
        class _Broken(_RecordingBackend):
            def generate(self, kind, facts, recent):
                raise RuntimeError("boom")

        gen = CommentaryGenerator({0: "BRA", 1: "JPN"}, 30.0, backend=_Broken())
        for ins in _sequence():
            gen.plan(ins, defer=True)
        gen.finish()
        self.assertTrue(gen.history)
        self.assertTrue(all(line.source == "TEMPLATE" and line.text for line in gen.history))


class QuantOptionTests(unittest.TestCase):
    def test_unknown_quant_rejected(self):
        with self.assertRaises(ValueError):
            build_backend("local", device="cpu", quant="8bit")

    def test_4bit_without_bitsandbytes_is_reported(self):
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            with self.assertRaises(RuntimeError):
                build_backend("local", device="cuda", quant="4bit")
        else:
            self.skipTest("bitsandbytes is installed")


class LlmPreloadTests(unittest.TestCase):
    """--preload-llm: the dashboard backend is built on a worker thread and
    handed over; a failed preload falls back to the dashboard's own load."""

    def test_preload_hands_the_backend_over(self):
        from unittest import mock

        import master_main
        sentinel = object()
        with mock.patch("analytics.nlp.build_backend", return_value=sentinel) as build:
            handle = master_main._LlmPreload("cpu", quant="4bit")
            self.assertIs(handle.take(), sentinel)
        build.assert_called_once_with("local", device="cpu", quant="4bit")
        self.assertGreaterEqual(handle.elapsed, 0.0)

    def test_failed_preload_falls_back(self):
        from unittest import mock

        import master_main
        with mock.patch("analytics.nlp.build_backend", side_effect=RuntimeError("no bitsandbytes")):
            handle = master_main._LlmPreload("cpu")
            self.assertIsNone(handle.take())
        self.assertIsInstance(handle.error, RuntimeError)


if __name__ == "__main__":
    unittest.main()
