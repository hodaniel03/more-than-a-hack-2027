"""Tests for the learned evidence (possession/learned.py). No match data and no scikit-learn needed:

    python -m unittest discover tests
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import config, learned  # noqa: E402
from possession.pipeline import predict_blocks  # noqa: E402
from possession.states import viterbi, viterbi_fast  # noqa: E402


def tiny_model():
    """One tree: x0 <= 1 -> -2, else (x1 <= 0 -> 0.5, else 3). Missing x0 goes right."""
    return dict(feature=np.array([0, 0, 1, 0, 0]), threshold=np.array([1.0, 0, 0.0, 0, 0]),
                left=np.array([1, 0, 3, 0, 0]), right=np.array([2, 0, 4, 0, 0]),
                value=np.array([0.0, -2.0, 0.0, 0.5, 3.0]),
                missing_left=np.array([False, False, True, False, False]),
                is_leaf=np.array([False, True, False, True, True]),
                offsets=np.array([0]), baseline=np.float64(0.25))


class TreeTest(unittest.TestCase):
    def test_predict_logit_follows_thresholds_and_missing_values(self):
        X = np.array([[0.0, 5.0], [2.0, -1.0], [2.0, 1.0], [np.nan, 1.0], [2.0, np.nan]])
        np.testing.assert_allclose(learned.predict_logit(tiny_model(), X),
                                   [-1.75, 0.75, 3.25, 3.25, 0.75])

    def test_stored_model_loads_and_names_match_the_features(self):
        if not os.path.isfile(learned.MODEL_PATH):
            self.skipTest("models/learned.npz not built")
        models = learned.load_model()
        self.assertFalse([n for n in models["dead"]["names"]
                          if n.startswith(learned.DEAD_EXCLUDED)])
        for kind in ("dead", "team"):
            self.assertEqual(int(models[kind]["feature"].max()) < len(models[kind]["names"]), True)


class FeatureTest(unittest.TestCase):
    def test_window_mean_ignores_nan_and_respects_the_window(self):
        x = np.array([1.0, np.nan, 3.0, 5.0])
        np.testing.assert_allclose(learned.window_mean(x, -1, 0), [1.0, 1.0, 3.0, 4.0])
        self.assertTrue(np.isnan(learned.window_mean(np.array([np.nan, 1.0]), 0, 0)[0]))

    def test_window_min_and_row_percentile(self):
        x = np.array([5.0, np.nan, 3.0, 7.0, 1.0, 9.0])
        np.testing.assert_allclose(learned.window_min(x, 1, 3), [3.0, 1.0, 1.0, 1.0, 9.0, np.nan])
        np.testing.assert_allclose(learned.window_min(x, -3, -1), [np.nan, 5.0, 5.0, 3.0, 3.0, 1.0])
        rows = np.array([[1.0, np.nan, 3.0, 2.0, 4.0], [np.nan] * 5], dtype=np.float32)
        np.testing.assert_allclose(learned.row_percentile(rows, 0.75), [3.25, np.nan])

    def test_since_and_until(self):
        flag = np.array([False, True, False, False, True])
        np.testing.assert_array_equal(learned.since_true(flag)[1:], [0, 1, 2, 0])
        np.testing.assert_array_equal(learned.until_true(flag)[:4], [1, 0, 2, 1])

    def test_swapping_teams_twice_is_the_identity_and_once_flips(self):
        names = ["d_a", "d_h", "excl", "u_conf"]
        X = np.array([[1.0, 2.0, 0.5, 0.9]])
        once = learned.swap_teams(names, X)
        np.testing.assert_allclose(once, [[2.0, 1.0, -0.5, 0.9]])
        np.testing.assert_allclose(learned.swap_teams(names, once), X)

    def test_team_features_flip_when_the_teams_are_swapped(self):
        rng = np.random.default_rng(0)
        n = 400
        ball = np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0).astype(np.float32) + (50, 30)
        ball[100:160] = np.nan
        players = (rng.uniform(0, 1, (n, 16, 2)) * (105, 68)).astype(np.float32)
        players[:, 0] = (2, 34)                        # a keeper each, so the ends are known
        players[:, 8] = (103, 34)
        team = np.array([0] * 8 + [1] * 8)
        speed = rng.uniform(0, 6, (n, 16)).astype(np.float32)
        blocks = [(0, n)]
        names, X = learned.matrix(learned.team_features(ball, players, team, speed, (105, 68),
                                                        blocks))
        _, Y = learned.matrix(learned.team_features(ball, players, 1 - team, speed, (105, 68),
                                                    blocks))
        np.testing.assert_allclose(learned.swap_teams(names, X), Y, atol=1e-4, equal_nan=True)


class DecoderTest(unittest.TestCase):
    def test_fast_viterbi_matches_the_reference(self):
        em = np.random.default_rng(1).normal(0, 3, (3000, 3))
        for costs in ((8.0, 8.0), (2.5, 6.0), (0.0, 0.0)):
            np.testing.assert_array_equal(viterbi(em, *costs), viterbi_fast(em, *costs))


class PipelineTest(unittest.TestCase):
    def test_learned_pipeline_labels_every_frame_in_a_block_and_is_symmetric(self):
        if not os.path.isfile(learned.MODEL_PATH):
            self.skipTest("models/learned.npz not built")
        rng = np.random.default_rng(2)
        n = 1500
        team = np.array([0] * 11 + [1] * 11)
        players = (rng.uniform(0.1, 0.9, (1, 22, 2)) * (105, 68)
                   + np.cumsum(rng.normal(0, 0.15, (n, 22, 2)), axis=0)).astype(np.float32)
        speed = rng.uniform(1, 6, (n, 22)).astype(np.float32)
        ball = players[:, 3].copy()                    # a home player carries the ball
        ball[800:] = players[800:, 15]                 # then an away player does
        blocks = [(100, n)]
        saved = config.MODEL
        config.MODEL = "learned"
        try:
            labels = predict_blocks(ball, players, team, blocks, (105.0, 68.0), speed)
            mirrored = predict_blocks(ball, players, 1 - team, blocks, (105.0, 68.0), speed)
        finally:
            config.MODEL = saved
        self.assertTrue((labels[:100] == "X").all())
        self.assertFalse((labels[100:] == "X").any())
        self.assertGreater((labels[200:700] == "H").mean(), 0.9)
        self.assertGreater((labels[900:1400] == "A").mean(), 0.9)
        flip = {"H": "A", "A": "H", "D": "D", "X": "X"}
        agree = np.mean([flip[a] == b for a, b in zip(labels, mirrored)])
        self.assertGreater(agree, 0.98)


if __name__ == "__main__":
    unittest.main()
