"""Smoke tests on synthetic data -- they need no match data:  python -m unittest discover tests"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import config, shape  # noqa: E402
from possession.deadball import dead_evidence, dead_probability  # noqa: E402
from possession.features import (ball_touches, control_vote, interpolate_ball,  # noqa: E402
                                 team_distances)
from possession.pipeline import log_emissions, predict  # noqa: E402
from possession.tagging import team_side  # noqa: E402
from possession.states import A, D, H, drop_short_runs, snap_handovers, viterbi  # noqa: E402
from games import GAMES, GAME, half_video_blocks  # noqa: E402


class StatesTest(unittest.TestCase):
    def test_viterbi_removes_flicker(self):
        em = np.log(np.array([[.9, .05, .05]] * 40 + [[.05, .9, .05]] * 40))
        em[20] = np.log([.05, .9, .05])               # one-frame glitch
        path = viterbi(em, 5.0, 5.0)
        self.assertTrue((path[:40] == H).all() and (path[40:] == A).all())

    def test_snap_moves_handover_to_new_teams_first_control(self):
        path = np.array([H] * 10 + [A] * 10)
        home = np.zeros(20, dtype=bool)
        away = np.zeros(20, dtype=bool)
        home[2:5] = True                                 # home's last touch at 4
        away[14:18] = True                               # away first controls at 14
        out = snap_handovers(path, home, away, max_shift=10)
        self.assertTrue((out[:14] == H).all() and (out[14:] == A).all())

    def test_snap_leaves_dead_ball_alone(self):
        path = np.array([H] * 5 + [D] * 5 + [A] * 5)
        none = np.zeros(15, dtype=bool)
        np.testing.assert_array_equal(snap_handovers(path, none, none, 10), path)

    def test_drop_short_runs(self):
        path = np.array([H] * 5 + [D] * 2 + [A] * 5 + [D] * 6 + [H] * 3 + [D] * 2)
        out = drop_short_runs(path, D, 3)
        self.assertTrue((out[5:7] == H).all())         # short run filled from before
        self.assertTrue((out[12:18] == D).all())       # long run kept
        self.assertTrue((out[-2:] == D).all())         # run at the end kept


class FeaturesTest(unittest.TestCase):
    def test_interpolation_respects_max_gap(self):
        ball = np.full((10, 2), np.nan, dtype=np.float32)
        ball[0], ball[3], ball[9] = (0, 0), (3, 0), (9, 0)
        filled, conf = interpolate_ball(ball, max_gap=3)
        self.assertAlmostEqual(float(filled[1, 0]), 1.0, places=5)
        self.assertTrue(np.isnan(filled[5, 0]))        # gap of 5 > 3
        self.assertEqual(float(conf[5]), 0.0)

    def test_team_distances(self):
        ball = np.array([[0, 0]], dtype=np.float32)
        players = np.array([[[1, 0], [5, 0], [np.nan, np.nan]]], dtype=np.float32)
        home, away = team_distances(ball, players, np.array([0, 1, 1]))
        self.assertEqual((float(home[0]), float(away[0])), (1.0, 5.0))

    def test_ball_touches_finds_the_kick_and_who_made_it(self):
        n = 120
        ball = np.zeros((n, 2), dtype=np.float32)
        ball[:60, 0] = 50.0                               # at a home player's feet
        ball[60:, 0] = 50.0 + np.arange(n - 60) * 0.6     # struck at frame 60: 15 m/s
        ball[:, 1] = 30.0
        d_home = np.where(np.arange(n) < 62, 0.5, 8.0)
        d_away = np.full(n, 10.0)
        touch = ball_touches(ball, d_home, d_away, jump=6.0, radius=2.0, lag=3)
        hits = np.flatnonzero(touch)
        self.assertEqual(len(hits), 1)
        self.assertLessEqual(abs(int(hits[0]) - 60), 2)
        self.assertEqual(int(touch[hits[0]]), 1)
        far = ball_touches(ball, np.full(n, 5.0), d_away, jump=6.0, radius=2.0, lag=3)
        self.assertFalse(far.any())                       # nobody within reach: no touch

    def test_control_vote_only_counts_uncontested_control(self):
        d_home = np.array([1.0, 1.0, 4.0, 4.0, np.nan, 1.0])
        d_away = np.array([5.0, 1.2, 1.0, 6.0, np.nan, 5.0])
        conf = np.array([1, 1, 1, 1, 1, 0.5])
        vote = control_vote(d_home, d_away, conf, radius=1.5, certainty=0.9)
        np.testing.assert_allclose(vote, [0.9, 0.5, 0.1, 0.5, 0.5, 0.7])


class DeadBallTest(unittest.TestCase):
    N = 400

    def rolling_ball(self):
        ball = np.full((self.N, 2), 34.0, dtype=np.float32)
        ball[:, 0] = 20 + np.arange(self.N) * 0.1        # 2.5 m/s, inside the pitch
        return ball

    def test_walking_players_mean_dead_running_players_alive(self):
        speed = np.full((self.N, 20), 3.0, dtype=np.float32)
        speed[200:] = 0.8
        p = dead_probability(self.rolling_ball(), speed)
        self.assertLess(p[100], 0.2)
        self.assertGreater(p[300], 0.6)

    def test_ball_out_is_latched_through_a_tracking_gap(self):
        ball = self.rolling_ball()
        ball[100:110] = (50, -3)                         # over the touchline
        ball[110:200] = np.nan                           # then lost
        p = dead_probability(ball)
        self.assertGreater(p[150], 0.9)
        self.assertLess(p[250], 0.5)                     # seen inside again: no longer latched

    def test_missing_ball_alone_is_not_dead(self):
        ball = self.rolling_ball()
        ball[100:300] = np.nan
        self.assertLess(dead_probability(ball)[200], 0.5)

    def test_set_piece_and_still_ball_add_up(self):
        ball = self.rolling_ball()
        ball[150:250] = (60, 34)                          # placed ball
        saved, config.W_SET_PIECE = config.W_SET_PIECE, 1.5   # off in config.py; test the term
        try:
            alone = dead_probability(ball)[200]
            both = dead_probability(ball, set_piece_starts=[200])[200]
        finally:
            config.W_SET_PIECE = saved
        self.assertGreater(both, alone)
        self.assertGreater(both, 0.8)
        self.assertEqual(set(dead_evidence(ball)), {"outside", "still", "set_piece", "missing"})


class TaggingTest(unittest.TestCase):
    @staticmethod
    def event(code, team, start, end):
        return {"code": code, "start_frame": start, "end_frame": end, "tags": {"Team": team}}

    def test_team_side_reads_the_tag_by_code(self):
        events = [
            self.event("BUILD UP", "Barcelona", 0, 10),                    # attacker has it
            self.event("DEFENDING IN MIDDLE THIRD", "Barcelona", 10, 20),  # defender: flip
            self.event("SET PIECES", "N/A", 20, 30),                       # throw-in: no side
            self.event("PROGRESSION", "Atletic Club", 30, 40),
            self.event("CREATING CHANCES", "barcelona", 35, 40),           # disagree: none
            self.event("GOALS", "Barcelona", 40, 50),                      # not a phase
        ]
        side = team_side(events, 50, "Barcelona", "Atletic Club")
        self.assertTrue((side[:10] == 1).all())
        self.assertTrue((side[10:20] == -1).all())
        self.assertTrue((side[20:30] == 0).all())
        self.assertTrue((side[30:35] == -1).all())
        self.assertTrue((side[35:] == 0).all())

    def test_tag_is_used_only_where_the_ball_says_nothing(self):
        d_home = np.array([1.0, 1.0])                  # home in control on both frames
        d_away = np.array([5.0, 5.0])
        confidence = np.array([1.0, 0.0])              # ...but the ball is only seen on the first
        side = np.array([-1, -1], dtype=np.int8)       # tags say away
        em = log_emissions(d_home, d_away, confidence, np.full(2, 0.1), side)
        self.assertGreater(em[0, H], em[0, A])         # seen ball wins over the tag
        self.assertGreater(em[1, A], em[1, H])         # blind frame follows the tag


class ShapeTest(unittest.TestCase):
    N = 200

    def players(self, home_attacks_right=True, home_spread=10.0, away_spread=4.0):
        """Home keeper in its box, away keeper in its box, outfielders around midfield."""
        rng = np.random.default_rng(0)
        p = np.full((self.N, 6, 2), np.nan, dtype=np.float32)
        left, right = 3.0, 102.0
        p[:, 0] = (left if home_attacks_right else right, 34)          # home keeper
        p[:, 3] = (right if home_attacks_right else left, 34)          # away keeper
        p[:, 1:3, 0] = 52 + rng.normal(0, home_spread, (self.N, 2))
        p[:, 1:3, 1] = 34 + rng.normal(0, home_spread, (self.N, 2))
        p[:, 4:6, 0] = 52 + rng.normal(0, away_spread, (self.N, 2))
        p[:, 4:6, 1] = 34 + rng.normal(0, away_spread, (self.N, 2))
        return p, np.array([0, 0, 0, 1, 1, 1])

    def test_attack_sign_from_where_the_keepers_stand(self):
        for right in (True, False):
            p, team = self.players(home_attacks_right=right)
            sign = shape.attack_sign(p, team, [(0, self.N)], 105.0)
            self.assertTrue((sign == (1 if right else -1)).all())

    def test_wider_team_reads_as_in_possession(self):
        p, team = self.players(home_spread=10.0, away_spread=3.0)
        sign = shape.attack_sign(p, team, [(0, self.N)], 105.0)
        feats = shape.features(p, team, sign, 105.0)
        self.assertGreater(np.nanmean(feats[:, 1]), 0)             # width points to home
        z = shape.shape_logit(feats, (0.0, 1.0, 0.0))
        self.assertGreater(z[100], 0)

    def test_shape_hint_only_where_the_ball_is_blind(self):
        d = np.array([1.0, 1.0])
        far = np.array([5.0, 5.0])
        confidence = np.array([1.0, 0.0])
        saved, config.SHAPE_WEIGHT = config.SHAPE_WEIGHT, 1.0
        try:
            with_hint = log_emissions(far, d, confidence, np.full(2, 0.1), None,
                                      np.array([9.0, 9.0]))
        finally:
            config.SHAPE_WEIGHT = saved
        self.assertGreater(with_hint[0, A], with_hint[0, H])          # seen ball: away controls
        self.assertGreater(with_hint[1, H], with_hint[1, A])          # blind: shape says home


class PipelineTest(unittest.TestCase):
    def test_labels_cover_every_frame_and_respect_graded_blocks(self):
        n = GAMES[GAME]["n_video_frames"]
        ball = np.full((n, 2), 34.0, dtype=np.float32)
        ball[:, 0] = 30 + (np.arange(n) % 500) * 0.08    # rolling at 2 m/s, never placed
        players = np.full((n, 2, 2), np.nan, dtype=np.float32)
        players[:, 0] = ball - (1, 0)                    # home slot next to the ball
        players[:, 1] = ball + (10, 0)
        labels = predict(ball, players, np.array([0, 1]))
        self.assertEqual(len(labels), n)
        (_, f1, l1), (_, f2, l2) = half_video_blocks(GAME)
        self.assertEqual(labels[0], "X")
        self.assertEqual(labels[l1 + 1], "X")
        self.assertEqual(labels[f1 + 100], "H")
        self.assertEqual(labels[f2 + 10], "D")        # 2nd half pre-kickoff
        self.assertEqual(labels[l2 + 1], "X")


if __name__ == "__main__":
    unittest.main()
