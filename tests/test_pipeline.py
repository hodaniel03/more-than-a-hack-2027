"""Smoke tests on synthetic data -- they need no match data:  python -m unittest discover tests"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession.deadball import dead_evidence, dead_probability  # noqa: E402
from possession.features import control_vote, interpolate_ball, team_distances  # noqa: E402
from possession.pipeline import log_emissions, predict  # noqa: E402
from possession.tagging import team_side  # noqa: E402
from possession.states import A, D, H, drop_short_runs, viterbi  # noqa: E402
from games import GAMES, GAME, half_video_blocks  # noqa: E402


class StatesTest(unittest.TestCase):
    def test_viterbi_removes_flicker(self):
        em = np.log(np.array([[.9, .05, .05]] * 40 + [[.05, .9, .05]] * 40))
        em[20] = np.log([.05, .9, .05])               # one-frame glitch
        path = viterbi(em, 5.0, 5.0)
        self.assertTrue((path[:40] == H).all() and (path[40:] == A).all())

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
        alone = dead_probability(ball)[200]
        both = dead_probability(ball, set_piece_starts=[200])[200]
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
