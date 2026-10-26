"""The validation tooling on synthetic data -- no download needed:  python -m unittest discover tests"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import degrade  # noqa: E402
from validation.metrica import ground_truth  # noqa: E402


def ev(team, kind, start, end=None, subtype="", period=1):
    return {"team": team, "type": kind, "subtype": subtype, "period": period,
            "start": start, "end": start if end is None else end}


class GroundTruthTest(unittest.TestCase):
    def test_possession_stoppage_and_restart(self):
        events = [
            ev("H", "SET PIECE", 10, subtype="KICK OFF"),
            ev("H", "PASS", 10, 20),
            ev("A", "CHALLENGE", 25, subtype="AERIAL-WON"),   # a won duel is not control
            ev("H", "PASS", 30, 40),
            ev("H", "BALL LOST", 40, subtype="INTERCEPTION"),
            ev("A", "RECOVERY", 42, subtype="INTERCEPTION"),
            ev("A", "BALL OUT", 50, 60),                     # crosses the line at 60
            ev("A", "CARD", 65, subtype="YELLOW"),           # off the ball, ignored
            ev("H", "SET PIECE", 70, subtype="THROW IN"),
            ev("H", "PASS", 70, 80),
        ]
        gt = ground_truth(events, 100)
        self.assertTrue((gt[:10] == "X").all())             # before kick-off
        self.assertTrue((gt[10:42] == "H").all())           # duel at 25 changes nothing
        self.assertTrue((gt[42:60] == "A").all())           # recovery hands it over
        self.assertTrue((gt[60:70] == "D").all())           # out until the throw-in
        self.assertTrue((gt[70:81] == "H").all())
        self.assertTrue((gt[81:] == "X").all())             # after the last event

    def test_foul_without_logged_stop_is_dead_from_the_foul(self):
        events = [
            ev("H", "SET PIECE", 0, subtype="KICK OFF"),
            ev("H", "PASS", 0, 10),
            ev("A", "CHALLENGE", 15, subtype="TACKLE-FAULT-LOST"),
            ev("H", "SET PIECE", 40, subtype="FREE KICK"),
            ev("H", "PASS", 40, 50),
        ]
        gt = ground_truth(events, 60)
        self.assertTrue((gt[15:40] == "D").all())
        self.assertTrue((gt[40:51] == "H").all())


class DegradeTest(unittest.TestCase):
    def test_blackouts_reach_target_and_are_seeded(self):
        ball = np.zeros((50000, 2), dtype=np.float32)
        alive = np.ones(50000, dtype=bool)
        a = degrade.ball_blackouts(ball, alive, np.random.default_rng(0), target=0.25)
        b = degrade.ball_blackouts(ball, alive, np.random.default_rng(0), target=0.25)
        missing = np.isnan(a[:, 0]).mean()
        self.assertTrue(0.25 <= missing < 0.35)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))

    def test_player_dropouts_share(self):
        players = np.zeros((40000, 10, 2), dtype=np.float32)
        speed = np.zeros((40000, 10), dtype=np.float32)
        p, s = degrade.player_dropouts(players, speed, np.random.default_rng(1), share=0.14)
        self.assertAlmostEqual(float(np.isnan(s).mean()), 0.14, delta=0.04)
        np.testing.assert_array_equal(np.isnan(p[:, :, 0]), np.isnan(s))


if __name__ == "__main__":
    unittest.main()
