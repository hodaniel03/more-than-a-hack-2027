"""Smart Tagging as evidence. Automatic, not ground truth: false positives and negatives included."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, require, source_frames  # noqa: E402
from starter.baseline import team_names  # noqa: E402
from starter.load_smart_tagging import load  # noqa: E402

SET_PIECES = "set pieces"

# What the Team tag means depends on the code. Attacking phases are tagged with the team on the
# ball; defending phases with the team *without* it (a PROGRESSION by one side comes paired with a
# DEFENDING IN MIDDLE THIRD by the other, over the same interval). SET PIECES names the team taking
# the restart, which is the team in possession once play resumes. GOALS is left out: the ball is
# dead after it and the other side kicks off.
POSSESSOR_CODES = {
    "build up", "progression", "creating chances", "attacking transition", "long ball",
    "ball in final third", "ball in the box", "players in the box", SET_PIECES,
}
DEFENDER_CODES = {
    "defending in attacking third", "defending in middle third", "defending in defensive third",
    "defensive transition",
}

HOME, NONE, AWAY = 1, 0, -1


def set_piece_starts(game=GAME, data_dir=None):
    """Sorted video frames where a SET PIECES tag starts. Codes are matched case-insensitively."""
    return sorted(event["start_frame"] for event in load(game, data_dir)
                  if (event["code"] or "").strip().lower() == SET_PIECES)


def team_side(events, n_frames, home, away):
    """(N,) int8: HOME or AWAY where the tags covering a frame agree on who has the ball, NONE where
    no tag names a side (N/A teams included -- every throw-in is one) or where they disagree."""
    names = {home.strip().lower(): HOME, away.strip().lower(): AWAY}
    home_votes = np.zeros(n_frames, dtype=np.int16)
    away_votes = np.zeros(n_frames, dtype=np.int16)
    for event in events:
        code = (event["code"] or "").strip().lower()
        side = names.get((event["tags"].get("Team") or "").strip().lower())
        if side is None or code not in POSSESSOR_CODES | DEFENDER_CODES:
            continue
        if code in DEFENDER_CODES:
            side = -side
        first = max(0, event["start_frame"])
        last = min(n_frames, event["end_frame"])
        (home_votes if side == HOME else away_votes)[first:last] += 1

    out = np.full(n_frames, NONE, dtype=np.int8)
    out[(home_votes > 0) & (away_votes == 0)] = HOME
    out[(away_votes > 0) & (home_votes == 0)] = AWAY
    return out


def tag_side(game=GAME, data_dir=None):
    """team_side() for the tactical video, with club names read from the ATD metadata."""
    meta_path = require(["atd_meta"], game, data_dir)[0]
    home, away = team_names(meta_path)
    return team_side(load(game, data_dir), source_frames("tactical", game), home, away)
