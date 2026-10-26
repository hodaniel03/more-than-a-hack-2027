"""Smart Tagging as evidence. Automatic, not ground truth: false positives and negatives included."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME  # noqa: E402
from starter.load_smart_tagging import load  # noqa: E402

SET_PIECES = "set pieces"


def set_piece_starts(game=GAME, data_dir=None):
    """Sorted video frames where a SET PIECES tag starts. Codes are matched case-insensitively."""
    return sorted(event["start_frame"] for event in load(game, data_dir)
                  if (event["code"] or "").strip().lower() == SET_PIECES)
