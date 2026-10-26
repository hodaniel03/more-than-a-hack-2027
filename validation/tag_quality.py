"""How good is BAR-ATH's Smart Tagging? Estimated without any BAR-ATH ground truth:

    python validation/tag_quality.py

The tags are compared with our own prediction on frames where that prediction is at its most
reliable (ball seen, alive). The prediction has errors too, so the raw agreement A understates or
overstates the tags; with q, the prediction's team accuracy on the same kind of frames measured
on the dev validation games, and assuming the two err independently,

    A = t q + (1 - t)(1 - q)   =>   t = (A - (1 - q)) / (2 q - 1).

Independence is the weak point -- both are likely to struggle on the same chaotic frames -- so
treat t as a rough figure. For SET PIECES the script reports where our predicted restarts fall
relative to each tag's start, and how many predicted stoppages have a tag near them.

The prediction used here ignores the tags entirely (no tag hint, W_SET_PIECE = 0); otherwise it
would partly agree with them because it was told to.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks  # noqa: E402
from possession import config  # noqa: E402
from possession.features import interpolate_ball  # noqa: E402
from possession.pipeline import predict, predict_blocks  # noqa: E402
from possession.tagging import set_piece_starts, tag_side  # noqa: E402
from possession.tracking import load_tracking  # noqa: E402
from validation.metrica import PITCH  # noqa: E402
from validation.run import blocks_of, inputs, load_games, params  # noqa: E402


def prediction_accuracy_on_seen(games):
    """q: share of alive, ball-seen frames (both by truth and by prediction) with the right team."""
    right = total = 0
    for data in games.values():
        ball, players, speed = inputs(data, "atd", 0)
        pitch = tuple(data["pitch"]) if "pitch" in data else PITCH
        pred = predict_blocks(ball, players, data["team"],
                              blocks_of(data["gt"], data.get("periods")), pitch, speed)
        _, confidence = interpolate_ball(ball, config.MAX_BALL_GAP)
        mask = (confidence > 0) & np.isin(data["gt"], ("H", "A")) & np.isin(pred, ("H", "A"))
        right += (pred[mask] == data["gt"][mask]).sum()
        total += mask.sum()
    return right / total


def restarts(labels):
    """Frames where a predicted stoppage ends (D followed by H or A)."""
    return np.flatnonzero((labels[:-1] == "D") & np.isin(labels[1:], ("H", "A"))) + 1


if __name__ == "__main__":
    q = prediction_accuracy_on_seen(load_games(split="dev"))

    ball, players, team, speed = load_tracking()
    side = tag_side()
    starts = np.array(set_piece_starts())
    with params(W_SET_PIECE=0.0):
        labels = predict(ball, players, team, "BAR-ATH", speed)
    _, confidence = interpolate_ball(ball, config.MAX_BALL_GAP)
    graded = np.zeros(len(ball), dtype=bool)
    for _, first, last in half_video_blocks("BAR-ATH"):
        graded[first:last + 1] = True
    alive = np.isin(labels, ("H", "A")) & graded
    mask = alive & (confidence > 0) & (side != 0)
    agree = float(((side[mask] > 0) == (labels[mask] == "H")).mean())
    t = (agree - (1 - q)) / (2 * q - 1)
    print("team tags (phases)")
    print("  cover {:.0f}% of predicted-alive frames".format(100 * (side[alive] != 0).mean()))
    print("  agree with our prediction on {:.1f}% of ball-seen alive frames (A)".format(100 * agree))
    print("  our prediction is right on {:.1f}% of such frames in validation (q)".format(100 * q))
    print("  => tags right on about {:.0f}% of the frames they cover (t)".format(100 * t))

    ends = restarts(labels)
    offsets = []
    for s in starts:
        near = ends[(ends > s - 40 * 25) & (ends < s + 40 * 25)]
        if len(near):
            offsets.append((near[np.argmin(np.abs(near - s))] - s) / 25.0)
    offsets = np.array(offsets)
    stops_with_tag = np.mean([np.any(np.abs(starts - e) < 30 * 25) for e in ends
                              if graded[e]])
    print("\nSET PIECES ({} tags, {} predicted restarts)".format(len(starts), len(ends)))
    print("  predicted restart minus tag start: median {:+.1f} s, middle half {:+.1f} .. {:+.1f} s,"
          " within +/-10 s: {:.0f}% of tags".format(
              np.median(offsets), np.percentile(offsets, 25), np.percentile(offsets, 75),
              100 * np.mean(np.abs(offsets) <= 10) * len(offsets) / len(starts)))
    print("  predicted stoppages with a tag within 30 s of their restart: {:.0f}%".format(
        100 * stops_with_tag))
