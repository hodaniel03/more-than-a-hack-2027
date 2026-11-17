"""Which true possessions does the pipeline miss, and why? By possession length:

    python validation/diagnose.py                   # dev games, atd inputs, seed 0
    python validation/diagnose.py --set MIN_CONTROL_S=0.1

A true possession counts as found when a predicted spell of the same team overlaps it with
IoU >= 0.5 (the scorer's headline threshold). A missed one gets the first reason that applies:

  no ball      the ball is seen on under a quarter of its frames
  no control   the ball is seen, but no frame has the right team in uncontested control
  contested    the right team is within reach on more frames than it is alone there
  skipped      there is control evidence, but no predicted spell of that team overlaps it
  boundary     a predicted spell of that team overlaps it, with IoU under 0.5
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import config  # noqa: E402
from possession.features import interpolate_ball, team_distances  # noqa: E402
from possession.pipeline import predict_blocks  # noqa: E402
from validation.metrica import PITCH  # noqa: E402
from validation.run import (SPLITS, blocks_of, inputs, load_games, params,  # noqa: E402
                            parse_overrides)

BUCKETS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, None))
REASONS = ("found", "no ball", "no control", "contested", "skipped", "boundary")


def spells(labels):
    """[(start, stop, team)] maximal runs of H or A."""
    out = []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(labels)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if labels[a] in ("H", "A"):
            out.append((int(a), int(b), labels[a]))
    return out


def best_iou(spell, predicted, starts):
    s, e, team = spell
    best = 0.0
    k = max(0, np.searchsorted(starts, s, side="right") - 1)
    while k < len(predicted) and predicted[k][0] < e:
        ps, pe, pt = predicted[k]
        if pt == team and pe > s:
            best = max(best, (min(e, pe) - max(s, ps)) / (max(e, pe) - min(s, ps)))
        k += 1
    return best


def diagnose(games, seed=0):
    counts = {b: dict.fromkeys(REASONS, 0) for b in BUCKETS}
    for data in games.values():
        ball, players, speed = inputs(data, "atd", seed)
        pitch = tuple(data["pitch"]) if "pitch" in data else PITCH
        pred = predict_blocks(ball, players, data["team"],
                              blocks_of(data["gt"], data.get("periods")), pitch, speed)
        masked = np.where(np.isnan(speed)[:, :, None], np.nan, players)
        filled, confidence = interpolate_ball(ball, config.MAX_BALL_GAP)
        d_home, d_away = team_distances(filled, masked, data["team"])
        r = config.CONTROL_RADIUS
        with np.errstate(invalid="ignore"):
            near = {"H": d_home <= r, "A": d_away <= r}
        seen = confidence > 0

        predicted = spells(pred)
        starts = np.array([p[0] for p in predicted])
        for spell in spells(data["gt"]):
            s, e, team = spell
            other = "A" if team == "H" else "H"
            length = (e - s) / 25.0
            bucket = next(b for b in BUCKETS if length >= b[0] and (b[1] is None or length < b[1]))
            iou = best_iou(spell, predicted, starts)
            own, opp, vis = near[team][s:e], near[other][s:e], seen[s:e]
            alone = (own & ~opp & vis).sum()
            if iou >= 0.5:
                reason = "found"
            elif vis.mean() < 0.25:
                reason = "no ball"
            elif (own & vis).sum() == 0:
                reason = "no control"
            elif (own & opp & vis).sum() > alone:
                reason = "contested"
            elif iou == 0.0:
                reason = "skipped"
            else:
                reason = "boundary"
            counts[bucket][reason] += 1
    return counts


def print_counts(counts):
    total = sum(sum(c.values()) for c in counts.values())
    print("{:<9} {:>6} {:>6}  ".format("length", "n", "share") + " ".join(
        "{:>10}".format(r) for r in REASONS))
    for b, c in counts.items():
        n = sum(c.values())
        label = "{}-{} s".format(b[0], b[1]) if b[1] else "{}+ s".format(b[0])
        print("{:<9} {:>6} {:>5.0f}%  ".format(label, n, 100.0 * n / total) + " ".join(
            "{:>9.0f}%".format(100.0 * c[r] / max(n, 1)) for r in REASONS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--games", nargs="*", metavar="PREFIX")
    args = parser.parse_args()
    games = load_games(args.games, args.split)
    with params(**parse_overrides(args.set)):
        print_counts(diagnose(games))
