"""Is B better than A? A paired, game-by-game answer on the validation games:

    python validation/compare.py --b SNAP_MAX_S=0              # A = config.py as it stands
    python validation/compare.py --a W_SPEED=1.5 --b W_SPEED=0.5
    python validation/compare.py --b NEW_FEATURE=1 --split test # confirm once, on the locked games

Both sides run on exactly the same degraded inputs (same games, same seeds), so the per-game
difference is the effect of the change alone. Reported: the mean difference with its standard
error across games, and on how many games B won. A change is worth keeping when it wins on most
games by more than about two standard errors -- one big game is not a result.

New behaviour should sit behind a config switch, so that it can be compared this way.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from validation.run import SPLITS, evaluate, load_games, parse_overrides  # noqa: E402

METRICS = ("spellF1", "macroF1", "D_F1", "team_acc", "poss")


def per_game(rows):
    out = {}
    for r in rows:
        out.setdefault(r["game"], []).append(r)
    return {g: {k: float(np.mean([r[k] for r in rs])) for k in METRICS + ("gt_poss",)}
            for g, rs in out.items()}


def compare(games, a, b, modes=("atd",), tags=False):
    ra = per_game(evaluate(games, modes, with_baseline=False, tags=tags, **a))
    rb = per_game(evaluate(games, modes, with_baseline=False, tags=tags, **b))
    return ra, rb


def report(ra, rb, a, b):
    print("A: {}\nB: {}\n".format(a or "config.py", b or "config.py"))
    print("{:<14} {:>8} {:>8} {:>8}   {:>6} {:>6} {:>6}".format(
        "game", "A", "B", "B-A", "possA", "possB", "gt"))
    deltas = []
    for g in ra:
        d = rb[g]["spellF1"] - ra[g]["spellF1"]
        deltas.append(d)
        print("{:<14} {:>8.4f} {:>8.4f} {:>+8.4f}   {:>6.0f} {:>6.0f} {:>6.0f}".format(
            g, ra[g]["spellF1"], rb[g]["spellF1"], d, ra[g]["poss"], rb[g]["poss"],
            ra[g]["gt_poss"]))
    deltas = np.array(deltas)
    se = deltas.std(ddof=1) / np.sqrt(len(deltas)) if len(deltas) > 1 else float("nan")
    print("\nspellF1  A {:.4f}  B {:.4f}  B-A {:+.4f} +/- {:.4f} (SE)  B better on {}/{} games"
          .format(np.mean([ra[g]["spellF1"] for g in ra]), np.mean([rb[g]["spellF1"] for g in rb]),
                  deltas.mean(), se, int((deltas > 0).sum()), len(deltas)))
    for k in METRICS[1:4]:
        print("{:<8} A {:.4f}  B {:.4f}  B-A {:+.4f}".format(
            k, np.mean([ra[g][k] for g in ra]), np.mean([rb[g][k] for g in rb]),
            np.mean([rb[g][k] - ra[g][k] for g in ra])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--b", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--games", nargs="*", metavar="PREFIX")
    parser.add_argument("--modes", nargs="*", default=["atd"])
    parser.add_argument("--tags", action="store_true", help="synthetic Smart Tagging on both sides")
    args = parser.parse_args()
    a, b = parse_overrides(args.a), parse_overrides(args.b)
    games = load_games(args.games, args.split)
    report(*compare(games, a, b, tuple(args.modes), args.tags), a, b)
