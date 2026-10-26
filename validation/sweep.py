"""Grid search over config values on the sample games, with leave-one-game-out:

    python validation/sweep.py MIN_CONTROL_S=0.25,0.5,1 CONTROL_RADIUS=1,1.5,2 --out sweep.csv

Every combination is scored on every game (mode `atd` by default, 3 seeds). The leave-one-game-out
block then picks the best combination on two games and reports it on the third: if the held-out
score keeps up with the in-sample one, the choice generalises rather than fitting one game.
"""

import argparse
import csv
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from validation.run import evaluate, load_games, parse_overrides  # noqa: E402

METRICS = ("spellF1", "macroF1", "D_F1", "team_acc", "poss", "gt_poss")


def grid(specs):
    names, values = [], []
    for spec in specs:
        name, raw = spec.split("=", 1)
        parse_overrides(["{}=0".format(name)])           # fails early on a typo
        names.append(name)
        values.append([float(v) for v in raw.split(",")])
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def run(games, combos, modes):
    rows = []
    for k, combo in enumerate(combos):
        for r in evaluate(games, modes, with_baseline=False, **combo):
            rows.append(dict(combo, **r))
        print("{}/{} {}".format(k + 1, len(combos), combo), flush=True)
    return rows


def per_game(rows, names):
    """{combo_key: {game: mean spellF1 over seeds}}"""
    out = {}
    for r in rows:
        key = tuple(r[n] for n in names)
        out.setdefault(key, {}).setdefault(r["game"], []).append(r["spellF1"])
    return {k: {g: float(np.mean(v)) for g, v in games.items()} for k, games in out.items()}


def leave_one_out(scores, names):
    games = sorted(next(iter(scores.values())))
    lines = []
    for held in games:
        train = [g for g in games if g != held]
        best = max(scores, key=lambda k: np.mean([scores[k][g] for g in train]))
        lines.append("  hold out {:<14} best on the others {}  -> train {:.4f}  held-out {:.4f}"
                     .format(held, dict(zip(names, best)),
                             np.mean([scores[best][g] for g in train]), scores[best][held]))
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specs", nargs="+", metavar="NAME=V1,V2,...")
    parser.add_argument("--modes", nargs="*", default=["atd"])
    parser.add_argument("--out")
    args = parser.parse_args()

    combos = grid(args.specs)
    names = list(combos[0])
    games = load_games()
    rows = run(games, combos, args.modes)

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    scores = per_game(rows, names)
    ranked = sorted(scores, key=lambda k: -np.mean(list(scores[k].values())))
    print("\nmean spellF1 over games ({}), best first".format(",".join(args.modes)))
    for key in ranked[:15]:
        print("  {:<60} {:.4f}   {}".format(str(dict(zip(names, key))),
                                           np.mean(list(scores[key].values())),
                                           "  ".join("{:.3f}".format(scores[key][g])
                                                     for g in sorted(scores[key]))))
    print("\nleave one game out")
    print("\n".join(leave_one_out(scores, names)))
