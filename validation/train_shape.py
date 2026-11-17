"""Fit the team-shape coefficients (config.SHAPE_COEF) on the validation games, and test them:

    python validation/train_shape.py                       # fit on all games, print SHAPE_COEF
    python validation/train_shape.py --logo 0.05 0.1 0.2   # leave-one-game-out at those weights

Training data is other games only (RULES.md allows it in any amount): alive frames of the
degraded (`atd`) inputs, labelled by those games' ground truth. Leave-one-game-out refits the
coefficients without the game it then scores, so the reported spellF1 never saw its own labels.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import shape  # noqa: E402
from validation.metrica import PITCH  # noqa: E402
from validation.run import blocks_of, evaluate, inputs, load_games, summarise  # noqa: E402

STRIDE = 5          # every 5th alive frame: neighbours carry the same information anyway
L2 = 1e-3


def design(data, seed=0):
    """(X, y) for one game: shape features of alive frames, y = 1 for home."""
    ball, players, speed = inputs(data, "atd", seed)
    players = np.where(np.isnan(speed)[:, :, None], np.nan, players)
    pitch = tuple(data["pitch"]) if "pitch" in data else PITCH
    blocks = blocks_of(data["gt"], data.get("periods"))
    sign = shape.attack_sign(players, data["team"], blocks, pitch[0])
    feats = shape.features(players, data["team"], sign, pitch[0])
    keep = np.isin(data["gt"], ("H", "A")) & ~np.isnan(feats).any(axis=1)
    idx = np.flatnonzero(keep)[::STRIDE]
    return feats[idx], (data["gt"][idx] == "H").astype(float)


def fit(xs, ys, iters=25):
    """Logistic regression without intercept, by Newton's method."""
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-x @ w))
        grad = x.T @ (p - y) / len(y) + L2 * w
        hess = (x * (p * (1 - p))[:, None]).T @ x / len(y) + L2 * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return w


def accuracy(x, y, w):
    return float(((x @ w > 0) == (y > 0.5)).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logo", nargs="*", type=float, metavar="SHAPE_WEIGHT")
    parser.add_argument("--games", nargs="*", metavar="PREFIX")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    args = parser.parse_args()

    games = load_games(args.games, args.split)
    data = {name: design(g) for name, g in games.items()}
    w = fit([d[0] for d in data.values()], [d[1] for d in data.values()])
    print("SHAPE_COEF = ({:.4f}, {:.4f}, {:.4f})".format(*w))
    for name, (x, y) in data.items():
        print("  {:<14} frame accuracy {:.3f}".format(name, accuracy(x, y, w)))

    if args.logo:
        rows = []
        for held in games:
            w_out = fit([d[0] for n, d in data.items() if n != held],
                        [d[1] for n, d in data.items() if n != held])
            for weight in [0.0] + list(args.logo):
                t = summarise(evaluate({held: games[held]}, ("atd",), with_baseline=False,
                                       SHAPE_COEF=tuple(w_out), SHAPE_WEIGHT=weight))
                rows.append((held, weight, t[0]["spellF1"], t[0]["team_acc"], t[0]["poss"]))
                print("  held out {:<14} weight {:<5} spellF1 {:.4f}  team_acc {:.3f}  poss {:.0f}"
                      .format(held, weight, *rows[-1][2:]), flush=True)
        print("\nmean over held-out games")
        for weight in [0.0] + list(args.logo):
            sel = [r for r in rows if r[1] == weight]
            metr = [r[2] for r in sel if not r[0].startswith("IDSSE")]
            ids = [r[2] for r in sel if r[0].startswith("IDSSE")]
            print("  weight {:<5} spellF1 all {:.4f}   Metrica {:.4f}   IDSSE {:.4f}".format(
                weight, np.mean([r[2] for r in sel]), np.mean(metr) if metr else np.nan,
                np.mean(ids) if ids else np.nan))
