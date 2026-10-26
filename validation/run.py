"""Score the team pipeline on other games, where scoring against a ground truth is allowed:

    python validation/download.py                    # once: fetch the sample games
    python validation/run.py                         # current config.py, both modes
    python validation/run.py --set MIN_CONTROL_S=0.5 CONTROL_RADIUS=2.0

Modes: `raw` runs on the sample tracking as published; `atd` first degrades it to look like the
provided automatic tracking (validation/degrade.py), which is the one that predicts BAR-ATH.
Smart Tagging does not exist for these games. By default the SET PIECES term and the tag hint
are therefore off; with --tags, synthetic tags of BAR-ATH's measured quality are made for each
game (validation/tags_sim.py), so the use of tags can be tested too.

The baseline row is the published reference algorithm (nearest player, majority vote +/- 50
frames) on the same arrays, so every number has a floor next to it.
"""

import argparse
import contextlib
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import config  # noqa: E402
from possession.features import team_distances  # noqa: E402
from possession.pipeline import predict_blocks  # noqa: E402
from starter.baseline import smooth  # noqa: E402
from validation import degrade, tags_sim  # noqa: E402
from validation import idsse, metrica  # noqa: E402

MODES = ("raw", "atd")
SEEDS = (0, 1, 2)

# Locked away from every choice: one Metrica game and two IDSSE matches. Settings are picked on
# the other seven ("dev"); these are scored only to confirm a decision already made, so that the
# number reported for them was never optimised. (Everything up to commit 44c40eb was chosen on all
# ten, so they are clean only for decisions made after that.)
TEST_GAMES = ("Sample_Game_3", "IDSSE_J03WQQ", "IDSSE_J03WR9")
SPLITS = ("dev", "test", "all")


def load_games(prefixes=None, split="dev"):
    """Validation games on disk: the three Metrica games, plus IDSSE if downloaded, restricted to
    `split` ("dev" by default, "test", or "all"). With `prefixes`, only the games whose name
    starts with one of them (e.g. "IDSSE", "Sample")."""
    names = list(metrica.GAMES) + ["IDSSE_" + match for match in idsse.available()]
    if split == "dev":
        names = [n for n in names if n not in TEST_GAMES]
    elif split == "test":
        names = [n for n in names if n in TEST_GAMES]
    if prefixes:
        names = [n for n in names if any(n.startswith(p) for p in prefixes)]
    return {n: idsse.load_game(n[6:]) if n.startswith("IDSSE_") else metrica.load_game(n)
            for n in names}


@contextlib.contextmanager
def params(**overrides):
    """Temporarily set config values. Derived costs follow their inputs unless set themselves."""
    saved = {name: getattr(config, name) for name in
             set(overrides) | {"SWITCH_HOME_AWAY", "TAG_CERTAINTY"}}
    for name, value in overrides.items():
        setattr(config, name, value)
    if "SWITCH_HOME_AWAY" not in overrides:
        config.SWITCH_HOME_AWAY = config.MIN_CONTROL_S * config.FPS * math.log(
            config.CONTROL_CERTAINTY / (1 - config.CONTROL_CERTAINTY))
    if "TAG_CERTAINTY" not in overrides:
        config.TAG_CERTAINTY = 1.0 / (1.0 + math.exp(
            -config.SWITCH_HOME_AWAY / (config.TAG_SWITCH_S * config.FPS)))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(config, name, value)


def blocks_of(gt, periods=None):
    """Graded [(first, stop)] spans, also split where the period changes: the IDSSE games run
    straight from one half into the next, with no ungraded frames between them."""
    graded = (gt != "X").astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    blocks = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    if periods is None:
        return blocks
    cuts = set((np.flatnonzero(periods[1:] != periods[:-1]) + 1).tolist())
    out = []
    for first, stop in blocks:
        inner = sorted(c for c in cuts if first < c < stop)
        out += list(zip([first] + inner, inner + [stop]))
    return out


def baseline(ball, players, team, blocks):
    d_home, d_away = team_distances(ball, players, team)
    codes = np.full(len(ball), -1, dtype=np.int8)
    seen = ~np.isnan(d_home) | ~np.isnan(d_away)
    codes[seen] = np.where(np.nan_to_num(d_home, nan=np.inf) <= np.nan_to_num(d_away, nan=np.inf),
                           0, 1)[seen]
    labels = np.full(len(ball), "X", dtype="<U1")
    for first, stop in blocks:
        labels[first:stop] = smooth(codes[first:stop], 50)
    return labels


def inputs(data, mode, seed):
    ball, players, speed = data["ball"], data["players"], data["speed"]
    if mode == "atd":
        ball = degrade.ball_smoothing(ball)
        if "ball_error" not in data:
            seen = (data["gt"] != "X") & ~np.isnan(ball[:, 0])
            data["ball_error"] = degrade.calibrated_error(ball, players, seen)
        rng = np.random.default_rng(seed)
        alive = np.isin(data["gt"], ("H", "A"))
        ball = degrade.ball_blackouts(ball, alive, rng)
        ball = degrade.ball_blackouts(ball, data["gt"] == "D", rng, target=degrade.DEAD_MISSING,
                                      confine=True)
        ball = degrade.ball_error(ball, rng, sigma=data["ball_error"])
        players, speed = degrade.player_dropouts(players, speed, rng)
    return ball, players, speed


def score(gt, labels):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels))
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "D_F1": r["per_class"]["D"]["f1"], "team_acc": r["team_acc_gt_alive"],
            "poss": r["n_pred_spells"], "gt_poss": r["n_gt_spells"]}


def synthetic_tags(data, seed, quality=None):
    """(side, set_piece_starts) for one game and seed. `quality` may override keep / flip (team
    phases) and stop_cover / false_tags (set pieces)."""
    quality = quality or {}
    rng = np.random.default_rng(1000 + seed)
    side = tags_sim.team_side(data["gt"], rng, **{k: v for k, v in quality.items()
                                                  if k in ("keep", "flip")})
    starts = tags_sim.set_piece_starts(data["gt"], rng, **{k: v for k, v in quality.items()
                                                           if k in ("stop_cover", "false_tags")})
    return side, starts


def evaluate(games, modes=MODES, seeds=SEEDS, with_baseline=True, tags=False, **overrides):
    """[{game, mode, seed, system, metrics...}] for every combination. `tags`: False, True for
    synthetic tags, or a dict of tag quality ({"keep": .., "flip": ..})."""
    rows = []
    for game in games:
        data = games[game]
        blocks = blocks_of(data["gt"], data.get("periods"))
        for mode in modes:
            for seed in (seeds if mode == "atd" else (0,)):
                ball, players, speed = inputs(data, mode, seed)
                side, starts = (synthetic_tags(data, seed, tags if isinstance(tags, dict) else None)
                                if tags else (None, ()))
                with params(**overrides):
                    pitch = tuple(data["pitch"]) if "pitch" in data else metrica.PITCH
                    labels = predict_blocks(ball, players, data["team"], blocks, pitch, speed,
                                            starts, side)
                rows.append(dict(game=game, mode=mode, seed=seed, system="pipeline",
                                 **score(data["gt"], labels)))
                if with_baseline:
                    rows.append(dict(game=game, mode=mode, seed=seed, system="baseline50",
                                     **score(data["gt"], baseline(ball, players, data["team"],
                                                                  blocks))))
    return rows


def summarise(rows):
    keys = ("spellF1", "macroF1", "D_F1", "team_acc", "poss", "gt_poss")
    out = {}
    for row in rows:
        out.setdefault((row["system"], row["mode"], row["game"]), []).append(row)
    table = []
    for (system, mode, game), group in sorted(out.items()):
        table.append(dict(system=system, mode=mode, game=game,
                          **{k: float(np.mean([g[k] for g in group])) for k in keys}))
    return table


def print_table(table):
    print("{:<11} {:<4} {:<14} {:>8} {:>8} {:>6} {:>8} {:>6} {:>6}".format(
        "system", "mode", "game", "spellF1", "macroF1", "D_F1", "team_acc", "poss", "gt"))
    for r in table:
        print("{:<11} {:<4} {:<14} {:>8.4f} {:>8.4f} {:>6.3f} {:>8.3f} {:>6.0f} {:>6.0f}".format(
            r["system"], r["mode"], r["game"], r["spellF1"], r["macroF1"], r["D_F1"],
            r["team_acc"], r["poss"], r["gt_poss"]))


def parse_overrides(pairs):
    out = {}
    for pair in pairs or ():
        name, value = pair.split("=", 1)
        if not hasattr(config, name):
            raise SystemExit("config has no {}".format(name))
        out[name] = float(value)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs="*", metavar="NAME=VALUE")
    parser.add_argument("--modes", nargs="*", default=list(MODES))
    parser.add_argument("--games", nargs="*", metavar="PREFIX")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--tags", action="store_true", help="synthetic Smart Tagging")
    args = parser.parse_args()
    games = load_games(args.games, args.split)
    print_table(summarise(evaluate(games, args.modes, tags=args.tags,
                                   **parse_overrides(args.set))))
