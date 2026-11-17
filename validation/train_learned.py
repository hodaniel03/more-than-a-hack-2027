"""Train the learned evidence models (possession/learned.py) on other games:

    python validation/learned_data.py                 # once: feature tables
    python validation/train_learned.py --oof          # leave-one-game-out probabilities (dev)
    python validation/train_learned.py --final        # fit on the dev games, write models/learned.npz

Gradient-boosted trees (scikit-learn's HistGradientBoostingClassifier, fixed seed), exported to
plain arrays so the pipeline needs only numpy. Training needs scikit-learn; inference does not.

Only games other than BAR-ATH are read here (RULES.md: labelled data from other games is allowed
in any amount). The three locked test games (validation/run.py TEST_GAMES) are never trained on
by --oof or --final; they are scored afterwards to confirm the result.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import learned  # noqa: E402
from validation.learned_data import FEAT_DIR, path  # noqa: E402
from validation.run import SEEDS, TEST_GAMES  # noqa: E402

OOF_DIR = os.path.join(ROOT, "data", "external", "oof")
STEP = 4                    # every 4th frame: neighbours are near-duplicates
PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=400,
              l2_regularization=1.0, early_stopping=False, random_state=0)


def games_on_disk():
    names = sorted({f.rsplit("_", 1)[0] for f in os.listdir(FEAT_DIR) if f.endswith(".npz")})
    return names


def load(game, seed):
    return np.load(path(game, seed), allow_pickle=False)


def training_rows(games, seeds=SEEDS, step=STEP):
    """(Xd, yd, Xt, yt) stacked over games and seeds; the team rows come twice, once swapped."""
    Xd, yd, Xt, yt = [], [], [], []
    names_t = None
    for game in games:
        for seed in seeds:
            d = load(game, seed)
            gt = d["gt"]
            graded = np.flatnonzero(gt != "X")[seed % step::step]
            Xd.append(d["Xd"][graded][:, dead_columns(d)])
            yd.append(gt[graded] == "D")
            alive = np.flatnonzero((gt == "H") | (gt == "A"))[seed % step::step]
            names_t = [str(s) for s in d["team_names"]]
            x = d["Xt"][alive]
            y = gt[alive] == "H"
            Xt += [x, learned.swap_teams(names_t, x)]
            yt += [y, ~y]
    return np.concatenate(Xd), np.concatenate(yd), np.concatenate(Xt), np.concatenate(yt)


def dead_columns(d):
    """Column indices of the dead-ball features the model is given (learned.DEAD_EXCLUDED)."""
    names = [str(s) for s in d["dead_names"]]
    keep = set(learned.dead_feature_names(names))
    return [j for j, n in enumerate(names) if n in keep]


def fit(X, y, **overrides):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(**dict(PARAMS, **overrides))
    model.fit(X, y)
    return model


def export(model):
    """A fitted binary HistGradientBoostingClassifier as flat arrays for learned.predict_logit."""
    feature, threshold, left, right, value, missing_left, is_leaf, offsets = ([] for _ in range(8))
    total = 0
    for stage in model._predictors:
        nodes = stage[0].nodes
        offsets.append(total)
        feature.append(nodes["feature_idx"].astype(np.int32))
        threshold.append(nodes["num_threshold"].astype(np.float64))
        left.append(nodes["left"].astype(np.int32))
        right.append(nodes["right"].astype(np.int32))
        value.append(nodes["value"].astype(np.float64))
        missing_left.append(nodes["missing_go_to_left"].astype(bool))
        is_leaf.append(nodes["is_leaf"].astype(bool))
        total += len(nodes)
    cat = np.concatenate
    return dict(feature=cat(feature), threshold=cat(threshold), left=cat(left), right=cat(right),
                value=cat(value), missing_left=cat(missing_left), is_leaf=cat(is_leaf),
                offsets=np.array(offsets, dtype=np.int64),
                baseline=np.float64(np.ravel(model._baseline_prediction)[0]))


def save_models(dead, team, dead_names, team_names, out):
    arrays = {}
    for kind, model, names in (("dead", dead, dead_names), ("team", team, team_names)):
        for key, value in export(model).items():
            arrays["{}_{}".format(kind, key)] = value
        arrays[kind + "_names"] = np.array(names)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **arrays)


def names_of(game):
    d = load(game, 0)
    return (learned.dead_feature_names([str(s) for s in d["dead_names"]]),
            [str(s) for s in d["team_names"]])


def predict_game(models, game, seed):
    """(p_dead, p_home) for one feature table, with the team answer symmetrised."""
    d = load(game, seed)
    zd = learned.predict_logit(models["dead"], d["Xd"][:, dead_columns(d)])
    Xt = d["Xt"]
    names = models["team"]["names"]
    zt = 0.5 * (learned.predict_logit(models["team"], Xt)
                - learned.predict_logit(models["team"], learned.swap_teams(names, Xt)))
    return 1.0 / (1.0 + np.exp(-zd)), 1.0 / (1.0 + np.exp(-zt))


def as_models(dead, team, dead_names, team_names):
    return {"dead": dict(export(dead), names=dead_names),
            "team": dict(export(team), names=team_names)}


def out_of_fold(games, tag="oof", **overrides):
    """Leave-one-game-out probabilities for every dev game and seed, saved under data/external/."""
    os.makedirs(OOF_DIR, exist_ok=True)
    for held in games:
        rest = [g for g in games if g != held]
        Xd, yd, Xt, yt = training_rows(rest)
        dn, tn = names_of(held)
        models = as_models(fit(Xd, yd, **overrides), fit(Xt, yt, **overrides), dn, tn)
        for seed in SEEDS:
            p_dead, p_home = predict_game(models, held, seed)
            np.savez_compressed(os.path.join(OOF_DIR, "{}_{}_{}.npz".format(tag, held, seed)),
                                p_dead=p_dead.astype(np.float32), p_home=p_home.astype(np.float32))
        print("oof", held, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oof", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--all-games", action="store_true",
                        help="--final on all ten games instead of the seven dev games")
    args = parser.parse_args()
    dev = [g for g in games_on_disk() if g not in TEST_GAMES]
    if args.oof:
        out_of_fold(dev)
    if args.final:
        games = games_on_disk() if args.all_games else dev
        Xd, yd, Xt, yt = training_rows(games)
        dn, tn = names_of(games[0])
        save_models(fit(Xd, yd), fit(Xt, yt), dn, tn, learned.MODEL_PATH)
        print("wrote", os.path.relpath(learned.MODEL_PATH, ROOT), "from", len(games), "games")
