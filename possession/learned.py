"""Learned evidence: per-frame probabilities from gradient-boosted trees trained on *other* games.

Two models, both trained by validation/train_learned.py on public games with a ground truth
(Metrica's sample games and the IDSSE Bundesliga matches), degraded to look like the provided
ATD. Nothing here is fitted on BAR-ATH:

  * dead:  P(ball is dead)                      -- team-agnostic features
  * team:  P(home has the ball | ball alive)    -- features signed towards home, and trained on
           every game twice, once with the teams swapped, so the model cannot prefer a side

The features are windows of the same signals the rule-based model uses (reach, touches, player
speed, ball position and visibility, team shape), looking up to 10 s back and forward -- the
pipeline labels a finished match, so the future is available.

The trees are stored as plain arrays (models/learned.npz) and evaluated with numpy, so inference
needs no scikit-learn and gives the same numbers on every machine.
"""

import os
import warnings

import numpy as np

from . import config, shape
from .deadball import ball_outside, ball_still, latch_out
from .features import ball_touches, interpolate_ball, team_distances

FPS = 25
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "learned.npz")

# (from, to) in seconds relative to the frame, inclusive: past, then future.
WINDOWS = ((-1.0, 0.0), (-3.0, -1.0), (-8.0, -3.0), (0.0, 1.0), (1.0, 3.0), (3.0, 8.0))
LONG = ((-20.0, -8.0), (8.0, 20.0))
CAP_S = 30.0

# Dead-ball features that are computed but not given to the model, by name prefix. Their
# distribution in the training games is an artefact of those games, not of football:
#   still, bspeed   the IDSSE (DFL) ball stands still on the restart spot for the whole stoppage,
#                   which the ATD ball almost never does (it is simply not seen);
#   out, edge_rate  no training game ever shows the ball outside the lines, the ATD does -- that
#                   signal is added by rule instead (config.LEARNED_W_OUTSIDE).
# Trained on IDSSE and scored on the Metrica games, leaving them out raises the dead-ball F1
# from 0.80 to 0.84 at the same spellF1 (0.61).
DEAD_EXCLUDED = ("still", "since_still", "until_still", "bspeed", "out", "since_out", "edge_rate")


def window_mean(x, a, b):
    """NaN-aware mean of x over frames [t + a, t + b] (inclusive, in frames). NaN if empty."""
    n = len(x)
    ok = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(n)
    lo = np.clip(i + a, 0, n)
    hi = np.clip(i + b + 1, 0, n)
    c = count[hi] - count[lo]
    out = np.full(n, np.nan)
    np.divide(total[hi] - total[lo], c, out=out, where=c > 0)
    return out


def since_true(flag):
    """Frames since the last True (0 on a True frame); a large number before the first."""
    n = len(flag)
    idx = np.where(flag, np.arange(n), -10 ** 9)
    return np.arange(n) - np.maximum.accumulate(idx)


def until_true(flag):
    return since_true(flag[::-1])[::-1]


def fill_forward(x, valid):
    """x with invalid entries replaced by the last valid value before them (NaN before any)."""
    idx = np.where(valid, np.arange(len(x)), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, x[np.maximum(last, 0)], np.nan)
    return out


def fill_backward(x, valid):
    return fill_forward(x[::-1], valid[::-1])[::-1]


def row_percentile(values, q):
    """NaN-aware percentile (0 <= q <= 1, linear interpolation) of each row, in float64.

    Written out instead of np.nanpercentile because that function's rounding on float32 input
    differs between numpy versions, and one frame of difference breaks "the same file"."""
    ordered = np.sort(values.astype(np.float64), axis=1)          # NaN last
    count = (~np.isnan(ordered)).sum(axis=1)
    pos = q * np.maximum(count - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(count - 1, 0))
    rows = np.arange(len(ordered))
    frac = pos - lo
    out = ordered[rows, lo] * (1.0 - frac) + ordered[rows, hi] * frac
    out[count == 0] = np.nan
    return out


def window_min(x, a, b):
    """NaN-aware minimum of x over frames [t + a, t + b] (inclusive, in frames). NaN if empty."""
    n = len(x)
    pad = max(abs(a), abs(b))
    padded = np.concatenate((np.full(pad, np.inf), np.where(np.isnan(x), np.inf, x),
                             np.full(pad, np.inf)))
    view = np.lib.stride_tricks.sliding_window_view(padded, b - a + 1)
    out = view[np.arange(n) + pad + a].min(axis=1)
    out[~np.isfinite(out)] = np.nan
    return out


def _windows(name, x, feats, windows=WINDOWS):
    for a, b in windows:
        feats["{}[{:g},{:g}]".format(name, a, b)] = window_mean(x, int(a * FPS), int(b * FPS))


def base_signals(ball, players, team, speed, pitch):
    """Everything both feature sets share, computed once."""
    length, width = pitch
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    filled, conf = interpolate_ball(ball, config.MAX_BALL_GAP)
    d_home, d_away = team_distances(filled, players, team)

    lag = 3
    v = np.full((n, 2), np.nan)
    v[lag:n - lag] = (ball[2 * lag:] - ball[:n - 2 * lag]) / (2.0 * lag) * FPS
    bspeed = np.sqrt((v ** 2).sum(axis=1))

    x, y = ball[:, 0], ball[:, 1]
    edge = np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))     # < 0: outside
    goal_edge = np.minimum(x, length - x)
    edge_rate = np.full(n, np.nan)
    edge_rate[lag:n - lag] = (edge[2 * lag:] - edge[:n - 2 * lag]) / (2.0 * lag) * FPS

    valid = ~np.isnan(speed)
    count = valid.sum(axis=1).astype(float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(speed, axis=1).astype(np.float64)
        p75 = row_percentile(speed, 0.75)
        run = np.where(count > 0, (np.nan_to_num(speed, nan=0.0) > 4.0).sum(axis=1) / count, np.nan)
        walk = np.where(count > 0, ((speed < 1.5) & valid).sum(axis=1) / count, np.nan)
    few = count < config.MIN_PLAYERS_FOR_SPEED
    for arr in (med, p75, run, walk):
        arr[few] = np.nan

    return dict(n=n, seen=seen, filled=filled, conf=conf, d_home=d_home, d_away=d_away, v=v,
                bspeed=bspeed, edge=edge, goal_edge=goal_edge, edge_rate=edge_rate, med=med,
                p75=p75, run=run, walk=walk, count=count)


def dead_features(ball, players, team, speed, pitch, base=None):
    """({name: (N,) float32}) team-agnostic features for P(dead)."""
    s = base or base_signals(ball, players, team, speed, pitch)
    seen = s["seen"]
    f = {}
    f["seen"] = seen.astype(float)
    f["since_seen"] = np.minimum(since_true(seen) / FPS, CAP_S)
    f["until_seen"] = np.minimum(until_true(seen) / FPS, CAP_S)
    f["gap_len"] = np.where(seen, 0.0, np.minimum(f["since_seen"] + f["until_seen"], 2 * CAP_S))

    f["edge"] = s["edge"]
    f["edge_last"] = fill_forward(s["edge"], seen)
    f["edge_next"] = fill_backward(s["edge"], seen)
    f["goal_edge_last"] = fill_forward(s["goal_edge"], seen)
    f["goal_edge_next"] = fill_backward(s["goal_edge"], seen)
    ok = ~np.isnan(s["edge_rate"])
    f["edge_rate_last"] = fill_forward(s["edge_rate"], ok)
    f["bspeed"] = s["bspeed"]
    okv = ~np.isnan(s["bspeed"])
    f["bspeed_last"] = fill_forward(s["bspeed"], okv)
    f["bspeed_next"] = fill_backward(s["bspeed"], okv)

    outside = ball_outside(ball, pitch)
    latched = latch_out(outside, seen & ~outside, config.OUT_LATCH_FRAMES)
    f["out_latched"] = latched.astype(float)
    f["since_out"] = np.minimum(since_true(outside) / FPS, CAP_S)
    still = ball_still(ball, config.STILL_FRAMES)
    f["still"] = still.astype(float)
    f["since_still"] = np.minimum(since_true(still) / FPS, CAP_S)
    f["until_still"] = np.minimum(until_true(still) / FPS, CAP_S)

    dmin = np.fmin(s["d_home"], s["d_away"])
    f["dmin"] = np.minimum(dmin, 30.0)
    f["count"] = s["count"]
    f["med"], f["p75"], f["run"], f["walk"] = s["med"], s["p75"], s["run"], s["walk"]

    # Set-piece shape: how many players stand in the fuller penalty area, and how stretched the
    # two teams are along the pitch (a restart draws everyone into one zone).
    px, py = players[:, :, 0], players[:, :, 1]
    with np.errstate(invalid="ignore"):
        in_y = np.abs(py - pitch[1] / 2.0) <= 20.16
        left = ((px <= 16.5) & in_y).sum(axis=1)
        right = ((px >= pitch[0] - 16.5) & in_y).sum(axis=1)
    f["box"] = np.maximum(left, right).astype(float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["xspread"] = row_percentile(px, 0.9) - row_percentile(px, 0.1)
    _windows("box", f["box"], f, ((-3.0, 0.0), (0.0, 3.0)))

    _windows("seen", seen.astype(float), f, WINDOWS + LONG)
    _windows("med", s["med"], f, WINDOWS + LONG)
    _windows("run", s["run"], f)
    _windows("walk", s["walk"], f)
    _windows("p75", s["p75"], f)
    _windows("bspeed", s["bspeed"], f)
    _windows("dmin", np.minimum(dmin, 30.0), f)
    _windows("still", still.astype(float), f)
    _windows("out", latched.astype(float), f)
    return {k: v.astype(np.float32) for k, v in f.items()}


def dead_feature_names(names):
    """The dead-ball features the model is given: all but DEAD_EXCLUDED."""
    return [n for n in names if not n.startswith(DEAD_EXCLUDED)]


def team_features(ball, players, team, speed, pitch, blocks, base=None):
    """({name: (N,) float32}) features for P(home | alive). Every feature flips sign, or swaps with
    its partner (`*_h` <-> `*_a`), when the teams are swapped -- see swap_teams()."""
    s = base or base_signals(ball, players, team, speed, pitch)
    seen, conf = s["seen"], s["conf"]
    d_home, d_away = s["d_home"], s["d_away"]
    r = config.CONTROL_RADIUS
    with np.errstate(invalid="ignore"):
        home_in = np.nan_to_num(d_home, nan=np.inf) <= r
        away_in = np.nan_to_num(d_away, nan=np.inf) <= r
    excl = (home_in & ~away_in).astype(float) - (away_in & ~home_in).astype(float)
    excl *= conf
    both = (home_in & away_in).astype(float) * conf
    # Which team is nearer, saturating at 2 m of difference. Plain arithmetic in float64 on
    # purpose: tanh() can differ in the last bit between numpy builds.
    with np.errstate(invalid="ignore"):
        near = np.clip((d_away.astype(np.float64) - d_home.astype(np.float64)) / 2.0, -1.0, 1.0)
    near = np.nan_to_num(near * conf, nan=0.0)

    touch = ball_touches(ball, *team_distances(ball, players, team), config.TOUCH_JUMP,
                         config.TOUCH_RADIUS, int(config.TOUCH_LAG)).astype(float)

    f = {}
    f["excl"] = excl
    f["near"] = near
    f["d_h"] = np.minimum(np.nan_to_num(d_home, nan=40.0), 40.0)
    f["d_a"] = np.minimum(np.nan_to_num(d_away, nan=40.0), 40.0)
    f["u_conf"] = conf
    f["u_both"] = both
    f["u_since_seen"] = np.minimum(since_true(seen) / FPS, CAP_S)
    f["u_until_seen"] = np.minimum(until_true(seen) / FPS, CAP_S)
    _windows("excl", excl, f, WINDOWS + LONG)
    _windows("near", near, f, WINDOWS + LONG)
    _windows("u_both", both, f)
    for a, b in ((-2.0, 0.0), (-6.0, -2.0), (0.0, 2.0), (2.0, 6.0)):
        f["touch[{:g},{:g}]".format(a, b)] = window_mean(touch, int(a * FPS), int(b * FPS)) * FPS

    # Closest approach of each team to the ball in a window: a touch is a moment, not a mean.
    dh = np.minimum(np.where(conf > 0, d_home, np.nan), 40.0)
    da = np.minimum(np.where(conf > 0, d_away, np.nan), 40.0)
    for a, b in ((-1.0, 0.0), (-3.0, -1.0), (0.0, 1.0), (1.0, 3.0)):
        f["mind[{:g},{:g}]_h".format(a, b)] = window_min(dh, int(a * FPS), int(b * FPS))
        f["mind[{:g},{:g}]_a".format(a, b)] = window_min(da, int(a * FPS), int(b * FPS))
    # How many of each team are around the ball (within 6 m).
    with np.errstate(invalid="ignore"):
        close = np.sqrt(((players - s["filled"][:, None, :]) ** 2).sum(axis=2)) <= 6.0
    f["n6_h"] = np.where(conf > 0, close[:, team == 0].sum(axis=1), np.nan).astype(float)
    f["n6_a"] = np.where(conf > 0, close[:, team == 1].sum(axis=1), np.nan).astype(float)

    # Who was last / is next alone at the ball, and how long ago: carries a blackout.
    alone = excl != 0
    f["excl_last"] = np.nan_to_num(fill_forward(excl, alone), nan=0.0)
    f["excl_next"] = np.nan_to_num(fill_backward(excl, alone), nan=0.0)
    f["u_since_alone"] = np.minimum(since_true(alone) / FPS, CAP_S)
    f["u_until_alone"] = np.minimum(until_true(alone) / FPS, CAP_S)
    hit = touch != 0
    f["touch_last"] = np.nan_to_num(fill_forward(touch, hit), nan=0.0)
    f["touch_next"] = np.nan_to_num(fill_backward(touch, hit), nan=0.0)
    f["u_since_touch"] = np.minimum(since_true(hit) / FPS, CAP_S)
    f["u_until_touch"] = np.minimum(until_true(hit) / FPS, CAP_S)

    sign = shape.attack_sign(players, team, blocks, pitch[0]).astype(float)
    sh = shape.features(players, team, sign.astype(np.int8), pitch[0])
    f["shape_territory"], f["shape_width"], f["shape_direction"] = sh[:, 0], sh[:, 1], sh[:, 2]
    for j, name in enumerate(("shape_territory", "shape_width", "shape_direction")):
        for a, b in ((-5.0, -1.0), (1.0, 5.0)):
            f["{}[{:g},{:g}]".format(name, a, b)] = window_mean(sh[:, j], int(a * FPS),
                                                                int(b * FPS))
    # Ball towards the away goal, and how far into the away half it is.
    f["ball_forward"] = s["v"][:, 0] * sign
    f["ball_territory"] = (s["filled"][:, 0] - pitch[0] / 2.0) * sign
    _windows("ball_forward", f["ball_forward"], f, ((-2.0, 0.0), (0.0, 2.0)))
    f["u_med"] = s["med"]
    return {k: v.astype(np.float32) for k, v in f.items()}


def swap_teams(names, X):
    """The feature matrix the same frames would give with home and away exchanged."""
    out = -X
    index = {name: j for j, name in enumerate(names)}
    for j, name in enumerate(names):
        if name.startswith("u_"):
            out[:, j] = X[:, j]
        elif name.endswith("_h"):
            out[:, j] = X[:, index[name[:-2] + "_a"]]
        elif name.endswith("_a"):
            out[:, j] = X[:, index[name[:-2] + "_h"]]
    return out


def matrix(feats, names=None):
    names = names or sorted(feats)
    return names, np.stack([feats[k] for k in names], axis=1)


# ---------------------------------------------------------------- tree inference (numpy only)

def predict_logit(model, X):
    """Raw score of a boosted-tree model stored as arrays. `model` has, per tree, flat node arrays:
    feature, threshold, left, right, value, missing_left, is_leaf, plus `offsets` and `baseline`."""
    out = np.full(len(X), float(model["baseline"]))
    feature, threshold = model["feature"], model["threshold"]
    left, right, value = model["left"], model["right"], model["value"]
    missing_left, is_leaf = model["missing_left"], model["is_leaf"]
    rows = np.arange(len(X))
    for start in model["offsets"]:
        node = np.full(len(X), start, dtype=np.int64)
        active = ~is_leaf[node]
        while active.any():
            idx = rows[active]
            nd = node[idx]
            x = X[idx, feature[nd]]
            go_left = np.where(np.isnan(x), missing_left[nd], x <= threshold[nd])
            node[idx] = start + np.where(go_left, left[nd], right[nd])
            active[idx] = ~is_leaf[node[idx]]
        out += value[node]
    return out


def load_model(path=MODEL_PATH):
    data = np.load(path, allow_pickle=False)
    models = {}
    for kind in ("dead", "team"):
        models[kind] = {k[len(kind) + 1:]: data[k] for k in data.files if k.startswith(kind + "_")}
        models[kind]["names"] = [str(s) for s in models[kind]["names"]]
    return models


def probabilities(models, ball, players, team, speed, pitch, blocks):
    """(p_dead, p_home) per frame from the stored models."""
    base = base_signals(ball, players, team, speed, pitch)
    fd = dead_features(ball, players, team, speed, pitch, base)
    ft = team_features(ball, players, team, speed, pitch, blocks, base)
    _, Xd = matrix(fd, models["dead"]["names"])
    names, Xt = matrix(ft, models["team"]["names"])
    zd = predict_logit(models["dead"], Xd)
    # Symmetrise: average the model's answer with the mirrored question's.
    zt = 0.5 * (predict_logit(models["team"], Xt)
                - predict_logit(models["team"], swap_teams(names, Xt)))
    return 1.0 / (1.0 + np.exp(-zd)), 1.0 / (1.0 + np.exp(-zt))
