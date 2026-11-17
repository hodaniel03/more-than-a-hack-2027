"""Who has the ball, read from the two teams' shape -- for the frames where the ball is not seen.

Three textbook signs, none of which needs the ball:

  * territory: both blocks follow the ball, so where they stand says which half it is in, and a
    team spends more of its possession in the opponent's half than in its own. (Which block is
    nearer which goal says nothing: each team stands goal-side of the other almost always.)
  * width: in possession a team makes the pitch big, out of possession it makes it small;
  * direction: when the ball moves forward, both teams move towards the defending team's goal.

Each is a signed home-minus-away number, positive when it points to home. A logistic model over
them (coefficients in config.SHAPE_COEF, fitted on other games by validation/train_shape.py) turns
them into a log-odds that home has the ball. Everything is collective -- no player identities --
and robust to slots that come and go: medians, not means, and the goalkeeper (each team's deepest
player) left out.
"""

import warnings

import numpy as np

from .deadball import moving_average

HOME, AWAY = 0, 1
N_FEATURES = 3
VELOCITY_LAG = 12          # frames either side: velocity over ~1 s
SMOOTH = 25                # +/- 1 s


def attack_sign(players, team, blocks, pitch_length):
    """(N,) +1 where home attacks towards +x, -1 where towards -x, 0 outside the blocks.

    Read per block, with no metadata, from how often each team has someone inside the penalty
    area at either end: its goalkeeper makes that its own end. Counting frames rather than taking
    a median distance matters on the ATD, where the far keeper often goes untracked."""
    sign = np.zeros(len(players), dtype=np.int8)
    x = players[:, :, 0]
    box = 16.5
    for first, stop in blocks:
        own_left = 0.0                      # > 0: home's own goal is at x = 0
        for side, flip in ((HOME, 1), (AWAY, -1)):
            xs = x[first:stop][:, team == side]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # frames with nobody of a team
                left = np.nanmean(np.nanmin(xs, axis=1) < box)
                right = np.nanmean(np.nanmax(xs, axis=1) > pitch_length - box)
            own_left += flip * (left - right)
        sign[first:stop] = 1 if own_left >= 0 else -1
    return sign


def _team_arrays(players, team, side):
    return players[:, team == side, :]


def features(players, team, sign, pitch_length):
    """(N, 3) shape features pointing to home, smoothed over +/- 1 s. NaN where unknown:
    territory (m into the away half), width (home minus away spread, m), direction (m/s towards
    the away goal)."""
    n = len(players)
    out = np.full((n, N_FEATURES), np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN frames: no players seen
        centre, spread, forward = [], [], []
        for side in (HOME, AWAY):
            p = _team_arrays(players, team, side)
            own = -sign if side == HOME else sign          # +1: own goal towards +x
            depth = p[:, :, 0] * own[:, None]               # larger = nearer own goal
            keeper = np.nanargmax(np.where(np.isnan(depth), -np.inf, depth), axis=1)
            outfield = p.copy()
            outfield[np.arange(n), keeper] = np.nan
            centre.append(np.nanmedian(outfield[:, :, 0], axis=1))
            spread.append(np.sqrt(np.nanvar(outfield[:, :, 0], axis=1)
                                  + np.nanvar(outfield[:, :, 1], axis=1)))
            step = np.full(n, np.nan)
            k = VELOCITY_LAG
            dx = p[2 * k:, :, 0] - p[:-2 * k, :, 0]
            step[k:-k] = np.nanmedian(dx, axis=1) / (2 * k) * 25.0
            forward.append(step * sign)                     # m/s towards the away goal
        out[:, 0] = ((centre[HOME] + centre[AWAY]) / 2.0 - pitch_length / 2.0) * sign
        out[:, 1] = spread[HOME] - spread[AWAY]
        out[:, 2] = (forward[HOME] + forward[AWAY]) / 2.0
    out[sign == 0] = np.nan
    for j in range(N_FEATURES):
        out[:, j] = moving_average(out[:, j], SMOOTH)
    return out


def shape_logit(feats, coef):
    """(N,) log-odds that home has the ball; 0 where any feature is unknown. No intercept: the
    model must not prefer a side."""
    z = np.nan_to_num(feats, nan=0.0) @ np.asarray(coef, dtype=float)
    z[np.isnan(feats).any(axis=1)] = 0.0
    return z
