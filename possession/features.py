"""Per-frame signals from the tracking. Everything here must survive a missing ball."""

import numpy as np

from .tracking import AWAY, HOME


def interpolate_ball(ball, max_gap):
    """Linearly bridge ball gaps of at most `max_gap` frames.

    Returns (filled, confidence). Confidence is 1 on observed frames, falls linearly to ~0 at the
    centre of a bridged gap, and is 0 where the gap is too long to trust.
    """
    n = len(ball)
    filled = ball.copy()
    confidence = np.zeros(n, dtype=np.float32)
    seen = ~np.isnan(ball[:, 0])
    confidence[seen] = 1.0

    index = np.flatnonzero(seen)
    for a, b in zip(index[:-1], index[1:]):
        gap = b - a - 1
        if gap == 0 or gap > max_gap:
            continue
        t = np.arange(1, gap + 1) / (gap + 1.0)
        filled[a + 1:b] = ball[a] + (ball[b] - ball[a]) * t[:, None]
        confidence[a + 1:b] = np.abs(2 * t - 1)
    return filled, confidence


def control_vote(d_home, d_away, confidence, radius, certainty):
    """(N,) probability-like vote that home has the ball, 0.5 where nobody is in control.

    A team is in control only when one of its players is within `radius` of the ball and no
    opponent is. A ball in flight, a loose ball or a 50/50 is no evidence either way, so the state
    model simply keeps the team it already had -- which is what the possession rules ask for.
    """
    home = d_home <= radius
    away = d_away <= radius
    with np.errstate(invalid="ignore"):
        home &= ~away
        away &= ~(d_home <= radius)
    edge = (certainty - 0.5) * confidence
    vote = np.full(len(d_home), 0.5)
    vote[home] += edge[home]
    vote[away] -= edge[away]
    return vote


def ball_touches(ball, d_home, d_away, jump, radius, lag=3):
    """(N,) int8: +1 where a home player touches the ball, -1 an away player, 0 elsewhere.

    A touch is where the ball's velocity changes abruptly -- a pass struck, a ball received or
    intercepted. The change is |v(t+lag) - v(t-lag)|, v a centred difference over `lag` frames,
    and a touch is a local peak of it of at least `jump` m/s, credited to the nearest player if
    he is within `radius` m. In open play the ball is otherwise smooth (median change ~3 m/s),
    while true changes of possession come with changes of ~11 m/s or more. Only seen balls count:
    an interpolated one has no touches by construction.
    """
    n = len(ball)
    v = np.full((n, 2), np.nan)
    v[lag:n - lag] = (ball[2 * lag:] - ball[:n - 2 * lag]) / (2 * lag) * 25.0
    change = np.full(n, np.nan)
    change[lag:n - lag] = np.sqrt(((v[2 * lag:] - v[:n - 2 * lag]) ** 2).sum(axis=1))
    padded = np.concatenate((np.full(lag, -np.inf), np.nan_to_num(change, nan=-np.inf),
                             np.full(lag, -np.inf)))
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * lag + 1).max(axis=1)
    with np.errstate(invalid="ignore"):
        peak = (change >= jump) & (change >= local_max)
        home = peak & (np.nan_to_num(d_home, nan=np.inf) <= np.fmin(np.nan_to_num(
            d_away, nan=np.inf), radius))
        away = peak & ~home & (np.nan_to_num(d_away, nan=np.inf) <= radius)
    out = np.zeros(n, dtype=np.int8)
    out[home] = 1
    out[away] = -1
    return out


def team_distances(ball, players, team):
    """Distance from the ball to the closest player of each team: (home, away), NaN if unknown."""
    delta = players - ball[:, None, :]
    dist = np.sqrt(np.nansum(delta ** 2, axis=2))
    dist[np.isnan(players[:, :, 0])] = np.inf
    out = []
    for side in (HOME, AWAY):
        closest = dist[:, team == side].min(axis=1)
        closest[~np.isfinite(closest)] = np.nan
        out.append(closest)
    out[0][np.isnan(ball[:, 0])] = np.nan
    out[1][np.isnan(ball[:, 0])] = np.nan
    return out[0], out[1]
