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
