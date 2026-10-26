"""Probability that the ball is dead, per frame.

Evidence is summed in log-odds, one term per signal, so each can be inspected on its own
(`dead_evidence`) and none of them has to be right on every frame:

  * collective player speed -- during a stoppage players walk to their positions, in play they run
  * ball outside the pitch -- latched until the ball is seen back inside, because the ball often
    vanishes from the tracking once it has gone out
  * ball standing still -- a placed ball for a free kick, corner or goal kick
  * Smart Tagging SET PIECES -- a restart is near the start of the tag
  * ball missing -- weak on its own: the ball also vanishes for long spells in open play

What this deliberately does not do any more: treat a long ball gap as "probably dead". Almost all
missing-ball frames sit in gaps longer than 3 s, and many of those are open play.
"""

import numpy as np

from games import GAME, GAMES
from . import config


def moving_average(values, window):
    """Centred NaN-aware mean over +/- `window` frames. NaN where the window has no data."""
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low = np.maximum(0, index - window)
    high = np.minimum(len(values), index + window + 1)
    n = count[high] - count[low]
    out = np.full(len(values), np.nan)
    out[n > 0] = (total[high] - total[low])[n > 0] / n[n > 0]
    return out


def team_speed(speed):
    """(N,) median speed of the tracked players, NaN where too few are tracked to trust it."""
    valid = ~np.isnan(speed)
    enough = valid.sum(axis=1) >= config.MIN_PLAYERS_FOR_SPEED
    out = np.full(len(speed), np.nan)
    if enough.any():
        out[enough] = np.nanmedian(speed[enough], axis=1)
    return out


def ball_outside(ball, pitch):
    """(N,) True where the observed ball is clearly beyond a touchline or goal line."""
    length, width = pitch
    m = config.OUT_MARGIN
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < -m) | (x > length + m) | (y < -m) | (y > width + m)


def latch_out(outside, inside_seen, max_frames):
    """Carry "ball is out" forward until the ball is seen inside again, at most `max_frames`."""
    latched = np.zeros(len(outside), dtype=bool)
    since = None
    for i in range(len(outside)):
        if outside[i]:
            since = i
        elif inside_seen[i]:
            since = None
        if since is not None and i - since <= max_frames:
            latched[i] = True
    return latched


def ball_still(ball, min_frames):
    """(N,) True inside runs of at least `min_frames` frames where the seen ball barely moves."""
    step = np.sqrt((np.diff(ball, axis=0) ** 2).sum(axis=1)) * config.FPS     # m/s
    slow = np.concatenate(([False], np.nan_to_num(step, nan=np.inf) < config.STILL_SPEED))
    out = np.zeros(len(slow), dtype=bool)
    start = None
    for i in range(len(slow) + 1):
        if i < len(slow) and slow[i]:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_frames:
                out[start:i] = True
            start = None
    return out


def near_set_piece(n_frames, set_piece_starts):
    """(N,) True within the window around each SET PIECES tag start."""
    out = np.zeros(n_frames, dtype=bool)
    before = int(config.SET_PIECE_BEFORE_S * config.FPS)
    after = int(config.SET_PIECE_AFTER_S * config.FPS)
    for start in set_piece_starts:
        out[max(0, start - before):max(0, min(n_frames, start + after))] = True
    return out


def dead_evidence(ball, speed=None, set_piece_starts=(), pitch=None):
    """Each signal's log-odds contribution, by name. All arrays are (N,)."""
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    pitch = pitch or GAMES[GAME]["pitch"]
    out = {}

    if speed is not None:
        smooth = moving_average(team_speed(speed), config.SPEED_WINDOW)
        term = np.clip((config.WALK_SPEED - smooth) / config.SPEED_SCALE, -2.0, 2.0)
        out["speed"] = config.W_SPEED * np.nan_to_num(term, nan=0.0)

    outside = ball_outside(ball, pitch)
    latched = latch_out(outside, seen & ~outside, config.OUT_LATCH_FRAMES)
    out["outside"] = config.W_OUTSIDE * latched
    out["still"] = config.W_STILL * ball_still(ball, config.STILL_FRAMES)
    out["set_piece"] = config.W_SET_PIECE * near_set_piece(n, set_piece_starts)
    out["missing"] = config.W_MISSING * ~seen
    return out


def dead_probability(ball, speed=None, set_piece_starts=(), pitch=None, forced_dead=()):
    """(N,) float in [0, 1]. `forced_dead` is [(first, stop)] spans known to be dead, such as the
    graded frames before a kick-off."""
    logit = np.full(len(ball), config.DEAD_PRIOR_LOGIT, dtype=np.float64)
    for term in dead_evidence(ball, speed, set_piece_starts, pitch).values():
        logit += term
    p = 1.0 / (1.0 + np.exp(-logit))
    for first, stop in forced_dead:
        p[first:stop] = 1.0
    return p.astype(np.float32)
