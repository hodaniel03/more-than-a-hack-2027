"""Make a clean sample game look like the provided automatic tracking (ATD).

The sample games already lose the ball, but almost only while it is dead; in play it is seen on
84-93% of frames. Two things are added, both seeded so every run is reproducible:

  * ball blackouts that start in open play, with the length profile of the BAR-ATH tactical ATD
    (gaps over 3 s: median 11.6 s, 90th percentile 37 s), until a target share of alive frames has
    no ball. On BAR-ATH our own predicted-alive frames lack the ball 16% of the time; the default
    target of 25% is deliberately harder than that.
  * player drop-outs: each player vanishes for spells, so that ~14% of player-frames are missing
    (the ATD tracks 18.9 moving players of 22 on average). Speed goes with the position.
  * ball position error. In Sample Games 1 and 2 the ball sits within 10 cm of the nearest player
    on a quarter of all frames -- it is attached to whoever has it -- while in the ATD that
    distance has a 10th percentile of 0.45 m and a median of 2 m. Without this, any control
    radius looks good down to a few centimetres. The error is smooth in time (tracking error
    drifts, it does not jump), and its size is set so the low percentiles of the ball-to-nearest-
    player distance match the ATD's (see nearest_percentiles()).

Nothing here reads the BAR-ATH ground truth: the numbers are properties of its tracking.
"""

import numpy as np

FPS = 25
GAP_MEDIAN_S = 11.6
GAP_P90_S = 37.2
GAP_RANGE_S = (3.0, 220.0)
ALIVE_MISSING = 0.25
PLAYER_MISSING = 0.14
PLAYER_SPELL_S = 5.0
BALL_ERROR_M = 0.7           # per-axis std: matches the ATD p5 and p10 (nearest_percentiles)
BALL_ERROR_TAU_S = 0.4       # correlation time of the error
# Ball-to-nearest-player distance in the BAR-ATH tactical ATD, graded frames with a ball.
ATD_NEAREST_PERCENTILES = {5: 0.28, 10: 0.45, 25: 0.90}


def ball_error(ball, rng, sigma=BALL_ERROR_M, tau_s=BALL_ERROR_TAU_S):
    """Copy of `ball` plus a smooth (AR(1)) position error of per-axis std `sigma` metres."""
    n = len(ball)
    rho = np.exp(-1.0 / (tau_s * FPS))
    shocks = rng.normal(0.0, sigma * np.sqrt(1 - rho ** 2), size=(n, 2))
    error = np.empty((n, 2))
    error[0] = rng.normal(0.0, sigma, size=2)
    for t in range(1, n):
        error[t] = rho * error[t - 1] + shocks[t]
    return (ball + error).astype(ball.dtype)


def nearest_percentiles(ball, players, mask, qs=tuple(ATD_NEAREST_PERCENTILES)):
    delta = players - ball[:, None, :]
    dist = np.sqrt((delta ** 2).sum(axis=2))
    nearest = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    nearest = nearest[mask & np.isfinite(nearest)]
    return {q: float(np.percentile(nearest, q)) for q in qs}


def ball_blackouts(ball, alive, rng, target=ALIVE_MISSING):
    """Copy of `ball` with lognormal blackouts started in alive frames until `target` of them are
    missing."""
    ball = ball.copy()
    sigma = np.log(GAP_P90_S / GAP_MEDIAN_S) / 1.2816
    mu = np.log(GAP_MEDIAN_S * FPS)
    lo, hi = (s * FPS for s in GAP_RANGE_S)
    starts = np.flatnonzero(alive)
    missing = np.isnan(ball[:, 0])
    n_alive = alive.sum()
    while (missing & alive).sum() < target * n_alive:
        length = int(np.clip(rng.lognormal(mu, sigma), lo, hi))
        start = int(rng.choice(starts))
        ball[start:start + length] = np.nan
        missing[start:start + length] = True
    return ball


def player_dropouts(players, speed, rng, share=PLAYER_MISSING, spell_s=PLAYER_SPELL_S):
    """Copies of `players` and `speed` with each slot alternating present / missing spells, the
    missing ones exponential with mean `spell_s`, so that `share` of frames are missing."""
    players = players.copy()
    speed = speed.copy()
    n, slots = speed.shape
    gone_mean = spell_s * FPS
    here_mean = gone_mean * (1 - share) / share
    for k in range(slots):
        t = int(rng.exponential(here_mean))
        while t < n:
            length = max(1, int(rng.exponential(gone_mean)))
            players[t:t + length, k] = np.nan
            speed[t:t + length, k] = np.nan
            t += length + max(1, int(rng.exponential(here_mean)))
    return players, speed
