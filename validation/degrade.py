"""Make a clean sample game look like the provided automatic tracking (ATD).

The sample games already lose the ball, but almost only while it is dead; in play it is seen on
84-93% of frames. Two things are added, both seeded so every run is reproducible:

  * ball blackouts that start in open play, with the length profile of the BAR-ATH tactical ATD
    (gaps over 3 s: median 11.6 s, 90th percentile 37 s), until a target share of alive frames has
    no ball. On BAR-ATH our own predicted-alive frames lack the ball 16% of the time; the default
    target of 25% is deliberately harder than that.
  * player drop-outs: each player vanishes for spells, so that ~14% of player-frames are missing
    (the ATD tracks 18.9 moving players of 22 on average). Speed goes with the position.

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
