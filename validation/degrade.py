"""Make a clean sample game look like the provided automatic tracking (ATD).

The sample games already lose the ball, but almost only while it is dead; in play it is seen on
84-93% of frames. Two things are added, both seeded so every run is reproducible:

  * ball blackouts that start in open play, with the length profile of the BAR-ATH tactical ATD
    (gaps over 3 s: median 11.6 s, 90th percentile 37 s), until a target share of alive frames has
    no ball. On BAR-ATH our own predicted-alive frames lack the ball 16% of the time; the default
    target of 25% is deliberately harder than that. Likewise for dead frames, up to the 66% that
    BAR-ATH's predicted-dead frames lack the ball: the Metrica games already lose it on ~95% of
    them, the IDSSE games (official DFL tracking) never do.
  * a smoothed ball trajectory. The ATD's ball speeds are normal (99th percentile 24.5 m/s) but
    its changes of velocity are smeared out in time: over +/- 3 frames the 99th percentile change
    is 8.8 m/s against 18-20 m/s in the sample games, over +/- 10 frames they agree (~21 m/s).
    A Gaussian of 5 frames (0.2 s) brings the short-scale figure to 9.2-9.7 m/s; without it
    every touch looks sharper than the ATD can show.
  * player drop-outs: each player vanishes for spells, so that ~14% of player-frames are missing
    (the ATD tracks 18.9 moving players of 22 on average). Speed goes with the position.
  * ball position error. In Sample Games 1 and 2 the ball sits within 10 cm of the nearest player
    on a quarter of all frames -- it is attached to whoever has it -- while in the ATD that
    distance has a 10th percentile of 0.45 m and a median of 2 m. Without this, any control
    radius looks good down to a few centimetres. Its size is set so the low percentiles of the
    ball-to-nearest-player distance match the ATD's (see nearest_percentiles()), and its
    smoothness so the ball's frame-to-frame jitter does too: the ATD ball is smooth (median
    second difference 0.009 m), an offset that drifts, not one that shakes. A first version used
    an AR(1) error, which looked right on the distance and was fifty times too jittery -- enough
    to drown every touch and every placed ball in the Metrica games.

Nothing here reads the BAR-ATH ground truth: the numbers are properties of its tracking.
"""

import numpy as np

FPS = 25
GAP_MEDIAN_S = 11.6
GAP_P90_S = 37.2
GAP_RANGE_S = (3.0, 220.0)
ALIVE_MISSING = 0.25
DEAD_MISSING = 0.66
PLAYER_MISSING = 0.14
PLAYER_SPELL_S = 5.0
BALL_ERROR_M = 0.7           # per-axis std: matches the ATD p5 and p10 (nearest_percentiles)
BALL_ERROR_TAU_S = 0.4       # Gaussian smoothing std: second difference 0.008 m on Sample Game 1
# Ball-to-nearest-player distance in the BAR-ATH tactical ATD, graded frames with a ball.
ATD_NEAREST_PERCENTILES = {5: 0.28, 10: 0.45, 25: 0.90}
ATD_SECOND_DIFF_M = 0.009    # median |b(t+1) - 2 b(t) + b(t-1)| of the BAR-ATH ATD ball
BALL_SMOOTH_FRAMES = 5       # Gaussian std; ATD jump p99 over +/- 3 frames: 8.8 m/s


def ball_smoothing(ball, frames=BALL_SMOOTH_FRAMES):
    """Copy of `ball` with its trajectory smoothed by a NaN-aware Gaussian of std `frames`.
    Missing frames stay missing."""
    if frames <= 0:
        return ball.copy()
    t = np.arange(-int(4 * frames), int(4 * frames) + 1)
    kernel = np.exp(-0.5 * (t / frames) ** 2)
    seen = ~np.isnan(ball[:, 0])
    weight = np.convolve(seen.astype(float), kernel, "same")
    out = np.full_like(ball, np.nan)
    for i in (0, 1):
        total = np.convolve(np.where(seen, ball[:, i], 0.0), kernel, "same")
        out[seen, i] = total[seen] / weight[seen]
    return out


def ball_error(ball, rng, sigma=BALL_ERROR_M, tau_s=BALL_ERROR_TAU_S):
    """Copy of `ball` plus a smooth position error of per-axis std `sigma` metres: white noise
    smoothed by a Gaussian of std `tau_s`, rescaled to `sigma`."""
    n = len(ball)
    tau = tau_s * FPS
    t = np.arange(-int(4 * tau), int(4 * tau) + 1)
    kernel = np.exp(-0.5 * (t / tau) ** 2)
    white = rng.normal(size=(n, 2))
    error = np.stack([np.convolve(white[:, i], kernel, "same") for i in (0, 1)], axis=1)
    error *= sigma / error.std(axis=0)
    return (ball + error).astype(ball.dtype)


def nearest_percentiles(ball, players, mask, qs=tuple(ATD_NEAREST_PERCENTILES)):
    delta = players - ball[:, None, :]
    dist = np.sqrt((delta ** 2).sum(axis=2))
    nearest = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    nearest = nearest[mask & np.isfinite(nearest)]
    return {q: float(np.percentile(nearest, q)) for q in qs}


def calibrated_error(ball, players, mask, grid=np.arange(0.0, 1.31, 0.1)):
    """The ball error (per-axis std, metres) that brings this game's 10th percentile of the
    ball-to-nearest-player distance closest to the ATD's. 0 for a game already as loose as the
    ATD -- the IDSSE games are; the Metrica games need ~0.7 m. Seeded, so deterministic."""
    target = ATD_NEAREST_PERCENTILES[10]
    if nearest_percentiles(ball, players, mask, (10,))[10] >= target:
        return 0.0
    gaps = [abs(nearest_percentiles(ball_error(ball, np.random.default_rng(0), s), players, mask,
                                    (10,))[10] - target) for s in grid]
    return float(grid[int(np.argmin(gaps))])


def ball_blackouts(ball, frames, rng, target=ALIVE_MISSING, confine=False):
    """Copy of `ball` with lognormal blackouts started in `frames` (a mask) until `target` of
    them are missing. With `confine`, a blackout only blanks frames inside the mask."""
    ball = ball.copy()
    sigma = np.log(GAP_P90_S / GAP_MEDIAN_S) / 1.2816
    mu = np.log(GAP_MEDIAN_S * FPS)
    lo, hi = (s * FPS for s in GAP_RANGE_S)
    starts = np.flatnonzero(frames)
    missing = np.isnan(ball[:, 0])
    n = frames.sum()
    while (missing & frames).sum() < target * n:
        length = int(np.clip(rng.lognormal(mu, sigma), lo, hi))
        start = int(rng.choice(starts))
        span = slice(start, start + length)
        blank = frames[span] if confine else np.ones(len(ball[span]), dtype=bool)
        ball[span][blank] = np.nan
        missing[span] |= blank
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
