"""End to end: data bundle in, 199,319 labels out. Deterministic, no randomness."""

import numpy as np

from games import GAME, GAMES, half_video_blocks
from . import config
from .deadball import dead_probability
from .features import control_vote, interpolate_ball, team_distances
from .states import A, D, H, LETTERS, drop_short_runs, viterbi
from .tagging import set_piece_starts, tag_side
from .tracking import load_tracking


def log_emissions(d_home, d_away, confidence, p_dead, side=None):
    """(N, 3) log-evidence for H, A, D from distances to the ball and the dead-ball prior.

    The team vote enters as a likelihood ratio (2 * vote), so an uninformative frame -- no ball,
    vote 0.5 -- adds nothing to H or A instead of halving both and handing the frame to D.

    `side` is the Smart Tagging hint (+1 home, -1 away, 0 none). It is used only where the ball
    gives no evidence at all, and weakly: the tags are phases, not possessions, so they overrun
    hand-overs and are trusted only when they disagree with the current team for a while.
    """
    vote = control_vote(d_home, d_away, confidence, config.CONTROL_RADIUS,
                        config.CONTROL_CERTAINTY)
    if side is not None:
        blind = confidence == 0
        vote[blind] = 0.5 + (config.TAG_CERTAINTY - 0.5) * side[blind]
    eps = 1e-6
    alive = 1.0 - p_dead
    out = np.empty((len(vote), 3), dtype=np.float64)
    out[:, H] = np.log(alive * 2.0 * vote + eps)
    out[:, A] = np.log(alive * 2.0 * (1.0 - vote) + eps)
    out[:, D] = np.log(p_dead + eps)
    return out


def predict(ball, players, team, game=GAME, speed=None, set_piece_starts=(), side=None):
    """Labels for every frame: viterbi inside the graded halves, `X` elsewhere."""
    n = GAMES[game]["n_video_frames"]
    if speed is not None:
        # Slots the tracker gives no speed for include two parked at a fixed spot near each goal;
        # left in, they "control" every ball that comes near that spot.
        players = np.where(np.isnan(speed)[:, :, None], np.nan, players)
    filled, confidence = interpolate_ball(ball, config.MAX_BALL_GAP)
    d_home, d_away = team_distances(filled, players, team)
    emission = log_emissions(d_home, d_away, confidence,
                             dead_probability(ball, speed, set_piece_starts, game), side)

    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        path = viterbi(emission[first:last + 1], config.SWITCH_HOME_AWAY, config.SWITCH_DEAD)
        path = drop_short_runs(path, D, int(config.MIN_DEAD_S * config.FPS))
        labels[first:last + 1] = LETTERS[path]
    return labels


def write_csv(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def run(out_path, game=GAME, data_dir=None):
    ball, players, team, speed = load_tracking(game, data_dir)
    labels = predict(ball, players, team, game, speed, set_piece_starts(game, data_dir),
                     tag_side(game, data_dir))
    write_csv(out_path, labels)
    return labels
