"""End to end: data bundle in, 199,319 labels out. Deterministic, no randomness."""

import numpy as np

from games import GAME, GAMES, half_video_blocks
from . import config, learned, shape
from .deadball import ball_outside, dead_probability
from .features import ball_touches, control_vote, interpolate_ball, team_distances
from .states import (A, D, H, LETTERS, drop_short_runs, snap_handovers, viterbi,
                     viterbi_fast)
from .tagging import set_piece_starts, tag_side
from .tracking import load_tracking


def log_emissions(d_home, d_away, confidence, p_dead, side=None, shape_z=None, touch=None):
    """(N, 3) log-evidence for H, A, D from distances to the ball and the dead-ball prior.

    The team vote enters as a likelihood ratio (2 * vote), so an uninformative frame -- no ball,
    vote 0.5 -- adds nothing to H or A instead of halving both and handing the frame to D.

    Where the ball gives no evidence at all, two weak hints take over, added in log-odds:
    `side`, the Smart Tagging hint (+1 home, -1 away, 0 none) -- phases, not possessions, so they
    overrun hand-overs and are trusted only when they disagree with the current team for a while --
    and `shape_z`, the team-shape log-odds (shape.py), tempered by SHAPE_WEIGHT because
    neighbouring frames of it are anything but independent.

    `touch` (+1 / -1 / 0, features.ball_touches) adds W_TOUCH log-odds for the touching team on
    the frame of the touch: a reception or a pass is sharper evidence than standing near the ball,
    and it is what lets a short possession pay for its two switches.
    """
    vote = control_vote(d_home, d_away, confidence, config.CONTROL_RADIUS,
                        config.CONTROL_CERTAINTY)
    blind = confidence == 0
    z = np.zeros(int(blind.sum()))
    if side is not None:
        tag = np.log(config.TAG_CERTAINTY / (1.0 - config.TAG_CERTAINTY))
        z += tag * side[blind]
    if shape_z is not None:
        z += config.SHAPE_WEIGHT * shape_z[blind]
    if side is not None or shape_z is not None:
        vote[blind] = 1.0 / (1.0 + np.exp(-z))
    if touch is not None and config.W_TOUCH > 0:
        hit = touch != 0
        logit = np.log(vote[hit] / (1.0 - vote[hit])) + config.W_TOUCH * touch[hit]
        vote[hit] = 1.0 / (1.0 + np.exp(-logit))
    eps = 1e-6
    alive = 1.0 - p_dead
    out = np.empty((len(vote), 3), dtype=np.float64)
    out[:, H] = np.log(alive * 2.0 * vote + eps)
    out[:, A] = np.log(alive * 2.0 * (1.0 - vote) + eps)
    out[:, D] = np.log(p_dead + eps)
    return out


def predict(ball, players, team, game=GAME, speed=None, set_piece_starts=(), side=None):
    """Labels for every frame of a provided game: viterbi inside the graded halves, `X` elsewhere."""
    blocks = [(first, last + 1) for _, first, last in half_video_blocks(game)]
    # The 2nd half is graded from the first frame of the block but kicks off later: all `D`.
    forced_dead = [(blocks[1][0], config.SECOND_HALF_KICKOFF)]
    return predict_blocks(ball, players, team, blocks, GAMES[game]["pitch"], speed,
                          set_piece_starts, side, forced_dead)


def predict_blocks(ball, players, team, blocks, pitch, speed=None, set_piece_starts=(),
                   side=None, forced_dead=()):
    """The same model for any game: arrays indexed by frame, `blocks` the [(first, stop)] spans to
    label (one per half; the state model never runs across a gap), `X` everywhere else."""
    if speed is not None:
        # Slots the tracker gives no speed for include two parked at a fixed spot near each goal;
        # left in, they "control" every ball that comes near that spot.
        players = np.where(np.isnan(speed)[:, :, None], np.nan, players)
    if config.MODEL == "learned":
        return predict_blocks_learned(ball, players, team, blocks, pitch, speed, side,
                                      forced_dead)
    filled, confidence = interpolate_ball(ball, config.MAX_BALL_GAP)
    d_home, d_away = team_distances(filled, players, team)
    p_dead = dead_probability(ball, speed, set_piece_starts, pitch, forced_dead)
    shape_z = None
    if config.SHAPE_WEIGHT > 0:
        sign = shape.attack_sign(players, team, blocks, pitch[0])
        shape_z = shape.shape_logit(shape.features(players, team, sign, pitch[0]),
                                    config.SHAPE_COEF)
    touch = None
    if config.W_TOUCH > 0:
        touch = ball_touches(ball, d_home, d_away, config.TOUCH_JUMP, config.TOUCH_RADIUS,
                             int(config.TOUCH_LAG))
    emission = log_emissions(d_home, d_away, confidence, p_dead, side, shape_z, touch)

    control = control_vote(d_home, d_away, confidence, config.CONTROL_RADIUS,
                           config.CONTROL_CERTAINTY)
    home_control, away_control = control > 0.5, control < 0.5
    if touch is not None:
        home_control |= touch > 0
        away_control |= touch < 0

    labels = np.full(len(ball), "X", dtype="<U1")
    for first, stop in blocks:
        path = viterbi(emission[first:stop], config.SWITCH_HOME_AWAY, config.SWITCH_DEAD)
        path = drop_short_runs(path, D, int(config.MIN_DEAD_S * config.FPS))
        if config.SNAP_MAX_S > 0:
            path = snap_handovers(path, home_control[first:stop], away_control[first:stop],
                                  int(config.SNAP_MAX_S * config.FPS))
        labels[first:stop] = LETTERS[path]
    return labels


_MODELS = {}


def predict_blocks_learned(ball, players, team, blocks, pitch, speed, side=None,
                           forced_dead=()):
    """predict_blocks() with the learned evidence: P(dead) and P(home | alive) from the stored
    trees, a Viterbi against flicker, and stoppages too short to be real dropped."""
    if speed is None:
        raise ValueError("the learned model needs player speeds")
    if "models" not in _MODELS:
        _MODELS["models"] = learned.load_model()
    p_dead, p_home = learned.probabilities(_MODELS["models"], ball, players, team, speed, pitch,
                                           blocks)
    eps = 1e-4
    z_dead = np.log(np.clip(p_dead, eps, 1 - eps) / np.clip(1 - p_dead, eps, 1 - eps))
    z_dead += config.LEARNED_W_OUTSIDE * ball_outside(ball, pitch)
    p_dead = 1.0 / (1.0 + np.exp(-z_dead))
    for first, stop in forced_dead:
        p_dead[first:stop] = 1.0
    p_dead = np.clip(p_dead, eps, 1 - eps)
    p_home = np.clip(p_home, eps, 1 - eps)
    if side is not None and config.LEARNED_W_TAG > 0:
        # Smart Tagging phases as a weak hint, only where the ball is not seen (see config).
        blind = interpolate_ball(ball, config.MAX_BALL_GAP)[1] == 0
        z_home = np.log(p_home / (1 - p_home)) + config.LEARNED_W_TAG * side * blind
        p_home = np.clip(1.0 / (1.0 + np.exp(-z_home)), eps, 1 - eps)
    emission = np.empty((len(ball), 3), dtype=np.float64)
    emission[:, H] = np.log(1 - p_dead) + np.log(p_home)
    emission[:, A] = np.log(1 - p_dead) + np.log(1 - p_home)
    emission[:, D] = np.log(p_dead)

    labels = np.full(len(ball), "X", dtype="<U1")
    for first, stop in blocks:
        path = viterbi_fast(emission[first:stop], config.LEARNED_SWITCH_HOME_AWAY,
                            config.LEARNED_SWITCH_DEAD)
        path = drop_short_runs(path, D, int(config.LEARNED_MIN_DEAD_S * config.FPS))
        labels[first:stop] = LETTERS[path]
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
