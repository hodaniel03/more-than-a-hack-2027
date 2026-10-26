"""Turn noisy per-frame evidence into long, stable segments (Viterbi with switch costs)."""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def viterbi(log_emission, switch_home_away, switch_dead):
    """Most likely H/A/D path. log_emission is (N, 3); returns (N,) ints.

    Staying in a state is free; H<->A costs `switch_home_away`; entering or leaving D costs
    `switch_dead`. Large costs mean few, long possessions -- the thing spellF1 rewards.
    """
    n = len(log_emission)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    cost = np.array([[0, switch_home_away, switch_dead],
                     [switch_home_away, 0, switch_dead],
                     [switch_dead, switch_dead, 0]], dtype=np.float64)

    score = log_emission[0].astype(np.float64)
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        candidate = score[:, None] - cost          # [from, to]
        back[t] = np.argmax(candidate, axis=0)
        score = candidate[back[t], np.arange(3)] + log_emission[t]

    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap_handovers(path, home_control, away_control, max_shift):
    """Move each direct H<->A switch to where the new team takes control.

    Between the old team's last touch and the new team's first one -- a pass in flight, a loose
    ball -- nobody is in control, so the Viterbi path may switch anywhere in that stretch. The
    possession rules say the ball stays with the last team until the other one controls it, so
    within +/- `max_shift` frames (and never past the neighbouring switches) the switch goes to the
    split that agrees best with the control evidence, the latest such split on a tie.
    """
    path = path.copy()
    n = len(path)
    change = np.flatnonzero(path[1:] != path[:-1]) + 1
    bounds = np.concatenate(([0], change, [n]))
    ctrl = {H: home_control, A: away_control}
    left = 0                                           # where the current run really starts
    for k in range(1, len(bounds) - 1):
        i = bounds[k]
        old, new = path[i - 1], path[i]
        if D in (old, new):
            left = i
            continue
        a = max(left, i - max_shift)
        b = min(bounds[k + 1], i + max_shift)
        left = i
        old_c = np.concatenate(([0], np.cumsum(ctrl[old][a:b])))
        new_c = np.concatenate(([0], np.cumsum(ctrl[new][a:b])))
        agree = old_c + (new_c[-1] - new_c)            # old before split s, new from s on
        best = np.flatnonzero(agree == agree.max())
        best = best[(best > 0) & (best < b - a)]       # keep at least one frame of each
        if len(best):
            left = a + best[-1]
            path[a:left] = old
            path[left:b] = new
    return path


def drop_short_runs(path, state, min_len):
    """Replace runs of `state` shorter than `min_len` with the label before them (after, at the
    start). A run touching either end of the path is kept: it may be cut off, not short."""
    path = path.copy()
    n = len(path)
    i = 0
    while i < n:
        if path[i] != state:
            i += 1
            continue
        j = i
        while j < n and path[j] == state:
            j += 1
        if j - i < min_len and i > 0 and j < n:
            path[i:j] = path[i - 1]
        i = j
    return path
