"""Synthetic Smart Tagging for the validation games, with the quality BAR-ATH's real tags show.

The real tags exist for BAR-ATH only, so their weight cannot be validated directly. These copies
behave like them by the measures validation/tag_quality.py takes on BAR-ATH without its ground
truth, so the pipeline's use of tags can be tested -- and how much it depends on their quality:

  team phases   cover ~67% of alive frames and name the right team on ~85% of what they cover.
                A phase follows a true possession of at least PHASE_MIN_S, with its edges
                jittered, and some phases run on past the hand-over (the real ones are phases,
                not possessions: CREATING CHANCES is always 30 s long).
  set pieces    the real restart comes a median 5.6 s after a tag starts (middle half 3.3 to
                8.2 s); ~62% of stoppages have a tag; ~31% of tags are not near any stoppage.

This tests a model of the tags, not the tags: it says what a given tag quality is worth, not
that BAR-ATH's tags have exactly that quality.
"""

import numpy as np

FPS = 25
PHASE_MIN_S = 4.0
EDGE_JITTER_S = 2.0
OVERRUN_SHARE = 0.35          # phases that run on into the next possession
OVERRUN_S = (3.0, 15.0)
KEEP = 0.80                   # possessions that get a phase at all
FLIP = 0.10                   # phases that name the wrong team
# (KEEP, FLIP) = (0.80, 0.10) gives coverage 0.68 and accuracy ~0.85 on the dev games.
RESTART_AFTER_S = (5.6, 3.6)  # mean, std of restart minus tag start
STOP_COVER = 0.62
FALSE_TAGS = 0.45             # false set-piece tags per true one


def spells(gt):
    change = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(gt)]))
    return [(int(a), int(b), gt[a]) for a, b in zip(bounds[:-1], bounds[1:])]


def team_side(gt, rng, keep=KEEP, flip=FLIP, phase_min_s=PHASE_MIN_S,
              overrun_share=OVERRUN_SHARE):
    """(N,) int8 +1 / -1 / 0 like tagging.team_side, from the true possessions."""
    n = len(gt)
    home = np.zeros(n, dtype=np.int16)
    away = np.zeros(n, dtype=np.int16)
    for a, b, lab in spells(gt):
        if lab not in ("H", "A") or (b - a) < phase_min_s * FPS or rng.random() > keep:
            continue
        if rng.random() < flip:
            lab = "A" if lab == "H" else "H"
        start = a + int(rng.normal(0, EDGE_JITTER_S) * FPS)
        stop = b + int(rng.normal(0, EDGE_JITTER_S) * FPS)
        if rng.random() < overrun_share:
            stop = b + int(rng.uniform(*OVERRUN_S) * FPS)
        start, stop = max(0, start), min(n, stop)
        (home if lab == "H" else away)[start:stop] += 1
    out = np.zeros(n, dtype=np.int8)
    out[(home > 0) & (away == 0)] = 1
    out[(away > 0) & (home == 0)] = -1
    out[gt == "X"] = 0
    return out


def set_piece_starts(gt, rng, stop_cover=STOP_COVER, false_tags=FALSE_TAGS):
    """Sorted tag starts: before a share of the true restarts, plus some false ones."""
    restarts = np.flatnonzero((gt[:-1] == "D") & np.isin(gt[1:], ("H", "A"))) + 1
    true = [int(r - max(0.0, rng.normal(*RESTART_AFTER_S)) * FPS)
            for r in restarts if rng.random() < stop_cover]
    alive = np.flatnonzero(np.isin(gt, ("H", "A")))
    false = rng.choice(alive, int(round(false_tags * len(true))), replace=False).tolist()
    return sorted(max(0, s) for s in true + false)


def quality(gt, side):
    """(coverage of alive frames, share of covered alive frames with the right team)."""
    alive = np.isin(gt, ("H", "A"))
    covered = alive & (side != 0)
    right = (side[covered] > 0) == (gt[covered] == "H")
    return float(covered.sum() / alive.sum()), float(right.mean())
