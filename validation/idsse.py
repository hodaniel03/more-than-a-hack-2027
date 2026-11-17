"""The IDSSE dataset (Bassek et al., Scientific Data 2025, CC-BY 4.0) as validation data.

Seven 2022/23 Bundesliga / 2. Bundesliga matches of official DFL position data at 25 Hz. Every
frame carries the DFL's own BallStatus (alive / dead) and BallPossession (home / away), which is
the same kind of signal the challenge ground truth is derived from -- corrected tracking naming
the ball dead or the team owning it -- so no event interpretation is needed:

    D where BallStatus is dead, otherwise H or A from BallPossession.

One difference matters. The DFL flag follows every touch: a deflection or a lost duel flips it
for a few frames (a median spell of 3 s, ~90 per match under 1 s). The challenge does not count a
touch without control as a change of possession (CHALLENGE.md), so `gt` folds an A-B-A spell of B
shorter than TOUCH_S back into A. The flag as delivered is kept as `gt_raw`.

The tracking is semi-automatic optical tracking checked by operators, i.e. cleaner than the ATD,
so validation/degrade.py applies here exactly as for the Metrica games.

Download (~2.6 GB, the position files are ~350-420 MB each):  python validation/download.py --idsse
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE_DIR = os.path.join(ROOT, "data", "external", "idsse")
CACHE_DIR = os.path.join(ROOT, "data", "external", "cache")

# The download_url the figshare API (v2/articles/28196177/files) gives for each file. The
# springernature.figshare.com address kloppy uses sits behind a browser challenge.
FIGSHARE = "https://ndownloader.figshare.com/files/{}"
# match id -> (competition, meta file id, position file id), from the figshare record
# 10.6084/m9.figshare.28196177 (the same ids kloppy's sportec.load_open_tracking_data uses).
MATCHES = {
    "J03WMX": ("DFL-COM-000001", 51643475, 51643514),
    "J03WN1": ("DFL-COM-000001", 51643472, 51643517),
    "J03WOH": ("DFL-COM-000002", 51643478, 51643520),
    "J03WOY": ("DFL-COM-000002", 51643481, 51643523),
    "J03WPY": ("DFL-COM-000002", 51643487, 51643526),
    "J03WQQ": ("DFL-COM-000002", 51643484, 51643529),
    "J03WR9": ("DFL-COM-000002", 51643490, 51643532),
}


TOUCH_S = 1.0
FPS = 25


def fold_touches(gt, max_frames):
    """Relabel every H/A spell shorter than `max_frames` that sits between two spells of the other
    team (A B A) as that team: a touch, not a change of possession."""
    out = gt.copy()
    change = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(gt)]))
    for k in range(1, len(bounds) - 2):
        start, stop = bounds[k], bounds[k + 1]
        before, here, after = gt[bounds[k - 1]], gt[start], gt[stop]
        if (here in ("H", "A") and before == after and before in ("H", "A") and before != here
                and stop - start < max_frames):
            out[start:stop] = before
    return out


def paths(match):
    competition = MATCHES[match][0]
    stem = "{}_DFL-MAT-{}.xml".format(competition, match)
    return (os.path.join(IDSSE_DIR, "DFL_02_01_matchinformation_" + stem),
            os.path.join(IDSSE_DIR, "DFL_04_03_positions_raw_observed_" + stem))


def downloads():
    """[(url, local path)] for every IDSSE file this module needs."""
    out = []
    for match, (_, meta_id, pos_id) in MATCHES.items():
        meta, positions = paths(match)
        out += [(FIGSHARE.format(meta_id), meta), (FIGSHARE.format(pos_id), positions)]
    return out


def available():
    return [m for m in MATCHES if all(os.path.isfile(p) for p in paths(m))]


def load_game(match):
    """Same dict as validation.metrica.load_game: ball, players, team, speed, periods, gt."""
    cache = os.path.join(CACHE_DIR, "IDSSE_" + match + ".npz")
    if not os.path.isfile(cache):
        _build_cache(match, cache)
    data = dict(np.load(cache))
    data["gt"] = fold_touches(data["gt_raw"], int(TOUCH_S * FPS))
    return data


def _build_cache(match, cache):

    from kloppy import sportec
    from validation.metrica import player_speed

    meta, positions = paths(match)
    dataset = sportec.load_tracking(raw_data=positions, meta_data=meta,
                                    coordinates="sportec", only_alive=False)
    df = dataset.to_df()
    n = len(df)
    length = dataset.metadata.pitch_dimensions.pitch_length
    width = dataset.metadata.pitch_dimensions.pitch_width
    # Sportec coordinates are metres from the centre spot; shift to the 0..length convention.
    ball = np.stack([df["ball_x"].to_numpy(dtype=float) + length / 2,
                     df["ball_y"].to_numpy(dtype=float) + width / 2], axis=1).astype(np.float32)

    slots, team = [], []
    for side, club in enumerate(dataset.metadata.teams):        # home first
        for player in club.players:
            if player.player_id + "_x" in df.columns:
                slots.append(player.player_id)
                team.append(side)
    players = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
    for k, pid in enumerate(slots):
        players[:, k, 0] = df[pid + "_x"].to_numpy(dtype=float) + length / 2
        players[:, k, 1] = df[pid + "_y"].to_numpy(dtype=float) + width / 2

    home_id = dataset.metadata.teams[0].team_id
    state = df["ball_state"].astype(str).str.lower().to_numpy()
    owner = df["ball_owning_team_id"].astype(str).to_numpy()
    gt = np.where(owner == str(home_id), "H", "A").astype("<U1")
    gt[state != "alive"] = "D"
    periods = df["period_id"].to_numpy().astype(np.int8)
    # Graded like the challenge: from each period's first alive frame to its last frame.
    for period in np.unique(periods):
        idx = np.flatnonzero(periods == period)
        alive = idx[state[idx] == "alive"]
        gt[idx[idx < alive[0]]] = "X"

    data = {"ball": ball, "players": players, "team": np.array(team, dtype=np.int8),
            "periods": periods, "speed": player_speed(players), "gt_raw": gt,
            "pitch": np.array([length, width], dtype=np.float32)}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, **data)
