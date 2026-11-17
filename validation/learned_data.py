"""Feature tables for the learned models (possession/learned.py), one file per game and seed:

    python validation/learned_data.py            # all ten games, seeds 0-2, into data/external/feat/

Each game is degraded to look like the ATD exactly as validation/run.py does it (mode "atd"), so
the models are trained on the inputs they will meet.
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import learned  # noqa: E402
from validation import metrica  # noqa: E402
from validation.run import SEEDS, blocks_of, inputs, load_games  # noqa: E402

FEAT_DIR = os.path.join(ROOT, "data", "external", "feat")


def path(game, seed):
    return os.path.join(FEAT_DIR, "{}_{}.npz".format(game, seed))


def build(job):
    game, seed = job
    out = path(game, seed)
    if os.path.isfile(out):
        return out
    data = load_games([game], "all")[game]
    blocks = blocks_of(data["gt"], data.get("periods"))
    ball, players, speed = inputs(data, "atd", seed)
    players = np.where(np.isnan(speed)[:, :, None], np.nan, players)
    pitch = tuple(float(v) for v in data["pitch"]) if "pitch" in data else metrica.PITCH
    base = learned.base_signals(ball, players, data["team"], speed, pitch)
    dn, Xd = learned.matrix(learned.dead_features(ball, players, data["team"], speed, pitch, base))
    tn, Xt = learned.matrix(learned.team_features(ball, players, data["team"], speed, pitch,
                                                  blocks, base))
    os.makedirs(FEAT_DIR, exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", gt=data["gt"], blocks=np.array(blocks), Xd=Xd, Xt=Xt,
                        dead_names=np.array(dn), team_names=np.array(tn), ball=ball,
                        players=players, team=data["team"], speed=speed, pitch=np.array(pitch))
    os.replace(out + ".tmp.npz", out)
    return out


if __name__ == "__main__":
    games = list(load_games(None, "all"))
    jobs = [(g, s) for g in games for s in SEEDS]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for done in pool.imap_unordered(build, jobs):
            print("built", os.path.relpath(done, ROOT), flush=True)
