"""Metrica's public sample games as validation data: tracking in, H/A/D/X ground truth out.

The ground truth is derived from the games' *event* data, which is hand-annotated -- allowed, since
these are other games (RULES.md). It follows the challenge's own possession rules (CHALLENGE.md):

  * The ball is dead from the moment play stops -- it crosses a line (BALL OUT, a shot that goes
    out or in), or the whistle goes for a foul or offside -- until the next restart: a SET PIECE,
    or a goal kick, which Games 1 and 2 log as a PASS / BALL LOST with a GOAL KICK subtype. Where
    no explicit stop is logged before a restart, play stopped at the last on-ball event.
  * While the ball is alive, it belongs to the team of the last *control* event: a pass, carry,
    shot, recovery, restart, or a ball lost / ball out by the team that had it. Challenges are not
    control -- a won duel or a touch does not change possession -- so the ball stays with the last
    team that had it through a contested phase.
  * Graded frames run from each period's kick-off to its last event; everything else is X.

Event frames are 1-based, as in the tracking; every array here is 0-based (index = frame - 1).
"""

import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICA_DIR = os.path.join(ROOT, "data", "external", "metrica")
CACHE_DIR = os.path.join(ROOT, "data", "external", "cache")

GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
PITCH = (105.0, 68.0)
FPS = 25

CONTROL = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT"}
NOT_ON_BALL = {"CARD"}


def _event(team, kind, subtype, period, start, end):
    start = int(start) - 1
    end = int(end) - 1 if end not in (None, "") else start
    return {"team": team, "type": kind.strip().upper(), "subtype": (subtype or "").strip().upper(),
            "period": int(period), "start": start, "end": max(start, end)}


def load_events(game):
    """[{team: 'H'/'A'/None, type, subtype, period, start, end}] in file order."""
    folder = os.path.join(METRICA_DIR, game)
    path = os.path.join(folder, game + "_RawEventsData.csv")
    side = {"home": "H", "away": "A", "team a": "H", "team b": "A"}
    out = []
    if os.path.isfile(path):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                out.append(_event(side.get(row["Team"].strip().lower()), row["Type"],
                                  row["Subtype"], row["Period"], row["Start Frame"],
                                  row["End Frame"]))
        return out

    # Game 3: Metrica's JSON export. Team A is listed first in the metadata, i.e. home.
    with open(os.path.join(folder, game + "_events.json")) as handle:
        data = json.load(handle)["data"]
    for item in data:
        subtypes = item.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        team = (item.get("team") or {}).get("name") or ""
        out.append(_event(side.get(team.strip().lower()), item["type"]["name"],
                          "-".join(s["name"] for s in subtypes), item["period"],
                          item["start"]["frame"], item["end"]["frame"]))
    return out


def is_restart(event):
    return event["type"] == "SET PIECE" or "GOAL KICK" in event["subtype"]


def ground_truth(events, n_frames):
    """(N,) array of 'H' / 'A' / 'D' / 'X' from the event log, by the rules in the module doc."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for period in sorted({e["period"] for e in events}):
        evs = sorted((e for e in events if e["period"] == period), key=lambda e: e["start"])
        kickoff = next(e for e in evs if is_restart(e))
        first = kickoff["start"]
        stop = min(n_frames, max(e["end"] for e in evs) + 1)

        # Possession: last control event wins, from its start frame on.
        team = np.full(stop - first, "", dtype="<U1")
        for e in evs:
            if e["type"] in CONTROL and e["team"] and e["start"] >= first:
                team[e["start"] - first:] = e["team"]
        team[team == ""] = kickoff["team"] or "H"
        labels[first:stop] = team

        # Stoppages: from where the last on-ball event ended (the ball crossing the line, the foul,
        # the offside reception) to the restart.
        last_end = None
        for e in evs:
            if e["start"] < first or e["type"] in NOT_ON_BALL:
                continue
            if is_restart(e) and e is not kickoff and last_end is not None:
                labels[last_end:e["start"]] = "D"
            last_end = e["end"]
    return labels


def _tracking_from_kloppy(game):
    from kloppy import metrica

    folder = os.path.join(METRICA_DIR, game)
    csv_home = os.path.join(folder, game + "_RawTrackingData_Home_Team.csv")
    if os.path.isfile(csv_home):
        dataset = metrica.load_tracking_csv(
            home_data=csv_home,
            away_data=os.path.join(folder, game + "_RawTrackingData_Away_Team.csv"))
    else:
        dataset = metrica.load_tracking_epts(
            meta_data=os.path.join(folder, game + "_metadata.xml"),
            raw_data=os.path.join(folder, game + "_tracking.txt"))
    df = dataset.to_df()

    frames = df["frame_id"].to_numpy().astype(np.int64) - 1
    n = int(frames.max()) + 1
    length, width = PITCH
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[frames, 0] = df["ball_x"].to_numpy(dtype=float) * length
    ball[frames, 1] = df["ball_y"].to_numpy(dtype=float) * width

    slots, team = [], []
    for side, club in enumerate(dataset.metadata.teams):        # home first
        for player in club.players:
            if player.player_id + "_x" in df.columns:
                slots.append(player.player_id)
                team.append(side)
    players = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
    for k, pid in enumerate(slots):
        players[frames, k, 0] = df[pid + "_x"].to_numpy(dtype=float) * length
        players[frames, k, 1] = df[pid + "_y"].to_numpy(dtype=float) * width
    periods = np.zeros(n, dtype=np.int8)
    periods[frames] = df["period_id"].to_numpy()
    return ball, players, np.array(team, dtype=np.int8), periods


def player_speed(players, window=2):
    """(N, S) m/s from positions, centred difference smoothed over +/- `window` frames. The
    sample games carry no speed channel; the ATD does, computed by the tracker."""
    step = np.full(players.shape[:2], np.nan, dtype=np.float32)
    step[1:] = np.sqrt(((players[1:] - players[:-1]) ** 2).sum(axis=2)) * FPS
    kernel = np.ones(2 * window + 1)
    ok = ~np.isnan(step)
    total = np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, np.where(ok, step, 0))
    count = np.apply_along_axis(lambda c: np.convolve(c, kernel, "same"), 0, ok.astype(float))
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
    out[np.isnan(players[:, :, 0])] = np.nan
    return out


def load_game(game):
    """dict(ball, players, team, speed, periods, gt) for one sample game, cached as .npz."""
    cache = os.path.join(CACHE_DIR, game + ".npz")
    if os.path.isfile(cache):
        data = dict(np.load(cache))
    else:
        ball, players, team, periods = _tracking_from_kloppy(game)
        data = {"ball": ball, "players": players, "team": team, "periods": periods,
                "speed": player_speed(players)}
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, **data)
    data["gt"] = ground_truth(load_events(game), len(data["ball"]))
    return data
