"""Every tunable number in one place.

Rule from RULES.md: none of these may be tuned on the public 5 minutes. Choose them from football
reasoning or from other games, and write the reason next to the value.

"Validated" below means checked on other games: Metrica's three public sample games, degraded to
look like the provided ATD (python validation/run.py, validation/sweep.py; see validation/). Each
value marked so was the best, or level with the best, with every game held out in turn.
"""

import math

# Ball gaps up to this many frames are bridged by linear interpolation (25 frames = 1 s).
MAX_BALL_GAP = 25

FPS = 25

# --- Dead-ball detector (deadball.py). Log-odds units: +1 is roughly "2.7x more likely dead". ---

# Base rate. Effective playing time in a top-flight match is about 55 of ~100 minutes, so the ball
# is dead a bit under half the time: log(0.4 / 0.6).
DEAD_PRIOR_LOGIT = -0.4

# Collective speed. Human walking pace is ~1.4 m/s; outfield players cover ~10-11 km per 90 min,
# i.e. ~2 m/s on average in play, while at a stoppage most of them walk or stand.
WALK_SPEED = 1.5
SPEED_SCALE = 0.4            # m/s per unit of evidence, clipped at +/-2 units
# Validated: 0.5 beat 0.75, 1.5 and 3 on all three games (spellF1 0.444 / 0.430 / 0.384 / 0.354),
# also with the missing-ball term switched off. At 0 the detector loses most stoppages, so speed is
# needed -- but heavily weighted it calls slow build-up play dead and splits the possession.
W_SPEED = 0.5
SPEED_WINDOW = 25            # +/- 1 s: a stoppage is a team-wide slowdown, not one frame
MIN_PLAYERS_FOR_SPEED = 8    # fewer tracked players than this and the median is not a team

# Ball out. Law 9: the whole ball must cross the line. 1 m covers the ball's radius plus the
# position noise of automatic tracking, so a ball on the line is not called out.
OUT_MARGIN = 1.0
W_OUTSIDE = 4.0
OUT_LATCH_FRAMES = 60 * 25   # longest a single out-of-play spell plausibly lasts

# Ball standing still: a placed ball. In play a ball is rarely below walking speed for a second.
STILL_SPEED = 0.5
STILL_FRAMES = 25
W_STILL = 1.5

# Smart Tagging SET PIECES. The tag is a fixed-length window (35 s, 5 s for kick-offs), so only its
# start says anything. The published first-half kick-off (frame 11185) falls 3.5 s after the start
# of its tag, so the restart is assumed to be within a few seconds of the tag start.
SET_PIECE_BEFORE_S = 5.0
SET_PIECE_AFTER_S = 5.0
W_SET_PIECE = 1.5

# The ball also vanishes for long spells in open play (DATA.md), so a missing ball is weak evidence.
W_MISSING = 0.3

# Shortest stoppage that is real: even a quick throw-in needs the ball fetched and taken from the
# line, which takes a few seconds. Shorter `D` runs are noise and would split a possession in two.
# Validated: 5 s beat 0, 2, 3, 7 and 10 (stoppages in the sample games have a median of 13-19 s).
# Enforcing it inside the Viterbi instead of filling short runs afterwards was tried and scored
# lower on all three games.
MIN_DEAD_S = 5.0

# --- Team in control (features.control_vote). ---

# A player in control has the ball within a stride; automatic tracking places a player by the
# centre of the body and adds about half a metre of noise on top. Validated: 1.5 m beat 0.5-2 m
# once the sample games' ball was given the ATD's position error. Without that error -- the sample
# ball sits on the controlling player's feet -- ever smaller radii look better, which would not
# carry over to the ATD.
CONTROL_RADIUS = 1.5
# How often an uncontested "my player is next to the ball" frame really is that team's ball.
# Tracking noise and identity swaps make roughly one frame in ten wrong.
CONTROL_CERTAINTY = 0.9

# A change of possession needs *controlled* possession (CHALLENGE.md): a touch, a deflection or a
# won header is not one. The new team's clean control needed before switching sets the H<->A cost:
# each frame of it is worth log(0.9 / 0.1) in the Viterbi. Validated: 0.2-0.5 s are level and
# clearly better than 0.1 s or less (0.3 s is a reception and a touch).
#
# Note the ceiling: H -> D -> A costs 2 * SWITCH_DEAD, so a switch never costs more than that
# however large this is (at SWITCH_DEAD = 8, anything above ~0.3 s behaves the same).
MIN_CONTROL_S = 0.3

# Viterbi switch costs (log-probability units). Larger = fewer, longer possessions.
SWITCH_HOME_AWAY = MIN_CONTROL_S * FPS * math.log(CONTROL_CERTAINTY / (1 - CONTROL_CERTAINTY))
SWITCH_DEAD = 8.0            # validated: 4 and 8 level, 16 worse

# A hand-over is moved to where the new team takes control (states.snap_handovers), searching at
# most this far either side: a long pass or a clearance is in the air for two to three seconds.
# 0 turns it off. Validated: +0.01 spellF1 on every game over 0; 3 s and 5 s level.
SNAP_MAX_S = 3.0

# --- Smart Tagging hint where the ball is not seen (tagging.team_side). ---

# The tags are phases, accurate to about a second, and several codes are fixed-length windows
# (CREATING CHANCES is always 30 s) that run on past the end of a possession. So a single frame of
# tag is worth little; what counts is a tag that keeps disagreeing with the current team. Asking for
# TAG_SWITCH_S of uncontradicted disagreement inside a ball blackout before switching sets the
# per-frame certainty: each tagged frame is worth SWITCH_HOME_AWAY / (TAG_SWITCH_S * FPS).
TAG_SWITCH_S = 5.0
TAG_CERTAINTY = 1.0 / (1.0 + math.exp(-SWITCH_HOME_AWAY / (TAG_SWITCH_S * FPS)))

# Kickoff of the 2nd half; frames from the start of the graded block to here are all `D`
# (FAQ.md). Source: starter/load_smart_tagging.py KICKOFF_VIDEO_FRAMES.
SECOND_HALF_KICKOFF = 113742
