"""Every tunable number in one place.

Rule from RULES.md: none of these may be tuned on the public 5 minutes. Choose them from football
reasoning or from other games, and write the reason next to the value. The values below are
starting points, not results.
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
W_SPEED = 1.5
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
MIN_DEAD_S = 3.0

# --- Team in control (features.control_vote). ---

# A player in control has the ball within a stride; automatic tracking places a player by the
# centre of the body and adds about half a metre of noise on top.
CONTROL_RADIUS = 1.5
# How often an uncontested "my player is next to the ball" frame really is that team's ball.
# Tracking noise and identity swaps make roughly one frame in ten wrong.
CONTROL_CERTAINTY = 0.9

# A change of possession needs *controlled* possession (CHALLENGE.md): a touch, a deflection or a
# won header is not one. Asking for a second of the new team's control before switching -- one
# player holding it, or a reception plus a pass to a teammate -- sets the H<->A cost:
# each frame of clean control is worth log(0.9 / 0.1) in the Viterbi.
MIN_CONTROL_S = 1.0

# Viterbi switch costs (log-probability units). Larger = fewer, longer possessions.
SWITCH_HOME_AWAY = MIN_CONTROL_S * FPS * math.log(CONTROL_CERTAINTY / (1 - CONTROL_CERTAINTY))
SWITCH_DEAD = 8.0

# Kickoff of the 2nd half; frames from the start of the graded block to here are all `D`
# (FAQ.md). Source: starter/load_smart_tagging.py KICKOFF_VIDEO_FRAMES.
SECOND_HALF_KICKOFF = 113742
