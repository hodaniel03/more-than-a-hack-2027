# TEAM_NAME — technical document

*More than a Hack 2027, Phase 1: frame-level ball possession (BAR-ATH).*

## 1. Result

- Submission: `TEAM_NAME.csv`, 199,319 rows, passes `evaluation/validate_submission.py`.
- It reports 289 possessions of mean 13.0 s; `D` on 40.7% of graded frames; no `X`
  on a graded frame.
- On other games, with tracking degraded to look like the provided ATD (section 4), the model
  scores **spellF1 0.636 on three games it was never trained or tuned on**, against
  0.443 for our earlier rule-based model and 0.227 for the reference baseline
  (`baseline_smooth50`) on the same inputs.
- The public 5 minutes were used only to check that the pipeline is correct (frames aligned,
  labels meaning what we think). No parameter was chosen on them and nothing was trained on them.

## 2. Approach

The question is "who is in control, or is the ball dead", for every frame, from 22 anonymous
players and a ball that is missing 41% of the time. We answer it in three steps.

1. **Signals** (`possession/learned.py`, `features.py`, `shape.py`). From the tactical ATD we
   compute per-frame signals and their means over windows up to 20 s back and forward (the
   match is finished, so the future is available):
   - *reach*: which team has a player alone within 1.25 m of the ball, which team is nearer, the
     closest approach of each team in a window, how many players of each team are around the ball;
   - *touches*: sharp changes of ball velocity next to a player;
   - *ball visibility*: seen or not, how long since / until it is seen, where it was last seen
     relative to the lines;
   - *collective player speed*: median, upper quartile, share running, share walking;
   - *team shape*: territory, width difference, direction of collective movement, and the number
     of players in a penalty area.
2. **Two learned models** (gradient-boosted trees, section 4): `P(ball dead)` from the
   team-agnostic signals, and `P(home | alive)` from signals signed towards home. The second
   model is trained on every game twice, once with the teams swapped, and at inference its answer
   is averaged with the answer to the mirrored question, so it cannot prefer a side.
3. **Decoding** (`states.viterbi_fast`). A Viterbi path over H / A / D with a fixed cost for
   changing state removes single-frame flicker; stoppages shorter than 2 s are dropped. This is
   what turns per-frame probabilities into possessions, which is what spellF1 counts.

Two things are added by rule, because no training game contains them:
- a ball *seen* more than 1 m beyond the lines is dead (Law 9): +4 log-odds on those frames;
- the graded frames of the second half before its kick-off are `D` (FAQ.md).

**Why this rather than rules.** Our first model was hand-written rules with the same signals
(`config.MODEL = "rules"`, still in the repository): uncontested reach, touches, a dead-ball
detector summing log-odds, a Viterbi and a hand-over placement step. On the validation games it
scores about 0.42-0.44. The learned model uses the same signals but learns how to weigh them
together and how to use the seconds before and after a frame; that is worth about +0.19 spellF1
on games it never saw (section 4).

## 3. Assets used and dropped

| asset | used | how / why not |
|---|---|---|
| tactical ATD (players, ball, speed) | yes | every signal above |
| tactical ATD metadata | yes | team of each track slot |
| Smart Tagging | weakly | team phases as +1 log-odds for the tagged team on frames where the ball is not seen; SET PIECES not used (tested on synthetic tags: no gain) |
| tactical homography | no | the ATD is already in pitch coordinates |
| tactical video | no | not needed by the model; no own detection |
| broadcast ATD + homography | tried, dropped | synchronised to the tactical timeline automatically (constant offset per half, found by cross-correlating ball positions and refined on player positions); it adds the ball on only 3.6 percentage points more graded frames, and where both feeds see the ball they disagree by more than 5 m on 11% of frames |
| broadcast video | no | |

## 4. Training and validation data

**Nothing is trained, tuned or validated on BAR-ATH labels.** All labelled data are other games:

| dataset | games | ground truth | who labelled |
|---|---|---|---|
| [Metrica sample data](https://github.com/metrica-sports/sample-data) | 3 | H / A / D derived by our code (`validation/metrica.py`) from the published event data | Metrica Sports (events) |
| [IDSSE](https://doi.org/10.6084/m9.figshare.28196177) (Bassek et al., Sci Data 2025, CC-BY 4.0) | 7 Bundesliga / 2. Bundesliga | the DFL's per-frame BallStatus and BallPossession flags, spells under 1 s between two spells of the other team folded back (`validation/idsse.py`) | DFL / Sportec |

`python validation/download.py --idsse` fetches both (2.8 GB).

**Making them look like the ATD** (`validation/degrade.py`). The public tracking is cleaner than
the provided ATD, so before any training or scoring each game gets: ball blackouts with the gap
profile of the BAR-ATH ATD (25% of live frames, 66% of dead frames), a smoothed ball trajectory,
a ball position error calibrated to the ATD's ball-to-nearest-player distances, and player
drop-outs (14% of player-frames). These are properties of the BAR-ATH *tracking*; its ground
truth is not read.

**Split.** Seven games are "dev" (Sample Games 1-2, five IDSSE matches); three are locked as
"test" (Sample Game 3, IDSSE J03WQQ, J03WR9) and are scored only to confirm decisions already
made. Feature choices, tree settings and decoder costs were chosen leave-one-game-out on dev; the
model in `models/learned.npz` is fitted on the seven dev games only.

**Results** (spellF1, ATD-like inputs, mean over three degradation seeds):

| games | reference baseline (`smooth50`) | rule-based model | learned model |
|---|---|---|---|
| 7 dev games, leave-one-game-out | 0.234 | 0.421 | 0.607 |
| Sample Game 3 (locked test) | 0.238 | 0.429 | 0.653 |
| IDSSE J03WQQ (locked test) | 0.236 | 0.441 | 0.609 |
| IDSSE J03WR9 (locked test) | 0.209 | 0.459 | 0.647 |
| **locked test, mean** | **0.227** | **0.443** | **0.636** |

On the locked games the learned model also has macroF1 0.86-0.90 and the right team on 86-89% of
live frames (rules: 0.78-0.83 and 77-81%). The dev row for the learned model is out-of-fold: each
game is scored by a model fitted on the other six. The rule-based and test rows use synthetic
Smart Tagging of BAR-ATH's estimated quality (`validation/tags_sim.py`).

**Does it transfer to a third data source?** BAR-ATH is neither Metrica nor DFL data, so we
checked what happens across sources:

| trained on | scored on | learned | rules |
|---|---|---|---|
| 5 IDSSE dev games | Metrica Sample Games 1-2 | 0.611 | 0.413 |
| Metrica Sample Games 1-2 | 5 IDSSE dev games | 0.475 | 0.424 |

Even trained on two games of the other source it stays above the rules; the final model is
trained on both sources.

**A check on BAR-ATH itself, without its ground truth.** The Smart Tagging is an independent
automatic signal. With the tag hint switched off in both models:

| on BAR-ATH | rules | learned |
|---|---|---|
| agrees with the tagged team, ball seen | 72.1% | 79.6% |
| agrees with the tagged team, ball not seen | 49.1% | 75.3% |
| SET PIECES tags with a predicted restart within 10 s | 69% | 80% |

**Tried and not kept.** A second model smoothing the first one's probabilities (+0.008 on dev,
inside the noise), the rule model's hand-over placement on top of the learned probabilities
(-0.10: the learned hand-overs are already better placed), and larger trees (too slow to validate
in time).

**Features left out on purpose.** Signals whose distribution in the training games is an artefact
of those games are computed but not given to the dead-ball model (`learned.DEAD_EXCLUDED`): the
DFL ball stands still on the restart spot through a stoppage, the ATD ball almost never does; no
training game shows the ball outside the lines, the ATD does. We found them by comparing each
feature's distribution on BAR-ATH with the training games (no labels involved).

## 5. Possession rules implemented

The model learns the convention of its training labels; these are the conventions, and they
follow CHALLENGE.md:

- **Loose and contested balls** stay with the last team in control. In the Metrica-derived truth
  only control events (pass, carry, shot, recovery, restart) change possession, challenges do not;
  in the DFL truth a spell under 1 s between two spells of the other team is folded back.
- **Deflections, lone touches, aerial duels without control**: not a change, by the same rule.
- **Passes in flight** belong to the passer until the opponent controls the ball.
- **Dead ball** from the moment play stops (ball out, foul, offside, goal) to the restart. Dead
  ball has no owner: the label is `D`.
- **Restarts**: the team taking it has the ball from the first live frame.
- **Second half before kick-off** (frames 110,808-113,741): `D`.
- **`X`**: never on a graded frame. The scorer fills an `X` with the previous label for spellF1
  and counts it as a miss for macroF1, so writing the label is never worse. Outside the two
  halves every frame is `X`.

## 6. Missing and noisy tracking

- **Ball missing (41% of graded frames).** Gaps up to 1 s are interpolated for the reach
  signals. For longer gaps the models see how long the ball has been gone, where and with whom it
  was last and next seen, the players' speed and the teams' shape; a missing ball is not treated
  as dead by itself.
- **46 slots for 22 players, no identities.** Every signal is collective or "nearest player of
  a team"; nothing follows a slot through time.
- **Parked slots.** Slots the tracker gives no speed for (two sit at fixed spots near the goals)
  are dropped before any distance is taken.
- **Too few players tracked.** Speed statistics are left missing when fewer than 8 players are
  tracked; the trees handle missing values natively.

## 7. Parameters

All in `possession/config.py`, each with its reason and the validation result next to it. The
decoder: H<->A cost 8 and dead cost 8 (6-12 and 4-12 are level on dev), minimum stoppage 2 s (0-3 s
level), ball-outside rule +4 log-odds, tag hint +1 log-odds on ball-not-seen frames. Trees: 300
iterations, 31 leaves, learning rate 0.08, at least 400 samples per leaf, every 4th frame.
**None was tuned on the public 5 minutes.**

## 8. Limitations

- The validation games are simulations of the ATD, not the ATD. Identity swaps, wrong team
  assignments and false ball detections are not simulated.
- The two training sources define possession slightly differently (the DFL flag changes hands
  more often); BAR-ATH's convention may match neither exactly.
- Short possessions are under-reported: the model finds about 300 possessions where the
  validation truth has about 390.
- Dead-ball timing is the weakest part on real tracking: it rests on player speed and ball
  visibility, since the ball's own behaviour during a stoppage could not be learned from the
  training games.
- With more time: train on more games, simulate the ATD's errors more faithfully, and use the
  video to recover the ball where both trackers lose it.

## 9. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-pipeline.txt
# data bundle in ./data (or export MTAH_DATA_DIR=...), see DATA.md
python run_pipeline.py --team TEAM_NAME      # writes submissions/TEAM_NAME.csv and validates it
python -m unittest discover tests
```

Deterministic: no randomness at inference, and the trees are stored as plain arrays in
`models/learned.npz` and evaluated with numpy. We checked that Python 3.9 + numpy 1.26.1 and
Python 3.13 + numpy 2.5.4 write byte-identical files.

To retrain the model from the public games (not needed to reproduce the submission):

```bash
pip install -r requirements-training.txt
python validation/download.py --idsse
python validation/learned_data.py
python validation/train_learned.py --final
python validation/run.py --modes atd --split test --tags
```

## 10. Team and LLM use

Team: Ho Daniel, Hashim, Muhayudin Awan.

LLMs (Claude, via Claude Code) were used to write and review code and this document. Only code,
documentation and aggregate statistics were shared with the model; the scripts ran locally, and
no video, tracking file or extract of the match data was uploaded.
