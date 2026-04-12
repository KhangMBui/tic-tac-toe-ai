# Milestone 6 — Heuristic Evaluation Function

## What is this milestone about?

In earlier milestones the AI used **Minimax with Alpha-Beta pruning** to find the best move.
That approach works by exploring every possible future game state down to the very end of the
game, then picking the move that leads to the best guaranteed outcome.

The problem: on a large board (say 10×10), the number of possible game states is astronomical.
Exploring all of them is computationally impossible in any reasonable amount of time.

The solution used in real game-playing AI is a **heuristic evaluation function** — a formula
that looks at the _current_ board and produces a number estimating how good that position is,
without needing to search all the way to the end. When the AI's search reaches a certain depth
limit and hasn't found a final result yet, it calls this function instead and uses the estimate.

Milestone 6 builds and integrates that function.

---

## Key concept: what is a heuristic?

A **heuristic** is an educated guess. It is not guaranteed to be perfectly correct, but it is
fast to compute and "good enough" to guide the AI toward better moves.

Think of it like a chess player who cannot calculate every possible sequence of moves to the
end of the game. Instead, they look at the board and think: "I have more pieces, I control the
center, and my opponent's king is exposed — this position looks good for me." That intuitive
assessment is a heuristic.

In this project the heuristic outputs a single number (the **score**):

- **Positive score** → the position is good for the AI (X)
- **Negative score** → the position is good for the opponent (O)
- **Near zero** → roughly equal

---

## What was built

### 1. `src/heuristics.py` — the evaluation function

The core deliverable. A class called `HeuristicEvaluator` with one main method:

```python
evaluator = HeuristicEvaluator()
score = evaluator.evaluate(board, my_player="X", opponent_player="O")
```

Internally it works in two steps:

**Step A — extract features.**
It calls `FeatureExtractor` (built in Milestone 4) to convert the board into a dictionary of
numbers. Each number describes one measurable aspect of the position. For example:

| Feature               | What it counts                                                                         |
| --------------------- | -------------------------------------------------------------------------------------- |
| `my_immediate_wins`   | Windows where I have k−1 marks and 1 empty cell — one move from winning                |
| `opp_immediate_wins`  | Same but for the opponent                                                              |
| `my_open_2`           | Windows where I have exactly 2 marks and the rest are empty                            |
| `opp_open_2`          | Same but for the opponent                                                              |
| `my_two_way_threats`  | Empty cells where, if I play there, I create 2 or more simultaneous threats (a "fork") |
| `opp_two_way_threats` | Same but for the opponent                                                              |
| `my_center_control`   | How many central cells I occupy                                                        |
| `opp_center_control`  | How many central cells the opponent occupies                                           |

**Step B — compute a weighted sum.**
Each feature is multiplied by a weight, and all the products are added together:

```
score = w1 * my_immediate_wins
      + w2 * opp_immediate_wins
      + w3 * my_open_2
      + ...
```

With the actual weights substituted in, the full formula used in this project is:

```
score = 1000.0 × my_winning_windows
      − 1000.0 × opp_winning_windows
      +    2.0 × my_immediate_wins
      −    2.0 × opp_immediate_wins
      +    1.0 × my_open_2
      −    1.0 × opp_open_2
      +    0.5 × my_two_way_threats
      −    1.0 × opp_two_way_threats
      +    1.0 × my_center_control
      −    1.0 × opp_center_control
```

**How did we arrive at this formula?**

The weights were not invented from scratch — they were derived from real game data collected in
Milestone 5. Here is the process step by step:

1. **Simulate games and record data.** The M5 pipeline played thousands of games between a
   random player and a tactical player. At every move it recorded the current board, extracted
   all features (immediate wins, open lines, forks, center control, etc.), and noted who
   eventually won that game. This produced a dataset of rows, each containing feature values
   and a final outcome (+1 = win, −1 = loss).

2. **Group rows by outcome.** All rows where the current player eventually won were separated
   from all rows where they eventually lost.

3. **Compute the average of each feature per group.** For example, `my_immediate_wins`
   averaged 0.533 in winning positions and only 0.014 in losing positions. That large gap means
   having an immediate win threat is a reliable sign of a winning position.

4. **Assign sign based on which group has the higher average.**
   - Feature higher in wins → positive weight (more of this = better for me)
   - Feature higher in losses → negative weight (more of this = worse for me)

5. **Assign magnitude based on the size of the gap.** A bigger difference between the win
   average and the loss average means the feature is more predictive, so it gets a larger
   weight. `my_immediate_wins` (diff +0.518) gets weight +2.0. `my_open_2` has the same
   numerical diff but is a weaker strategic signal than an immediate win threat, so it gets
   the smaller weight of +1.0.

6. **Apply domain knowledge for noisy features.** `my_two_way_threats` had the largest raw
   difference (+0.962) but was given a conservative weight of only +0.5. The reason: the
   opponent's fork count is also high even in positions where you win (1.821 vs 1.731), which
   means the feature fluctuates a lot and is less reliable as a standalone predictor. A
   smaller weight guards against over-trusting a noisy signal.

7. **Handle terminal positions separately.** The ±1000 weights for `my_winning_windows` and
   `opp_winning_windows` were not derived from the dataset — they were set by hand to a value
   much larger than any combination of the other features could ever produce. This guarantees
   that a board where someone has already won always gets the most extreme score, overriding
   every other consideration.

Features that help the AI get a **positive** weight. Features that help the opponent get a
**negative** weight. The magnitude of the weight reflects how strongly that feature predicts
winning or losing.

---

### 2. How the weights were chosen — the data-driven process

This is what makes the approach "data-driven." The weights were not just guessed; they were
grounded in evidence from the Milestone 5 dataset.

The M5 pipeline simulated thousands of games and recorded the board state and features at every
move, along with who eventually won. That produces rows like:

| my_immediate_wins | opp_immediate_wins | my_center_control | …   | final_outcome |
| ----------------- | ------------------ | ----------------- | --- | ------------- |
| 1                 | 0                  | 1                 | …   | +1 (win)      |
| 0                 | 1                  | 0                 | …   | −1 (loss)     |

By grouping these rows by outcome and computing the average of each feature, we can see which
features actually differ between wins and losses:

```
Feature                    Win avg   Loss avg   Diff (W-L)
----------------------------------------------------------
my_immediate_wins            0.533      0.014      +0.518
opp_immediate_wins           0.248      0.703      -0.455
my_open_2                    0.533      0.014      +0.518
opp_open_2                   0.248      0.703      -0.455
my_two_way_threats           1.731      0.769      +0.962
opp_two_way_threats          1.821      2.532      -0.711
my_center_control            0.971      0.150      +0.820
opp_center_control           0.256      1.165      -0.909
```

**Reading the table:**

- `my_immediate_wins` averages 0.533 in winning positions and only 0.014 in losing ones.
  That large positive difference confirms it is a strong predictor of winning → give it a
  positive weight.
- `opp_immediate_wins` is much higher in losses → give it a negative weight.
- `my_two_way_threats` has the largest raw difference (+0.962) but is given a smaller weight
  (+0.5) than `my_immediate_wins` (+2.0). That is intentional: forks are contextual and the
  opponent's fork count is also high even in winning positions (1.821 vs 1.731 when you win),
  so the signal is noisier. A smaller weight reflects that uncertainty.

The final weights chosen:

| Feature               | Weight | Reasoning                                                    |
| --------------------- | ------ | ------------------------------------------------------------ |
| `my_winning_windows`  | +1000  | Someone already won — must dominate all other signals        |
| `opp_winning_windows` | −1000  | Same, for the opponent                                       |
| `my_immediate_wins`   | +2.0   | Strong predictor of winning                                  |
| `opp_immediate_wins`  | −2.0   | Strong predictor of losing                                   |
| `my_open_2`           | +1.0   | Positive predictor, same magnitude as immediate wins in data |
| `opp_open_2`          | −1.0   | Negative predictor                                           |
| `my_two_way_threats`  | +0.5   | Positive but noisy — conservative weight                     |
| `opp_two_way_threats` | −1.0   | More reliably negative                                       |
| `my_center_control`   | +1.0   | Strongly positive in data                                    |
| `opp_center_control`  | −1.0   | Strongly negative in data                                    |

The ±1000 values for `winning_windows` are special: they represent a _completed_ game (someone
has already won k-in-a-row). They are set far larger than any other feature combination so that
the AI always correctly identifies a finished position as the most extreme possible outcome.

---

### 3. `src/ai.py` — integrating the heuristic into search

Two new methods were added to `MinimaxAI`, leaving all existing code untouched:

**`get_best_move_heuristic(game, ai_player, human_player, max_depth=4)`**

The public entry point. Works exactly like `get_best_move_ab` (Alpha-Beta pruning), except it
takes a `max_depth` parameter that caps how deep the search goes.

**`_minimax_ab_h(...)` — internal search with heuristic cutoff**

The key change from plain Alpha-Beta is one extra rule: _if the search has reached `max_depth`
and the game is not yet over, call the heuristic instead of continuing._

```
At each node:
  1. Is there a winner? → return exact terminal score (±10000)
  2. Is the board full (draw)? → return 0
  3. Have we reached max_depth? → return heuristic score  ← NEW
  4. Otherwise → recurse deeper with Alpha-Beta pruning
```

Terminal scores use ±10000 (much larger than any heuristic output) to guarantee the AI always
prefers an actual win over a good-looking estimate.

**Practical result — same move, fewer nodes:**

```
Scenario: X to move — immediate win available at (0,2)
  Heuristic AB  → move (0, 2)   94 nodes visited
  Exhaustive AB → move (0, 2)  161 nodes visited   ✓ agree

Scenario: X to move — must block O's immediate win at (1,2)
  Heuristic AB  → move (1, 2)  190 nodes visited
  Exhaustive AB → move (1, 2)  310 nodes visited   ✓ agree
```

Both approaches find the same correct move. The heuristic version visits ~40% fewer nodes on
3×3. The gap widens drastically on larger boards, which is the motivation for this approach.

---

### 4. `tests/test_heuristics.py` — automated verification

Six test cases verify the heuristic behaves correctly on crafted board positions:

| Test                       | Board setup                  | What is asserted           |
| -------------------------- | ---------------------------- | -------------------------- |
| `test_winning_board_for_X` | X has a completed row        | Score > 100                |
| `test_losing_board_for_X`  | O has a completed column     | Score < −100               |
| `test_draw_board`          | Full board, no winner        | \|Score\| < 10             |
| `test_immediate_win_for_X` | X has 2-in-a-row, O does not | Score > 1                  |
| `test_immediate_win_for_O` | O has 2-in-a-row, X does not | Score < −1                 |
| `test_symmetry`            | Same board, swap X and O     | Scores are exact negatives |

All six pass.

---

### 5. `demo.ipynb` — notebook documentation

Three cells were added to the demo notebook documenting the M6 process end-to-end:

1. **Feature correlation table** — loads the M5 CSV, prints win/loss averages and the
   difference for every selected feature, and shows the weight assigned to each. This is the
   direct evidence linking the data to the weight choices.

2. **Board score gallery** — evaluates six crafted boards (won, lost, draw, immediate win,
   immediate loss, early game) and prints their scores with human-readable interpretations.
   Confirms the heuristic assigns the right sign and a sensible magnitude.

3. **Heuristic AI vs exhaustive AI** — runs both methods on two mid-game scenarios, prints
   the move each picks and how many nodes each explored. Confirms they agree on the correct
   move while the heuristic version is faster.

---

## Files changed in Milestone 6

| File                       | Change                                                           |
| -------------------------- | ---------------------------------------------------------------- |
| `src/heuristics.py`        | New class `HeuristicEvaluator` with data-driven weights          |
| `src/ai.py`                | New methods `get_best_move_heuristic` and `_minimax_ab_h`        |
| `tests/test_heuristics.py` | Six test cases covering all key board scenarios                  |
| `demo.ipynb`               | Three new cells documenting the feature analysis and integration |

---

## How it all fits together

```
Board state
    │
    ▼
FeatureExtractor (M4)
    │  converts board → dictionary of numbers
    ▼
HeuristicEvaluator (M6)
    │  weighted sum → single score
    ▼
MinimaxAI._minimax_ab_h (M6 integration)
    │  calls heuristic at depth limit instead of searching further
    ▼
Best move returned to the game
```

The heuristic sits between the feature extractor and the search algorithm. It is the bridge
that makes depth-limited search possible: without it, stopping the search early would produce
no meaningful score. With it, the AI can make a reasonable decision even when it cannot see
all the way to the end of the game.
