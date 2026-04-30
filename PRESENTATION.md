# Tic-Tac-Toe AI: From First Principles to 100×100

A structured walkthrough of the entire system — game engine, search algorithms, heuristic evaluation,
and scalability engineering. Designed to let you explain every decision confidently in a class presentation.

---

## 1. Big Picture: What This Project Is

This project builds a generalized Tic-Tac-Toe engine and AI that works on any board size, from the
classic 3×3 up to 100×100.

The core question is: **how do you make an AI that plays well when exhaustive search is impossible?**

On 3×3, the full game tree has ~255,168 leaf nodes — a modern computer solves it instantly. On a
10×10 board with k=5, the branching factor is ~100 and the game lasts ~50 moves, giving a tree with
roughly 100^50 nodes. That is more atoms than in the observable universe. You cannot search it all.

The project works through that problem in stages: exhaustive Minimax → Alpha-Beta pruning →
depth-limited search with a learned heuristic → candidate-move restriction. Each stage makes the AI
faster; together they make it practical on boards up to 100×100.

---

## 2. Game Generalization

Standard Tic-Tac-Toe is 3×3, 3-in-a-row. This engine is parameterized by two numbers:

- **n** — board side length (n×n grid)
- **k** — number of consecutive marks needed to win

This is called the **m,n,k-game** in combinatorial game theory. Classic examples:

| Game | n | k |
|---|---|---|
| Tic-Tac-Toe | 3 | 3 |
| Connect Four (simplified) | 6 | 4 |
| Gomoku | 15 | 5 |
| This project (max) | 100 | 5 |

Every class, algorithm, and data structure in this codebase is parameterized by (n, k). There is no
hardcoded 3×3 logic anywhere.

The default GUI rule: **k = min(n, 5)**. This keeps small boards playable (3×3 still uses k=3) while
preventing trivially easy wins on large boards.

---

## 3. System Architecture Overview

The system has a clean layered architecture where each layer depends only on layers below it:

```
Board (src/board.py)
  — raw grid state, move placement, O(4k) winner check

Game (src/game.py)
  — rules, turn management, win/draw detection

FeatureExtractor (src/features.py)
  — converts board state → interpretable feature dict

HeuristicEvaluator (src/heuristics.py)
  — feature dict → numeric score (how good is this board for me?)

MinimaxAI (src/ai.py)
  — search: Minimax, Alpha-Beta, depth-limited heuristic AB

DataCollection (src/data_collection.py)
  — simulates games, records features + outcomes → CSV

Benchmark (src/benchmark.py)
  — reproducible experiments, performance measurement

GUI (src/gui.py)
  — Tkinter human-vs-AI, canvas rendering, background AI thread
```

Data flows upward: the AI calls the evaluator which calls the extractor which reads the board. The GUI
sits at the top and orchestrates everything.

---

## 4. Core Game Engine

### Board (`src/board.py`)

The `Board` class is the foundation. It stores the grid and maintains two auxiliary sets for O(1)
lookups:

```python
self._grid     # list[list[str|None]] — the n×n grid
self._empty    # set of (r, c) — all unoccupied cells
self._occupied # set of (r, c) — all occupied cells
```

`is_full()` checks `len(self._empty) == 0` — O(1). Without the set, you'd scan every cell: O(n²).

`make_move` and `undo_move` keep the sets synchronized:

```python
def make_move(self, r, c, player):
    self._grid[r][c] = player
    self._empty.discard((r, c))
    self._occupied.add((r, c))

def undo_move(self, r, c):
    self._grid[r][c] = None
    self._empty.add((r, c))
    self._occupied.discard((r, c))
```

### The Key Winner Check: `check_line_at`

This is the most performance-critical method in the codebase. Instead of scanning the entire board
after every move (O(n²k)), it only checks lines that pass through the cell just placed:

```python
def check_line_at(self, r, c, player):
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 1
        for sign in (1, -1):
            for step in range(1, self.k):
                nr = r + dr * sign * step
                nc = c + dc * sign * step
                if (0 <= nr < self.n and 0 <= nc < self.n
                        and self._grid[nr][nc] == player):
                    count += 1
                else:
                    break
        if count >= self.k:
            return True
    return False
```

Why is this correct? A new piece at (r, c) can only create a winning line that passes through (r, c).
So you only need to check the 4 directions (horizontal, vertical, two diagonals) through that cell.

**Cost: O(4k)** — 4 directions × up to k steps each way. On a 100×100 board with k=5 that is 40
operations, vs. n²k = 100×100×5 = 50,000 for a full scan. This is a **~1,250× speedup** at every
node in the search tree.

### Game (`src/game.py`)

`Game` wraps `Board` and adds rules:
- `switch_turn()` — alternates between the two players
- `make_move(r, c)` — places the current player's piece
- `get_available_moves()` — returns list of (r, c) from `board._empty`
- `check_winner()` — **full board scan** O(n²k), used only at the game-rule level
- `is_draw()` — `board.is_full()` and no winner

The full-board `check_winner()` is used in two places: the benchmarking code (for correctness
verification) and the test suite. Inside the AI search tree, only `check_line_at` is used.

---

## 5. Minimax Algorithm (Foundation)

### What Problem Minimax Solves

Minimax answers: **"assuming both players play perfectly, what move should I make?"**

It builds the full game tree, assigns scores to terminal states, and propagates those scores back up
to the root. The AI (maximizer) picks the branch with the highest score; the opponent (minimizer)
picks the lowest.

### Terminal Scores

```
AI win   =  10 − depth   (prefer faster wins)
Human win = depth − 10   (prefer slower losses)
Draw      =  0
```

Subtracting depth means the AI prefers to win in fewer moves (it scores higher the sooner it wins).
Similarly, it delays losses as long as possible.

### The Algorithm

```python
def _minimax(self, game, depth, is_maximizing, ai_player, human_player):
    winner = game.check_winner()
    if winner == ai_player:
        return 10 - depth
    if winner == human_player:
        return depth - 10
    if game.is_draw():
        return 0

    if is_maximizing:
        best = float("-inf")
        for r, c in game.get_available_moves():
            game.board.make_move(r, c, ai_player)
            score = self._minimax(game, depth+1, False, ai_player, human_player)
            game.board.undo_move(r, c)
            best = max(best, score)
        return best
    else:
        best = float("inf")
        for r, c in game.get_available_moves():
            game.board.make_move(r, c, human_player)
            score = self._minimax(game, depth+1, True, ai_player, human_player)
            game.board.undo_move(r, c)
            best = min(best, score)
        return best
```

### Move Ordering

Before exploring, moves are sorted: center first, then corners, then edges. This does not change the
result, but helps Alpha-Beta pruning (the next section) find good moves early and cut more branches.

### Why It Is Exact but Impractical

Minimax always finds the optimal move. But it explores every possible game state. On a 3×3 board
from the opening position, it visits **7,979 nodes**. That is fast. On larger boards, the tree grows
exponentially and the algorithm becomes infeasible.

---

## 6. Alpha-Beta Pruning

### The Core Insight

In Minimax, you often explore branches you already know cannot influence the result. Alpha-Beta
skips them.

Two values are tracked throughout the search:
- **alpha** — the best score the maximizer has found so far (a lower bound)
- **beta** — the best score the minimizer has found so far (an upper bound)

**Pruning rule: if `beta ≤ alpha`, stop exploring the current branch.** The minimizer has already
found a path that is ≤ alpha, so the maximizer would never choose to enter this subtree — it already
has something better.

### The Code

```python
def _minimax_ab(self, game, depth, alpha, beta, is_maximizing, ai_player, human_player):
    winner = game.check_winner()
    if winner == ai_player: return 10 - depth
    if winner == human_player: return depth - 10
    if game.is_draw(): return 0

    if is_maximizing:
        best = float("-inf")
        for r, c in game.get_available_moves():
            game.board.make_move(r, c, ai_player)
            val = self._minimax_ab(game, depth+1, alpha, beta, False, ...)
            game.board.undo_move(r, c)
            best = max(best, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break   # beta cut-off
        return best
    else:
        best = float("inf")
        for r, c in game.get_available_moves():
            game.board.make_move(r, c, human_player)
            val = self._minimax_ab(game, depth+1, alpha, beta, True, ...)
            game.board.undo_move(r, c)
            best = min(best, val)
            beta = min(beta, val)
            if beta <= alpha:
                break   # alpha cut-off
        return best
```

### Measured Impact

On a 3×3 board from the opening position (2 pieces placed):

| Method | Nodes visited | Reduction |
|---|---|---|
| Minimax | 7,979 | — |
| Alpha-Beta | 1,988 | **75% fewer** |

Same result. Same optimal move. 75% less work.

Alpha-Beta's best case (perfect move ordering) cuts the tree from O(b^d) to O(b^(d/2)), where b is
the branching factor and d is depth. That means you can search twice as deep in the same time. In
practice, with good move ordering, you get 40–75% pruning.

---

## 7. Why Full Search Does Not Scale

Even with Alpha-Beta, exhaustive search breaks down quickly.

Consider a 5×5 board, k=4:
- ~25 moves in a game
- Branching factor ~20 mid-game (after Alpha-Beta)
- Tree depth = 25 plies
- Estimated nodes: 20^25 ≈ 10^32

That computation would take longer than the age of the universe.

The fundamental problem: **the game tree grows exponentially with board size and game length**.
Alpha-Beta reduces the exponent but does not change its exponential nature.

We need a different approach: stop searching at a fixed depth and estimate the value of non-terminal
states using a heuristic function.

---

## 8. Key Idea: Heuristic Evaluation

Instead of searching to the end of the game, we search to a fixed depth and then **estimate** how
good the position is using a heuristic.

This is the same idea behind chess engines like Stockfish: they search 20–30 plies deep and then
evaluate the position using features like material balance, king safety, pawn structure.

For this project, the heuristic score comes from a weighted sum of board features:

```
score = Σ weight_i × feature_i
```

If the score is positive, the position favors the AI. If negative, it favors the human.

The challenge: what features should we use, and what weights? This is where data collection (M5)
and feature engineering (M4, M6) come in.

---

## 9. Feature Extraction (State Representation)

`FeatureExtractor.extract(board, my_player, opp_player)` converts a board position into a
dictionary of 25+ interpretable features.

### Feature Groups

**Basic counts:**
```
my_marks, opp_marks, empty_cells
```

**Positional:**
```
my_center_control, opp_center_control
```
Center is defined as the single center cell (odd n) or the 2×2 center block (even n).

**Window-based features:**
A *window* is any contiguous length-k segment in one of 4 directions (horizontal, vertical, two
diagonals). For each window, we categorize it:

```
my_open_i     — exactly i of my marks, rest empty (i = 1 .. k-1)
opp_open_i    — same for opponent
my_immediate_wins   — k-1 of my marks + 1 empty (one move from winning)
opp_immediate_wins  — same for opponent
my_winning_windows  — k of my marks (already won)
opp_winning_windows — k of opponent's marks
blocked_windows     — both players present (neither can win here)
neutral_windows     — all empty
```

**Tactical (fork detection):**
```
my_two_way_threats   — empty cells that, if played, create ≥2 simultaneous immediate threats
opp_two_way_threats  — same for opponent
```
A fork is the strongest tactical weapon: you create two winning threats at once, and the opponent can
only block one.

### Window Enumeration Caching

Enumerating all length-k windows on an n×n board takes O(n²) work. Since this is called at every
node in the search tree, and the board dimensions never change during a game, the result is cached:

```python
@lru_cache(maxsize=64)
def _enumerate_windows(n, k):
    # returns list of window coordinate tuples for an n×n, k-in-a-row board
    ...
```

This runs once per (n, k) pair — ever. Every subsequent call hits the cache in O(1).

### Fork Detection and Scaling

Detecting forks requires simulating each empty cell placement and checking if it creates ≥2
simultaneous immediate threats. This is O(n²) per board evaluation.

For large boards, this becomes a bottleneck. The solution: stratify by board size.

| Board size | Strategy |
|---|---|
| n ≤ 15 | Full scan of all empty cells |
| 15 < n ≤ 20 | Scan only cells within radius-2 of occupied pieces |
| n > 20 | Skip entirely (return 0) |

The radius-2 restriction mirrors what `_get_candidate_moves` uses in the AI — cells far from any
occupied piece are not strategically relevant in k-in-a-row games.

---

## 10. Data Collection (M5)

`collect_dataset()` simulates games between different AI policies and records the board state,
all extracted features, the move played, and the final game outcome (+1 win / 0 draw / −1 loss)
at every turn.

### Policies

```
random_policy    — picks a random legal move
tactical_policy  — prefers: immediate wins > blocks > center > random
```

The dataset is saved to `report/results/m5_dataset.csv`. It captures thousands of (state, outcome)
pairs that reveal which features actually correlate with winning.

### What the Data Showed

After collecting the M5 dataset (4×4, k=3, tactical vs random):

- **`my_immediate_wins` / `opp_immediate_wins`**: strongest predictor — having k-1 in a row nearly
  guarantees a win if not blocked
- **`my_two_way_threats` / `opp_two_way_threats`**: high positive/negative correlation with outcomes
  — forks are reliable win predictors
- **`my_open_{k-2}`**: strategically relevant at intermediate depth (two marks short of a win)
- **`my_center_control`**: useful on small boards, noise on large ones

This data directly informed the heuristic weights in M6.

---

## 11. Heuristic Function (M6)

`HeuristicEvaluator.evaluate(board, my_player, opp_player)` computes:

```
score = 1000 × my_winning_windows   − 1000 × opp_winning_windows
       +   50 × my_immediate_wins   −    50 × opp_immediate_wins
       +   30 × my_two_way_threats  −    30 × opp_two_way_threats
       +   15 × my_open_{k-2}       −    15 × opp_open_{k-2}
       +    1 × my_center_control   −     1 × opp_center_control   (n ≤ 10 only)
```

Positive = good for my_player. Negative = good for opponent.

### Key Design Choices

**`open_{k-2}` not `open_{k-1}`:**
`open_{k-1}` is the same as `my_immediate_wins` (k-1 marks + 1 empty). Using `open_{k-2}` avoids
double-counting and captures the strategically important "building" phase — two marks short of a win.
For k=5, this is `open_3` (three in a row), which is the most important positional threat.

**Center weight disabled for n > 10:**
The 2×2 center on a 50×50 board represents < 0.16% of cells. Weighting it would add noise, not
signal.

**±10000 terminal scores in search (not ±1000):**
In the search tree, terminal states (actual wins/losses) are scored ±10000, while the heuristic
can produce values at most ±1000. This guarantees the AI **never mistakes a heuristic estimate for
an actual win** — real wins always dominate.

**Symmetric weights (`opp_X = −my_X`):**
This is a correctness invariant: evaluating a position from X's perspective and then from O's
perspective should give equal and opposite scores. It is enforced by the `test_symmetry` test.

### Weight Building

Weights are built dynamically per (k, n) via `_build_weights(k, n)` — also cached with `@lru_cache`.
This makes the heuristic self-adjusting: it automatically uses `open_3` for k=5 and `open_1` for k=3
without any manual tuning.

---

## 12. Depth-Limited Alpha-Beta (M7)

`get_best_move_heuristic(game, ai_player, human_player, max_depth)` is the primary AI method.

It runs the same Alpha-Beta search as M3, but adds a **depth cutoff**: when the search reaches
`max_depth`, it calls `evaluator.evaluate()` instead of searching further.

```python
def _minimax_ab_h(self, game, depth, max_depth, alpha, beta,
                  is_maximizing, ai_player, human_player, evaluator,
                  last_move, last_player):

    # Fast O(4k) terminal check — only through the last-placed cell.
    if last_move is not None:
        if game.board.check_line_at(last_move[0], last_move[1], last_player):
            return (10000 - depth) if last_player == ai_player else (depth - 10000)
        if game.board.is_full():
            return 0

    # Depth cutoff: score with heuristic instead of searching deeper.
    if depth >= max_depth:
        return evaluator.evaluate(game.board, ai_player, human_player)

    # ... standard Alpha-Beta recursion ...
```

### Why `last_move` and `last_player` Are Passed Explicitly

At every node in the search tree, we need to check if the position is terminal. We could call
`check_winner()` (full board scan, O(n²k)), but that is wasteful — a new piece can only complete a
line through itself. By passing `last_move` and `last_player` down the recursion, we call
`check_line_at` (O(4k)) instead. This is a ~200× speedup per node on large boards.

### Default Depth Schedule

| Board size | Auto depth |
|---|---|
| n ≤ 5 | 4 |
| n ≤ 10 | 3 |
| n > 10 | 2 |

Chosen to keep AI response time under ~1 second. Users can override via the GUI's depth field.

---

## 13. Pre-Search Tactical Checks (Very Important)

Before entering the full Alpha-Beta search, `get_best_move_heuristic` runs three fast checks in
priority order. If any fires, the function returns immediately without searching further.

### Check 1 — Take an Immediate Win

```python
for row, col in moves:
    game.board.make_move(row, col, ai_player)
    won = game.board.check_line_at(row, col, ai_player)
    game.board.undo_move(row, col)
    if won:
        return (row, col)   # take the win immediately
```

If the AI can win right now, it does. No search needed.

### Check 2 — Block an Immediate Human Win

Same loop, but simulating the human's move. If the human would win at any candidate cell, block the
first such cell found.

### Check 3 — Block Open-(k-1) Forced Threats

This is the subtlest and most important check. An **open-(k-1) threat** is exactly k-1 consecutive
marks with **both adjacent ends empty**. This is a forced loss: if you do not block it, the opponent
wins on the next turn by playing at either open end.

The function `_find_open_forced_threats(board, player)` scans for these:

```python
# Runs through every cell, identifies starts of runs of exactly k-1,
# checks that both ends are empty and in-bounds.
if length == k - 1 and e1_open and e2_open:
    threats.append(((e1r, e1c), (e2r, e2c)))
```

If any such threat exists, the AI blocks one of the two open ends immediately.

**Why not rely on Alpha-Beta to handle this?** Alpha-Beta at depth=2 can miss this. At depth=1, it
sees only the AI's immediate moves (not the opponent's response), so it would not detect that the
opponent wins on the following turn. At depth=2, move-ordering and pruning might result in the
blocking move being explored last. The pre-search check guarantees this critical defensive move is
always made.

### Why the Order Matters

The checks must run in this exact order:
1. Win immediately (never miss a winning move)
2. Block opponent's win (never let opponent win when you could block)
3. Block forced threat (never enter a position you cannot escape)

If check 1 and 2 both apply (you can win AND the opponent can win), check 1 takes priority — you win.

---

## 14. Candidate Move Restriction (Scaling Breakthrough)

For n > 5, `get_available_moves()` can return up to n² candidates. On a 50×50 board, that is 2,500
moves. At depth=2, you evaluate 2,500² = 6.25 million leaf nodes. That is completely infeasible.

The insight: **cells far from any occupied piece have zero immediate strategic relevance in k-in-a-row
games.** A threat on the left side of the board has no interaction with moves on the right side.

`_get_candidate_moves(board, radius=2)` restricts to cells within Chebyshev radius 2 of any occupied
piece:

```python
candidates = set()
for pr, pc in board._occupied:
    for dr in range(-2, radius + 1):
        for dc in range(-2, radius + 1):
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < n and 0 <= nc < n and board._grid[nr][nc] is None:
                candidates.add((nr, nc))
return list(candidates)
```

On an empty board, returns just the center — the universally strongest opening in k-in-a-row games.

**Result:** Branching factor collapses from n² (~2,500 on 50×50) to ~20–50 in typical mid-game
positions. This makes depth=2 feasible on boards up to 100×100.

This restriction is applied at two points:
- In `get_best_move_heuristic` (the root move selection)
- Inside `_minimax_ab_h` (at every recursive node)

---

## 15. Scalability Strategy

Every component was designed with large boards in mind. The full picture:

| Problem | Solution | Where |
|---|---|---|
| Branching factor n² | Candidate-move radius=2 around occupied cells | `ai.py: _get_candidate_moves` |
| Winner check O(n²k) at every node | `check_line_at` O(4k) through last-placed cell | `board.py: check_line_at` |
| Window enumeration repeated per leaf | `@lru_cache` on `_enumerate_windows(n, k)` | `features.py` |
| Fork scan O(n²) per heuristic call | Full scan n≤15, radius-2 scan 15<n≤20, skip n>20 | `features.py: _count_two_way_threats` |
| Center control noise on large boards | Center weight only for n≤10 | `heuristics.py: _build_weights` |
| Depth=3 too slow for n>10 | Auto depth: n≤10→3, n>10→2 | `gui.py: _auto_depth` |
| Pre-search checks on large move set | Only run on candidate moves (already restricted) | `ai.py: get_best_move_heuristic` |

These optimizations compose: the candidate restriction reduces branching factor, the `lru_cache` on
windows means the cost of feature extraction is amortized, and `check_line_at` keeps terminal
detection cheap at every node.

---

## 16. Benchmarking (M8)

Three reproducible experiments in `src/benchmark.py`.

### Experiment 1 — Minimax vs Alpha-Beta (3×3, k=3)

Both methods produce the identical optimal move. Alpha-Beta prunes 41–75% of nodes depending on
game phase.

| Position | Minimax nodes | AB nodes | Pruned |
|---|---|---|---|
| Early (2 placed) | 7,979 | 1,988 | **75%** |
| Mid (4 placed) | 149 | 88 | 41% |
| Late (6 placed) | 11 | 11 | 0% |

Late-game pruning drops to 0% because there are so few moves left that Alpha-Beta has nothing to cut.

### Experiment 2 — Full Search vs Depth-Limited (4×4, k=3)

| Method | Nodes visited | Same move as full search? |
|---|---|---|
| Full Alpha-Beta (reference) | 47,183 | — |
| Heuristic depth=1 | 12 | No |
| Heuristic depth=2 | 144 | **Yes — 34× fewer nodes** |
| Heuristic depth=3 | 288 | Yes |
| Heuristic depth=4 | 1,371 | Yes |

At depth=2, the heuristic AI finds the correct move using 34× fewer node evaluations than exhaustive
search. This demonstrates that the heuristic is accurately capturing board quality — the AI does not
need to see the entire game tree.

### Experiment 3 — Heuristic AI vs Random (20 games each)

| Board | k | Depth | Win% | Draw% | Loss% |
|---|---|---|---|---|---|
| 3×3 | 3 | 4 | 85% | 15% | 0% |
| 4×4 | 3 | 3 | **100%** | 0% | 0% |
| 5×5 | 4 | 2 | **100%** | 0% | 0% |

The AI never loses to a random opponent. On 3×3 the draws are expected — optimal play leads to
draws on this board.

---

## 17. GUI (M9)

The GUI is built with Tkinter and designed to remain responsive at any board size.

### Canvas-Based Rendering

The board is rendered as canvas rectangle + text items, not as Button widgets. This is a critical
design choice: a 100×100 grid of Button widgets would require 10,000 widget objects, causing severe
layout and rendering overhead. With canvas items, the entire board is one bitmap that scrolls natively.

```python
for r in range(n):
    for c in range(n):
        x1, y1 = c * step + GAP, r * step + GAP
        x2, y2 = x1 + cell, y1 + cell
        self._rects[(r, c)] = self._canvas.create_rectangle(
            x1, y1, x2, y2, fill=CELL_BG, outline=GRID_LINE
        )
        self._texts[(r, c)] = self._canvas.create_text(
            (x1+x2)//2, (y1+y2)//2, text="", font=...
        )
```

### AI on Background Thread

Tkinter runs on a single thread. If the AI computation ran on the main thread, the window would
freeze during thinking. The solution: run the AI on a background thread and post the result back:

```python
def _do_ai_move(self):
    def _think():
        move = self._ai.get_best_move_heuristic(...)
        self.root.after(0, lambda: self._apply_ai_result(move))
    threading.Thread(target=_think, daemon=True).start()
```

`root.after(0, callback)` safely posts the result back to the main thread's event loop. This is
the correct pattern for Tkinter + threads.

### Status Commentary

When it is the AI's turn, the status bar shows:
- `"Seems like I'm losing."` — if the human has 4+ in a row with both ends open (only on k=5 boards)
- `"AI is thinking..."` — otherwise

When it becomes the human's turn:
- `"I already win."` — if the AI has 4+ in a row with both ends open
- `"Your turn"` — otherwise

These messages require detecting **open-4 threats** (not open-3, not blocked threats). The
`_has_open_4(player)` method scans for runs of length ≥ 4 with both adjacent ends in-bounds and
empty. It requires `k ≥ 5` — this commentary only makes sense in games where 4-in-a-row is
strategically decisive without immediately winning.

### Other Features

- **Scrollbars:** vertical (MouseWheel) and horizontal (Shift+MouseWheel)
- **Board sizes (dropdown):** 3, 30, 50, 100 — k auto-syncs to min(n, 5)
- **Last-placed cell:** highlighted with a green 3px outline
- **Winning k cells:** turn green with `WIN_COLOR` fill
- **New Game dialog:** configure n, k, depth, and which player you are

---

## 18. Testing Strategy

57 tests across 6 files, all passing. Run with `python -m pytest tests/ -q`.

| File | What it covers | Count |
|---|---|---|
| `test_board.py` | Grid state, make/undo move, set sync, clone, check_line_at | 9 |
| `test_game.py` | Win detection in all 4 directions, draw, turn switching | 9 |
| `test_ai.py` | Minimax correctness, Alpha-Beta equivalence, heuristic AB cutoff, efficiency | 20 |
| `test_features.py` | Feature extraction, perspective symmetry, even/odd boards | 8 |
| `test_heuristics.py` | Score sign on crafted boards, symmetry invariant | 6 |
| `test_data_collection.py` | Dataset generation, outcome encoding, policy correctness | 5 |

### Key Test: `test_symmetry`

The heuristic must satisfy `evaluate(board, X, O) == −evaluate(board, O, X)`. This is tested by
placing pieces on a crafted board and asserting the two evaluations are equal and opposite. If this
fails, the AI has a systematic bias toward one player — a correctness bug, not just a strength issue.

### Key Test: Heuristic AB Efficiency

Tests verify that `nodes_explored_h` is strictly less than `nodes_explored_ab` on the same 4×4
position. This confirms that (a) the depth cutoff is actually firing, and (b) the heuristic is
making decisions that prune branches differently than full search.

### Design Philosophy

Tests are written at the **behavior level**, not the implementation level. They specify what the
AI should do (always take a winning move, always block an immediate loss) rather than how many
nodes it should visit. This makes the tests resilient to internal refactoring.

---

## 19. Key Design Decisions

### 1. `check_line_at` vs `check_winner`

`check_winner()` scans the entire board (O(n²k)) and returns the winner's symbol. `check_line_at(r,
c, player)` checks only lines through the given cell (O(4k)).

**Decision:** Use `check_winner` only in game rules and the GUI end-of-game. Use `check_line_at`
inside the search tree and in `_apply_move` in the GUI.

**Why:** At every node in a depth-2 search on a 100×100 board, `check_winner` would cost 50,000
operations. `check_line_at` costs 40. Over millions of nodes, this difference is the difference
between a playable game and a frozen UI.

### 2. ±10000 terminal scores vs ±10 (Minimax) or ±heuristic

**Decision:** Terminal scores in the depth-limited search are ±10000. The heuristic can produce at
most ±1000 (one winning window = 1000 points).

**Why:** This guarantees that any actual win or loss always beats any heuristic estimate. Without
this, the AI might rank a heuristically-scored position higher than a position it actually wins in
two moves — a catastrophic error.

### 3. `open_{k-2}` in heuristic, not `open_{k-1}`

**Decision:** The heuristic uses the `open_{k-2}` feature (two marks short of an immediate win)
rather than `open_{k-1}` (one mark short, the same as `my_immediate_wins`).

**Why:** `open_{k-1}` duplicates `my_immediate_wins`, which is already weighted at ±50. Using
`open_{k-2}` captures the "building" phase — lines that will become dangerous in two more moves.
For k=5, this is three-in-a-row (`open_3`), the most important positional feature in Gomoku.

### 4. Fork weight ±30 (raised from ±5)

**Decision:** `my_two_way_threats` weight is +30, `opp_two_way_threats` is −30.

**Why:** M5 data showed that opponent forks are among the most reliable loss predictors. A fork
(two simultaneous threats) is nearly impossible to defend against; the AI must treat it as near-
critical at the heuristic cutoff. The original ±5 was too conservative.

### 5. Candidate-move radius=2 (not radius=1 or radius=3)

**Decision:** Radius=2 Chebyshev neighborhood around all occupied cells.

**Why:** Radius=1 misses moves that set up threats two cells away — a common pattern in Gomoku.
Radius=3 gives too many candidates (~100+) and slows the AI. Radius=2 gives ~20–50 candidates
in typical mid-game: enough to find all tactically relevant moves, small enough to keep depth=2
fast on any board size.

---

## 20. Limitations

**1. Depth-2 on large boards is beatable by a patient human.**
With n > 10, the AI uses depth=2. An experienced player can create threats across two moves that
the AI cannot see coming. Depth=3 would significantly improve strength but requires ~10× more
computation.

**2. Fork detection is disabled for n > 20.**
On boards larger than 20×20, `_count_two_way_threats` returns 0. The heuristic loses an important
signal. The radius-2 filtering in the 15<n≤20 range helps but is not available for larger boards
without further optimization.

**3. No transposition table.**
The same board position can be reached via different move sequences (transpositions). Without a
hash table recording previously evaluated positions, the AI re-evaluates duplicates. Adding a
transposition table (Zobrist hashing) is a standard improvement in game AI.

**4. No iterative deepening.**
The depth is fixed before the search starts. Iterative deepening deepens the search incrementally,
enabling better move ordering at deeper levels and allowing the AI to use any amount of time
productively rather than having a fixed depth.

**5. The heuristic was not machine-learned.**
The feature weights were manually tuned based on M5 correlation analysis. A proper supervised or
reinforcement learning approach (training weights from millions of self-play games) would produce
a significantly stronger heuristic.

**6. Static evaluation only.**
The heuristic does not perform any lookahead at the cutoff node (no quiescence search). In
positions with many captures or tactical complications, the static score can be misleading.

---

## 21. Future Work

**Transposition table (Zobrist hashing):**
Hash each board state to a 64-bit integer, cache the evaluated score. On large boards with many
repeated positions via different move sequences, this can reduce the effective search tree by 30–50%.

**Iterative deepening with time limit:**
Instead of a fixed depth, search progressively deeper (depth 1, 2, 3, ...) until a time budget
is exhausted. The best move from the previous iteration guides move ordering for the next, improving
pruning. The AI uses the deepest completed search result.

**Learned evaluation function:**
Train a neural network (or even linear regression) on millions of self-play games to predict board
outcome directly from raw features. This is how AlphaZero works at its core.

**Threat-space search:**
Instead of searching all candidate moves, only consider moves that are direct responses to existing
threats. On boards where most of the position is quiet, this dramatically reduces the branching
factor while focusing on the tactically critical cells.

**Opening book:**
For 15×15 k=5 (standard Gomoku), precomputed optimal openings exist. An opening book stores the
known best responses for the first ~10 moves, bypassing search entirely in the opening phase.

**Two-player mode:**
The GUI currently supports only human vs AI. Human vs human is already implemented in `main.py`
(CLI); extending it to the GUI is straightforward.

---

## 22. How to Run the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the GUI — recommended entry point
python src/gui.py

# Play via CLI (human vs human)
python main.py

# Run all 57 tests
python -m pytest tests/ -q

# Run the benchmarks (writes CSVs and plots to report/)
python src/benchmark.py

# Open the end-to-end demo notebook
jupyter notebook demo.ipynb
```

### Project Structure

```
tic_tac_toe_ai/
├── src/
│   ├── board.py           # Board state, O(4k) winner check
│   ├── game.py            # Game rules, turn management
│   ├── ai.py              # Minimax, Alpha-Beta, depth-limited heuristic search
│   ├── features.py        # Board → feature dict (windows, forks, center)
│   ├── heuristics.py      # Weighted-sum heuristic evaluation
│   ├── data_collection.py # Game simulation → CSV/JSON datasets
│   ├── benchmark.py       # Reproducible experiments
│   └── gui.py             # Tkinter GUI
├── tests/                 # 57 tests, all passing
├── demo.ipynb             # End-to-end notebook (M4–M9)
├── main.py                # CLI runner (human vs human)
├── report/
│   ├── results/           # CSV experiment outputs
│   └── figures/           # Matplotlib plots
└── PRESENTATION.md        # This file
```

### Quick Demo (3×3, optimal AI)

1. Run `python src/gui.py`
2. Select board size **3**, k=**3**, depth=**auto**, play as **X**
3. The AI plays as O and will never lose — every game either ends in a draw or O wins

### Playing a Large Board (50×50, k=5)

1. Run `python src/gui.py`
2. Select board size **50**, k=**5** (auto-set), depth=**auto** (= 2)
3. The AI responds in under 1 second using candidate-move restriction
4. Use scroll/wheel to navigate the board

---

*End of presentation. All code is in `src/`. Tests are in `tests/`. Benchmark results are in `report/`.*
