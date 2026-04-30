# Glossary — Tic-Tac-Toe AI Cheat Sheet

Quick-reference definitions for every important term in the project, organized by topic.

---

## Game Fundamentals

| Term | Definition |
|---|---|
| **n** | Board side length. The board is n×n. |
| **k** | Number of consecutive marks in a row/column/diagonal needed to win. |
| **m,n,k-game** | The formal name for this class of game. Tic-Tac-Toe = 3,3,3-game. Gomoku = 15,15,5-game. |
| **Turn** | One player placing one piece at an empty cell. |
| **Ply** | One half-move (one player's single action). Depth is measured in plies. |
| **Terminal state** | A board position where the game is over: someone won, or the board is full (draw). |
| **Draw** | Game ends with no winner because the board is full with no k-in-a-row. |
| **Available moves** | The set of all empty cells on the board right now. |
| **Board state** | The complete grid at a given moment — who is on every cell. |

---

## Search Algorithms

### Minimax

| Term | Definition |
|---|---|
| **Minimax** | An exhaustive algorithm that builds the full game tree and finds the optimal move assuming both players play perfectly. |
| **Maximizer** | The AI's role in Minimax. It always picks the move with the highest score. |
| **Minimizer** | The human's role in Minimax. It always picks the move with the lowest score (worst for the AI). |
| **Terminal score** | The score assigned to a game-over state. In Minimax: AI win = 10−depth, human win = depth−10, draw = 0. |
| **Depth** | How many moves deep the current search is from the root. Depth 0 = the current real board. Depth 1 = one move ahead. |
| **Score propagation** | After assigning scores to leaf nodes (game-over states), Minimax bubbles those scores back up — max at AI's turns, min at human's turns. |
| **Game tree** | The full branching structure of all possible moves and counter-moves from a given position. |
| **Nodes explored** | Counter of how many board positions the algorithm visited. Used to measure efficiency. |

### Alpha-Beta Pruning

| Term | Definition |
|---|---|
| **Alpha-Beta Pruning** | An optimization of Minimax that skips branches which cannot affect the final decision. Produces the identical move as Minimax but visits fewer nodes. |
| **Alpha (α)** | The best score the maximizer (AI) can guarantee so far. A lower bound. Starts at −∞. |
| **Beta (β)** | The best score the minimizer (human) can guarantee so far. An upper bound. Starts at +∞. |
| **Pruning condition** | When β ≤ α, the current branch is pruned (cut off). The opponent would never allow this path because they already have a better option elsewhere. |
| **Beta cut-off** | Pruning that happens at a maximizer node — the minimizer already has a result ≤ alpha, so this branch is ignored. |
| **Alpha cut-off** | Pruning that happens at a minimizer node — the maximizer already has a result ≥ beta, so this branch is ignored. |
| **Move ordering** | Sorting moves before exploring them (center → corners → edges). Good ordering means better moves are explored first, which helps Alpha-Beta cut more branches. |
| **Best case** | With perfect move ordering, Alpha-Beta reduces the tree from O(b^d) to O(b^(d/2)) — effectively doubling the searchable depth. |

### Depth-Limited Search

| Term | Definition |
|---|---|
| **Depth-limited search** | Alpha-Beta search that stops at a fixed depth instead of searching all the way to terminal states. |
| **Max depth** | The depth at which the search stops and evaluates the board with a heuristic instead of searching further. |
| **Cutoff policy** | The rule that decides what to return at each node: (1) terminal win/loss/draw if detected; (2) heuristic score if at max depth; (3) recurse otherwise. |
| **Depth cutoff** | The moment the search hits max_depth and calls the heuristic evaluator instead of going deeper. |
| **Heuristic cutoff** | Same as depth cutoff — the search is cut off and the heuristic takes over. |
| **Terminal score (heuristic AB)** | ±10000 (not ±10). Much larger than any heuristic value to guarantee real wins/losses always dominate estimated scores. |

---

## Candidate Moves & Scalability

| Term | Definition |
|---|---|
| **Candidate moves** | The subset of empty cells that the AI actually considers. On large boards, restricted to cells near existing pieces. |
| **Chebyshev distance** | A distance measure where diagonal moves cost the same as horizontal/vertical. Chebyshev radius 2 means a 5×5 square neighborhood around a cell. |
| **Radius-2 restriction** | For boards with n > 5, only empty cells within Chebyshev distance 2 of any occupied piece are considered. Collapses branching factor from n² to ~20–50. |
| **Branching factor** | The average number of candidate moves at each node in the search tree. Determines how fast the tree grows. Smaller = faster AI. |
| **Search tree explosion** | The exponential blowup of the game tree as board size grows. The main reason full search does not scale. |
| **Empty board special case** | When no pieces have been placed yet, `_get_candidate_moves` returns only the center cell — the strongest opening in any k-in-a-row game. |

---

## Pre-Search Tactical Checks

| Term | Definition |
|---|---|
| **Pre-search checks** | Fast pattern-matching passes that run before Alpha-Beta. If any fires, the function returns immediately without a full search. |
| **Immediate win** | A move that completes k-in-a-row for the AI right now. Always taken first. |
| **Immediate block** | A move that blocks the human from winning on the very next turn. Taken if no immediate win exists. |
| **Open-(k-1) forced threat** | Exactly k-1 consecutive human marks with BOTH adjacent ends empty. The human wins next turn at either end regardless of where you play — you must block one end right now. |
| **Forced loss** | A position where the opponent will win no matter what you do, unless you block the specific threat identified by the pre-search check. |

---

## Board Features

Features are numbers extracted from the board state. They are the input to the heuristic.

### Basic Counts

| Feature | What it means |
|---|---|
| `my_marks` | How many of my pieces are on the board. |
| `opp_marks` | How many opponent pieces are on the board. |
| `empty_cells` | How many cells are still empty. |

### Positional

| Feature | What it means |
|---|---|
| `my_center_control` | Number of my pieces in the center region (single cell for odd n; 2×2 block for even n). |
| `opp_center_control` | Same for the opponent. |

### Window-Based

A **window** is any contiguous k-length segment in one of 4 directions (row, column, diagonal ↘, diagonal ↗). Every possible winning line is a window.

| Feature | What it means |
|---|---|
| `my_open_1` | Windows with exactly 1 of my marks, rest empty. Early-stage coverage. |
| `my_open_2` … `my_open_{k-3}` | Windows with 2 to k-3 of my marks, rest empty. Partial lines in the building phase. |
| `my_open_{k-2}` | Windows with k-2 of my marks, rest empty. **The most strategically important partial line** — two marks short of a win. For k=5 this is three-in-a-row (`open_3`). Weighted ±15 in heuristic. |
| `my_immediate_wins` | Windows with k-1 of my marks + 1 empty. One move from winning. Weighted ±50 in heuristic. |
| `my_winning_windows` | Windows with all k of my marks. Already a completed winning line. Weighted ±1000 in heuristic. |
| `opp_open_i` | Same as above but for the opponent (negative signal). |
| `blocked_windows` | Windows containing pieces from BOTH players. Neither player can win through this window — it is dead. |
| `neutral_windows` | Windows with no pieces at all. Pure untouched potential. |

### Tactical — Fork Detection

| Feature | What it means |
|---|---|
| `my_two_way_threats` | Count of empty cells that, if played by me, create ≥2 simultaneous `immediate_win` threats. This is a **fork** — the opponent can only block one. Weighted +30. |
| `opp_two_way_threats` | Same for the opponent. If they have a fork available, you are about to lose unless you prevent it. Weighted −30. |

---

## Heuristic Evaluation

| Term | Definition |
|---|---|
| **Heuristic** | An estimate of board quality. Not exact (no full search), but good enough to guide decisions at the depth cutoff. |
| **Heuristic evaluation function** | The function that converts a board into a single number. Positive = good for AI. Negative = good for human. |
| **Weighted sum (linear evaluation)** | The formula: score = Σ (weight × feature). The oldest and most common structure for game AI evaluation functions. Used since Shannon's 1950 chess paper. |
| **Weight** | A number that says how much a feature matters. Higher weight = more influence on the final score. |
| **Priority hierarchy** | The ordering implied by the weights: winning line (1000) >> immediate threat (50) >> fork (30) >> open line (15) >> center (1). Each tier dominates all below it. |
| **Symmetry invariant** | The rule that `evaluate(board, X, O) = −evaluate(board, O, X)`. Evaluating the same board from the other player's perspective must give the equal and opposite score. |
| **`_build_weights(k, n)`** | The function that generates weights dynamically per (k, n). Cached via `@lru_cache`. Ensures the heuristic generalizes correctly to any board size. |
| **`open_{k-2}` not `open_{k-1}`** | Design choice: `open_{k-1}` would double-count `immediate_wins` (already weighted at ±50). `open_{k-2}` captures the "building phase" two moves short of a win. For k=5, this is `open_3`. |

---

## Performance & Optimization

| Term | Definition |
|---|---|
| **`check_line_at(r, c, player)`** | O(4k) winner check. Scans only lines through the last-placed cell. Used inside the search tree. ~1,250× faster than a full board scan on a 100×100 board. |
| **`check_winner()`** | O(n²k) winner check. Scans every cell on the entire board. Used only in game rules, the test suite, and end-of-game GUI detection. |
| **Window enumeration caching** | `@lru_cache(maxsize=64)` on `_enumerate_windows(n, k)`. Computes all possible k-length windows once per (n, k) pair. Every subsequent call is O(1). |
| **`@lru_cache`** | Python's built-in memoization decorator. Stores previously computed return values keyed by arguments. Prevents recomputing the same result. |
| **Memoization** | Caching the result of a function call so if the same inputs appear again, you return the stored answer instead of recomputing. |
| **Fork scan stratification** | `_count_two_way_threats`: full O(n²) scan for n≤15; radius-2 filtered scan for 15<n≤20; returns 0 for n>20. Keeps the heuristic fast on large boards. |
| **Background thread** | The AI computation runs on a separate thread so the GUI window never freezes while the AI is thinking. |
| **`root.after(0, callback)`** | Tkinter-safe way to post a result from a background thread back to the main UI thread. |
| **Transposition table** | (Not implemented.) A hash table storing previously evaluated board states. Prevents re-evaluating the same position reached via different move orders. Standard improvement in game AI. |
| **Zobrist hashing** | (Not implemented.) A fast technique to hash a board state to a 64-bit integer by XOR-ing random numbers assigned to each (cell, player) pair. Used to build transposition tables. |
| **Iterative deepening** | (Not implemented.) Search at depth 1, then depth 2, then depth 3, etc. until a time budget runs out. Uses the deepest completed result. Better move ordering than fixed-depth search. |

---

## Benchmarking

| Term | Definition |
|---|---|
| **Benchmark** | A controlled, reproducible experiment measuring AI performance — typically node count, time, or win rate under fixed conditions. |
| **Nodes explored** | How many board positions the algorithm visited during one move decision. Lower = faster AI. |
| **Pruning rate** | Percentage of nodes that Alpha-Beta skips compared to Minimax. 75% pruning = Alpha-Beta visited only 25% as many nodes. |
| **Exp 1 (3×3)** | Minimax vs Alpha-Beta: same move, 75% fewer nodes at game start. |
| **Exp 2 (4×4, k=3)** | Full Alpha-Beta (47,183 nodes) vs heuristic depth=2 (144 nodes): same move, 34× fewer nodes. |
| **Exp 3 (win rates)** | Heuristic AI vs random opponent over 20 games: 85% win on 3×3 (15% draw), 100% win on 4×4 and 5×5. |
| **Random policy** | A game policy that picks any legal move uniformly at random. Used as a baseline opponent in data collection and benchmarking. |
| **Tactical policy** | A game policy that prefers: (1) take immediate win, (2) block immediate loss, (3) play center, (4) random. Stronger than random, weaker than heuristic AI. |

---

## Data Collection

| Term | Definition |
|---|---|
| **Data collection** | Simulating many games and recording (board state, features, move played, final outcome) at every turn. Used to discover which features correlate with winning. |
| **Feature–outcome correlation** | Measuring how strongly a feature value predicts the final game result. High correlation → high heuristic weight. |
| **Outcome encoding** | +1 for a win, 0 for a draw, −1 for a loss. Used as the label in the dataset. |
| **Dataset (M5)** | `report/results/m5_dataset.csv` — thousands of game turns from 4×4 k=3 tactical-vs-random games, with all features and outcomes. |

---

## GUI

| Term | Definition |
|---|---|
| **Canvas-based rendering** | Drawing the board as rectangle + text items on a single Tkinter Canvas, not as Button widgets. The whole board is one bitmap — smooth scrolling, no per-widget overhead. 100×100 = 10,000 canvas items, not 10,000 Button objects. |
| **`_has_open_4(player)`** | Returns True if the given player has a run of ≥4 marks with BOTH adjacent ends empty (k≥5 only). Used to trigger "I already win." / "Seems like I'm losing." status messages. |
| **Auto depth** | The default search depth chosen by board size: n≤5 → 4, n≤10 → 3, n>10 → 2. Balances AI strength vs. response time. |
| **Status commentary** | "I already win." when AI has open-4; "Seems like I'm losing." when human does. Only shown on k=5 boards where 4-in-a-row is decisive but not yet a win. |
