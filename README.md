# Tic-Tac-Toe AI

A generalized n×n, k-in-a-row Tic-Tac-Toe engine with progressively advanced AI — from exhaustive Minimax to depth-limited Alpha-Beta search with a data-driven heuristic, playable on any board up to 100×100.

---

## Project Structure

```
tic_tac_toe_ai/
├── src/
│   ├── board.py              # Board state, move/undo, O(4k) winner check
│   ├── game.py               # Game rules, win/draw detection, turn management
│   ├── ai.py                 # Minimax, Alpha-Beta, depth-limited heuristic search
│   ├── features.py           # Board → feature dict (windows, threats, center control)
│   ├── heuristics.py         # Weighted-sum heuristic evaluation
│   ├── data_collection.py    # Game simulation → CSV/JSON datasets
│   ├── benchmark.py          # Reproducible experiments & result files
│   └── gui.py                # Tkinter GUI for human vs AI play
├── tests/
│   ├── test_board.py
│   ├── test_game.py
│   ├── test_ai.py
│   ├── test_features.py
│   ├── test_heuristics.py
│   └── test_data_collection.py
├── demo.ipynb                # End-to-end notebook (M4–M9)
├── main.py                   # CLI runner
├── conftest.py               # pytest path setup
├── requirements.txt
└── report/
    ├── results/              # CSV experiment outputs
    └── figures/              # Matplotlib plots
```

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the GUI (recommended entry point)
python src/gui.py

# Play via CLI (human vs human)
python main.py

# Run all tests
python -m pytest tests/ -q

# Run benchmarks
python src/benchmark.py

# Open the demo notebook
jupyter notebook demo.ipynb
```

---

## Milestones

### M1 — Game Engine ✅

- `Board` class: n×n grid, configurable k-in-a-row win condition, O(1) `is_full()` via `_empty` set
- `Game` class: turn management, win/draw detection, `check_winner()` full-board scan
- CLI runner (`main.py`) for human vs human play
- 18 unit tests covering Board and Game

### M2 — Minimax AI ✅

- `get_best_move(game, ai_player, human_player)` — exhaustive Minimax
- Terminal scoring: AI win = `10 − depth`, human win = `depth − 10`, draw = 0
- Move ordering: center → corners → edges
- `nodes_explored` counter for benchmarking

### M3 — Alpha-Beta Pruning ✅

- `get_best_move_ab(game, ai_player, human_player)` — same move quality as Minimax
- Prunes when `beta ≤ alpha`; typically 40–75% fewer nodes at game start
- `nodes_explored_ab` verified ≤ `nodes_explored` on identical boards

### M4 — Feature Extraction ✅

`FeatureExtractor.extract(board, my_player, opp_player)` → dict with 25+ interpretable features, generalized for any (n, k):

| Feature group | Features |
|---|---|
| Basic counts | `my_marks`, `opp_marks`, `empty_cells` |
| Positional | `my_center_control`, `opp_center_control` |
| Window patterns | `my_open_i` / `opp_open_i` (i = 1..k−1), `my_immediate_wins`, `opp_immediate_wins`, `my_winning_windows`, `opp_winning_windows`, `blocked_windows` |
| Tactical | `my_two_way_threats`, `opp_two_way_threats` (fork cells) |

Windows are all length-k contiguous segments in 4 directions. `_enumerate_windows(n, k)` is `@lru_cache`'d — computed once per (n, k).

### M5 — Data Collection ✅

`collect_dataset(num_games, n, k, matchup, seed)` simulates games and logs per-turn:
- Board state, all features, current player, move played, final outcome (+1 win / 0 draw / −1 loss)

Game policies: `random_policy`, `tactical_policy` (prefers immediate wins, blocks, center).

Dataset saved to `report/results/m5_dataset.csv`.

### M6 — Heuristic Evaluation ✅

`HeuristicEvaluator.evaluate(board, my_player, opp_player)` → float

Weighted sum with weights derived from M5 feature–outcome correlations:

```
score = 1000 × my_winning_windows   − 1000 × opp_winning_windows
       +  50 × my_immediate_wins    −   50 × opp_immediate_wins
       +  30 × my_two_way_threats   −   30 × opp_two_way_threats
       +  15 × my_open_{k-2}        −   15 × opp_open_{k-2}
       +   1 × my_center_control    −    1 × opp_center_control  (n ≤ 10 only)
```

Key design choices:
- **`open_{k-2}` not `open_{k-1}`**: for k=5 this is `open_3` (three-in-a-row), far more strategically relevant than `open_2`
- **Center weight disabled for n > 10**: the 2×2 center represents < 0.16% of a 50×50 board — noise, not signal
- **Weights symmetric**: `opp_X = −my_X` (required by `test_symmetry`)
- **`±10000` terminal scores** in search tree: guarantees actual wins/losses always dominate heuristic estimates

### M7 — Depth-Limited Alpha-Beta ✅

`get_best_move_heuristic(game, ai_player, human_player, max_depth)` is the scalable search method used by the GUI.

**Pre-search passes** (before full Alpha-Beta, in priority order):
1. Take an immediate AI win
2. Block an immediate human win
3. Block any open-(k−1) forced threat — k−1 consecutive human marks with both ends empty (opponent wins next turn whichever end you pick; must block now)

**Candidate move restriction** (`_get_candidate_moves`):
- For n ≤ 5: all available moves
- For n > 5: empty cells within **Chebyshev radius 2** of any occupied cell — collapses branching factor from n² to ~20–50

**Default depth schedule** (GUI "auto"):

| Board size | Depth |
|---|---|
| n ≤ 5 | 4 |
| n ≤ 10 | 3 |
| n > 10 | 2 |

### M8 — Benchmarking ✅

Three reproducible experiments in `src/benchmark.py`:

#### Exp 1 — Minimax vs Alpha-Beta (3×3, k=3)

| Position | Minimax nodes | AB nodes | Pruned |
|---|---|---|---|
| Early (2 placed) | 7,979 | 1,988 | **75%** |
| Mid (4 placed) | 149 | 88 | 41% |
| Late (6 placed) | 11 | 11 | 0% |

Alpha-Beta prunes up to 75% of nodes while returning the identical move.

#### Exp 2 — Full Search vs Depth-Limited Heuristic (4×4, k=3)

| Method | Nodes | Correct? |
|---|---|---|
| Full Alpha-Beta (reference) | 47,183 | — |
| Heuristic depth=1 | 12 | ✗ |
| Heuristic depth=2 | 144 | ✓ **34× fewer nodes** |
| Heuristic depth=3 | 288 | ✓ |
| Heuristic depth=4 | 1,371 | ✓ |

#### Exp 3 — Heuristic AI vs Random (20 games per config)

| Board | k | Depth | Win% | Draw% | Loss% |
|---|---|---|---|---|---|
| 3×3 | 3 | 4 | 85% | 15% | 0% |
| 4×4 | 3 | 3 | **100%** | 0% | 0% |
| 5×5 | 4 | 2 | **100%** | 0% | 0% |

Results and plots saved to `report/results/` and `report/figures/`.

### M9 — GUI, Notebook & Report ✅

**GUI (`src/gui.py`):**
- Canvas-based board rendering — single bitmap surface, smooth scroll on any board size
- Scrollbars + mouse-wheel scroll (Shift+wheel for horizontal)
- Board size dropdown: 3, 30, 50, 100 — k auto-syncs to min(n, 5)
- AI runs on a background thread (UI never freezes)
- Last-placed cell highlighted with a green outline
- Winning k-in-a-row cells highlighted in green
- Status commentary: "I already win." / "Seems like I'm losing." when a player has 4-in-a-row with both ends open (k=5 only)
- User-overridable search depth

**Notebook (`demo.ipynb`):** demonstrates M4–M9 end-to-end — feature extraction, data collection, heuristic scoring, benchmarks, and a programmatic AI move example. All cells run top-to-bottom without errors.

---

## Scalability

The AI is designed to play on boards up to 100×100 with k=5:

| Technique | Where | Effect |
|---|---|---|
| Candidate-move restriction (radius=2) | `ai.py: _get_candidate_moves` | Branching factor n²→20–50 |
| Winner check O(4k) instead of O(n²k) | `board.py: check_line_at` | ~200× faster at each node |
| Window enumeration caching | `features.py: _enumerate_windows` | Computed once per (n, k) |
| Fork scan guard (n > 20 → skip) | `features.py: _count_two_way_threats` | Prevents O(n²) bottleneck |
| Fork scan filter (15 < n ≤ 20) | `features.py: _count_two_way_threats` | ~5× fewer cells scanned |
| Pre-search forced-win checks | `ai.py: get_best_move_heuristic` | Avoids full search when answer is obvious |
| k-relative open-line weights | `heuristics.py: _build_weights` | Generalizes meaningfully to any k |

---

## Tests

57 tests, all passing. Run with:

```bash
python -m pytest tests/ -q
```

| File | Coverage | Count |
|---|---|---|
| `test_board.py` | Board state, moves, undo, clone | 9 |
| `test_game.py` | Win detection (all 4 directions), draw, turn switching | 9 |
| `test_ai.py` | Minimax, Alpha-Beta, heuristic AB (cutoff, terminal bypass, efficiency) | 20 |
| `test_features.py` | Feature extraction, symmetry, perspective, even/odd boards | 8 |
| `test_heuristics.py` | Score sign/magnitude on crafted boards, symmetry invariant | 6 |
| `test_data_collection.py` | Dataset generation, outcome encoding, policy correctness | 5 |
