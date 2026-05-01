# Tic-Tac-Toe AI

A generalized n×n, k-in-a-row Tic-Tac-Toe engine with progressively advanced AI, from Minimax to feature-based heuristic search.

---

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Play via CLI
python main.py

# Run all tests
python -m pytest tests/

# Run benchmarks
python src/benchmark.py

# Launch GUI
python src/gui.py
```

---

## Project Milestones

### Milestone 1 — Game Engine ✅
**Feb 25 – Mar 7**

Objective: Build a fully playable CLI Tic-Tac-Toe game on an n×n board.

- Generalized `Board` class supporting any n×n size and k-in-a-row win condition
- `Game` class managing turns, win/draw detection, and player input
- CLI runner for human vs human play
- Unit tests for `Board` and `Game`

---

### Milestone 2 — Minimax AI ✅
**Mar 8 – Mar 18**

Objective: Implement an AI player using the Minimax algorithm.

- `AIPlayer` class in `ai.py` using full Minimax search
- Node count tracking for performance analysis
- Unit tests verifying correctness on 3×3 boards (optimal play, never loses)

---

### Milestone 3 — Alpha-Beta Pruning ✅
**Mar 19 – Mar 28**

Objective: Improve search efficiency with Alpha-Beta pruning and depth-limited search.

- Alpha-Beta pruning integrated into Minimax
- Depth-limited search with a basic heuristic at cutoff
- Node count comparison: Minimax vs Alpha-Beta
- Unit tests for pruning correctness

---

### Milestone 4 — Feature Extraction & State Representation
**Mar 29 – Apr 4**

Objective: Design and implement a feature extraction system for board states.

**Why:** Enables interpretable, generalizable heuristics and data-driven evaluation.

**Deliverables:**
- `src/features.py` — functions/classes to extract board features (open lines, center control, near-wins, threat counts, blocked windows, etc.)

**Approach:**
- Features must generalize across any (n, k) — base them on windows/segments of length k, not hardcoded 3×3 patterns
- Keep features interpretable: counts, pattern flags (e.g. `ai_open_i`, `opp_near_win`, `center_control`)

**Acceptance criteria:**
- Feature extractor works for any n×n, k-in-a-row board
- Unit tests on crafted boards covering: immediate AI threat, immediate opponent threat, center-dominant position, blocked windows, near-win vs neutral (`tests/test_features.py`)
- Feature vectors printed and explained for sample states to confirm correctness before heuristic work begins

---

### Milestone 5 — Data Collection & Outcome Logging
**Apr 5 – Apr 9**

Objective: Simulate games and log board states, feature vectors, and outcomes to inform heuristic design.

**Why:** Grounds the heuristic in evidence rather than pure intuition — this is the "data-driven" part your professor is asking for. Building this before the heuristic gives the project a much stronger narrative.

**Deliverables:**
- `src/data_collection.py` — simulate random-vs-random and shallow AI-vs-random games, extract features at each state, record final outcome from the current player's perspective, save to CSV/JSON

**Approach:**
- Keep the pipeline simple: simulate → extract features → log outcome
- No ML required at this stage — the goal is to produce evidence of which features correlate with winning

**Acceptance criteria:**
- Data files contain varied states with correct feature/outcome labels
- Exploratory analysis shows which features correlate with winning (can be a simple printout or notebook cell)

---

### Milestone 6 — Heuristic Evaluation Function
**Apr 10 – Apr 16**

Objective: Implement a heuristic evaluation function using features extracted in M4, informed by data from M5.

**Why:** Allows the AI to evaluate non-terminal states, enabling practical depth-limited search. Building this after data collection means the weights can be visibly tied to evidence.

**Deliverables:**
- `src/heuristics.py` — documented evaluation function combining features into a numeric score

**Approach:**
- Weighted sum of features with initial hand-tuned weights
- Lightly refine weights using correlations observed in M5 data
- Keep logic explainable — one clear formula, easy to justify in a report

**Acceptance criteria:**
- Heuristic returns sensible scores (positive = good for maximizer, negative = good for minimizer)
- Unit tests verify correct sign and relative magnitude on crafted boards (`tests/test_heuristics.py`)
- Heuristic rankings shown for "good" vs "bad" positions to demonstrate alignment with intuition

---

### Milestone 7 — Depth-Limited Alpha-Beta with Cutoff Policy
**Apr 17 – Apr 23**

Objective: Update the AI to use depth-limited Alpha-Beta, calling the heuristic at the depth cutoff.

**Why:** Makes the AI practical for larger boards (4×4+); matches project requirements for scalable, cutoff-driven search.

**Deliverables:**
- Updated `src/ai.py` with depth-limited Alpha-Beta and explicit cutoff logic

**Approach:**
- Add a `max_depth` parameter; at cutoff call `heuristics.evaluate()` instead of continuing search
- Implement a clear cutoff policy — e.g. full search on 3×3, fixed depth on 4×4+, or empty-cells-based depth scaling
- The cutoff policy itself should be visible and documented (the professor called this out specifically)

**Acceptance criteria:**
- AI responds in practical time on 4×4+ boards
- Unit tests verify correct cutoff behavior and heuristic call
- Benchmark showing node counts and runtimes for various depths and board sizes

---

### Milestone 8 — Benchmarking & Experiments
**Apr 24 – Apr 28**

Objective: Systematically benchmark AI performance, scalability, and heuristic quality.

**Why:** Provides evidence for the final report; supports claims about efficiency and generalization.

**Deliverables:**
- `src/benchmark.py` — reproducible experiment scripts and result files
- Tables/plots of node counts, runtimes, and win rates saved to `report/results/` and `report/figures/`

**Experiments:**
- Minimax vs Alpha-Beta node count and runtime comparison
- Full search vs depth-limited search outcome comparison
- Heuristic quality across board sizes (3×3, 4×4, 5×5)
- Effect of move ordering on pruning efficiency (if implemented)

---

### Milestone 9 — GUI, Notebook & Report
**Apr 29 – May 5**

Objective: Add a GUI, a demo notebook, and complete project documentation.

**Why:** Required for presentation, usability, and reproducibility.

**Deliverables:**
- `src/gui.py` — Tkinter GUI for human vs AI play
- `demo.ipynb` — Jupyter notebook demonstrating AI, benchmarks, and experiments end-to-end
- Final `README.md` and report artifacts in `report/`

**Acceptance criteria:**
- GUI supports human vs AI with configurable board size
- Notebook runs top-to-bottom without errors
- README is clear, complete, and includes experiment results

---

## Experiment Results

All experiments are reproducible by running `python src/benchmark.py`.

### Exp 1 — Minimax vs Alpha-Beta (3×3, k=3)

| Position | Minimax nodes | Alpha-Beta nodes | Pruned |
|---|---|---|---|
| Early (2 placed) | 7,979 | 1,988 | **75%** |
| Mid (4 placed) | 149 | 88 | 41% |
| Late (6 placed) | 11 | 11 | 0% |

Alpha-Beta prunes up to 75% of nodes while always returning the identical move.

### Exp 2 — Full Search vs Depth-Limited Heuristic (4×4, k=3)

| Method | Nodes | Correct move? |
|---|---|---|
| Full Alpha-Beta (reference) | 47,183 | — |
| Heuristic depth=1 | 12 | ✗ (misses threat) |
| Heuristic depth=2 | 144 | ✓ |
| Heuristic depth=3 | 288 | ✓ |
| Heuristic depth=4 | 1,371 | ✓ |

At depth=2 the heuristic finds the correct move using **34× fewer nodes** than full search.

### Exp 3 — Heuristic AI vs Random (20 games per config)

| Board | k | Depth | Win% | Draw% | Loss% |
|---|---|---|---|---|---|
| 3×3 | 3 | 4 | 85% | 15% | 0% |
| 4×4 | 3 | 3 | **100%** | 0% | 0% |
| 5×5 | 4 | 2 | **100%** | 0% | 0% |

The heuristic AI never loses to random play. The 15% draw rate on 3×3 reflects depth-4 not being full exhaustive search.

### Practical Depth Limits

| Board size | Recommended depth | Approx. time per move |
|---|---|---|
| n ≤ 5 | 4 | < 400ms |
| n ≤ 10 | 3 | < 6s |
| n > 10 | 2 | < 2s |

Bottleneck for large boards is Python-level feature extraction (~0.5ms/node). The candidate-move restriction (`radius=2`) keeps branching factor at 20–50 regardless of board size.

---

## Nice-to-Have

- **Move ordering for Alpha-Beta** — sort moves by a quick heuristic before expanding to improve pruning; benchmark the speedup
- **Heuristic tuning via data** — use collected game data to tune feature weights (grid search or simple regression)
- **Configurable AI parameters in CLI/GUI** — let users set search depth, heuristic, and player symbol at runtime

## Stretch Goals

- **Learning-based heuristic** — train a logistic regression model on collected game data to replace hand-tuned weights
- **Generalization tests** — evaluate AI on board sizes not seen during heuristic tuning
- **AI vs human logging** — log and analyze real human vs AI games

---

## Project Structure

```
tic_tac_toe_ai/
├── src/
│   ├── board.py
│   ├── game.py
│   ├── ai.py
│   ├── features.py          # M4
│   ├── data_collection.py   # M5
│   ├── heuristics.py        # M6
│   ├── benchmark.py         # M8
│   ├── gui.py               # M9
│   └── utils.py
├── tests/
│   ├── test_board.py
│   ├── test_game.py
│   ├── test_ai.py
│   ├── test_features.py     # M4
│   └── test_heuristics.py   # M6
├── demo.ipynb               # M9
├── main.py
├── README.md
├── requirements.txt
└── report/
    ├── figures/
    └── results/
```
