"""
Milestone 8 — Benchmarking & Experiments

Run:  python src/benchmark.py

Outputs
-------
report/results/exp1_minimax_vs_ab.csv
report/results/exp2_search_vs_heuristic.csv
report/results/exp3_win_rates.csv
report/figures/exp1_node_counts.png       (requires matplotlib)
report/figures/exp2_nodes_by_depth.png    (requires matplotlib)
report/figures/exp3_win_rates.png         (requires matplotlib)
"""

import csv
import os
import random
import sys
import time

# Allow `python src/benchmark.py` to find the src package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.ai import MinimaxAI
from src.game import Game

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP1_POSITIONS = [
    {
        "name": "early_2placed",
        "n": 3, "k": 3,
        "placements": [("X", 0, 0), ("O", 2, 2)],
    },
    {
        "name": "mid_4placed",
        "n": 3, "k": 3,
        "placements": [("X", 0, 0), ("O", 0, 1), ("O", 1, 0), ("X", 2, 2)],
    },
    {
        "name": "late_6placed",
        "n": 3, "k": 3,
        "placements": [("X", 0, 0), ("O", 0, 1), ("X", 0, 2),
                       ("O", 1, 0), ("X", 1, 1), ("O", 2, 2)],
    },
]

# 4×4 k=3 midgame: O has immediate win threat at (0,2); depth=1 misses it.
EXP2_PLACEMENTS = [("O", 0, 0), ("O", 0, 1), ("X", 1, 0), ("X", 2, 2)]
EXP2_DEPTHS = [1, 2, 3, 4]

EXP3_CONFIGS = [
    {"n": 3, "k": 3, "max_depth": 4},
    {"n": 4, "k": 3, "max_depth": 3},
    {"n": 5, "k": 4, "max_depth": 2},
]
EXP3_GAMES_PER_SIDE = 10  # 10 as X + 10 as O = 20 games per config

RESULTS_DIR = "report/results"
FIGURES_DIR = "report/figures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_board(game: Game, placements: list) -> None:
    for player, r, c in placements:
        game.board.make_move(r, c, player)


def _write_csv(path: str, rows: list) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows -> {path}")


def _simulate_one_game(n: int, k: int, ai_player: str, random_player: str,
                       max_depth: int, rng: random.Random):
    """Play one game: heuristic AI vs random. Returns winner symbol or None."""
    ai = MinimaxAI()
    game = Game(n, k, "X", "O")
    while not game.is_terminal():
        cur = game.current_player
        if cur == ai_player:
            move = ai.get_best_move_heuristic(game, ai_player, random_player,
                                               max_depth=max_depth)
        else:
            move = rng.choice(game.get_available_moves())
        if move is not None:
            game.make_move(move[0], move[1])
        if not game.is_terminal():
            game.switch_turn()
    return game.check_winner()


# ---------------------------------------------------------------------------
# Experiment 1 — Minimax vs Alpha-Beta
# ---------------------------------------------------------------------------

def run_exp1_minimax_vs_ab() -> list:
    """Compare node counts: exhaustive Minimax vs Alpha-Beta on 3×3 positions."""
    ai = MinimaxAI()
    rows = []
    for pos in EXP1_POSITIONS:
        for method in ("minimax", "alpha_beta"):
            game = Game(pos["n"], pos["k"], "X", "O")
            _seed_board(game, pos["placements"])
            t0 = time.perf_counter()
            if method == "minimax":
                move = ai.get_best_move(game, "X", "O")
                nodes = ai.nodes_explored
            else:
                move = ai.get_best_move_ab(game, "X", "O")
                nodes = ai.nodes_explored_ab
            elapsed_ms = (time.perf_counter() - t0) * 1000
            rows.append({
                "position_name": pos["name"],
                "n": pos["n"],
                "k": pos["k"],
                "pieces_placed": len(pos["placements"]),
                "method": method,
                "nodes_explored": nodes,
                "time_ms": round(elapsed_ms, 3),
                "best_move_row": move[0] if move else None,
                "best_move_col": move[1] if move else None,
            })
    return rows


# ---------------------------------------------------------------------------
# Experiment 2 — Full Search vs Depth-Limited Heuristic
# ---------------------------------------------------------------------------

def run_exp2_search_vs_heuristic() -> list:
    """Compare exhaustive AB (reference) vs heuristic at increasing depths."""
    ai = MinimaxAI()
    rows = []

    game = Game(4, 3, "X", "O")
    _seed_board(game, EXP2_PLACEMENTS)
    t0 = time.perf_counter()
    ref_move = ai.get_best_move_ab(game, "X", "O")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    rows.append({
        "method": "full_ab",
        "max_depth": "",
        "nodes_explored": ai.nodes_explored_ab,
        "time_ms": round(elapsed_ms, 3),
        "best_move_row": ref_move[0] if ref_move else None,
        "best_move_col": ref_move[1] if ref_move else None,
        "matches_full_ab": True,
    })

    for d in EXP2_DEPTHS:
        game = Game(4, 3, "X", "O")
        _seed_board(game, EXP2_PLACEMENTS)
        t0 = time.perf_counter()
        move = ai.get_best_move_heuristic(game, "X", "O", max_depth=d)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        rows.append({
            "method": "heuristic_ab",
            "max_depth": d,
            "nodes_explored": ai.nodes_explored_h,
            "time_ms": round(elapsed_ms, 3),
            "best_move_row": move[0] if move else None,
            "best_move_col": move[1] if move else None,
            "matches_full_ab": (move == ref_move),
        })
    return rows


# ---------------------------------------------------------------------------
# Experiment 3 — Heuristic Quality (win rate vs random)
# ---------------------------------------------------------------------------

def run_exp3_win_rates() -> list:
    """Play heuristic AI vs random over 20 games per board config."""
    rows = []
    for cfg in EXP3_CONFIGS:
        n, k, depth = cfg["n"], cfg["k"], cfg["max_depth"]
        rng = random.Random(42)
        game_id = 0
        for ai_player, random_player in [("X", "O"), ("O", "X")]:
            for _ in range(EXP3_GAMES_PER_SIDE):
                winner = _simulate_one_game(n, k, ai_player, random_player, depth, rng)
                if winner == ai_player:
                    ai_result = "win"
                elif winner is None:
                    ai_result = "draw"
                else:
                    ai_result = "loss"
                rows.append({
                    "n": n, "k": k, "max_depth": depth,
                    "game_id": game_id,
                    "ai_player": ai_player,
                    "random_player": random_player,
                    "winner": winner if winner else "DRAW",
                    "ai_result": ai_result,
                })
                game_id += 1
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_exp1(rows: list) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("  [skip] matplotlib not installed — skipping exp1 plot")
        return
    import numpy as np
    positions = [r["position_name"] for r in rows if r["method"] == "minimax"]
    mm_nodes  = [r["nodes_explored"]  for r in rows if r["method"] == "minimax"]
    ab_nodes  = [r["nodes_explored"]  for r in rows if r["method"] == "alpha_beta"]

    x = np.arange(len(positions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, mm_nodes, width, label="Minimax", color="steelblue")
    bars2 = ax.bar(x + width / 2, ab_nodes, width, label="Alpha-Beta", color="darkorange")
    ax.set_yscale("log")
    ax.set_xlabel("Board Position")
    ax.set_ylabel("Nodes Explored (log scale)")
    ax.set_title("Node Count: Minimax vs Alpha-Beta (3×3, k=3)")
    ax.set_xticks(x)
    ax.set_xticklabels(positions, rotation=15, ha="right")
    ax.bar_label(bars1, padding=3)
    ax.bar_label(bars2, padding=3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp1_node_counts.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_exp2(rows: list) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("  [skip] matplotlib not installed — skipping exp2 plot")
        return
    h_rows   = [r for r in rows if r["method"] == "heuristic_ab"]
    full_row = next(r for r in rows if r["method"] == "full_ab")
    depths = [r["max_depth"] for r in h_rows]
    nodes  = [r["nodes_explored"] for r in h_rows]
    colors = ["red" if not r["matches_full_ab"] else "green" for r in h_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, nodes, marker="o", color="steelblue", label="Heuristic AB")
    for d, n, c in zip(depths, nodes, colors):
        ax.plot(d, n, "o", color=c, markersize=10, zorder=5)
    ax.axhline(full_row["nodes_explored"], linestyle="--", color="gray",
               label=f"Full Alpha-Beta ({full_row['nodes_explored']:,} nodes)")
    ax.set_yscale("log")
    ax.set_xlabel("Max Search Depth")
    ax.set_ylabel("Nodes Explored (log scale)")
    ax.set_title("Nodes Explored: Full Alpha-Beta vs Depth-Limited Heuristic (4×4, k=3)")
    ax.legend()
    ax.annotate("red = wrong move  |  green = correct move",
                xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8, color="gray")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp2_nodes_by_depth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_exp3(rows: list) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("  [skip] matplotlib not installed — skipping exp3 plot")
        return
    configs = EXP3_CONFIGS
    labels, wins_pct, draws_pct, losses_pct = [], [], [], []
    for cfg in configs:
        n, k, d = cfg["n"], cfg["k"], cfg["max_depth"]
        subset = [r for r in rows if r["n"] == n and r["k"] == k and r["max_depth"] == d]
        total = len(subset)
        wins   = sum(1 for r in subset if r["ai_result"] == "win")
        draws  = sum(1 for r in subset if r["ai_result"] == "draw")
        losses = sum(1 for r in subset if r["ai_result"] == "loss")
        labels.append(f"{n}×{n}\nd={d}")
        wins_pct.append(wins / total * 100)
        draws_pct.append(draws / total * 100)
        losses_pct.append(losses / total * 100)

    import numpy as np
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, wins_pct,   label="Win",  color="seagreen")
    ax.bar(x, draws_pct,  bottom=wins_pct, label="Draw", color="lightgray")
    ax.bar(x, losses_pct, bottom=[w + d for w, d in zip(wins_pct, draws_pct)],
           label="Loss", color="tomato")
    ax.set_ylim(0, 110)
    ax.set_xlabel("Board Config")
    ax.set_ylabel("Outcome Rate (%)")
    ax.set_title("Heuristic AI vs Random: Win/Draw/Loss Rates")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for i, (w, d, _) in enumerate(zip(wins_pct, draws_pct, losses_pct)):
        ax.text(i, w / 2, f"{w:.0f}%", ha="center", va="center",
                color="white", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "exp3_win_rates.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def _print_exp1_summary(rows: list) -> None:
    print(f"\n  {'Position':<18} {'Method':<12} {'Nodes':>8}  {'Time(ms)':>9}  {'Move':>8}")
    print("  " + "-" * 60)
    for r in rows:
        move = f"({r['best_move_row']},{r['best_move_col']})"
        print(f"  {r['position_name']:<18} {r['method']:<12} {r['nodes_explored']:>8,}"
              f"  {r['time_ms']:>9.2f}  {move:>8}")


def _print_exp2_summary(rows: list) -> None:
    print(f"\n  {'Method':<14} {'Depth':>6}  {'Nodes':>8}  {'Time(ms)':>9}  "
          f"{'Move':>8}  {'Matches AB':>10}")
    print("  " + "-" * 62)
    for r in rows:
        move = f"({r['best_move_row']},{r['best_move_col']})"
        depth = str(r["max_depth"]) if r["max_depth"] != "" else "full"
        print(f"  {r['method']:<14} {depth:>6}  {r['nodes_explored']:>8,}"
              f"  {r['time_ms']:>9.2f}  {move:>8}  {str(r['matches_full_ab']):>10}")


def _print_exp3_summary(rows: list) -> None:
    print(f"\n  {'Board':<8} {'k':>3} {'depth':>6}  {'W':>4} {'D':>4} {'L':>4}  "
          f"{'Win%':>6}")
    print("  " + "-" * 44)
    for cfg in EXP3_CONFIGS:
        n, k, d = cfg["n"], cfg["k"], cfg["max_depth"]
        subset = [r for r in rows if r["n"] == n and r["k"] == k and r["max_depth"] == d]
        total  = len(subset)
        wins   = sum(1 for r in subset if r["ai_result"] == "win")
        draws  = sum(1 for r in subset if r["ai_result"] == "draw")
        losses = sum(1 for r in subset if r["ai_result"] == "loss")
        print(f"  {f'{n}×{n}':<8} {k:>3} {d:>6}  {wins:>4} {draws:>4} {losses:>4}"
              f"  {wins/total*100:>5.0f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("[Exp 1] Minimax vs Alpha-Beta (3×3)...")
    rows1 = run_exp1_minimax_vs_ab()
    _write_csv(os.path.join(RESULTS_DIR, "exp1_minimax_vs_ab.csv"), rows1)
    _print_exp1_summary(rows1)
    _plot_exp1(rows1)

    print("\n[Exp 2] Full Alpha-Beta vs Depth-Limited Heuristic (4×4, k=3)...")
    rows2 = run_exp2_search_vs_heuristic()
    _write_csv(os.path.join(RESULTS_DIR, "exp2_search_vs_heuristic.csv"), rows2)
    _print_exp2_summary(rows2)
    _plot_exp2(rows2)

    print("\n[Exp 3] Heuristic AI vs Random — win rates...")
    rows3 = run_exp3_win_rates()
    _write_csv(os.path.join(RESULTS_DIR, "exp3_win_rates.csv"), rows3)
    _print_exp3_summary(rows3)
    _plot_exp3(rows3)

    print("\nDone. Results in report/results/  Figures in report/figures/")


if __name__ == "__main__":
    main()
