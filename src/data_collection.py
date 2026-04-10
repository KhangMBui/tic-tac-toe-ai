import argparse
import csv
import json
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from src.features import FeatureExtractor
from src.game import Game

# A move is represented as (row, col)
Move = Tuple[int, int]

# A policy is any function that:
# - looks at the current game
# - knows which player is moving and who the opponent is
# - can use randomness if needed
# - returns one legal move
Policy = Callable[[Game, str, str, random.Random], Move]


def board_to_string(grid: List[List[Optional[str]]]) -> str:
    """
    Serialize the board grid into a compact JSON string.

    Why store the board as a string?
    --------------------------------
    When we log dataset rows, we want to preserve the exact board state
    before each move. Converting the board to a JSON string makes it easy
    to save into CSV/JSON files and inspect later.

    Example:
        [[None, "X"], ["O", None]]
    might become:
        [[null,"X"],["O",null]]
    """
    return json.dumps(grid, separators=(",", ":"))


def outcome_from_perspective(winner: Optional[str], perspective_player: str) -> int:
    """
    Encode the final result from one player's point of view.

    Returns:
        1  -> this player eventually wins
        0  -> the game ends in a draw
        -1 -> this player eventually loses

    Why this matters:
    -----------------
    Our dataset logs each row from the CURRENT player's perspective at the
    time the row is created. So later, when the game ends, we need to label
    that row with the outcome relative to that same player.

    Example:
        winner = "X"
        perspective_player = "X"  -> +1
        perspective_player = "O"  -> -1
        winner = None             -> 0
    """
    if winner is None:
        return 0
    return 1 if winner == perspective_player else -1


def _find_immediate_winning_move(game: Game, player: str) -> Optional[Move]:
    """
    Return a legal move that wins immediately for 'player', if one exists.

    Strategy:
    ---------
    For every legal move:
    1. Temporarily make the move
    2. Check whether it produces a winning board
    3. Undo the move
    4. Return the move if it wins immediately

    If no such move exists, return None.

    Why this helper exists:
    -----------------------
    It supports the simple tactical policy below, which is meant to be a
    stronger baseline than pure random play without running full search.
    """
    for row, col in game.get_available_moves():
        game.board.make_move(row, col, player)
        is_win = game.check_winner() == player
        game.board.undo_move(row, col)
        if is_win:
            return row, col
    return None


def random_policy(game: Game, player: str, opponent: str, rng: random.Random) -> Move:
    """
    Choose any legal move uniformly at random.

    Why include this policy:
    ------------------------
    Random play is useful for:
    - generating a diverse variety of board states
    - building a baseline dataset
    - comparing stronger policies against a weak opponent
    """
    return rng.choice(game.get_available_moves())


def tactical_policy(game: Game, player: str, opponent: str, rng: random.Random) -> Move:
    """
    A shallow, non-search policy.

    Decision order:
    ---------------
    1. If I can win immediately, do it.
    2. Else, if the opponent can win immediately next turn, block it.
    3. Else, prefer central cells or corners.
    4. Else, choose randomly among equally preferred moves.

    Why this policy exists:
    -----------------------
    This is a useful "middle ground" policy:
    - stronger than random play
    - much cheaper than Minimax/Alpha-Beta
    - can generate more meaningful training data than random-only self-play

    This makes it a good candidate for early data collection before the
    full heuristic-guided AI is ready.
    """
    # Step 1: take a winning move if available
    win_move = _find_immediate_winning_move(game, player)
    if win_move is not None:
        return win_move

    # Step 2: block opponent's immediate win if needed
    block_move = _find_immediate_winning_move(game, opponent)
    if block_move is not None:
        return block_move

    moves = game.get_available_moves()
    n = game.n

    def move_priority(move: Move) -> int:
        """
        Lower number = better priority.

        Priority order:
        0 -> center / center region
        1 -> corners
        2 -> everything else

        This is a lightweight positional preference, not a deep search.
        """
        r, c = move

        # Odd-sized board: one exact center cell
        if n % 2 == 1 and (r, c) == (n // 2, n // 2):
            return 0

        # Even-sized board: treat the 2x2 middle block as the center region
        if n % 2 == 0:
            a = n // 2 - 1
            b = n // 2
            if (r, c) in [(a, a), (a, b), (b, a), (b, b)]:
                return 0

        # Corners are usually stronger than generic edge/interior cells
        if (r, c) in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]:
            return 1
        return 2

    # Sort by the heuristic priority above
    moves = sorted(moves, key=move_priority)

    # Keep some randomness among equally preferred moves so the dataset
    # does not become too deterministic and repetitive.
    best_priority = move_priority(moves[0])
    best_moves = [m for m in moves if move_priority(m) == best_priority]
    return rng.choice(best_moves)


def _simulate_one_game(
    game_id: int,
    n: int,
    k: int,
    policy_x: Policy,
    policy_o: Policy,
    rng: random.Random,
    extractor: FeatureExtractor,
) -> List[Dict]:
    """
    Simulate one full game and return one logged row per turn.

    What each row represents:
    -------------------------
    A single row corresponds to one decision point in the game:
    - the board state BEFORE the move
    - features extracted from that board
    - the move chosen
    - the final game outcome (filled in after the game ends)

    Very important detail:
    ----------------------
    Features are always extracted from the CURRENT player's perspective.

    That means:
    - "my_open_2" refers to the player whose turn it is in that row
    - "opp_open_2" refers to the opposing player in that row

    This perspective-based logging makes the dataset more consistent for
    later heuristic analysis.
    """
    game = Game(n, k, "X", "O")
    rows: List[Dict] = []
    turn_index = 0

    while not game.is_terminal():
        current = game.current_player
        opponent = "O" if current == "X" else "X"

        # Extract features BEFORE making the move, from the perspective
        # of the player who is about to act.
        features = extractor.extract(game.board, current, opponent)

        # Select which policy controls the current player
        policy = policy_x if current == "X" else policy_o
        move_row, move_col = policy(game, current, opponent, rng)

        # Log the board/features/action for this turn.
        # The final outcome is not known yet, so we append it later after
        # the game finishes.
        row = {
            "game_id": game_id,
            "turn_index": turn_index,
            "current_player": current,
            "opponent_player": opponent,
            "n": n,
            "k": k,
            "board_before_move": board_to_string(game.board.grid),
            "move_row": move_row,
            "move_col": move_col,
        }
        row.update(features)
        rows.append(row)

        game.make_move(move_row, move_col)

        # If the game is not over, switch to the next player
        if not game.is_terminal():
            game.switch_turn()
        turn_index += 1

    # Once the game ends, attach the final label to every row.
    winner = game.check_winner()  # None => draw
    winner_label = winner if winner is not None else "DRAW"

    for row in rows:
        row["final_winner"] = winner_label
        row["final_outcome"] = outcome_from_perspective(winner, row["current_player"])

    return rows


def collect_dataset(
    num_games: int,
    n: int,
    k: int,
    matchup: str = "random_vs_random",
    seed: int = 42,
) -> List[Dict]:
    """
    Generate dataset rows by simulating many games.

    Supported matchups:
    -------------------
    random_vs_random
        Both players use uniformly random legal moves.

    tactical_vs_random
        One player uses the shallow tactical policy and the other uses random.
        We alternate which side gets the tactical policy each game to reduce
        first-player bias.

    Why this function matters:
    --------------------------
    This is the main data collection entry point. It creates the dataset that
    will later help us:
    - inspect feature distributions
    - compare winning vs losing states
    - design or refine heuristic weights
    """
    if num_games <= 0:
        raise ValueError("num_games must be > 0")

    rng = random.Random(seed)
    extractor = FeatureExtractor()
    all_rows: List[Dict] = []

    for game_id in range(num_games):
        # Choose policies based on the requested matchup
        if matchup == "random_vs_random":
            policy_x = random_policy
            policy_o = random_policy

        elif matchup == "tactical_vs_random":
            # Alternate sides so the tactical policy is not always X.
            # This helps reduce bias from always moving first.
            if game_id % 2 == 0:
                policy_x = tactical_policy
                policy_o = random_policy
            else:
                policy_x = random_policy
                policy_o = tactical_policy

        else:
            raise ValueError(
                "Unknown matchup. Use random_vs_random or tactical_vs_random."
            )

        game_rows = _simulate_one_game(
            game_id=game_id,
            n=n,
            k=k,
            policy_x=policy_x,
            policy_o=policy_o,
            rng=rng,
            extractor=extractor,
        )
        all_rows.extend(game_rows)

    return all_rows


def save_rows_csv(rows: List[Dict], csv_path: str) -> None:
    """
    Save dataset rows to CSV.

    Column ordering strategy:
    -------------------------
    1. Put metadata first
    2. Put extracted feature columns next
    3. Put final labels at the end

    This makes the CSV easier to inspect and analyze manually.
    """
    if not rows:
        raise ValueError("No rows to save.")

    # Stable column order: metadata first, features next, labels last.
    preferred = [
        "game_id",
        "turn_index",
        "current_player",
        "opponent_player",
        "n",
        "k",
        "board_before_move",
        "move_row",
        "move_col",
    ]
    label_cols = ["final_winner", "final_outcome"]

    # Collect all keys that appear anywhere in the dataset
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    # Feature columns are everything that is not metadata or final labels
    feature_cols = sorted(
        [k for k in all_keys if k not in preferred and k not in label_cols]
    )
    columns = preferred + feature_cols + label_cols

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_rows_json(rows: List[Dict], json_path: str) -> None:
    """
    Save dataset rows to JSON.

    Why JSON too?
    -------------
    JSON keeps the row structure flexible and is convenient for later scripts
    or notebook-based analysis.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def summarize_rows(rows: List[Dict]) -> Dict[str, int]:
    """
    Compute a simple game-level summary from the row-level dataset.

    Returns counts of:
    - X wins
    - O wins
    - draws
    - number of games

    Important note:
    ---------------
    The dataset has one row per turn, not one row per game.
    So to summarize winners, we first collapse rows by game_id.
    """
    per_game_winner = {}
    for row in rows:
        gid = row["game_id"]
        if gid not in per_game_winner:
            per_game_winner[gid] = row["final_winner"]

    summary = {"X_wins": 0, "O_wins": 0, "draws": 0, "num_games": len(per_game_winner)}
    for w in per_game_winner.values():
        if w == "X":
            summary["X_wins"] += 1
        elif w == "O":
            summary["O_wins"] += 1
        else:
            summary["draws"] += 1
    return summary


def main() -> None:
    """
    Command-line entry point for Milestone 5 data collection.

    Example usage:
        python data_collection.py --num-games 200 --n 4 --k 3 \
            --matchup tactical_vs_random --out-dir report/results

    Output:
    -------
    - CSV dataset
    - JSON dataset
    - short summary printed to terminal
    """
    parser = argparse.ArgumentParser(description="Milestone 5 data collection")
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--matchup",
        type=str,
        default="tactical_vs_random",
        choices=["random_vs_random", "tactical_vs_random"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="report/results")
    parser.add_argument("--base-name", type=str, default="m5_dataset")

    args = parser.parse_args()

    rows = collect_dataset(
        num_games=args.num_games,
        n=args.n,
        k=args.k,
        matchup=args.matchup,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.base_name}.csv")
    json_path = os.path.join(args.out_dir, f"{args.base_name}.json")

    save_rows_csv(rows, csv_path)
    save_rows_json(rows, json_path)

    summary = summarize_rows(rows)
    print(f"Saved rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print("Game summary:", summary)


if __name__ == "__main__":
    main()
