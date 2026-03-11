# Responsible for:
#   Minimax
#   Alpha-Beta pruning
#   best-move selection

"""
Minimax AI for classic 3x3 Tic-Tac-Toe.
No Alpha-Beta pruning.

- get_best_move(): Loops over all available moves, simulates each, and picks the move with the highest Minimax score.

- _minimax(): Recursively explores all possible moves, alternating between maximizing (AI) and minimizing (human).
Terminal states are scored using depth for faster wins/slower losses.

- nodes_explored is incremented for benchmarking.
"""

from typing import Tuple, Optional
from src.game import Game


class MinimaxAI:
    def __init__(self):
        self.nodes_explored = 0

    def get_best_move(self, game: Game, ai_player: str, human_player: str):
        self.nodes_explored = 0
        best_score = float("-inf")
        best_move = None

        # Prefer center, then corners, then edges
        n = game.board.n
        moves = game.get_available_moves()

        def move_priority(move):
            row, col = move
            # Center
            if (row, col) == (n // 2, n // 2):
                return 0
            # Corners
            if (row, col) in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]:
                return 1
            # Edges
            return 2

        moves = sorted(moves, key=move_priority)

        for row, col in moves:
            game.board.make_move(row, col, ai_player)
            score = self._minimax(game, 0, False, ai_player, human_player)
            game.board.undo_move(row, col)
            if score > best_score:
                best_score = score
                best_move = (row, col)
        return best_move

    def _minimax(
        self,
        game: Game,
        depth: int,
        is_maximizing: bool,
        ai_player: str,
        human_player: str,
    ) -> int:
        """
        Recursive Minimax search
        Returns score for the current board state.
        """
        self.nodes_explored += 1
        winner = game.check_winner()
        if winner == ai_player:
            return 10 - depth
        elif winner == human_player:
            return depth - 10
        elif game.is_draw():
            return 0

        if is_maximizing:
            max_eval = float("-inf")
            for row, col in game.get_available_moves():
                game.board.make_move(row, col, ai_player)
                eval = self._minimax(game, depth + 1, False, ai_player, human_player)
                game.board.undo_move(row, col)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float("inf")
            for row, col in game.get_available_moves():
                game.board.make_move(row, col, human_player)
                eval = self._minimax(game, depth + 1, True, ai_player, human_player)
                game.board.undo_move(row, col)
                min_eval = min(min_eval, eval)
            return min_eval
