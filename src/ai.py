# Responsible for:
#   Minimax
#   Alpha-Beta pruning
#   best-move selection

"""
Minimax AI for Tic-Tac-Toe (supports any n x n board with k-in-a-row win).

Plain Minimax (get_best_move / _minimax):
- Exhaustively explores all possible game states.
- Terminal states scored as: AI win = 10 - depth, Human win = depth - 10, Draw = 0.
- nodes_explored tracks total nodes visited for benchmarking.

Alpha-Beta Pruning (get_best_move_ab / _minimax_ab):
- Same search and scoring logic as plain Minimax.
- Prunes branches that cannot affect the final decision:
    alpha = best score the maximizer is guaranteed so far
    beta  = best score the minimizer is guaranteed so far
    A branch is pruned when beta <= alpha (the opponent would never allow this path).
- nodes_explored_ab tracks nodes visited; will be <= nodes_explored on the same board.
"""

from typing import Tuple, Optional
from src.game import Game


class MinimaxAI:
    def __init__(self):
        self.nodes_explored = 0
        self.nodes_explored_ab = 0

    def get_best_move(self, game: Game, ai_player: str, human_player: str):
        self.nodes_explored = 0
        best_score = float("-inf")
        best_move = None

        # Prefer center, then corners, then edges
        n = game.board.n
        moves = sorted(game.get_available_moves(), key=lambda m: self._move_priority(m, n))

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
        Scoring: AI win = 10 - depth, Human win = depth - 10, Draw = 0.
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

    def get_best_move_ab(self, game: Game, ai_player: str, human_player: str):
        """
        Return the best move for ai_player using Minimax with Alpha-Beta pruning.
        Produces the same move quality as get_best_move but visits fewer nodes.
        """
        self.nodes_explored_ab = 0
        best_score = float("-inf")
        best_move = None

        n = game.board.n
        moves = sorted(game.get_available_moves(), key=lambda m: self._move_priority(m, n))

        for row, col in moves:
            game.board.make_move(row, col, ai_player)
            score = self._minimax_ab(
                game, 0, float("-inf"), float("inf"), False, ai_player, human_player
            )
            game.board.undo_move(row, col)
            if score > best_score:
                best_score = score
                best_move = (row, col)
        return best_move
    
    def _minimax_ab(
        self,
        game: Game,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
        ai_player: str,
        human_player: str,
    ) -> int:
        """
        Recursive Minimax search with Alpha-Beta pruning.

        alpha: the best (highest) score the maximizer can already guarantee.
        beta:  the best (lowest)  score the minimizer can already guarantee.

        Pruning rule: when beta <= alpha, the current branch cannot influence
        the final decision — the opponent would have chosen a different path
        earlier — so we stop exploring it.

        Scoring is identical to plain Minimax:
            AI win    = 10 - depth
            Human win = depth - 10
            Draw      = 0
        """
        self.nodes_explored_ab += 1
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
                eval = self._minimax_ab(game, depth + 1, alpha, beta, False, ai_player, human_player)
                game.board.undo_move(row, col)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    # Beta cut-off: minimizer already has a better option elsewhere,
                    # so it will never choose this branch — prune remaining siblings.
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for row, col in game.get_available_moves():
                game.board.make_move(row, col, human_player)
                eval = self._minimax_ab(game, depth + 1, alpha, beta, True, ai_player, human_player)
                game.board.undo_move(row, col)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    # Alpha cut-off: maximizer already has a better option elsewhere,
                    # so it will never choose this branch — prune remaining siblings.
                    break
            return min_eval
        
    @staticmethod
    def _move_priority(move: tuple, n: int) -> int:
        row, col = move
        # Center
        if (row, col) == (n // 2, n // 2):
            return 0
        # Corners
        if (row, col) in [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]:
            return 1
        # Edges
        return 2