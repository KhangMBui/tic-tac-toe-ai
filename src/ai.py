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
        moves = sorted(
            game.get_available_moves(), key=lambda m: self._move_priority(m, n)
        )

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
        Minimax recursively evaluates all possible future game states.

        Parameters:
        - game: current game object
        - depth: how many moves deep we are in the search tree
        - is_maximizing: True if it is the AI's turn, False if it is the human's turn
        - ai_player: the symbol/player controlled by the AI
        - human_player: the symbol/player controlled by the human

        Returns:
        - positive score if the AI is winning
        - negative score if the human is winning
        - 0 if the game ends in a draw
        """
        # Count this recursive call as one explored node.
        # This is useful for measuring how much work the AI searched.
        self.nodes_explored += 1

        # First, check if the current board already has a winner.
        winner = game.check_winner()

        # If the AI wins, return a positive score.
        # We subtract depth so the AI prefers winning sooner.
        # Example:
        # winning in 1 move gives a higher score than winning in 3 moves.
        if winner == ai_player:
            return 10 - depth
        
        # If the human wins, return a negative score.
        # We add depth so the AI prefers delaying a loss.
        # Example:
        # losing in 3 moves is less bad than losing in 1 move.
        elif winner == human_player:
            return depth - 10
        
        # If nobody wins and the board is full, this path is a draw.
        elif game.is_draw():
            return 0


        # Maximizing turn:
        # This is the AI's turn, so we want to find the move
        # that gives the highest possible score.
        if is_maximizing:
            max_eval = float("-inf")

            # Try every legal move the AI can make.
            for row, col in game.get_available_moves():
                
                # Make a temporary move for the AI.
                game.board.make_move(row, col, ai_player)

                # Recursively evaluate what happens after this move.
                # After AI moves, it becomes the human's turn,
                # so is_maximizing becomes False.
                eval = self._minimax(game, depth + 1, False, ai_player, human_player)

                # Undo the temporary move so we can try the next possible move.
                game.board.undo_move(row, col)

                # Keep the best/highest score found so far.
                max_eval = max(max_eval, eval)
            return max_eval
        
        # Minimizing turn:
        # This is the human's turn, so we assume the human will choose
        # the move that is worst for the AI.
        else:
            min_eval = float("inf")

            # Try every legal move the human can make.
            for row, col in game.get_available_moves():
                # Make a temporary move for the human.
                game.board.make_move(row, col, human_player)

                # Recursively evaluate what happens after this move.
                # After human moves, it becomes the AI's turn,
                # so is_maximizing becomes True.
                eval = self._minimax(game, depth + 1, True, ai_player, human_player)

                # Undo the temporary move so we can try the next possible move.
                game.board.undo_move(row, col)

                # Keep the lowest score found so far,
                # because the human is trying to minimize the AI's outcome.
                min_eval = min(min_eval, eval)
            # Return the worst score the human can force against the AI.
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
        moves = sorted(
            game.get_available_moves(), key=lambda m: self._move_priority(m, n)
        )

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
        
        Returns:
        - positive score if the AI is winning
        - negative score if the human is winning
        - 0 if the game ends in a draw
        """

        # Count this recursive call as one explored node.
        # This is useful for measuring how much work the AI searched.
        self.nodes_explored_ab += 1

        # First, check if the current board already has a winner.
        winner = game.check_winner()

        # If the AI wins, return a positive score.
        # Subtract depth so the AI prefers winning sooner.
        if winner == ai_player:
            return 10 - depth
        
        # If the human wins, return a negative score.
        # Add depth so the AI prefers losing later instead of losing immediately.
        elif winner == human_player:
            return depth - 10

        # If nobody wins and there are no moves left, this path is a draw.
        elif game.is_draw():
            return 0
        
        # Maximizing turn:
        # This is the AI's turn, so the AI wants the highest possible score.
        if is_maximizing:
            max_eval = float("-inf")

            # Try every legal AI move.
            for row, col in game.get_available_moves():
                game.board.make_move(row, col, ai_player)

                # Recursively evaluate the board after this move.
                # After AI moves, it becomes the human's turn.
                eval = self._minimax_ab(
                    game, depth + 1, alpha, beta, False, ai_player, human_player
                )

                # Undo the temporary move so we can test another move.
                game.board.undo_move(row, col)

                # Keep the best score found for the AI.
                max_eval = max(max_eval, eval)

                # Update alpha.
                # Alpha means: the best score the AI can guarantee so far
                alpha = max(alpha, eval)

                # Pruning condition:
                # If beta <= alpha, the minimizing player already has a better option elsewhere.
                # So there is no reason to keep checking this branch
                if beta <= alpha:
                    break # beta cut off
            return max_eval
        
        # Minimizing turn:
        # This is the human's turn, so we assume the human chooses the move
        # that gives the lowest score for the AI.
        else:
            min_eval = float("inf")

            # Try every legal human move.
            for row, col in game.get_available_moves():
                 # Make a temporary move for the human.
                game.board.make_move(row, col, human_player)

                # Recursively evaluate the board after this move.
                # After human moves, it becomes the AI's turn.
                eval = self._minimax_ab(
                    game, depth + 1, alpha, beta, True, ai_player, human_player
                )

                # Undo the temporary move so we can test another move.
                game.board.undo_move(row, col)

                # Keep the worst/lowest score from the AI's perspective
                min_eval = min(min_eval, eval)

                # Update beta.
                # Beta means: the best score the human can guarantee so far,
                # which is the lowest score for the AI.
                beta = min(beta, eval)

                # Pruning condition:
                # If beta <= alpha, the maximizing player already has a better option elsewhere.
                # So there is no reason to keep checking this branch.
                if beta <= alpha:
                    break # Alpha cut-off
                
            # Return the lowest score the human can force against the AI.
            return min_eval

    def get_best_move_heuristic(
        self,
        game: Game,
        ai_player: str,
        human_player: str,
        max_depth: int = 4,
    ):
        """
        Return the best move using depth-limited Alpha-Beta with heuristic cutoff.

        When the search reaches max_depth without reaching a terminal state, the
        board is scored with HeuristicEvaluator.evaluate() instead of searching
        further. This bounds the search tree to a fixed number of plies and makes
        the AI practical on larger boards where exhaustive search is infeasible.

        Cutoff policy:
          - On 3×3 boards the full tree is small; max_depth is effectively ignored
            because terminal states are always reached before the cutoff.
          - On 4×4+ boards max_depth caps the search. Recommended starting values:
            depth 3–4 for 4×4, depth 2–3 for 5×5 and above.
        """
        from src.heuristics import HeuristicEvaluator

        self.nodes_explored_h = 0
        evaluator = HeuristicEvaluator()
        best_score = float("-inf")
        best_move = None

        n = game.board.n
        candidates = (
            self._get_candidate_moves(game.board)
            if n > 5
            else game.get_available_moves()
        )
        moves = sorted(candidates, key=lambda m: self._move_priority(m, n))

        # Take an immediate win before doing full search.
        for row, col in moves:
            game.board.make_move(row, col, ai_player)
            won = game.board.check_line_at(row, col, ai_player)
            game.board.undo_move(row, col)
            if won:
                self.nodes_explored_h = len(moves)
                return (row, col)

        # Block an immediate human win before doing full search.
        for row, col in moves:
            game.board.make_move(row, col, human_player)
            won = game.board.check_line_at(row, col, human_player)
            game.board.undo_move(row, col)
            if won:
                self.nodes_explored_h = len(moves)
                return (row, col)

        # Block open-(k-1) forced threats: k-1 consecutive human marks with
        # BOTH adjacent ends empty. Playing at either end wins outright next
        # turn, so no alpha-beta search can fix it — block immediately.
        if game.board.k >= 3:
            forced = self._find_open_forced_threats(game.board, human_player)
            if forced:
                moves_set = set(moves)
                end1, end2 = forced[0]
                for cell in (end1, end2):
                    if cell in moves_set:
                        self.nodes_explored_h = len(moves)
                        return cell

        for row, col in moves:
            game.board.make_move(row, col, ai_player)
            score = self._minimax_ab_h(
                game, 1, max_depth,
                float("-inf"), float("inf"),
                False,
                ai_player, human_player,
                evaluator,
                (row, col), ai_player,
            )
            game.board.undo_move(row, col)
            if score > best_score:
                best_score = score
                best_move = (row, col)
        return best_move

    def _minimax_ab_h(
        self,
        game: Game,
        depth: int,
        max_depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
        ai_player: str,
        human_player: str,
        evaluator,
        last_move: tuple = None,
        last_player: str = None,
    ) -> float:
        """
        Alpha-Beta Minimax with a heuristic evaluation function.

        This is an optimized version of Minimax for larger boards.

        Main ideas:
        1. Use alpha-beta pruning to skip useless branches.
        2. Stop searching once depth reaches max_depth.
        3. Use a heuristic evaluator to estimate the board score at the cutoff.
        4. Use last_move / last_player to check wins faster.

        Parameters:
        - game: current game object
        - depth: current depth in the search tree
        - max_depth: maximum depth we are willing to search
        - alpha: best score the AI/maximizing player can guarantee so far
        - beta: best score the human/minimizing player can guarantee so far
        - is_maximizing: True if it is the AI's turn, False if it is the human's turn
        - ai_player: symbol/player controlled by the AI
        - human_player: symbol/player controlled by the human
        - evaluator: heuristic object/function used to score non-terminal boards
        - last_move: the most recent move played, as (row, col)
        - last_player: the player who made the most recent move

        Returns:
        - a large positive score if the AI wins
        - a large negative score if the human wins
        - 0 for a draw
        - otherwise, a heuristic estimate of the board position
        """

        # Count this recursive call as one explored node.
        # This is useful for measuring how much work the AI searched.
        self.nodes_explored_h += 1

        # ------------------------------------------------------------
        # 1. Fast terminal check
        # ------------------------------------------------------------
        # Instead of scanning the entire board to check for a winner,
        # we only check lines passing through the last move.
        #
        # This works because the only way a new win can happen is if
        # the most recent piece completed a line.
        #
        # For example, if the last move was at (row, col), then only
        # rows/columns/diagonals touching that cell could have changed.
        if last_move is not None:
            
            # If the last player was the AI, this is an AI win.
            # Return a very large positive score.
            #
            # Subtract depth so faster wins are preferred.
            # On the other hand, 
            # If the last player was the human, this is a human win.
            # Return a very large negative score.
            #
            # Add depth so slower losses are preferred.
            if game.board.check_line_at(last_move[0], last_move[1], last_player):
                return (10000 - depth) if last_player == ai_player else (depth - 10000)
            
            # If the board is full and there is no winner, this path is a draw.
            if game.board.is_full():    # O(1) via _empty set
                return 0

        # ------------------------------------------------------------
        # 2. Depth cutoff
        # ------------------------------------------------------------
        # If we reached the maximum search depth, stop searching deeper.
        #
        # Instead of playing the game out completely, estimate how good
        # the current board is using the heuristic evaluator.
        #
        # Terminal states were checked before this, so real wins/losses/draws
        # still take priority over heuristic guesses.
        if depth >= max_depth:
            return evaluator.evaluate(game.board, ai_player, human_player)

        # ------------------------------------------------------------
        # 3. Move generation
        # ------------------------------------------------------------
        # For small boards, we can just try every available move.
        #
        # For large boards, trying every empty cell is too expensive.
        # So we use candidate moves, usually moves near existing pieces,
        # because those are more likely to matter.
        moves = (
            self._get_candidate_moves(game.board)
            if game.board.n > 5
            else game.get_available_moves()
        )

        # ------------------------------------------------------------
        # 4. Maximizing turn: AI's turn
        # ------------------------------------------------------------
        # The AI wants to choose the move with the highest score.
        if is_maximizing:
            max_eval = float("-inf")

            # Try each possible AI move.
            for row, col in moves:
                
                # Temporarily place the AI's move.
                game.board.make_move(row, col, ai_player)

                # Recursively evaluate the position after this move.
                # Since AI just moved, the next turn is the human's turn,
                # so is_maximizing becomes False.
                #
                # We pass (row, col) and ai_player as the last move info,
                # so the next recursive call can do a fast win check.
                val = self._minimax_ab_h(
                    game, depth + 1, max_depth, alpha, beta,
                    False, ai_player, human_player, evaluator,
                    (row, col), ai_player,
                )

                # Undo the temporary move so we can test another move.
                game.board.undo_move(row, col)

                # Keep the highest score found so far.
                max_eval = max(max_eval, val)

                # Update alpha.
                # Alpha is the best score the AI can guarantee so far.
                alpha = max(alpha, val)

                # Alpha-beta pruning:
                # If beta <= alpha, the human already has a better option
                # in a previous branch, so continuing here will not affect
                # the final decision.
                if beta <= alpha:
                    break # beta cut-off
                
            # Return the best score the AI can force from this position.
            return max_eval
        
        # ------------------------------------------------------------
        # 5. Minimizing turn: human's turn
        # ------------------------------------------------------------
        # The human is assumed to play optimally against the AI.
        # So the human chooses the move with the lowest score for the AI.
        else:
            min_eval = float("inf")
             # Try each possible human move.
            for row, col in moves:
                game.board.make_move(row, col, human_player)

                # Recursively evaluate the position after this move.
                # Since human just moved, the next turn is the AI's turn,
                # so is_maximizing becomes True.
                #
                # We pass (row, col) and human_player as the last move info,
                # so the next recursive call can do a fast win check.
                val = self._minimax_ab_h(
                    game, depth + 1, max_depth, alpha, beta,
                    True, ai_player, human_player, evaluator,
                    (row, col), human_player,
                )

                # Undo the temporary move so we can test another move.
                game.board.undo_move(row, col)

                # Keep the lowest score found so far.
                min_eval = min(min_eval, val)

                # Update beta.
                # Beta is the best score the human can guarantee so far,
                # meaning the lowest score the AI may be forced into.
                beta = min(beta, val)

                # Alpha-beta pruning:
                # If beta <= alpha, the AI already has a better option
                # in a previous branch, so this branch can be skipped.
                if beta <= alpha:
                    break # beta cut-off
                
            # Return the lowest score the human can force against the AI.
            return min_eval

    @staticmethod
    def _find_open_forced_threats(board, player: str):
        """
        Return (end1, end2) blocking pairs for every open-(k-1) threat by `player`.

        An open-(k-1) is exactly k-1 consecutive marks in one direction with BOTH
        adjacent cells empty and in-bounds. Either end cell wins immediately, so
        the opponent must play at one of them or lose the following turn.

        Only the start of each run is examined (skip cells with a same-player
        predecessor) to avoid counting the same sequence multiple times.
        """
        k, n = board.k, board.n
        threats = []
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            for r in range(n):
                for c in range(n):
                    if board._grid[r][c] != player:
                        continue
                    pr, pc = r - dr, c - dc
                    if 0 <= pr < n and 0 <= pc < n and board._grid[pr][pc] == player:
                        continue  # not the start of this run
                    length, nr, nc = 0, r, c
                    while 0 <= nr < n and 0 <= nc < n and board._grid[nr][nc] == player:
                        length += 1
                        nr += dr
                        nc += dc
                    if length != k - 1:
                        continue
                    e1r, e1c = r - dr, c - dc
                    e2r, e2c = nr, nc
                    e1_open = (0 <= e1r < n and 0 <= e1c < n
                               and board._grid[e1r][e1c] is None)
                    e2_open = (0 <= e2r < n and 0 <= e2c < n
                               and board._grid[e2r][e2c] is None)
                    if e1_open and e2_open:
                        threats.append(((e1r, e1c), (e2r, e2c)))
        return threats

    @staticmethod
    def _get_candidate_moves(board, radius: int = 2):
        """
        Return a focused set of candidate moves for large-board heuristic search.

        Only considers empty cells within `radius` squares (Chebyshev distance)
        of any occupied cell. On an empty board, returns just the center cell —
        the universally strongest opening for any k-in-a-row game.

        Why this is necessary for scale:
          get_available_moves() returns up to n² candidates. On a 50×50 board
          that is a branching factor of ~2,500 — depth=2 alone visits 6.25M
          nodes. Restricting to radius=2 around existing pieces caps the
          effective branching factor at ~20–50 in typical mid-game positions,
          making depth=4 or 5 feasible on 50×50+ boards.

        Cells far from any existing piece have no immediate strategic relevance
        in k-in-a-row games; this is the standard approach in Gomoku AI.
        Only applied when board.n > 5; smaller boards use all available moves.
        """
        n = board.n

        if not board._occupied:
            mid = n // 2
            return [(mid, mid)]

        candidates = set()
        for pr, pc in board._occupied:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < n and 0 <= nc < n and board._grid[nr][nc] is None:
                        candidates.add((nr, nc))

        return list(candidates)

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
