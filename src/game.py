# Responsible for rules.

# Handles:
# current player
# checking winner
# checking draw
# switching turns
# deciding whether game is over

"""
Game class for Tic-Tac-Toe AI project.
Handles turn management, win/draw detection, and terminal state checking.
"""

from typing import Optional, Tuple, List
from .board import Board


class Game:
    def __init__(self, n: int, k: int, player1: str = "X", player2: str = "O"):
        """
        Initialize a new game.
        :param n: Board size (n x n)
        :param k: Win length (number in a row to win)
        :param player1: Symbol for player 1
        :param player2: Symbol for player 2
        """
        self.board = Board(n, k)
        self.n = n
        self.k = k
        self.players = [player1, player2]
        self.current_player_idx = 0  # 0 for player1, 1 for player2

    @property
    def current_player(self) -> str:
        return self.players[self.current_player_idx]

    def switch_turn(self) -> None:
        """
        Switch to the other player's turn.
        """
        self.current_player_idx = 1 - self.current_player_idx

    def make_move(self, row: int, col: int) -> bool:
        """
        Place a move for the current player.
        :return: True if move was successful, False otherwise.
        """
        return self.board.make_move(row, col, self.current_player)

    def undo_move(self, row: int, col: int) -> None:
        """
        Undo the move at (row, col).
        """
        self.board.undo_move(row, col)

    def check_winner(self) -> Optional[str]:
        """
        Check if there is a winner.
        :return: Player symbol if winner, None otherwise.
        """
        for player in self.players:
            if self._has_winning_line(player):
                return player
        return None

    def is_draw(self) -> bool:
        """
        Check if the game is a draw (board full, no winner).
        """
        return self.board.is_full() and self.check_winner() is None

    def is_terminal(self) -> bool:
        """
        Check if the game is over (win or draw).
        """
        return self.check_winner() is not None or self.is_draw()

    def reset(self) -> None:
        """
        Reset the game to initial state.
        """
        self.board.reset()
        self.current_player_idx = 0

    def _has_winning_line(self, player: str) -> bool:
        """
        Check all directions for a winning line for the given player.
        """
        directions = [
            (0, 1),  # right
            (1, 0),  # down
            (1, 1),  # down-right
            (1, -1),  # down-left
        ]
        for r in range(self.n):
            for c in range(self.n):
                for dr, dc in directions:
                    if self._check_line(r, c, dr, dc, player):
                        return True
        return False

    def _check_line(self, r: int, c: int, dr: int, dc: int, player: str) -> bool:
        """
        Check if there is a line of length k starting at (r, c) in direction (dr, dc) for player.
        """
        for i in range(self.k):
            nr = r + dr * i
            nc = c + dc * i
            if not (0 <= nr < self.n and 0 <= nc < self.n):
                return False
            if self.board.grid[nr][nc] != player:
                return False
        return True

    def get_status(self) -> str:
        """
        Return a string describing the current game status.
        """
        winner = self.check_winner()
        if winner:
            return f"Player {winner} wins!"
        elif self.is_draw():
            return "Draw!"
        else:
            return f"Player {self.current_player}'s turn."

    def get_available_moves(self) -> List[Tuple[int, int]]:
        """
        Return a list of available moves.
        """
        return self.board.get_available_moves()
