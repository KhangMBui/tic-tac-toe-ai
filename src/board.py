# Responsible only for board state.
# Handles:
# creating an n x n board
# checking whether a cell is empty
# placing a mark
# returning available moves
# printing/displaying board for CLI debugging

"""
Board class for Tic-Tac-Toe AI project.
Handles board state, move placement, undo, and utility functions.
"""
from typing import List, Optional, Tuple
import copy


class Board:
    def __init__(self, n: int, k: int):
        """
        Initialize an empty board.
        :param n: Board size (n x n)
        :param k: Win length (number in a row to win)
        """
        self.n = n
        self.k = k
        self.grid = [[None for _ in range(n)] for _ in range(n)]

    def make_move(self, row: int, col: int, player: str) -> bool:
        """
        Place a move for the player at (row, col).
        :return: True if move was successful, False if invalid.
        """
        if self.is_valid_move(row, col):
            self.grid[row][col] = player
            return True
        return False

    def undo_move(self, row: int, col: int) -> None:
        """
        Undo the move at (row, col)
        """
        self.grid[row][col] = None

    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if the move at (row, col) is valid (empty and within bounds).
        """
        return 0 <= row < self.n and 0 <= col < self.n and self.grid[row][col] is None

    def is_full(self) -> bool:
        """
        Check if the board is full.
        """
        return all(cell is not None for row in self.grid for cell in row)

    def reset(self) -> None:
        """
        Reset the board to empty.
        """
        self.grid = [[None for _ in range(self.n)] for _ in range(self.n)]

    def clone(self) -> "Board":
        """
        Return a deep copy of the board.
        """
        new_board = Board(self.n, self.k)
        new_board.grid = copy.deepcopy(self.grid)
        return new_board

    def display(self) -> None:
        """
        Print the board in a readable CLI form.
        """
        for row in self.grid:
            print(" | ".join([cell if cell is not None else " " for cell in row]))
            print("-" * (self.n * 4 - 1))

    def get_available_moves(self) -> List[Tuple[int, int]]:
        """
        Return a list of available (row, col) moves.
        """
        return [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if self.grid[r][c] is None
        ]
