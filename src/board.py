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
        self._grid = [[None for _ in range(n)] for _ in range(n)]
        self._empty: set = {(r, c) for r in range(n) for c in range(n)}
        self._occupied: set = set()

    # ------------------------------------------------------------------
    # grid property — allows tests to assign board.grid = [...] directly
    # while keeping _empty/_occupied in sync via _sync_sets().
    # ------------------------------------------------------------------

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, new_grid):
        self._grid = new_grid
        self._sync_sets()

    def _sync_sets(self) -> None:
        """Rebuild _empty and _occupied from the current _grid. O(n²)."""
        self._empty = set()
        self._occupied = set()
        for r in range(self.n):
            for c in range(self.n):
                if self._grid[r][c] is None:
                    self._empty.add((r, c))
                else:
                    self._occupied.add((r, c))

    # ------------------------------------------------------------------
    # Move primitives
    # ------------------------------------------------------------------

    def make_move(self, row: int, col: int, player: str) -> bool:
        """
        Place a move for the player at (row, col).
        :return: True if move was successful, False if invalid.
        """
        if self.is_valid_move(row, col):
            self._grid[row][col] = player
            self._empty.discard((row, col))
            self._occupied.add((row, col))
            return True
        return False

    def undo_move(self, row: int, col: int) -> None:
        """Undo the move at (row, col)."""
        self._grid[row][col] = None
        self._occupied.discard((row, col))
        self._empty.add((row, col))

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if the move at (row, col) is valid (empty and within bounds)."""
        return 0 <= row < self.n and 0 <= col < self.n and self._grid[row][col] is None

    # ------------------------------------------------------------------
    # Board state queries
    # ------------------------------------------------------------------

    def is_full(self) -> bool:
        """Check if the board is full. O(1)."""
        return len(self._empty) == 0

    def get_available_moves(self) -> List[Tuple[int, int]]:
        """Return a list of available (row, col) moves. O(available)."""
        return list(self._empty)

    def check_line_at(self, r: int, c: int, player: str) -> bool:
        """
        Return True if `player` has k-in-a-row passing through (r, c). O(4k).

        Only valid to call immediately after player placed a piece at (r, c) —
        a new piece can only complete a line through itself, so checking all four
        axes through (r, c) is both necessary and sufficient.
        """
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in (1, -1):
                for step in range(1, self.k):
                    nr = r + dr * sign * step
                    nc = c + dc * sign * step
                    if (0 <= nr < self.n and 0 <= nc < self.n
                            and self._grid[nr][nc] == player):
                        count += 1
                    else:
                        break
            if count >= self.k:
                return True
        return False

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the board to empty."""
        self._grid = [[None for _ in range(self.n)] for _ in range(self.n)]
        self._empty = {(r, c) for r in range(self.n) for c in range(self.n)}
        self._occupied = set()

    def clone(self) -> "Board":
        """Return a deep copy of the board."""
        new_board = Board(self.n, self.k)
        new_board._grid = copy.deepcopy(self._grid)
        new_board._empty = set(self._empty)
        new_board._occupied = set(self._occupied)
        return new_board

    def display(self) -> None:
        """Print the board in a readable CLI form with row and column headers."""
        col_headers = "   " + "   ".join(str(c) for c in range(self.n))
        print(col_headers)
        print("  " + "-" * (self.n * 4 - 1))
        for r, row in enumerate(self._grid):
            row_str = f"{r} " + " | ".join(
                [cell if cell is not None else " " for cell in row]
            )
            print(row_str)
            print("  " + "-" * (self.n * 4 - 1))
