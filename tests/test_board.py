import unittest
from src.board import Board


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board(3, 3)

    def test_empty_board(self):
        for row in self.board.grid:
            for cell in row:
                self.assertIsNone(cell)

    def test_valid_move(self):
        self.assertTrue(self.board.make_move(0, 0, "X"))
        self.assertEqual(self.board.grid[0][0], "X")

    def test_invalid_move(self):
        self.board.make_move(0, 0, "X")
        self.assertFalse(self.board.make_move(0, 0, "O"))  # Already occupied
        self.assertFalse(self.board.make_move(-1, 0, "X"))  # Out of bounds
        self.assertFalse(self.board.make_move(3, 3, "X"))  # Out of bounds

    def test_undo_move(self):
        self.board.make_move(1, 1, "O")
        self.board.undo_move(1, 1)
        self.assertIsNone(self.board.grid[1][1])

    def test_get_available_moves(self):
        self.board.make_move(0, 0, "X")
        moves = self.board.get_available_moves()
        self.assertNotIn((0, 0), moves)
        self.assertEqual(len(moves), 8)

    def test_is_full(self):
        for r in range(3):
            for c in range(3):
                self.board.make_move(r, c, "X")
        self.assertTrue(self.board.is_full())

    def test_reset(self):
        self.board.make_move(0, 0, "X")
        self.board.reset()
        for row in self.board.grid:
            for cell in row:
                self.assertIsNone(cell)

    def test_clone(self):
        self.board.make_move(0, 0, "X")
        clone = self.board.clone()
        self.assertEqual(clone.grid, self.board.grid)
        clone.make_move(1, 1, "O")
        self.assertNotEqual(clone.grid, self.board.grid)

    def test_generalized_board(self):
        board4 = Board(4, 3)
        self.assertEqual(board4.n, 4)
        self.assertEqual(board4.k, 3)
        self.assertEqual(len(board4.grid), 4)
        self.assertEqual(len(board4.grid[0]), 4)


if __name__ == "__main__":
    unittest.main()
