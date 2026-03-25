import unittest
from src.board import Board
from src.features import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.fx = FeatureExtractor()

    def test_empty_board_symmetry_3x3(self):
        board = Board(3, 3)
        f = self.fx.extract(board, "X", "O")

        self.assertEqual(f["my_marks"], 0)
        self.assertEqual(f["opp_marks"], 0)
        self.assertEqual(f["empty_cells"], 9)
        self.assertEqual(f["my_center_control"], 0)
        self.assertEqual(f["opp_center_control"], 0)
        self.assertEqual(f["blocked_windows"], 0)
        self.assertEqual(f["my_open_lines"], 0)
        self.assertEqual(f["opp_open_lines"], 0)
        self.assertEqual(f["my_immediate_wins"], 0)
        self.assertEqual(f["opp_immediate_wins"], 0)

    def test_center_control_odd_board(self):
        board = Board(3, 3)
        board.make_move(1, 1, "X")
        f = self.fx.extract(board, "X", "O")
        self.assertEqual(f["my_center_control"], 1)
        self.assertEqual(f["opp_center_control"], 0)

    def test_center_control_even_board(self):
        board = Board(4, 3)
        board.make_move(1, 1, "O")
        board.make_move(2, 2, "O")
        f = self.fx.extract(board, "X", "O")
        self.assertEqual(f["my_center_control"], 0)
        self.assertEqual(f["opp_center_control"], 2)

    def test_immediate_win_detection(self):
        board = Board(3, 3)
        board.grid = [
            ["X", "X", None],
            [None, "O", None],
            [None, None, "O"],
        ]
        f = self.fx.extract(board, "X", "O")
        self.assertGreaterEqual(f["my_immediate_wins"], 1)

    def test_opponent_immediate_win_detection(self):
        board = Board(3, 3)
        board.grid = [
            ["O", "O", None],
            [None, "X", None],
            [None, None, "X"],
        ]
        f = self.fx.extract(board, "X", "O")
        self.assertGreaterEqual(f["opp_immediate_wins"], 1)

    def test_blocked_windows(self):
        board = Board(3, 3)
        board.grid = [
            ["X", "O", None],
            [None, None, None],
            [None, None, None],
        ]
        f = self.fx.extract(board, "X", "O")
        self.assertGreaterEqual(f["blocked_windows"], 1)

    def test_generalized_board_4x4_k3(self):
        board = Board(4, 3)
        board.grid = [
            ["X", None, None, None],
            [None, "O", None, None],
            [None, None, "X", None],
            [None, None, None, "O"],
        ]
        f = self.fx.extract(board, "X", "O")

        self.assertEqual(f["n"], 4)
        self.assertEqual(f["k"], 3)
        self.assertEqual(f["my_marks"], 2)
        self.assertEqual(f["opp_marks"], 2)
        self.assertEqual(f["empty_cells"], 12)
        self.assertTrue(f["total_windows"] > 0)

    def test_swap_players_changes_perspective(self):
        board = Board(3, 3)
        board.grid = [
            ["X", "X", None],
            [None, "O", None],
            [None, None, None],
        ]

        f_x = self.fx.extract(board, "X", "O")
        f_o = self.fx.extract(board, "O", "X")

        self.assertEqual(f_x["my_marks"], f_o["opp_marks"])
        self.assertEqual(f_x["opp_marks"], f_o["my_marks"])
        self.assertEqual(f_x["my_immediate_wins"], f_o["opp_immediate_wins"])


if __name__ == "__main__":
    unittest.main()
