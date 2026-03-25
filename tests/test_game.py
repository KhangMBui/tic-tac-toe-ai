import unittest
from src.game import Game


class TestGame(unittest.TestCase):
    def setUp(self):
        self.game = Game(3, 3)

    def test_row_win(self):
        self.game.board.make_move(0, 0, "X")
        self.game.board.make_move(0, 1, "X")
        self.game.board.make_move(0, 2, "X")
        self.assertEqual(self.game.check_winner(), "X")

    def test_column_win(self):
        self.game.board.make_move(0, 1, "O")
        self.game.board.make_move(1, 1, "O")
        self.game.board.make_move(2, 1, "O")
        self.assertEqual(self.game.check_winner(), "O")

    def test_diagonal_win(self):
        self.game.board.make_move(0, 0, "X")
        self.game.board.make_move(1, 1, "X")
        self.game.board.make_move(2, 2, "X")
        self.assertEqual(self.game.check_winner(), "X")

    def test_anti_diagonal_win(self):
        self.game.board.make_move(0, 2, "O")
        self.game.board.make_move(1, 1, "O")
        self.game.board.make_move(2, 0, "O")
        self.assertEqual(self.game.check_winner(), "O")

    def test_draw(self):
        moves = [
            (0, 0, "X"),
            (0, 1, "O"),
            (0, 2, "X"),
            (1, 0, "O"),
            (1, 1, "X"),
            (1, 2, "O"),
            (2, 0, "O"),
            (2, 1, "X"),
            (2, 2, "O"),
        ]
        for r, c, p in moves:
            self.game.board.make_move(r, c, p)
        self.assertTrue(self.game.is_draw())
        self.assertIsNone(self.game.check_winner())
        self.assertTrue(self.game.is_terminal())

    def test_non_terminal(self):
        self.game.board.make_move(0, 0, "X")
        self.assertFalse(self.game.is_terminal())
        self.assertIsNone(self.game.check_winner())
        self.assertFalse(self.game.is_draw())

    def test_switch_turn(self):
        self.assertEqual(self.game.current_player, "X")
        self.game.switch_turn()
        self.assertEqual(self.game.current_player, "O")
        self.game.switch_turn()
        self.assertEqual(self.game.current_player, "X")

    def test_configurable_k(self):
        game4 = Game(4, 4)
        for i in range(4):
            game4.board.make_move(i, i, "X")
        self.assertEqual(game4.check_winner(), "X")

    def test_status_messages(self):
        self.assertIn("turn", self.game.get_status())
        self.game.board.make_move(0, 0, "X")
        self.game.board.make_move(0, 1, "X")
        self.game.board.make_move(0, 2, "X")
        self.assertIn("wins", self.game.get_status())


if __name__ == "__main__":
    unittest.main()
