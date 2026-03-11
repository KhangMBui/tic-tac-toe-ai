import unittest
from src.game import Game
from src.ai import MinimaxAI


class TestMinimaxAI(unittest.TestCase):
    def setUp(self):
        self.ai = MinimaxAI()
        self.ai_player = "O"
        self.human_player = "X"

    def test_immediate_win(self):
        game = Game(3, 3, "X", "O")
        # AI can win immediately
        game.board.grid = [["O", "O", None], ["X", "X", None], [None, None, None]]
        move = self.ai.get_best_move(game, self.ai_player, self.human_player)
        self.assertEqual(move, (0, 2))  # AI should win

    def test_immediate_block(self):
        game = Game(3, 3, "X", "O")
        # Human can win, AI must block
        game.board.grid = [["X", "X", None], ["O", None, None], [None, None, None]]
        move = self.ai.get_best_move(game, self.ai_player, self.human_player)
        self.assertEqual(move, (0, 2))  # AI should block

    def test_draw(self):
        game = Game(3, 3, "X", "O")
        # Draw scenario
        game.board.grid = [["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]]
        move = self.ai.get_best_move(game, self.ai_player, self.human_player)
        self.assertIsNone(move)  # No moves left

    def test_best_move_center(self):
        game = Game(3, 3, "X", "O")
        # Empty board, AI should prefer center
        move = self.ai.get_best_move(game, self.ai_player, self.human_player)
        self.assertEqual(move, (1, 1))

    def test_nodes_explored(self):
        game = Game(3, 3, "X", "O")
        self.ai.get_best_move(game, self.ai_player, self.human_player)
        self.assertTrue(self.ai.nodes_explored > 0)


if __name__ == "__main__":
    unittest.main()
