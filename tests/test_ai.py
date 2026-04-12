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

class TestAlphaBetaAI(unittest.TestCase):
    def setUp(self):
        self.ai = MinimaxAI()
        self.ai_player = "O"
        self.human_player = "X"

    def test_ab_immediate_win(self):
        """Alpha-Beta should find the winning move."""
        game = Game(3, 3, "X", "O")
        game.board.grid = [["O", "O", None], ["X", "X", None], [None, None, None]]
        move = self.ai.get_best_move_ab(game, self.ai_player, self.human_player)
        self.assertEqual(move, (0, 2))

    def test_ab_immediate_block(self):
        """Alpha-Beta should block the human from winning."""
        game = Game(3, 3, "X", "O")
        game.board.grid = [["X", "X", None], ["O", None, None], [None, None, None]]
        move = self.ai.get_best_move_ab(game, self.ai_player, self.human_player)
        self.assertEqual(move, (0, 2))

    def test_ab_draw(self):
        """Alpha-Beta returns None when the board is full."""
        game = Game(3, 3, "X", "O")
        game.board.grid = [["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]]
        move = self.ai.get_best_move_ab(game, self.ai_player, self.human_player)
        self.assertIsNone(move)

    def test_ab_best_move_center(self):
        """Alpha-Beta should prefer center on an empty board."""
        game = Game(3, 3, "X", "O")
        move = self.ai.get_best_move_ab(game, self.ai_player, self.human_player)
        self.assertEqual(move, (1, 1))

    def test_ab_move_matches_minimax(self):
        """Alpha-Beta and plain Minimax must choose the same best move."""
        boards = [
            # Empty board
            [[None, None, None], [None, None, None], [None, None, None]],
            # AI can win immediately
            [["O", "O", None], ["X", "X", None], [None, None, None]],
            # Human threatens, AI must block
            [["X", "X", None], ["O", None, None], [None, None, None]],
            # Mid-game board
            [["X", None, None], [None, "O", None], [None, None, "X"]],
        ]
        for grid in boards:
            game_mm = Game(3, 3, "X", "O")
            game_ab = Game(3, 3, "X", "O")
            game_mm.board.grid = [row[:] for row in grid]
            game_ab.board.grid = [row[:] for row in grid]
            move_mm = self.ai.get_best_move(game_mm, self.ai_player, self.human_player)
            move_ab = self.ai.get_best_move_ab(game_ab, self.ai_player, self.human_player)
            self.assertEqual(move_mm, move_ab, f"Mismatch on board: {grid}")

    def test_ab_nodes_fewer_or_equal(self):
        """Alpha-Beta must explore no more nodes than plain Minimax."""
        boards = [
            # Empty board — largest search space
            [[None, None, None], [None, None, None], [None, None, None]],
            # Mid-game
            [["X", None, None], [None, "O", None], [None, None, "X"]],
        ]
        for grid in boards:
            game_mm = Game(3, 3, "X", "O")
            game_ab = Game(3, 3, "X", "O")
            game_mm.board.grid = [row[:] for row in grid]
            game_ab.board.grid = [row[:] for row in grid]
            self.ai.get_best_move(game_mm, self.ai_player, self.human_player)
            nodes_mm = self.ai.nodes_explored
            self.ai.get_best_move_ab(game_ab, self.ai_player, self.human_player)
            nodes_ab = self.ai.nodes_explored_ab
            self.assertLessEqual(nodes_ab, nodes_mm,
                f"Alpha-Beta ({nodes_ab}) explored more nodes than Minimax ({nodes_mm}) on board: {grid}")

    def test_ab_nodes_explored(self):
        """nodes_explored_ab must be positive after a search."""
        game = Game(3, 3, "X", "O")
        self.ai.get_best_move_ab(game, self.ai_player, self.human_player)
        self.assertTrue(self.ai.nodes_explored_ab > 0)

class TestHeuristicAI(unittest.TestCase):
    """
    Tests for get_best_move_heuristic — depth-limited Alpha-Beta with heuristic cutoff.

    Coverage:
      - Correctness: finds wins and blocks on 3×3 and 4×4 boards
      - Cutoff behavior: shallower depth visits fewer nodes; deeper depth visits more
      - Terminal bypass: actual wins/losses are detected before the depth cutoff fires
      - Efficiency: depth-limited always explores fewer nodes than exhaustive AB
    """

    def setUp(self):
        self.ai = MinimaxAI()
        self.ai_player = "O"
        self.human_player = "X"

    def test_h_immediate_win(self):
        """Heuristic AB takes an immediate winning move."""
        game = Game(3, 3, "X", "O")
        game.board.grid = [["O", "O", None], ["X", "X", None], [None, None, None]]
        move = self.ai.get_best_move_heuristic(game, self.ai_player, self.human_player, max_depth=4)
        self.assertEqual(move, (0, 2))

    def test_h_immediate_block(self):
        """Heuristic AB blocks the human from winning on the next move."""
        game = Game(3, 3, "X", "O")
        game.board.grid = [["X", "X", None], ["O", None, None], [None, None, None]]
        move = self.ai.get_best_move_heuristic(game, self.ai_player, self.human_player, max_depth=4)
        self.assertEqual(move, (0, 2))

    def test_h_full_board_returns_none(self):
        """Heuristic AB returns None when no moves are available."""
        game = Game(3, 3, "X", "O")
        game.board.grid = [["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]]
        move = self.ai.get_best_move_heuristic(game, self.ai_player, self.human_player, max_depth=4)
        self.assertIsNone(move)

    def test_h_nodes_explored_positive(self):
        """nodes_explored_h must be positive after any non-trivial search."""
        game = Game(3, 3, "X", "O")
        self.ai.get_best_move_heuristic(game, self.ai_player, self.human_player, max_depth=4)
        self.assertGreater(self.ai.nodes_explored_h, 0)

    def test_h_terminal_bypass_at_depth_1(self):
        """
        Terminal wins are detected before the depth cutoff.
        Even max_depth=1 must find an immediate winning move because the terminal
        check runs first — the depth limit is never reached for a winning position.
        """
        game = Game(3, 3, "X", "O")
        game.board.grid = [["O", "O", None], ["X", "X", None], [None, None, None]]
        move = self.ai.get_best_move_heuristic(game, self.ai_player, self.human_player, max_depth=1)
        self.assertEqual(move, (0, 2),
            "Immediate win must be found at max_depth=1 — terminals bypass the depth cutoff")

    def test_h_depth_limit_reduces_nodes(self):
        """
        A shallower depth limit must visit strictly fewer nodes than a deeper one.
        This directly verifies that max_depth is actually bounding the search.
        """
        game_shallow = Game(4, 3, "X", "O")
        game_deep    = Game(4, 3, "X", "O")

        self.ai.get_best_move_heuristic(
            game_shallow, self.ai_player, self.human_player, max_depth=1
        )
        nodes_shallow = self.ai.nodes_explored_h

        self.ai.get_best_move_heuristic(
            game_deep, self.ai_player, self.human_player, max_depth=3
        )
        nodes_deep = self.ai.nodes_explored_h

        self.assertLess(
            nodes_shallow, nodes_deep,
            f"max_depth=1 ({nodes_shallow} nodes) should be less than "
            f"max_depth=3 ({nodes_deep} nodes)",
        )

    def test_h_fewer_nodes_than_exhaustive(self):
        """Depth-limited AB must explore fewer nodes than exhaustive AB on the same board."""
        grid = [["X", None, None], [None, "O", None], [None, None, "X"]]

        game_h  = Game(3, 3, "X", "O")
        game_ab = Game(3, 3, "X", "O")
        game_h.board.grid  = [row[:] for row in grid]
        game_ab.board.grid = [row[:] for row in grid]

        self.ai.get_best_move_heuristic(
            game_h, self.ai_player, self.human_player, max_depth=2
        )
        nodes_h = self.ai.nodes_explored_h

        self.ai.get_best_move_ab(game_ab, self.ai_player, self.human_player)
        nodes_ab = self.ai.nodes_explored_ab

        self.assertLess(
            nodes_h, nodes_ab,
            f"Depth-limited ({nodes_h} nodes) should be less than "
            f"exhaustive AB ({nodes_ab} nodes)",
        )

    def test_h_correct_move_on_4x4_board(self):
        """Heuristic AB finds the winning move on a 4×4 board under a depth cap."""
        game = Game(4, 3, "X", "O")
        game.board.grid = [
            ["O", "O", None, None],
            ["X", "X", None, None],
            [None, None, None, None],
            [None, None, None, None],
        ]
        move = self.ai.get_best_move_heuristic(
            game, self.ai_player, self.human_player, max_depth=3
        )
        self.assertEqual(move, (0, 2), "AI should win immediately on the 4×4 board")


if __name__ == "__main__":
    unittest.main()
