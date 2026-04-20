import unittest

from src.data_collection import (
    collect_dataset,
    outcome_from_perspective,
    tactical_policy,
)
from src.game import Game


class TestDataCollection(unittest.TestCase):
    def test_outcome_encoding(self):
        self.assertEqual(outcome_from_perspective("X", "X"), 1)
        self.assertEqual(outcome_from_perspective("O", "X"), -1)
        self.assertEqual(outcome_from_perspective(None, "X"), 0)

    def test_collect_dataset_non_empty(self):
        rows = collect_dataset(
            num_games=5,
            n=3,
            k=3,
            matchup="random_vs_random",
            seed=1,
        )
        self.assertTrue(len(rows) > 0)

    def test_row_has_required_fields(self):
        rows = collect_dataset(
            num_games=3,
            n=3,
            k=3,
            matchup="tactical_vs_random",
            seed=2,
        )
        row = rows[0]
        for key in [
            "game_id",
            "turn_index",
            "current_player",
            "opponent_player",
            "n",
            "k",
            "board_before_move",
            "move_row",
            "move_col",
            "final_winner",
            "final_outcome",
            "my_marks",
            "opp_marks",
            "empty_cells",
            "my_immediate_wins",
            "opp_immediate_wins",
        ]:
            self.assertIn(key, row)

    def test_final_outcome_value_range(self):
        rows = collect_dataset(
            num_games=5,
            n=4,
            k=3,
            matchup="tactical_vs_random",
            seed=7,
        )
        valid = {-1, 0, 1}
        for row in rows:
            self.assertIn(row["final_outcome"], valid)

    def test_tactical_policy_takes_immediate_win(self):
        game = Game(3, 3, "X", "O")
        game.board.grid = [
            ["X", "X", None],
            ["O", None, None],
            [None, None, None],
        ]
        move = tactical_policy(game, "X", "O", __import__("random").Random(0))
        self.assertEqual(move, (0, 2))


if __name__ == "__main__":
    unittest.main()
