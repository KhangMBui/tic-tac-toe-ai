# tests/test_heuristics.py

import pytest
from src.board import Board
from src.heuristics import HeuristicEvaluator


def make_board(grid, n, k):
    """Helper to create a Board with a given grid."""
    board = Board(n, k)
    board.grid = [row[:] for row in grid]
    return board


def test_winning_board_for_X():
    # X has a full row (win)
    grid = [
        ["X", "X", "X"],
        [None, "O", None],
        [None, None, "O"],
    ]
    board = make_board(grid, 3, 3)
    h = HeuristicEvaluator()
    score = h.evaluate(board, "X", "O")
    assert (
        score > 100
    ), f"Winning board for X should have high positive score, got {score}"


def test_losing_board_for_X():
    # O has a full column (loss for X)
    grid = [
        ["X", "O", None],
        ["X", "O", None],
        [None, "O", "X"],
    ]
    board = make_board(grid, 3, 3)
    h = HeuristicEvaluator()
    score = h.evaluate(board, "X", "O")
    assert (
        score < -100
    ), f"Losing board for X should have high negative score, got {score}"


def test_draw_board():
    # Full board, no winner
    grid = [
        ["X", "O", "X"],
        ["O", "X", "O"],
        ["O", "X", "O"],
    ]
    board = make_board(grid, 3, 3)
    h = HeuristicEvaluator()
    score = h.evaluate(board, "X", "O")
    assert abs(score) < 10, f"Draw board should have near-zero score, got {score}"


def test_immediate_win_for_X():
    # X can win next move; O has no immediate threat
    grid = [
        ["X", "X", None],
        ["O", None, None],
        [None, None, None],
    ]
    board = make_board(grid, 3, 3)
    h = HeuristicEvaluator()
    score = h.evaluate(board, "X", "O")
    assert score > 1, f"Immediate win for X should have positive score, got {score}"


def test_immediate_win_for_O():
    # O can win next move; X has no immediate threat
    grid = [
        ["O", "O", None],
        ["X", None, None],
        [None, None, None],
    ]
    board = make_board(grid, 3, 3)
    h = HeuristicEvaluator()
    score = h.evaluate(board, "X", "O")
    assert score < -1, f"Immediate win for O should have negative score, got {score}"


def test_symmetry():
    # Board is symmetric: swapping players should flip sign
    grid = [
        ["X", "X", None],
        ["O", "O", None],
        [None, None, None],
    ]
    board = make_board(grid, 3, 3)
    h = HeuristicEvaluator()
    score_X = h.evaluate(board, "X", "O")
    score_O = h.evaluate(board, "O", "X")
    assert (
        abs(score_X + score_O) < 1e-6
    ), f"Scores should be symmetric, got {score_X} and {score_O}"


if __name__ == "__main__":
    pytest.main([__file__])
