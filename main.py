"""
CLI runner for Tic-Tac-Toe AI project.
Allows two human players to play in the terminal.
"""

from src.game import Game
from src.ai import MinimaxAI


def get_move_input(n):
    while True:
        try:
            move = input(f"Enter your move as 'row col' (0-based): ")
            row, col = map(int, move.strip().split())
            if 0 <= row < n and 0 <= col < n:
                return row, col
            else:
                print(f"Invalid input: row and col must be between 0 and {n-1}.")
        except Exception:
            print("Invalid input format. Please enter two integers separated by space.")


def main():
    print("Welcome to Tic-Tac-Toe!")
    n = int(input("Enter board size n (e.g. 3 for 3x3): "))
    k = int(input("Enter win length k (e.g. 3 for 3-in-a-row): "))
    game = Game(n, k)
    ai = MinimaxAI()
    human_player = "X"
    ai_player = "O"

    while True:
        game.board.display()
        print(game.get_status())
        if game.is_terminal():
            break

        if game.current_player == human_player:
            row, col = get_move_input(n)
            if not game.make_move(row, col):
                print("Invalid move! Cell is already occupied or out of bounds.")
                continue
        else:
            print("AI is thinking...")
            move = ai.get_best_move(game, ai_player, human_player)
            if move is not None:
                row, col = move
                game.make_move(row, col)
                print(f"AI played at ({row}, {col})")

        if not game.is_terminal():
            game.switch_turn()

    game.board.display()
    print(game.get_status())
    print("Game over.")


if __name__ == "__main__":
    main()
