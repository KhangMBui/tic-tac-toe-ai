# Responsible for evaluation when search is depth-limited.
# For example:
#   center preference
#   counting open lines
#   rewarding near-complete rows/cols/diagonals

from src.features import FeatureExtractor


class HeuristicEvaluator:
    """
    Data-driven, interpretable heuristic evaluation for n x n, k-in-a-row Tic-Tac-Toe.
    The weights are chosen based on feature correlation with winning from your dataset.
    """

    def __init__(self, weights=None):
        # Data-driven, interpretable weights based on Milestone 5 analysis.
        # Rationale for each feature:
        # - my_immediate_wins: Much higher in wins than losses. Strong positive.
        # - opp_immediate_wins: Much higher in losses. Strong negative.
        # - my_open_2: Higher in wins. Positive.
        # - opp_open_2: Higher in losses. Negative.
        # - my_two_way_threats: Higher in wins, but also high in losses. Mild positive.
        # - opp_two_way_threats: Higher in losses, but also high in wins. Mild negative.
        # - my_center_control: Higher in wins. Positive.
        # - opp_center_control: Higher in losses. Negative.
        #
        # Features not included: open_lines, open_3, winning_windows, etc., as they were not as discriminative in your data.
        self.weights = weights or {
            # Terminal positions — someone already won
            "my_winning_windows": 1000.0,   # Board already won by me
            "opp_winning_windows": -1000.0,  # Board already won by opponent
            # Immediate win/loss patterns
            "my_immediate_wins": 2.0,  # Strong positive for immediate win
            "opp_immediate_wins": -2.0,  # Strong negative for opponent's immediate win
            # Open (k-1) lines
            "my_open_2": 1.0,  # Positive for open lines close to winning
            "opp_open_2": -1.0,  # Negative for opponent's open lines
            # Two-way threats
            "my_two_way_threats": 0.5,  # Mild positive (predictive but less discriminative)
            "opp_two_way_threats": -1.0,  # Mild negative (higher in losses)
            # Center control
            "my_center_control": 1.0,  # Positive for controlling center
            "opp_center_control": -1.0,  # Negative for opponent's center control
        }
        self.extractor = FeatureExtractor()

    def evaluate(self, board, my_player, opponent_player):
        """
        Returns a heuristic score for the given board and player.
        Positive = good for 'player', negative = bad for 'player'
        """
        features = self.extractor.extract(board, my_player, opponent_player)
        score = 0.0
        for feat, weight in self.weights.items():
            score += weight * features.get(feat, 0)
        return score
