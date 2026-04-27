# Responsible for evaluation when search is depth-limited.
# For example:
#   center preference
#   counting open lines
#   rewarding near-complete rows/cols/diagonals

from functools import lru_cache

from src.features import FeatureExtractor


class HeuristicEvaluator:
    """
    Data-driven, interpretable heuristic evaluation for n×n, k-in-a-row Tic-Tac-Toe.

    Weights are built dynamically per board so the heuristic generalizes to any (n, k):
      - Open-line weights reference open_{k-2} rather than a hardcoded open_2.
        For k=3 that is open_1; for k=5 that is open_3 (three-in-a-row), which is
        far more strategically relevant than open_2 on a large board.
      - Center control is dropped for n > 10, where the 2×2 center region represents
        less than 0.16 % of the board and carries no meaningful strategic signal.
      - Two-way-threat detection is skipped for n > 15 (handled in FeatureExtractor)
        because the O(n²) scan per heuristic call becomes the search bottleneck.

    Custom weights can still be passed to the constructor; they are used as-is and
    bypass all dynamic adjustment.
    """

    def __init__(self, weights=None):
        # None → build weights dynamically in evaluate() via _build_weights().
        # Provided dict → use exactly as given (caller takes responsibility for k-scaling).
        self._custom_weights = weights
        self.extractor = FeatureExtractor()

    @staticmethod
    @lru_cache(maxsize=32)
    def _build_weights(k: int, n: int) -> dict:
        """
        Return a weight dictionary tuned for the given board dimensions.

        Design rationale (grounded in M5 feature-correlation data):
          my_winning_windows  +1000 : terminal win — must dominate every other signal.
          opp_winning_windows −1000 : terminal loss.
          my_immediate_wins   +2.0  : k-1 marks + 1 empty; diff +0.52 in M5 data.
          opp_immediate_wins  −2.0  : symmetric.
          my_open_{k-2}       +1.0  : two marks short of a win — scales with k so
                                      the feature always represents a meaningful
                                      positional threat (open_3 for k=5, open_1 for k=3).
          opp_open_{k-2}      −1.0  : symmetric.
          my_two_way_threats  +0.5  : fork cells; positive but noisy (M5 diff +0.96
                                      yet opponent forks also high in wins → conservative).
          opp_two_way_threats −1.0  : more reliably negative per M5 data.
          center control      ±1.0  : only on small boards (n ≤ 10); on larger boards
                                      the 2×2 center is strategically negligible.
        """
        w = {
            "my_winning_windows":   1000.0,
            "opp_winning_windows": -1000.0,
            # Near-wins dominate evaluation — opponent with k-1 in a row will win
            # next turn if not blocked; own k-1 in a row is nearly a forced win.
            "my_immediate_wins":     50.0,
            "opp_immediate_wins":   -50.0,
            # Fork threats: opponent with two simultaneous threats is very dangerous.
            "my_two_way_threats":    5.0,
            "opp_two_way_threats":  -5.0,
        }

        # k-relative open-line weights.
        # open_{k-2} is the longest partial line that is NOT yet an immediate win
        # threat. For k=5 this is open_3 (three in a row, two short of a win).
        # Skipped for k < 3 because k-2 < 1 has no corresponding feature.
        # High weights force the AI to block building threats before they become
        # immediate wins — critical for k=5 boards where open-3 leads to open-4.
        if k >= 3:
            w[f"my_open_{k - 2}"]  =  15.0
            w[f"opp_open_{k - 2}"] = -15.0

        # Center control only matters on small boards.
        if n <= 10:
            w["my_center_control"]  =  1.0
            w["opp_center_control"] = -1.0

        return w

    def evaluate(self, board, my_player, opponent_player):
        """
        Return a heuristic score for the board from my_player's perspective.
        Positive = good for my_player. Negative = good for opponent_player.
        """
        features = self.extractor.extract(board, my_player, opponent_player)
        weights = (
            self._custom_weights
            if self._custom_weights is not None
            else self._build_weights(board.k, board.n)
        )
        score = 0.0
        for feat, weight in weights.items():
            score += weight * features.get(feat, 0)
        return score
