from typing import Dict, List, Tuple
from src.board import Board


class FeatureExtractor:
    """
    Convert a board state into an interpretable feature dictionary.

    Why this class exists:
    ----------------------
    Minimax with full search works well on small boards, but on larger boards
    it becomes too expensive to explore every possible future state.

    To handle larger boards, we will eventually use a heuristic evaluation
    function. That heuristic needs a good "state representation" — a way to
    describe the board numerically.

    This class builds that representation.

    Design idea:
    ------------
    All features are extracted from the perspective of:
        - my_player       -> the player we are evaluating for
        - opponent_player -> the opposing player

    So the exact same board can produce different feature values depending on
    whose perspective we use.

    Example:
        If we evaluate from O's perspective, then "my_open_2" means
        "windows where O has 2 marks and the rest are empty."
    """

    def extract(
        self, board: Board, my_player: str, opponent_player: str
    ) -> Dict[str, int]:
        """
        Extract a feature dictionary for the given board state.

        Main feature groups:
        --------------------
        1. Basic board statistics
           - number of my marks
           - number of opponent marks
           - number of empty cells

        2. Positional control
           - how many center cells I control
           - how many center cells the opponent controls

        3. Window-based pattern features
           We look at every contiguous segment ("window") of length k in:
           - rows
           - columns
           - main diagonals
           - anti-diagonals

           Each window is classified as one of:
           - winning window
           - blocked window
           - open window for me
           - open window for opponent
           - neutral window

        4. Tactical threat features
           - immediate wins
           - fork-like / two-way threats
        """
        n = board.n
        k = board.k

        # This dictionary is the final numeric representation of the board state.
        # It is intentionally simple and interpretable so it can be used later
        # in a hand-designed or data-informed heuristic evaluation function.
        features: Dict[str, int] = {
            "n": n,
            "k": k,
            # Basic counts over the whole board
            "my_marks": 0,
            "opp_marks": 0,
            "empty_cells": 0,
            # Positional feature: controlling central cells is often useful
            "my_center_control": 0,
            "opp_center_control": 0,
            # Summary counts over all length-k windows
            "total_windows": 0,
            "neutral_windows": 0,
            "blocked_windows": 0,
            "my_open_lines": 0,
            "opp_open_lines": 0,
            # Tactical pattern counts
            "my_immediate_wins": 0,  # windows with k-1 my marks and 1 empty
            "opp_immediate_wins": 0,  # windows with k-1 opp marks and 1 empty
            # Fully completed windows
            "my_winning_windows": 0,  # windows fully occupied by my_player
            "opp_winning_windows": 0,  # windows fully occupied by opponent_player
            # "Fork-like" threats:
            # number of empty cells that would create 2 or more strong threats
            "my_two_way_threats": 0,  # empty cells that create >= 2 immediate wins
            "opp_two_way_threats": 0,
        }

        # Dynamic features:
        # my_open_i / opp_open_i count windows that contain exactly i marks
        # for one player and the rest empty.
        #
        # Example for k = 4:
        #   my_open_1 -> 1 of my marks + 3 empty
        #   my_open_2 -> 2 of my marks + 2 empty
        #   my_open_3 -> 3 of my marks + 1 empty
        #
        # We only create counts for 1..k-1 because k itself is tracked separately
        # as a winning window.
        for i in range(1, k):
            features[f"my_open_{i}"] = 0
            features[f"opp_open_{i}"] = 0

        # ------------------------------------------------------------
        # 1) Basic board counts
        # ------------------------------------------------------------
        # Count how many cells belong to me, the opponent, or are empty.
        for r in range(n):
            for c in range(n):
                cell = board.grid[r][c]
                if cell is None:
                    features["empty_cells"] += 1
                elif cell == my_player:
                    features["my_marks"] += 1
                elif cell == opponent_player:
                    features["opp_marks"] += 1

        # ------------------------------------------------------------
        # 2) Center control
        # ------------------------------------------------------------
        # Center squares are usually strategically useful because they often
        # participate in more possible winning lines than edge/corner cells.
        #
        # For odd n: there is 1 center cell.
        # For even n: we treat the 2x2 middle block as the "center region."
        for r, c in self._center_cells(n):
            cell = board.grid[r][c]
            if cell == my_player:
                features["my_center_control"] += 1
            elif cell == opponent_player:
                features["opp_center_control"] += 1

        # ------------------------------------------------------------
        # 3) Window-based feature extraction
        # ------------------------------------------------------------
        # A "window" is any contiguous sequence of length k in one of 4 directions:
        #   - right
        #   - down
        #   - down-right
        #   - down-left
        #
        # Why windows matter:
        # A player wins by completing k in a row, so these windows naturally
        # describe all potential winning segments on the board.
        windows = self._enumerate_windows(n, k)
        for coords in windows:
            my_count = 0
            opp_count = 0
            empty_count = 0

            # Count how many cells in this window belong to each side.
            for r, c in coords:
                cell = board.grid[r][c]
                if cell is None:
                    empty_count += 1
                elif cell == my_player:
                    my_count += 1
                elif cell == opponent_player:
                    opp_count += 1

            features["total_windows"] += 1

            # Case 1: already a completed winning segment for me
            if my_count == k:
                features["my_winning_windows"] += 1
                continue

            # Case 2: already a completed winning segment for opponent
            if opp_count == k:
                features["opp_winning_windows"] += 1
                continue

            # Case 3: blocked window
            # Both players appear, so neither side can fully complete this window.
            if my_count > 0 and opp_count > 0:
                features["blocked_windows"] += 1

            # Case 4: open window for me
            # Only my marks + empty spaces appear here.
            elif my_count > 0 and opp_count == 0:
                features[f"my_open_{my_count}"] += 1
                features["my_open_lines"] += 1

                # Immediate win pattern:
                # I already have k-1 marks in this window and only 1 empty remains.
                if my_count == k - 1 and empty_count == 1:
                    features["my_immediate_wins"] += 1

            # Case 5: open window for opponent
            # Only opponent marks + empty spaces appear here.
            elif opp_count > 0 and my_count == 0:
                features[f"opp_open_{opp_count}"] += 1
                features["opp_open_lines"] += 1

                # Opponent immediate win pattern
                if opp_count == k - 1 and empty_count == 1:
                    features["opp_immediate_wins"] += 1

            # Case 6: completely empty window
            else:
                features["neutral_windows"] += 1

        # ------------------------------------------------------------
        # 4) Fork-like tactical features
        # ------------------------------------------------------------
        # A "two-way threat" means there exists an empty cell such that, if a
        # player moves there, the position creates at least two strong winning
        # threats at once.
        #
        # This is very important strategically because the opponent may not be
        # able to block both threats with a single move.
        features["my_two_way_threats"] = self._count_two_way_threats(
            board, my_player, opponent_player
        )
        features["opp_two_way_threats"] = self._count_two_way_threats(
            board, opponent_player, my_player
        )

        return features

    def _count_two_way_threats(
        self,
        board: Board,
        player_to_place: str,
        other_player: str,
    ) -> int:
        """
        Count how many empty cells create a "two-way threat" for player_to_place.

        Idea:
        -----
        For each empty cell:
        1. Temporarily place player_to_place there.
        2. Examine all windows that include that cell.
        3. Count how many strong winning patterns this move creates.
        4. If it creates at least 2 such patterns, treat that square as a
           two-way threat cell.

        Why this matters:
        -----------------
        A move that creates multiple threats at once is usually much stronger
        than a normal move because it can force the opponent into a losing
        situation.

        Note:
        -----
        This method temporarily modifies board.grid, then restores it.
        That is safe here because the move is undone immediately after testing.
        """
        n = board.n
        k = board.k

        # On large boards this scan is O(n² × windows_per_cell) per heuristic
        # call. Because the heuristic is called at every search-tree leaf, the
        # cost multiplies with the number of nodes explored and becomes the
        # dominant bottleneck. Two-way threats are also sparser and less
        # discriminative on large boards, so skipping them is an acceptable
        # trade-off beyond this threshold.
        if n > 15:
            return 0

        windows = self._enumerate_windows(n, k)

        # Precompute cell → windows that contain it.
        # The naive approach checks every window for every empty cell using
        # `if (r, c) not in coords`, which is O(k) per check (list scan).
        # Precomputing this map reduces each lookup to O(1) and eliminates
        # scanning windows that cannot be affected by the candidate move.
        cell_windows: Dict[Tuple[int, int], List] = {
            (r, c): [] for r in range(n) for c in range(n)
        }
        for coords in windows:
            for rc in coords:
                cell_windows[rc].append(coords)

        threat_cells = 0
        for r in range(n):
            for c in range(n):
                # Skip occupied cells; only empty cells can be tested as candidate moves.
                if board.grid[r][c] is not None:
                    continue

                # Temporarily simulate placing a piece here.
                board.grid[r][c] = player_to_place
                immediate_wins_created = 0

                # Only windows containing (r, c) can be affected by this move.
                for coords in cell_windows[(r, c)]:
                    player_count = 0
                    other_count = 0
                    empty_count = 0

                    for wr, wc in coords:
                        cell = board.grid[wr][wc]
                        if cell is None:
                            empty_count += 1
                        elif cell == player_to_place:
                            player_count += 1
                        elif cell == other_player:
                            other_count += 1

                    # Strong pattern 1: this move already completes a winning window.
                    if player_count == k:
                        immediate_wins_created += 1

                    # Strong pattern 2: this move creates a near-complete open line
                    # with only one empty cell remaining.
                    elif (
                        player_count == k - 1 and other_count == 0 and empty_count == 1
                    ):
                        immediate_wins_created += 1

                # Undo the temporary move so the board returns to its original state.
                board.grid[r][c] = None

                # If one move creates at least two strong threats, count it as a fork.
                if immediate_wins_created >= 2:
                    threat_cells += 1

        return threat_cells

    @staticmethod
    def _center_cells(n: int) -> List[Tuple[int, int]]:
        """
        Return the coordinates considered to be the center of the board.

        For odd-sized boards:
            there is one true center cell.

        For even-sized boards:
            there is no single center, so we use the middle 2x2 block.
        """
        if n % 2 == 1:
            center = n // 2
            return [(center, center)]
        # 4-center block for even boards
        a = (n // 2) - 1
        b = n // 2
        return [(a, a), (a, b), (b, a), (b, b)]

    @staticmethod
    def _enumerate_windows(n: int, k: int) -> List[List[Tuple[int, int]]]:
        """
        Return every contiguous length-k window on the board.

        A window is a sequence of k cells in one of 4 directions:
            - right
            - down
            - down-right
            - down-left

        Why this helper is important:
        -----------------------------
        Tic-Tac-Toe and its generalized versions are won by completing k in a row.
        So if we want features that generalize across board sizes, it makes sense
        to analyze the board through all possible length-k windows.

        Example:
            On a 4x4 board with k=3, this method returns every possible
            horizontal, vertical, and diagonal 3-cell segment.
        """
        directions = [
            (0, 1),  # right
            (1, 0),  # down
            (1, 1),  # down-right
            (1, -1),  # down-left
        ]
        windows: List[List[Tuple[int, int]]] = []

        for r in range(n):
            for c in range(n):
                for dr, dc in directions:
                    end_r = r + dr * (k - 1)
                    end_c = c + dc * (k - 1)
                    if 0 <= end_r < n and 0 <= end_c < n:
                        coords = []
                        for i in range(k):
                            coords.append((r + dr * i, c + dc * i))
                        windows.append(coords)

        return windows
