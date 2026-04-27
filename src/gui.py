"""
Milestone 9 -- GUI (Tkinter)
Human vs AI on any n x n, k-in-a-row board.

Run:  python src/gui.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import tkinter as tk
from tkinter import messagebox

from src.ai import MinimaxAI
from src.game import Game

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
PLAYER_COLORS  = {"X": "#4A90D9", "O": "#E74C3C"}
WIN_COLOR       = "#2ECC71"
LAST_OUTLINE    = "#27AE60"   # green border on the most-recently placed cell
CELL_BG         = "white"
GRID_LINE       = "#CCCCCC"
BG              = "#F5F5F5"
STATUS_FONT     = ("Helvetica", 12)
MAX_CELL_PX     = 72
MIN_CELL_PX     = 28
GAP             = 2           # pixels between cells


def _auto_depth(n: int) -> int:
    if n <= 5:
        return 4
    if n <= 10:
        return 3
    return 2


# ---------------------------------------------------------------------------
# Start dialog
# ---------------------------------------------------------------------------

class _StartDialog(tk.Toplevel):
    """Modal config dialog; sets self.result to a dict or None on cancel."""

    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.title("New Game")
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        pad = {"padx": 10, "pady": 4}

        tk.Label(self, text="Board size  n:").grid(row=0, column=0, sticky="e", **pad)
        self._n = tk.StringVar(value="3")
        tk.Entry(self, textvariable=self._n, width=6).grid(row=0, column=1, **pad)
        tk.Label(self, text="(3–100)", font=("Helvetica", 9), fg="gray").grid(
            row=0, column=2, sticky="w")

        tk.Label(self, text="Win length  k:").grid(row=1, column=0, sticky="e", **pad)
        self._k = tk.StringVar(value="3")
        tk.Entry(self, textvariable=self._k, width=6).grid(row=1, column=1, **pad)
        tk.Label(self, text="(auto: min(n, 5) — updates as you type n)",
                 font=("Helvetica", 9), fg="gray").grid(row=1, column=2, sticky="w")

        def _sync_k(*_):
            try:
                self._k.set(str(min(int(self._n.get()), 5)))
            except ValueError:
                pass

        self._n.trace_add("write", _sync_k)

        tk.Label(self, text="Search depth:").grid(row=2, column=0, sticky="e", **pad)
        self._depth = tk.StringVar(value="auto")
        tk.Entry(self, textvariable=self._depth, width=6).grid(row=2, column=1, **pad)
        tk.Label(self, text='("auto" picks by board size)',
                 font=("Helvetica", 9), fg="gray").grid(row=2, column=2, sticky="w")

        tk.Label(self, text="You play as:").grid(row=3, column=0, sticky="e", **pad)
        self._human = tk.StringVar(value="X")
        frame = tk.Frame(self)
        frame.grid(row=3, column=1, columnspan=2, sticky="w", padx=10)
        tk.Radiobutton(frame, text="X  (first)",  variable=self._human, value="X").pack(side="left")
        tk.Radiobutton(frame, text="O  (second)", variable=self._human, value="O").pack(side="left")

        tk.Button(self, text="Start", command=self._ok,
                  width=12).grid(row=4, column=0, columnspan=3, pady=12)
        self.wait_window()

    def _ok(self) -> None:
        try:
            n = int(self._n.get())
            k = int(self._k.get())
            raw = self._depth.get().strip().lower()
            depth = _auto_depth(n) if raw == "auto" else int(raw)
            human = self._human.get()
            if n < 2 or k < 2 or k > n or depth < 1:
                raise ValueError
            self.result = {"n": n, "k": k, "depth": depth, "human": human}
            self.destroy()
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Requirements: n >= 2,  2 <= k <= n,  depth >= 1",
                parent=self,
            )


# ---------------------------------------------------------------------------
# Main game window
# ---------------------------------------------------------------------------

class TicTacToeGUI:
    """
    Renders the board as canvas rectangles + text items.

    Using canvas items instead of Button widgets means the entire board is a
    single bitmap that scrolls natively — no per-widget repositioning, no
    flicker on large boards.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Tic-Tac-Toe AI")
        self.root.configure(bg=BG)
        self.root.withdraw()   # hidden until first game config is confirmed

        self._ai = MinimaxAI()
        self._game: Game | None = None
        self._human = "X"
        self._ai_player = "O"
        self._depth = 4
        self._active = False
        self._last_move: tuple | None = None   # (r, c) of most recent piece

        # Canvas item id maps — populated in _build_board
        self._rects: dict = {}   # (r, c) -> rectangle id
        self._texts: dict = {}   # (r, c) -> text id
        self._cell_size = MIN_CELL_PX

        self._build_chrome()
        self._new_game()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_chrome(self) -> None:
        menubar = tk.Menu(self.root)
        m = tk.Menu(menubar, tearoff=0)
        m.add_command(label="New Game", command=self._new_game)
        m.add_separator()
        m.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="Game", menu=m)
        self.root.config(menu=menubar)

        self._status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self._status_var,
                 font=STATUS_FONT, bg=BG, pady=6).pack()

        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True, padx=12, pady=4)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(container, bg=CELL_BG, highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = tk.Scrollbar(container, orient="vertical",   command=self._canvas.yview)
        h_scroll = tk.Scrollbar(container, orient="horizontal", command=self._canvas.xview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        self._canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self._canvas.bind("<Button-1>", self._on_canvas_click)

        # Mouse-wheel scroll — bind_all so it works when hovering over cells
        self.root.bind_all(
            "<MouseWheel>",
            lambda e: self._canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"),
        )
        self.root.bind_all(
            "<Shift-MouseWheel>",
            lambda e: self._canvas.xview_scroll(-1 if e.delta > 0 else 1, "units"),
        )

        tk.Button(self.root, text="New Game", command=self._new_game,
                  font=STATUS_FONT, width=14).pack(pady=10)

    def _build_board(self, n: int) -> None:
        self._canvas.delete("all")
        self._rects = {}
        self._texts = {}

        cell = max(MIN_CELL_PX, min(MAX_CELL_PX, 480 // n))
        font_size = max(9, cell // 3)
        self._cell_size = cell
        step = cell + GAP

        for r in range(n):
            for c in range(n):
                x1 = c * step + GAP
                y1 = r * step + GAP
                x2 = x1 + cell
                y2 = y1 + cell
                self._rects[(r, c)] = self._canvas.create_rectangle(
                    x1, y1, x2, y2, fill=CELL_BG, outline=GRID_LINE, width=1
                )
                self._texts[(r, c)] = self._canvas.create_text(
                    (x1 + x2) // 2, (y1 + y2) // 2,
                    text="", font=("Helvetica", font_size, "bold"), fill="white"
                )

        board_px = n * step + GAP
        vp = min(board_px, 700)
        self._canvas.configure(
            scrollregion=(0, 0, board_px, board_px),
            width=vp,
            height=vp,
        )

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    def _new_game(self) -> None:
        dlg = _StartDialog(self.root)
        if dlg.result is None:
            if self._game is None:   # cancelled before any game ever started
                self.root.quit()
            return
        cfg = dlg.result
        self._human     = cfg["human"]
        self._ai_player = "O" if self._human == "X" else "X"
        self._depth     = cfg["depth"]
        self._game      = Game(cfg["n"], cfg["k"], "X", "O")
        self._active    = True
        self._last_move = None
        self._build_board(cfg["n"])
        self.root.title(f"Tic-Tac-Toe AI  |  {cfg['n']}x{cfg['n']}  k={cfg['k']}")
        self._set_status(
            f"You are {self._human}  |  AI is {self._ai_player}  "
            f"|  {cfg['n']}x{cfg['n']} k={cfg['k']} depth={self._depth}  |  Your turn"
        )
        self.root.deiconify()   # show window (no-op on subsequent calls)
        self.root.geometry("")  # let tkinter resize to fit the new board
        if self._game.current_player == self._ai_player:
            self.root.after(300, self._do_ai_move)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event) -> None:
        if not self._active:
            return
        if self._game.current_player != self._human:
            return
        # Convert widget coords to canvas coords (accounts for scroll offset)
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        step = self._cell_size + GAP
        c = int(cx) // step
        r = int(cy) // step
        n = self._game.board.n
        if not (0 <= r < n and 0 <= c < n):
            return
        # Reject clicks that land in the gap between cells
        x1 = c * step + GAP
        y1 = r * step + GAP
        if not (x1 <= cx <= x1 + self._cell_size and y1 <= cy <= y1 + self._cell_size):
            return
        if self._game.board.grid[r][c] is not None:
            return
        self._apply_move(r, c)
        if self._active:
            self.root.after(120, self._do_ai_move)

    # ------------------------------------------------------------------
    # AI move
    # ------------------------------------------------------------------

    def _do_ai_move(self) -> None:
        if not self._active:
            return
        self._set_status("AI is thinking...")

        def _think() -> None:
            move = self._ai.get_best_move_heuristic(
                self._game, self._ai_player, self._human, max_depth=self._depth
            )
            self.root.after(0, lambda: self._apply_ai_result(move))

        threading.Thread(target=_think, daemon=True).start()

    def _apply_ai_result(self, move) -> None:
        if not self._active:
            return
        if move:
            self._apply_move(move[0], move[1])

    # ------------------------------------------------------------------
    # Shared move application
    # ------------------------------------------------------------------

    def _apply_move(self, r: int, c: int) -> None:
        player = self._game.current_player
        self._game.make_move(r, c)

        # Remove green outline from the previous last-move cell
        if self._last_move is not None:
            pr, pc = self._last_move
            self._canvas.itemconfig(self._rects[(pr, pc)], outline=GRID_LINE, width=1)

        # Fill the new cell and add a green outline to mark it as last-placed
        color = PLAYER_COLORS[player]
        self._canvas.itemconfig(self._rects[(r, c)], fill=color,
                                outline=LAST_OUTLINE, width=3)
        self._canvas.itemconfig(self._texts[(r, c)], text=player, fill="white")
        self._last_move = (r, c)

        winner = self._game.check_winner()
        if winner:
            self._finish(winner)
        elif self._game.is_draw():
            self._finish(None)
        else:
            self._game.switch_turn()
            cur = self._game.current_player
            self._set_status(
                "Your turn" if cur == self._human else "AI is thinking..."
            )

    # ------------------------------------------------------------------
    # End state
    # ------------------------------------------------------------------

    def _finish(self, winner) -> None:
        self._active = False
        if winner:
            for r, c in self._winning_cells(winner):
                self._canvas.itemconfig(self._rects[(r, c)],
                                        fill=WIN_COLOR, outline=WIN_COLOR, width=1)
                self._canvas.itemconfig(self._texts[(r, c)], fill="white")
            msg = "You win!" if winner == self._human else "AI wins!"
            self._set_status(f"{msg}  ({winner})  --  click New Game to play again")
        else:
            self._set_status("Draw!  --  click New Game to play again")

    def _winning_cells(self, player: str) -> list:
        board = self._game.board
        n, k = board.n, board.k
        for r in range(n):
            for c in range(n):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    cells = [(r + i * dr, c + i * dc) for i in range(k)]
                    if all(0 <= nr < n and 0 <= nc < n
                           and board.grid[nr][nc] == player
                           for nr, nc in cells):
                        return cells
        return []

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    root.resizable(True, True)
    TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
