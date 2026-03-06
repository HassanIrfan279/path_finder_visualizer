"""
Dynamic Pathfinding Agent
Algorithms : Greedy Best-First Search (GBFS) & A*
Heuristics : Manhattan & Euclidean
GUI        : tkinter  (built-in — no pip install needed)
Python     : 3.x compatible (including 3.14)
"""

import tkinter as tk
from tkinter import ttk
import heapq
import math
import random
import time

# ─────────────────────────────────────────────────────────
#  GRID SETTINGS
# ─────────────────────────────────────────────────────────
ROWS      = 20
COLS      = 28
CELL      = 30          # pixel size of one cell

# ─────────────────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────────────────
C_BG       = "#1e1e2e"
C_EMPTY    = "#ffffff"
C_WALL     = "#2a2a3a"
C_START    = "#00c853"   # green
C_GOAL     = "#ff1744"   # red
C_VISITED  = "#5b9bd5"   # blue  – expanded nodes
C_PATH     = "#00e676"   # teal  – final path
C_AGENT    = "#ff6d00"   # orange circle
C_GRID     = "#cccccc"   # grid lines

STEP_MS    = 150         # milliseconds between agent steps

# ─────────────────────────────────────────────────────────
#  HEURISTICS
# ─────────────────────────────────────────────────────────
def manhattan(a, b):
    """h(n) = |delta_row| + |delta_col|"""
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    """h(n) = sqrt(delta_row^2 + delta_col^2)"""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ─────────────────────────────────────────────────────────
#  GRID NEIGHBOURS  (4-directional, no diagonals)
# ─────────────────────────────────────────────────────────
def get_neighbours(pos, rows, cols, walls):
    r, c = pos
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) not in walls:
            result.append((nr, nc))
    return result

# ─────────────────────────────────────────────────────────
#  A* SEARCH   f(n) = g(n) + h(n)
# ─────────────────────────────────────────────────────────
def astar(start, goal, rows, cols, walls, hfn):
    heap = []
    heapq.heappush(heap, (0, start))

    came_from = {start: None}
    g_cost    = {start: 0}
    visited   = []

    while heap:
        _, cur = heapq.heappop(heap)

        if cur == goal:
            return reconstruct(came_from, goal), visited

        visited.append(cur)

        for nb in get_neighbours(cur, rows, cols, walls):
            new_g = g_cost[cur] + 1
            if nb not in g_cost or new_g < g_cost[nb]:
                g_cost[nb]    = new_g
                came_from[nb] = cur
                priority      = new_g + hfn(nb, goal)
                heapq.heappush(heap, (priority, nb))

    return None, visited   # no path found

# ─────────────────────────────────────────────────────────
#  GREEDY BEST-FIRST SEARCH   f(n) = h(n)
# ─────────────────────────────────────────────────────────
def gbfs(start, goal, rows, cols, walls, hfn):
    heap = []
    heapq.heappush(heap, (hfn(start, goal), start))

    came_from = {start: None}
    visited   = []

    while heap:
        _, cur = heapq.heappop(heap)

        if cur == goal:
            return reconstruct(came_from, goal), visited

        visited.append(cur)

        for nb in get_neighbours(cur, rows, cols, walls):
            if nb not in came_from:
                came_from[nb] = cur
                heapq.heappush(heap, (hfn(nb, goal), nb))

    return None, visited

# ─────────────────────────────────────────────────────────
#  RECONSTRUCT PATH from came_from dict
# ─────────────────────────────────────────────────────────
def reconstruct(came_from, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path

# ─────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────
class PathfinderApp:

    def __init__(self, root):
        self.root = root
        root.title("Dynamic Pathfinding Agent")
        root.configure(bg=C_BG)
        root.resizable(False, False)

        # grid state
        self.rows  = ROWS
        self.cols  = COLS
        self.walls = set()
        self.start = (2, 2)
        self.goal  = (self.rows-3, self.cols-3)

        # animation state
        self.path          = []
        self.visited_cells = set()
        self.agent_index   = 0
        self.animating     = False
        self.draw_mode     = "wall"    # "wall" | "start" | "goal"

        # metrics (StringVar so labels auto-update)
        self.var_nodes  = tk.StringVar(value="0")
        self.var_cost   = tk.StringVar(value="0")
        self.var_time   = tk.StringVar(value="0 ms")
        self.var_status = tk.StringVar(value="Ready")

        # algorithm / heuristic choices
        self.algo_var = tk.StringVar(value="A*")
        self.heur_var = tk.StringVar(value="Manhattan")
        self.dyn_var  = tk.BooleanVar(value=False)

        self._build_ui()
        self._draw_all()

        # mouse bindings on canvas
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>",     self._on_left_drag)
        self.canvas.bind("<ButtonPress-3>", self._on_right_press)
        self.canvas.bind("<B3-Motion>",     self._on_right_drag)

    # ─────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────
    def _build_ui(self):
        # ── canvas (left side) ──
        self.canvas = tk.Canvas(
            self.root,
            width=self.cols * CELL,
            height=self.rows * CELL,
            bg=C_EMPTY,
            highlightthickness=2,
            highlightbackground="#444"
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # ── panel (right side) ──
        panel = tk.Frame(self.root, bg=C_BG, width=230)
        panel.grid(row=0, column=1, sticky="ns", padx=(0,10), pady=10)
        panel.grid_propagate(False)

        # helpers for adding widgets
        def sep():
            tk.Frame(panel, bg="#444444", height=1).pack(fill="x", pady=5, padx=4)

        def heading(text):
            tk.Label(panel, text=text, bg=C_BG, fg="#aaaacc",
                     font=("Consolas", 10, "bold")).pack(anchor="w", padx=8, pady=(4,0))

        def metric_row(label, var):
            row = tk.Frame(panel, bg=C_BG)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=label, bg=C_BG, fg="#aaaacc",
                     font=("Consolas", 9), width=15, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, bg=C_BG, fg="#ddeeff",
                     font=("Consolas", 9, "bold")).pack(side="left")

        def btn(text, cmd, color="#3c3c7a"):
            b = tk.Button(
                panel, text=text, command=cmd,
                bg=color, fg="white",
                activebackground="#5555aa",
                font=("Consolas", 10, "bold"),
                relief="flat", cursor="hand2", pady=5
            )
            b.pack(fill="x", padx=8, pady=2)
            return b

        # ── Algorithm section ──
        heading("Algorithm")
        for label, value in [("A*  (g + h)", "A*"), ("Greedy BFS  (h only)", "GBFS")]:
            tk.Radiobutton(
                panel, text=label, variable=self.algo_var, value=value,
                bg=C_BG, fg="white", selectcolor="#444",
                activebackground=C_BG, font=("Consolas", 10)
            ).pack(anchor="w", padx=18)

        sep()

        # ── Heuristic section ──
        heading("Heuristic")
        for label, value in [("Manhattan  |dr|+|dc|", "Manhattan"),
                              ("Euclidean  sqrt(..)", "Euclidean")]:
            tk.Radiobutton(
                panel, text=label, variable=self.heur_var, value=value,
                bg=C_BG, fg="white", selectcolor="#444",
                activebackground=C_BG, font=("Consolas", 10)
            ).pack(anchor="w", padx=18)

        sep()

        # ── Action buttons ──
        heading("Controls")
        btn("▶  RUN",       self._on_run,     "#006633")
        btn("Clear Path",   self._clear_path)
        btn("Reset Grid",   self._reset_grid)
        btn("Random Maze",  self._random_maze)

        sep()

        # ── Dynamic mode ──
        tk.Checkbutton(
            panel, text="Dynamic Mode  (live obstacles)",
            variable=self.dyn_var,
            bg=C_BG, fg="white", selectcolor="#444",
            activebackground=C_BG, font=("Consolas", 10, "bold")
        ).pack(anchor="w", padx=8, pady=4)

        sep()

        # ── Draw mode ──
        heading("Set Nodes")
        btn("Click to Set START", lambda: self._set_draw("start"), "#1a5c1a")
        btn("Click to Set GOAL",  lambda: self._set_draw("goal"),  "#5c1a1a")

        sep()

        # ── Metrics ──
        heading("Metrics")
        metric_row("Nodes Visited :", self.var_nodes)
        metric_row("Path Cost     :", self.var_cost)
        metric_row("Time          :", self.var_time)

        sep()

        # ── Status ──
        heading("Status")
        tk.Label(
            panel, textvariable=self.var_status,
            bg=C_BG, fg="#ffdd55",
            font=("Consolas", 10, "bold"),
            wraplength=210, justify="left"
        ).pack(anchor="w", padx=8, pady=2)

        sep()

        # ── Colour legend ──
        heading("Legend")
        for color, text in [
            (C_START,   "Start node"),
            (C_GOAL,    "Goal node"),
            (C_WALL,    "Wall"),
            (C_VISITED, "Visited nodes"),
            (C_PATH,    "Final path"),
            (C_AGENT,   "Agent"),
        ]:
            row = tk.Frame(panel, bg=C_BG)
            row.pack(fill="x", padx=8, pady=1)
            tk.Canvas(row, width=14, height=14,
                      bg=color, highlightthickness=0).pack(side="left", padx=(0,6))
            tk.Label(row, text=text, bg=C_BG, fg="white",
                     font=("Consolas", 9)).pack(side="left")

        sep()

        # ── Tips ──
        tk.Label(
            panel,
            text="Left-click / drag = place wall\nRight-click / drag = erase wall",
            bg=C_BG, fg="#888899",
            font=("Consolas", 8), justify="left"
        ).pack(anchor="w", padx=8, pady=(0,4))

    # ─────────────────────────────────────────
    #  DRAW GRID
    # ─────────────────────────────────────────
    def _draw_all(self):
        self.canvas.delete("all")
        path_set = set(self.path)

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * CELL
                y1 = r * CELL
                x2 = x1 + CELL
                y2 = y1 + CELL
                pos = (r, c)

                # pick fill colour
                if pos == self.start:
                    fill = C_START
                elif pos == self.goal:
                    fill = C_GOAL
                elif pos in self.walls:
                    fill = C_WALL
                elif pos in path_set:
                    fill = C_PATH
                elif pos in self.visited_cells:
                    fill = C_VISITED
                else:
                    fill = C_EMPTY

                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=fill, outline=C_GRID, width=1
                )

        # draw S / G labels
        for pos, letter in [(self.start, "S"), (self.goal, "G")]:
            r, c = pos
            self.canvas.create_text(
                c * CELL + CELL // 2,
                r * CELL + CELL // 2,
                text=letter,
                font=("Consolas", 11, "bold"),
                fill="white"
            )

        # draw moving agent
        if self.animating and self.path:
            ar, ac = self.path[self.agent_index]
            cx = ac * CELL + CELL // 2
            cy = ar * CELL + CELL // 2
            rad = CELL // 2 - 4
            self.canvas.create_oval(
                cx - rad, cy - rad,
                cx + rad, cy + rad,
                fill=C_AGENT, outline=""
            )

    # ─────────────────────────────────────────
    #  HELPER: which heuristic function to use
    # ─────────────────────────────────────────
    def _hfn(self):
        if self.heur_var.get() == "Manhattan":
            return manhattan
        return euclidean

    # ─────────────────────────────────────────
    #  RUN ALGORITHM & UPDATE METRICS
    # ─────────────────────────────────────────
    def _run_algo(self, start=None):
        s = start if start is not None else self.start

        t0 = time.perf_counter()

        if self.algo_var.get() == "A*":
            path, visited = astar(s, self.goal,
                                  self.rows, self.cols,
                                  self.walls, self._hfn())
        else:
            path, visited = gbfs(s, self.goal,
                                 self.rows, self.cols,
                                 self.walls, self._hfn())

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # update metrics panel
        self.visited_cells = set(visited)
        self.var_nodes.set(str(len(visited)))
        self.var_time.set(f"{elapsed_ms:.2f} ms")

        if path:
            self.var_cost.set(str(len(path) - 1))
            self.var_status.set("Path found! ✓")
        else:
            self.var_cost.set("0")
            self.var_status.set("No path found!")

        return path

    # ─────────────────────────────────────────
    #  SET DRAW MODE
    # ─────────────────────────────────────────
    def _set_draw(self, mode):
        self.draw_mode = mode
        self.var_status.set(f"Click a cell to set {mode.upper()}...")

    # ─────────────────────────────────────────
    #  CLEAR / RESET
    # ─────────────────────────────────────────
    def _clear_path(self):
        self.path          = []
        self.visited_cells = set()
        self.animating     = False
        self.agent_index   = 0
        self.var_nodes.set("0")
        self.var_cost.set("0")
        self.var_time.set("0 ms")
        self.var_status.set("Ready")
        self._draw_all()

    def _reset_grid(self):
        self.walls = set()
        self._clear_path()

    def _random_maze(self, density=0.28):
        self.walls = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in (self.start, self.goal):
                    continue
                if random.random() < density:
                    self.walls.add((r, c))
        self._clear_path()

    # ─────────────────────────────────────────
    #  RUN BUTTON
    # ─────────────────────────────────────────
    def _on_run(self):
        self._clear_path()
        path = self._run_algo()
        if path:
            self.path        = path
            self.agent_index = 0
            self.animating   = True
            self._draw_all()
            self.root.after(STEP_MS, self._animate_step)

    # ─────────────────────────────────────────
    #  ANIMATION (one step at a time via after())
    # ─────────────────────────────────────────
    def _animate_step(self):
        if not self.animating:
            return

        if self.agent_index < len(self.path) - 1:
            self.agent_index += 1

            # maybe spawn a random obstacle in dynamic mode
            if self.dyn_var.get():
                self._maybe_spawn_obstacle()

            self._draw_all()
            self.root.after(STEP_MS, self._animate_step)
        else:
            # agent reached the goal
            self.animating = False
            self.var_status.set("Goal reached! ✓")
            self._draw_all()

    # ─────────────────────────────────────────
    #  DYNAMIC OBSTACLE SPAWNING
    # ─────────────────────────────────────────
    def _maybe_spawn_obstacle(self):
        """6% chance per step to place a new wall somewhere."""
        if random.random() > 0.06:
            return   # no obstacle this step

        r = random.randint(0, self.rows - 1)
        c = random.randint(0, self.cols - 1)
        cell = (r, c)

        # do not overwrite start, goal, or already-walked cells
        walked = set(self.path[:self.agent_index + 1])
        if cell in self.walls or cell in (self.start, self.goal) or cell in walked:
            return

        self.walls.add(cell)

        # check if new wall blocks the remaining path
        remaining = set(self.path[self.agent_index:])
        if cell in remaining:
            self.var_status.set("Obstacle! Re-planning...")
            current  = self.path[self.agent_index]
            new_path = self._run_algo(start=current)
            if new_path:
                self.path        = new_path
                self.agent_index = 0
            else:
                self.animating = False
                self.var_status.set("Path blocked — no route!")

    # ─────────────────────────────────────────
    #  MOUSE HELPERS
    # ─────────────────────────────────────────
    def _cell_from_event(self, event):
        c = event.x // CELL
        r = event.y // CELL
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def _on_left_press(self, event):
        cell = self._cell_from_event(event)
        if cell is None:
            return

        if self.draw_mode == "start":
            # set new start position
            if cell not in self.walls and cell != self.goal:
                self.start = cell
            self.draw_mode = "wall"
            self.var_status.set("Ready")
            self._draw_all()

        elif self.draw_mode == "goal":
            # set new goal position
            if cell not in self.walls and cell != self.start:
                self.goal = cell
            self.draw_mode = "wall"
            self.var_status.set("Ready")
            self._draw_all()

        else:
            # place a wall
            if cell not in (self.start, self.goal):
                self.walls.add(cell)
                self._draw_all()

    def _on_left_drag(self, event):
        """Drag to paint walls."""
        cell = self._cell_from_event(event)
        if cell and self.draw_mode == "wall" and cell not in (self.start, self.goal):
            self.walls.add(cell)
            self._draw_all()

    def _on_right_press(self, event):
        """Right-click to erase a wall."""
        cell = self._cell_from_event(event)
        if cell:
            self.walls.discard(cell)
            self._draw_all()

    def _on_right_drag(self, event):
        """Drag to erase walls."""
        cell = self._cell_from_event(event)
        if cell:
            self.walls.discard(cell)
            self._draw_all()


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = PathfinderApp(root)
    root.mainloop()