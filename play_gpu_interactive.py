import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from match_gpu_env import MatchVecEnv
from config import GRID_SIZE, N_TILES

# Colors for tiles (0 to N_TILES-1)
TILE_COLORS = [
    "#FF6B6B", # Red/Pink
    "#4ECDC4", # Teal
    "#45B7D1", # Blue
    "#FFA07A", # Light Salmon
    "#98D8C8", # Mint
    "#F7DC6F", # Yellow
    "#BB8FCE", # Purple
]

class MatchInteractive:
    def __init__(self):
        print("Initializing GPU Environment...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = MatchVecEnv(num_envs=1, device=self.device)
        self.obs = self.env.reset() # (1, C, H, W)
        
        # Get the actual grid state (integers)
        # MatchVecEnv stores the grid in self.grids (num_envs, H, W)
        self.grid_state = self.env.grids[0].cpu().numpy()
        
        self.selected_cell = None # Tuple (row, col)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title("Match-3 GPU Interactive")
        
        self.step_count = 0
        self.last_reward = 0.0
        self.info_text = None
        
        # Connect event events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.render()
        
        plt.show()

    def render(self):
        self.ax.clear()
        
        # Draw cells
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                val = int(self.grid_state[r, c])
                color_idx = val % len(TILE_COLORS)
                color = TILE_COLORS[color_idx]
                
                # Matplotlib Rectangle (x, y) is bottom-left
                # We want row 0 at top. y = GRID_SIZE - 1 - r
                x = c
                y = GRID_SIZE - 1 - r
                
                edge_color = "white"
                line_width = 1
                
                # Highlight selection
                if self.selected_cell and self.selected_cell == (r, c):
                    edge_color = "black"
                    line_width = 4
                elif val == -1:
                    color = "#333333" # Empty/Background
                
                rect = Rectangle((x, y), 1, 1, facecolor=color, edgecolor=edge_color, linewidth=line_width)
                self.ax.add_patch(rect)
                
                # Draw text ID
                if val != -1:
                    self.ax.text(x + 0.5, y + 0.5, str(val), 
                                 ha="center", va="center", fontsize=12, 
                                 color="white", fontweight="bold")

        # Set limits and aspect
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Title/Status
        title = f"Step: {self.step_count} | Last Reward: {self.last_reward:.1f}"
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        self.fig.canvas.draw()

    def get_cell_from_coords(self, x, y):
        # x is column (0 to GRID)
        # y is from bottom (0 to GRID)
        if x is None or y is None: return None
        
        c = int(x)
        # y = GRID - 1 - r  =>  r = GRID - 1 - y
        r = GRID_SIZE - 1 - int(y)
        
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return (r, c)
        return None

    def on_click(self, event):
        if event.inaxes != self.ax: return
        
        cell = self.get_cell_from_coords(event.xdata, event.ydata)
        if not cell: return
        
        if self.selected_cell is None:
            self.selected_cell = cell
            self.render()
        else:
            # If clicked same cell, deselect
            if self.selected_cell == cell:
                self.selected_cell = None
                self.render()
            else:
                # If clicked adjacent, try to move
                r1, c1 = self.selected_cell
                r2, c2 = cell
                
                dist = abs(r1 - r2) + abs(c1 - c2)
                if dist == 1:
                    # Determine direction
                    direction = -1
                    if r2 < r1: direction = 0 # Up
                    elif r2 > r1: direction = 1 # Down
                    elif c2 < c1: direction = 2 # Left
                    elif c2 > c1: direction = 3 # Right
                    
                    self.execute_move(r1, c1, direction)
                    self.selected_cell = None # Deselect after move
                else:
                    # Select new cell
                    self.selected_cell = cell
                    self.render()

    def on_key(self, event):
        if self.selected_cell is None: return
        
        r, c = self.selected_cell
        direction = -1
        
        if event.key == 'up': direction = 0
        elif event.key == 'down': direction = 1
        elif event.key == 'left': direction = 2
        elif event.key == 'right': direction = 3
        
        if direction != -1:
            self.execute_move(r, c, direction)

    def execute_move(self, r, c, direction):
        # Action encoding
        # action = tile_idx * 4 + direction
        tile_idx = r * GRID_SIZE + c
        action = tile_idx * 4 + direction
        
        # Run step on GPU env
        actions = np.array([action])
        self.env.step_async(actions)
        obs, rewards, dones, infos = self.env.step_wait()
        
        # Update state
        self.grid_state = self.env.grids[0].cpu().numpy()
        self.last_reward = rewards[0]
        self.step_count += 1
        
        # Print stats to console as well
        stats = self.env.get_stats()
        print(f"Step {self.step_count}: Reward={self.last_reward:.1f}, "
              f"Matches={stats['match']}, Popped={stats['popped']}")
        
        self.render()

if __name__ == "__main__":
    game = MatchInteractive()
