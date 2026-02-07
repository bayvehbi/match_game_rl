
import torch
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from config import GRID_SIZE, N_TILES, DIRS

class MatchVecEnv(VecEnv):
    def __init__(self, num_envs, device="cuda", grid_size=GRID_SIZE, n_tiles=N_TILES):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.grid_size = grid_size
        self.n_tiles = n_tiles
        
        # Define spaces
        # Observation: One-Hot floats (C, H, W)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.n_tiles, self.grid_size, self.grid_size), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size * 4)
        
        super().__init__(num_envs, self.observation_space, self.action_space)
        
        # --- State Tensors (N_ENVS, H, W) ---
        self.grids = torch.zeros((num_envs, self.grid_size, self.grid_size), 
                                 dtype=torch.long, device=self.device)
        
        # --- Kernels ---
        self.k_h = torch.ones((self.n_tiles, 1, 1, 3), device=self.device, dtype=torch.float32)
        self.k_v = torch.ones((self.n_tiles, 1, 3, 1), device=self.device, dtype=torch.float32)
        
        # --- Stats (Zero Overhead) ---
        # [match, no_match, illegal, popped, total_moves]
        self.stats = torch.zeros(5, dtype=torch.long, device=self.device)
        self.shared_infos = [{} for _ in range(num_envs)] # Pre-allocate empty dicts
        
        # Directions tensor: (4, 2) -> [[-1,0], [1,0], ...]
        self.dir_vecs = torch.zeros((4, 2), dtype=torch.long, device=self.device)
        for d, (dr, dc) in DIRS.items():
            self.dir_vecs[d] = torch.tensor([dr, dc], device=self.device)
            
        self.actions = None
        
        # Initial Reset
        print(f"Initializing {num_envs} GPU Environments...")
        self._reset_all_grids()

    def reset(self):
        # SB3 expects numpy array from reset
        return self._get_obs().cpu().numpy()

    def step_async(self, actions):
        # Store actions for step_wait
        if isinstance(actions, np.ndarray):
            self.actions = torch.from_numpy(actions).to(self.device)
        else:
            self.actions = torch.tensor(actions, device=self.device)

    def step_wait(self):
        # 1. Parse Actions
        # action = tile_idx * 4 + direction_idx
        tile_indices = self.actions // 4
        directions = self.actions % 4
        
        r = tile_indices // self.grid_size
        c = tile_indices % self.grid_size
        
        # Get Delta (dr, dc)
        deltas = self.dir_vecs[directions] # (N, 2)
        dr, dc = deltas[:, 0], deltas[:, 1]
        
        r2 = r + dr
        c2 = c + dc
        
        # 2. Check Valid Bounds
        in_bounds = (r2 >= 0) & (r2 < self.grid_size) & (c2 >= 0) & (c2 < self.grid_size)
        
        # 3. Phase-1 Shaping: Check if move is "valid" (results in match)
        # We need to simulate the swap for checking validity.
        
        # Prepare grids for check
        temp_grids = self.grids.clone()
        
        # Apply swaps where bounds are valid
        batch_idx = torch.arange(self.num_envs, device=self.device)
        
        # Swap logic
        valid_b = batch_idx[in_bounds]
        valid_r, valid_c = r[in_bounds], c[in_bounds]
        valid_r2, valid_c2 = r2[in_bounds], c2[in_bounds]
        
        val_source = temp_grids[valid_b, valid_r, valid_c]
        val_dest = temp_grids[valid_b, valid_r2, valid_c2]
        
        temp_grids[valid_b, valid_r, valid_c] = val_dest
        temp_grids[valid_b, valid_r2, valid_c2] = val_source
        
        # Check matches on temp_grids
        matches_mask = self._find_matches_mask(temp_grids) # (N, H, W) bool
        has_match = matches_mask.view(self.num_envs, -1).any(dim=1) # (N) bool
        
        # Shaping reward for valid move
        shaping_reward = torch.where(has_match, torch.tensor(10.0, device=self.device), torch.tensor(-3.0, device=self.device))
        
        # 4. Apply Moves to Real Grid
        applied_mask = in_bounds # Initially try all in-bounds
        
        # Apply swaps to a candidate grid to resolve cascades. 
        # If nothing pops, we revert (so we don't update self.grids).
        
        final_grids = self.grids.clone()
        
        # Apply Swap
        b_idx = batch_idx[applied_mask]
        final_grids[b_idx, r[applied_mask], c[applied_mask]] = self.grids[b_idx, r2[applied_mask], c2[applied_mask]]
        final_grids[b_idx, r2[applied_mask], c2[applied_mask]] = self.grids[b_idx, r[applied_mask], c[applied_mask]]
        
        # 5. Resolve Cascades (Loop)
        total_popped = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        first_match_size = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        is_first_iter = True
        
        active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device) # Envs that are still cascading
        
        MAX_CASCADE = 15
        
        for _ in range(MAX_CASCADE):
            matches = self._find_matches_mask(final_grids) # (N, H, W)
            
            # Count matches per env
            # Be careful: matches is a bool mask of CELLS.
            # Number of matched cells.
            num_matched_cells = matches.view(self.num_envs, -1).sum(dim=1)
            
            # If an env has 0 matches, it stops cascading
            has_matches = num_matched_cells > 0
            active_mask = active_mask & has_matches
            
            if not active_mask.any():
                break
                
            # Update Metrics
            # If this is the first iteration, record first_match
            if is_first_iter:
                # Only for those that have matches
                first_match_size = torch.where(has_matches, num_matched_cells, first_match_size)
                is_first_iter = False
            
            # Accumulate total popped
            total_popped += (num_matched_cells * active_mask.long())
            
            # Remove Matches (Set to -1)
            # Only in active envs
            # We can just apply -1 where matches is True? 
            # Yes, matches is False for inactive envs (because find_matches returns False)
            final_grids[matches] = -1
            
            # Gravity & Refill
            self._apply_gravity_and_refill(final_grids, active_mask)
            
        
        # 6. Calc Rewards & Obs
        success_mask = (total_popped > 0)
        
        # Valid Move but No Match:
        # Original: -3 + shaping. 
        # Illegal: -4
        
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        # Case 1: Match
        rewards[success_mask] = 16.0
        
        # Case 2: No Match (but in bounds)
        fail_mask = in_bounds & (~success_mask)
        rewards[fail_mask] = -3.0
        
        # Case 3: Illegal (Out of bounds)
        illegal_mask = ~in_bounds
        rewards[illegal_mask] = -4.0
        
        # Add shaping
        rewards += shaping_reward
        
        # Update State
        self.grids = torch.where(success_mask.view(-1, 1, 1), final_grids, self.grids)
        
        # 7. Dead Board Check
        # If grid has no valid moves, we should regenerate.
        # Currently skipping for optimization purposes as dead boards are rare.
        refreshed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Return
        obs = self._get_obs()
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # Infinite horizon usually?
        # Or use max steps? The original env doesn't have max steps in code shown.
        # Usually handled by wrapper.
        
        infos = []
        # ZERO-OVERHEAD LOGGING STRATEGY:
        # Instead of creating python dicts (slow), we accumulate stats on GPU.
        # We return empty infos here. The callback will fetch self.stats periodically.
        
        # Determine current stats
        with torch.no_grad():
            matches_count = success_mask.sum()
            illegal_count = illegal_mask.sum()
            no_match_count = fail_mask.sum()
            popped_sum = total_popped.sum()
            
            self.stats[0] += matches_count
            self.stats[1] += no_match_count
            self.stats[2] += illegal_count
            self.stats[3] += popped_sum
            self.stats[4] += self.num_envs # total moves attempted
            
        # PPO requires infos to be a list of len(num_envs)
        # We return empty dicts to satisfy type check but avoid data transfer completely
        
        return obs.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), self.shared_infos

    def get_stats(self):
        # Called by Callback to get accumulated stats
        cpu_stats = self.stats.cpu().numpy()
        return {
            "match": cpu_stats[0],
            "no_match": cpu_stats[1],
            "illegal": cpu_stats[2],
            "popped": cpu_stats[3],
            "total_moves": cpu_stats[4]
        }
    
    def reset_stats(self):
        self.stats.zero_()

    def _get_obs(self):
        # Convert (N, H, W) int64 -> (N, C, H, W) float32 OneHot
        return F.one_hot(self.grids, num_classes=self.n_tiles).permute(0, 3, 1, 2).float()

    def _find_matches_mask(self, grids_tensor):
        # grids: (B, H, W)
        one_hot = self._get_one_hot(grids_tensor)
        
        match_h = F.conv2d(one_hot, self.k_h, padding=(0, 1), groups=self.n_tiles)
        match_v = F.conv2d(one_hot, self.k_v, padding=(1, 0), groups=self.n_tiles)
        
        mask_h = (match_h >= 2.9).float()
        mask_v = (match_v >= 2.9).float()
        
        final_mask_h = F.conv2d(mask_h, self.k_h, padding=(0, 1), groups=self.n_tiles)
        final_mask_v = F.conv2d(mask_v, self.k_v, padding=(1, 0), groups=self.n_tiles)
        
        matches = (final_mask_h + final_mask_v) > 0.9
        return matches.max(dim=1).values # (B, H, W) bool

    def _get_one_hot(self, grids):
        # Helper for internal use (grids can be temp)
        return F.one_hot(grids, num_classes=self.n_tiles).permute(0, 3, 1, 2).float()

    def _apply_gravity_and_refill(self, grids, active_mask):
        # grids: (B, H, W). -1 are empty.
        
        # 1. Gravity
        # Shift non -1 values down.
        # Approach: For each column, we want -1s to bubble up (to index 0).
        # We use a stable sort where 'is_tile' (1) > 'is_empty' (0).
        # This keeps the relative order of tiles but moves empty spaces to top.
        
        keys = (grids != -1).long()
        _, indices = torch.sort(keys, dim=1, stable=True)
        
        # Gather
        grids_gravity = torch.gather(grids, 1, indices)
        
        # 2. Refill
        # Replace -1s with random
        mask_empty = (grids_gravity == -1)
        if mask_empty.any():
            random_tiles = torch.randint(0, self.n_tiles, grids.shape, device=self.device)
            grids_gravity = torch.where(mask_empty, random_tiles, grids_gravity)
            
        # Update original grids tensor
        # Gravity and refill only affect columns with gaps (where values are -1).
        # For stable grids, this operation is an identity transformation, so it's safe to apply globally.
        
        # In-place update
        grids[:] = grids_gravity

    def _reset_all_grids(self):
        # Generates Pure Random grids.
        # Then resolves matches iteratively until stable.
        self.grids = torch.randint(0, self.n_tiles, (self.num_envs, self.grid_size, self.grid_size), device=self.device)
        
        # Loop until no matches exist
        stable_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for _ in range(20): # Safety break
            matches = self._find_matches_mask(self.grids)
            has_match = matches.view(self.num_envs, -1).any(dim=1)
            
            if not has_match.any():
                break
                
            self.grids[matches] = -1
            self._apply_gravity_and_refill(self.grids, None)

    # Required for SB3
    def seed(self, seed=None):
        pass # Handle seeding if needed

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, **method_kwargs):
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

