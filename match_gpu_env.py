"""
match_gpu_env.py
================
Fully GPU-vectorised Match-3 environment that supports MaskablePPO.

Key design choices
------------------
* action_masks() runs entirely on-device using chunked tensor ops.
* Match detection in the masking path uses direct cell-value comparisons
  (no one-hot expansion) → very small VRAM footprint.
* Dead boards are auto-refreshed so the mask is always non-empty.
"""

import torch
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from config import (
    GRID_SIZE,
    N_TILES,
    DIRS,
    MASK_CHUNK_SIZE,
    REWARD_MATCH,
    REWARD_NO_MATCH,
    REWARD_ILLEGAL,
    REWARD_SHAPING_MATCH,
    REWARD_SHAPING_NOMATCH,
    REWARD_COMBO_EXPONENT,
)


class MatchVecEnv(VecEnv):
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        grid_size: int = GRID_SIZE,
        n_tiles: int = N_TILES,
        mask_chunk_size: int = MASK_CHUNK_SIZE,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.grid_size = grid_size
        self.n_tiles = n_tiles
        self.mask_chunk_size = mask_chunk_size
        self.n_actions = grid_size * grid_size * 4

        # SB3 spaces (CPU / numpy world)
        obs_shape = (self.n_tiles, self.grid_size, self.grid_size)
        observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)
        action_space = spaces.Discrete(self.n_actions)
        super().__init__(num_envs, observation_space, action_space)

        # ── State (on GPU) ────────────────────────────────────────────
        self.grids = torch.zeros(
            (num_envs, grid_size, grid_size), dtype=torch.long, device=self.device
        )

        # ── Conv kernels for cascade match detection ──────────────────
        self.k_h = torch.ones((n_tiles, 1, 1, 3), device=self.device, dtype=torch.float32)
        self.k_v = torch.ones((n_tiles, 1, 3, 1), device=self.device, dtype=torch.float32)

        # ── Direction vectors ─────────────────────────────────────────
        self.dir_vecs = torch.zeros((4, 2), dtype=torch.long, device=self.device)
        for d, (dr, dc) in DIRS.items():
            self.dir_vecs[d] = torch.tensor([dr, dc], device=self.device)

        # ── Pre-computed swap-pair tensors (for action_masks) ─────────
        (
            self.swap_action_ids,
            self.swap_r1s,
            self.swap_c1s,
            self.swap_r2s,
            self.swap_c2s,
        ) = self._precompute_swap_pairs()

        # ── Stats ─────────────────────────────────────────────────────
        # [match, no_match, illegal, popped, total_moves, total_reward]
        self.stats = torch.zeros(6, dtype=torch.float32, device=self.device)
        self.shared_infos = [{} for _ in range(num_envs)]

        self.actions = None

        print(f"Initializing {num_envs} GPU environments "
              f"(grid={grid_size}x{grid_size}, tiles={n_tiles}, "
              f"action_space={self.n_actions})…")
        self._reset_all_grids()

    # ------------------------------------------------------------------
    # SB3 API
    # ------------------------------------------------------------------
    def reset(self):
        return self._get_obs().cpu().numpy()

    def step_async(self, actions):
        if isinstance(actions, np.ndarray):
            self.actions = torch.from_numpy(actions).to(self.device)
        else:
            self.actions = torch.tensor(actions, device=self.device)

    def step_wait(self):
        # ── 1. Parse actions ──────────────────────────────────────────
        tile_indices = self.actions // 4
        directions   = self.actions % 4
        r = tile_indices // self.grid_size
        c = tile_indices %  self.grid_size
        deltas = self.dir_vecs[directions]        # (N, 2)
        dr, dc = deltas[:, 0], deltas[:, 1]
        r2, c2 = r + dr, c + dc

        # ── 2. Bounds check ───────────────────────────────────────────
        in_bounds = (r2 >= 0) & (r2 < self.grid_size) & (c2 >= 0) & (c2 < self.grid_size)

        # ── 3. Phase-1 shaping: peek which moves create matches ───────
        temp_grids = self.grids.clone()
        batch_idx  = torch.arange(self.num_envs, device=self.device)
        valid_b    = batch_idx[in_bounds]
        vr, vc    = r[in_bounds], c[in_bounds]
        vr2, vc2  = r2[in_bounds], c2[in_bounds]

        val_src = temp_grids[valid_b, vr,  vc ].clone()
        val_dst = temp_grids[valid_b, vr2, vc2].clone()
        temp_grids[valid_b, vr,  vc ] = val_dst
        temp_grids[valid_b, vr2, vc2] = val_src

        has_match_after = self._find_matches_mask(temp_grids).view(self.num_envs, -1).any(1)
        shaping = torch.where(
            has_match_after,
            torch.tensor(REWARD_SHAPING_MATCH, device=self.device),
            torch.tensor(REWARD_SHAPING_NOMATCH, device=self.device),
        )

        # ── 4. Apply move to working copy ─────────────────────────────
        final_grids = self.grids.clone()
        b_idx = batch_idx[in_bounds]
        final_grids[b_idx, r[in_bounds],  c[in_bounds] ] = self.grids[b_idx, r2[in_bounds], c2[in_bounds]]
        final_grids[b_idx, r2[in_bounds], c2[in_bounds]] = self.grids[b_idx, r[in_bounds],  c[in_bounds] ]

        # ── 5. Cascade resolution ─────────────────────────────────────
        total_popped     = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        active_mask      = torch.ones (self.num_envs, dtype=torch.bool,  device=self.device)
        first_match_size = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        is_first_iter    = True

        for _ in range(15):
            matches       = self._find_matches_mask(final_grids)      # (N,H,W)
            matched_cells = matches.view(self.num_envs, -1).sum(1)
            has_matches   = matched_cells > 0
            active_mask   = active_mask & has_matches
            if not active_mask.any():
                break
            if is_first_iter:
                first_match_size = torch.where(has_matches, matched_cells, first_match_size)
                is_first_iter = False
            total_popped += (matched_cells * active_mask.long())
            final_grids[matches] = -1
            self._apply_gravity_and_refill(final_grids, active_mask)

        # ── 6. Rewards ────────────────────────────────────────────────
        success_mask = total_popped > 0
        fail_mask    = in_bounds & (~success_mask)
        illegal_mask = ~in_bounds

        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        if success_mask.any():
            p = total_popped[success_mask].float()
            # Multipliers requested by user:
            # 3 -> 1x
            # 4 -> 4x
            # 5 -> 6x
            # 6 -> 9x
            # and so on (+3 per tile)
            
            # Linear multiplier for N >= 6: 9 + (N-6)*3 => 3*N - 9
            mult = torch.where(p == 3, torch.tensor(1.0, device=self.device),
                   torch.where(p == 4, torch.tensor(4.0, device=self.device),
                   torch.where(p == 5, torch.tensor(6.0, device=self.device),
                   3.0 * p - 9.0)))
            
            rewards[success_mask] = mult * REWARD_MATCH
        
        rewards[fail_mask]    = REWARD_NO_MATCH
        rewards[illegal_mask] = REWARD_ILLEGAL
        rewards += shaping

        # ── 7. Update real grid ───────────────────────────────────────
        self.grids = torch.where(success_mask.view(-1, 1, 1), final_grids, self.grids)

        # ── 8. Accumulate stats ───────────────────────────────────────
        with torch.no_grad():
            self.stats[0] += success_mask.sum()
            self.stats[1] += fail_mask.sum()
            self.stats[2] += illegal_mask.sum()
            self.stats[3] += total_popped.sum()
            self.stats[4] += self.num_envs
            self.stats[5] += rewards.sum()

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (
            self._get_obs().cpu().numpy(),
            rewards.cpu().numpy(),
            dones.cpu().numpy(),
            self.shared_infos,
        )

    # ------------------------------------------------------------------
    # Action Masking  ← the critical new piece
    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """
        Compute a boolean action mask (N_envs, N_actions) entirely on GPU.

        Strategy
        --------
        For every pre-computed in-bounds swap pair p=(r1,c1,r2,c2):
          1. Expand self.grids to (N, K, G, G) for a chunk of K pairs.
          2. Apply all K swaps via advanced indexing (no Python loop over swaps).
          3. Flatten to (N*K, G, G) and run _find_matches_any_fast() which
             uses plain equality comparisons — no one-hot, minimal VRAM.
          4. Scatter results back into the (N, A) mask tensor.

        Dead boards are refreshed in-place so the mask is always non-empty.
        """
        N = self.num_envs
        P = self.swap_action_ids.shape[0]
        G = self.grid_size
        C = self.mask_chunk_size

        masks = torch.zeros((N, self.n_actions), dtype=torch.bool, device=self.device)

        for start in range(0, P, C):
            end = min(start + C, P)
            K   = end - start

            r1s  = self.swap_r1s [start:end]   # (K,)
            c1s  = self.swap_c1s [start:end]
            r2s  = self.swap_r2s [start:end]
            c2s  = self.swap_c2s [start:end]
            aids = self.swap_action_ids[start:end]

            # Expand grids: (N,G,G) → (N,K,G,G) [cloned for in-place swap]
            exp = self.grids.unsqueeze(1).expand(N, K, G, G).clone()  # (N,K,G,G)

            # Advanced indexing: k_idx[i]==i, so exp[:,k_idx,r1s,c1s] == exp[:,i,r1s[i],c1s[i]]
            k_idx = torch.arange(K, device=self.device)
            src = exp[:, k_idx, r1s, c1s].clone()  # (N,K)
            dst = exp[:, k_idx, r2s, c2s].clone()  # (N,K)
            exp[:, k_idx, r1s, c1s] = dst
            exp[:, k_idx, r2s, c2s] = src

            # Flatten → (N*K, G, G)
            flat = exp.view(N * K, G, G)

            # Fast match detection (no one-hot)
            has_match = self._find_matches_any_fast(flat)  # (N*K,)
            has_match = has_match.view(N, K)                # (N,K)

            masks[:, aids] = has_match

        # ── Dead-board handling ───────────────────────────────────────
        dead = ~masks.any(dim=1)   # (N,) — True where no valid move exists
        if dead.any():
            dead_idx = dead.nonzero(as_tuple=True)[0]
            self._reset_specific_grids(dead_idx)
            # Allow all in-bounds swaps for refreshed boards
            masks[dead_idx.unsqueeze(1), self.swap_action_ids.unsqueeze(0)] = True

        return masks.cpu().numpy()

    # ------------------------------------------------------------------
    # Fast match detection (masking path — no one-hot)
    # ------------------------------------------------------------------
    def _find_matches_any_fast(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Pure equality-comparison match detector.
        grids : (B, H, W)  int64
        returns: (B,)  bool  — True if any 3-in-a-row/column exists
        """
        B, H, W = grids.shape
        result = torch.zeros(B, dtype=torch.bool, device=self.device)

        if W >= 3:
            eq_h  = (grids[:, :, :-1] == grids[:, :, 1:])         # (B,H,W-1)
            run_h = eq_h[:, :, :-1] & eq_h[:, :, 1:]               # (B,H,W-2)
            result |= run_h.reshape(B, -1).any(1)

        if H >= 3:
            eq_v  = (grids[:, :-1, :] == grids[:, 1:, :])         # (B,H-1,W)
            run_v = eq_v[:, :-1, :] & eq_v[:, 1:, :]               # (B,H-2,W)
            result |= run_v.reshape(B, -1).any(1)

        return result

    # ------------------------------------------------------------------
    # Conv-based match mask (cascade path — needs per-cell info)
    # ------------------------------------------------------------------
    def _find_matches_mask(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, H, W) bool: True for every cell that is part of a match.
        Uses depthwise conv on one-hot encoding for exact cell locations.
        Only called during cascade resolution, not in action_masks().
        """
        one_hot = F.one_hot(grids, num_classes=self.n_tiles).permute(0, 3, 1, 2).float()

        mh = F.conv2d(one_hot, self.k_h, padding=(0, 1), groups=self.n_tiles)
        mv = F.conv2d(one_hot, self.k_v, padding=(1, 0), groups=self.n_tiles)

        fh = F.conv2d((mh >= 2.9).float(), self.k_h, padding=(0, 1), groups=self.n_tiles)
        fv = F.conv2d((mv >= 2.9).float(), self.k_v, padding=(1, 0), groups=self.n_tiles)

        return ((fh + fv) > 0.9).max(dim=1).values  # (B,H,W)

    # ------------------------------------------------------------------
    # Gravity + refill
    # ------------------------------------------------------------------
    def _apply_gravity_and_refill(self, grids: torch.Tensor, active_mask):
        keys = (grids != -1).long()
        _, indices = torch.sort(keys, dim=1, stable=True)
        grids_sorted = torch.gather(grids, 1, indices)

        empty_mask = grids_sorted == -1
        if empty_mask.any():
            rand_tiles = torch.randint(0, self.n_tiles, grids.shape, device=self.device)
            grids_sorted = torch.where(empty_mask, rand_tiles, grids_sorted)

        grids[:] = grids_sorted

    # ------------------------------------------------------------------
    # Grid initialisation helpers
    # ------------------------------------------------------------------
    def _reset_all_grids(self):
        self.grids = torch.randint(
            0, self.n_tiles,
            (self.num_envs, self.grid_size, self.grid_size),
            device=self.device,
        )
        for _ in range(20):
            matches = self._find_matches_mask(self.grids)
            if not matches.view(self.num_envs, -1).any(1).any():
                break
            self.grids[matches] = -1
            self._apply_gravity_and_refill(self.grids, None)

    def _reset_specific_grids(self, env_indices: torch.Tensor):
        """Re-generate boards only for the given env indices."""
        n = env_indices.shape[0]
        new_grids = torch.randint(
            0, self.n_tiles,
            (n, self.grid_size, self.grid_size),
            device=self.device,
        )
        for _ in range(20):
            matches = self._find_matches_mask(new_grids)
            if not matches.view(n, -1).any(1).any():
                break
            new_grids[matches] = -1
            self._apply_gravity_and_refill(new_grids, None)
        self.grids[env_indices] = new_grids

    # ------------------------------------------------------------------
    # Pre-compute all in-bounds swap pairs
    # ------------------------------------------------------------------
    def _precompute_swap_pairs(self):
        """
        For every (row, col, direction) that stays in-bounds, store:
          action_id  = (row*G + col)*4 + dir
          r1, c1     = source cell
          r2, c2     = destination cell
        Returns five GPU tensors of shape (P,).
        """
        G = self.grid_size
        action_ids, r1s, c1s, r2s, c2s = [], [], [], [], []

        for r in range(G):
            for c in range(G):
                for d, (dr, dc) in DIRS.items():
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < G and 0 <= c2 < G:
                        action_ids.append((r * G + c) * 4 + d)
                        r1s.append(r);  c1s.append(c)
                        r2s.append(r2); c2s.append(c2)

        mk = lambda lst: torch.tensor(lst, dtype=torch.long, device=self.device)
        return mk(action_ids), mk(r1s), mk(c1s), mk(r2s), mk(c2s)

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------
    def get_stats(self):
        s = self.stats.cpu().numpy()
        return {
            "match":       int(s[0]),
            "no_match":    int(s[1]),
            "illegal":     int(s[2]),
            "popped":      int(s[3]),
            "total_moves": int(s[4]),
            "total_reward": float(s[5]),
        }

    def reset_stats(self):
        self.stats.zero_()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        return F.one_hot(self.grids, num_classes=self.n_tiles).permute(0, 3, 1, 2).float()

    # ------------------------------------------------------------------
    # Required VecEnv stubs
    # ------------------------------------------------------------------
    def seed(self, seed=None):
        pass

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, **method_kwargs):
        """Standard SB3 VecEnv method caller."""
        if method_name == "action_masks":
            # MaskablePPO on a VecEnv calls env_method("action_masks") 
            # and then np.stack() on the result.
            masks = self.action_masks() # (N, A)
            return [masks[i] for i in range(self.num_envs)]
        
        # For any other method, try to call it on self
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
