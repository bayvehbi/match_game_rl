"""
train.py
========
Train a Match-3 agent with MaskablePPO (sb3-contrib).

Usage
-----
    python train.py                        # defaults from config.py
    python train.py --grid 6 --tiles 4     # 6x6 board, 4 tile types
    python train.py --envs 256 --steps 5000000
    python train.py --resume models/best   # continue from checkpoint
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

# ── SB3 / sb3-contrib ─────────────────────────────────────────────────────────
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    sys.exit(
        "sb3-contrib not found.\n"
        "Install with:  pip install sb3-contrib"
    )

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecMonitor

import torch.nn as nn
import gymnasium as gym

from match_gpu_env import MatchVecEnv
import config as C


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CNN Feature Extractor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MatchCNN(BaseFeaturesExtractor):
    """
    Convolutional feature extractor for the one-hot grid observation.

    Input shape : (n_tiles, grid_size, grid_size)  — float32
    Output      : flat vector of `features_dim` floats
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_tiles, h, w = observation_space.shape

        self.cnn = nn.Sequential(
            # Spatial feature extraction
            nn.Conv2d(n_tiles, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flat size after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, n_tiles, h, w)
            flat_size = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(obs))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logging Callback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StatsCallback(BaseCallback):
    """
    Periodically reads the GPU stat counters from MatchVecEnv and logs them.
    Avoids per-step Python overhead — stats stay on GPU between reads.
    """

    def __init__(self, log_interval: int = C.LOG_INTERVAL, save_interval: int = C.SAVE_INTERVAL,
                 model_dir: str = C.MODEL_DIR, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval  = log_interval
        self.save_interval = save_interval
        self.model_dir     = model_dir
        self._last_log     = 0
        self._last_save    = 0
        self._t0           = time.time()
        os.makedirs(model_dir, exist_ok=True)

        # CSV log
        os.makedirs(C.LOG_DIR, exist_ok=True)
        self._csv = open(os.path.join(C.LOG_DIR, "training_metrics.csv"), "w")
        self._csv.write("timestep,match_rate,illegal_rate,match_per_move,reward_per_move,steps_per_sec\n")

    def _on_step(self) -> bool:
        ts = self.num_timesteps

        if ts - self._last_log >= self.log_interval:
            env: MatchVecEnv = self.training_env.unwrapped  # raw GPU env
            stats = env.get_stats()
            env.reset_stats()

            total = max(stats["total_moves"], 1)
            match_rate   = stats["match"]   / total
            illegal_rate = stats["illegal"] / total
            reward_move  = stats["total_reward"] / total
            elapsed      = time.time() - self._t0
            sps          = (ts - self._last_log) / max(elapsed - getattr(self, "_last_elapsed", 0), 1e-6)

            if self.verbose:
                print(
                    f"[{ts:>10,}]  match={match_rate:.2%}  illegal={illegal_rate:.2%}"
                    f"  popped/move={stats['popped']/total:.2f}  rew/move={reward_move:.1f}  SPS={sps:,.0f}"
                )

            self._csv.write(f"{ts},{match_rate:.4f},{illegal_rate:.4f},"
                            f"{stats['popped']/total:.4f},{reward_move:.4f},{sps:.0f}\n")
            self._csv.flush()

            self._last_log     = ts
            self._last_elapsed = elapsed

        if ts - self._last_save >= self.save_interval:
            path = os.path.join(self.model_dir, f"checkpoint_{ts}")
            self.model.save(path)
            if self.verbose:
                print(f"  ✓ Saved checkpoint → {path}.zip")
            self._last_save = ts

        return True

    def _on_training_end(self):
        self._csv.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mask function shim (required by ActionMasker wrapper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _mask_fn(env):
    """Called by ActionMasker; env here is the MatchVecEnv itself."""
    return env.action_masks()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser(description="Train Match-3 agent with MaskablePPO")
    p.add_argument("--grid",    type=int,   default=C.GRID_SIZE,    help="Board size NxN")
    p.add_argument("--tiles",   type=int,   default=C.N_TILES,      help="Number of tile types")
    p.add_argument("--envs",    type=int,   default=C.NUM_ENVS,     help="Parallel GPU environments")
    p.add_argument("--steps",   type=int,   default=C.TOTAL_STEPS,  help="Total training timesteps")
    p.add_argument("--lr",      type=float, default=C.LR,           help="Learning rate")
    p.add_argument("--device",  type=str,   default="cuda",         help="cuda or cpu")
    p.add_argument("--resume",  type=str,   default=None,           help="Path to checkpoint .zip (no ext)")
    p.add_argument("--chunk",   type=int,   default=C.MASK_CHUNK_SIZE, help="Mask computation chunk size")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"  Match-3 RL Training  —  MaskablePPO + GPU env")
    print(f"  Grid: {args.grid}x{args.grid}   Tiles: {args.tiles}   Envs: {args.envs}")
    print(f"  Device: {args.device}   Steps: {args.steps:,}")
    print("=" * 60)

    # ── Build vectorised GPU environment ──────────────────────────────
    env = MatchVecEnv(
        num_envs=args.envs,
        device=args.device,
        grid_size=args.grid,
        n_tiles=args.tiles,
        mask_chunk_size=args.chunk,
    )

    # MaskablePPO requires the vec_env to expose action_masks() directly.
    # MatchVecEnv already has this method, so we wrap with VecMonitor only.
    env = VecMonitor(env)

    # ── Policy kwargs ─────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=MatchCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=nn.ReLU,
    )

    # ── Build or resume model ─────────────────────────────────────────
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)
        model.learning_rate = args.lr
    else:
        model = MaskablePPO(
            policy          = "CnnPolicy",
            env             = env,
            n_steps         = C.N_STEPS,
            batch_size      = C.BATCH_SIZE,
            n_epochs        = C.N_EPOCHS,
            learning_rate   = args.lr,
            gamma           = C.GAMMA,
            gae_lambda      = C.GAE_LAMBDA,
            clip_range      = C.CLIP_RANGE,
            ent_coef        = C.ENT_COEF,
            vf_coef         = C.VF_COEF,
            max_grad_norm   = C.MAX_GRAD_NORM,
            policy_kwargs   = policy_kwargs,
            verbose         = 0,       # we use our own callback
            device          = args.device,
            tensorboard_log = C.LOG_DIR,
        )

    print(f"\n  Policy parameters : {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"  Action space      : {env.action_space.n}")
    print(f"  Observation space : {env.observation_space.shape}\n")

    # ── Callbacks ────────────────────────────────────────────────────
    callbacks = CallbackList([
        StatsCallback(
            log_interval  = C.LOG_INTERVAL,
            save_interval = C.SAVE_INTERVAL,
            model_dir     = C.MODEL_DIR,
        ),
    ])

    # ── Train ─────────────────────────────────────────────────────────
    model.learn(
        total_timesteps    = args.steps,
        callback           = callbacks,
        use_masking        = True,       # ← enable action masking in MaskablePPO
        reset_num_timesteps= args.resume is None,
    )

    # ── Save final model ──────────────────────────────────────────────
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    final_path = os.path.join(C.MODEL_DIR, "final_model")
    model.save(final_path)
    print(f"\n  ✓ Final model saved → {final_path}.zip")


if __name__ == "__main__":
    main()
