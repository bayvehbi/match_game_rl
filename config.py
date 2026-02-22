# ─── Game Settings ────────────────────────────────────────────────────────────
GRID_SIZE = 8          # Board is GRID_SIZE x GRID_SIZE
N_TILES   = 5          # Number of distinct tile colours

DIRS = {
    0: (-1, 0),  # up
    1: ( 1, 0),  # down
    2: ( 0,-1),  # left
    3: ( 0, 1),  # right
}

# ─── Training Settings ────────────────────────────────────────────────────────
NUM_ENVS        = 512          # Parallel GPU environments
TOTAL_STEPS     = 30_000_000   # Total training timesteps
N_STEPS         = 512          # PPO rollout steps per update
BATCH_SIZE      = 4096         # Mini-batch size
N_EPOCHS        = 4            # PPO update epochs
LR              = 3e-4         # Learning rate
GAMMA           = 0.99         # Discount factor
GAE_LAMBDA      = 0.95         # GAE lambda
CLIP_RANGE      = 0.2          # PPO clip range
ENT_COEF        = 0.01         # Entropy coefficient
VF_COEF         = 0.5          # Value function coefficient
MAX_GRAD_NORM   = 0.5          # Gradient clipping

# ─── Environment Settings ─────────────────────────────────────────────────────
MAX_STEPS_PER_EP = 1000        # Steps before episode resets (uses TimeLimit wrapper)
MASK_CHUNK_SIZE  = 32          # Swap pairs processed per chunk in action_masks()

# ─── Reward Settings ──────────────────────────────────────────────────────────
REWARD_MATCH    =  16.0
REWARD_NO_MATCH =  -3.0
REWARD_ILLEGAL  =  -4.0
REWARD_SHAPING_MATCH   =  10.0
REWARD_SHAPING_NOMATCH =  -3.0
REWARD_COMBO_EXPONENT  =  1.2    # Power exponent for non-linear rewards (total_popped ^ exponent)

# ─── Logging & Saving ─────────────────────────────────────────────────────────
LOG_DIR          = "./logs"
MODEL_DIR        = "./models"
LOG_INTERVAL     = 20_000      # Steps between console logs
SAVE_INTERVAL    = 200_000     # Steps between model saves
