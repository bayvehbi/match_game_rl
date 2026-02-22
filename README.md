# Match-3 GPU RL

A high-performance Reinforcement Learning project for Match-3 games.

## Why it is good
- **Massive Scale**: Fully GPU-vectorized. Simulates over 20,000 game steps per second.
- **Strategic AI**: Uses Action Masking and exponential reward multipliers to prioritize complex cascades and high-score moves.
- **Fast Training**: Processes 10M+ steps in minutes on consumer GPUs.

## Usage
```bash
python train.py
```
To play manually or test the environment:
```bash
python play_gpu_interactive.py
```
