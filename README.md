# Overcooked2-RL

Train two agents to rack up scores in **Overcooked! 2** using reinforcement learning.

## Roadmap

```
Phase 1: Overcooked-AI simulator ──► Phase 2: FCP population training ──► Phase 3: Real-game transfer
        PPO self-play baseline        Cross-partner generalization       Visual policy sim-to-real
```

## Environment

```bash
conda activate overcooked   # python 3.10, torch 2.6+cu124
```

## Phase 1: Simulator Baseline (src/sim/)

```bash
cd src/sim
python train_sp.py --layout cramped_room --timesteps 5000000 --n-envs 32
```

- `gym_env.py` — Gymnasium wrapper (96-dim featurized observations, ~5k steps/s/core)
- `train_sp.py` — PPO self-play (partner is periodically synced to a snapshot of the current policy)

## Phase 2: Population Training (FCP)

```bash
# 1) Build the population: self-play agents from multiple seeds + a random agent
python seed_population.py --layout cramped_room --n-seeds 4 --timesteps 3000000

# 2) FCP training: the protagonist trains against teammates sampled from the population
python train_fcp.py --layout cramped_room --timesteps 5000000

# 3) Evaluation: self-play score / cross-play matrix
python evaluate.py --layout cramped_room --model checkpoints/sim_sp/xxx.zip
python evaluate.py --layout cramped_room --population population/cramped_room --episodes 20
```

Key metric: **cross-play (paired with unseen teammates) score / self-play score ≥ 80%**.

- `population.py` — Population management (checkpoint registry, sampling, ELO)
- `train_fcp.py` — FCP trainer
- `evaluate.py` — Self-play / cross-play evaluation

## Phase 3: Real Game (existing code, to be integrated)

- `src/screen_io.py`, `src/env.py` — Real-game environment via screen capture + keyboard control
- `src/train.py`, `src/play.py` — Real-game PPO / inference

## Monitoring

```bash
tail -f logs/cramped_room_sp.log
tensorboard --logdir logs/tb
```
