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

## 训练完成后待办 (Post-training checklist)

`src/sim/train_sp.py` 的 5M 步 self-play 训练跑完后，按顺序做：

### 1. 查看脚本末尾自动打印的 eval 结果
训练结束时脚本会自动跑 10 局 deterministic self-play eval 并打印：
```
eval sparse reward (10 ep, deterministic): mean=X std=Y all=[...]
```
- cramped_room 参考基准：随机策略 ~0-5 分；训练良好的 self-play PPO 应达 **100+ 分**
- **2026-09-19 中间检查**：20k 步 checkpoint 实测 20 局 mean=0.0，早期策略尚未学会上菜，属正常（稀疏奖励任务早期普遍如此）；但需确认后期是否爬起来

### 2. 若分数仍然接近 0（疑似未学会）
- 检查 `logs/cramped_room_sp/progress.csv` 中 `env/sparse_reward_rate` —— **注意：该指标当前是坏的**（只统计非零奖励步的均值，恒为 20，不反映上菜频率），不要用它判断学习进度
- 修复 `RewardLogCallback`：改为统计每局 sparse 总和，或给 env 包 `Monitor` 让 SB3 记录 `rollout/ep_rew_mean`
- 检查 `checkpoints/sim_sp/` 是否有 20k 之后的新 checkpoint（截至 2026-09-19 只有 ppo_sp_cramped_room_20000.zip，之后的保存疑似未生效，需排查 CheckpointCallback save_freq）
- 考虑启用 reward shaping（`rew_shaping_horizon`）或改用 `train_fcp.py`，纯稀疏奖励在 cramped_room 上样本效率极低

### 3. 若分数正常（100+）
- 用最终权重跑 self-play eval 确认稳定性（可多跑几个 seed 的 20 局）
- 作为 FCP（fictitious co-play）的 population 种子，启动 `train_fcp.py`
- 最终目标：迁移到真实游戏 `src/train.py` 微调前，先在模拟器里验证人机协作表现
