# Overcooked-Agent

用强化学习在 **Overcooked! 2** 中训练双 agent 刷分。

## 路线

```
阶段1: Overcooked-AI 模拟器 ──► 阶段2: FCP 种群训练 ──► 阶段3: 真机迁移
        PPO self-play 基线        跨队友泛化              视觉策略 sim-to-real
```

## 环境

```bash
conda activate overcooked   # python 3.10, torch 2.6+cu124
```

## 阶段 1：模拟器基线（src/sim/）

```bash
cd src/sim
python train_sp.py --layout cramped_room --timesteps 5000000 --n-envs 32
```

- `gym_env.py` — Gymnasium 封装（96 维 featurized 观测，~5k steps/s/核）
- `train_sp.py` — PPO self-play（partner 定期同步为当前策略快照）

## 阶段 2：种群训练（FCP）

```bash
# 1) 生成种群：多个 seed 的自对弈 agent + random
python seed_population.py --layout cramped_room --n-seeds 4 --timesteps 3000000

# 2) FCP 训练：主角 vs 种群中采样的队友
python train_fcp.py --layout cramped_room --timesteps 5000000

# 3) 评估：self-play 分数 / cross-play 矩阵
python evaluate.py --layout cramped_room --model checkpoints/sim_sp/xxx.zip
python evaluate.py --layout cramped_room --population population/cramped_room --episodes 20
```

核心指标：**cross-play（与陌生队友配对）得分 / self-play 得分 ≥ 80%**。

- `population.py` — 种群管理（checkpoint 注册、采样、ELO）
- `train_fcp.py` — FCP 训练器
- `evaluate.py` — self-play / cross-play 评估

## 阶段 3：真机（原有代码，待接入）

- `src/screen_io.py`、`src/env.py` — 截屏 + 键盘控制真机环境
- `src/train.py`、`src/play.py` — 真机 PPO / 推理

## 监控

```bash
tail -f logs/cramped_room_sp.log
tensorboard --logdir logs/tb
```
