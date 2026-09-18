"""Evaluate agents: self-play score + cross-play matrix + ELO.

Usage:
    # self-play score of one checkpoint
    python evaluate.py --layout cramped_room --model checkpoints/x.zip

    # cross-play matrix across a whole population
    python evaluate.py --layout cramped_room --population population/cramped_room --episodes 20
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO

from gym_env import OvercookedSimEnv
from population import Population, PolicyAgent


def eval_pair(layout, focal, partner, episodes=10, horizon=400, seed=0):
    """focal + partner cooperate; returns mean sparse reward."""
    env = OvercookedSimEnv(layout, horizon=horizon, agent_index=0, seed=seed)
    env.set_partner_policy(partner)
    rng = np.random.default_rng(seed)
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        total = 0.0
        for _ in range(horizon):
            a = focal(obs)
            obs, r, done, trunc, info = env.step(a)
            total += info.get("sparse_reward", 0)
            if done or trunc:
                break
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores)), scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="cramped_room")
    p.add_argument("--model", default=None)
    p.add_argument("--population", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.model:
        focal = PolicyAgent.from_sb3_model(args.model, name="focal",
                                           deterministic=True)
        # self-play: partner = same weights, stochastic
        partner = PolicyAgent.from_sb3_model(args.model, name="partner")
        mean, std, scores = eval_pair(args.layout, focal, partner,
                                      args.episodes, args.horizon, args.seed)
        print(f"self-play sparse reward over {args.episodes} eps: "
              f"mean={mean:.1f} std={std:.1f}\n  scores={scores}")
        return

    if args.population:
        pop = Population(args.population)
        names = pop.names()
        print(f"cross-play matrix ({args.episodes} eps per cell): {names}")
        agents = {n: pop.load_agent(n, deterministic=True) for n in names}
        matrix = np.zeros((len(names), len(names)))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                mean, _, _ = eval_pair(args.layout, agents[a], agents[b],
                                       args.episodes, args.horizon,
                                       args.seed + i * 1000 + j)
                matrix[i, j] = mean
        header = "          " + "".join(f"{n[:9]:>10}" for n in names)
        print(header)
        for i, a in enumerate(names):
            print(f"{a[:9]:>10}" + "".join(f"{matrix[i,j]:>10.1f}"
                                           for j in range(len(names))))
        # cross-play generalization: mean off-diagonal
        offdiag = [matrix[i, j] for i in range(len(names))
                   for j in range(len(names)) if i != j]
        if offdiag:
            print(f"\ncross-play (off-diagonal) mean: {np.mean(offdiag):.1f}")
            print(f"self-play (diagonal) mean:      {np.mean(np.diag(matrix)):.1f}")
        np.save(os.path.join(args.population, "crossplay_matrix.npy"), matrix)


if __name__ == "__main__":
    main()
