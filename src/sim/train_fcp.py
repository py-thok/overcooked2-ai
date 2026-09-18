"""FCP (Fictitious Co-Play) training: train a focal agent against partners
sampled from a population (Strouse et al. 2021), maximizing cross-play score.

Usage:
    python train_fcp.py --layout cramped_room --timesteps 5000000 \
        --population population/cramped_room
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from gym_env import OvercookedSimEnv
from population import Population


def make_env(layout, horizon, rank, seed):
    def _init():
        return OvercookedSimEnv(layout_name=layout, horizon=horizon,
                                partner_policy=None, agent_index=rank % 2,
                                seed=seed + rank)
    return _init


class FCPPartnerCallback(BaseCallback):
    """Re-sample partners from the population every `resample_every` steps."""

    def __init__(self, population: Population, resample_every=25_000,
                 seed=0, verbose=0):
        super().__init__(verbose)
        self.population = population
        self.resample_every = resample_every
        self.rng = np.random.default_rng(seed)
        self._last = -1
        self.partner_counts = {}

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.resample_every:
            n_envs = self.training_env.num_envs
            for i in range(n_envs):
                agent = self.population.sample_agent(
                    seed=int(self.rng.integers(2**31)))
                self.training_env.env_method("set_partner_policy", agent,
                                             indices=[i])
                self.partner_counts[agent.name] = \
                    self.partner_counts.get(agent.name, 0) + 1
            self._last = self.num_timesteps
            if self.verbose:
                print(f"[fcp] resampled partners @ {self.num_timesteps}: "
                      f"{self.partner_counts}")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="cramped_room")
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--population", default="population/cramped_room")
    p.add_argument("--resample-every", type=int, default=25_000)
    p.add_argument("--init-from", default=None,
                   help="optional checkpoint to initialize focal agent")
    p.add_argument("--save-dir", default="checkpoints/sim_fcp")
    p.add_argument("--name", default=None, help="run name (default auto)")
    p.add_argument("--device", default="cuda:2")
    args = p.parse_args()

    name = args.name or f"fcp_{args.layout}_seed{args.seed}"
    os.makedirs(args.save_dir, exist_ok=True)

    pop = Population(args.population)
    print(f"population members: {pop.names()}")

    env = SubprocVecEnv([make_env(args.layout, args.horizon, i, args.seed)
                         for i in range(args.n_envs)])

    if args.init_from:
        model = PPO.load(args.init_from, env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=max(2048 // args.n_envs, 64),
            batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1, seed=args.seed, device=args.device,
            tensorboard_log="logs/tb",
        )
    model.set_logger(configure(f"logs/{name}", ["stdout", "csv", "tensorboard"]))

    ckpt_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=args.save_dir, name_prefix=name)
    model.learn(total_timesteps=args.timesteps,
                callback=[FCPPartnerCallback(pop, args.resample_every,
                                             seed=args.seed, verbose=1),
                          ckpt_cb],
                tb_log_name=name)

    path = os.path.join(args.save_dir, f"{name}.zip")
    model.save(path)
    print(f"saved {path}")

    # register into population for future runs
    pop.add(path, name, kind="fcp", notes=f"trained vs pop of {len(pop.names())}")
    print(f"added '{name}' to population")


if __name__ == "__main__":
    main()
