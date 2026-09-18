"""Self-play PPO baseline on the Overcooked-AI simulator.

Both chefs share the same policy (classic self-play from the NeurIPS paper).
Partner policies inside vectorized envs are periodically refreshed from the
latest model weights.

Usage:
    python train_sp.py --layout cramped_room --timesteps 5000000
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from gym_env import OvercookedSimEnv, N_ACTIONS


def make_env(layout, horizon, rank, seed):
    def _init():
        env = OvercookedSimEnv(layout_name=layout, horizon=horizon,
                               partner_policy=None, agent_index=rank % 2,
                               seed=seed + rank)
        return env
    return _init


class PicklablePolicyPartner:
    """Frozen snapshot of an SB3 policy, CPU-only, picklable for SubprocVecEnv."""

    def __init__(self, policy, deterministic=False):
        import copy
        self.policy = copy.deepcopy(policy).to("cpu")
        state_dict = {k: v.detach().cpu().clone()
                      for k, v in policy.state_dict().items()}
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        self.deterministic = deterministic

    def __call__(self, obs):
        with th.no_grad():
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            if self.deterministic:
                return int(self.policy.predict(obs_t, deterministic=True)[0][0])
            return int(self.policy.get_distribution(obs_t).sample().item())

    def __getstate__(self):
        # keep only weights + config; rebuild a fresh MlpPolicy on unpickle
        return {
            "state_dict": self.policy.state_dict(),
            "obs_dim": self.policy.observation_space.shape[0],
            "deterministic": self.deterministic,
        }

    def __setstate__(self, state):
        from stable_baselines3.common.policies import ActorCriticPolicy
        from gymnasium import spaces
        import torch as th
        obs_dim = state["obs_dim"]
        self.policy = ActorCriticPolicy(
            observation_space=spaces.Box(-np.inf, np.inf, shape=(obs_dim,),
                                         dtype=np.float32),
            action_space=spaces.Discrete(N_ACTIONS),
            lr_schedule=lambda _: 3e-4,
            net_arch=[256, 256],
        )
        self.policy.load_state_dict(state["state_dict"])
        self.policy.eval()
        self.deterministic = state["deterministic"]


class SelfPlayCallback(BaseCallback):
    """Periodically sync the current policy into all envs as the partner."""

    def __init__(self, sync_every_steps=50_000, deterministic_partner=False,
                 verbose=0):
        super().__init__(verbose)
        self.sync_every = sync_every_steps
        self.deterministic = deterministic_partner
        self._last_sync = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_sync >= self.sync_every:
            partner = PicklablePolicyPartner(self.model.policy,
                                             deterministic=self.deterministic)
            self.training_env.env_method("set_partner_policy", partner)
            self._last_sync = self.num_timesteps
            if self.verbose:
                print(f"[self-play] synced partner @ {self.num_timesteps}")
        return True


class RewardLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.sparse, self.shaped, self.count = [], [], 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode_sparse_reward" in info and ("done" in info or True):
                pass
        # SB3 puts terminal info under "episode" for Monitor-wrapped envs only;
        # we track sparse reward directly from infos
        for info in self.locals["infos"]:
            if info.get("sparse_reward"):
                self.sparse.append(info["sparse_reward"])
        if len(self.sparse) >= 20:
            self.logger.record("env/sparse_reward_rate", np.mean(self.sparse[-100:]))
            self.sparse = []
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="cramped_room")
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="checkpoints/sim_sp")
    p.add_argument("--sync-every", type=int, default=50_000)
    p.add_argument("--device", default="cuda:2")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(args.layout, args.horizon, i, args.seed)
                         for i in range(args.n_envs)])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048 // args.n_envs if 2048 // args.n_envs > 0 else 128,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log="logs/tb",
    )
    model.set_logger(configure(f"logs/{args.layout}_sp", ["stdout", "csv", "tensorboard"]))

    ckpt_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=args.save_dir,
        name_prefix=f"ppo_sp_{args.layout}_seed{args.seed}")
    cb = [SelfPlayCallback(sync_every_steps=args.sync_every, verbose=1),
          RewardLogCallback(), ckpt_cb]

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb,
                tb_log_name=f"sp_{args.layout}")
    elapsed = time.time() - t0

    path = os.path.join(args.save_dir, f"ppo_sp_{args.layout}_seed{args.seed}_{args.timesteps}.zip")
    model.save(path)
    print(f"saved {path} | {args.timesteps/elapsed:.0f} steps/s")

    # quick eval vs itself
    from stable_baselines3.common.evaluation import evaluate_policy
    eval_env = OvercookedSimEnv(args.layout, args.horizon)
    obs_dim = eval_env.observation_space.shape[0]
    def partner_fn(obs):
        a, _ = model.predict(obs, deterministic=False)
        return int(a)
    eval_env.set_partner_policy(partner_fn)
    rews = []
    for _ in range(10):
        obs, _ = eval_env.reset()
        ep_sparse = 0
        for _ in range(args.horizon):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, tr, info = eval_env.step(int(a))
            ep_sparse += info.get("sparse_reward", 0)
            if d or tr: break
        rews.append(ep_sparse)
    print(f"eval sparse reward (10 ep, deterministic): mean={np.mean(rews):.1f} "
          f"std={np.std(rews):.1f} all={rews}")


if __name__ == "__main__":
    main()
