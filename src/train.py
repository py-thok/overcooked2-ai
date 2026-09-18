"""Train two PPO agents against each other in the *real* Overcooked! 2.

Usage:
  python src/train.py [--config configs/default.json] [--total-timesteps N]

Because the real game is a single instance, the two agents share the
same environment: agent A controls WASD-style keys for chef 1, agent B
controls the arrow-key chef 2.  Both observe the same stacked frames.
Rewards are shared (score changes benefit both), which encourages
cooperation -- exactly what you want for score farming.

NOTE: This is deliberately simple.  For serious training, use the
Overcooked-AI simulator first, then fine-tune here.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from env import OvercookedEnv

# Chef-2 uses arrow keys + right-ctrl / right-shift style binds.
CHEF2_KEYS = {
    "up": "up", "down": "down", "left": "left", "right": "right",
    "interact": "ctrl", "chop_throw": "shift",
}


class DualAgentEnv(OvercookedEnv):
    """Wraps OvercookedEnv so a single PPO policy controls *both* chefs.

    The action space becomes Discrete(N_ACTIONS ** 2) -- the joint action
    of (chef1, chef2).  Self-play with a shared policy is a solid default
    for cooperative score farming.
    """

    def __init__(self, *args, chef2_cfg=None, **kw):
        super().__init__(*args, **kw)
        from screen_io import ScreenController, N_ACTIONS, ACTIONS
        self._n = N_ACTIONS
        self._actions = ACTIONS
        self.action_space = __import__("gymnasium").spaces.Discrete(N_ACTIONS ** 2)
        chef2_cfg = chef2_cfg or {}
        self.ctrl2 = ScreenController(
            monitor=chef2_cfg.get("monitor", 1),
            capture_size=tuple(chef2_cfg.get("capture_size", (1280, 720))),
            resize_to=tuple(chef2_cfg.get("resize_to", (320, 180))),
            keys=CHEF2_KEYS,
            action_repeat_ms=chef2_cfg.get("action_repeat_ms", 100),
        )

    def step(self, joint_action):
        a1 = joint_action // self._n
        a2 = joint_action % self._n
        # Apply both chefs' actions before stepping the world.
        self.ctrl.act(int(a1))
        self.ctrl2.act(int(a2))
        # Reuse parent logic for observation/reward but skip its act().
        if self.step_delay:
            time.sleep(self.step_delay)
        frame = self.ctrl.grab()
        obs = self._obs(frame)
        self._steps += 1
        reward = 0.0
        sig = self._score_signature(self._frames[-1])
        if sig is not None and self._prev_score_sig is not None and sig != self._prev_score_sig:
            reward = 1.0
        self._prev_score_sig = sig
        terminated = False
        truncated = self._steps >= self.max_steps
        if self.episode_time_s and (time.time() - self._t0) > self.episode_time_s:
            truncated = True
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.ctrl2.close()
        super().close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--total-timesteps", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    total = args.total_timesteps or cfg["train"]["total_timesteps"]
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    def make_env():
        return DualAgentEnv(
            screen_cfg=cfg["screen"],
            control_cfg=cfg["control"],
            agent_cfg=cfg["agent"],
            score_roi=cfg.get("score_roi"),
            episode_time_s=cfg.get("episode_time_s"),
        )

    env = DummyVecEnv([make_env])
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="logs",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    cb = CheckpointCallback(save_freq=cfg["train"]["checkpoint_every_steps"],
                            save_path=save_dir, name_prefix="ppo_overcooked")
    model.learn(total_timesteps=total, callback=cb, progress_bar=True)
    model.save(os.path.join(save_dir, "ppo_overcooked_final"))
    env.close()


if __name__ == "__main__":
    main()
