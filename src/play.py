"""Run a trained PPO policy in the real game (or manual playtest).

Usage:
  python src/play.py --model checkpoints/ppo_overcooked_final.zip
"""
import argparse
import time

from stable_baselines3 import PPO
from train import DualAgentEnv  # reuses config-free env construction
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="configs/default.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    env = DualAgentEnv(
        screen_cfg=cfg["screen"],
        control_cfg=cfg["control"],
        agent_cfg=cfg["agent"],
        score_roi=cfg.get("score_roi"),
        episode_time_s=cfg.get("episode_time_s"),
    )
    model = PPO.load(args.model)
    obs, _ = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(int(action))
            if term or trunc:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
