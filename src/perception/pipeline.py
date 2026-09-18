"""End-to-end plan-B pipeline: screenshot -> 96-dim vector -> MLP action.

Usage (simulation validation, no X display needed):
    python pipeline.py --backend sim --steps 50

Usage (real game, requires calibrated RealDetector + display):
    python pipeline.py --backend real --checkpoint <path>
"""
import sys, argparse, time
sys.path.insert(0, "/root/data/overcooked2-ai/src/sim")
sys.path.insert(0, "/root/data/overcooked2-ai/src/perception")

import numpy as np
from feature_builder import FeatureBuilder
from detectors import SimDetector


def run_sim(steps, checkpoint):
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as OAIEnv
    from stable_baselines3 import PPO

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OAIEnv.from_mdp(mdp, horizon=400, info_level=0)
    fb = FeatureBuilder()
    det = SimDetector()
    model = PPO.load(checkpoint, device="cpu")

    env.reset()
    state = env.state
    ep_sparse = 0.0
    t0 = time.time()
    for t in range(steps):
        ps = det.detect(state)               # perception (sim = ground truth seam)
        vec = fb.build(ps, agent_index=0)    # -> 96-dim
        a, _ = model.predict(vec, deterministic=True)
        # self-play: same model drives partner
        vec_p = fb.build(ps, agent_index=1)
        a_p, _ = model.predict(vec_p, deterministic=True)
        ja = (Action.INDEX_TO_ACTION[int(a)], Action.INDEX_TO_ACTION[int(a_p)])
        state, r, done, info = env.step(ja)
        ep_sparse += r
        if done:
            env.reset(); state = env.state
    dt = time.time() - t0
    print(f"sim pipeline: {steps} steps in {dt:.2f}s ({steps/dt:.0f} Hz decision rate)")
    print(f"sparse reward collected: {ep_sparse:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "real"], default="sim")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--checkpoint",
                    default="/root/data/overcooked2-ai/src/sim/checkpoints/sim_sp/ppo_sp_cramped_room_20000.zip")
    args = ap.parse_args()
    if args.backend == "sim":
        run_sim(args.steps, args.checkpoint)
    else:
        raise NotImplementedError("real backend needs RealDetector calibration")
