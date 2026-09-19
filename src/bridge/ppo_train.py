"""PPO fine-tuning on the real sushi level via the state bridge.

Self-play: one shared policy drives both chefs; team reward is shared.
Loads the BC checkpoint as initialization (same architecture).

    python src/bridge/ppo_train.py --bc checkpoints/bc_sushi.pt \
        --timescale 3 --decision-hz 5 --total-steps 500000

Logs to logs/ppo_sushi/ (tensorboard), checkpoints to checkpoints/ppo_sushi_*.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import OC2Env, N_ACTIONS
from bc_train import BCPolicy


class ActorCritic(nn.Module):
    """BCPolicy trunk/heads + value head. Loads BC weights by matching keys."""

    def __init__(self, grid_shape, g_dim):
        super().__init__()
        c, h, w = grid_shape
        self.bc = BCPolicy(c, h, w, g_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, grid, globals_):
        h = self.bc.trunk(torch.cat([self.bc.conv(grid), globals_], dim=1))
        return self.bc.move_head(h), self.bc.btn_head(h), self.value_head(h).squeeze(-1)

    def load_bc(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.bc.load_state_dict(ckpt["state_dict"])
        print(f"BC weights loaded from {path}")


def sample(lm, lb):
    dm = torch.distributions.Categorical(logits=lm)
    db = torch.distributions.Categorical(logits=lb)
    am, ab = dm.sample(), db.sample()
    logp = dm.log_prob(am) + db.log_prob(ab)
    ent = dm.entropy() + db.entropy()
    return am, ab, logp, ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="checkpoints/bc_sushi.pt")
    ap.add_argument("--scene", default="s_sushi_1_4")
    ap.add_argument("--timescale", type=float, default=3.0)
    ap.add_argument("--decision-hz", type=float, default=5.0)
    ap.add_argument("--total-steps", type=int, default=500_000)
    ap.add_argument("--rollout", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--log-dir", default="logs/ppo_sushi")
    ap.add_argument("--save-every", type=int, default=20)
    args = ap.parse_args()

    env = OC2Env(args.scene, decision_hz=args.decision_hz, timescale=args.timescale)

    # probe obs dims
    obs = env.reset()
    grid_shape = obs["grid"].shape
    g_dim = obs["globals"][0].shape[0]
    print(f"obs grid={grid_shape} globals={g_dim}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActorCritic(grid_shape, g_dim).to(dev)
    if os.path.exists(args.bc):
        model.load_bc(args.bc)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.log_dir)

    gs = 0  # global step (per-agent transitions)
    update = 0
    ep_scores = []
    t_start = time.time()
    try:
        while gs < args.total_steps:
            # ---------------- rollout ----------------
            G, Gl, AM, AB, LP, R, V, D = [], [], [], [], [], [], [], []
            obs = env._obs(env._prev) if env._prev else env.reset()
            for _ in range(args.rollout):
                grid = torch.tensor(obs["grid"], dtype=torch.float32, device=dev).unsqueeze(0)
                step_lp, step_ent = [], []
                acts = []
                vals = []
                for p in range(2):
                    g = torch.tensor(obs["globals"][p], dtype=torch.float32, device=dev).unsqueeze(0)
                    with torch.no_grad():
                        lm, lb, v = model(grid, g)
                    am, ab, logp, ent = sample(lm, lb)
                    acts.append(int(am.item()) * 4 + int(ab.item()))
                    vals.append(float(v.item()))
                    G.append(obs["grid"]); Gl.append(obs["globals"][p])
                    AM.append(int(am.item())); AB.append(int(ab.item()))
                    LP.append(float(logp.item())); V.append(vals[-1])
                obs2, r, done, info = env.step(tuple(acts))
                for _p in range(2):
                    R.append(r); D.append(done)
                obs = obs2
                gs += 2
                if done:
                    ep_scores.append(info["score"])
                    writer.add_scalar("episode/score", info["score"], gs)
                    print(f"[{gs}] episode done: score={info['score']} "
                          f"(last5 avg={np.mean(ep_scores[-5:]):.0f})")
                    obs = env.reset()

            # ---------------- GAE ----------------
            with torch.no_grad():
                grid = torch.tensor(obs["grid"], dtype=torch.float32, device=dev).unsqueeze(0)
                next_v = 0.0
                if not D[-1]:
                    g = torch.tensor(obs["globals"][1], dtype=torch.float32, device=dev).unsqueeze(0)
                    _, _, nv = model(grid, g)
                    next_v = float(nv.item())
            T = len(R)
            adv = np.zeros(T, dtype=np.float32)
            lastgae = 0.0
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else V[t + 1]
                nonterminal = 1.0 - float(D[t])
                delta = R[t] + args.gamma * nv * nonterminal - V[t]
                lastgae = delta + args.gamma * args.lam * nonterminal * lastgae
                adv[t] = lastgae
            ret = adv + np.array(V, dtype=np.float32)

            # ---------------- PPO update ----------------
            tG = torch.tensor(np.stack(G), dtype=torch.float32, device=dev)
            tL = torch.tensor(np.stack(Gl), dtype=torch.float32, device=dev)
            tAM = torch.tensor(AM, dtype=torch.long, device=dev)
            tAB = torch.tensor(AB, dtype=torch.long, device=dev)
            tLP = torch.tensor(LP, dtype=torch.float32, device=dev)
            tA = torch.tensor(adv, dtype=torch.float32, device=dev)
            tR = torch.tensor(ret, dtype=torch.float32, device=dev)
            tA = (tA - tA.mean()) / (tA.std() + 1e-8)

            idx = np.arange(T)
            for _ in range(args.epochs):
                np.random.shuffle(idx)
                for s in range(0, T, args.minibatch):
                    b = idx[s:s + args.minibatch]
                    lm, lb, v = model(tG[b], tL[b])
                    dm = torch.distributions.Categorical(logits=lm)
                    db = torch.distributions.Categorical(logits=lb)
                    logp = dm.log_prob(tAM[b]) + db.log_prob(tAB[b])
                    ratio = torch.exp(logp - tLP[b])
                    s1 = ratio * tA[b]
                    s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * tA[b]
                    pi_loss = -torch.min(s1, s2).mean()
                    v_loss = ((v - tR[b]) ** 2).mean()
                    ent = (dm.entropy() + db.entropy()).mean()
                    loss = pi_loss + args.vf * v_loss - args.ent * ent
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()

            update += 1
            writer.add_scalar("train/pi_loss", pi_loss.item(), gs)
            writer.add_scalar("train/v_loss", v_loss.item(), gs)
            writer.add_scalar("train/entropy", ent.item(), gs)
            if update % args.save_every == 0:
                path = f"checkpoints/ppo_sushi_{gs}.pt"
                torch.save({"state_dict": model.state_dict(),
                            "grid_shape": list(grid_shape), "g_dim": g_dim}, path)
                print(f"saved {path}")
            rate = gs / max(time.time() - t_start, 1)
            print(f"[{gs}/{args.total_steps}] update {update} done, "
                  f"pi={pi_loss.item():+.3f} v={v_loss.item():.3f} ent={ent.item():.3f} "
                  f"({rate:.0f} trans/s)")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        torch.save({"state_dict": model.state_dict(),
                    "grid_shape": list(grid_shape), "g_dim": g_dim}, "checkpoints/ppo_sushi_final.pt")
        env.close()
        writer.close()
        print("saved checkpoints/ppo_sushi_final.pt")


if __name__ == "__main__":
    main()
