"""PPO fine-tuning on the real sushi level via the state bridge.

Self-play: one shared policy drives both chefs; team reward is shared.
Loads the BC checkpoint as initialization (same architecture).

    python src/bridge/ppo_train.py --bc checkpoints/bc_sushi.pt \
        --timescale 3 --decision-hz 5 --total-steps 500000
    # parallel game instances (one bridge port per instance):
    python src/bridge/ppo_train.py --ports 8765 8767 ...

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

    def _load_compat(self, state_dict):
        """load_state_dict with zero-padding for NEW input channels: encoder
        v1->v2 added grid channels 13/14, so old checkpoints' conv1 weight
        (32, 14, 3, 3) gets two zero channel slices -> (32, 16, 3, 3). The
        new channels start invisible and are learned from zero — old
        behavior fully preserved at init."""
        conv_key = "bc.conv.0.weight"
        own = self.state_dict()[conv_key]
        old = state_dict[conv_key]
        if old.shape[1] < own.shape[1]:
            pad = torch.zeros(old.shape[0], own.shape[1] - old.shape[1],
                              *old.shape[2:])
            state_dict = dict(state_dict)
            state_dict[conv_key] = torch.cat([old, pad], dim=1)
            print(f"conv1 input channels padded {old.shape[1]} -> {own.shape[1]}")
        self.load_state_dict(state_dict)

    def load_bc(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self._load_compat(ckpt["state_dict"])
        print(f"BC weights loaded from {path}")


def batch_obs(obs, dev):
    """Stack per-env obs into one forward batch over all agents of all
    instances; rows ordered env-major: [env0 p0, env0 p1, env1 p0, ...]"""
    N = len(obs)
    grid_np = np.stack([obs[i]["grid"] for i in range(N) for _ in range(2)])
    glob_np = np.stack([obs[i]["globals"][p] for i in range(N) for p in range(2)])
    return (grid_np, glob_np,
            torch.tensor(grid_np, dtype=torch.float32, device=dev),
            torch.tensor(glob_np, dtype=torch.float32, device=dev))


def collect_rollout(model, envs, args, dev, writer, gs, ep_scores):
    """Run args.rollout steps on every env. Buffers are (T, 2N agents) so
    GAE runs per agent along time. Returns (batch, obs, parts_sum, gs)."""
    N = len(envs)
    NA = 2 * N
    G, Gl = [], []          # per (t, agent): obs
    AM, AB = [], []         # actions
    LP = np.zeros((0, NA), np.float32)
    V = np.zeros((0, NA), np.float32)
    R = np.zeros((0, NA), np.float32)
    D = np.zeros((0, NA), np.float32)
    parts_sum = {}
    ep_ret = [0.0] * N
    ep_len = [0] * N

    obs = [e._obs(e._prev) if e._prev else e.reset() for e in envs]
    for _t in range(args.rollout):
        grid_np, glob_np, tG, tL = batch_obs(obs, dev)
        with torch.no_grad():
            lm, lb, v = model(tG, tL)
        dm = torch.distributions.Categorical(logits=lm)
        db = torch.distributions.Categorical(logits=lb)
        am, ab = dm.sample(), db.sample()
        logp = dm.log_prob(am) + db.log_prob(ab)
        am_l, ab_l = am.tolist(), ab.tolist()
        acts = [am_l[r] * 4 + ab_l[r] for r in range(NA)]

        # overlap the per-step sleeps: apply everywhere first
        for i, e in enumerate(envs):
            e.apply((acts[2 * i], acts[2 * i + 1]))
        results = [e.collect() for e in envs]

        G.extend(grid_np); Gl.extend(glob_np)
        AM.append([a // 4 for a in acts]); AB.append([a % 4 for a in acts])
        LP = np.vstack([LP, [logp.tolist()]])
        V = np.vstack([V, [v.tolist()]])
        R = np.vstack([R, [[results[i][1] for i in range(N)
                           for _ in range(2)]]])
        D = np.vstack([D, [[float(results[i][2]) for i in range(N)
                            for _ in range(2)]]])

        for i, (obs2, r, done, info) in enumerate(results):
            for k, vv in info.get("reward_parts", {}).items():
                parts_sum[k] = parts_sum.get(k, 0.0) + vv
            ep_ret[i] += r; ep_len[i] += 1
            if done:
                ep_scores.append(info["score"])
                writer.add_scalar("episode/score", info["score"], gs)
                writer.add_scalar("episode/return_shaped", ep_ret[i], gs)
                writer.add_scalar("episode/length", ep_len[i], gs)
                writer.add_scalar("episode/time_remaining",
                                  info["time_remaining"], gs)
                print(f"[{gs}] env{i} episode done: score={info['score']} "
                      f"(last5 avg={np.mean(ep_scores[-5:]):.0f})")
                ep_ret[i], ep_len[i] = 0.0, 0
                obs[i] = envs[i].reset()
            else:
                obs[i] = obs2
        gs += NA

    batch = {"G": np.stack(G), "Gl": np.stack(Gl),
             "AM": np.array(AM).reshape(-1), "AB": np.array(AB).reshape(-1),
             "LP": LP.reshape(-1), "V": V, "R": R, "D": D}
    return batch, obs, parts_sum, gs


def compute_gae(model, batch, obs, args, dev):
    """GAE per agent column. Returns flat (adv, ret, v_old)."""
    N = len(obs)
    NA = 2 * N
    V, R, D = batch["V"], batch["R"], batch["D"]
    T = V.shape[0]
    adv = np.zeros((T, NA), np.float32)
    # bootstrap values for non-terminal rollout tails (batched)
    _, _, tG, tL = batch_obs(obs, dev)
    with torch.no_grad():
        _, _, nv = model(tG, tL)
    next_vs = nv.tolist()
    for a in range(NA):
        next_v = 0.0 if D[-1, a] else next_vs[a]
        lastgae = 0.0
        for t in reversed(range(T)):
            nv_a = next_v if t == T - 1 else V[t + 1, a]
            nonterminal = 1.0 - D[t, a]
            delta = R[t, a] + args.gamma * nv_a * nonterminal - V[t, a]
            lastgae = delta + args.gamma * args.lam * nonterminal * lastgae
            adv[t, a] = lastgae
    ret = adv + V
    return adv.reshape(-1), ret.reshape(-1), V.reshape(-1)


def ppo_update(model, opt, batch, adv_f, ret_f, args, dev):
    """Clipped PPO epochs over the flattened batch. Returns log metrics."""
    tG = torch.tensor(batch["G"], dtype=torch.float32, device=dev)
    tL = torch.tensor(batch["Gl"], dtype=torch.float32, device=dev)
    tAM = torch.tensor(batch["AM"], dtype=torch.long, device=dev)
    tAB = torch.tensor(batch["AB"], dtype=torch.long, device=dev)
    tLP = torch.tensor(batch["LP"], dtype=torch.float32, device=dev)
    tR = torch.tensor(ret_f, dtype=torch.float32, device=dev)
    tA = torch.tensor(adv_f, dtype=torch.float32, device=dev)
    tA = (tA - tA.mean()) / (tA.std() + 1e-8)

    B = len(tAM)
    idx = np.arange(B)
    approx_kl = clipfrac = 0.0
    pi_loss = v_loss = ent = None
    for _ in range(args.epochs):
        np.random.shuffle(idx)
        for s in range(0, B, args.minibatch):
            b = idx[s:s + args.minibatch]
            lm, lb, v = model(tG[b], tL[b])
            dm = torch.distributions.Categorical(logits=lm)
            db = torch.distributions.Categorical(logits=lb)
            logp = dm.log_prob(tAM[b]) + db.log_prob(tAB[b])
            logratio = logp - tLP[b]
            ratio = torch.exp(logratio)
            s1 = ratio * tA[b]
            s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * tA[b]
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = ((v - tR[b]) ** 2).mean()
            ent = (dm.entropy() + db.entropy()).mean()
            loss = pi_loss + args.vf * v_loss - args.ent * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            approx_kl = float((-logratio).mean().item())
            clipfrac = float((torch.abs(ratio - 1) > args.clip).float().mean().item())
    # explained variance of the value function
    ev = float(1 - np.var(ret_f - batch["V"].reshape(-1)) / (np.var(ret_f) + 1e-8))
    return {"pi_loss": pi_loss.item(), "v_loss": v_loss.item(),
            "entropy": ent.item(), "approx_kl": approx_kl,
            "clipfrac": clipfrac, "explained_var": ev}


def save_ckpt(model, grid_shape, g_dim, path):
    torch.save({"state_dict": model.state_dict(),
                "grid_shape": list(grid_shape), "g_dim": g_dim}, path)


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
    ap.add_argument("--resume", default=None, help="PPO checkpoint to resume from")
    ap.add_argument("--ports", type=int, nargs="+", default=[8765],
                    help="bridge ports — one per parallel game instance")
    args = ap.parse_args()

    envs = [OC2Env(args.scene, decision_hz=args.decision_hz,
                   timescale=args.timescale, port=p) for p in args.ports]
    N = len(envs)          # parallel game instances

    # probe obs dims
    obs = [e.reset() for e in envs]
    grid_shape = obs[0]["grid"].shape
    g_dim = obs[0]["globals"][0].shape[0]
    print(f"obs grid={grid_shape} globals={g_dim} instances={N}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActorCritic(grid_shape, g_dim).to(dev)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model._load_compat(ckpt["state_dict"])
        print(f"resumed PPO weights from {args.resume}")
    elif os.path.exists(args.bc):
        model.load_bc(args.bc)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.log_dir)

    gs = 0  # global step (per-agent transitions)
    update = 0
    ep_scores = []
    t_start = time.time()
    try:
        while gs < args.total_steps:
            batch, obs, parts_sum, gs = collect_rollout(
                model, envs, args, dev, writer, gs, ep_scores)
            adv_f, ret_f, _ = compute_gae(model, batch, obs, args, dev)
            metrics = ppo_update(model, opt, batch, adv_f, ret_f, args, dev)

            update += 1
            sps = gs / max(time.time() - t_start, 1)
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, gs)
            writer.add_scalar("perf/sps", sps, gs)
            for k, vv in parts_sum.items():
                writer.add_scalar(f"reward/{k}", vv, gs)
            if update % args.save_every == 0:
                path = f"checkpoints/ppo_sushi_{gs}.pt"
                save_ckpt(model, grid_shape, g_dim, path)
                print(f"saved {path}")
            print(f"[{gs}/{args.total_steps}] update {update} done, "
                  f"pi={metrics['pi_loss']:+.3f} v={metrics['v_loss']:.3f} "
                  f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f} "
                  f"ev={metrics['explained_var']:.2f} ({sps:.0f} trans/s)")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        save_ckpt(model, grid_shape, g_dim, "checkpoints/ppo_sushi_final.pt")
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
        writer.close()
        print("saved checkpoints/ppo_sushi_final.pt")


if __name__ == "__main__":
    main()
