"""Behavior cloning on recorded sushi demos.

Reads raw state JSONL + quantized actions, encodes with the shared
ObsEncoder, trains a small CNN+globals policy with cross-entropy.

    python src/bridge/bc_train.py --rounds 2 3 4 --epochs 40

Outputs checkpoints/bc_sushi.pt and prints per-class accuracy.
Runs fine on CPU (tiny net, ~7k samples).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder import ObsEncoder

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """CNN over grid tensor + MLP over globals -> two heads: move(9), button(4)."""

    def __init__(self, in_channels, gh, gw, g_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        gh2, gw2 = (gh + 1) // 2, (gw + 1) // 2
        gh2, gw2 = (gh2 + 1) // 2, (gw2 + 1) // 2
        flat = 32 * gh2 * gw2
        self.trunk = nn.Sequential(
            nn.Linear(flat + g_dim, 256), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.move_head = nn.Linear(256, 9)
        self.btn_head = nn.Linear(256, 4)

    def forward(self, grid, globals_):
        h = self.trunk(torch.cat([self.conv(grid), globals_], dim=1))
        return self.move_head(h), self.btn_head(h)


def load_dataset(demo_dir, rounds, time_limit=210.0):
    enc = ObsEncoder(json.load(open(os.path.join(demo_dir, "stations.json"), encoding="utf-8")))
    grids, globs, acts = [], [], []
    val_mask = []
    for rnd in rounds:
        base = os.path.join(demo_dir, f"round_{rnd:03d}")
        actions = np.load(base + "_actions.npz")
        a0, a1 = actions["a0"], actions["a1"]
        with open(base + ".jsonl", encoding="utf-8") as f:
            states = [json.loads(l) for l in f]
        n = min(len(states), len(a0))
        # time-block split: last 15% of each round is validation (no temporal leakage)
        cut = int(n * 0.85)
        for i in range(n):
            st = states[i]
            g = enc.encode_globals(st, time_limit)
            grid = enc.encode(st)
            for a, onehot in ((a0[i], (1.0, 0.0)), (a1[i], (0.0, 1.0))):
                grids.append(grid)
                globs.append(np.concatenate([g, onehot]).astype(np.float32))
                acts.append(int(a))
                val_mask.append(i >= cut)
    return (np.stack(grids), np.stack(globs), np.array(acts), np.array(val_mask)), enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", default="demos/sushi")
    ap.add_argument("--rounds", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out", default="checkpoints/bc_sushi.pt")
    args = ap.parse_args()

    (grids, globs, acts, val_mask), enc = load_dataset(args.demo_dir, args.rounds)
    print(f"dataset: {len(acts)} samples, grid {grids.shape}, globals {globs.shape[1]}")

    # split joint action id (move*4+btn) into two targets
    mv_t = torch.tensor(acts // 4, dtype=torch.long)
    bt_t = torch.tensor(acts % 4, dtype=torch.long)
    mv_counts = np.bincount(acts // 4, minlength=9)
    bt_counts = np.bincount(acts % 4, minlength=4)
    print("move counts:", mv_counts.tolist())
    print("btn counts:", bt_counts.tolist())
    # buttons are rare -> mild weights; moves are balanced enough -> plain CE
    bt_w = torch.tensor(np.clip(len(acts) / (4 * np.maximum(bt_counts, 1)), 0.2, 5.0), dtype=torch.float32)

    val_idx = np.where(val_mask)[0]
    tr_idx = np.where(~val_mask)[0]
    print(f"train {len(tr_idx)} / val {len(val_idx)} (time-blocked)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = BCPolicy(grids.shape[1], grids.shape[2], grids.shape[3], globs.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    mv_lossf = nn.CrossEntropyLoss()
    bt_lossf = nn.CrossEntropyLoss(weight=bt_w.to(dev))

    tg = torch.tensor(grids, dtype=torch.float32)
    tl = torch.tensor(globs, dtype=torch.float32)
    for ep in range(args.epochs):
        model.train()
        np.random.shuffle(tr_idx)
        tot_loss, nb = 0.0, 0
        for s in range(0, len(tr_idx), args.batch):
            b = tr_idx[s:s + args.batch]
            lm, lb = model(tg[b].to(dev), tl[b].to(dev))
            loss = mv_lossf(lm, mv_t[b].to(dev)) + bt_lossf(lb, bt_t[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                lm, lb = model(tg[val_idx].to(dev), tl[val_idx].to(dev))
                macc = (lm.argmax(1).cpu() == mv_t[val_idx]).float().mean().item()
                bacc = (lb.argmax(1).cpu() == bt_t[val_idx]).float().mean().item()
            print(f"epoch {ep+1}: loss={tot_loss/max(nb,1):.4f} val_move_acc={macc:.3f} val_btn_acc={bacc:.3f}")

    model.eval()
    with torch.no_grad():
        lm, lb = model(tg[val_idx].to(dev), tl[val_idx].to(dev))
        pm, pb = lm.argmax(1).cpu(), lb.argmax(1).cpu()
    print("\nper-move val acc:", {
        m: round(float((pm[val_m] == mv_t[val_idx][val_m]).float().mean()), 2)
        for m in range(9) if (val_m := (mv_t[val_idx] == m)).sum() > 5})
    print("per-btn val acc:", {
        b: round(float((pb[val_b] == bt_t[val_idx][val_b]).float().mean()), 2)
        for b in range(4) if (val_b := (bt_t[val_idx] == b)).sum() > 5})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "grid_shape": list(grids.shape[1:]),
        "g_dim": int(globs.shape[1]),
        "two_heads": True,
    }, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
