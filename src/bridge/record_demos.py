"""Record human demonstrations for behavior cloning.

The mod is switched to sniff mode: the human plays with the keyboard while
this script logs (raw_state, quantized_action_p0, quantized_action_p1) at
~10 Hz. Raw states (not encoded tensors) are stored so the encoder can evolve
without invalidating old demos.

    python record_demos.py --rounds 3 --out demos/sushi

Controls during recording: play normally. One file per round:
    demos/sushi/round_000.jsonl   (one JSON state per line)
    demos/sushi/round_000_actions.npz

Stop anytime with Ctrl+C (mod is restored to drive mode).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_client import StateBridgeClient
from env import MOVES

MOVE_VECS = np.array(MOVES, dtype=float)


def quantize(player):
    """State player entry -> action id (9 moves x 4 buttons)."""
    inp = player.get("input")
    if not inp:
        return 0
    sign = player.get("move_sign") or (1, 1)
    mx, my = inp["move"]
    # world (dx, dz) from raw input
    w = np.array([mx * sign[0], -my * sign[1]], dtype=float)
    if np.linalg.norm(w) > 1e-3:
        w /= np.linalg.norm(w)
    d = np.linalg.norm(MOVE_VECS - w, axis=1)
    move_idx = int(np.argmin(d))
    if np.linalg.norm(w) < 0.3:
        move_idx = 0
    btn = 0
    if inp.get("pickup"):
        btn = 1
    elif inp.get("use"):
        btn = 2
    elif inp.get("dash"):
        btn = 3
    return move_idx * 4 + btn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="demos/sushi")
    ap.add_argument("--scene", default="s_sushi_1_4")
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cli = StateBridgeClient().connect()
    cli.set_mode("sniff")
    print("sniff mode on - you control the chefs. Recording...")
    try:
        for rnd in range(args.rounds):
            st = cli.get_state()
            if st.get("scene") != args.scene or not st.get("in_round"):
                cli.load_level(args.scene)
                cli.wait_in_round(True, timeout=90)
            # skip intro
            while cli.get_state()["round"]["time_elapsed"] < 4:
                time.sleep(0.3)
            print(f"round {rnd}: recording (play!)")

            states, a0s, a1s = [], [], []
            t_next = time.time()
            while True:
                st = cli.get_state()
                if not st.get("in_round"):
                    break
                players = st.get("players", [])
                if len(players) >= 2:
                    states.append(st)
                    a0s.append(quantize(players[0]))
                    a1s.append(quantize(players[1]))
                t_next += 1.0 / args.hz
                time.sleep(max(0.0, t_next - time.time()))

            score = st.get("round", {}).get("score", 0) if st else 0
            base = os.path.join(args.out, f"round_{rnd:03d}")
            with open(base + ".jsonl", "w") as f:
                for st in states:
                    f.write(json.dumps(st) + "\n")
            np.savez_compressed(base + "_actions.npz", a0=a0s, a1=a1s)
            print(f"round {rnd}: saved {len(states)} frames, score={score}")

            if rnd + 1 < args.rounds:
                print("restarting level for next round...")
                cli.load_level(args.scene)
                cli.wait_in_round(True, timeout=90)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cli.set_mode("drive")
        print("drive mode restored")


if __name__ == "__main__":
    main()
