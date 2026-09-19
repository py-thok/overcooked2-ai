"""OC2 gym-style environment over the state bridge.

    env = OC2Env("s_sushi_1_4", decision_hz=5, timescale=2)
    obs = env.reset()
    obs, reward, done, info = env.step((a0, a1))

Action space (v1): discrete 36 = 9 moves x 4 buttons.
  move: 0=stay, 1..8 = N/NE/E/SE/S/SW/W/NW (world/grid axes)
  button: 0=none, 1=pickup(tap), 2=use(hold while moving), 3=dash(tap)
Both chefs are driven; the same policy net can be shared (agent one-hot is
appended to that agent's globals).

Reward: team score delta + small shaping for pot pipeline events.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_client import StateBridgeClient
from encoder import ObsEncoder, N_CHANNELS

# world (dx, dz) for the 9 moves
MOVES = [
    (0, 0),
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]
N_MOVE = 9
N_BTN = 4
N_ACTIONS = N_MOVE * N_BTN

BTN_NAMES = {0: None, 1: "pickup", 2: "use", 3: "dash"}


class OC2Env:
    def __init__(self, scene="s_sushi_1_4", decision_hz=5.0, timescale=1.0,
                 time_limit=210.0, max_steps=2000):
        self.scene = scene
        self.hz = decision_hz
        self.timescale = timescale
        self.time_limit = time_limit
        self.max_steps = max_steps
        self.cli = StateBridgeClient().connect()
        self._enc = None
        self._prev = None
        self._steps = 0
        self._btn_held = [{}, {}]  # per player: buttons currently held down

    # ------------------------------------------------------------------
    def reset(self):
        st = self.cli.get_state()
        if st.get("scene") != self.scene or not st.get("in_round"):
            self.cli.load_level(self.scene)
            if not self.cli.wait_in_round(True, timeout=90):
                raise RuntimeError("failed to enter level " + self.scene)
        else:
            self.cli.reset_level()
            self.cli.wait_in_round(False, timeout=30)
            if not self.cli.wait_in_round(True, timeout=90):
                raise RuntimeError("failed to restart level " + self.scene)

        # wait for intro so chefs are controllable
        while True:
            st = self.cli.get_state()
            if st["round"]["time_elapsed"] > 4 and len(st.get("players", [])) >= 2:
                break
            time.sleep(0.3)

        self.cli.set_timescale(self.timescale)
        self._enc = ObsEncoder(self.cli.get_stations())
        self._release_all()
        self._steps = 0
        self._prev = self.cli.get_state()
        return self._obs(self._prev)

    def step(self, actions):
        t0 = time.time()
        for p in range(2):
            self._apply_action(p, int(actions[p]))

        # wait one decision period (wall clock accounts for timescale)
        dt = (1.0 / self.hz) / max(self.timescale, 0.25)
        time.sleep(max(0.0, dt - (time.time() - t0)))

        st = self.cli.get_state()
        reward = self._reward(self._prev, st)
        self._steps += 1
        done = (not st.get("in_round")) or self._steps >= self.max_steps
        info = {"score": st.get("round", {}).get("score", 0),
                "time_remaining": st.get("round", {}).get("time_remaining", -1)}
        self._prev = st
        return self._obs(st), reward, done, info

    def close(self):
        self._release_all()
        self.cli.set_timescale(1.0)
        self.cli.close()

    # ------------------------------------------------------------------
    def _obs(self, st):
        tensor = self._enc.encode(st)
        g = self._enc.encode_globals(st, self.time_limit)
        g0 = np.concatenate([g, [1.0, 0.0]])
        g1 = np.concatenate([g, [0.0, 1.0]])
        return {"grid": tensor, "globals": (g0, g1)}

    def _apply_action(self, player, action):
        move_idx, btn = divmod(action, N_BTN)
        dx, dz = MOVES[move_idx]
        sign = self._move_sign(player)

        held = self._btn_held[player]
        # release previous tap/hold buttons first
        for b in list(held):
            self.cli.send_action(player=player, **{b: False})
            held.clear()

        msg = {"move": (dx * sign[0], -dz * sign[1])}
        name = BTN_NAMES[btn]
        if name == "pickup" or name == "dash":
            self.cli.send_action(player=player, **{name: True})  # tap: released next step
            held[name] = True
        elif name == "use":
            self.cli.send_action(player=player, use=True)
            held["use"] = True
        self.cli.send_action(player=player, move=msg["move"])

    def _release_all(self):
        for p in range(2):
            self.cli.send_action(player=p, move=(0, 0), pickup=False, use=False, dash=False)
        self._btn_held = [{}, {}]

    def _move_sign(self, player):
        players = self._prev.get("players", []) if self._prev else []
        if player < len(players) and players[player].get("move_sign"):
            return players[player]["move_sign"]
        return (1, 1)

    def _reward(self, prev, cur):
        r = 0.0
        pr, cr = prev.get("round", {}), cur.get("round", {})
        r += (cr.get("score", 0) - pr.get("score", 0)) / 20.0
        # shaping: pot pipeline (contents added / cooking progressed / cooked)
        prev_pots = {tuple(c["grid"]): c for c in prev.get("cookers", []) if c.get("grid")}
        for c in cur.get("cookers", []):
            if not c.get("grid"):
                continue
            p = prev_pots.get(tuple(c["grid"]))
            if p is None:
                continue
            pn = len(p.get("contents") or [])
            cn = len(c.get("contents") or [])
            if cn > pn:
                r += 0.1 * (cn - pn)                      # ingredient added to pot
            if p["progress"] == 0 and c["progress"] > 0:
                r += 0.2                                  # cooking started
            if not p["is_cooked"] and c["is_cooked"]:
                r += 0.3                                  # cooking finished
        return r


if __name__ == "__main__":
    # random-action smoke test
    env = OC2Env(decision_hz=5.0, timescale=2.0)
    obs = env.reset()
    print("reset ok; obs grid:", obs["grid"].shape, "globals:", obs["globals"][0].shape)
    rng = np.random.default_rng(0)
    total_r = 0.0
    t0 = time.time()
    try:
        for i in range(120):  # 24 game-seconds at 5Hz... x2 timescale = 48 game-s
            a = (rng.integers(N_ACTIONS), rng.integers(N_ACTIONS))
            obs, r, done, info = env.step(a)
            total_r += r
            if i % 20 == 0:
                print(f"step {i}: r={r:+.2f} score={info['score']} t_left={info['time_remaining']:.0f}")
            if done:
                print("done at step", i)
                break
    finally:
        env.close()
    print(f"smoke test: total_r={total_r:+.2f} wall={time.time()-t0:.0f}s")
