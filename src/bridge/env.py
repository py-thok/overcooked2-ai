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
            self._load_level_robust()
        else:
            self.cli.reset_level()
            self.cli.wait_in_round(False, timeout=30)
            if not self.cli.wait_in_round(True, timeout=90):
                raise RuntimeError("failed to restart level " + self.scene)

        # wait for intro so chefs are controllable
        t0 = time.time()
        while True:
            st = self.cli.get_state()
            if st["round"]["time_elapsed"] > 4 and len(st.get("players", [])) >= 2:
                break
            if time.time() - t0 > 60:
                raise RuntimeError("players never spawned in " + self.scene)
            time.sleep(0.3)

        self.cli.set_timescale(self.timescale)
        self.cli.set_mode("drive")
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
        reward, parts = self._reward(self._prev, st)
        self._steps += 1
        # A GET issued just before scene reload is answered only after the new
        # round starts (main thread busy), so in_round=False is never observed
        # for that transition. Detect the round restart via the timer jump.
        tr_prev = self._prev.get("round", {}).get("time_remaining", -1)
        tr_cur = st.get("round", {}).get("time_remaining", -1)
        round_restarted = (st.get("in_round") and self._prev.get("in_round")
                           and tr_prev >= 0 and tr_cur > tr_prev + 5)
        if round_restarted and parts.get("score", 0) < 0:
            # score reset 0 with the new round is not a penalty
            reward -= parts["score"]
            parts["score"] = 0.0
        done = (not st.get("in_round")) or round_restarted or self._steps >= self.max_steps
        # score resets to 0 with the new round; the episode's final score was
        # visible in the previous state
        score = (self._prev.get("round", {}).get("score", 0) if round_restarted
                 else st.get("round", {}).get("score", 0))
        info = {"score": score,
                "time_remaining": tr_cur,
                "reward_parts": parts}
        self._prev = st
        return self._obs(st), reward, done, info

    def close(self):
        self._release_all()
        self.cli.set_timescale(1.0)
        self.cli.close()

    # ------------------------------------------------------------------
    def _load_level_robust(self):
        """From a fresh boot the StartScreen needs one programmatic ENGAGE
        (the 'press any key' prompt) before a campaign session exists; after
        that LOADLEVEL bootstraps everything itself. Retry until players
        actually spawn."""
        for attempt in range(4):
            st = self.cli.get_state()
            if st.get("scene") == "StartScreen":
                self.cli.engage()
                time.sleep(5)
            self.cli.load_level(self.scene)
            if self.cli.wait_in_round(True, timeout=90):
                t0 = time.time()
                while time.time() - t0 < 30:
                    if len(self.cli.get_state().get("players", [])) >= 2:
                        return
                    time.sleep(1)
                # in_round but no players = broken bare load; loop and retry
            else:
                st = self.cli.get_state()
                if st.get("scene") == "WorldMap":
                    continue  # session bootstrapped, next attempt works
        raise RuntimeError("failed to load level " + self.scene)

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
        if name == "pickup":
            # semantic pickup/place via the game's own interaction events —
            # raw button taps race the JustPressed claim chain and get lost
            self.cli.interact(player)
        elif name == "dash":
            self.cli.send_action(player=player, dash=True)  # tap: released next step
            held["dash"] = True
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
        parts = {"score": 0.0, "pot_add": 0.0, "pot_start": 0.0, "pot_cooked": 0.0}
        pr, cr = prev.get("round", {}), cur.get("round", {})
        parts["score"] = (cr.get("score", 0) - pr.get("score", 0)) / 20.0
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
                parts["pot_add"] += 0.1 * (cn - pn)          # ingredient added to pot
            if p["progress"] == 0 and c["progress"] > 0:
                parts["pot_start"] += 0.2                    # cooking started
            if not p["is_cooked"] and c["is_cooked"]:
                parts["pot_cooked"] += 0.3                   # cooking finished
        return sum(parts.values()), parts


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
