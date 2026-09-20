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
        self._interact_ready = [0.0, 0.0]  # wall-clock when INTERACT next allowed

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
        stations = self.cli.get_stations()
        self._enc = ObsEncoder(stations)
        slist = stations.get("stations", stations) if isinstance(stations, dict) else stations
        self._crates = [s["pos"] for s in slist if "DispenserCrate" in s.get("name", "")]
        self._cooker_pos = [s["pos"] for s in slist if "workstation_cooker" in s.get("name", "")]
        self._release_all()
        self._interact_ready = [0.0, 0.0]
        self._steps = 0
        self._curriculum_spawn()
        self._prev = self.cli.get_state()
        return self._obs(self._prev)

    def _curriculum_spawn(self):
        """Place the chefs at useful starting spots facing their first
        station: P0 before the rice crate (8.4,-2.4), P1 before the chopping
        board (15.6,-2.4). Exposes the policy to the first link of the
        pipeline every round instead of hoping random walks find crates."""
        spots = [(8.4, -3.6), (15.6, -3.7)]
        for p, (x, z) in enumerate(spots):
            self.cli.set_player_pos(x, 0, z, player=p)
        time.sleep(0.25)
        players = self.cli.get_state().get("players", [])
        for p in range(2):
            sign = (1, 1)
            if p < len(players) and players[p].get("move_sign"):
                sign = players[p]["move_sign"]
            # face north (world +z): input y = -sign1
            self.cli.send_action(player=p, move=(0.0, -1.0 * sign[1]))
        time.sleep(0.25)
        for p in range(2):
            self.cli.send_action(player=p, move=(0.0, 0.0))

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
        # Round boundary detection. The flow controller can flicker null
        # mid-round (false in_round=False), and scene reload can swallow the
        # in_round=False window entirely (main thread busy) — so candidates
        # are confirmed with re-reads, and timer discontinuities count too.
        tr_prev = self._prev.get("round", {}).get("time_remaining", -1)
        tr_cur = st.get("round", {}).get("time_remaining", -1)
        done = False
        round_restarted = False
        if self._steps >= self.max_steps:
            done = True
        elif self._boundary_candidate(st, tr_prev):
            st2, round_restarted = self._confirm_boundary(st, tr_prev)
            if st2 is not None:
                st = st2
                tr_cur = st.get("round", {}).get("time_remaining", -1)
                done = True
        if round_restarted and parts.get("score", 0) < 0:
            # score reset 0 with the new round is not a penalty
            reward -= parts["score"]
            parts["score"] = 0.0
        # score resets to 0 with the new round; the episode's final score was
        # visible in the previous state
        score = (self._prev.get("round", {}).get("score", 0)
                 if round_restarted or tr_cur < 0
                 else st.get("round", {}).get("score", 0))
        info = {"score": score,
                "time_remaining": tr_cur,
                "reward_parts": parts}
        self._prev = st
        return self._obs(st), reward, done, info

    def _boundary_candidate(self, st, tr_prev):
        tr_cur = st.get("round", {}).get("time_remaining", -1)
        if not st.get("in_round"):
            return True
        if not self._prev.get("in_round"):
            return False
        if tr_prev == 0 and tr_cur == 0:
            return True  # timer pinned at 0 = outro screen, round is over
        return (tr_prev >= 0 and tr_cur < 0) or tr_cur > tr_prev + 5

    def _confirm_boundary(self, st0, tr_prev):
        """Confirm a round-end candidate with a few re-reads (rejects
        transient flow/timer glitches). Returns (state_to_use, restarted)
        or (None, False) if the candidate was a glitch."""
        votes = 0
        st = st0
        for _ in range(3):
            if not self._boundary_candidate(st, tr_prev):
                break
            votes += 1
            time.sleep(0.1)
            st = self.cli.get_state()
        if votes < 3:
            return None, False
        tr_cur = st.get("round", {}).get("time_remaining", -1)
        restarted = st.get("in_round") and tr_cur > 0
        return st, restarted

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
            # raw button taps race the JustPressed claim chain and get lost.
            # Cooldown: the pickup result reaches our state ~0.6s after the
            # command; without it a policy spamming pickup cancels its own
            # pickups by placing the item right back before it ever sees it.
            now = time.time()
            if now >= self._interact_ready[player]:
                self.cli.interact(player)
                self._interact_ready[player] = now + 0.7
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

    @staticmethod
    def _held_name(player_state):
        h = player_state.get("held")
        return h.get("name") if h else None

    @staticmethod
    def _is_ingredient(name):
        # SushiRice, Cucumber, ChoppedCucumber, fish, nori, ... — anything
        # that is not cookware/tableware
        return name is not None and not name.startswith(("utensil_", "equipment_"))

    def _reward(self, prev, cur):
        parts = {"score": 0.0, "pot_add": 0.0, "pot_start": 0.0, "pot_cooked": 0.0,
                 "held_ing": 0.0, "held_utensil": 0.0, "nav": 0.0,
                 "metric_pickup_ing": 0.0}
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
        # shaping: held-item transitions (symmetric so pick/drop cycling nets 0)
        pp, cp = prev.get("players", []), cur.get("players", [])
        for i in range(min(len(pp), len(cp))):
            ph, ch = self._held_name(pp[i]), self._held_name(cp[i])
            if ph != ch:
                if ch is not None and self._is_ingredient(ch):
                    parts["held_ing"] += 0.05                    # picked up an ingredient
                    parts["metric_pickup_ing"] += 1.0            # count (not a reward)
                if ph is not None and self._is_ingredient(ph) and ch is None:
                    parts["held_ing"] -= 0.05                    # put it down (pot_add nets +)
                if ch is not None and ch.startswith("utensil_"):
                    parts["held_utensil"] -= 0.1                 # grabbed a pot/extinguisher
            # navigation shaping: delta distance to the relevant target
            # (carrying ingredient -> nearest cooker; empty-handed -> crate)
            targets = self._cooker_pos if self._is_ingredient(ch) else (
                self._crates if ch is None else None)
            if targets:
                d_prev = self._nearest_dist(pp[i].get("pos"), targets)
                d_cur = self._nearest_dist(cp[i].get("pos"), targets)
                w = 0.02 if self._is_ingredient(ch) else 0.01
                parts["nav"] += w * (d_prev - d_cur)
        return sum(v for k, v in parts.items() if not k.startswith("metric_")), parts

    @staticmethod
    def _nearest_dist(pos, targets):
        if not pos:
            return 0.0
        return min((pos[0] - t[0]) ** 2 + (pos[2] - t[2]) ** 2 for t in targets) ** 0.5


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
