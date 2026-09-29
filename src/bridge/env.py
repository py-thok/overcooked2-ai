"""OC2 gym-style environment over the state bridge.

    env = OC2Env("s_sushi_1_4", decision_hz=5, timescale=2)
    obs = env.reset()
    obs, reward, done, info = env.step((a0, a1))

Action space (v1): discrete 36 = 9 moves x 4 buttons.
  move: 0=stay, 1..8 = N/NE/E/SE/S/SW/W/NW (world/grid axes)
  button: 0=none, 1=pickup(tap), 2=use(hold while moving), 3=dash(tap)
Both chefs are driven; the same policy net can be shared (agent one-hot is
appended to that agent's globals).

Reward shaping lives in reward.py (weights in REWARD_WEIGHTS), navigation
routing in nav.py, level coordinates in level_config.py.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_client import StateBridgeClient
from encoder import ObsEncoder
from level_config import get_level
from nav import content_names
from reward import compute_reward

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

# rounds are cut this many game-seconds early (nothing can finish cooking
# in the tail, so the shaped reward there just teaches freezing)
EARLY_DONE_SECONDS = 20.0


class OC2Env:
    def __init__(self, scene="s_sushi_1_4", decision_hz=5.0, timescale=1.0,
                 time_limit=210.0, max_steps=2000, port=8765):
        self.scene = scene
        self.level = get_level(scene)
        self.hz = decision_hz
        self.timescale = timescale
        self.time_limit = time_limit
        self.max_steps = max_steps
        self.cli = StateBridgeClient(port=port).connect()
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
                # round restart occasionally collapses to Loading/WorldMap —
                # fall back to a full reload instead of dying
                print(f"reset_level failed on port {self.cli.port}; reloading level")
                self._load_level_robust()

        # wait for intro so chefs are controllable; a collapsed reload can
        # leave the round running with no players — retry the full load
        # instead of dying (run33 crashed here)
        for attempt in range(3):
            t0 = time.time()
            while True:
                st = self.cli.get_state()
                if st["round"]["time_elapsed"] > 4 and len(st.get("players", [])) >= 2:
                    break
                if time.time() - t0 > 60:
                    break
                time.sleep(0.3)
            else:
                continue
            if st["round"]["time_elapsed"] > 4 and len(st.get("players", [])) >= 2:
                break
            print(f"players never spawned on port {self.cli.port} "
                  f"(attempt {attempt+1}); reloading level")
            self._load_level_robust()
        else:
            raise RuntimeError("players never spawned in " + self.scene)

        self.cli.set_timescale(self.timescale)
        self.cli.set_mode("drive")
        stations = self.cli.get_stations()
        self._enc = ObsEncoder(stations)
        slist = stations.get("stations", stations) if isinstance(stations, dict) else stations
        self._reward_ctx = {
            "chop_boards": self.level["chop_boards"],
            "plate_station": self.level["plate_station"],
            "crates": [s["pos"] for s in slist if "DispenserCrate" in s.get("name", "")],
            "cooker_pos": [s["pos"] for s in slist if "workstation_cooker" in s.get("name", "")],
            "chopped_paid": set(),  # chopped pickups already paid this episode
        }
        self._release_all()
        self._interact_ready = [0.0, 0.0]
        self._steps = 0
        self._curriculum_spawn()
        self._prev = self.cli.get_state()
        return self._obs(self._prev)

    def _curriculum_spawn(self):
        """Place the chefs at useful starting spots facing their first
        station. Exposes the policy to the first link of the pipeline every
        round instead of hoping random walks find crates."""
        for p, (x, z) in enumerate(self.level["spawn_spots"]):
            self.cli.set_player_pos(x, 0, z, player=p)
        time.sleep(0.25)
        self._face_north()
        self._preload_chop_board()
        self._preload_chopped()
        self._preload_rice()
        self._preload_plate()
        self._preload_sushi()

    def _preload_sushi(self):
        """Assemble a COMPLETE sushi into P0's hand at spawn: scoop the
        pre-cooked rice onto the preloaded plate, then P1 plate-locks the
        seaweed and the pre-chopped cucumber onto the HELD plate (verified:
        plate-lock works on plates held by the other chef).

        Why: with every episode opening one INTERACT-at-hatch away from a
        real score, serve discovery stops depending on the whole assembly
        chain aligning by chance (plate_complete stayed 0/50 long after
        every link was individually verified). The score reward (+3 per
        sushi) then anchors credit backward along the chain. Costs ~10s
        per reset; retire once serves are routine."""
        pot_stand = self.level.get("rice_pot_stand")
        spawn0 = self.level["spawn_spots"][0]
        board2 = self.level["chop_boards"][1]
        try:
            # 1. P0 scoop: empty plate + cooked pot (both preloaded)
            h0 = self._held_of(0) or {}
            if not h0.get("is_plate"):
                print("sushi preload: P0 has no plate, skipping")
                return
            if not any(c.get("is_cooked") for c in self.cli.get_state().get("cookers", [])):
                print("sushi preload: no cooked rice, skipping")
                return
            self.cli.set_player_pos(pot_stand[0], 0, pot_stand[1], player=0)
            time.sleep(0.4)
            self.cli.interact(0)
            time.sleep(0.8)
            h0 = self._held_of(0) or {}
            if "SushiRice" not in content_names(h0.get("contents")):
                print("sushi preload: scoop failed, skipping")
                self.cli.set_player_pos(spawn0[0], 0, spawn0[1], player=0)
                return
            # park P0 at spawn; assembly happens there (open floor, no pots)
            self.cli.set_player_pos(spawn0[0], 0, spawn0[1], player=0)
            time.sleep(0.3)

            # 2. P1: seaweed -> add onto P0's held plate
            for attempt in range(2):
                if self._held_of(1):
                    self.cli.throw_item(1)
                    time.sleep(0.4)
                self.cli.set_player_pos(8.4, 0, -3.6, player=1)
                time.sleep(0.3)
                self.cli.fetch_crate(1, 8.4, -1.2)
                time.sleep(0.9)
                if (self._held_of(1) or {}).get("name") == "Seaweed":
                    break
            if (self._held_of(1) or {}).get("name") == "Seaweed":
                self.cli.set_player_pos(spawn0[0] + 1.5, 0, spawn0[1], player=1)
                time.sleep(0.4)
                self.cli.interact(1)   # plate-lock -> P0's held plate
                time.sleep(0.8)

            # 3. P1: pick up the pre-chopped cucumber from board 2, add it
            b2stand = (board2[0], board2[2] - 1.2)
            for attempt in range(3):
                self.cli.set_player_pos(b2stand[0], 0, b2stand[1], player=1)
                time.sleep(0.3)
                self.cli.interact(1)
                time.sleep(1.3)   # pickup animation — never teleport early
                h1 = self._held_of(1) or {}
                if (h1.get("name") or "").startswith("Chopped"):
                    break
            h1 = self._held_of(1) or {}
            if (h1.get("name") or "").startswith("Chopped"):
                self.cli.set_player_pos(spawn0[0] + 1.5, 0, spawn0[1], player=1)
                time.sleep(0.4)
                self.cli.interact(1)   # plate-lock again
                time.sleep(0.8)

            h0 = self._held_of(0) or {}
            n = len(content_names(h0.get("contents")))
            print(f"sushi preload: P0 plate has {n} contents")
            # re-park P1 at board 1 (its chop-practice spot)
            bs = self.level["board_stand"]
            self.cli.set_player_pos(bs[0], 0, bs[1], player=1)
            time.sleep(0.15)
            self._face_north()
        except Exception as e:
            print(f"sushi preload failed ({e}); continuing without it")

    def _preload_chopped(self):
        """Scripted-chop a cucumber on board 2 at spawn, so every episode
        opens with a READY chopped filling. The 3rd plate layer has been
        supply-starved for hours (chop_done 0/50 across runs): the policy
        unlearned use-hold chopping under the scoop-churn economy and
        re-discovery from exploration alone is glacial. With a chopped item
        on a board every episode, chopped pickup -> plate-lock ->
        plate_complete -> serve becomes discoverable NOW, and the score
        reward can pull the whole chop chain back up from the top.
        Board 1 still gets the raw cucumber (chop practice stays available).
        Speed-chops at 8x like the rice speed-cook; failure just skips."""
        crate = self.level.get("cucumber_crate")
        if not crate or len(self.level.get("chop_boards", [])) < 2:
            return
        board = self.level["chop_boards"][1]
        stand = (board[0], board[2] - 1.2)
        try:
            got = False
            for attempt in range(2):
                if self._held_of(1):
                    self.cli.throw_item(1)
                    time.sleep(0.3)
                self.cli.set_player_pos(crate[0], 0, crate[2] - 1.2, player=1)
                time.sleep(0.3)
                self.cli.fetch_crate(1, crate[0], crate[2])
                time.sleep(0.7)
                name = (self._held_of(1) or {}).get("name")
                if name and self._is_choppable(name):
                    got = True
                    break
            if not got:
                print("chopped preload: no choppable, skipping")
                return
            self.cli.set_player_pos(stand[0], 0, stand[1], player=1)
            time.sleep(0.3)
            self.cli.place_board(1, board[0], board[2])
            time.sleep(0.5)
            # chop at 8x: use-hold until the item transforms (progress 0.99
            # crossing isn't always observable; the Chopped* name is)
            self.cli.set_timescale(8.0)
            self.cli.send_action(player=1, use=True)
            t0 = time.time()
            done = False
            while time.time() - t0 < 10:
                time.sleep(0.4)
                names = {it.get("name", "") for it in self.cli.get_state().get("items", [])}
                if any(n.startswith("Chopped") for n in names):
                    done = True
                    break
            self.cli.send_action(player=1, use=False)
            self.cli.set_timescale(self.timescale)
            print("chopped preload: chopped ready" if done
                  else "chopped preload: chop timed out, continuing")
            # re-park P1 at board 1 (chop preload's spot), facing it
            bs = self.level["board_stand"]
            self.cli.set_player_pos(bs[0], 0, bs[1], player=1)
            time.sleep(0.15)
            self._face_north()
        except Exception as e:
            print(f"chopped preload failed ({e}); continuing without it")
            self.cli.set_timescale(self.timescale)

    def _preload_plate(self):
        """P0 ends the spawn holding an empty plate: with rice already
        cooking (rice preload), the scoop window opens mid-round every
        episode. scoop_stance stayed 0/50 without this — the plate station
        is far south of the action and nobody ever fetched a plate."""
        stand = self.level.get("plate_stand")
        if not stand:
            return
        try:
            for attempt in range(3):
                if self._held_of(0):
                    self.cli.throw_item(0)
                    time.sleep(0.3)
                self.cli.set_player_pos(stand[0], 0, stand[1], player=0)
                time.sleep(0.3)
                self.cli.fetch_plate(0)
                time.sleep(0.7)
                held = self._held_of(0) or {}
                print(f"plate preload try {attempt+1}: held={held.get('name')}")
                if held.get("is_plate"):
                    break
        except Exception as e:
            print(f"plate preload failed ({e}); continuing without it")

    def _preload_rice(self):
        """P0 starts a pot of rice at spawn, so every episode has a cooked
        rice window in the mid-round (scooping — plated — stayed ~2/50
        because cooked rice and a fetched plate almost never coincided).
        Same direct-event pattern as the chop preload: FETCHCRATE at the
        rice crate, then INTERACT next to the cooker — the pot-lock routes
        raw rice straight into the pot."""
        crate = self.level.get("rice_crate")
        if not crate:
            return
        cooker = self.level["rice_cooker"]
        pot_stand = self.level["rice_pot_stand"]
        try:
            got = False
            for attempt in range(3):
                if self._held_of(0):
                    self.cli.throw_item(0)
                    time.sleep(0.3)
                self.cli.set_player_pos(self.level["rice_stand"][0], 0,
                                        self.level["rice_stand"][1], player=0)
                time.sleep(0.3)
                self.cli.fetch_crate(0, crate[0], crate[2])
                time.sleep(0.7)
                name = (self._held_of(0) or {}).get("name")
                print(f"rice preload fetch try {attempt+1}: held={name}")
                if name == "SushiRice":
                    got = True
                    break
            if not got:
                print("rice preload: no rice, skipping")
                return
            self.cli.set_player_pos(pot_stand[0], 0, pot_stand[1], player=0)
            time.sleep(0.3)
            self.cli.interact(0)  # pot-lock: rice -> nearest pot container
            time.sleep(0.5)
            loaded = any(abs((c.get("pos") or [9e9])[0] - cooker[0]) < 1.5
                         and abs((c.get("pos") or [0, 0, 9e9])[2] - cooker[2]) < 1.5
                         and c.get("contents")
                         for c in self.cli.get_state().get("cookers", []))
            print("rice preload: pot loaded" if loaded
                  else f"rice preload: pot empty after interact, "
                       f"held={(self._held_of(0) or {}).get('name')}")
            if loaded:
                # speed-cook at 8x so the episode OPENS with cooked rice
                # (plated stayed 1-2/50: even with rice cooking and a plate
                # in hand, the mid-round cooked window never got exploited)
                self.cli.set_timescale(8.0)
                t0 = time.time()
                while time.time() - t0 < 8:
                    if any(c.get("is_cooked")
                           for c in self.cli.get_state().get("cookers", [])):
                        break
                    time.sleep(0.5)
                self.cli.set_timescale(self.timescale)
                print("rice preload: rice cooked")
            # leave P0 back near its spawn, not blocking the cooker
            self.cli.set_player_pos(self.level["spawn_spots"][0][0], 0,
                                    self.level["spawn_spots"][0][1], player=0)
            time.sleep(0.15)
        except Exception as e:
            print(f"rice preload failed ({e}); continuing without it")

    def _face_north(self):
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

    def _preload_chop_board(self):
        """Fetch a cucumber from the crate and put it on chopping board 1
        for P1, so every episode starts with a ready-to-chop board.

        Uses the mod's direct-event commands (FETCHCRATE/PLACEBOARD) instead
        of the proximity scan: near the cucumber crate the scan keeps losing
        to the fire extinguisher at (14.4,-2.4) and to floor litter, no
        matter where the chef stands. Every step is still verified and
        retried; total failure just skips the pre-load."""
        crate = self.level.get("cucumber_crate")
        if not crate:
            return
        crate_stand = self.level["cucumber_stand"]
        board_stand = self.level["board_stand"]
        board = self.level["chop_boards"][0]
        try:
            choppable = False
            for attempt in range(3):
                held = self._held_of(1)
                if held:
                    self.cli.throw_item(1)
                    time.sleep(0.3)
                self.cli.set_player_pos(crate_stand[0], 0, crate_stand[1], player=1)
                time.sleep(0.3)
                self.cli.fetch_crate(1, crate[0], crate[2])
                time.sleep(0.7)  # pickup reaches state ~0.6s after the command
                held = self._held_of(1)
                name = held.get("name") if held else None
                print(f"chop preload fetch try {attempt+1}: held={name}")
                if name and self._is_choppable(name):
                    choppable = True
                    break
            if not choppable:
                print("chop preload: no choppable, skipping")
                return
            for attempt in range(2):
                self.cli.set_player_pos(board_stand[0], 0, board_stand[1], player=1)
                time.sleep(0.3)
                self.cli.place_board(1, board[0], board[2])
                time.sleep(0.5)
                items = self.cli.get_state().get("items", [])
                if any(it.get("kind") == "carryable"
                       and abs((it.get("pos") or [9e9])[0] - board[0]) < 1.0
                       and abs((it.get("pos") or [0, 0, 9e9])[2] - board[2]) < 1.0
                       for it in items):
                    print("chop preload: board loaded")
                    break
                print(f"chop preload place try {attempt+1}: board still empty, "
                      f"held={(self._held_of(1) or {}).get('name')}")
            # leave P1 parked in front of the loaded board, facing it
            self.cli.set_player_pos(board_stand[0], 0, board_stand[1], player=1)
            time.sleep(0.15)
            self._face_north()
        except Exception as e:
            print(f"chop preload failed ({e}); continuing without it")

    @staticmethod
    def _is_choppable(name):
        # raw fillings that must be chopped; rice cooks, nori is direct
        return (name is not None
                and not name.startswith(("utensil_", "equipment_", "Chopped"))
                and name not in ("Seaweed", "SushiRice"))

    def _held_of(self, player):
        players = self.cli.get_state().get("players", [])
        if player < len(players):
            return players[player].get("held")
        return None

    def step(self, actions):
        self.apply(actions)
        return self.collect()

    def apply(self, actions):
        """Send actions to both chefs (no waiting). Vectorized trainers call
        apply() on every env first, then collect() on every env — the
        per-step sleeps overlap across game instances."""
        self._step_t0 = time.time()
        for p in range(2):
            self._apply_action(p, int(actions[p]))

    def collect(self):
        """Wait out the decision period, read state, compute reward/done."""
        t0 = getattr(self, "_step_t0", time.time())
        dt = (1.0 / self.hz) / max(self.timescale, 0.25)
        time.sleep(max(0.0, dt - (time.time() - t0)))

        st = self.cli.get_state()
        reward, parts = compute_reward(self._prev, st, self._reward_ctx)
        self._steps += 1
        # A post-round read (results screen / level unloaded) has no items —
        # diffing it against the last in-round state makes every plate's food
        # "vanish", firing plate_waste for the round-end wipe. That penalty
        # is policy-independent (observed: constant ~-4.2/rollout, zero
        # adaptation over 2h). Zero it like the score-reset handling below.
        post_round_read = not st.get("in_round")
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
        if (post_round_read or round_restarted) and parts.get("plate_waste", 0.0) < 0.0:
            # round-end item wipe is not food waste (see above)
            reward -= parts["plate_waste"]
            parts["plate_waste"] = 0.0
        if (post_round_read or round_restarted) and parts.get("board_add", 0.0) < 0.0:
            # same wipe: raw fillings vanishing with the round are not
            # ping-pong removals (symmetric board_add is the anti-farm)
            reward -= parts["board_add"]
            parts["board_add"] = 0.0
        if not done:
            # Last ~20 game-seconds no pipeline link can complete (cooking
            # alone takes that long), so under the shaping any action only
            # risks floor_drop/held_utensil penalties — the policy correctly
            # learns to freeze. Cut the idle tail; keep playing only while a
            # loaded plate is being carried (last-second serves are real).
            tr_now = st.get("round", {}).get("time_remaining", -1)
            if 0 <= tr_now < EARLY_DONE_SECONDS:
                loaded_plate = any(
                    (p.get("held") or {}).get("is_plate")
                    and (p.get("held") or {}).get("contents")
                    for p in st.get("players", []))
                if not loaded_plate:
                    done = True
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
