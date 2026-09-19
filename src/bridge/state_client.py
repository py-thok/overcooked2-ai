"""Python client for the OC2StateBridge BepInEx mod.

The mod (mod/OC2StateBridge) runs inside Overcooked! 2 and serves ground-truth
kitchen state on 127.0.0.1:8765. This client fetches state and sends actions.

Protocol (newline-delimited text):
    -> "GET"                  <- one line of JSON state
    -> "ACTION {json}"        <- "OK"
    -> "PING"                 <- "PONG"

Usage:
    python state_client.py              # print state at ~4 Hz
    python state_client.py --once       # print one snapshot (pretty)
    python state_client.py --drive      # random-walk demo driving player 0
"""
import argparse
import json
import socket
import time


class StateBridgeClient:
    def __init__(self, host="127.0.0.1", port=8765, timeout=5.0):
        self.host, self.port, self.timeout = host, port, timeout
        self._sock = None

    def connect(self):
        self.close()
        self._sock = socket.create_connection((self.host, self.port), self.timeout)
        self._file = self._sock.makefile("rw", encoding="utf-8", newline="\n")
        return self

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _cmd(self, line):
        self._file.write(line + "\n")
        self._file.flush()
        return self._file.readline().strip()

    def ping(self):
        return self._cmd("PING") == "PONG"

    def get_state(self):
        return json.loads(self._cmd("GET"))

    def get_stations(self):
        """Static station census for the current level (refreshed every 2s)."""
        return json.loads(self._cmd("STATIONS"))

    def send_action(self, move=(0.0, 0.0), player=0, pickup=None, use=None, dash=None):
        """move: (x, y) analog axes in [-1, 1]. Buttons are hold-semantics:
        pass True to hold, False to release, None to leave unchanged."""
        msg = {"player": player, "move": [float(move[0]), float(move[1])]}
        if pickup is not None:
            msg["pickup"] = bool(pickup)
        if use is not None:
            msg["use"] = bool(use)
        if dash is not None:
            msg["dash"] = bool(dash)
        return self._cmd("ACTION " + json.dumps(msg)) == "OK"

    def tap(self, button, player=0, hold_s=0.12):
        """Press and release a button ('pickup' | 'use' | 'dash')."""
        self.send_action(player=player, **{button: True})
        time.sleep(hold_s)
        self.send_action(player=player, **{button: False})

    def reset_level(self):
        """Restart the current level via the game's own flow."""
        return self._cmd("RESET") == "OK"

    def load_level(self, scene_name):
        """Load a kitchen level by scene name, from anywhere."""
        return self._cmd(f"LOADLEVEL {scene_name}") == "OK"

    def set_player_pos(self, x, y, z, player=0):
        """Teleport a player to a world position (calibration/scenario setup)."""
        return self._cmd(f"SETPOS {player} {float(x)} {float(y)} {float(z)}") == "OK"

    def set_mode(self, mode):
        """'drive': bot controls chefs (default). 'sniff': human controls,
        inputs are echoed in state for demonstration recording."""
        assert mode in ("drive", "sniff")
        return self._cmd(f"MODE {mode}") == "OK"

    def set_timescale(self, scale):
        """Scale game speed (clamped to [0.25, 8] by the mod)."""
        return self._cmd(f"TIMESCALE {float(scale)}") == "OK"

    def wait_in_round(self, want=True, timeout=60.0, poll=0.5):
        """Block until the game is in/out of a round."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if bool(self.get_state().get("in_round")) == want:
                    return True
            except (OSError, json.JSONDecodeError):
                self._reconnect()
            time.sleep(poll)
        return False

    def _reconnect(self):
        for _ in range(20):
            try:
                self.connect()
                return
            except OSError:
                time.sleep(0.5)
        raise ConnectionError("state server unreachable")


def summarize(st):
    r = st.get("round", {})
    print(f"scene={st.get('scene')} in_round={st.get('in_round')} "
          f"time={r.get('time_remaining', -1):.0f}s score={r.get('score')}")
    for p in st.get("players", []):
        held = p.get("held")
        held_s = held["name"] if held else "-"
        print(f"  P{p.get('i')} id={p.get('player_id')} grid={p.get('grid')} held={held_s}")
    for c in st.get("cookers", []):
        print(f"  cooker grid={c.get('grid')} {c.get('state')} "
              f"{c.get('progress', 0):.1f}/{c.get('cook_time', 0):.1f} "
              f"contents={_names(c.get('contents'))}")
    for o in st.get("orders", []):
        print(f"  order#{o.get('id')} {o.get('recipe')} "
              f"{o.get('remaining', 0):.0f}/{o.get('lifetime', 0):.0f}s")


def _names(contents):
    if not contents:
        return []
    return [n.get("name") for n in contents if isinstance(n, dict)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--hz", type=float, default=4.0)
    args = ap.parse_args()

    cli = StateBridgeClient().connect()
    print("connected:", cli.ping())

    if args.once:
        print(json.dumps(cli.get_state(), indent=2, ensure_ascii=False))
        return

    if args.drive:
        import random
        try:
            while True:
                mv = (random.uniform(-1, 1), random.uniform(-1, 1))
                cli.send_action(move=mv, player=0)
                time.sleep(0.5)
                if random.random() < 0.2:
                    cli.tap("pickup", player=0)
        except KeyboardInterrupt:
            cli.send_action(move=(0, 0), player=0)
        return

    try:
        while True:
            summarize(cli.get_state())
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
