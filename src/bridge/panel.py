"""OC2StateBridge control panel (BetterGI-style always-on-top mini window).

Shows live mod state and exposes the common controls:
  connection / scene / round / mode / timescale / players / pots / orders
Buttons: toggle drive<->sniff, timescale presets, RESET, LOADLEVEL.

    python src/bridge/panel.py
"""
import os
import sys
import time
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_client import StateBridgeClient

SCENE = "s_sushi_1_4"
BG = "#1e1e1e"
FG = "#d4d4d4"
ACCENT = "#569cd6"
OK = "#4ec9b0"
WARN = "#dcdcaa"
ERR = "#f44747"


class Panel:
    def __init__(self):
        self.cli = None
        self.mode = "sniff"  # mod now boots in sniff mode
        self.root = tk.Tk()
        r = self.root
        r.title("OC2 Bridge")
        r.attributes("-topmost", True)
        r.configure(bg=BG)
        r.resizable(False, False)

        self.vars = {}
        row = 0

        def section(text):
            nonlocal row
            tk.Label(r, text=text, bg=BG, fg=ACCENT, anchor="w",
                     font=("Segoe UI", 9, "bold")).grid(row=row, column=0, columnspan=3,
                                                        sticky="w", padx=8, pady=(8, 0))
            row += 1

        def line(key, label):
            nonlocal row
            tk.Label(r, text=label, bg=BG, fg="#808080", anchor="w",
                     font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", padx=(8, 4))
            v = tk.StringVar(value="-")
            self.vars[key] = v
            tk.Label(r, textvariable=v, bg=BG, fg=FG, anchor="w",
                     font=("Consolas", 9)).grid(row=row, column=1, columnspan=2, sticky="w")
            row += 1

        section("连接")
        line("conn", "状态")
        line("scene", "场景")

        section("回合")
        line("round", "计时/分数")
        line("mode", "输入模式")
        line("timescale", "游戏速度")

        section("控制")
        fr = tk.Frame(r, bg=BG)
        fr.grid(row=row, column=0, columnspan=3, padx=6, pady=2)
        row += 1
        self._btn(fr, "模式切换", self.toggle_mode, 0)
        self._btn(fr, "1x", lambda: self.ts(1), 1)
        self._btn(fr, "2x", lambda: self.ts(2), 2)
        self._btn(fr, "4x", lambda: self.ts(4), 3)
        fr2 = tk.Frame(r, bg=BG)
        fr2.grid(row=row, column=0, columnspan=3, padx=6, pady=2)
        row += 1
        self._btn(fr2, "RESET", self.reset, 0)
        self._btn(fr2, "LOADLEVEL", self.load, 1)

        section("玩家")
        line("p0", "P0")
        line("p1", "P1")

        section("训练 (PPO)")
        line("tr_step", "步数")
        line("tr_score", "近5局分数")
        line("tr_kl", "KL/EV")

        section("锅")
        line("pots", "状态")

        section("订单")
        self.order_labels = []
        for i in range(6):
            v = tk.StringVar(value="")
            self.vars[f"order{i}"] = v
            tk.Label(r, textvariable=v, bg=BG, fg=FG, anchor="w",
                     font=("Consolas", 9)).grid(row=row, column=0, columnspan=3,
                                                sticky="w", padx=(16, 4))
            row += 1

        self.tick()
        r.mainloop()

    def _btn(self, parent, text, cmd, col):
        tk.Button(parent, text=text, command=cmd, bg="#2d2d30", fg=FG,
                  activebackground="#3e3e42", relief="flat", width=9,
                  font=("Segoe UI", 8)).grid(row=0, column=col, padx=2)

    # ------------------------------------------------ actions ----------
    def toggle_mode(self):
        if not self.cli:
            return
        self.mode = "sniff" if self.mode == "drive" else "drive"
        try:
            self.cli.set_mode(self.mode)
        except OSError:
            pass

    def ts(self, x):
        if self.cli:
            try:
                self.cli.set_timescale(x)
                self._ts_val = float(x)
            except OSError:
                pass

    def reset(self):
        if self.cli:
            try:
                self.cli.reset_level()
            except OSError:
                pass

    def load(self):
        if self.cli:
            try:
                self.cli.load_level(SCENE)
            except OSError:
                pass

    # ------------------------------------------------ polling ----------
    def tick(self):
        if self.cli is None:
            try:
                self.cli = StateBridgeClient().connect()
            except OSError:
                self.vars["conn"].set("未连接 (游戏未启动?)")
                self._clear()
                self.root.after(1000, self.tick)
                return
        try:
            st = self.cli.get_state()
        except (OSError, ValueError):
            self.cli = None
            self.root.after(500, self.tick)
            return

        self.vars["conn"].set("已连接  seq=%d" % st.get("seq", 0))
        self.vars["scene"].set("%s %s" % (st.get("scene", "?"),
                                          "IN ROUND" if st.get("in_round") else ""))
        r = st.get("round", {})
        if st.get("in_round"):
            self.vars["round"].set("%3.0fs  score=%d" % (r.get("time_remaining", -1),
                                                         r.get("score", 0)))
        else:
            self.vars["round"].set("-")
        self.vars["mode"].set(self.mode)
        self.vars["timescale"].set("x%.2f" % self._ts)

        for i in range(2):
            players = st.get("players", [])
            if i < len(players):
                p = players[i]
                held = p.get("held")
                held_s = held["name"] if held else "-"
                inp = p.get("input") or {}
                mv = inp.get("move", [0, 0])
                self.vars[f"p{i}"].set("g=%s held=%s in=[%+.1f,%+.1f]%s%s%s" % (
                    p.get("grid"), held_s, mv[0], mv[1],
                    " P" if inp.get("pickup") else "",
                    " U" if inp.get("use") else "",
                    " D" if inp.get("dash") else ""))
            else:
                self.vars[f"p{i}"].set("-")

        pots = st.get("cookers", [])
        self.vars["pots"].set("  ".join(
            "%s %.0f/%.0f[%d]" % (c.get("state", "?")[:4], c.get("progress", 0),
                                  c.get("cook_time", 0), len(c.get("contents") or []))
            for c in pots) or "-")

        self._poll_training()

        orders = st.get("orders", [])
        for i in range(6):
            if i < len(orders):
                o = orders[i]
                pct = o.get("remaining", 0) / max(o.get("lifetime", 1), 1)
                color_mark = "!" if pct < 0.25 else " "
                self.vars[f"order{i}"].set("%s %-16s %3.0fs" % (color_mark, o.get("recipe", "?"),
                                                                o.get("remaining", 0)))
            else:
                self.vars[f"order{i}"].set("")

        self.root.after(250, self.tick)

    # ------------------------------------------------ training readout ----------
    def _poll_training(self):
        """Read TensorBoard event files every ~5s (cheap tail-parse)."""
        now = time.time()
        if now - getattr(self, "_tr_last", 0) < 5:
            return
        self._tr_last = now
        logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "..", "logs", "ppo_sushi")
        try:
            import glob
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            files = sorted(glob.glob(os.path.join(logdir, "events.*")),
                           key=os.path.getmtime)
            if not files:
                return
            ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
            ea.Reload()

            def last(tag):
                try:
                    ev = ea.Scalars(tag)
                    return ev[-1].value if ev else None
                except KeyError:
                    return None

            step = last("perf/sps")
            scores = ea.Scalars("episode/score") if "episode/score" in ea.Tags().get("scalars", []) else []
            recent = [e.value for e in scores[-5:]]
            kl = last("train/approx_kl")
            evv = last("train/explained_var")
            if scores:
                self.vars["tr_step"].set("%d steps" % scores[-1].step)
                self.vars["tr_score"].set("/".join("%.0f" % s for s in recent))
            if kl is not None:
                self.vars["tr_kl"].set("kl=%.3f ev=%.2f" % (kl, evv or 0))
        except Exception:
            pass

    @property
    def _ts(self):
        return getattr(self, "_ts_val", 1.0)

    def _clear(self):
        for k in ("scene", "round", "p0", "p1", "pots"):
            self.vars[k].set("-")
        for i in range(6):
            self.vars[f"order{i}"].set("")


if __name__ == "__main__":
    Panel()
