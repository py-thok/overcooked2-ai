"""Layout-agnostic observation encoder: bridge state JSON -> grid tensor + globals.

Shared by the real-game env and (later) the simulator pretrain, so the policy
sees one format everywhere. Layout comes from the station census (STATIONS
command), dynamic entities from the per-frame state (GET).

Tensor layout (C, H, W) over the level's native grid (GridManager indices,
offset so min index = 0). Channels v1:

  0  station_counter      7  pot_progress (0..1 fraction cooked)
  1  station_chop         8  item_carryable (loose ingredients/items)
  2  station_cooker       9  item_plate
  3  station_crate       10  item_utensil (pots not on a burner)
  4  station_serving     11  player_0
  5  station_wash        12  player_1
  6  station_plate_ret   13  plate_food_depth (recipe items on plate / 3)
 14  item_work_progress (chop progress of board items)
 15  pot_contents (count/4)  16  pot_is_cooked  17  pot_is_burning

v2 added channels 13/14: without them a rice-loaded plate and an empty
plate were pixel-identical on the grid (assembly concentration had no
signal), and a half-chopped cucumber looked identical to a raw one (chop
resumption had no signal). Checkpoints resume via zero-padded conv1.

v3 added channels 15-17: a rice-loaded pot and an empty pot differed only
in the progress channel, and "cooked, waiting to scoop" was
indistinguishable from "still cooking at progress 1.0".

Globals vector v3: [time_remaining_norm, score_norm,
                    per player: held_kind_onehot(4) + plate_contents/4,
                    order slots x N: recipe_id_onehot(K) + remaining_norm,
                    per player: fwd_x, fwd_z, subcell_off_x, subcell_off_z,
                    per player: held_identity_onehot(10),
                    is_most_urgent_onehot(N), min_remaining_norm]
New dims are appended AFTER the original 36; the agent one-hot is
concatenated downstream (env._obs / bc_train) and stays last.
Checkpoints resume via zero-spliced trunk columns (ppo_train._load_compat).
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav import content_names, plate_food_count

N_CHANNELS = 18
MAX_ORDERS = 6
RECIPES = ["Sushi_Fish", "Sushi_Cucumber"]  # extend as needed
HELD_IDS = ["SushiRice", "Seaweed", "Cucumber", "SushiFish",
            "ChoppedCucumber", "ChoppedSushiFish"]

_NAME_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


def norm_name(name):
    """Strip Unity instance suffixes: 'countertop_01 (32)' -> 'countertop_01'."""
    return _NAME_SUFFIX.sub("", name)


def classify_station(name):
    n = norm_name(name).lower()
    if "chopping" in n:
        return 1
    if "cooker" in n or "fryer" in n or "oven" in n:
        return 2
    if "dispensercrate" in n or "crate" in n:
        return 3
    if "plate_station" in n:
        return 4  # serving hatch (ServerPlateStation.FoodDelivered)
    if "washing" in n or "drying" in n or "sink" in n:
        return 5
    if "plate_return" in n:
        return 6
    return 0  # countertops, bins, anything else stand-on-able



class LevelGrid:
    """Grid geometry for one level, derived from stations + state."""

    def __init__(self, stations):
        xs = [s["grid"][0] for s in stations if s.get("grid")]
        zs = [s["grid"][2] for s in stations if s.get("grid")]
        self.x0, self.z0 = min(xs) - 1, min(zs) - 1
        self.w = max(xs) - self.x0 + 2
        self.h = max(zs) - self.z0 + 2
        # world = pitch * grid + origin, least-squares fit per axis over the
        # station census (every station carries both grid and pos) — pitch is
        # ~1.2 m but never hardcode level geometry
        pairs = [(s["grid"], s["pos"]) for s in stations
                 if s.get("grid") and s.get("pos")]
        gx = np.array([g[0] for g, _ in pairs], np.float64)
        px = np.array([p[0] for _, p in pairs], np.float64)
        gz = np.array([g[2] for g, _ in pairs], np.float64)
        pz = np.array([p[2] for _, p in pairs], np.float64)
        self.pitch_x, self.origin_x = np.polyfit(gx, px, 1)
        self.pitch_z, self.origin_z = np.polyfit(gz, pz, 1)
        assert 0.8 < abs(self.pitch_x) < 2.0, f"bad pitch_x {self.pitch_x}"
        assert 0.8 < abs(self.pitch_z) < 2.0, f"bad pitch_z {self.pitch_z}"

    def idx(self, grid):
        return grid[2] - self.z0, grid[0] - self.x0  # (row, col)

    def contains(self, grid):
        r, c = self.idx(grid)
        return 0 <= r < self.h and 0 <= c < self.w

    def cell_center(self, grid):
        return (self.pitch_x * grid[0] + self.origin_x,
                self.pitch_z * grid[2] + self.origin_z)


class ObsEncoder:
    def __init__(self, stations):
        self.level = LevelGrid(stations["stations"])
        self.static = np.zeros((N_CHANNELS, self.level.h, self.level.w), np.float32)
        for s in stations["stations"]:
            if not s.get("grid") or not self.level.contains(s["grid"]):
                continue
            ch = classify_station(s["name"])
            r, c = self.level.idx(s["grid"])
            self.static[ch, r, c] = 1.0

    def encode(self, st):
        t = np.copy(self.static)
        lv = self.level

        for ck in st.get("cookers", []):
            if ck.get("grid") and lv.contains(ck["grid"]):
                r, c = lv.idx(ck["grid"])
                frac = ck["progress"] / ck["cook_time"] if ck["cook_time"] > 0 else 0.0
                t[7, r, c] = max(t[7, r, c], min(frac, 2.0) / 2.0)
                t[15, r, c] = max(t[15, r, c],
                                  min(len(ck.get("contents") or []), 4) * 0.25)
                if ck.get("is_cooked"):
                    t[16, r, c] = 1.0
                if ck.get("is_burning"):
                    t[17, r, c] = 1.0

        for it in st.get("items", []):
            if not it.get("grid") or not lv.contains(it["grid"]):
                continue
            r, c = lv.idx(it["grid"])
            ch = {"plate": 9, "utensil": 10}.get(it.get("kind"), 8)
            t[ch, r, c] = 1.0
            # plate food depth: which plate actually carries the assembly
            if it.get("kind") == "plate":
                t[13, r, c] = max(t[13, r, c],
                                  min(plate_food_count(it.get("contents")) / 3.0, 1.0))
            # chop progress: a half-chopped item is worth resuming
            wp = it.get("work_progress")
            if wp:
                t[14, r, c] = max(t[14, r, c], min(float(wp), 1.0))

        for i, p in enumerate(st.get("players", [])[:2]):
            if p.get("grid") and lv.contains(p["grid"]):
                r, c = lv.idx(p["grid"])
                t[11 + i, r, c] = 1.0

        return t

    def encode_globals(self, st, time_limit=210.0):
        r = st.get("round", {})
        tl = r.get("time_limit") if r.get("time_limit", -1) > 0 else time_limit
        g = [r.get("time_remaining", 0) / tl, r.get("score", 0) / 500.0]
        players = st.get("players", [])
        for i in range(2):
            held = None
            if i < len(players):
                held = players[i].get("held")
            g += _held_onehot(held["name"] if held else None)
            # plate contents matter: empty plate vs loaded plate look the same by name
            # (flattened — plate contents nest inside one composite wrapper node)
            n_contents = len(content_names(held.get("contents"))) if held else 0
            g.append(min(n_contents, 4) / 4.0)
        orders = st.get("orders", [])[:MAX_ORDERS]
        for k in range(MAX_ORDERS):
            if k < len(orders):
                o = orders[k]
                g += _recipe_onehot(o.get("recipe"))
                g.append(o.get("remaining", 0) / max(o.get("lifetime", 1), 1))
            else:
                g += [0.0] * (len(RECIPES) + 2)  # recipe onehot + remaining

        # v3 additions (appended after the original 36 — see module docstring)
        # player motion: facing + sub-cell offset (the grid is 1.2m cells;
        # within-cell position is invisible in the grid tensor)
        for i in range(2):
            p = players[i] if i < len(players) else None
            if p and p.get("pos") and p.get("grid"):
                fwd = p.get("fwd") or [0.0, 0.0, 1.0]
                cx, cz = self.level.cell_center(p["grid"])
                ox = (p["pos"][0] - cx) * (2.0 / self.level.pitch_x)
                oz = (p["pos"][2] - cz) * (2.0 / self.level.pitch_z)
                g += [fwd[0], fwd[2], _clamp(ox), _clamp(oz)]
            else:
                g += [0.0, 1.0, 0.0, 0.0]
        # held-item specific identity (the coarse onehot above can't tell a
        # raw cucumber from a chopped one)
        for i in range(2):
            held = players[i].get("held") if i < len(players) else None
            g += _held_identity(held)
        # order urgency: which slot is closest to expiring + how close
        urg = [0.0] * MAX_ORDERS
        best_k, best_r = -1, 1.0
        for k, o in enumerate(orders):
            rem = o.get("remaining", 0) / max(o.get("lifetime", 1), 1)
            if rem < best_r:
                best_r, best_k = rem, k
        if best_k >= 0:
            urg[best_k] = 1.0
        g += urg
        g.append(best_r if best_k >= 0 else 1.0)  # 1.0 = no orders (neutral)
        return np.asarray(g, np.float32)


def _clamp(x, lo=-1.5, hi=1.5):
    # clamp, don't wrap: collision ejection legitimately overshoots a cell
    return max(lo, min(hi, x))


def _held_identity(held):
    """10-dim one-hot: HELD_IDS(6) + plate + utensil + other + none."""
    v = [0.0] * 10
    if not held:
        v[9] = 1.0
        return v
    name = norm_name(held.get("name") or "")
    if name in HELD_IDS:
        v[HELD_IDS.index(name)] = 1.0
    elif held.get("is_plate"):
        v[6] = 1.0
    elif name.startswith(("utensil_", "equipment_")):
        v[7] = 1.0
    else:
        v[8] = 1.0
    return v


def _held_onehot(name):
    kinds = ["plate", "utensil", "ingredient", "other"]
    v = [0.0] * 4
    if name:
        n = name.lower()
        for i, k in enumerate(kinds):
            if k in n:
                v[i] = 1.0
                break
        else:
            v[3] = 1.0
    return v


def _recipe_onehot(name):
    v = [0.0] * (len(RECIPES) + 1)
    if name in RECIPES:
        v[RECIPES.index(name)] = 1.0
    elif name:
        v[-1] = 1.0
    return v
