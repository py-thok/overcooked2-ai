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
  6  station_plate_ret   13  reserved

Globals vector v1: [time_remaining_norm, score_norm,
                    held_p0_onehot(4), held_p1_onehot(4),
                    order slots: recipe_id_onehot(K) x remaining_norm x N]
"""
import json
import os
import re

import numpy as np

N_CHANNELS = 14
MAX_ORDERS = 6
RECIPES = ["Sushi_Fish", "Sushi_Cucumber"]  # extend as needed

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

    def idx(self, grid):
        return grid[2] - self.z0, grid[0] - self.x0  # (row, col)

    def contains(self, grid):
        r, c = self.idx(grid)
        return 0 <= r < self.h and 0 <= c < self.w


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

        for it in st.get("items", []):
            if not it.get("grid") or not lv.contains(it["grid"]):
                continue
            r, c = lv.idx(it["grid"])
            ch = {"plate": 9, "utensil": 10}.get(it.get("kind"), 8)
            t[ch, r, c] = 1.0

        for i, p in enumerate(st.get("players", [])[:2]):
            if p.get("grid") and lv.contains(p["grid"]):
                r, c = lv.idx(p["grid"])
                t[11 + i, r, c] = 1.0

        return t

    def encode_globals(self, st, time_limit=210.0):
        r = st.get("round", {})
        g = [r.get("time_remaining", 0) / time_limit, r.get("score", 0) / 500.0]
        for i in range(2):
            held = None
            players = st.get("players", [])
            if i < len(players):
                held = players[i].get("held")
            g += _held_onehot(held["name"] if held else None)
        orders = st.get("orders", [])[:MAX_ORDERS]
        for k in range(MAX_ORDERS):
            if k < len(orders):
                o = orders[k]
                g += _recipe_onehot(o.get("recipe"))
                g.append(o.get("remaining", 0) / max(o.get("lifetime", 1), 1))
            else:
                g += [0.0] * (len(RECIPES) + 2)  # recipe onehot + remaining
        return np.asarray(g, np.float32)


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
