"""Reward shaping for the sushi pipeline.

`compute_reward(prev, cur, ctx)` diffs two consecutive bridge states and
returns (total, parts). `parts` doubles as a metrics channel: keys starting
with "metric_" are logged but excluded from the reward sum.

All weights live in REWARD_WEIGHTS — tune there, not in the logic.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav import board_items, nav_targets, plate_food_count, content_names

REWARD_WEIGHTS = {
    "score": 0.05,        # per score point (served dish = ~20-40 pts)
    "pot_add": 0.3,       # ingredient added to pot
    "pot_start": 0.2,     # cooking started
    "pot_cooked": 0.3,    # cooking finished
    "held_ing": 0.05,     # symmetric: +pickup / -putdown (pot_add nets +)
    "held_utensil": -0.1, # grabbed a pot/extinguisher
    "held_plate": 0.05,   # symmetric: +took / -put down
    "chopped": 1.0,       # held item became Chopped* (0.5->1.0: chopping
                          # boycotted for 3h — chop_done 0/50 — because the
                          # scoop-churn loop out-paid it; completion-only,
                          # cannot be farmed without real chopped output)
    "plated": 0.6,        # held plate gained its first contents (1.0->0.6:
                          # scooping is over-learned — plated hit 7/rollout,
                          # plates churn scoop->dump->rescope because emptying
                          # resets "first contents". The churn cycle must not
                          # out-earn a real serve)
    "plate_add": 0.3,     # any plate (held or on a surface) gained a content —
                          # assembly needs rice+nori+filling; plating only the
                          # first content left additions 2/3 unrewarded
    "plate_waste": -1.0,  # per plate-food item LOST without a score increase.
                          # -0.2->-1.0: churn economics. A scoop->dump cycle
                          # earned plated+plate_add-waste = +1.1 at -0.2, so
                          # dumping was rational (waste kept climbing:
                          # -0.56 -> -0.79). At -1.0 a churn cycle nets
                          # 0.6+0.3-1.0 < 0 while a real 3-layer serve keeps
                          # ~+3.5 (adds+complete+score).
    "plate_complete": 1.0, # a plate reaching 3 recipe items (rice+nori+
                          # chopped filling = a servable sushi). 2-layer
                          # plates (rice+nori) appeared in run42 but the 3rd
                          # layer starves: scooping pays ~1.3 per INTERACT
                          # while a full chop pays ~1.5 over ~20 use-hold
                          # steps, so the policy rationally skips chopping —
                          # yet orders are undeliverable without it.
    "board_add": 0.3,     # raw filling placed on a chopping board
    "chop_prog": 0.5,     # per unit of chop progress on a board (dense;
                          # "chopped" only fires when the item is picked up,
                          # so without this chop completion is invisible)
    "chop_done": 0.8,     # a board item's progress crossing ~1.0 — finishing
                          # the chop pays immediately, not only on pickup
                          # (partial-chop farming otherwise beats finishing).
                          # 0.5->0.8: run42 collapsed chopping (chop_done 0/50
                          # for 2h) — per-effort it lost to scooping, but the
                          # 3rd plate layer is undeliverable without it.
    "chop_stance": 0.0,   # RETIRED (0): scaffolding from before chop_prog —
                          # policy became a stance farmer (mean +31/rollout,
                          # 50/50, entropy collapsed to 2.1, plated 0/50).
                          # Dense progress + completion carry chopping now.
    "scoop_stance": 0.02, # per step: empty plate in hand next to a cooked pot
                          # (0.05->0.02: same farming shape as chop_stance —
                          # stance climbed 27/50 while plated conversion
                          # stayed ~1/8; guidance yes, destination no)
    "floor_drop": -0.05,  # ingredient dropped on the floor (anti-churn)
}

# chop stance gate: must stand within this range and face the board
_STANCE_MIN_DIST = 0.01
_STANCE_MAX_DIST = 1.6
_STANCE_FACING_DOT = 0.5
# an item below this height after a putdown counts as floor-dropped
_FLOOR_Y = 0.3
# look for the dropped item within this radius of the chef
_DROP_RADIUS = 2.0


def held_name(player_state):
    h = player_state.get("held")
    return h.get("name") if h else None


def is_ingredient(name):
    # SushiRice, Cucumber, ChoppedCucumber, fish, nori, ... — anything
    # that is not cookware/tableware
    return name is not None and not name.startswith(("utensil_", "equipment_"))


def find_item_near(items, name, pos, max_dist):
    if not pos:
        return None
    best, best_d = None, max_dist
    for it in items:
        if it.get("name") != name or it.get("kind") != "carryable":
            continue
        p = it.get("pos")
        if not p or len(p) < 3:
            continue
        d = ((p[0] - pos[0]) ** 2 + (p[2] - pos[2]) ** 2) ** 0.5
        if d < best_d:
            best, best_d = it, d
    return best


def nearest_dist(pos, targets):
    if not pos:
        return 0.0
    return min((pos[0] - t[0]) ** 2 + (pos[2] - t[2]) ** 2 for t in targets) ** 0.5


def _held_names(st):
    """Names of items currently in a player's hand. The items[] scan
    (PhysicalAttachment, 0.3s refresh) INCLUDES held items — they attach to
    the player's hand — so without this exclusion every loaded plate counts
    twice while carried: phantom plate_add on pickup, phantom plate_waste
    on putdown (observed: every 'loss' event was a world+held pair at the
    same position — one plate, not two)."""
    names = set()
    for p in st.get("players", []):
        h = p.get("held")
        if h and h.get("name"):
            names.add(h["name"])
    return names


def _plate_food_counts(st):
    """Per-plate recipe-food counts (held + loose), held plates counted once."""
    held = _held_names(st)
    counts = []
    for p in st.get("players", []):
        h = p.get("held")
        if h and h.get("is_plate"):
            counts.append(plate_food_count(h.get("contents")))
    for it in st.get("items", []):
        if it.get("kind") == "plate" and it.get("name") not in held:
            counts.append(plate_food_count(it.get("contents")))
    return counts


def _total_plate_contents(st):
    """Total RECIPE-RELEVANT ingredient count across every visible plate
    (held + loose). Nested composite nodes are flattened and raw
    ingredients excluded — anything else rewards plate-poisoning."""
    return sum(_plate_food_counts(st))


def _pot_rice_count(st):
    """Rice sitting in ANY pot, each pot counted once. A pot on a burner
    appears in BOTH cookers[] and items[] (dedup by cooker proximity); a
    carried pot appears only via its items[] ghost (players[].held also
    carries contents, so counting items-only avoids that double-count)."""
    total = 0
    cooker_pos = [c["pos"] for c in st.get("cookers", []) if c.get("pos")]
    for it in st.get("items", []):
        if it.get("kind") != "utensil" or not it.get("pos"):
            continue
        if any(abs(it["pos"][0] - cp[0]) < 1.0 and abs(it["pos"][2] - cp[2]) < 1.0
               for cp in cooker_pos):
            continue  # on a burner — counted via cookers[] below
        total += sum(1 for n in content_names(it.get("contents"))
                     if n == "SushiRice")
    for c in st.get("cookers", []):
        total += sum(1 for n in content_names(c.get("contents"))
                     if n == "SushiRice")
    return total


def _board_chop_stats(items, chop_boards):
    """(sum, max) of work_progress (0..1) over items on chopping boards."""
    total, best = 0.0, 0.0
    for it in items:
        wp = it.get("work_progress")
        if not wp:
            continue
        p = it.get("pos")
        if not p:
            continue
        for b in chop_boards:
            if abs(p[0] - b[0]) < 1.0 and abs(p[2] - b[2]) < 1.0:
                total += wp
                if wp > best:
                    best = wp
                break
    return total, best


def compute_reward(prev, cur, ctx):
    """Diff two states -> (reward, parts).
    ctx: {"chop_boards", "plate_station", "crates", "cooker_pos"}
    (from level_config.get_level + the station census)."""
    W = REWARD_WEIGHTS
    parts = {"score": 0.0, "pot_add": 0.0, "pot_start": 0.0, "pot_cooked": 0.0,
             "held_ing": 0.0, "held_utensil": 0.0, "held_plate": 0.0,
             "chopped": 0.0, "plated": 0.0, "plate_add": 0.0, "plate_waste": 0.0,
             "plate_complete": 0.0, "board_add": 0.0, "chop_prog": 0.0,
             "chop_done": 0.0, "chop_stance": 0.0, "scoop_stance": 0.0, "floor_drop": 0.0, "nav": 0.0,
             "metric_pickup_ing": 0.0}
    pr, cr = prev.get("round", {}), cur.get("round", {})
    parts["score"] = (cr.get("score", 0) - pr.get("score", 0)) * W["score"]
    # chopping: raw filling placed on a board. SYMMETRIC: removing a raw
    # item from a board without a chop transform un-earns the placement —
    # positive-only made board ping-pong (place b1, pick up, place b2, ...)
    # an infinite +0.3 farm (observed run48: +4.1/rollout, 50/50, chopping
    # dead). A raw item vanishing as a Chopped* appears IS the completion —
    # exempt, or finishing a chop would be penalized.
    prev_boards = board_items(prev.get("items", []), ctx["chop_boards"])
    cur_boards = board_items(cur.get("items", []), ctx["chop_boards"])
    prev_board, cur_board = len(prev_boards), len(cur_boards)
    board_delta = cur_board - prev_board
    if board_delta > 0:
        parts["board_add"] += W["board_add"] * board_delta
    elif board_delta < 0:
        prev_chopped = sum(1 for i in prev.get("items", [])
                           if (i.get("name") or "").startswith("Chopped"))
        cur_chopped = sum(1 for i in cur.get("items", [])
                          if (i.get("name") or "").startswith("Chopped"))
        if cur_chopped <= prev_chopped:
            parts["board_add"] += W["board_add"] * board_delta  # negative
    # dense chop progress: sum of work_progress over items on boards.
    # Positive deltas only — removing/losing an item must not claw reward
    # back, and the completion bonus comes from "chopped" on pickup.
    prev_sum, prev_max = _board_chop_stats(prev.get("items", []), ctx["chop_boards"])
    cur_sum, cur_max = _board_chop_stats(cur.get("items", []), ctx["chop_boards"])
    if cur_sum > prev_sum:
        parts["chop_prog"] += W["chop_prog"] * (cur_sum - prev_sum)
    if prev_max < 0.99 <= cur_max:
        parts["chop_done"] += W["chop_done"]
    # assembly: total contents across ALL plates (held + surfaces). Waste is
    # judged on plate food + pot rice together: a plate->pot pour keeps the
    # rice (re-scoopable), only food that leaves BOTH is truly wasted
    # (partial serves, binning) — unless score increased (legit serve).
    plate_delta = _total_plate_contents(cur) - _total_plate_contents(prev)
    if plate_delta > 0:
        parts["plate_add"] += W["plate_add"] * plate_delta
    else:
        food_delta = (plate_delta
                      + _pot_rice_count(cur) - _pot_rice_count(prev))
        if food_delta < 0 and parts["score"] == 0.0:
            parts["plate_waste"] += W["plate_waste"] * (-food_delta)
    # a plate reaching a servable 3 items (crossing, not per-step)
    if (max(_plate_food_counts(prev), default=0) < 3
            <= max(_plate_food_counts(cur), default=0)):
        parts["plate_complete"] += W["plate_complete"]
    # pot pipeline (contents added / cooking progressed / cooked)
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
            parts["pot_add"] += W["pot_add"] * (cn - pn)
        if p["progress"] == 0 and c["progress"] > 0:
            parts["pot_start"] += W["pot_start"]
        if not p["is_cooked"] and c["is_cooked"]:
            parts["pot_cooked"] += W["pot_cooked"]
    # plates visible this step (they move; recompute every step)
    plate_items = [i for i in cur.get("items", [])
                   if i.get("kind") == "plate" and i.get("pos")]
    plates_pos = [i["pos"] for i in plate_items] or [ctx["plate_station"]]
    fullest = max(plate_items, key=lambda i: plate_food_count(i.get("contents")),
                  default=None)
    plates_best = ([fullest["pos"]]
                   if fullest is not None and plate_food_count(fullest.get("contents")) > 0
                   else None)
    # chopped fillings waiting in the world (exclude held ghosts — items[]
    # includes what players carry; chasing a carried item is a dogpile)
    held_now = _held_names(cur)
    chopped_pos = [i["pos"] for i in cur.get("items", [])
                   if (i.get("name") or "").startswith("Chopped")
                   and i.get("pos") and i.get("name") not in held_now]
    # held-item transitions (symmetric so pick/drop cycling nets 0)
    pp, cp = prev.get("players", []), cur.get("players", [])
    for i in range(min(len(pp), len(cp))):
        ph, ch = pp[i].get("held"), cp[i].get("held")
        phn = ph.get("name") if ph else None
        chn = ch.get("name") if ch else None
        if phn != chn:
            if chn is not None and is_ingredient(chn):
                parts["held_ing"] += W["held_ing"]               # picked up an ingredient
                parts["metric_pickup_ing"] += 1.0                # count (not a reward)
                # pay each distinct chopped item ONCE per episode — the
                # chopped-preload cucumber was being yo-yo'd (pick up +1.0,
                # put down, repeat: +6/rollout with zero plates assembled)
                if (chn.startswith("Chopped")
                        and not (phn or "").startswith("Chopped")
                        and chn not in ctx.setdefault("chopped_paid", set())):
                    ctx["chopped_paid"].add(chn)
                    parts["chopped"] += W["chopped"]             # chopping paid off
            if phn is not None and is_ingredient(phn) and chn is None:
                parts["held_ing"] -= W["held_ing"]               # put it down
                # anti-churn: dropped on the FLOOR (not a station) costs
                # extra — pick/drop cycling used to net zero
                it = find_item_near(cur.get("items", []), phn,
                                    cp[i].get("pos"), _DROP_RADIUS)
                if it is not None and (it.get("pos") or [0, 1, 0])[1] < _FLOOR_Y:
                    parts["floor_drop"] += W["floor_drop"]
            if chn is not None and chn.startswith("utensil_"):
                parts["held_utensil"] += W["held_utensil"]
            p_plate = bool(ph and ph.get("is_plate"))
            c_plate = bool(ch and ch.get("is_plate"))
            if c_plate and not p_plate:
                parts["held_plate"] += W["held_plate"]           # took a plate
            elif p_plate and not c_plate:
                parts["held_plate"] -= W["held_plate"]           # put the plate down
        # plating: an EMPTY held plate gains contents while held (the scoop).
        # NOT "picked up an already-loaded plate" — that refires on every
        # re-pickup of a loaded plate (observed: plated 9.6/rollout with
        # plate_waste 0 — a pickup yo-yo, no food moved at all).
        if (ph and ph.get("is_plate") and not ph.get("contents")
                and ch and ch.get("is_plate") and ch.get("contents")):
            parts["plated"] += W["plated"]
        # navigation shaping: delta distance to the recipe-relevant target
        targets, w = nav_targets(ch, cur.get("cookers", []), plates_pos,
                                 cur_boards, ctx, plates_best=plates_best,
                                 chopped_pos=chopped_pos)
        if targets:
            d_prev = nearest_dist(pp[i].get("pos"), targets)
            d_cur = nearest_dist(cp[i].get("pos"), targets)
            parts["nav"] += w * (d_prev - d_cur)
        # chopping stance: USE held while FACING a board that has a raw
        # filling on it — the facing gate kills the stance-farming exploit
        # (holding use with your back to the board never progresses the
        # chop but used to pay per-step forever)
        if cur_board > 0:
            inp = cp[i].get("input") or {}
            pos = cp[i].get("pos")
            fwd = cp[i].get("fwd") or [0.0, 0.0, 1.0]
            if inp.get("use") and pos:
                for b in cur_boards:
                    dx, dz = b[0] - pos[0], b[2] - pos[2]
                    dist = (dx * dx + dz * dz) ** 0.5
                    if (_STANCE_MIN_DIST < dist < _STANCE_MAX_DIST
                            and (fwd[0] * dx + fwd[2] * dz) / dist > _STANCE_FACING_DOT):
                        parts["chop_stance"] += W["chop_stance"]
                        break
        # scoop stance: holding an EMPTY plate within reach of a cooked pot —
        # the exact setup for scooping rice (mirrors chop_stance)
        if ch and ch.get("is_plate") and not ch.get("contents"):
            pos = cp[i].get("pos")
            if pos:
                for c in cur.get("cookers", []):
                    cpos = c.get("pos")
                    if not c.get("is_cooked") or not cpos:
                        continue
                    d = ((cpos[0] - pos[0]) ** 2 + (cpos[2] - pos[2]) ** 2) ** 0.5
                    if d < _STANCE_MAX_DIST:
                        parts["scoop_stance"] += W["scoop_stance"]
                        break
    return sum(v for k, v in parts.items() if not k.startswith("metric_")), parts
