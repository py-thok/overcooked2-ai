"""Recipe-aware navigation shaping (sushi 1-4).

Given what a chef holds, decide where they should head; the reward module
pays a potential-based delta (weight * distance closed) toward the target.

sushi 1-4 recipe routing: raw rice cooks, raw cucumber/fish chop,
nori and chopped fillings go straight onto a plate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def is_choppable(name):
    # raw fillings that must be chopped; rice cooks, nori is direct
    return (name is not None
            and not name.startswith(("utensil_", "equipment_", "Chopped"))
            and name not in ("Seaweed", "SushiRice"))


def content_names(contents):
    """Flatten nested contents nodes -> ingredient names.

    Plate contents serialize as ONE CookedCompositeAssembledNode wrapper
    whose CHILDREN carry the names — len(contents) counts the wrapper, not
    the food. A complete sushi (rice+nori+filling) looked like "1 content",
    which silently broke serve routing (nav), assembly rewards and the
    plate-contents observation feature."""
    names = []
    for n in (contents or []):
        if not isinstance(n, dict):
            continue
        if n.get("name"):
            names.append(n["name"])
        for c in (n.get("children") or []):
            if isinstance(c, dict) and c.get("name"):
                names.append(c["name"])
    return names


def plate_food_count(contents):
    """Recipe-relevant items on a plate: rice, nori, chopped fillings.

    NOTE on names: the assembled node for a chopped filling LOSES its
    "Chopped" prefix — a complete sushi serializes as [SushiRice, Seaweed,
    Cucumber] (serve probe: delivered +63, matched Sushi_Cucumber). Counting
    only Chopped* made the 3rd layer invisible: plate_complete was
    unreachable and nav never routed full plates to the hatch. Accepting
    bare filling names here is safe because the mod REFUSES raw fillings
    onto plates (drop-refuse-plate-poison) — bare "Cucumber"/"SushiFish" in
    plate contents can only arrive chopped via the plate-lock."""
    return sum(1 for n in content_names(contents)
               if n in ("SushiRice", "Seaweed", "Cucumber", "SushiFish")
               or n.startswith("Chopped"))


def board_items(items, chop_boards):
    """Positions of chopping boards holding a raw choppable ingredient."""
    busy = []
    for it in items:
        if it.get("kind") != "carryable" or not is_choppable(it.get("name")):
            continue
        p = it.get("pos")
        if not p:
            continue
        for b in chop_boards:
            if abs(p[0] - b[0]) < 1.0 and abs(p[2] - b[2]) < 1.0:
                busy.append(b)
                break
    return busy


def nav_targets(held, cookers, plates_pos, boards_busy, ctx, plates_best=None,
                chopped_pos=None):
    """Where this chef should head, given what they hold.
    ctx: {"chop_boards", "plate_station", "crates", "cooker_pos"}.
    plates_best: [pos] of the plate with most contents, if any plate is
    loaded — assembly must CONCENTRATE on one plate (nearest-plate routing
    used to scatter seaweed/rice/filling across different plates).
    chopped_pos: positions of Chopped* items in the world — a finished
    filling is the rarest object in the kitchen; fetching it beats starting
    a new chop, and a partial plate should co-locate with it.
    Returns (target_list, weight) or (None, 0)."""
    if held is None:
        # a finished chopped filling beats everything: it is one pickup away
        # from the 3rd plate layer
        if chopped_pos:
            return chopped_pos, 0.02
        # unfinished chopping beats the rest: pull the chef back to
        # the occupied board so the chop can be resumed/finished
        if boards_busy:
            return boards_busy, 0.02
        # rice ready somewhere -> next link in the chain is fetching a
        # plate; without this reroute nothing ever pulls a chef toward
        # the plate station (held_plate was 0/68 episodes in run10)
        cooked = [c["pos"] for c in cookers if c.get("is_cooked") and c.get("pos")]
        if cooked:
            return (plates_pos, 0.02) if plates_pos else (None, 0.0)
        return ctx["crates"], 0.01
    name = held.get("name") or ""
    if held.get("is_plate"):
        n_food = plate_food_count(held.get("contents"))
        if n_food >= 3:
            # complete sushi: ClientPlateStation (the plate station doubles
            # as the delivery hatch — CanAddItem placement triggers delivery).
            # 0.02->0.05: the ~10m carry must visibly out-earn the policy's
            # old habits (nav ran -0.45/rollout — chefs drifted AWAY).
            return [ctx["plate_station"]], 0.05
        if n_food >= 1:
            # partial plate: co-locate with a finished filling if one exists
            # (drop the plate next to it, then the same chef can fetch the
            # chopped item and plate-lock it on); else head for the crates
            # so the chef can put it down and fetch the missing ingredients.
            if chopped_pos:
                return chopped_pos, 0.01
            return ctx["crates"], 0.01
        cooked = [c["pos"] for c in cookers if c.get("is_cooked") and c.get("pos")]
        return (cooked, 0.02) if cooked else (None, 0.0)
    if name.startswith(("utensil_", "equipment_")):
        return None, 0.0
    if name == "SushiRice":
        return ctx["cooker_pos"], 0.02
    if name.startswith("Chopped") or name == "Seaweed":
        if plates_best:
            return plates_best, 0.02
        return (plates_pos, 0.02) if plates_pos else (None, 0.0)
    return ctx["chop_boards"], 0.02  # raw cucumber/fish -> chopping board
