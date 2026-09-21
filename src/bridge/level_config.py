"""Per-level layout constants.

Currently only sushi 1-4 (campaign 1-4) is mapped. To generalize to another
level, add its entry here: board/plate-station coordinates come from the
station census (`work/probes/find_boards.py`) and spawn spots are picked by
hand near the first stations of the recipe chain.
"""

SUSHI_1_4 = {
    "scene": "s_sushi_1_4",
    # chopping boards ("countertop_01_chopping_board_wood_no_edge")
    "chop_boards": [(15.6, 0.0, -2.4), (20.4, 0.0, -2.4)],
    # serve hatch: ClientPlateStation doubles as delivery point
    "plate_station": (15.0, 0.0, -12.0),
    # curriculum spawn (x, z) per chef: P0 before the rice crate (8.4,-2.4),
    # P1 before the first chopping board
    "spawn_spots": [(8.4, -3.6), (15.6, -3.7)],
    # chop curriculum: pre-load the first board with a cucumber so every
    # episode opens with an immediate chop opportunity (chopped stayed 0
    # for hours without this — the stance needs ~20 consecutive use-hold
    # steps, too long to discover by chance)
    "cucumber_crate": (13.2, 0.0, -2.4),
    "cucumber_stand": (13.2, -3.6),   # probes used crate_z - 1.2 successfully
    "board_stand": (15.6, -3.6),      # south of board 1, facing north
    # rice curriculum: start rice cooking at spawn (fetch verified:
    # (8.4,-2.4) dispenses SushiRice; plated stayed ~2/50 because the
    # cooked-rice window almost never aligned with a fetched plate)
    "rice_crate": (8.4, 0.0, -2.4),
    "rice_stand": (8.4, -3.6),        # below the crate, next to P0's spawn
    "rice_cooker": (22.8, 0.0, -4.8), # middle cooker
    "rice_pot_stand": (22.8, -6.0),   # south of that cooker
    # plate preload: P0 spawns holding an empty plate (scoop_stance stayed
    # 0/50 — nobody fetched a plate at all). Plates are LOOSE items on the
    # counters around (14.4,-6.6); the "plate station" is the serve hatch.
    "plate_stand": (14.4, -5.4),      # just north of the plate counters
}

LEVELS = {SUSHI_1_4["scene"]: SUSHI_1_4}


def get_level(scene):
    """Level config for a scene name; falls back to sushi 1-4."""
    return LEVELS.get(scene, SUSHI_1_4)
