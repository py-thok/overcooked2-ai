"""Rebuild the 96-dim overcooked-ai featurize_state vector from a *perceived* state.

The perception layer (detectors.py) produces a `PerceivedState` — grid positions,
held objects, pot states. This module turns that into the exact feature vector the
MLP was trained on, by delegating distance computation to the SAME MotionPlanner
the simulator uses (critical: dx/dy are A* path costs, not euclidean).
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from dataclasses import dataclass, field
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld, OvercookedState, PlayerState, ObjectState, SoupState,
)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as OAIEnv

# canonical object/feature order used by featurize_state
ITEM_TYPES = ["onion", "tomato", "dish", "soup", "serving", "empty_counter"]


@dataclass
class PerceivedState:
    """What the vision layer claims to see. All grid coords are (x, y) ints."""
    layout: str = "cramped_room"
    players: list = field(default_factory=list)   # list of (pos, orientation, held_name|None)
    objects: dict = field(default_factory=dict)    # {pos: ObjectState}
    # pot states are just objects on pot cells; included in `objects`


class FeatureBuilder:
    """Convert PerceivedState -> 96-dim vector, ground-truth aligned."""

    def __init__(self, layout_name="cramped_room", horizon=400, num_pots=2):
        self.layout_name = layout_name
        self.num_pots = num_pots
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.env = OAIEnv.from_mdp(self.mdp, horizon=horizon, info_level=0)
        # mlam (motion planner) is built inside env; reused for A* distances
        self.mlam = self.env.mlam

    def to_overcooked_state(self, ps: PerceivedState) -> OvercookedState:
        players = []
        for pos, orient, held in ps.players:
            held_obj = None
            if held is not None:
                # minimal ObjectState for held item
                if held == "soup":
                    held_obj = SoupState(pos, ingredients=[ObjectState("onion", pos) for _ in range(3)])
                else:
                    held_obj = ObjectState(held, pos)
            players.append(PlayerState(pos, orient, held_object=held_obj))
        return OvercookedState(players=players, objects=dict(ps.objects))

    def build(self, ps: PerceivedState, agent_index=0) -> np.ndarray:
        state = self.to_overcooked_state(ps)
        feats = self.mdp.featurize_state(state, self.mlam, num_pots=self.num_pots)
        return np.asarray(feats[agent_index], dtype=np.float32)

    def ground_truth(self, state: OvercookedState, agent_index=0) -> np.ndarray:
        feats = self.mdp.featurize_state(state, self.mlam, num_pots=self.num_pots)
        return np.asarray(feats[agent_index], dtype=np.float32)


if __name__ == "__main__":
    fb = FeatureBuilder()
    st = fb.mdp.get_standard_start_state()
    ps = PerceivedState(
        players=[(p.position, p.orientation,
                  p.held_object.name if p.held_object else None) for p in st.players],
        objects=dict(st.objects),
    )
    v_rebuilt = fb.build(ps)
    v_true = fb.ground_truth(st)
    print("rebuilt shape:", v_rebuilt.shape)
    print("max abs diff vs ground truth:", np.abs(v_rebuilt - v_true).max())
    print("exact match:", np.array_equal(v_rebuilt, v_true))
