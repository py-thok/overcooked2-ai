"""Vision detectors: game screenshot -> PerceivedState.

Two backends:
  - SimDetector: reads from overcooked-ai renderer/state (for offline validation,
    no X display needed). This is the reference implementation whose OUTPUT FORMAT
    the real detectors must match.
  - RealDetector: template matching / color segmentation / OCR on actual OC2
    screenshots. Requires calibrated templates in assets/templates/. NOT YET
    CALIBRATED — the interface is the deliverable of this scaffold.

The seam between vision and features is PerceivedState: as long as a detector
produces a correct PerceivedState, feature_builder guarantees a training-aligned
96-dim vector (verified exact-match against ground truth over random rollouts).
"""
import sys
sys.path.insert(0, "/root/data/overcooked2-ai/src/perception")
from feature_builder import PerceivedState


class SimDetector:
    """'Perception' from the simulator's own state. Ground truth stand-in."""

    def __init__(self):
        pass

    def detect(self, state) -> PerceivedState:
        return PerceivedState(
            players=[(p.position, p.orientation,
                      p.held_object.name if p.held_object else None)
                     for p in state.players],
            objects=dict(state.objects),
        )


class RealDetector:
    """Template-matching detector for real OC2 screenshots.

    Pipeline per frame:
      1. grid calibration: locate the kitchen grid in the screenshot (fixed
         layout -> can be calibrated once per resolution)
      2. player detection: color-segment chef sprites, snap pixel centroid to
         nearest grid cell, estimate facing from sprite template (4 orientations)
      3. held-item: match small icon above chef's head
      4. pot state: template-match pot sprite per state (empty/filled/cooking/
         ready); cook_time from progress-bar length (linear, needs calibration)
      5. soup ingredient count: count small icons in the bubble above the pot

    All coordinates must be converted to the overcooked-ai grid convention
    ((x, y), origin top-left, x right, y down) before filling PerceivedState.
    """

    def __init__(self, template_dir, grid_calib):
        import glob, os, cv2
        self.templates = {}
        for f in glob.glob(os.path.join(template_dir, "*.png")):
            self.templates[os.path.splitext(os.path.basename(f))[0]] = cv2.imread(f)
        self.grid_calib = grid_calib  # dict: {origin_xy, cell_px, grid_w, grid_h}

    def _snap_to_grid(self, px, py):
        ox, oy = self.grid_calib["origin_xy"]
        cs = self.grid_calib["cell_px"]
        return (round((px - ox) / cs), round((py - oy) / cs))

    def detect(self, frame_bgr) -> PerceivedState:
        raise NotImplementedError(
            "RealDetector requires calibrated templates. "
            "Capture screenshots at target resolution, crop sprites for each "
            "entity class into assets/templates/, fill grid_calib, then "
            "implement the 5 detection steps documented in the class docstring."
        )
