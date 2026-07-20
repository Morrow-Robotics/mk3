"""Bench perceiver adapter — the physical implementation of the `Perceiver` boundary.

Skeleton. `observe()` must capture an overhead RGB-D frame, segment the product
and carton (a SAM-class model or a fine-tuned detector), and return a
`SceneState` in the *same* mask/surface representation the sim produces — no
rigid 6-DoF pose. Yaw is returned as symmetry-equivalent candidates because a
pouch's principal axis flips.

The helpers below are the exact fill-in points; each is a small, testable unit
you can develop against recorded frames before going live.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Transform
from ..scene import SceneState

_TODO = "BenchPerceiver: fill against recorded overhead RGB-D frames"


class BenchPerceiver:
    def __init__(self, camera=None, segmenter=None, calibration=None, table_height: float = 0.0):
        # camera: RGB-D capture handle. segmenter: SAM-class model. calibration: intrinsics + T_base_camera.
        self.camera = camera
        self.segmenter = segmenter
        self.calibration = calibration
        self.table_height = table_height

    # --- fill-in points (develop each against saved frames) ------------------
    def _capture(self):
        raise NotImplementedError(f"{_TODO}: return (rgb, depth) from self.camera")

    def _segment(self, rgb, depth):
        raise NotImplementedError(f"{_TODO}: return (product_mask, carton_mask) via self.segmenter")

    def _deproject_centroid(self, mask, depth) -> np.ndarray:
        raise NotImplementedError(f"{_TODO}: masked depth -> base-frame 3D centroid via calibration")

    def _footprint(self, mask, depth) -> np.ndarray:
        raise NotImplementedError(f"{_TODO}: mask -> base-frame [xmin,ymin,xmax,ymax]")

    def _yaw_candidates(self, mask) -> list[float]:
        raise NotImplementedError(f"{_TODO}: principal-axis yaw and its 180-degree flip")

    def _carton(self, mask, depth) -> tuple[Transform, np.ndarray]:
        raise NotImplementedError(f"{_TODO}: carton -> (4x4 frame, opening bbox)")

    def _confidence(self, product_mask, carton_mask, depth) -> float:
        raise NotImplementedError(f"{_TODO}: a 0..1 perception confidence for the fail-closed gate")

    # --- the boundary method -------------------------------------------------
    def observe(self) -> SceneState:
        raise NotImplementedError(
            f"{_TODO}: compose the helpers above into a SceneState. "
            "Grasp is verified by the robot's vacuum sensor, not here."
        )
