"""What one perception step returns.

`SceneState` is deliberately mask-and-surface centric, not rigid-pose centric.
A deformable pouch has no stable 6-DoF pose; it has a footprint mask, a depth
surface, a centroid, and an *ambiguous* yaw. So yaw is a list of candidates
(a square or wrinkled pouch is symmetric to within 180 degrees), and every
scene carries a perception confidence the feasibility gate can veto on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Transform


@dataclass
class SceneState:
    product_mask: np.ndarray  # boolean HxW, overhead camera
    product_depth: np.ndarray  # HxW, base-frame surface height where masked
    product_centroid: np.ndarray  # (3,) base frame
    product_footprint: np.ndarray  # (4,) [xmin, ymin, xmax, ymax] base-frame
    product_yaw_candidates: list[float]  # symmetry-equivalent yaws (radians)
    carton_frame: Transform  # base-frame carton pose
    carton_opening: np.ndarray  # (4,) [xmin, ymin, xmax, ymax] base-frame footprint
    gripper_pose: Transform  # current T_base_ee
    gripper_signal: float  # raw hardware reading (vacuum: low == holding)
    holding: bool  # hardware verdict: is something attached right now
    perception_confidence: float  # 0..1
    timestamp: float
    uncertainty: dict = field(default_factory=dict)
    occupied_regions: list = field(default_factory=list)  # placed-item footprints in the carton

    @property
    def product_top_z(self) -> float:
        """Base-frame z of the top surface of the product (for grasp depth)."""
        return float(self.product_centroid[2])
