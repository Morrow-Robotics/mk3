"""A tabletop world: one product, one carton, one tool.

Deliberately analytic (no physics engine, numpy only) so the whole stack runs
anywhere, deterministically, in milliseconds. It is faithful to exactly the
things the skill program reasons about: where the product footprint is, how
tall it is, whether a top-down suction grasp would seal, and whether a released
product lands in the carton. It is honest about being a simulator — the real
world's slips and wrinkles are the reason the bench exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import Transform, frame, translation, wrap_angle, yaw_of


def rect_aabb(cx: float, cy: float, hx: float, hy: float, yaw: float) -> np.ndarray:
    """Axis-aligned bounding box of a rotated, centered rectangle footprint."""
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    aw = hx * c + hy * s
    ah = hx * s + hy * c
    return np.array([cx - aw, cy - ah, cx + aw, cy + ah], dtype=np.float64)


@dataclass
class Product:
    kind: str  # "box" | "cylinder" | "pouch"
    hx: float  # footprint half-extent x (radius for cylinder)
    hy: float  # footprint half-extent y (radius for cylinder)
    height: float  # full height
    grasp_radius: float  # radius of the reliable top-center seal region
    cx: float
    cy: float
    yaw: float
    # Object-frame preferred grasp yaw. When set, a top-down grasp seals
    # reliably only when the tool yaw aligns with (product.yaw + this); the
    # 180-flipped grasp — geometrically identical — seals worse. This is the
    # signal the analytic ranker structurally ignores and the learned one finds.
    seal_yaw_pref: float | None = None

    def footprint(self) -> np.ndarray:
        return rect_aabb(self.cx, self.cy, self.hx, self.hy, self.yaw)


@dataclass
class Carton:
    cx: float
    cy: float
    yaw: float
    hx: float  # inner opening half-extent x
    hy: float  # inner opening half-extent y
    wall_z: float  # base-frame z of the carton rim
    floor_z: float  # base-frame z of the inner floor

    def opening(self) -> np.ndarray:
        return rect_aabb(self.cx, self.cy, self.hx, self.hy, self.yaw)

    def pose(self) -> Transform:
        return frame((self.cx, self.cy, self.floor_z), self.yaw)


class World:
    """Mutable world state. The SimRobot mutates it; the SimPerceiver reads it."""

    GRASP_Z_TOL = 0.02  # how close to the top the tool must be to seal
    SAFE_Z = 0.35  # travel/park height above the table
    YAW_SLIP_PENALTY = 0.7  # extra slip when the grasp yaw is fully misaligned

    def __init__(self, product: Product, carton: Carton, table_height: float = 0.0,
                 workspace=(-0.4, 0.4, -0.4, 0.4), force_fail_seals: int = 0,
                 slip_prob: float = 0.0, perception_dropout_prob: float = 0.0,
                 occlusion_prob: float = 0.0, distractors=None, occupied=None, rng=None):
        self.product = product
        self.carton = carton
        self.table_height = table_height
        self.workspace = workspace  # (xmin, xmax, ymin, ymax)
        self.force_fail_seals = force_fail_seals  # scripted misses (bench demo)
        self.slip_prob = slip_prob  # random suction miss on an otherwise-good seal
        self.perception_dropout_prob = perception_dropout_prob  # low-confidence frames
        self.occlusion_prob = occlusion_prob  # partial mask -> jittered centroid/footprint
        self.distractors = list(distractors) if distractors else []  # non-target clutter
        self.occupied = [np.asarray(f) for f in occupied] if occupied else []  # placed items in carton
        self.rng = rng  # seeded RandomState for reproducible slips/dropouts/occlusion
        self.ee_pose: Transform = frame((0.0, 0.0, self.SAFE_Z), 0.0)
        self.vacuum_on = False
        self.attached = False
        self._grasp_offset = np.zeros(2)  # product_xy - ee_xy at seal time
        self.in_carton = False
        self.flagged = False
        self._grasp_attempts = 0

    # --- queries ---------------------------------------------------------
    def product_top_z(self) -> float:
        if self.attached:
            return float(translation(self.ee_pose)[2])
        if self.in_carton:
            return self.carton.floor_z + self.product.height
        return self.table_height + self.product.height

    def product_centroid(self) -> np.ndarray:
        return np.array([self.product.cx, self.product.cy, self.product_top_z()])

    def in_workspace(self, xy) -> bool:
        x0, x1, y0, y1 = self.workspace
        return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)

    # --- mutations (called by SimRobot) ----------------------------------
    def set_ee(self, pose: Transform) -> None:
        self.ee_pose = pose.copy()
        if self.attached:
            ee_xy = translation(pose)[:2]
            self.product.cx = float(ee_xy[0] + self._grasp_offset[0])
            self.product.cy = float(ee_xy[1] + self._grasp_offset[1])

    def try_seal(self) -> bool:
        """Attempt a suction seal at the current tool pose. Returns success."""
        self.vacuum_on = True
        self._grasp_attempts += 1
        if self._grasp_attempts <= self.force_fail_seals:
            return False  # scripted miss to exercise recovery on the bench demo
        ee = translation(self.ee_pose)
        dxy = float(np.linalg.norm(ee[:2] - np.array([self.product.cx, self.product.cy])))
        dz = abs(ee[2] - self.product_top_z())
        if dxy <= self.product.grasp_radius and dz <= self.GRASP_Z_TOL:
            eff_slip = self.slip_prob
            pref = self.product.seal_yaw_pref
            if pref is not None:
                align = float(np.cos(wrap_angle(yaw_of(self.ee_pose) - self.product.yaw - pref)))
                eff_slip = min(0.95, self.slip_prob + self.YAW_SLIP_PENALTY * (1.0 - align) / 2.0)
            if self.rng is not None and eff_slip > 0.0 and self.rng.random() < eff_slip:
                return False  # good geometry, but the film wrinkled and the seal slipped
            self.attached = True
            self.in_carton = False
            self._grasp_offset = np.array([self.product.cx, self.product.cy]) - ee[:2]
            return True
        return False

    def release_seal(self) -> None:
        self.vacuum_on = False
        if self.attached:
            self.attached = False
            inside = _xy_in_bbox((self.product.cx, self.product.cy), self.carton.opening())
            self.in_carton = bool(inside)

    def signal(self) -> float:
        if self.attached:
            return 0.1  # strong vacuum
        return 0.5 if self.vacuum_on else 0.95

    def reset_episode(self) -> None:
        self._grasp_attempts = 0
        self.flagged = False


def _xy_in_bbox(xy, bbox) -> bool:
    x0, y0, x1, y1 = bbox
    return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)
