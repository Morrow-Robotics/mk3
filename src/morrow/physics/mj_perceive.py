"""MjPerceiver — the Perceiver boundary over a MuJoCo cell.

Reads the physics ground truth into the same mask/footprint `SceneState` the
analytic perceiver produces, so `run_skill` is byte-identical across sim and
physics. Also renders camera frames for the dashboard.
"""

from __future__ import annotations

import time

import numpy as np

import mujoco

from ..geometry import frame, wrap_angle
from ..scene import SceneState
from ..sim.world import rect_aabb
from .world import MjWorld

_PLACEHOLDER = np.zeros((2, 2), dtype=bool)  # mask/depth are unused perception-contract fields


class MjPerceiver:
    def __init__(self, world: MjWorld):
        self.world = world
        self._renderer = None

    def observe(self) -> SceneState:
        w = self.world
        xy = w.product_xy()
        yaw = w.product_yaw()
        centroid = np.array([xy[0], xy[1], w.product_top_z()])
        yaws = [0.0] if w.kind == "cylinder" else [wrap_angle(yaw), wrap_angle(yaw + np.pi)]
        c = w.carton
        return SceneState(
            product_mask=_PLACEHOLDER,
            product_depth=_PLACEHOLDER.astype(float),
            product_centroid=centroid,
            product_footprint=w.product_footprint(),
            product_yaw_candidates=yaws,
            carton_frame=frame((c["cx"], c["cy"], 0.0), 0.0),
            carton_opening=rect_aabb(c["cx"], c["cy"], c["hx"], c["hy"], 0.0),
            gripper_pose=frame(w.ee_pos(), w.ee_yaw()),
            gripper_signal=w.jaw_gap(),
            holding=w.grasped(),
            perception_confidence=0.98,
            timestamp=time.time(),
        )

    def render(self, camera: str = "top", height: int = 240, width: int = 320) -> np.ndarray:
        if self._renderer is None or self._renderer.height != height or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.world.model, height=height, width=width)
        self._renderer.update_scene(self.world.data, camera=camera)
        return self._renderer.render()
