"""A simulated overhead RGB-D perceiver that renders a `SceneState` from a `World`.

It produces the same mask-and-surface representation the real perceiver will:
a footprint mask, a depth surface, a centroid, and *ambiguous* yaw candidates.
No rigid 6-DoF pose is assumed, because a pouch does not have one.
"""

from __future__ import annotations

import time

import numpy as np

from ..geometry import wrap_angle
from ..scene import SceneState
from .world import World


class SimPerceiver:
    def __init__(self, world: World, resolution: int = 80):
        self.world = world
        self.res = resolution

    def _render(self):
        w = self.world
        x0, x1, y0, y1 = w.workspace
        xs = np.linspace(x0, x1, self.res)
        ys = np.linspace(y0, y1, self.res)
        gx, gy = np.meshgrid(xs, ys)
        p = w.product
        # rotate grid into the product footprint frame
        c, s = np.cos(-p.yaw), np.sin(-p.yaw)
        lx = c * (gx - p.cx) - s * (gy - p.cy)
        ly = s * (gx - p.cx) + c * (gy - p.cy)
        mask = (np.abs(lx) <= p.hx) & (np.abs(ly) <= p.hy)
        depth = np.where(mask, w.product_top_z(), w.table_height).astype(np.float64)
        return mask, depth

    def observe(self) -> SceneState:
        w = self.world
        mask, depth = self._render()
        p = w.product
        if p.kind == "cylinder":
            yaws = [0.0]  # rotationally symmetric; any grasp yaw is fine
        else:
            yaws = [wrap_angle(p.yaw), wrap_angle(p.yaw + np.pi)]  # 180 ambiguity
        return SceneState(
            product_mask=mask,
            product_depth=depth,
            product_centroid=w.product_centroid(),
            product_footprint=p.footprint(),
            product_yaw_candidates=yaws,
            carton_frame=w.carton.pose(),
            carton_opening=w.carton.opening(),
            gripper_pose=w.ee_pose.copy(),
            gripper_signal=w.signal(),
            holding=w.attached,
            perception_confidence=0.98,
            timestamp=time.time(),
        )
