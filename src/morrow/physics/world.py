"""MjWorld — a MuJoCo packing cell and the queries the boundaries need.

Holds the model/data and answers: where is the product, where is the tool, how
open are the jaws, and is the product actually pinched between the fingers
(contact-based — the honest parallel-jaw grasp verdict). Grasp success is
physics, not a flag.
"""

from __future__ import annotations

import numpy as np

import mujoco

from ..sim.world import rect_aabb
from .scene import CARTON, build_mjcf

SIZES = {  # product half-extents (hx, hy, hz)
    "box": (0.03, 0.03, 0.03),
    "cylinder": (0.028, 0.028, 0.045),
    # A *stand-up* pouch (tall, narrow) — a real form factor the standard
    # parallel jaw can actually grip. Modeled RIGID: true deformable / flat-bag
    # handling needs soft-body sim and likely a different end-effector (not built).
    "pouch": (0.02, 0.03, 0.05),
}


def _quat_yaw(q) -> float:
    qw, qx, qy, qz = q
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


class MjWorld:
    SAFE_Z = 0.26
    WORKSPACE = (-0.34, 0.40, -0.34, 0.34)

    def __init__(self, kind: str = "box", px: float = -0.15, py: float = 0.0,
                 pyaw: float = 0.0, carton: dict | None = None, size=None):
        self.kind = kind
        self.hx, self.hy, self.hz = size if size is not None else SIZES[kind]
        self.carton = carton or CARTON
        self.model = mujoco.MjModel.from_xml_string(
            build_mjcf(kind, self.hx, self.hy, self.hz, px, py, pyaw, self.carton))
        self.data = mujoco.MjData(self.model)
        self.pid = self.model.body("product").id
        self.palm_id = self.model.body("palm").id
        self.g_prod = self.model.geom("product").id
        self.g_lf = self.model.geom("lfg").id
        self.g_rf = self.model.geom("rfg").id
        self.lf_adr = self.model.joint("lf").qposadr[0]
        self.rf_adr = self.model.joint("rf").qposadr[0]
        self.flagged = False
        self.data.ctrl[:] = 0.0  # fingers open
        self.set_ee((px, py, self.SAFE_Z), pyaw)
        self.settle(200)

    # --- stepping --------------------------------------------------------
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def settle(self, n: int) -> None:
        self.step(n)

    # --- tool control ----------------------------------------------------
    def set_ee(self, xyz, yaw: float) -> None:
        self.data.mocap_pos[0] = np.asarray(xyz, dtype=float)
        self.data.mocap_quat[0] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    def set_jaws(self, closed: float) -> None:
        self.data.ctrl[:] = float(closed)

    # --- queries ---------------------------------------------------------
    def product_xy(self) -> np.ndarray:
        return self.data.xpos[self.pid][:2].copy()

    def product_top_z(self) -> float:
        return float(self.data.xpos[self.pid][2]) + self.hz

    def product_yaw(self) -> float:
        return _quat_yaw(self.data.xquat[self.pid])

    def product_footprint(self) -> np.ndarray:
        xy = self.product_xy()
        return rect_aabb(float(xy[0]), float(xy[1]), self.hx, self.hy, self.product_yaw())

    def ee_pos(self) -> np.ndarray:
        return self.data.xpos[self.palm_id].copy()

    def ee_yaw(self) -> float:
        return _quat_yaw(self.data.xquat[self.palm_id])

    def jaw_gap(self) -> float:
        # 1.0 fully open, ~0 closed; each finger slides up to 0.035 inward.
        return float(1.0 - (self.data.qpos[self.lf_adr] + self.data.qpos[self.rf_adr]) / 0.07)

    def grasped(self) -> bool:
        """Product pinched between BOTH fingers — the contact-based verdict."""
        left = right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if self.g_prod in pair and self.g_lf in pair:
                left = True
            if self.g_prod in pair and self.g_rf in pair:
                right = True
        return left and right

    def in_workspace(self, xy) -> bool:
        x0, x1, y0, y1 = self.WORKSPACE
        return bool(x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)
