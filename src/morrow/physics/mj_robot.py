"""MjRobot — the Robot boundary over a MuJoCo cell (standard LeRobot parallel jaw).

The mocap tool follows the SkillProgram's Cartesian EE waypoints; MuJoCo handles
contact, gravity, and the friction grasp. No suction: `holding` is the
contact-based two-finger verdict, and `engage`/`release` close/open the jaws.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Transform, frame, translation, wrap_angle, yaw_of
from .world import MjWorld

MOVE_STEPS = 70  # physics steps to traverse one waypoint (smooth, stable weld)
CAPTURE_EVERY = 10  # steps between frames when a capture hook is attached


class MjRobot:
    has_vacuum = False

    def __init__(self, world: MjWorld, on_frame=None):
        self.world = world
        self.on_frame = on_frame  # optional callable() -> capture a render frame

    def get_ee_pose(self) -> Transform:
        return frame(self.world.ee_pos(), self.world.ee_yaw())

    def reachable(self, pose: Transform) -> bool:
        xy = translation(pose)[:2]
        z = float(translation(pose)[2])
        return self.world.in_workspace(xy) and -0.02 <= z <= self.world.SAFE_Z + 0.05

    def gripper_signal(self) -> float:
        return self.world.jaw_gap()

    def holding(self) -> bool:
        return self.world.grasped()

    def _capture(self) -> None:
        if self.on_frame is not None:
            self.on_frame()

    def _move_to(self, xyz, yaw: float, steps: int = MOVE_STEPS) -> None:
        w = self.world
        start = w.data.mocap_pos[0].copy()
        yaw0 = w.ee_yaw()
        dyaw = wrap_angle(yaw - yaw0)
        target = np.asarray(xyz, dtype=float)
        for k in range(1, steps + 1):
            f = k / steps
            w.set_ee(start + (target - start) * f, yaw0 + dyaw * f)
            w.step(1)
            if self.on_frame is not None and k % CAPTURE_EVERY == 0:
                self._capture()

    def follow(self, waypoints: list[Transform]) -> None:
        for wp in waypoints:
            self._move_to(translation(wp), yaw_of(wp))

    def engage(self) -> None:
        self.world.set_jaws(0.035)
        for _ in range(14):
            self.world.step(10)
            self._capture()

    def release(self) -> None:
        self.world.set_jaws(0.0)
        for _ in range(9):
            self.world.step(10)
            self._capture()

    def safe_retract(self) -> None:
        ee = self.world.ee_pos()
        self._move_to((ee[0], ee[1], self.world.SAFE_Z), self.world.ee_yaw())

    def park_and_flag(self) -> None:
        self.release()
        self._move_to((0.0, -0.28, self.world.SAFE_Z), 0.0)
        self.world.flagged = True
