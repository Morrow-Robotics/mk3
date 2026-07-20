"""Bench robot adapter — the physical implementation of the `Robot` boundary.

This is a skeleton, not a working driver: every method that needs hardware
raises `NotImplementedError` with a note on what to wire. It satisfies the
`Robot` Protocol structurally, so `run_skill` will accept it the moment the
TODOs are filled. Swap `SimRobot` for `BenchRobot` in `pipeline`/`eval` to move
from bench to hardware; nothing above the boundary changes.

Hardware assumed: a top-down arm (LeRobot-class or industrial), a suction
end-effector, and a vacuum pressure sensor on one analog input. Grasp
verification MUST come from that sensor, not from vision.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Transform, translation

_TODO = "BenchRobot: wire this to the arm/vacuum driver"


class BenchRobot:
    has_vacuum = True

    def __init__(self, arm=None, vacuum=None, calibration=None,
                 vacuum_threshold: float = 0.5, safe_z: float = 0.35, park_xy=(0.0, 0.0)):
        # arm: motion driver (e.g. LeRobot follower). vacuum: pressure sensor + valve.
        # calibration: T_base_camera etc., from the camera-to-robot calibration step.
        self.arm = arm
        self.vacuum = vacuum
        self.calibration = calibration
        self.vacuum_threshold = vacuum_threshold
        self.safe_z = safe_z
        self.park_xy = park_xy

    def get_ee_pose(self) -> Transform:
        raise NotImplementedError(f"{_TODO}: read forward kinematics -> 4x4 base-frame pose")

    def reachable(self, pose: Transform) -> bool:
        # Replace with a real IK feasibility + joint-limit check on self.arm.
        raise NotImplementedError(f"{_TODO}: IK feasibility for {translation(pose)}")

    def gripper_signal(self) -> float:
        raise NotImplementedError(f"{_TODO}: read the vacuum pressure sensor (low == sealed)")

    def holding(self) -> bool:
        # Once gripper_signal() works, this is just the threshold — no vision.
        return self.gripper_signal() < self.vacuum_threshold

    def follow(self, waypoints: list[Transform]) -> None:
        raise NotImplementedError(f"{_TODO}: send a Cartesian/joint trajectory through {len(waypoints)} poses")

    def engage(self) -> None:
        raise NotImplementedError(f"{_TODO}: open the vacuum valve")

    def release(self) -> None:
        raise NotImplementedError(f"{_TODO}: vent the vacuum valve")

    def safe_retract(self) -> None:
        raise NotImplementedError(f"{_TODO}: lift straight to safe_z={self.safe_z} at current xy")

    def park_and_flag(self) -> None:
        raise NotImplementedError(f"{_TODO}: move to park_xy at safe_z and raise the operator flag/light")

    # Kept so 'world.flagged' probing in execute.py stays harmless on hardware.
    world = None
