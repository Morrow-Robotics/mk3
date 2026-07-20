"""A simulated arm implementing the `Robot` boundary against a `World`."""

from __future__ import annotations

import numpy as np

from ..geometry import Transform, frame, translation, yaw_of
from .world import World


class SimRobot:
    has_vacuum = True

    def __init__(self, world: World):
        self.world = world

    def get_ee_pose(self) -> Transform:
        return self.world.ee_pose.copy()

    def reachable(self, pose: Transform) -> bool:
        xy = translation(pose)[:2]
        z = float(translation(pose)[2])
        return self.world.in_workspace(xy) and (self.world.table_height - 0.02) <= z <= (self.world.SAFE_Z + 0.05)

    def gripper_signal(self) -> float:
        return self.world.signal()

    def holding(self) -> bool:
        return self.world.attached

    def follow(self, waypoints: list[Transform]) -> None:
        for wp in waypoints:
            self.world.set_ee(wp)

    def engage(self) -> None:
        self.world.try_seal()

    def release(self) -> None:
        self.world.release_seal()

    def safe_retract(self) -> None:
        ee = translation(self.world.ee_pose)
        self.world.set_ee(frame((ee[0], ee[1], self.world.SAFE_Z), yaw_of(self.world.ee_pose)))

    def park_and_flag(self) -> None:
        self.world.release_seal()
        self.world.set_ee(frame((0.0, 0.0, self.world.SAFE_Z), 0.0))
        self.world.flagged = True
