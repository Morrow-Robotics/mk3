"""The robot boundary.

The executor talks to *this* interface, never to a specific arm. A simulated
arm (`sim.sim_robot.SimRobot`) and, later, a real LeRobot/industrial adapter
both implement it, so the state machine, compiler, and evaluation code are
identical across bench and hardware. This is the same boundary discipline mk2
used for its model backend.

Grasp verification is a first-class part of the interface: `gripper_signal`
is the raw hardware reading and `holding` is its thresholded verdict. Vision
never decides whether a grasp succeeded.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .geometry import Transform


@runtime_checkable
class Robot(Protocol):
    has_vacuum: bool

    def get_ee_pose(self) -> Transform:
        """Current base-frame end-effector pose."""

    def reachable(self, pose: Transform) -> bool:
        """IK/workspace feasibility of a target pose (the fail-closed gate)."""

    def gripper_signal(self) -> float:
        """Raw hardware reading. Vacuum: low == strong seal == holding."""

    def holding(self) -> bool:
        """Hardware verdict: is something attached to the tool right now."""

    def follow(self, waypoints: list[Transform]) -> None:
        """Move the tool through a sequence of base-frame poses."""

    def engage(self) -> None:
        """Turn on suction / close the gripper."""

    def release(self) -> None:
        """Turn off suction / open the gripper."""

    def safe_retract(self) -> None:
        """Lift straight up to a safe travel height from wherever we are."""

    def park_and_flag(self) -> None:
        """Move to the park pose and raise an operator flag. Graceful stop."""
