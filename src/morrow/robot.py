"""The robot boundary.

The executor talks to *this* interface, never to a specific arm. A simulated
arm (`sim.sim_robot.SimRobot`) and, later, a physical LeRobot/industrial adapter
both implement it, so the state machine, compiler, and evaluation code are
identical across bench and hardware. This is the same boundary discipline mk2
used for its model backend.

Grasp verification is a first-class part of the interface: `gripper_signal`
is the raw hardware reading and `holding` is its thresholded verdict. Vision
never decides whether a grasp succeeded.

The interface is **end-effector-neutral**. `holding()` is the hardware grasp
verdict whatever the tool is — a vacuum seal, a parallel-jaw two-finger contact,
or a grip-force threshold — so a suction cell (the analytic sim) and a
parallel-jaw cell (the MuJoCo SO-101 model, and the physical arm) both satisfy
the same contract. `end_effector` names which one, for logging/display only; no
execution logic branches on it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .geometry import Transform


@runtime_checkable
class Robot(Protocol):
    end_effector: str  # "vacuum" | "parallel_jaw" | "force" — descriptive only

    def get_ee_pose(self) -> Transform:
        """Current base-frame end-effector pose."""

    def reachable(self, pose: Transform) -> bool:
        """IK/workspace feasibility of a target pose (the fail-closed gate)."""

    def gripper_signal(self) -> float:
        """Raw end-effector reading — vacuum pressure, jaw gap, or grip force.
        `holding()` interprets it; callers must not assume a specific tool."""

    def holding(self) -> bool:
        """Hardware grasp verdict: is the product secured to the tool right now
        (vacuum seal / two-finger contact / grip force). Never vision."""

    def follow(self, waypoints: list[Transform]) -> None:
        """Move the tool through a sequence of base-frame poses."""

    def engage(self) -> None:
        """Secure the product: close the jaws or enable suction."""

    def release(self) -> None:
        """Release the product: open the jaws or disable suction."""

    def safe_retract(self) -> None:
        """Lift straight up to a safe travel height from wherever we are."""

    def park_and_flag(self) -> None:
        """Move to the park pose and raise an operator flag. Graceful stop."""
