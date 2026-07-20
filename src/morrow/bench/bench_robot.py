"""Physical SO-101 adapter — the hardware implementation of the `Robot` boundary.

This is a scaffold, NOT a working driver: every method that needs the arm raises
`NotImplementedError` with a note on what to wire. It satisfies the `Robot`
Protocol structurally, so `run_skill` accepts it the moment the TODOs are filled.
Swap `SimRobot`/`ArmRobot` for `SO101BenchRobot` and nothing above the boundary
changes.

Hardware assumed (Phase 4, Real Bench v0): a **LeRobot SO-101 follower**, rigidly
mounted, top-down, driven via the LeRobot control API; the standard **parallel-jaw**
end-effector (NOT suction — that is why `end_effector = "parallel_jaw"` and the
grasp verdict is contact/force, not a vacuum seal). Grasp verification MUST come
from the gripper — closing width bottoming out on the product and/or a grip-force
threshold — never from vision, because the tool occludes the product at the
moment you need to confirm the grasp.

The MuJoCo `physics/arm.py` is the sim twin of this class: same 5-DOF SO-101, same
tool-down IK, same parallel-jaw contact grasp. Bring-up = re-calibrating the five
embodiment facts documented there (inverted gripper ctrl, ~2 cm tool-site offset,
IK on a scratch state only, abstract gripper intent, workspace limits) against the
physical arm, then measuring the sim-to-real gap on the known reach envelope.
"""

from __future__ import annotations

from ..geometry import Transform, translation

_TODO = "SO101BenchRobot: wire this to the LeRobot SO-101 follower driver"


class SO101BenchRobot:
    """Parallel-jaw SO-101, physical. End-effector-neutral `holding()`."""

    end_effector = "parallel_jaw"

    def __init__(self, arm=None, gripper=None, calibration=None,
                 grip_threshold: float = 0.5, safe_z: float = 0.20, park_xy=(0.0, 0.0)):
        # arm: LeRobot SO-101 follower motion driver.
        # gripper: jaw command + grip-strength/width sensor.
        # calibration: T_base_camera etc. from camera-to-robot calibration.
        self.arm = arm
        self.gripper = gripper
        self.calibration = calibration
        self.grip_threshold = grip_threshold
        self.safe_z = safe_z
        self.park_xy = park_xy

    def get_ee_pose(self) -> Transform:
        raise NotImplementedError(f"{_TODO}: read forward kinematics -> 4x4 base-frame pose")

    def reachable(self, pose: Transform) -> bool:
        raise NotImplementedError(f"{_TODO}: IK feasibility + joint limits for {translation(pose)}")

    def gripper_signal(self) -> float:
        # Normalized grip strength: high == the jaws are closed on something.
        raise NotImplementedError(f"{_TODO}: read jaw grip force / closed-on-product width")

    def holding(self) -> bool:
        # End-effector-neutral verdict: threshold on the grip signal, never vision.
        return self.gripper_signal() >= self.grip_threshold

    def follow(self, waypoints: list[Transform]) -> None:
        raise NotImplementedError(
            f"{_TODO}: send a Cartesian/joint trajectory through {len(waypoints)} poses")

    def engage(self) -> None:
        raise NotImplementedError(f"{_TODO}: close the jaws on the product")

    def release(self) -> None:
        raise NotImplementedError(f"{_TODO}: open the jaws")

    def safe_retract(self) -> None:
        raise NotImplementedError(f"{_TODO}: lift straight to safe_z={self.safe_z} at current xy")

    def park_and_flag(self) -> None:
        raise NotImplementedError(f"{_TODO}: move to park_xy at safe_z and raise the operator flag/light")

    def emergency_stop(self) -> None:
        """Immediate halt — cut motion and latch a safe state. Bring-up must
        implement this before any autonomous run."""
        raise NotImplementedError(f"{_TODO}: cut motion and latch a safe state")

    # Kept so `world.flagged` probing in execute.py stays harmless on hardware.
    world = None


# Back-compat alias: the boundary is embodiment-agnostic; this is the SO-101 one.
BenchRobot = SO101BenchRobot
