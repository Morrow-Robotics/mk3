"""SE(3) helpers, restricted to the yaw-about-vertical regime the bench uses.

Every frame and end-effector pose in this codebase is a 4x4 homogeneous
transform whose rotation is a yaw about the world +Z axis. That is all a
calibrated top-down packing cell needs: the tool points down, and the only
free orientation is yaw. Keeping the whole system in this regime makes the
object-relative / carton-relative math below exact and easy to read.

Real hardware with tilted approaches would add roll/pitch here; the rest of
the stack consumes 4x4 transforms and does not care.
"""

from __future__ import annotations

import numpy as np

Transform = np.ndarray  # 4x4, float64


def frame(position, yaw: float = 0.0) -> Transform:
    """Build a 4x4 transform from a 3D position and a yaw (radians)."""
    x, y, z = (float(v) for v in position)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0, x],
            [s, c, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def identity() -> Transform:
    return np.eye(4, dtype=np.float64)


def inv(T: Transform) -> Transform:
    """Inverse of a rigid transform (transpose-rotation trick, no solve)."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose(*transforms: Transform) -> Transform:
    """Chain transforms left-to-right: compose(A, B, C) == A @ B @ C."""
    out = identity()
    for T in transforms:
        out = out @ T
    return out


def relative(T_ref: Transform, T: Transform) -> Transform:
    """Express pose T in the frame of T_ref: inv(T_ref) @ T."""
    return inv(T_ref) @ T


def apply_frame(T_ref: Transform, T_rel: Transform) -> Transform:
    """Instantiate a frame-relative pose back into the world: T_ref @ T_rel."""
    return T_ref @ T_rel


def translation(T: Transform) -> np.ndarray:
    return T[:3, 3].copy()


def yaw_of(T: Transform) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))
