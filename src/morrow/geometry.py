"""SE(3) helpers, restricted to the yaw-about-vertical regime the bench uses.

Every frame and end-effector pose in this codebase is a 4x4 homogeneous
transform whose rotation is a yaw about the world +Z axis. That is all a
calibrated top-down packing cell needs: the tool points down, and the only
free orientation is yaw. Keeping the whole system in this regime makes the
object-relative / carton-relative math below exact and easy to read.

Real hardware with tilted approaches would add roll/pitch here; the rest of
the stack consumes 4x4 transforms and does not care.

It also holds the few small 2D helpers the planner, gates, and sim share —
planar rotation, point-in-box, box overlap — so that math lives in one place.
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


def inv(T: Transform) -> Transform:
    """Inverse of a rigid transform (transpose-rotation trick, no solve)."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def translation(T: Transform) -> np.ndarray:
    return T[:3, 3].copy()


def yaw_of(T: Transform) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


def rot2(theta: float) -> np.ndarray:
    """2x2 rotation matrix (the planar workhorse for object/carton-frame offsets)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def xy_in_bbox(xy, bbox) -> bool:
    """Is point xy inside axis-aligned [xmin, ymin, xmax, ymax]?"""
    return bool(bbox[0] <= xy[0] <= bbox[2] and bbox[1] <= xy[1] <= bbox[3])


def bbox_overlap_area(a, b) -> float:
    """Intersection area of two axis-aligned [xmin, ymin, xmax, ymax] boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(max(0.0, w) * max(0.0, h))
