"""The immutable demonstration contract.

`DemonstrationTrace` is the single, strict record every downstream stage
consumes. Nothing infers a skill from anything else. If a field is missing
here, the compiler cannot invent it — that is the point. The trace is frozen
so a demonstration cannot be mutated after capture, which keeps skill hashes
and evaluation reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Transform


@dataclass(frozen=True)
class DemonstrationTrace:
    """One synchronized recording of a human/teleop packing demonstration.

    All list fields are indexed by the same frame counter and share `timestamps`.
    Poses are 4x4 base-frame transforms. `gripper_signal` is the raw hardware
    reading (end-effector grip signal, normalized 0..1). Masks are
    boolean HxW arrays in the overhead camera image.
    """

    timestamps: np.ndarray  # (T,)
    ee_poses: list[Transform]  # T_base_ee per frame
    gripper_command: np.ndarray  # (T,) commanded: 1.0 == engage/close
    gripper_signal: np.ndarray  # (T,) measured hardware signal
    product_masks: list[np.ndarray]  # per-frame boolean masks (overhead)
    product_centroids: list[np.ndarray]  # (3,) base-frame centroid per frame
    product_footprints: list[np.ndarray]  # (4,) base-frame bbox per frame
    product_yaws: np.ndarray  # (T,) footprint yaw per frame
    carton_frame: Transform  # base-frame carton pose at start of demo
    table_height: float  # base-frame z of the table surface
    camera_calibration_id: str
    software_version: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        for name in (
            "ee_poses",
            "gripper_command",
            "gripper_signal",
            "product_masks",
            "product_centroids",
            "product_footprints",
            "product_yaws",
        ):
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(
                    f"DemonstrationTrace.{name} has length {got}, expected {n} "
                    f"(must match timestamps)"
                )
        if n < 2:
            raise ValueError("a demonstration needs at least 2 frames")

    def __len__(self) -> int:
        return len(self.timestamps)
