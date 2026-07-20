"""Bench demonstration recorder — fills the same immutable `DemonstrationTrace`.

The recorder is the one sim-specific piece (see BENCH.md). On hardware, a human
teleoperates the leader arm while this captures synchronized frames. The frame
*assembly* (`add_frame` / `finalize`) is real and tested here; only the live
capture loop (`record_live`) needs the teleop trigger and camera, so it is the
single fill-in point. Because it produces a `DemonstrationTrace`, `compile_skill`
is byte-for-byte unchanged between bench and sim.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Transform
from ..trace import DemonstrationTrace

_TODO = "BenchRecorder: wire the teleop trigger + synchronized capture"


class BenchRecorder:
    def __init__(self, camera_calibration_id: str, software_version: str, table_height: float):
        self.camera_calibration_id = camera_calibration_id
        self.software_version = software_version
        self.table_height = table_height
        self._t: list[float] = []
        self._poses: list[Transform] = []
        self._gcmd: list[float] = []
        self._gsig: list[float] = []
        self._masks: list[np.ndarray] = []
        self._centroids: list[np.ndarray] = []
        self._footprints: list[np.ndarray] = []
        self._yaws: list[float] = []

    def add_frame(self, timestamp: float, ee_pose: Transform, gripper_command: float,
                  gripper_signal: float, product_mask: np.ndarray, product_centroid: np.ndarray,
                  product_footprint: np.ndarray, product_yaw: float) -> None:
        """Append one synchronized frame. Perception fields come from the bench
        perceiver's helpers run per frame; robot fields from the arm + gripper grip-strength signal."""
        self._t.append(float(timestamp))
        self._poses.append(ee_pose)
        self._gcmd.append(float(gripper_command))
        self._gsig.append(float(gripper_signal))
        self._masks.append(product_mask)
        self._centroids.append(product_centroid)
        self._footprints.append(product_footprint)
        self._yaws.append(float(product_yaw))

    def finalize(self, carton_frame: Transform, meta: dict | None = None) -> DemonstrationTrace:
        return DemonstrationTrace(
            timestamps=np.array(self._t),
            ee_poses=self._poses,
            gripper_command=np.array(self._gcmd),
            gripper_signal=np.array(self._gsig),
            product_masks=self._masks,
            product_centroids=self._centroids,
            product_footprints=self._footprints,
            product_yaws=np.array(self._yaws),
            carton_frame=carton_frame,
            table_height=self.table_height,
            camera_calibration_id=self.camera_calibration_id,
            software_version=self.software_version,
            meta=meta or {},
        )

    def record_live(self, robot, perceiver, is_recording, carton_frame, meta=None) -> DemonstrationTrace:
        """Capture a live teleop demo. The only piece that needs hardware:
        loop while `is_recording()` calling `add_frame` from robot + perceiver,
        then `finalize`. Left unimplemented so it fails loud rather than faking data."""
        raise NotImplementedError(
            f"{_TODO}: while is_recording(): read a synchronized frame from robot + "
            "perceiver, add_frame(...), then return finalize(carton_frame, meta)."
        )
