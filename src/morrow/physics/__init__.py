"""MuJoCo physics backend: the LeRobot SO-101 model + the SAM2 watch pipeline.

Implements the same Robot/Perceiver boundaries as the analytic sim, so the FSM
(`run_skill`) drives real physics unchanged. The embodiment is the SO-101 model
(MuJoCo Menagerie), a standard parallel jaw — no suction, and no floating gripper.
Requires `mujoco` (optional dependency); the watch pipeline also needs SAM2.
"""

from .arm import ArmPerceiver, ArmRobot, ArmWorld, onboard_arm, record_arm_demo
from .arm_multi import MultiArmWorld, capture_pack_objects, pack_objects
from .pattern import carton_activity, packing_profile
from .reconstruct import VIDEO_SCENES, reconstruct_and_pack, reconstruct_scene
from .watch import (have_sam2, pack_annotation_on_arm, render_overlay, scene_to_annotation,
                    segment_scene, watch_and_pack_arm)

__all__ = ["ArmWorld", "ArmRobot", "ArmPerceiver", "record_arm_demo", "onboard_arm",
           "MultiArmWorld", "pack_objects", "capture_pack_objects",
           "reconstruct_scene", "reconstruct_and_pack", "VIDEO_SCENES",
           "have_sam2", "segment_scene", "render_overlay", "scene_to_annotation",
           "watch_and_pack_arm", "pack_annotation_on_arm", "packing_profile",
           "carton_activity"]
