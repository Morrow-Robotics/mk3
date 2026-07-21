"""MuJoCo physics backend: accurate contact/grasp/collision for the packing task.

Implements the same Robot/Perceiver boundaries as the analytic sim, so the FSM
(`run_skill`) drives real physics unchanged. Models the standard LeRobot
parallel-jaw gripper (no suction). Requires `mujoco` (optional dependency).
"""

from .annotate import build_skill_from_annotation, capture_annotation, run_annotation
from .arm import ArmPerceiver, ArmRobot, ArmWorld, onboard_arm, record_arm_demo
from .mj_perceive import MjPerceiver
from .mj_robot import MjRobot
from .multipack import MultiCell, capture_multi_pack, pack_boxes
from .pattern import carton_activity, packing_profile
from .record import onboard_mj, record_mj_demo
from .watch import (have_sam2, render_overlay, scene_to_annotation, segment_scene,
                    watch_and_pack_arm)
from .world import MjWorld

__all__ = ["MjWorld", "MjRobot", "MjPerceiver", "record_mj_demo", "onboard_mj",
           "build_skill_from_annotation", "run_annotation", "capture_annotation",
           "ArmWorld", "ArmRobot", "ArmPerceiver", "record_arm_demo", "onboard_arm",
           "have_sam2", "segment_scene", "render_overlay", "scene_to_annotation",
           "watch_and_pack_arm", "packing_profile", "carton_activity",
           "pack_boxes", "capture_multi_pack", "MultiCell"]
