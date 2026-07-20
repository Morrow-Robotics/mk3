import numpy as np
import pytest

pytest.importorskip("mujoco")  # physics backend is an optional dependency

from morrow import run_skill
from morrow.physics import MjPerceiver, MjRobot, MjWorld, onboard_mj


def _inside_carton(w) -> bool:
    px, py = w.product_xy()
    c = w.carton
    return (c["cx"] - c["hx"] <= px <= c["cx"] + c["hx"]
            and c["cy"] - c["hy"] <= py <= c["cy"] + c["hy"])


def test_physics_packs_a_box_into_the_carton():
    skill = onboard_mj("box", "box")
    w = MjWorld("box", px=-0.15)
    r = run_skill(skill, MjRobot(w), MjPerceiver(w), seed=0)
    assert r.success and r.final_state == "VERIFIED"
    assert _inside_carton(w)  # physically inside, not just the FSM saying so


def test_physics_grasp_is_contact_based_not_a_flag():
    # Close the jaws on empty space -> no two-finger contact -> not holding.
    w = MjWorld("box")
    robot = MjRobot(w)
    robot._move_to((0.10, 0.20, 0.15), 0.0)  # away from the product
    robot.engage()
    assert not robot.holding()


def test_physics_transfers_across_poses():
    skill = onboard_mj("cylinder", "cylinder")
    ok = 0
    rng = np.random.RandomState(1)
    for i in range(4):
        w = MjWorld("cylinder", px=rng.uniform(-0.20, -0.10),
                    py=rng.uniform(-0.08, 0.08), pyaw=rng.uniform(-0.4, 0.4))
        ok += int(run_skill(skill, MjRobot(w), MjPerceiver(w), seed=i).success and _inside_carton(w))
    assert ok == 4


def test_physics_packs_a_standup_pouch():
    skill = onboard_mj("pouch", "pouch")  # rigid stand-up pouch approximation
    ok = 0
    rng = np.random.RandomState(2)
    for i in range(4):
        w = MjWorld("pouch", px=rng.uniform(-0.20, -0.10),
                    py=rng.uniform(-0.08, 0.08), pyaw=rng.uniform(-0.4, 0.4))
        ok += int(run_skill(skill, MjRobot(w), MjPerceiver(w), seed=i).success and _inside_carton(w))
    assert ok == 4


def test_annotation_builds_and_packs_at_marked_size():
    from morrow.physics.annotate import run_annotation
    ann = {"sku": "cust", "image": {"w": 1280, "h": 720}, "scale_m_per_px": 0.0006,
           "product": {"kind": "box", "bbox_px": [560, 300, 677, 400], "height_m": 0.06}}
    skill, w, r = run_annotation(ann)
    assert r.success and _inside_carton(w)
    assert abs(w.hx - 0.0351) < 0.006  # the marked size (117 px * 0.6 mm/px / 2) is honored


def test_annotation_rejects_product_too_wide_for_the_jaw():
    from morrow.physics.annotate import build_skill_from_annotation
    ann = {"image": {"w": 100, "h": 100}, "scale_m_per_px": 0.002,
           "product": {"kind": "box", "bbox_px": [0, 0, 60, 60], "height_m": 0.06}}
    with pytest.raises(ValueError):
        build_skill_from_annotation(ann)  # 60 px * 2 mm/px = 12 cm wide -> no honest grasp


def test_so101_arm_physically_packs_a_box():
    # The REAL 5-DOF LeRobot SO-101 (not the floating gripper): orientation-aware
    # IK drives the position actuators tool-down, the parallel jaw grasps by
    # friction, and run_skill carries the FSM to VERIFIED with the box physically
    # in the carton — grasp is contact, lift is real z, place is footprint overlap.
    from morrow.physics.arm import ArmPerceiver, ArmRobot, ArmWorld, onboard_arm
    skill = onboard_arm("box", "so101-box")
    w = ArmWorld("box", px=0.27, py=-0.06)
    r = run_skill(skill, ArmRobot(w), ArmPerceiver(w), seed=0)
    assert r.success and r.final_state == "VERIFIED"
    assert _inside_carton(w)                       # physically inside, not an FSM claim
    assert w.data.xpos[w.pid][2] < 0.05            # settled to the carton floor


def test_so101_arm_grasp_is_contact_based():
    # Close the real jaws on empty space above the product -> no two-finger
    # contact -> not holding. The grasp verdict is physics, never a flag.
    from morrow.physics.arm import ArmRobot, ArmWorld
    w = ArmWorld("box")
    robot = ArmRobot(w)
    robot._move_to((0.27, 0.06, 0.09), 0.0)        # away from and above the product
    robot.engage()
    assert not robot.holding()


def test_physics_showcase_and_dashboard_render():
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available for mp4 encode")
    from morrow.physics.showcase import build_showcase
    from morrow.physics.webview import _slots, render_physics_page, runtime_info
    show = build_showcase(kinds=("box",))
    box = show["kinds"]["box"]
    assert box["inside_carton"] and len(box["mp4_b64"]) > 100  # a real rendered clip
    html = render_physics_page(show, _slots("videos", embed=False), runtime_info())
    assert "<title>" in html and "mp4_b64" in html and "Phase-3" in html
