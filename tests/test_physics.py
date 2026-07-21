import pytest

pytest.importorskip("mujoco")  # physics backend is an optional dependency

from morrow import run_skill
from morrow.physics.arm import ArmPerceiver, ArmRobot, ArmWorld, onboard_arm


def _inside_carton(w) -> bool:
    px, py = w.product_xy()
    c = w.carton
    return (c["cx"] - c["hx"] <= px <= c["cx"] + c["hx"]
            and c["cy"] - c["hy"] <= py <= c["cy"] + c["hy"])


def test_so101_arm_physically_packs_a_box():
    # The 5-DOF LeRobot SO-101 model (MuJoCo Menagerie): orientation-aware IK drives
    # the position actuators tool-down, the parallel jaw grasps by friction, and
    # run_skill carries the FSM to VERIFIED with the box physically in the carton —
    # grasp is contact, lift is real z, place is footprint overlap.
    skill = onboard_arm("box", "so101-box")
    w = ArmWorld("box", px=0.27, py=-0.06)
    r = run_skill(skill, ArmRobot(w), ArmPerceiver(w), seed=0)
    assert r.success and r.final_state == "VERIFIED"
    assert _inside_carton(w)                       # physically inside, not an FSM claim
    assert w.data.xpos[w.pid][2] < 0.05            # settled to the carton floor


def test_so101_arm_grasp_is_contact_based():
    # Close the jaws on empty space above the product -> no two-finger contact ->
    # not holding. The grasp verdict is physics, never a flag.
    w = ArmWorld("box")
    robot = ArmRobot(w)
    robot._move_to((0.27, 0.06, 0.09), 0.0)        # away from and above the product
    robot.engage()
    assert not robot.holding()


def test_arm_showcase_and_dashboard_render():
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available for mp4 encode")
    from morrow.physics.showcase import build_arm_showcase
    from morrow.physics.webview import _slots, render_physics_page, runtime_info
    show = build_arm_showcase(kinds=("box",))
    box = show["kinds"]["box"]
    assert box["inside_carton"] and len(box["mp4_b64"]) > 100  # a real rendered clip
    html = render_physics_page(_slots("videos", embed=False), runtime_info(), show, None)
    assert "<title>" in html and "SO-101" in html and "Phase-3" in html
