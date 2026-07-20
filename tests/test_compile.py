import numpy as np
import pytest

from morrow import compile_skill
from morrow.skill import EDGES, SkillState
from morrow.sim import make_world, record_demo, SimPerceiver, SimRobot


def _demo(kind="box"):
    w = make_world(kind)
    return record_demo(w, SimRobot(w), SimPerceiver(w))


def test_compiles_a_complete_fsm():
    skill = compile_skill([_demo()], "box")
    skill.validate()  # raises if any edge missing
    for edge in EDGES:
        assert edge in skill.transitions


def test_phase_indices_are_ordered():
    skill = compile_skill([_demo()], "box")
    p = skill.metadata["phase_indices"]
    assert p["grasp"] < p["lift"] <= p["over"] < p["release"]
    assert skill.metadata["attachment_confirmed"] is True


def test_hash_is_deterministic_and_content_addressed():
    a = compile_skill([_demo()], "box")
    b = compile_skill([_demo()], "box")
    assert a.version_hash == b.version_hash
    c = compile_skill([_demo("cylinder")], "cylinder")
    assert c.version_hash != a.version_hash


def test_missing_edge_fails_loud():
    skill = compile_skill([_demo()], "box")
    del skill.transitions[(SkillState.READY, SkillState.APPROACHED)]
    with pytest.raises(ValueError):
        skill.validate()


def test_grasp_offset_is_small_for_centered_demo():
    skill = compile_skill([_demo()], "box")
    off = skill.transition(SkillState.APPROACHED, SkillState.GRASPED).rel["grasp_offset"]
    assert np.linalg.norm(off) < 0.01
