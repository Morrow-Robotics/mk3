import numpy as np

from morrow import PackItem, demo_pack_sequence, run_pack_sequence, with_place_slot
from morrow.skill import SkillState
from morrow.sim import onboard


def test_high_mix_sequence_packs_all():
    r = demo_pack_sequence(seed=0)
    assert r.success and r.packed == 3
    assert all(x.final_state == "VERIFIED" for x in r.results)
    assert {x.kind for x in r.results} == {"box", "cylinder", "pouch"}


def test_place_slots_are_distinct():
    r = demo_pack_sequence(seed=0)
    slots = [x.slot for x in r.results]
    assert len(set(slots)) == len(slots)


def test_sequence_is_reproducible():
    a = demo_pack_sequence(seed=1)
    b = demo_pack_sequence(seed=1)
    assert [(x.final_state, x.failure_reason) for x in a.results] == \
           [(x.final_state, x.failure_reason) for x in b.results]


def test_same_kind_sequence_flags_ambiguous():
    skills = {"box": onboard("box", "box")}
    order = [PackItem("box-1", "box", (-0.05, 0.0)), PackItem("box-2", "box", (0.05, 0.0))]
    r = run_pack_sequence(order, skills, seed=0)
    assert r.results[0].failure_reason == "SELECTION:ambiguous"  # can't tell the two apart


def test_with_place_slot_shifts_offset_and_leaves_original():
    skill = onboard("box", "box")
    edge = (SkillState.OVER_CARTON, SkillState.PLACED)
    before = list(skill.transition(*edge).rel["place_offset"])
    shifted = with_place_slot(skill, (0.05, -0.03))
    after = shifted.transition(*edge).rel["place_offset"]
    assert np.allclose(after, [before[0] + 0.05, before[1] - 0.03])
    assert skill.transition(*edge).rel["place_offset"] == before  # original untouched
    assert shifted.version_hash != skill.version_hash  # content-addressed
