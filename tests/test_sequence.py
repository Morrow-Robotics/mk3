import numpy as np
import pytest

from morrow import PackItem, demo_pack_sequence, run_pack_sequence, run_skill, with_place_slot
from morrow.skill import SkillState
from morrow.sim import make_world, onboard, SimPerceiver, SimRobot


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


def test_placed_item_blocks_the_place():
    # An occupied region over the drop zone must make placement infeasible.
    skill = onboard("box", "box")
    w = make_world("box")
    w.occupied = [np.array([0.0, -0.3, 0.4, 0.3])]  # whole carton is taken
    r = run_skill(skill, SimRobot(w), SimPerceiver(w), seed=0)
    assert not r.success and r.flagged
    assert r.failure_reason is not None and "OVER_CARTON" in r.failure_reason


def test_empty_occupied_leaves_single_product_unchanged():
    skill = onboard("box", "box")
    w = make_world("box")
    assert not w.occupied  # default
    assert run_skill(skill, SimRobot(w), SimPerceiver(w), seed=0).success


def test_halt_policy_stops_at_first_failure():
    skills = {"box": onboard("box", "box")}
    order = [PackItem("b1", "box", (-0.05, 0.0)), PackItem("b2", "box", (0.05, 0.0))]
    halt = run_pack_sequence(order, skills, seed=0, policy="halt")
    skip = run_pack_sequence(order, skills, seed=0, policy="skip")
    assert len(halt.results) == 1 and not halt.results[0].success  # stopped after item 0
    assert len(skip.results) == 2  # kept going


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        run_pack_sequence([], {}, policy="nonsense")


def test_sequence_runs_are_tagged_in_the_journal():
    from morrow import EpisodeLog
    log = EpisodeLog()
    demo_pack_sequence(seed=0, journal=log)
    assert len(log) == 3
    assert all(r.get("in_sequence") for r in log.records)
    assert [r["seq_item"] for r in log.records] == [0, 1, 2]
    assert {r["kind"] for r in log.records} == {"box", "cylinder", "pouch"}
    # single-product runs stay untagged
    from morrow import run_skill
    from morrow.sim import make_world, onboard, SimPerceiver, SimRobot
    plain = EpisodeLog()
    w = make_world("box")
    run_skill(onboard("box", "box"), SimRobot(w), SimPerceiver(w), seed=0, journal=plain)
    assert "in_sequence" not in plain.records[0]
