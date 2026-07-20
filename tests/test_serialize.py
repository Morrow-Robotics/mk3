import numpy as np

from morrow import run_skill, skill_from_json, skill_to_json
from morrow.serialize import recompute_hash
from morrow.sim import onboard, randomize, SimPerceiver, SimRobot


def test_json_round_trip_preserves_hash():
    skill = onboard("box", "box")
    back = skill_from_json(skill_to_json(skill))
    assert back.version_hash == skill.version_hash
    assert back.sku_id == skill.sku_id
    assert set(back.transitions) == set(skill.transitions)


def test_reloaded_skill_recomputes_its_own_hash():
    # The core claim: a skill is content-addressed and free of callables.
    skill = onboard("cylinder", "cylinder")
    back = skill_from_json(skill_to_json(skill))
    assert recompute_hash(back) == back.version_hash == skill.version_hash


def test_reloaded_skill_runs_identically():
    skill = onboard("box", "box")
    back = skill_from_json(skill_to_json(skill))
    wa = randomize("box", np.random.RandomState(11))
    wb = randomize("box", np.random.RandomState(11))
    a = run_skill(skill, SimRobot(wa), SimPerceiver(wa), seed=11)
    b = run_skill(back, SimRobot(wb), SimPerceiver(wb), seed=11)
    assert (a.success, a.final_state, a.retries, a.recoveries) == \
           (b.success, b.final_state, b.retries, b.recoveries)


def test_save_and_load(tmp_path):
    from morrow import load_skill, save_skill
    skill = onboard("pouch", "pouch")
    p = tmp_path / "pouch.json"
    save_skill(skill, str(p))
    assert load_skill(str(p)).version_hash == skill.version_hash
