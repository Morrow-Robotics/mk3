import numpy as np

from morrow import run_skill
from morrow.sim import onboard, randomize, staged, stress, SimPerceiver, SimRobot


def test_transfers_under_carton_rotation():
    skill = onboard("box", "box")
    ok = 0
    for i in range(20):
        w = randomize("box", np.random.RandomState(200 + i), carton_yaw=True)
        ok += run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i).success
    assert ok == 20


def test_confidence_dropout_triggers_recovery_but_still_succeeds():
    skill = onboard("box", "box")
    results = []
    for i in range(40):
        w = randomize("box", np.random.RandomState(300 + i), perception_dropout=0.2)
        results.append(run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i))
    final = sum(r.success for r in results) / len(results)
    total_recoveries = sum(r.recoveries for r in results)
    assert final > 0.9  # the fail-closed gate causes recovery, not permanent failure
    assert total_recoveries > 0  # the confidence gate actually fired at least once


def test_stress_mode_is_reproducible():
    skill = onboard("box", "box")

    def batch():
        out = []
        for i in range(15):
            w = stress("box", np.random.RandomState(400 + i))
            out.append(run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i).final_state)
        return out

    assert batch() == batch()


def test_occlusion_stresses_grasp_but_recovers():
    skill = onboard("pouch", "pouch")
    results = []
    for i in range(40):
        w = randomize("pouch", np.random.RandomState(600 + i), occlusion=0.3)
        results.append(run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i))
    final = sum(r.success for r in results) / len(results)
    work = sum(r.retries + r.recoveries for r in results)
    assert final > 0.85  # geometric perception noise is recovered, not fatal
    assert work > 0  # occlusion actually forced re-grasps


def test_selects_the_right_sku_among_distractors():
    skill = onboard("box", "box")
    w = staged("box", np.random.RandomState(700))  # box target + cylinder/pouch distractors
    perceiver = SimPerceiver(w, target_descriptor=skill.object_descriptor)
    scene = perceiver.observe()
    assert scene.uncertainty["n_candidates"] == 3
    assert scene.uncertainty["selected_kind"] == "box"  # picked the target, not a distractor
    r = run_skill(skill, SimRobot(w), perceiver, seed=0)
    assert r.success


def test_selection_is_real_not_always_first():
    # A pouch descriptor over a box-target staged world must select the pouch distractor.
    box_skill = onboard("box", "box")
    pouch_skill = onboard("pouch", "pouch")
    w = staged("box", np.random.RandomState(701), distractor_kinds=["pouch", "cylinder"])
    assert SimPerceiver(w, target_descriptor=box_skill.object_descriptor).observe().uncertainty["selected_kind"] == "box"
    assert SimPerceiver(w, target_descriptor=pouch_skill.object_descriptor).observe().uncertainty["selected_kind"] == "pouch"


def test_staged_scenes_transfer():
    skill = onboard("cylinder", "cylinder")
    ok = 0
    for i in range(15):
        w = staged("cylinder", np.random.RandomState(720 + i))
        p = SimPerceiver(w, target_descriptor=skill.object_descriptor)
        ok += run_skill(skill, SimRobot(w), p, seed=i).success
    assert ok == 15
