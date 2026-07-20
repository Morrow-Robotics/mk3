import numpy as np

from morrow import run_skill
from morrow.sim import forced_failure_world, make_world, onboard, randomize, SimPerceiver, SimRobot


def test_clean_run_reaches_verified_first_attempt():
    skill = onboard("box", "box")
    w = make_world("box")  # canonical pose, no noise
    r = run_skill(skill, SimRobot(w), SimPerceiver(w), seed=0)
    assert r.success and r.final_state == "VERIFIED"
    assert r.first_attempt_success and r.recoveries == 0 and r.retries == 0


def test_transfers_to_randomized_poses():
    skill = onboard("box", "box")
    ok = 0
    for i in range(30):
        w = randomize("box", np.random.RandomState(i))
        ok += run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i).success
    assert ok == 30  # autonomous recovery makes final success total in sim


def test_forced_failure_recovers_from_current_state():
    skill = onboard("box", "box")
    w = forced_failure_world("box")
    r = run_skill(skill, SimRobot(w), SimPerceiver(w), seed=1)
    assert r.success  # recovered
    assert r.recoveries >= 1  # a cross-transition recovery actually fired
    assert not r.flagged  # did not need a human


def test_runs_are_deterministic():
    skill = onboard("box", "box")
    wa = randomize("box", np.random.RandomState(5))  # robot + perceiver share one world
    a = run_skill(skill, SimRobot(wa), SimPerceiver(wa), seed=5)
    wb = randomize("box", np.random.RandomState(5))  # identical world from the same seed
    b = run_skill(skill, SimRobot(wb), SimPerceiver(wb), seed=5)
    assert (a.success, a.retries, a.recoveries, a.final_state) == \
           (b.success, b.retries, b.recoveries, b.final_state)


def test_grasp_verified_by_hardware_not_vision():
    # A skill that never seals must never reach GRASPED, no matter what vision says.
    skill = onboard("box", "box")
    w = make_world("box", force_fail_seals=999)
    r = run_skill(skill, SimRobot(w), SimPerceiver(w), seed=0)
    assert not r.success and r.flagged
    assert r.final_state == "FAILED"


def test_failure_reason_names_the_stuck_edge():
    skill = onboard("box", "box")
    w = make_world("box", force_fail_seals=999)  # never seals
    r = run_skill(skill, SimRobot(w), SimPerceiver(w), seed=0)
    assert r.failure_reason == "APPROACHED->GRASPED:grasped"
