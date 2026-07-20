import numpy as np

from morrow import run_skill
from morrow.sim import onboard, randomize, stress, SimPerceiver, SimRobot


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
