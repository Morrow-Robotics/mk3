import numpy as np

from morrow import GraspRanker, run_skill
from morrow.eval import compare_ranker
from morrow.journal import EpisodeLog
from morrow.sim import onboard, structured, SimPerceiver, SimRobot


def _log(kind="box", n=60, pref=0.0):
    skill = onboard(kind, kind)
    log = EpisodeLog()
    for i in range(n):
        w = structured(kind, np.random.RandomState(5000 + i), pref)
        run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i, journal=log)
    return skill, log


def test_fit_is_deterministic():
    _, log = _log()
    a = GraspRanker.fit(log.records)
    b = GraspRanker.fit(log.records)
    assert np.allclose(a.weights, b.weights)


def test_recovers_the_yaw_prior():
    # The learned model must prefer the aligned grasp over its 180-flip.
    _, log = _log()
    r = GraspRanker.fit(log.records)
    assert r.prob(0.0, [0.0, 0.0]) > r.prob(np.pi, [0.0, 0.0]) + 0.3


def test_lifts_first_attempt_on_structured_sku():
    res = compare_ranker("box", n=60, n_train=60, seal_yaw_pref=0.0)
    assert res["ranker_first_attempt"] > res["analytic_first_attempt"] + 0.15


def test_ranker_run_is_reproducible():
    skill, log = _log()
    r = GraspRanker.fit(log.records)
    wa = structured("box", np.random.RandomState(6001), 0.0)
    wb = structured("box", np.random.RandomState(6001), 0.0)
    a = run_skill(skill, SimRobot(wa), SimPerceiver(wa), seed=1, ranker=r)
    b = run_skill(skill, SimRobot(wb), SimPerceiver(wb), seed=1, ranker=r)
    assert (a.success, a.first_attempt_success, a.final_state) == \
           (b.success, b.first_attempt_success, b.final_state)


def test_empty_log_gives_a_no_op_ranker():
    r = GraspRanker.fit([])
    assert abs(r.prob(0.0, [0.0, 0.0]) - 0.5) < 1e-9  # prob 0.5 -> adds nothing
