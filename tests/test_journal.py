import numpy as np

from morrow import EpisodeLog, read_log, run_skill, summarize_log
from morrow.sim import make_world, onboard, randomize, SimPerceiver, SimRobot


def test_run_appends_a_record():
    skill = onboard("box", "box")
    log = EpisodeLog()
    w = randomize("box", np.random.RandomState(0))
    run_skill(skill, SimRobot(w), SimPerceiver(w), seed=0, journal=log)
    assert len(log) == 1
    rec = log.records[0]
    assert rec["skill_hash"] == skill.version_hash and rec["seed"] == 0
    assert "grasp_attempts" in rec and rec["success"] is True


def test_record_is_deterministic_no_wallclock():
    skill = onboard("box", "box")

    def record():
        log = EpisodeLog()
        w = randomize("box", np.random.RandomState(3))
        run_skill(skill, SimRobot(w), SimPerceiver(w), seed=3, journal=log)
        return log.records[0]

    a, b = record(), record()
    assert a == b  # identical (skill, seed) -> identical record, no time field
    assert "timestamp" not in a and "time" not in a


def test_grasp_attempts_capture_seal_and_miss():
    skill = onboard("box", "box")
    log = EpisodeLog()
    w = make_world("box", force_fail_seals=2)  # first two grasps miss, then seal
    run_skill(skill, SimRobot(w), SimPerceiver(w), seed=1, journal=log)
    grasps = log.records[0]["grasp_attempts"]
    assert any(g["sealed"] for g in grasps) and any(not g["sealed"] for g in grasps)


def test_jsonl_persist_and_read(tmp_path):
    skill = onboard("box", "box")
    path = tmp_path / "episodes.jsonl"
    log = EpisodeLog(str(path))
    for i in range(5):
        w = randomize("box", np.random.RandomState(i))
        run_skill(skill, SimRobot(w), SimPerceiver(w), seed=i, journal=log)
    records = read_log(str(path))
    assert len(records) == 5
    summary = summarize_log(records)
    assert summary["episodes"] == 5 and summary["grasp_attempts"] >= 5
    assert 0.0 <= summary["grasp_seal_rate"] <= 1.0
