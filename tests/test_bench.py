import pytest

from morrow import Perceiver, Robot
from morrow.bench import BenchPerceiver, BenchRobot


def test_bench_adapters_satisfy_the_boundaries():
    # Structural conformance: run_skill will accept these once the TODOs are filled.
    assert isinstance(BenchRobot(), Robot)
    assert isinstance(BenchPerceiver(), Perceiver)


def test_bench_hardware_calls_are_honest_stubs():
    r = BenchRobot()
    with pytest.raises(NotImplementedError):
        r.get_ee_pose()
    with pytest.raises(NotImplementedError):
        r.follow([])
    with pytest.raises(NotImplementedError):
        BenchPerceiver().observe()


def test_bench_holding_is_derived_not_stubbed():
    # holding() should be pure threshold logic over gripper_signal, no vision.
    class FakeSensorRobot(BenchRobot):
        def gripper_signal(self):
            return 0.1  # sealed

    assert FakeSensorRobot(vacuum_threshold=0.5).holding() is True
