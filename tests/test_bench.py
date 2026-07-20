import numpy as np
import pytest

from morrow import DemonstrationTrace, Perceiver, Robot
from morrow.bench import BenchPerceiver, BenchRecorder, BenchRobot
from morrow.geometry import frame


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


def test_recorder_assembly_produces_a_valid_trace():
    # The frame-assembly path is real and testable without hardware.
    rec = BenchRecorder("cam-0", "test", table_height=0.0)
    mask = np.zeros((4, 4), dtype=bool)
    for i in range(3):
        rec.add_frame(timestamp=i * 0.1, ee_pose=frame((0, 0, 0.2 - i * 0.05)),
                      gripper_command=float(i == 2), gripper_signal=0.9 - 0.4 * i,
                      product_mask=mask, product_centroid=np.zeros(3),
                      product_footprint=np.array([-0.03, -0.03, 0.03, 0.03]), product_yaw=0.0)
    trace = rec.finalize(carton_frame=frame((0.2, 0, 0)), meta={"kind": "box"})
    assert isinstance(trace, DemonstrationTrace) and len(trace) == 3


def test_recorder_live_capture_is_an_honest_stub():
    rec = BenchRecorder("cam-0", "test", table_height=0.0)
    with pytest.raises(NotImplementedError):
        rec.record_live(robot=None, perceiver=None, is_recording=lambda: False,
                        carton_frame=frame((0.2, 0, 0)))
