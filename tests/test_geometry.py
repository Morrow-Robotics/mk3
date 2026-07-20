import numpy as np

from morrow.geometry import apply_frame, frame, inv, relative, translation, yaw_of


def test_inverse_round_trips():
    T = frame((0.1, -0.2, 0.3), yaw=0.7)
    assert np.allclose(inv(T) @ T, np.eye(4), atol=1e-12)


def test_relative_and_apply_are_inverses():
    ref = frame((0.2, 0.0, 0.1), yaw=0.5)
    pose = frame((0.25, 0.05, 0.15), yaw=0.9)
    rel = relative(ref, pose)
    assert np.allclose(apply_frame(ref, rel), pose, atol=1e-12)


def test_translation_and_yaw_readback():
    T = frame((1.0, 2.0, 3.0), yaw=-0.4)
    assert np.allclose(translation(T), [1.0, 2.0, 3.0])
    assert abs(yaw_of(T) - (-0.4)) < 1e-12
