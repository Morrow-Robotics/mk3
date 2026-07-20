import numpy as np

from morrow.geometry import bbox_overlap_area, frame, inv, rot2, translation, xy_in_bbox, yaw_of


def test_inverse_round_trips():
    T = frame((0.1, -0.2, 0.3), yaw=0.7)
    assert np.allclose(inv(T) @ T, np.eye(4), atol=1e-12)


def test_translation_and_yaw_readback():
    T = frame((1.0, 2.0, 3.0), yaw=-0.4)
    assert np.allclose(translation(T), [1.0, 2.0, 3.0])
    assert abs(yaw_of(T) - (-0.4)) < 1e-12


def test_rot2_rotates_a_vector():
    v = np.array([1.0, 0.0])
    assert np.allclose(rot2(np.pi / 2) @ v, [0.0, 1.0], atol=1e-12)


def test_xy_in_bbox():
    box = [0.0, 0.0, 1.0, 1.0]
    assert xy_in_bbox((0.5, 0.5), box)
    assert not xy_in_bbox((1.5, 0.5), box)


def test_bbox_overlap_area():
    a = [0.0, 0.0, 2.0, 2.0]
    b = [1.0, 1.0, 3.0, 3.0]
    assert bbox_overlap_area(a, b) == 1.0  # unit square overlap
    assert bbox_overlap_area(a, [3.0, 3.0, 4.0, 4.0]) == 0.0  # disjoint
