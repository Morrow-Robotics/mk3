"""Packing-activity extraction — pure classical CV (cv2 frame-diff), no SAM2, no
MuJoCo. Guarded on cv2 + a real clip being present.

These assert what is HONESTLY true: the signal is deterministic and bounded, and
the profile advertises its own low confidence. They deliberately do NOT assert an
exact item count, because a verified item count is not recoverable from these
occluded consumer clips (see pattern.py).
"""

import os

import pytest

pytest.importorskip("cv2")

from morrow.physics.pattern import carton_activity, packing_profile

_CLIP = os.path.join(os.path.dirname(__file__), os.pardir, "videos", "pexels_7581335.mp4")
_REGION = (0.16, 0.30, 0.80, 0.92)

if not os.path.isfile(_CLIP):
    pytest.skip("real clip not present in ./videos", allow_module_level=True)


def test_carton_activity_returns_a_signal():
    ts, motion, fps = carton_activity(_CLIP, _REGION)
    assert len(ts) == len(motion) and len(motion) > 5 and fps > 0
    assert (motion >= 0).all()  # frame-difference magnitudes


def test_packing_profile_is_deterministic_and_bounded():
    p1 = packing_profile(_CLIP, _REGION)
    p2 = packing_profile(_CLIP, _REGION)
    assert p1.n_events == p2.n_events and p1.confidence == p2.confidence
    assert p1.motion == p2.motion                      # fully deterministic
    assert 0.0 <= p1.confidence <= 1.0
    assert 0.0 <= p1.active_fraction <= 1.0
    assert p1.confidence_label in ("LOW", "MEDIUM", "HIGH")
    assert p1.n_events >= 1 and p1.duration_s > 0


def test_profile_note_is_honest_about_confidence():
    p = packing_profile(_CLIP, _REGION)
    assert "operator-confirmable" in p.note
    assert "not a verified item count" in p.note
