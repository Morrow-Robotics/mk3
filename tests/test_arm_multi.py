"""Multi-object pack-out on the SO-101 model (guarded on MuJoCo).

The SO-101 arm packs several boxes into one carton as a sequence, using the same
grasp-verify-retry and place-verify-recover the FSM uses — so it stays reliable on
this small arm AND ports 1:1 to the physical arm (`SO101BenchRobot`).
"""

import pytest

pytest.importorskip("mujoco")

from morrow.physics.arm_multi import pack_objects


def test_so101_packs_multiple_objects():
    w, seated, status = pack_objects([(0.015, 0.02)] * 3)
    assert seated == 3 and all(status)          # all three physically in the carton
    # distinct positions inside the carton (a pack, not a single pile)
    xy = {(round(float(w.data.xpos[b][0]), 2), round(float(w.data.xpos[b][1]), 2))
          for b in w.boxes}
    assert len(xy) == 3


def test_so101_multi_is_deterministic():
    _, s1, _ = pack_objects([(0.015, 0.02)] * 2)
    _, s2, _ = pack_objects([(0.015, 0.02)] * 2)
    assert s1 == s2 == 2
