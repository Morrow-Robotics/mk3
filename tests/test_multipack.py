"""Multi-item pack-out with the floating parallel-jaw gripper (guarded on MuJoCo).

Asserts what is honestly true and reliable: the floating gripper (mocap-exact
placement) grasps and seats a full grid of boxes into one carton — unlike the
SO-101 model, whose small reliable-reach zone can't place a grid.
"""

import pytest

pytest.importorskip("mujoco")

from morrow.physics.multipack import pack_boxes


def test_floating_gripper_packs_a_box_grid():
    cell, grasped, seated = pack_boxes(n=4)
    assert grasped == 4                      # every box picked (contact grasp)
    assert seated == 4                       # every box physically inside the carton
    # the boxes occupy DISTINCT slots (a grid), not a single pile
    xy = {(round(float(cell.data.xpos[b][0]), 2), round(float(cell.data.xpos[b][1]), 2))
          for b in cell.boxes}
    assert len(xy) == 4


def test_multipack_is_deterministic():
    _, g1, s1 = pack_boxes(n=4)
    _, g2, s2 = pack_boxes(n=4)
    assert (g1, s1) == (g2, s2) == (4, 4)
