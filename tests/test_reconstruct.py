"""Video → reconstructed SO-101 scene → the arm packs the objects (guarded).

Skipped unless BOTH MuJoCo and a real SAM2 checkpoint are present — this runs
genuine CV (SAM2 segments a real customer clip) end to end into a physics pack.
"""

import os

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("sam2")

from morrow.physics.reconstruct import VIDEO_SCENES, reconstruct_and_pack, reconstruct_scene
from morrow.physics.watch import have_sam2

if not have_sam2():
    pytest.skip("no SAM2 checkpoint (set $MORROW_SAM2_CKPT)", allow_module_level=True)

_VID = os.path.join(os.path.dirname(__file__), os.pardir, "videos")
if not os.path.isdir(_VID) or not os.listdir(_VID):
    pytest.skip("real customer clips not present in ./videos", allow_module_level=True)


def _cfg_for(name):
    for c in VIDEO_SCENES:
        if c["clip"].endswith(name) and os.path.isfile(c["clip"]):
            return c
    return None


def test_reconstruct_detects_carton_and_objects():
    cfg = _cfg_for("pexels_7581335.mp4")
    if cfg is None:
        pytest.skip("clip not present")
    carton, sizes, wh, dets = reconstruct_scene(cfg)
    assert wh == (1920, 1080)
    assert len(sizes) == len(cfg["object_boxes_frac"])   # one graspable box per object
    assert dets[0][2] is True and dets[0][1] > 0.2       # carton detected with a score
    assert len(dets) == 1 + len(cfg["object_boxes_frac"])


def test_so101_packs_the_reconstructed_objects():
    cfg = _cfg_for("pexels_7581335.mp4")
    if cfg is None:
        pytest.skip("clip not present")
    r = reconstruct_and_pack(cfg)
    assert r["n"] == 3
    assert r["seated"] == r["n"]                          # SO-101 seats every object
    assert len(r["frames"]) > 5                           # a real rendered clip
