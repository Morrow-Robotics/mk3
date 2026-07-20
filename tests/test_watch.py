"""Guarded tests for the SAM2 watch→workflow→SO-101 pipeline.

Skipped unless BOTH MuJoCo and a real SAM2 checkpoint are present — this exercises
genuine CV (SAM2 segments a real customer clip), so with no weights there is
nothing honest to assert.
"""

import os

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("sam2")

from morrow.physics.watch import (have_sam2, scene_to_annotation, segment_scene,
                                  watch_and_pack_arm)

if not have_sam2():
    pytest.skip("no SAM2 checkpoint (set $MORROW_SAM2_CKPT)", allow_module_level=True)

_CLIP = os.path.join(os.path.dirname(__file__), os.pardir, "videos", "pexels_7581335.mp4")

if not os.path.isfile(_CLIP):
    pytest.skip("real customer clip not present in ./videos", allow_module_level=True)


def test_sam2_segments_a_real_clip_frame():
    scene = segment_scene(_CLIP, 18, carton_box_frac=(0.16, 0.30, 0.80, 0.92),
                          product_box_frac=(0.30, 0.40, 0.62, 0.74))
    W, H = scene.image_wh
    assert (W, H) == (1920, 1080)                      # the real clip's resolution
    # SAM2 returned real, non-degenerate masks for both prompts
    assert scene.carton_score > 0.2 and scene.product_score > 0.2
    cbb = scene.carton_bbox_px
    assert 0 <= cbb[0] < cbb[2] <= W and 0 <= cbb[1] < cbb[3] <= H
    assert scene.carton_mask is not None and scene.carton_mask.any()


def test_watched_scene_drives_the_real_arm_to_verified():
    scene = segment_scene(_CLIP, 18, carton_box_frac=(0.16, 0.30, 0.80, 0.92),
                          product_box_frac=(0.30, 0.40, 0.62, 0.74))
    ann, size, carton, result, world = watch_and_pack_arm(
        scene, scale_m_per_px=0.00024, kind="box", height_m=0.06, seed=0)
    assert result.success and result.final_state == "VERIFIED"
    px, py = world.product_xy()
    c = world.carton
    assert c["cx"] - c["hx"] <= px <= c["cx"] + c["hx"]   # physically in the carton
    assert c["cy"] - c["hy"] <= py <= c["cy"] + c["hy"]
    # the mapped product stays inside the arm's proven graspable envelope
    assert max(size[0], size[1]) <= 0.02 and size[2] <= 0.03


def test_segment_click_refines_a_rough_box():
    # The interactive dashboard path: operator drags a rough box, SAM2 tightens it.
    from morrow.physics.watch import segment_click
    out = segment_click(_CLIP, 0.6, box_frac=(0.16, 0.30, 0.80, 0.92))
    assert out["wh"] == [1920, 1080]
    bf = out["bbox_frac"]
    assert 0.0 <= bf[0] < bf[2] <= 1.0 and 0.0 <= bf[1] < bf[3] <= 1.0
    assert out["score"] > 0.2


def test_marked_annotation_packs_on_the_so101():
    # An operator annotation (canvas px + scale) drives the REAL arm to VERIFIED.
    from morrow.physics.watch import pack_annotation_on_arm
    ann = {"sku": "marked", "image": {"w": 480, "h": 270}, "scale_m_per_px": 0.0007,
           "product": {"kind": "box", "bbox_px": [210, 120, 300, 190], "height_m": 0.06},
           "carton": {"bbox_px": [300, 150, 430, 250]}}
    _frames, size, _c, result, world = pack_annotation_on_arm(ann, capture=False)
    assert result.success and result.final_state == "VERIFIED"
    assert max(size[0], size[1]) <= 0.02 and size[2] <= 0.03


def test_arm_pack_rejects_pouch_and_missing_carton():
    from morrow.physics.watch import pack_annotation_on_arm
    with pytest.raises(ValueError):  # no soft-body pouch grasp on this arm
        pack_annotation_on_arm({"product": {"kind": "pouch", "bbox_px": [0, 0, 40, 40]},
                                "scale_m_per_px": 0.001, "carton": {"bbox_px": [0, 0, 80, 80]}})
    with pytest.raises(ValueError):  # arm needs a place target
        pack_annotation_on_arm({"product": {"kind": "box", "bbox_px": [0, 0, 40, 40]},
                                "scale_m_per_px": 0.001})


def test_scene_to_annotation_matches_schema():
    scene = segment_scene(_CLIP, 18, carton_box_frac=(0.16, 0.30, 0.80, 0.92),
                          product_box_frac=(0.30, 0.40, 0.62, 0.74))
    ann = scene_to_annotation(scene, scale_m_per_px=0.00024, kind="box", height_m=0.06)
    assert ann["scale_m_per_px"] == 0.00024
    assert ann["product"]["kind"] == "box" and len(ann["product"]["bbox_px"]) == 4
    assert len(ann["carton"]["bbox_px"]) == 4
