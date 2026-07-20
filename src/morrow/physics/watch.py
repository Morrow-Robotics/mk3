"""watch.py — SAM2-assisted "watch a real customer clip → workflow → SO-101 pack".

This is the Phase-3 watch→workflow step, built to the honest boundary rather than
a faked one:

  * SAM2 (real hiera-tiny weights, runs on MPS) segments the carton and an
    operator-seeded product on a chosen frame of a REAL clip → precise pixel
    geometry. Single-frame image segmentation is reliable. Fully UNATTENDED video
    tracking is NOT: on cluttered consumer clips it drifts under hand occlusion
    (measured on pexels_7581335: SAM2 held the carton for ~43/90 propagated
    frames and its centre wandered >250 px as papers were inserted). So the
    honest product is operator-SEEDED / semi-automatic — the human supplies a
    click-box the way they actually would, and SAM2 turns it into a precise mask.

  * A monocular consumer clip carries no metric scale, so the operator supplies
    metres-per-pixel + the product kind/height — the same honesty as `annotate`.
    Pixels → metres uses that scale; nothing invents depth.

  * The watched workflow (pick this product, place it in that carton) is then
    reproduced by the REAL SO-101 in its validated reach envelope. It is NOT a
    pixel-perfect metric replay (impossible from an uncalibrated clip) — it is
    the same task, executed and verified in MuJoCo physics on the real arm.

`have_sam2()` gates everything: if the checkpoint is absent the caller skips,
exactly like the MuJoCo optional dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# SAM2 hiera-tiny checkpoint. Overridable; falls back to the known local path.
_CKPT_ENV = "MORROW_SAM2_CKPT"
_CKPT_CANDIDATES = [
    os.path.expanduser("~/Documents/morrow-video/models/poc3/sam2.1_hiera_tiny.pt"),
    os.path.expanduser("~/.cache/morrow/sam2.1_hiera_tiny.pt"),
]
_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Arm-graspable clamps (metres, half-extents). The SO-101 jaw + reach are small
# and its top-down envelope is proven only up to z≈0.10, so a watched product is
# mapped into what the real arm can actually grip AND reach (height bounded so the
# demo's over-approach stays tool-down-reachable). Honest: the workflow transfers,
# the absolute size does not — a monocular clip can't fix metric scale anyway.
ARM_GRASP_HALF = 0.02
ARM_MAX_HZ = 0.03
ARM_CARTON_HALF = (0.045, 0.05)


def sam2_checkpoint() -> str | None:
    cands = [os.environ[_CKPT_ENV]] if os.environ.get(_CKPT_ENV) else []
    cands += _CKPT_CANDIDATES
    return next((p for p in cands if p and os.path.isfile(p)), None)


def have_sam2() -> bool:
    if sam2_checkpoint() is None:
        return False
    try:
        import sam2  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class WatchedScene:
    """What we honestly read from one frame of a real clip."""
    clip: str
    frame_idx: int
    image_wh: tuple[int, int]
    product_bbox_px: tuple[int, int, int, int]
    carton_bbox_px: tuple[int, int, int, int]
    product_score: float
    carton_score: float
    product_mask: np.ndarray | None = None
    carton_mask: np.ndarray | None = None


class Sam2Segmenter:
    """Lazy wrapper over the SAM2 image predictor (built once, reused)."""

    _shared = None

    def __init__(self, ckpt: str | None = None, cfg: str = _CFG, device: str | None = None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt = ckpt or sam2_checkpoint()
        if ckpt is None:
            raise FileNotFoundError("no SAM2 checkpoint; set $MORROW_SAM2_CKPT")
        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=device))

    @classmethod
    def shared(cls) -> "Sam2Segmenter":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def segment(self, frame_rgb: np.ndarray, box=None, point=None):
        """Return (mask bool[H,W], bbox (x0,y0,x1,y1), score) for one prompt."""
        self.predictor.set_image(frame_rgb)
        kw: dict = {"multimask_output": True}
        if box is not None:
            kw["box"] = np.asarray(box, dtype=np.float32)
        if point is not None:
            kw["point_coords"] = np.asarray([point], dtype=np.float32)
            kw["point_labels"] = np.asarray([1])
        masks, scores, _ = self.predictor.predict(**kw)
        b = int(np.argmax(scores))
        mask = masks[b] > 0
        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise ValueError("SAM2 returned an empty mask for the prompt")
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        return mask, bbox, float(scores[b])


def read_frame(clip: str, frame_idx: int) -> np.ndarray:
    """Read one RGB frame from a clip."""
    import cv2
    cap = cv2.VideoCapture(clip)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read frame {frame_idx} of {clip}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_frame_time(clip: str, t_s: float) -> np.ndarray:
    """Read one RGB frame at a timestamp (seconds) — matches the dashboard's
    client-side `video.currentTime` grab."""
    import cv2
    cap = cv2.VideoCapture(clip)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(t_s) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read {clip} at t={t_s}s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def segment_click(clip: str, t_s: float, box_frac=None, point_frac=None,
                  segmenter: Sam2Segmenter | None = None) -> dict:
    """SAM2 on one operator seed (a fractional box or a fractional click point)
    at a clip timestamp. Returns fractional bbox + score for the dashboard."""
    seg = segmenter or Sam2Segmenter.shared()
    frame = read_frame_time(clip, t_s)
    H, W = frame.shape[:2]
    if box_frac is not None:
        _, bbox, score = seg.segment(frame, box=_px_box(box_frac, W, H))
    elif point_frac is not None:
        _, bbox, score = seg.segment(frame, point=[point_frac[0] * W, point_frac[1] * H])
    else:
        raise ValueError("segment_click needs a box_frac or a point_frac")
    x0, y0, x1, y1 = bbox
    return {"bbox_frac": [x0 / W, y0 / H, x1 / W, y1 / H], "score": round(float(score), 3),
            "wh": [W, H]}


def _px_box(frac, W, H):
    return [frac[0] * W, frac[1] * H, frac[2] * W, frac[3] * H]


def segment_scene(clip: str, frame_idx: int, carton_box_frac, product_box_frac,
                  segmenter: Sam2Segmenter | None = None, keep_masks: bool = True) -> WatchedScene:
    """Operator seeds fractional click-boxes for the carton and the product;
    SAM2 turns each into a precise mask + bbox on a real frame."""
    seg = segmenter or Sam2Segmenter.shared()
    frame = read_frame(clip, frame_idx)
    H, W = frame.shape[:2]
    cmask, cbb, cs = seg.segment(frame, box=_px_box(carton_box_frac, W, H))
    pmask, pbb, ps = seg.segment(frame, box=_px_box(product_box_frac, W, H))
    return WatchedScene(
        clip=os.path.basename(clip), frame_idx=frame_idx, image_wh=(W, H),
        product_bbox_px=pbb, carton_bbox_px=cbb, product_score=ps, carton_score=cs,
        product_mask=pmask if keep_masks else None,
        carton_mask=cmask if keep_masks else None,
    )


def render_overlay(clip: str, scene: WatchedScene, path: str, width: int = 720) -> str:
    """Draw the SAM2 masks + bboxes on the real frame — honest proof the CV ran."""
    import cv2
    frame = read_frame(clip, scene.frame_idx)
    ov = frame.copy()
    if scene.carton_mask is not None:
        ov[scene.carton_mask] = (0.55 * ov[scene.carton_mask] + 0.45 * np.array([210, 140, 20])).astype(np.uint8)
    if scene.product_mask is not None:
        ov[scene.product_mask] = (0.55 * ov[scene.product_mask] + 0.45 * np.array([30, 210, 70])).astype(np.uint8)
    cv2.rectangle(ov, scene.carton_bbox_px[:2], scene.carton_bbox_px[2:], (230, 150, 20), 3)
    cv2.rectangle(ov, scene.product_bbox_px[:2], scene.product_bbox_px[2:], (30, 220, 70), 3)
    H, W = ov.shape[:2]
    ov = cv2.resize(ov, (width, int(width * H / W)))
    cv2.imwrite(path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    return path


def scene_to_annotation(scene: WatchedScene, scale_m_per_px: float, kind: str = "box",
                        height_m: float = 0.06) -> dict:
    """The watched pixel geometry + an operator scale → the annotation schema."""
    return {
        "sku": f"watched-{os.path.splitext(scene.clip)[0]}",
        "image": {"w": scene.image_wh[0], "h": scene.image_wh[1]},
        "scale_m_per_px": float(scale_m_per_px),
        "product": {"kind": kind, "bbox_px": list(scene.product_bbox_px), "height_m": float(height_m)},
        "carton": {"bbox_px": list(scene.carton_bbox_px)},
    }


def _arm_size_and_carton(ann: dict):
    """Map watched metric extents into what the SO-101 can grip and reach."""
    scale = float(ann["scale_m_per_px"])
    x0, y0, x1, y1 = ann["product"]["bbox_px"]
    hx = min(ARM_GRASP_HALF, max(0.012, abs(x1 - x0) * scale / 2))
    hy = min(ARM_GRASP_HALF, max(0.012, abs(y1 - y0) * scale / 2))
    hz = min(ARM_MAX_HZ, max(0.02, float(ann["product"].get("height_m", 0.06)) / 2))
    cx0, cy0, cx1, cy1 = ann["carton"]["bbox_px"]
    from .arm import ARM_CARTON
    carton = dict(ARM_CARTON)
    carton["hx"] = min(ARM_CARTON_HALF[1], max(ARM_CARTON_HALF[0], abs(cx1 - cx0) * scale / 2))
    carton["hy"] = min(ARM_CARTON_HALF[1], max(ARM_CARTON_HALF[0], abs(cy1 - cy0) * scale / 2))
    return (hx, hy, hz), carton


def watch_and_pack_arm(scene: WatchedScene, scale_m_per_px: float, kind: str = "box",
                       height_m: float = 0.06, seed: int = 0, on_frame=None):
    """Full pipeline: watched scene → annotation → SO-101 physics pack.
    Returns (annotation, size, carton, result, world)."""
    from ..compile import compile_skill
    from ..execute import run_skill
    from .arm import ArmPerceiver, ArmRobot, ArmWorld, record_arm_demo
    ann = scene_to_annotation(scene, scale_m_per_px, kind, height_m)
    size, carton = _arm_size_and_carton(ann)
    if kind == "cylinder":
        size = (min(size[0], size[1]),) * 2 + (size[2],)  # round footprint
    demo = record_arm_demo(ArmWorld(kind, size=size, carton=carton))
    skill = compile_skill([demo], ann["sku"])
    world = ArmWorld(kind, size=size, carton=carton)
    robot = ArmRobot(world, on_frame=on_frame)
    result = run_skill(skill, robot, ArmPerceiver(world), seed=seed)
    return ann, size, carton, result, world


def pack_annotation_on_arm(ann: dict, seed: int = 0, capture: bool = False,
                           camera: str = "cell", height: int = 260, width: int = 340):
    """Run an operator annotation (product+carton bboxes, scale) on the REAL
    SO-101. Returns (frames, size, carton, result, world). box/cylinder only —
    the arm has no soft-body pouch grasp."""
    from ..compile import compile_skill
    from ..execute import run_skill
    from .arm import ArmPerceiver, ArmRobot, ArmWorld, record_arm_demo
    kind = ann["product"].get("kind", "box")
    if kind not in ("box", "cylinder"):
        raise ValueError(f"the SO-101 arm path packs box/cylinder, not {kind!r} "
                         "(a stand-up pouch needs a different end-effector)")
    if "carton" not in ann or not ann["carton"].get("bbox_px"):
        raise ValueError("mark the carton box too — the arm needs a place target")
    size, carton = _arm_size_and_carton(ann)
    if kind == "cylinder":
        size = (min(size[0], size[1]),) * 2 + (size[2],)
    skill = compile_skill([record_arm_demo(ArmWorld(kind, size=size, carton=carton))],
                          ann.get("sku", "marked"))
    world = ArmWorld(kind, size=size, carton=carton)
    per = ArmPerceiver(world)
    frames: list = []
    if capture:
        per.render(camera, height, width)
        robot = ArmRobot(world, on_frame=lambda: frames.append(per.render(camera, height, width)))
    else:
        robot = ArmRobot(world)
    result = run_skill(skill, robot, per, seed=seed)
    return frames, size, carton, result, world


def capture_watch_pack(scene: WatchedScene, scale_m_per_px: float, kind: str = "box",
                       height_m: float = 0.06, seed: int = 0, camera: str = "cell",
                       height: int = 260, width: int = 340):
    """Like `watch_and_pack_arm` but renders the SO-101 pack to frames for the
    dashboard. Returns (frames, annotation, size, carton, result, world)."""
    from ..compile import compile_skill
    from ..execute import run_skill
    from .arm import ArmPerceiver, ArmRobot, ArmWorld, record_arm_demo
    ann = scene_to_annotation(scene, scale_m_per_px, kind, height_m)
    size, carton = _arm_size_and_carton(ann)
    if kind == "cylinder":
        size = (min(size[0], size[1]),) * 2 + (size[2],)
    skill = compile_skill([record_arm_demo(ArmWorld(kind, size=size, carton=carton))], ann["sku"])
    world = ArmWorld(kind, size=size, carton=carton)
    per = ArmPerceiver(world)
    per.render(camera, height, width)  # warm renderer
    frames: list = []
    robot = ArmRobot(world, on_frame=lambda: frames.append(per.render(camera, height, width)))
    result = run_skill(skill, robot, per, seed=seed)
    return frames, ann, size, carton, result, world
