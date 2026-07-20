"""Human-in-the-loop: a marked-up customer clip -> a physics-runnable skill.

This is NOT automatic video->skill extraction (no CV / SAM here). An operator
watches a customer packing clip and MARKS the ground truth: the product kind,
its real size (an image bounding box + a stated table scale in metres/pixel),
and — optionally — a place designation. That annotation configures a MuJoCo cell
matching the customer's product, records a parallel-jaw demo, and compiles a
verified skill that runs in real physics. The honesty is the point: the human
supplies what a human actually gets from watching, and nothing is faked.

Annotation JSON schema:
    {
      "sku": "cust-box",                         # optional label
      "image": {"w": 1280, "h": 720},            # informational
      "scale_m_per_px": 0.0006,                  # metres per pixel on the table plane
      "product": {
        "kind": "box" | "cylinder" | "pouch",
        "bbox_px": [x0, y0, x1, y1],             # product extent in the image
        "height_m": 0.06                         # stated product height
      }
    }
"""

from __future__ import annotations

from ..compile import compile_skill
from ..execute import run_skill
from .mj_perceive import MjPerceiver
from .mj_robot import MjRobot
from .record import record_mj_demo
from .scene import CARTON
from .world import MjWorld

GRIPPER_MAX_HALF = 0.038  # object half-width (m) the standard parallel jaw can straddle


def size_from_annotation(ann: dict) -> tuple[float, float, float]:
    scale = float(ann["scale_m_per_px"])
    x0, y0, x1, y1 = ann["product"]["bbox_px"]
    hx = abs(x1 - x0) * scale / 2
    hy = abs(y1 - y0) * scale / 2
    hz = float(ann["product"].get("height_m", 0.06)) / 2
    return hx, hy, hz


def _carton_from_annotation(ann: dict) -> dict:
    c = dict(CARTON)
    cm = ann.get("carton")
    if cm and cm.get("bbox_px"):
        scale = float(ann["scale_m_per_px"])
        x0, y0, x1, y1 = cm["bbox_px"]
        c = dict(c, hx=min(0.16, max(0.06, abs(x1 - x0) * scale / 2)),
                 hy=min(0.16, max(0.06, abs(y1 - y0) * scale / 2)))
    return c


def build_skill_from_annotation(ann: dict):
    """Return (skill, cfg) from an operator annotation; cfg has kind/size/carton."""
    kind = ann["product"].get("kind", "box")
    hx, hy, hz = size_from_annotation(ann)
    if min(hx, hy) > GRIPPER_MAX_HALF:
        raise ValueError(
            f"product half-width {min(hx, hy):.3f} m exceeds the parallel jaw's "
            f"{GRIPPER_MAX_HALF} m reach — this SKU needs a wider gripper (honest limit)")
    carton = _carton_from_annotation(ann)
    world = MjWorld(kind, size=(hx, hy, hz), carton=carton)
    skill = compile_skill([record_mj_demo(world)], ann.get("sku", kind))
    return skill, {"kind": kind, "size": (hx, hy, hz), "carton": carton}


def run_annotation(ann: dict, px: float = -0.15, py: float = 0.0, pyaw: float = 0.0,
                   seed: int = 0):
    skill, cfg = build_skill_from_annotation(ann)
    world = MjWorld(cfg["kind"], px=px, py=py, pyaw=pyaw, size=cfg["size"], carton=cfg["carton"])
    result = run_skill(skill, MjRobot(world), MjPerceiver(world), seed=seed)
    return skill, world, result


def capture_annotation(ann: dict, px: float = -0.15, py: float = 0.0, pyaw: float = 0.0,
                       seed: int = 0, camera: str = "side", height: int = 240, width: int = 320):
    skill, cfg = build_skill_from_annotation(ann)
    world = MjWorld(cfg["kind"], px=px, py=py, pyaw=pyaw, size=cfg["size"], carton=cfg["carton"])
    per = MjPerceiver(world)
    per.render(camera, height, width)
    frames: list = []
    robot = MjRobot(world, on_frame=lambda: frames.append(per.render(camera, height, width)))
    result = run_skill(skill, robot, per, seed=seed)
    return frames, result, world
