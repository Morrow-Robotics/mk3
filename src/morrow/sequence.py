"""Multi-object packing: several SKUs into one carton, in order.

A real FSM extension that adds *no* new state machine. Each item is packed by
the ordinary single-product `run_skill`; the only new pieces are:

- **per-item skill lookup** — the item's SKU picks which compiled skill runs;
- **descriptor selection** — the target is chosen among the still-staged
  products by the skill's descriptor (the ambiguity gate still applies, so two
  identical SKUs correctly refuse to guess);
- **place slots** — the item's skill is cloned (`with_place_slot`) with its
  carton-frame place offset shifted to a free slot; already-placed items are
  passed into the world as `occupied`, and the place feasibility gate rejects
  any candidate whose footprint would land on them. Slots are auto-assigned
  (first-fit with clearance) or given per item.

The single-product path is untouched: `with_place_slot` returns a new skill, the
world's `occupied` list is empty by default, and the executor never learns about
sequences. Clearance is checked from footprint AABBs — a 2.5D approximation an
honest bench would refine with real geometry.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .execute import run_skill
from .geometry import bbox_overlap_area
from .skill import SkillProgram, SkillState, hash_skill
from .sim.scenarios import make_product, pack_carton
from .sim.sim_perceive import SimPerceiver
from .sim.sim_robot import SimRobot
from .sim.world import World

_CARTON_EDGES = ((SkillState.LIFTED, SkillState.OVER_CARTON),
                 (SkillState.OVER_CARTON, SkillState.PLACED))


def with_place_slot(skill: SkillProgram, slot_xy) -> SkillProgram:
    """A copy of `skill` whose carton-frame place offset is shifted to `slot_xy`."""
    s = copy.deepcopy(skill)
    for edge in _CARTON_EDGES:
        t = s.transitions[edge]
        po = t.rel["place_offset"]
        t.rel = dict(t.rel, place_offset=[po[0] + slot_xy[0], po[1] + slot_xy[1]])
    s.version_hash = hash_skill(s.transitions, s.object_descriptor, s.grasp_regions)
    s.metadata = dict(s.metadata, place_slot=[float(slot_xy[0]), float(slot_xy[1])])
    return s


@dataclass
class PackItem:
    sku: str
    kind: str
    slot: tuple | None = None  # carton-frame xy; None => auto-assign a free slot


# Candidate slots (carton frame), tried in order — corners first so several
# items with rotated footprints fit inside one opening without overlapping.
SLOT_GRID = [(-0.07, -0.07), (0.07, -0.07), (0.0, 0.07),
             (-0.07, 0.07), (0.07, 0.07), (0.0, -0.07), (0.0, 0.0)]


def _aabb_half(product) -> tuple[float, float]:
    fp = product.footprint()
    return (fp[2] - fp[0]) / 2, (fp[3] - fp[1]) / 2


def _slot_footprint(product, carton, slot) -> np.ndarray:
    hx, hy = _aabb_half(product)
    bx, by = carton.cx + slot[0], carton.cy + slot[1]
    return np.array([bx - hx, by - hy, bx + hx, by + hy])


def _within(inner, outer) -> bool:
    return bool(inner[0] >= outer[0] and inner[1] >= outer[1]
                and inner[2] <= outer[2] and inner[3] <= outer[3])


def _overlaps(a, b, clearance: float = 0.005) -> bool:
    inflated = (a[0] - clearance, a[1] - clearance, a[2] + clearance, a[3] + clearance)
    return bbox_overlap_area(inflated, b) > 0.0


def _assign_slot(product, carton, placed):
    """First free slot whose footprint fits the opening and clears placed items."""
    opening = carton.opening()
    for slot in SLOT_GRID:
        fp = _slot_footprint(product, carton, slot)
        if not _within(fp, opening):
            continue
        if any(_overlaps(fp, p) for p in placed):
            continue
        return slot, fp
    return None, None


@dataclass
class ItemResult:
    sku: str
    kind: str
    slot: tuple
    success: bool
    final_state: str
    failure_reason: str | None


@dataclass
class SequenceResult:
    results: list = field(default_factory=list)
    packed: int = 0
    total: int = 0
    success: bool = False


def run_pack_sequence(items: list[PackItem], skills: dict, seed: int = 0,
                      layout_seed: int = 100, journal=None, ranker=None,
                      policy: str = "skip") -> SequenceResult:
    """Pack `items` (each an already-onboarded SKU) into one carton, in order.

    Placed items become obstacles for the ones after them (real clearance is
    verified by the place feasibility gate). Slots may be given per item or
    auto-assigned. `policy`: "skip" (record the failure, keep going) or "halt"
    (stop the whole sequence at the first item that fails)."""
    if policy not in ("skip", "halt"):
        raise ValueError(f"policy must be 'skip' or 'halt', got {policy!r}")
    rng = np.random.RandomState(layout_seed)
    n = len(items)
    products = []
    for i, it in enumerate(items):  # stage each item at a distinct spot on the infeed
        cx = -0.18 + rng.uniform(-0.02, 0.02)
        cy = -0.15 + i * (0.30 / max(1, n - 1)) + rng.uniform(-0.02, 0.02) if n > 1 else 0.0
        products.append(make_product(it.kind, cx, cy, rng.uniform(-np.pi, np.pi)))

    placed: list = []  # base-frame footprints already in the carton
    results = []
    for k, it in enumerate(items):
        product = products[k]
        carton = pack_carton()
        if it.slot is not None:
            slot, slot_fp = tuple(it.slot), _slot_footprint(product, carton, it.slot)
        else:
            slot, slot_fp = _assign_slot(product, carton, placed)
        if slot is None:
            results.append(ItemResult(it.sku, it.kind, None, False, "FAILED", "PLACE:no_slot"))
            if policy == "halt":
                break
            continue

        world = World(product, carton, slip_prob=0.12, distractors=products[k + 1:],
                      occupied=placed, rng=np.random.RandomState(seed * 1000 + k))
        skill = skills[it.kind]
        shifted = with_place_slot(skill, slot)
        perceiver = SimPerceiver(world, target_descriptor=skill.object_descriptor)
        r = run_skill(shifted, SimRobot(world), perceiver, seed=seed * 100 + k,
                      journal=journal, ranker=ranker,
                      extra={"in_sequence": True, "seq_item": k,
                             "seq_slot": [float(slot[0]), float(slot[1])], "kind": it.kind})
        results.append(ItemResult(it.sku, it.kind, tuple(slot), r.success,
                                  r.final_state, r.failure_reason))
        if r.success:
            placed.append(slot_fp)  # now occupies the carton for later items
        elif policy == "halt":
            break

    packed = sum(x.success for x in results)
    return SequenceResult(results=results, packed=packed, total=n, success=(packed == n))


# A canonical high-mix order: one of each; slots auto-assigned with clearance.
DEFAULT_ORDER = [
    PackItem("box-A", "box"),
    PackItem("cyl-B", "cylinder"),
    PackItem("pouch-C", "pouch"),
]


def demo_pack_sequence(seed: int = 0, journal=None) -> SequenceResult:
    """Onboard box/cylinder/pouch and pack one of each into a single carton."""
    from .sim.scenarios import onboard
    skills = {k: onboard(k, k) for k in ("box", "cylinder", "pouch")}
    return run_pack_sequence(DEFAULT_ORDER, skills, seed=seed, journal=journal)
