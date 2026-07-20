"""Multi-object packing: several SKUs into one carton, in order.

A real FSM extension that adds *no* new state machine. Each item is packed by
the ordinary single-product `run_skill`; the only new pieces are:

- **per-item skill lookup** — the item's SKU picks which compiled skill runs;
- **descriptor selection** — the target is chosen among the still-staged
  products by the skill's descriptor (the ambiguity gate still applies, so two
  identical SKUs correctly refuse to guess);
- **place slots** — the item's skill is cloned with its carton-frame place
  offset shifted to a distinct slot, so items don't land on each other.

The single-product path is untouched: `with_place_slot` returns a new skill and
the executor never learns about sequences.

Note: the analytic sim does not model item-on-item collision — slots are chosen
to be disjoint; a real cell would verify clearance. That honesty is the point.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .execute import run_skill
from .skill import SkillProgram, SkillState, hash_skill
from .sim.scenarios import default_carton, make_product
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
    slot: tuple  # carton-frame xy


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
                      layout_seed: int = 100, journal=None, ranker=None) -> SequenceResult:
    """Pack `items` (each an already-onboarded SKU) into one carton, in order."""
    rng = np.random.RandomState(layout_seed)
    n = len(items)
    products = []
    for i, it in enumerate(items):  # stage each item at a distinct spot on the infeed
        cx = -0.18 + rng.uniform(-0.02, 0.02)
        cy = -0.15 + i * (0.30 / max(1, n - 1)) + rng.uniform(-0.02, 0.02) if n > 1 else 0.0
        products.append(make_product(it.kind, cx, cy, rng.uniform(-np.pi, np.pi)))

    results = []
    for k, it in enumerate(items):
        # target = this item; distractors = the items still waiting behind it
        world = World(products[k], default_carton(), slip_prob=0.12,
                      distractors=products[k + 1:], rng=np.random.RandomState(seed * 1000 + k))
        skill = skills[it.kind]
        shifted = with_place_slot(skill, it.slot)
        perceiver = SimPerceiver(world, target_descriptor=skill.object_descriptor)
        r = run_skill(shifted, SimRobot(world), perceiver, seed=seed * 100 + k,
                      journal=journal, ranker=ranker)
        results.append(ItemResult(it.sku, it.kind, tuple(it.slot), r.success,
                                  r.final_state, r.failure_reason))

    packed = sum(x.success for x in results)
    return SequenceResult(results=results, packed=packed, total=n, success=(packed == n))


# A canonical high-mix order: one of each, three disjoint slots.
DEFAULT_ORDER = [
    PackItem("box-A", "box", (-0.055, -0.05)),
    PackItem("cyl-B", "cylinder", (0.055, -0.05)),
    PackItem("pouch-C", "pouch", (0.0, 0.055)),
]


def demo_pack_sequence(seed: int = 0, journal=None) -> SequenceResult:
    """Onboard box/cylinder/pouch and pack one of each into a single carton."""
    from .sim.scenarios import onboard
    skills = {k: onboard(k, k) for k in ("box", "cylinder", "pouch")}
    return run_pack_sequence(DEFAULT_ORDER, skills, seed=seed, journal=journal)
