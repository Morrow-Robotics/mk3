"""Run a SkillProgram as a verified state machine.

Loop invariant: at the top of every iteration the world is re-perceived, and
every candidate attempt re-perceives again before it instantiates its motion.
So a failed grasp that nudged the pouch is seen before the next try. Recovery
is per-transition and always resumes from the robot's actual current state; it
never restarts a full skill from a physical state that no longer matches READY.

On exhausting an edge's retries the cell parks and flags — a composed failure,
which is what an investor (and a plant manager) actually needs to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .candidates import apply_feasibility, generate_candidates, rank_candidates
from .conditions import check
from .geometry import wrap_angle
from .motion import instantiate_edge
from .skill import SkillProgram, SkillState, next_state

MAX_CANDIDATES = 6  # attempts per edge per round
MAX_EDGE_RETRIES = 3  # recovery rounds before giving up on an edge
MAX_STEPS = 60  # global safety cap


@dataclass
class RunResult:
    sku_id: str
    success: bool
    final_state: str
    first_attempt_success: bool  # verified with no retries and no recoveries
    retries: int  # extra within-cell candidate attempts (autonomous)
    recoveries: int  # cross-transition recovery rounds (autonomous)
    attempts: int  # total transition attempts
    steps: int
    flagged: bool  # did the cell park-and-flag for a human
    failure_reason: str | None = None  # "EDGE:condition" that exhausted, if flagged
    timeline: list = field(default_factory=list)
    grasp_attempts: list = field(default_factory=list)  # {grasp params, sealed} per try


def _actuate(edge, robot) -> None:
    if edge[1] is SkillState.GRASPED:
        robot.engage()
    elif edge[1] is SkillState.RELEASED:
        robot.release()


def _prep(edge, robot) -> None:
    # Clean reset before any fresh grasp attempt; never retract while carrying.
    if edge[1] in (SkillState.APPROACHED, SkillState.GRASPED):
        robot.release()
        robot.safe_retract()


def _recover(action: str, robot) -> None:
    if action in ("retract_choose_next_grasp", "lower_release_regrasp"):
        robot.release()
        robot.safe_retract()
    elif action == "reopen_withdraw":
        robot.release()
    # reobserve_and_regenerate, stop_replan_transport, reobserve_escalate: just re-perceive


def run_skill(skill: SkillProgram, robot, perceiver, seed: int = 42,
              journal=None, ranker=None) -> RunResult:
    skill.validate()
    current = SkillState.READY
    recoveries = 0
    retries = 0
    attempts_total = 0
    steps = 0
    edge_rounds: dict = {}
    timeline: list = []
    grasp_attempts: list = []
    failure_reason = None

    def emit(ev: dict) -> None:
        timeline.append(ev)

    # Selection gate: if perception can't tell the target from a distractor,
    # flag for a human rather than grasp the wrong SKU.
    if perceiver.observe().uncertainty.get("ambiguous"):
        robot.park_and_flag()
        current = SkillState.FAILED
        failure_reason = "SELECTION:ambiguous"
        emit({"edge": "SELECTION", "outcome": "failed", "reason": failure_reason})

    while current not in (SkillState.VERIFIED, SkillState.FAILED) and steps < MAX_STEPS:
        steps += 1
        target = next_state(current)
        edge = (current, target)
        tr = skill.transition(current, target)
        rnd = edge_rounds.get(edge, 0)

        scene = perceiver.observe()
        cands = generate_candidates(skill, scene, edge, seed, rnd, n=20)
        cands = apply_feasibility(cands, skill, scene, robot, edge)
        ranked = rank_candidates(cands, skill, scene, edge, ranker=ranker)[:MAX_CANDIDATES]

        advanced = False
        attempts = 0
        for c in ranked:
            attempts += 1
            _prep(edge, robot)
            scene = perceiver.observe()  # re-perceive before this attempt
            robot.follow(instantiate_edge(skill, edge, scene, robot, c.params))
            _actuate(edge, robot)
            scene = perceiver.observe()
            ok = check(tr.success, scene, robot)
            if edge[1] is SkillState.GRASPED:
                rel = wrap_angle(float(c.params["grasp_yaw"]) - scene.product_yaw_candidates[0])
                grasp_attempts.append({
                    "grasp_yaw": round(float(c.params["grasp_yaw"]), 4),
                    "grasp_yaw_rel": round(float(rel), 4),  # yaw vs detected axis: the learnable feature
                    "grasp_offset_noise": [round(float(v), 4) for v in c.params["grasp_offset_noise"]],
                    "sealed": bool(ok),
                })
            if ok:
                emit({"edge": f"{current.value}->{target.value}", "outcome": "ok",
                      "attempts": attempts, "candidate": c.id, "round": rnd})
                current = target
                advanced = True
                break

        attempts_total += attempts
        if advanced:
            retries += attempts - 1

        if not advanced:
            edge_rounds[edge] = rnd + 1
            recoveries += 1
            emit({"edge": f"{current.value}->{target.value}", "outcome": "recover",
                  "attempts": attempts, "action": tr.recovery["action"], "round": rnd})
            if edge_rounds[edge] > MAX_EDGE_RETRIES:
                current = SkillState.FAILED
                failure_reason = f"{edge[0].value}->{edge[1].value}:{tr.success['name']}"
                robot.park_and_flag()
                emit({"edge": f"{edge[0].value}->{edge[1].value}", "outcome": "failed",
                      "attempts": attempts, "reason": failure_reason})
                break
            _recover(tr.recovery["action"], robot)
            current = tr.recovery["next_state"]

    success = current is SkillState.VERIFIED
    flagged = getattr(getattr(robot, "world", None), "flagged", False)
    result = RunResult(
        sku_id=skill.sku_id,
        success=success,
        final_state=current.value,
        first_attempt_success=success and retries == 0 and recoveries == 0,
        retries=retries,
        recoveries=recoveries,
        attempts=attempts_total,
        steps=steps,
        flagged=bool(flagged),
        failure_reason=failure_reason,
        timeline=timeline,
        grasp_attempts=grasp_attempts,
    )
    if journal is not None:
        from .journal import record_from_result
        journal.append(record_from_result(result, skill, seed))
    return result
