"""Open-loop replay: the honest baseline for the changeover contrast.

This is NOT a strawman "policy with no recovery". It is exactly what a fixed,
taught-once industrial cell does: replay the demonstrated joint/pose trajectory
open-loop. It succeeds on the pose it was taught and fails the moment the
product moves — which is the whole reason perception + a verified skill matter.
"""

from __future__ import annotations

from ..conditions import check
from ..skill import SkillProgram, SkillState
from ..trace import DemonstrationTrace
from ..sim.sim_perceive import SimPerceiver
from ..sim.sim_robot import SimRobot
from ..sim.world import World


def run_fixed_replay(trace: DemonstrationTrace, skill: SkillProgram, world: World) -> bool:
    """Replay the demonstrated absolute poses on `world`; return whether it verified."""
    robot = SimRobot(world)
    perceiver = SimPerceiver(world)
    for pose, cmd in zip(trace.ee_poses, trace.gripper_command):
        robot.follow([pose])
        if cmd >= 0.5 and not robot.holding():
            robot.engage()
        elif cmd < 0.5 and robot.holding():
            robot.release()
    verified = skill.transition(SkillState.RELEASED, SkillState.VERIFIED).success
    return check(verified, perceiver.observe(), robot)
