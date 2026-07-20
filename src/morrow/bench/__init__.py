"""Physical-bench adapters (skeletons) for the Robot/Perceiver boundaries.

Not runnable without hardware — every hardware call raises NotImplementedError
with a wiring note. They satisfy the boundary Protocols structurally, so the
compiler, executor, and eval accept them unchanged once filled. See BENCH.md.
"""

from .bench_perceive import BenchPerceiver
from .bench_robot import BenchRobot

__all__ = ["BenchRobot", "BenchPerceiver"]
