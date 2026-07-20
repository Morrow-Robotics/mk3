"""`morrow` command line: onboard, run, eval, demo."""

from __future__ import annotations

import argparse

import numpy as np


def _onboard(args) -> None:
    from .pipeline import skill_graph
    from .sim import onboard
    skill = onboard(args.kind, args.kind, n_demos=args.demos)
    g = skill_graph(skill)
    print(f"skill {g['sku_id']}  hash {g['hash']}  ({g['descriptor']['kind']}, "
          f"symmetry {g['descriptor']['symmetry']})")
    print(f"phases {g['phase_indices']}")
    for e in g["edges"]:
        print(f"  {e['from']:>11} -> {e['to']:<11}  frame={e['frame']:<7} "
              f"verify={e['success']:<11} recover={e['recovery']}")


def _run(args) -> None:
    from morrow import run_skill
    from .sim import onboard, randomize, SimPerceiver, SimRobot
    skill = onboard(args.kind, args.kind)
    w = randomize(args.kind, np.random.RandomState(args.seed))
    r = run_skill(skill, SimRobot(w), SimPerceiver(w), seed=args.seed)
    print(f"{r.final_state}  success={r.success}  first_attempt={r.first_attempt_success}  "
          f"retries={r.retries}  recoveries={r.recoveries}")
    for ev in r.timeline:
        print("  ", ev)


def _eval(args) -> None:
    from .eval import format_report, run_benchmark
    print(format_report(run_benchmark(n=args.n, stress_mode=args.stress)))


def _save(args) -> None:
    from .serialize import save_skill
    from .sim import onboard
    skill = onboard(args.kind, args.kind)
    save_skill(skill, args.path)
    print(f"saved skill {skill.sku_id} (hash {skill.version_hash}) -> {args.path}")


def _demo(args) -> None:
    from .dashboard import serve
    serve(host=args.host, port=args.port, benchmark_n=args.n)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="morrow", description="demonstration -> verified skill (mk3)")
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("onboard", help="compile a skill from demonstrations and print it")
    o.add_argument("kind", choices=["box", "cylinder", "pouch"])
    o.add_argument("--demos", type=int, default=2)
    o.set_defaults(fn=_onboard)

    r = sub.add_parser("run", help="run one randomized packing cycle")
    r.add_argument("kind", choices=["box", "cylinder", "pouch"])
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=_run)

    e = sub.add_parser("eval", help="run the frozen benchmark")
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--stress", action="store_true", help="rotated carton + low-confidence frames")
    e.set_defaults(fn=_eval)

    sv = sub.add_parser("save", help="onboard a skill and write it to JSON")
    sv.add_argument("kind", choices=["box", "cylinder", "pouch"])
    sv.add_argument("path")
    sv.set_defaults(fn=_save)

    d = sub.add_parser("demo", help="serve the localhost dashboard")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8000)
    d.add_argument("--n", type=int, default=60)
    d.set_defaults(fn=_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
