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
    journal = None
    if args.log:
        from .journal import EpisodeLog
        journal = EpisodeLog(args.log)
    print(format_report(run_benchmark(n=args.n, stress_mode=args.stress, journal=journal)))
    if args.breakdown:
        from .eval import failure_breakdown
        print("where the box stress batch gets stuck:")
        for reason, count in failure_breakdown("box", n=args.n).items():
            print(f"  {count:4d}  {reason}")
    if journal is not None:
        print(f"logged {len(journal)} episodes -> {args.log}")


def _ranker(args) -> None:
    from .eval import compare_ranker
    r = compare_ranker(args.kind, n=args.n, n_train=args.n)
    print(f"[{args.kind}] structured SKU (seal depends on grasp yaw)")
    print(f"  analytic first-attempt : {r['analytic_first_attempt']*100:5.1f}%")
    print(f"  + learned ranker       : {r['ranker_first_attempt']*100:5.1f}%  "
          f"(+{(r['ranker_first_attempt']-r['analytic_first_attempt'])*100:.1f})")


def _pack(args) -> None:
    from .sequence import demo_pack_sequence
    r = demo_pack_sequence(seed=args.seed)
    print(f"packed {r.packed}/{r.total} into one carton  (success={r.success})")
    for it in r.results:
        tail = "" if it.failure_reason is None else f"  [{it.failure_reason}]"
        print(f"  {it.sku:9s} {it.kind:9s} slot {it.slot} -> {it.final_state}{tail}")


def _save(args) -> None:
    from .serialize import save_skill
    from .sim import onboard
    skill = onboard(args.kind, args.kind)
    save_skill(skill, args.path)
    print(f"saved skill {skill.sku_id} (hash {skill.version_hash}) -> {args.path}")


def _demo(args) -> None:
    if args.shot:
        from .dashboard import render_page, runtime_info
        from .pipeline import investor_sequence
        html = render_page(investor_sequence(benchmark_n=args.n), runtime_info())
        with open(args.shot, "w") as f:
            f.write(html)
        print(f"wrote dashboard -> {args.shot} ({len(html)} bytes)")
        return
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
    e.add_argument("--breakdown", action="store_true", help="tally where the stress batch gets stuck")
    e.add_argument("--log", help="append per-run episode records to this JSONL file")
    e.set_defaults(fn=_eval)

    rk = sub.add_parser("ranker", help="A/B the learned grasp ranker on a structured SKU")
    rk.add_argument("kind", choices=["box", "cylinder", "pouch"])
    rk.add_argument("--n", type=int, default=80)
    rk.set_defaults(fn=_ranker)

    pk = sub.add_parser("pack", help="pack one of each SKU into a single carton (high-mix)")
    pk.add_argument("--seed", type=int, default=0)
    pk.set_defaults(fn=_pack)

    sv = sub.add_parser("save", help="onboard a skill and write it to JSON")
    sv.add_argument("kind", choices=["box", "cylinder", "pouch"])
    sv.add_argument("path")
    sv.set_defaults(fn=_save)

    d = sub.add_parser("demo", help="serve the localhost dashboard (or --shot to a file)")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8000)
    d.add_argument("--n", type=int, default=60)
    d.add_argument("--shot", help="render the dashboard to this HTML file and exit")
    d.set_defaults(fn=_demo)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
