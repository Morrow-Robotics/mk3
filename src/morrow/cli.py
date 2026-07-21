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


def _cell(args) -> None:
    try:
        from .physics.webview import (find_clips, render_physics_page, runtime_info,
                                      serve_physics, _slots)
        from .physics.showcase import (build_arm_showcase, build_multipack_showcase,
                                        build_showcase, build_watch_showcase)
    except ImportError:
        print("physics cell needs MuJoCo: pip install -e '.[physics]'")
        return
    if args.shot:
        arm = None if args.no_arm else build_arm_showcase()
        multipack = build_multipack_showcase()
        watch = None
        if not args.no_arm:
            try:
                watch = build_watch_showcase()
            except Exception as e:
                print(f"watch panel skipped: {e}")
        html = render_physics_page(build_showcase(), _slots(args.videos, embed=True),
                                   runtime_info(), arm, watch, multipack)
        with open(args.shot, "w") as f:
            f.write(html)
        print(f"wrote physics dashboard -> {args.shot} ({len(html)} bytes); "
              f"clips found in {args.videos}/: {len(find_clips(args.videos))}")
        return
    serve_physics(host=args.host, port=args.port, videos_dir=args.videos)


def _annotate(args) -> None:
    import json
    try:
        from .physics.annotate import capture_annotation, run_annotation
    except ImportError:
        print("annotate needs MuJoCo: pip install -e '.[physics]'")
        return
    with open(args.json) as f:
        ann = json.load(f)
    if args.shot:
        from .physics.film import encode_mp4
        frames, result, world = capture_annotation(ann, seed=args.seed)
        encode_mp4(frames, args.shot, fps=18)
        print(f"annotation -> skill; physics run {result.final_state}; "
              f"wrote {args.shot} ({len(frames)} frames)")
        return
    skill, world, result = run_annotation(ann, seed=args.seed)
    size = tuple(round(v, 3) for v in (world.hx, world.hy, world.hz))
    print(f"annotation -> skill {skill.version_hash}  (kind {world.kind}, half-size {size} m)")
    print(f"physics run: {result.final_state}  success={result.success}")


def _watch(args) -> None:
    try:
        from .physics.showcase import WATCH_CLIP, _inside_carton
        from .physics.pattern import packing_profile
        from .physics.watch import (capture_watch_pack, have_sam2, render_overlay,
                                    segment_scene, watch_and_pack_arm)
    except ImportError:
        print("watch needs MuJoCo + SAM2: pip install -e '.[physics]' and set $MORROW_SAM2_CKPT")
        return
    if not have_sam2():
        print("SAM2 weights not found — set $MORROW_SAM2_CKPT to a sam2.1 checkpoint. "
              "The watch pipeline runs real segmentation, so it needs real weights.")
        return
    c = dict(WATCH_CLIP)
    for k, v in (("clip", args.clip), ("frame_idx", args.frame), ("scale_m_per_px", args.scale),
                 ("kind", args.kind), ("height_m", args.height)):
        if v is not None:
            c[k] = v
    if args.carton:
        c["carton_box_frac"] = tuple(float(x) for x in args.carton.split(","))
    if args.product:
        c["product_box_frac"] = tuple(float(x) for x in args.product.split(","))

    scene = segment_scene(c["clip"], c["frame_idx"], c["carton_box_frac"], c["product_box_frac"])
    W, H = scene.image_wh
    cbb = scene.carton_bbox_px
    prof = packing_profile(c["clip"], (cbb[0] / W, cbb[1] / H, cbb[2] / W, cbb[3] / H))
    print(f"watched {scene.clip} @ frame {scene.frame_idx}: "
          f"SAM2 carton {scene.carton_score:.2f} · product {scene.product_score:.2f}")
    print(f"packing activity: ~{prof.n_events} candidate events "
          f"(confidence {prof.confidence_label} {prof.confidence}), "
          f"active {prof.active_fraction:.0%} of {prof.duration_s}s — operator-confirmable")
    if args.shot:
        from .physics.film import encode_mp4
        render_overlay(c["clip"], scene, args.shot + "_overlay.png")
        frames, _a, size, _c, result, world = capture_watch_pack(
            scene, c["scale_m_per_px"], c["kind"], c["height_m"])
        encode_mp4(frames, args.shot + "_pack.mp4", fps=18)
        print(f"SO-101 pack: {result.final_state}  inside={_inside_carton(world)}  "
              f"-> {args.shot}_overlay.png, {args.shot}_pack.mp4")
    else:
        _a, size, _c, result, world = watch_and_pack_arm(
            scene, c["scale_m_per_px"], c["kind"], c["height_m"])
        print(f"SO-101 pack: {result.final_state}  success={result.success}  "
              f"inside={_inside_carton(world)}  product half-size "
              f"{tuple(round(v, 3) for v in size)} m")


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

    wt = sub.add_parser("watch", help="watch a real clip with SAM2 → SO-101 physics pack")
    wt.add_argument("--clip", help="path to a customer clip (default: the showcase clip)")
    wt.add_argument("--frame", type=int, help="frame index to segment")
    wt.add_argument("--scale", type=float, help="table scale, metres per pixel")
    wt.add_argument("--kind", choices=["box", "cylinder"])
    wt.add_argument("--height", type=float, help="product height in metres")
    wt.add_argument("--carton", help="carton seed box, fractional x0,y0,x1,y1")
    wt.add_argument("--product", help="product seed box, fractional x0,y0,x1,y1")
    wt.add_argument("--shot", help="render <shot>_overlay.png and <shot>_pack.mp4")
    wt.set_defaults(fn=_watch)

    an = sub.add_parser("annotate", help="build a physics skill from a marked-up clip (JSON)")
    an.add_argument("json", help="annotation JSON file (see examples/annotation_box.json)")
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--shot", help="render the resulting physics pack to this mp4")
    an.set_defaults(fn=_annotate)

    cl = sub.add_parser("cell", help="physics dashboard: customer video -> LeRobot MuJoCo sim")
    cl.add_argument("--host", default="127.0.0.1")
    cl.add_argument("--port", type=int, default=8001)
    cl.add_argument("--videos", default="videos", help="folder of drop-in customer clips")
    cl.add_argument("--shot", help="render the dashboard to an HTML file and exit")
    cl.add_argument("--no-arm", action="store_true",
                    help="skip the (slower) SO-101 model arm render in --shot")
    cl.set_defaults(fn=_cell)

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
