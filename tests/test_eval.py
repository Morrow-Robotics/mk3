from morrow.eval import run_benchmark


def test_benchmark_is_reproducible():
    a = run_benchmark(n=20)
    b = run_benchmark(n=20)
    assert a == b


def test_benchmark_report_carries_no_wallclock():
    # A reproducible artifact must not embed wall-clock timing (it made the
    # report flaky under CPU load). Onboarding keeps only deterministic fields.
    rep = run_benchmark(n=10)
    for r in rep["kinds"].values():
        assert set(r["onboarding"]) == {"n_demos", "code_changes"}


def test_candidate_seed_is_stable_not_salted():
    # A fixed golden value guarantees _seed never regresses to Python's
    # per-process salted string/tuple hashing (which breaks cross-process runs).
    from morrow.candidates import _seed
    from morrow.skill import SkillState
    assert _seed(42, (SkillState.READY, SkillState.APPROACHED), 0) == 42_000_126
    assert _seed(42, (SkillState.APPROACHED, SkillState.GRASPED), 0) == 42_000_257


def test_morrow_beats_open_loop_replay():
    rep = run_benchmark(n=40)
    for kind, r in rep["kinds"].items():
        morrow = r["morrow"]["final_success_rate"]
        replay = r["baseline_open_loop_success_rate"]
        assert morrow > 0.9, kind
        assert replay < morrow, kind  # perception + verification is the whole point


def test_onboarding_writes_no_code():
    rep = run_benchmark(n=10)
    for r in rep["kinds"].values():
        assert r["onboarding"]["code_changes"] == 0
