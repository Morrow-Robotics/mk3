from morrow.eval import run_benchmark


def test_benchmark_is_reproducible():
    a = run_benchmark(n=20)
    b = run_benchmark(n=20)
    assert a == b


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
