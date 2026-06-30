"""Structural verification of the CM prior/likelihood pipeline at k=1.

Requires ``../scenario1MC.mat`` (skipped if absent).
"""

import os

import numpy as np

MAT = os.path.join(os.path.dirname(__file__), "..", "..", "scenario1MC.mat")


def _run_k1():
    from cm import (
        InCol,
        compute_pre_cluster_threshold,
        initial_state,
        load_cm_scenario,
        run_prior_likelihood,
    )
    from cm.dump import DUMP_VARS

    scn = load_cm_scenario(MAT, pd_value=0.9)
    state = initial_state(InCol())
    config = {
        "pD": scn.pD, "gamma_gate": 9.0, "do_mechi": True,
        "do_semi_two_point": True, "adole_thres": 1.0,
        "nHypoMax": 30, "nHypoTotalMax": 150,
        "pre_cluster_threshold": compute_pre_cluster_threshold(scn.system, scn.params, scn.pD),
        "label_gen": 0,
    }
    dump, _ = run_prior_likelihood(state, scn, 1, config)
    return dump, DUMP_VARS, scn


def test_k1_has_all_variables():
    if not os.path.exists(MAT):
        print("SKIP (scenario1MC.mat not found)")
        return
    dump, dump_vars, _ = _run_k1()
    for v in dump_vars:
        assert v in dump, f"missing {v}"


def test_k1_shapes():
    if not os.path.exists(MAT):
        print("SKIP (scenario1MC.mat not found)")
        return
    dump, _, scn = _run_k1()
    m = scn.measurements_at(1).shape[1]
    assert dump["trackFile"].shape == (21, 0)
    assert dump["trackFileShadow"].shape == (21, 0, 8)
    assert dump["predX"].shape == (4, 0)
    assert dump["predP"].shape == (4, 4, 0)
    assert dump["predS"].shape == (2, 2, 0)
    assert dump["measurements"].shape == (2, m)
    assert dump["gainMatPostC"].shape == (m, m)
    assert dump["trackNumberLookup"].shape == (m, m)
    assert dump["meaHistColNew"].shape == (8, m)
    assert dump["hypos"].shape == (1, 0)
    assert dump["k"] == 1.0 and dump["pD"] == 0.9
    # k=1 specifics: each measurement spawns its own tentative track.
    assert list(np.asarray(dump["indicesOfNewbornTracks"]).ravel()) == list(range(1, m + 1))
    assert list(np.diag(np.asarray(dump["trackNumberLookup"]))) == list(range(1, m + 1))
    assert list(np.asarray(dump["meaHistColNew"])[-1, :]) == list(range(1, m + 1))
    assert np.allclose(np.diag(np.asarray(dump["gainMatPostC"])), 0.0)


def _run_through_k2():
    from cm import (
        InCol,
        compute_pre_cluster_threshold,
        initial_state,
        load_cm_scenario,
        run_prior_likelihood,
    )
    from cm.carryover import carry_over

    scn = load_cm_scenario(MAT, pd_value=0.9)
    state = initial_state(InCol())
    config = {
        "pD": scn.pD, "gamma_gate": 9.0, "do_mechi": True,
        "do_semi_two_point": True, "adole_thres": 1.0,
        "nHypoMax": 30, "nHypoTotalMax": 150,
        "pre_cluster_threshold": compute_pre_cluster_threshold(scn.system, scn.params, scn.pD),
        "label_gen": 0,
    }
    dump1, extra1 = run_prior_likelihood(state, scn, 1, config)
    carry_over(state, dump1, extra1)          # step-1 carry-over (no BB)
    dump2, extra2 = run_prior_likelihood(state, scn, 2, config)
    return dump1, dump2, extra2, scn


def test_k2_carryover_and_shapes():
    if not os.path.exists(MAT):
        print("SKIP (scenario1MC.mat not found)")
        return
    dump1, dump2, extra2, scn = _run_through_k2()
    m1 = scn.measurements_at(1).shape[1]
    m2 = scn.measurements_at(2).shape[1]
    # The m1 step-1 tracks are carried into step 2 as singleton clusters.
    assert dump2["trackFile"].shape == (21, m1)
    assert dump2["predX"].shape == (4, m1)
    assert dump2["predP"].shape == (4, 4, m1)
    assert dump2["measurements"].shape == (2, m2)
    assert dump2["gainMatPostC"].shape == (m1 + m2, m2 + m1)
    assert list(np.asarray(dump2["hypos"]).ravel().astype(int)) == list(range(1, m1 + 1))
    assert list(np.asarray(dump2["clusters"]).ravel().astype(int)) == list(range(1, m1 + 1))
    assert np.all(np.asarray(dump2["clustersCard"]).ravel() == 1)
    assert dump2["k"] == 2.0
    # Step-2 carry-over needs branch-and-bound (Stage 2b).
    assert extra2["masters"].size > 0


def _run_through_k3():
    from cm import (
        InCol,
        compute_pre_cluster_threshold,
        initial_state,
        load_cm_scenario,
        run_prior_likelihood,
    )
    from cm.carryover import carry_over

    scn = load_cm_scenario(MAT, pd_value=0.9)
    state = initial_state(InCol())
    config = {
        "pD": scn.pD, "gamma_gate": 9.0, "do_mechi": True,
        "do_semi_two_point": True, "adole_thres": 1.0,
        "nHypoMax": 30, "nHypoTotalMax": 150,
        "pre_cluster_threshold": compute_pre_cluster_threshold(scn.system, scn.params, scn.pD),
        "label_gen": 0, "sig": 0.05, "splitting_threshold": 0.018,
        "maha_cs_thres": 3, "non_split_lag": 2,
    }
    dump1, extra1 = run_prior_likelihood(state, scn, 1, config)
    carry_over(state, dump1, extra1, config)   # step-1 carry-over (no BB)
    dump2, extra2 = run_prior_likelihood(state, scn, 2, config)
    carry_over(state, dump2, extra2, config)   # step-2 carry-over (full Stage 2b)
    dump3, extra3 = run_prior_likelihood(state, scn, 3, config)
    return dump3, scn


def test_k3_carryover_and_shapes():
    """The step-2 carry-over runs the full Stage 2b pipeline (branch-and-bound,
    pruning, n-scan merge, cluster splitting, recycling); step 3 then consumes
    that state. Check the dump's ABC invariants hold."""
    if not os.path.exists(MAT):
        print("SKIP (scenario1MC.mat not found)")
        return
    dump3, scn = _run_through_k3()
    hypos = np.asarray(dump3["hypos"]).ravel()
    hypos_card = np.asarray(dump3["hyposCard"]).ravel().astype(int)
    clusters = np.asarray(dump3["clusters"]).ravel()
    clusters_card = np.asarray(dump3["clustersCard"]).ravel().astype(int)
    assert dump3["k"] == 3.0
    # ABC bookkeeping invariants.
    assert hypos.size == int(hypos_card.sum())
    assert clusters.size == int(clusters_card.sum())
    assert clusters.size == hypos_card.size
    # Hypotheses are capped at nHypoTotalMax per supercluster.
    assert hypos_card.size <= 150
    # trackFile has 21 rows; every value finite.
    assert dump3["trackFile"].shape[0] == 21
    assert np.all(np.isfinite(np.asarray(dump3["probLogHypos"])[np.isfinite(np.asarray(dump3["probLogHypos"]))]))


if __name__ == "__main__":
    import sys
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
