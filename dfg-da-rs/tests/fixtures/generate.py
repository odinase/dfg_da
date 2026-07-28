#!/usr/bin/env python3
"""Generate shared golden fixtures for the multicluster inclusion-exclusion solver.

Each fixture stores the inputs (R_LC with -inf encoded as JSON null, prior
hypotheses as {tracks, log_weight}, and assocLocal) together with the expected
outputs computed by the reference Python implementation
(`MulticlusterEfficientMarginalsLBPInclusionExclusion` + `LBPMarginalsByTotalProbBethe`).

Both the Python parity test (dfg_da/tests/test_multicluster_ie_parity.py) and the
Rust parity test (dfg-da-rs/tests/multicluster_ie_parity.rs) load these files,
re-run their own solver, and assert the outputs match the stored expected values,
so the two implementations are checked against one canonical result.

Run from the repo root:  .venv/bin/python dfg-da-rs/tests/fixtures/generate.py
"""
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import py_dfg_da as pdd

from dfg_da.cluster_bayes_tree_extra_term import edmund_to_lc
from dfg_da.marginals_computers import LBPMarginalsByTotalProbBethe
from dfg_da.cluster_conditioning_lbp_ie import (
    MulticlusterEfficientMarginalsLBPInclusionExclusion,
)

FIXTURE_DIR = Path(__file__).resolve().parent

NINF = -math.inf


def cl(*clusters):
    """A cluster's hypotheses: cl([tracks, prob], ...) -> [{tracks, log_weight}]."""
    return [{"tracks": list(tracks), "log_weight": math.log(prob)} for tracks, prob in clusters]


# --- Fixture definitions (inputs only; outputs are computed below) -------------

def test_case3_inputs():
    # 3 clusters that all merge into a single supercluster (the existing case).
    R = np.array([
        [    3.0, NINF, NINF, NINF,   -0.60, NINF, NINF, NINF, NINF, NINF, NINF],
        [    3.2, NINF, NINF, NINF, NINF,   -0.56, NINF, NINF, NINF, NINF, NINF],
        [   -3.0,     2.0,     1.2, NINF, NINF, NINF,   -0.46, NINF, NINF, NINF, NINF],
        [NINF, NINF,     3.0, NINF, NINF, NINF, NINF,   -0.62, NINF, NINF, NINF],
        [NINF,    -0.4,    -1.8, NINF, NINF, NINF, NINF, NINF,   -0.55, NINF, NINF],
        [NINF,     0.5, NINF,    -0.1, NINF, NINF, NINF, NINF, NINF,   -0.62, NINF],
        [NINF, NINF, NINF,     0.8, NINF, NINF, NINF, NINF, NINF, NINF,   -0.55],
    ], order='F')
    R_LC = edmund_to_lc(R)
    priors = [
        cl(([1, 2], 0.5), ([1, 3], 0.5)),
        cl(([4], 0.5), ([5], 0.5)),
        cl(([6, 7], 0.2), ([6], 0.8)),
    ]
    assoc_local = [[1, 1, 1], [1, 0, 0]]
    return R_LC, priors, assoc_local


def two_merge_one_separate_inputs():
    # Clusters 0 and 1 share measurement 2 (a linking measurement) so they merge;
    # cluster 2 is independent and goes through the unmerging path.
    R_LC = np.array([
        [0.0, 1.0, NINF, NINF, NINF],   # track 1 (c0) -> meas 1
        [0.0, NINF, 0.8, NINF, NINF],   # track 2 (c0) -> meas 2  \ linking
        [0.0, NINF, 0.9, NINF, NINF],   # track 3 (c1) -> meas 2  /
        [0.0, NINF, NINF, 1.2, NINF],   # track 4 (c1) -> meas 3
        [0.0, NINF, NINF, NINF, 0.5],   # track 5 (c2) -> meas 4
    ])
    priors = [
        cl(([1, 2], 0.6), ([1], 0.4)),
        cl(([3, 4], 0.7), ([3], 0.3)),
        cl(([5], 0.9), ([], 0.1)),
    ]
    assoc_local = [[1, 1, 2], [1, 0, 0]]
    return R_LC, priors, assoc_local


def all_separate_inputs():
    # Three clusters, none share a measurement -> no merging, all unmerging.
    R_LC = np.array([
        [0.0, 1.0, NINF, NINF],   # track 1 (c0) -> meas 1
        [0.0, NINF, 0.5, NINF],   # track 2 (c1) -> meas 2
        [0.0, NINF, NINF, 2.0],   # track 3 (c2) -> meas 3
    ])
    priors = [
        cl(([1], 0.8), ([], 0.2)),
        cl(([2], 1.0)),
        cl(([3], 0.6), ([], 0.4)),
    ]
    assoc_local = [[1, 2, 3], [1, 0, 0]]
    return R_LC, priors, assoc_local


def single_cluster_single_hypothesis_inputs():
    # Trivial edge case: one cluster, one hypothesis over two tracks.
    R_LC = np.array([
        [0.0, 1.0, NINF],
        [0.0, NINF, 1.0],
    ])
    priors = [cl(([1, 2], 1.0))]
    assoc_local = [[1], [1]]
    return R_LC, priors, assoc_local


def single_cluster_multi_hypothesis_empty_inputs():
    # Edge case: a single cluster whose hypotheses include the empty (nonexistence)
    # hypothesis, and measurements gated by both tracks.
    R_LC = np.array([
        [0.0, 0.7, 0.2],
        [0.0, 0.1, 0.9],
    ])
    priors = [cl(([1, 2], 0.5), ([1], 0.3), ([], 0.2))]
    assoc_local = [[1], [1]]
    return R_LC, priors, assoc_local


FIXTURES = {
    "test_case3": test_case3_inputs,
    "two_merge_one_separate": two_merge_one_separate_inputs,
    "all_separate": all_separate_inputs,
    "single_cluster_single_hypothesis": single_cluster_single_hypothesis_inputs,
    "single_cluster_multi_hypothesis_empty": single_cluster_multi_hypothesis_empty_inputs,
}


# --- Helpers ------------------------------------------------------------------

def build_hypotheses_list(priors):
    return pdd.hypothesis.HypothesesList([
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis(list(h["tracks"]), h["log_weight"]) for h in cluster
        ])
        for cluster in priors
    ])


def matrix_to_json(mat):
    # Encode -inf as JSON null; everything else as a plain float.
    return [[None if not math.isfinite(x) else float(x) for x in row] for row in np.asarray(mat)]


def run_reference(R_LC, priors, assoc_local):
    prior_list = build_hypotheses_list(deepcopy(priors))
    solver = MulticlusterEfficientMarginalsLBPInclusionExclusion(
        R_LC=R_LC.copy(),
        prior_hypotheses_per_cluster=prior_list,
        assocLocal=np.array(assoc_local).copy(),
        lbp_solver=LBPMarginalsByTotalProbBethe(),
    )
    out = solver.compute_marginals_likelihood()
    theta = [np.asarray(t).tolist() for t in out.theta_posteriors]
    return {
        "marginals": np.asarray(out.marginals).tolist(),
        "likelihood": float(out.likelihood),
        "theta_posteriors": theta,
    }


def main():
    for name, inputs_fn in FIXTURES.items():
        R_LC, priors, assoc_local = inputs_fn()
        expected = run_reference(R_LC, priors, assoc_local)
        fixture = {
            "name": name,
            "R_LC": matrix_to_json(R_LC),
            "prior_hypotheses_per_cluster": priors,
            "assoc_local": [[int(x) for x in row] for row in assoc_local],
            "expected": expected,
        }
        path = FIXTURE_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(fixture, f, indent=2)
        n_clusters = len(priors)
        print(f"wrote {path.name}: {len(expected['marginals'])} tracks, "
              f"{n_clusters} clusters, likelihood={expected['likelihood']:.6g}")


if __name__ == "__main__":
    main()
