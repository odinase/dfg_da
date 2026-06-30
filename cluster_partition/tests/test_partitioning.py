"""
test_partitioning.py
====================

Comprehensive verification of the cluster-partitioning / inclusion-exclusion
marginalization package against:

* the brute-force full-graph exact reference (ground truth),
* the EHM2 baseline (``pyehm``) for single-cluster marginals,
* the Bonferroni structural property of the first-stage approximation.

The test cases include the thesis Figure 7.7 toy problem, the upstream
``cluster_bayes_tree.test_case`` 5x7 problem, a three-cluster problem (so the
inclusion-exclusion weight ``1 - K`` takes the value ``-2``), and a problem with
two linking measurements (so multilinearity across delegating variables is
exercised).
"""
import numpy as np
import pytest

from cluster_partition import (
    ClusterLinks,
    ConditionalSuperclusterMarginals,
    EHM2Solver,
    ExactEnumerationSolver,
    LBPBetheSolver,
    MODES,
    MulticlusterPartitionedMarginals,
    NONE_EVENT,
    make_hypotheses,
    multicluster_exact_reference,
)

ATOL = 1e-9


# ---------------------------------------------------------------------------
# Test fixtures: each builds FRESH hypotheses (reindex_tracks mutates in place).
# ---------------------------------------------------------------------------
def case_fig77():
    """Thesis Fig. 7.7: two single-track clusters sharing one measurement."""
    R = np.array([[0.0, 1.3], [0.0, 0.7]])
    mk = lambda: [make_hypotheses([([1], 1.0)]), make_hypotheses([([2], 1.0)])]
    return R, mk


def case_upstream():
    """Upstream cluster_bayes_tree.test_case (5 tracks, 6 meas, 2 clusters)."""
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ])
    mk = lambda: [
        make_hypotheses([([1, 2], 0.5), ([1, 3], 0.5)]),
        make_hypotheses([([4], 0.5), ([5], 0.5)]),
    ]
    return R, mk


def case_three_clusters():
    """Three single-track clusters all gating measurement 1 (K = 3)."""
    R = np.array([
        [0.0, 1.1, -np.inf, -np.inf],
        [0.0, 0.6,    0.3 , -np.inf],
        [0.0, 0.9, -np.inf,    0.4 ],
    ])
    mk = lambda: [
        make_hypotheses([([1], 1.0)]),
        make_hypotheses([([2], 1.0)]),
        make_hypotheses([([3], 1.0)]),
    ]
    return R, mk


def case_two_linking():
    """Two clusters coupled by TWO linking measurements; cluster 0 multi-hyp."""
    R = np.array([
        [0.0,  0.8,  0.5, -np.inf, -np.inf],
        [0.0, -0.3, -np.inf,  0.7 , -np.inf],
        [0.0,  0.4,  0.9, -np.inf,    0.2 ],
    ])
    mk = lambda: [
        make_hypotheses([([1, 2], 0.5), ([1], 0.5)]),
        make_hypotheses([([3], 1.0)]),
    ]
    return R, mk


ALL_CASES = {
    "fig77": case_fig77,
    "upstream": case_upstream,
    "three_clusters": case_three_clusters,
    "two_linking": case_two_linking,
}


# ---------------------------------------------------------------------------
# 1. ClusterLinks: discovery of merging clusters and linking measurements.
# ---------------------------------------------------------------------------
def test_cluster_links_upstream_structure():
    R, mk = case_upstream()
    cl = ClusterLinks(R, mk())
    # Clusters 0 and 1 merge through the single shared measurement (column 1).
    assert cl.merging_clusters == [{0, 1}]
    assert cl.unmerging_clusters() == set()
    lm = cl.linking_mappings_per_merging_clusters()[0]
    assert lm.linking_measurements_to_clusters == {1: {0, 1}}
    assert lm.cluster_to_linking_measurements == {0: {1}, 1: {1}}


def test_cluster_links_three_clusters_K3():
    R, mk = case_three_clusters()
    cl = ClusterLinks(R, mk())
    assert cl.merging_clusters == [{0, 1, 2}]
    lm = cl.linking_mappings_per_merging_clusters()[0]
    assert lm.linking_measurements_to_clusters == {1: {0, 1, 2}}


def test_cluster_links_two_linking_measurements():
    R, mk = case_two_linking()
    cl = ClusterLinks(R, mk())
    assert cl.merging_clusters == [{0, 1}]
    lm = cl.linking_mappings_per_merging_clusters()[0]
    # Measurements 1 and 2 are both gated by tracks of both clusters.
    assert lm.linking_measurements_to_clusters == {1: {0, 1}, 2: {0, 1}}


def test_cluster_links_no_merging():
    """Two clusters with disjoint measurement support -> no superclusters."""
    R = np.array([
        [0.0,  1.0, -np.inf, -np.inf],
        [0.0, -np.inf, 0.5, -np.inf],
        [0.0, -np.inf, -np.inf, 0.8 ],
    ])
    mk = lambda: [make_hypotheses([([1], 1.0)]),
                  make_hypotheses([([2], 1.0), ([3], 1.0)])]
    cl = ClusterLinks(R, mk())
    assert cl.merging_clusters == []
    assert cl.unmerging_clusters() == {0, 1}


# ---------------------------------------------------------------------------
# 2. Delegating-variable event space: disjoint = {{0}} u {T^c}, enumeration
#    columns = [-1, *clusters].
# ---------------------------------------------------------------------------
def test_delegating_event_space_disjoint():
    R, mk = case_upstream()
    cl = ClusterLinks(R, mk())
    events = cl.delegating_variable_event_space(1)  # measurement 1
    # {0} block, then the FULL track-set T^c of each gating cluster (Eq. (7.5)).
    assert events[0] == frozenset({0})
    assert frozenset({1, 2, 3}) in events  # cluster 0's track set T^0
    assert frozenset({4, 5}) in events      # cluster 1's track set T^1


def test_enumeration_columns_are_none_then_clusters():
    R, mk = case_three_clusters()
    cl = ClusterLinks(R, mk())
    lm = cl.linking_mappings_per_merging_clusters()[0]
    sc = ConditionalSuperclusterMarginals(R, mk(), lm, ExactEnumerationSolver(),
                                          "overlap_ie")
    cols = sc._delegating_event_columns()
    assert len(cols) == 1                       # one linking measurement
    np.testing.assert_array_equal(cols[0], np.array([NONE_EVENT, 0, 1, 2]))


# ---------------------------------------------------------------------------
# 3. Inclusion-exclusion weights reproduce the thesis Fig. 7.7 formula
#    omega(c) = +1, omega(none) = 1 - K.
# ---------------------------------------------------------------------------
def test_ie_weights_fig77():
    R, mk = case_fig77()
    cl = ClusterLinks(R, mk())
    lm = cl.linking_mappings_per_merging_clusters()[0]
    sc = ConditionalSuperclusterMarginals(R, mk(), lm, ExactEnumerationSolver(),
                                          "overlap_ie")
    weights = sc._ie_weights()
    assert len(weights) == 1
    w = weights[0]
    # K = 2 gating clusters -> none weight = 1 - 2 = -1, each cluster weight +1.
    assert w[NONE_EVENT] == -1
    assert w[0] == 1 and w[1] == 1


def test_ie_weights_three_clusters():
    R, mk = case_three_clusters()
    cl = ClusterLinks(R, mk())
    lm = cl.linking_mappings_per_merging_clusters()[0]
    sc = ConditionalSuperclusterMarginals(R, mk(), lm, ExactEnumerationSolver(),
                                          "overlap_ie")
    w = sc._ie_weights()[0]
    # K = 3 -> none weight = 1 - 3 = -2.
    assert w[NONE_EVENT] == -2
    assert w[0] == w[1] == w[2] == 1


def test_fig77_inclusion_exclusion_identity():
    """Z_exact = P({0,1}) + P({0,2}) - P({0}) for the Fig. 7.7 problem."""
    R, mk = case_fig77()
    Z_exact = multicluster_exact_reference(R, mk())[2]
    first = MulticlusterPartitionedMarginals(
        R, mk(), ExactEnumerationSolver(), mode="overlap_firststage"
    ).compute_marginals_likelihood()[2]
    ie = MulticlusterPartitionedMarginals(
        R, mk(), ExactEnumerationSolver(), mode="overlap_ie"
    ).compute_marginals_likelihood()[2]
    # First stage sums P({0}) + sum_c P({0} u T^c) over the K+1 enumeration
    # tuples; the exact union is sum_c P({0} u T^c) + (1 - K) P({0}).  Hence
    # first_stage - exact = K * P({0}).  Here K = 2 and P({0}) = 1*1 = 1.
    assert first == pytest.approx(Z_exact + 2.0, abs=1e-9)
    assert ie == pytest.approx(Z_exact, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Exactness: disjoint_exact == overlap_ie == reference, on every case,
#    for BOTH marginals and likelihood, with the exact inner solver.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_CASES))
@pytest.mark.parametrize("mode", ["disjoint_exact", "overlap_ie"])
def test_exact_modes_match_reference(name, mode):
    R, mk = ALL_CASES[name]()
    ref_marg, ref_theta, ref_Z = multicluster_exact_reference(R, mk())
    out_marg, out_theta, out_Z = MulticlusterPartitionedMarginals(
        R, mk(), ExactEnumerationSolver(), mode=mode
    ).compute_marginals_likelihood()
    assert out_Z == pytest.approx(ref_Z, rel=1e-9)
    np.testing.assert_allclose(out_marg, ref_marg, atol=ATOL)
    for c in ref_theta:
        np.testing.assert_allclose(out_theta[c], ref_theta[c], atol=ATOL)


# ---------------------------------------------------------------------------
# 5. Bonferroni: the first-stage approximation OVER-estimates the constant
#    (it uses an odd number of inclusion-exclusion stages).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_CASES))
def test_firststage_overestimates_constant(name):
    R, mk = ALL_CASES[name]()
    ref_Z = multicluster_exact_reference(R, mk())[2]
    first_Z = MulticlusterPartitionedMarginals(
        R, mk(), ExactEnumerationSolver(), mode="overlap_firststage"
    ).compute_marginals_likelihood()[2]
    assert first_Z > ref_Z - 1e-9            # strictly >= exact (Bonferroni)


# ---------------------------------------------------------------------------
# 6. EHM2 comparison: EHM2Solver single-cluster marginals == ExactEnumeration.
# ---------------------------------------------------------------------------
def _single_cluster_inputs():
    R = np.array([
        [0.0,  1.3, -0.6, -np.inf],
        [0.0, -0.4,  0.9,    0.2 ],
        [0.0, -np.inf, 0.5,  1.1 ],
    ])
    mk = lambda: make_hypotheses([([1, 2, 3], 0.6), ([1, 2], 0.4)])
    return R, mk


def test_ehm2_matches_exact_single_cluster():
    pytest.importorskip("pyehm")
    R, mk = _single_cluster_inputs()
    ex_marg, ex_theta, ex_Z = ExactEnumerationSolver()(R, mk())
    eh_marg, eh_theta, eh_Z = EHM2Solver()(R, mk())
    np.testing.assert_allclose(eh_marg, ex_marg, atol=1e-9)
    np.testing.assert_allclose(eh_theta, ex_theta, atol=1e-9)
    assert eh_Z == pytest.approx(ex_Z, rel=1e-9)


@pytest.mark.parametrize("name", list(ALL_CASES))
def test_ehm2_as_inner_solver_matches_reference(name):
    """The whole partitioned pipeline with EHM2 inner solver stays exact."""
    pytest.importorskip("pyehm")
    R, mk = ALL_CASES[name]()
    ref_marg, _, ref_Z = multicluster_exact_reference(R, mk())
    out_marg, _, out_Z = MulticlusterPartitionedMarginals(
        R, mk(), EHM2Solver(), mode="overlap_ie"
    ).compute_marginals_likelihood()
    assert out_Z == pytest.approx(ref_Z, rel=1e-9)
    np.testing.assert_allclose(out_marg, ref_marg, atol=1e-7)


# ---------------------------------------------------------------------------
# 7. LBP inner solver: the Bethe constant is sane, and the inclusion-exclusion
#    correction DE-BIASES the first-stage likelihood (moves it toward exact).
# ---------------------------------------------------------------------------
def test_lbp_bethe_constant_underestimates_loopy():
    """On a loopy single cluster the Bethe constant underestimates the truth."""
    R, mk = _single_cluster_inputs()
    ex_Z = ExactEnumerationSolver()(R, mk())[2]
    lbp_Z = LBPBetheSolver()(R, mk())[2]
    assert 0 < lbp_Z <= ex_Z + 1e-9          # Bethe lower bound (Sec. 2.3.3)
    assert lbp_Z > 0.5 * ex_Z                 # and it is a *good* lower bound


def test_lbp_bethe_constant_exact_on_tree():
    """Two tracks, one measurement -> association graph is a tree -> exact."""
    psi1, psi2 = 2.0, 3.0
    R = np.array([[0.0, np.log(psi1)], [0.0, np.log(psi2)]])
    mk = lambda: make_hypotheses([([1, 2], 1.0)])
    lbp_Z = LBPBetheSolver()(R, mk())[2]
    assert lbp_Z == pytest.approx(1.0 + psi1 + psi2, abs=1e-9)


@pytest.mark.parametrize("name", list(ALL_CASES))
def test_lbp_ie_debiases_firststage(name):
    R, mk = ALL_CASES[name]()
    ref_Z = multicluster_exact_reference(R, mk())[2]
    first_Z = MulticlusterPartitionedMarginals(
        R, mk(), LBPBetheSolver(), mode="overlap_firststage"
    ).compute_marginals_likelihood()[2]
    ie_Z = MulticlusterPartitionedMarginals(
        R, mk(), LBPBetheSolver(), mode="overlap_ie"
    ).compute_marginals_likelihood()[2]
    # The IE-corrected constant is at least as close to the truth as first-stage.
    assert abs(ie_Z - ref_Z) <= abs(first_Z - ref_Z) + 1e-9


# ---------------------------------------------------------------------------
# 8. Dynamic-programming cache: ConditionedCluster reuses conditioned sub-problems.
# ---------------------------------------------------------------------------
def test_conditioned_cluster_caches_by_owned_mask():
    R, mk = case_two_linking()
    cl = ClusterLinks(R, mk())
    lm = cl.linking_mappings_per_merging_clusters()[0]
    sc = ConditionalSuperclusterMarginals(R, mk(), lm, ExactEnumerationSolver(),
                                          "overlap_ie")
    sc.compute_marginals_likelihood()
    # Cluster 0 owns 2 linking measurements -> at most 2^2 = 4 distinct sub-problems.
    for cc in sc.conditioned_clusters:
        n_owned = len(cc.actual_meas)
        assert len(cc.cache) <= 2 ** n_owned


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
