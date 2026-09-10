import unittest

import networkx as nx
import numpy as np
import py_dfg_da as pdd

from dfg_da.graph_stats import (
    cluster_graph_stats,
    cyclomatic_number,
    cyclomatic_number_from_gate,
    merged_cluster_members,
    scan_multihypothesis_cyclomatic,
)


def nx_cyclomatic(gate):
    """Reference implementation: build the bipartite graph and count independent cycles.

    This is the original ``ravens_parser_parallell_multicluster.cyclomatic_number`` body,
    kept here as the oracle the fast kernel is checked against.
    """
    track_idx, meas_idx = np.nonzero(np.asarray(gate, dtype=bool))
    if track_idx.size == 0:
        return 0
    G = nx.Graph()
    G.add_edges_from(zip((f"t{i}" for i in track_idx), (f"m{j}" for j in meas_idx)))
    return int(G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))


def lc(rows):
    """(n, m+1) reward matrix in the lc layout from a list of measurement rows.

    ``None`` marks an ungated (``-inf``) entry; the misdetection column is always finite.
    """
    out = []
    for row in rows:
        out.append([0.0] + [(-np.inf if v is None else float(v)) for v in row])
    return np.asfortranarray(np.array(out))


def hypotheses(*specs):
    return pdd.hypothesis.Hypotheses([
        pdd.hypothesis.Hypothesis(list(tracks), float(np.log(p))) for tracks, p in specs])


class TestCyclomaticKernel(unittest.TestCase):
    def test_four_cycle(self):
        # Two tracks, both gated to the same two measurements: a single 4-cycle.
        gate = np.array([[True, True], [True, True]])
        self.assertEqual(cyclomatic_number_from_gate(gate), 1)

    def test_forest(self):
        # Each track gated to its own measurement: a forest, so mu = 0.
        gate = np.eye(4, dtype=bool)
        self.assertEqual(cyclomatic_number_from_gate(gate), 0)

    def test_two_independent_cycles(self):
        # Two disjoint 4-cycles: mu adds up.
        gate = np.zeros((4, 4), dtype=bool)
        gate[:2, :2] = True
        gate[2:, 2:] = True
        self.assertEqual(cyclomatic_number_from_gate(gate), 2)

    def test_no_edges(self):
        self.assertEqual(cyclomatic_number_from_gate(np.zeros((5, 3), dtype=bool)), 0)
        self.assertEqual(cyclomatic_number_from_gate(np.zeros((0, 0), dtype=bool)), 0)

    def test_isolated_tracks_do_not_count(self):
        # A track whose only feasible event is misdetection adds one to both V and C.
        gate = np.array([[True, True], [True, True], [False, False]])
        self.assertEqual(cyclomatic_number_from_gate(gate), 1)

    def test_matches_networkx_on_random_masks(self):
        rng = np.random.default_rng(20260910)
        for _ in range(400):
            n = int(rng.integers(0, 12))
            m = int(rng.integers(0, 12))
            gate = rng.random((n, m)) < rng.uniform(0.0, 1.0)
            self.assertEqual(cyclomatic_number_from_gate(gate), nx_cyclomatic(gate),
                             msg=f"mismatch on\n{gate.astype(int)}")

    def test_scan_level_wrapper_skips_the_misdetection_column(self):
        R_LC = lc([[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(cyclomatic_number(R_LC), 1)


class TestClusterGraphStats(unittest.TestCase):
    def test_counts_and_multihypothesis_closed_form(self):
        # Tracks 1 and 2 both gated to measurements 1 and 2; track 3 never detected.
        R_LC = lc([[1.0, 1.0], [1.0, 1.0], [None, None]])
        ph = hypotheses(((1, 2, 3), 0.5), ((1, 2), 0.5))

        stats = cluster_graph_stats(R_LC, ph)

        self.assertEqual(list(stats.track_ids), [1, 2, 3])
        self.assertEqual(stats.member_prior_clusters, ())
        self.assertEqual(stats.n_tracks, 3)
        self.assertEqual(stats.n_gated_measurements, 2)
        self.assertEqual(stats.n_edges, 4)
        self.assertEqual(stats.mu_bipartite, 1)
        # Theta is wired to all three tracks, so it closes one more loop than the
        # bipartite graph has, and mu collapses to |E| - |M|.
        self.assertEqual(stats.mu_multihypothesis, 2)
        self.assertEqual(stats.mu_multihypothesis,
                         stats.n_edges - stats.n_gated_measurements)

    def test_conditioned_values_match_the_ehm2_slice(self):
        R_LC = lc([[1.0, 1.0], [1.0, 1.0], [1.0, None]])
        ph = hypotheses(((1, 2), 0.25), ((1, 3), 0.25), ((1,), 0.25), ((), 0.25))

        stats = cluster_graph_stats(R_LC, ph)

        self.assertEqual(stats.n_hypotheses, 4)
        self.assertEqual(stats.mu_conditioned.size, 4)
        for k, hypothesis in enumerate(ph):
            rows = np.array(hypothesis.tracks(), dtype=int) - 1
            expected = nx_cyclomatic(np.isfinite(np.asarray(R_LC)[rows, 1:])) if rows.size else 0
            self.assertEqual(int(stats.mu_conditioned[k]), expected)
        # Only {1, 2} closes a cycle; the empty hypothesis contributes nothing.
        self.assertEqual(list(stats.mu_conditioned), [1, 0, 0, 0])
        self.assertEqual(stats.mu_conditioned_max, 1)
        self.assertAlmostEqual(stats.mu_conditioned_mean, 0.25)
        self.assertAlmostEqual(stats.mu_conditioned_prior_mean, 0.25)
        self.assertAlmostEqual(stats.tree_hypothesis_fraction, 0.75)

    def test_prior_weighted_mean_follows_the_hypothesis_probabilities(self):
        R_LC = lc([[1.0, 1.0], [1.0, 1.0]])
        ph = hypotheses(((1, 2), 0.9), ((1,), 0.1))

        stats = cluster_graph_stats(R_LC, ph)

        self.assertEqual(list(stats.mu_conditioned), [1, 0])
        self.assertAlmostEqual(stats.mu_conditioned_mean, 0.5)
        self.assertAlmostEqual(stats.mu_conditioned_prior_mean, 0.9)

    def test_cluster_without_any_feasible_measurement(self):
        R_LC = lc([[None, None], [None, None]])
        stats = cluster_graph_stats(R_LC, hypotheses(((1, 2), 0.5), ((1,), 0.5)))

        self.assertEqual(stats.n_edges, 0)
        self.assertEqual(stats.n_gated_measurements, 0)
        self.assertEqual(stats.mu_bipartite, 0)
        self.assertEqual(stats.mu_multihypothesis, 0)
        self.assertEqual(stats.mu_conditioned_max, 0)
        self.assertAlmostEqual(stats.tree_hypothesis_fraction, 1.0)

    def test_member_prior_clusters_is_recorded(self):
        R_LC = lc([[1.0], [1.0]])
        stats = cluster_graph_stats(R_LC, hypotheses(((1, 2), 1.0)), (3, 5))
        self.assertEqual(stats.member_prior_clusters, (3, 5))

    def test_matches_networkx_on_random_clusters(self):
        rng = np.random.default_rng(4711)
        for _ in range(60):
            n = int(rng.integers(1, 7))
            m = int(rng.integers(1, 7))
            gate = rng.random((n, m)) < rng.uniform(0.2, 0.9)
            R_LC = lc([[1.0 if g else None for g in row] for row in gate])
            ph = hypotheses(
                (tuple(t for t in range(1, n + 1) if rng.random() < 0.7), 0.5),
                (tuple(range(1, n + 1)), 0.5),
            )
            stats = cluster_graph_stats(R_LC, ph)
            tracks = np.array(sorted(ph.tracks()), dtype=int) - 1
            self.assertEqual(stats.mu_bipartite,
                             nx_cyclomatic(np.isfinite(np.asarray(R_LC)[tracks, 1:])))
            self.assertEqual(stats.mu_multihypothesis,
                             stats.n_edges - stats.n_gated_measurements)


class TestScanLevel(unittest.TestCase):
    def test_shared_measurement_couples_two_clusters(self):
        # Cluster A = tracks 1,2 and cluster B = tracks 3,4, all four gated to the same
        # two measurements. Neither cluster's own theta sees the other's tracks, but the
        # measurement nodes are shared, so the scan graph is loopier than either cluster.
        R_LC = lc([[1.0, 1.0]] * 4)
        clusters = [hypotheses(((1, 2), 1.0)), hypotheses(((3, 4), 1.0))]

        per_cluster = [cluster_graph_stats(R_LC, ph) for ph in clusters]
        self.assertEqual([c.mu_multihypothesis for c in per_cluster], [2, 2])
        # V = 4 tracks + 2 measurements + 2 thetas = 8, E = 8 track-meas + 4 track-theta,
        # all one component: mu = 12 - 8 + 1 = 5.
        self.assertEqual(scan_multihypothesis_cyclomatic(R_LC, clusters), 5)

    def test_no_clusters(self):
        self.assertEqual(scan_multihypothesis_cyclomatic(lc([[1.0]]), []), 0)


class TestMergedClusterMembers(unittest.TestCase):
    def test_masters_and_slaves(self):
        # Row 0 is the 1-indexed master of each prior cluster, row 1 flags the masters.
        assocLocal = np.array([[1, 1, 1, 4, 4, 6],
                               [1, 0, 0, 1, 0, 1]], dtype=np.uint8)
        self.assertEqual(merged_cluster_members(assocLocal), [(0, 1, 2), (3, 4), (5,)])

    def test_master_followed_by_a_master(self):
        # The case the standalone merge_clusters() copies get wrong: a master that is not
        # immediately followed by its slaves.
        assocLocal = np.array([[1, 2, 1, 2],
                               [1, 1, 0, 0]], dtype=np.uint8)
        self.assertEqual(merged_cluster_members(assocLocal), [(0, 2), (1, 3)])

    def test_all_independent(self):
        assocLocal = np.array([[1, 2, 3],
                               [1, 1, 1]], dtype=np.uint8)
        self.assertEqual(merged_cluster_members(assocLocal), [(0,), (1,), (2,)])


if __name__ == "__main__":
    unittest.main()
