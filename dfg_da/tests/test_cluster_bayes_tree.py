import unittest
import py_dfg_da as pdd
import py_dfg_da
from dfg_da.cluster_bayes_tree import *
from dfg_da.marginals_computers import MulticlusterExactEHM2, MulticlusterExactOutput, MulticlusterExact
from cluster_data_asso import edmund_to_lc, lc_to_edmund


class TestClusterLinks(unittest.TestCase):
    def setUp(self):
        pass


    def test_measurement_to_cluster_map(self):
        assocLocal = np.array([
            [1, 1],
            [1, 0]
        ])

        R = np.array([
            [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')
        R_LC = edmund_to_lc(R)

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
                pdd.hypothesis.Hypothesis([1, 3], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([4], np.log(0.5)),
                pdd.hypothesis.Hypothesis([5], np.log(0.5))
            ])
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        # cluster links should actually just be a dict over measurement indices with set over cluster
        # We create a dict over linking measurement indices with set over cluster indices, but the inverted dict, i.e a dict over cluster indices with set over linking measurements, is more useful implementation-wise

        # In this case, the "culprit" is measurement 2, as it's gated by tracks in both cluster 0 and 1
        correct_mapping = {
            2: {0, 1}
        }

        for (correct_measurement, correct_cluster_idx_set), (measurement, cluster_idx_set) in zip(correct_mapping.items(), cluster_links.meas_to_clusters().items()):
            self.assertEqual(correct_measurement, measurement)
            self.assertEqual(correct_cluster_idx_set, cluster_idx_set)


    def test_cluster_to_measurements_map(self):
        assocLocal = np.array([
            [1, 1],
            [1, 0]
        ])

        R = np.array([
            [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')
        R_LC = edmund_to_lc(R)

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
                pdd.hypothesis.Hypothesis([1, 3], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([4], np.log(0.5)),
                pdd.hypothesis.Hypothesis([5], np.log(0.5))
            ])
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        correct_mapping = {
            0: { 2 },
            1: { 2 }
        }

        for (correct_cluster, correct_measurement_set), (cluster, measurement_set) in zip(correct_mapping.items(), cluster_links.cluster_to_linking_meas().items()):
            self.assertEqual(correct_cluster, cluster)
            self.assertEqual(correct_measurement_set, measurement_set)

    def test_measurement_to_cluster_map2(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf],
            [0.1,     1.0, -np.inf],
            [0.1,     1.0,     1.0],
            [0.1, -np.inf,     1.0],
            [0.1, -np.inf,     1.0],
            [0.1,     1.0, -np.inf],
        ], order='F')

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([4], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.5)),
                pdd.hypothesis.Hypothesis([6], np.log(0.5))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 1],
            [1, 0, 0]
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        correct_mapping = {
            1: {0, 1, 2},
            2: {1, 2}
        }

        for (correct_measurement, correct_cluster_idx_set), (measurement, cluster_idx_set) in zip(correct_mapping.items(), cluster_links.meas_to_clusters().items()):
            self.assertEqual(correct_measurement, measurement)
            self.assertEqual(correct_cluster_idx_set, cluster_idx_set)


    def test_cluster_to_measurements_map2(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf],
            [0.1,     1.0, -np.inf],
            [0.1,     1.0,     1.0],
            [0.1, -np.inf,     1.0],
            [0.1, -np.inf,     1.0],
            [0.1,     1.0, -np.inf],
        ], order='F')

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([4], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.5)),
                pdd.hypothesis.Hypothesis([6], np.log(0.5))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 1],
            [1, 0, 0]
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        correct_mapping = {
            0: { 1 },
            1: { 1, 2 },
            2: { 1, 2 }
        }

        for (correct_cluster, correct_measurement_set), (cluster, measurement_set) in zip(correct_mapping.items(), cluster_links.cluster_to_linking_meas().items()):
            self.assertEqual(correct_cluster, cluster)
            self.assertEqual(correct_measurement_set, measurement_set)

    def test_measurement_to_cluster_map3(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,     1.0],
        ], order='F')

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([4], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.3)),
                pdd.hypothesis.Hypothesis([6], np.log(0.3)),
                pdd.hypothesis.Hypothesis([7], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([8], np.log(0.3)),
                pdd.hypothesis.Hypothesis([9], np.log(0.3)),
                pdd.hypothesis.Hypothesis([10], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([11], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([12], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 3, 3, 1, 6],
            [1, 0, 1, 0, 0, 1]
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        correct_mapping = {
            1: {0, 1, 4},
            3: { 2, 3 },
            4: { 2, 3 }
        }

        for (correct_measurement, correct_cluster_idx_set), (measurement, cluster_idx_set) in zip(correct_mapping.items(), cluster_links.meas_to_clusters().items()):
            self.assertEqual(correct_measurement, measurement)
            self.assertEqual(correct_cluster_idx_set, cluster_idx_set)

        unmerged_clusters_correct = { 5 }
        self.assertEqual(unmerged_clusters_correct, cluster_links.unmerging_clusters())
 
    def test_cluster_to_measurements_map3(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,     1.0],
        ], order='F')

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([4], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.3)),
                pdd.hypothesis.Hypothesis([6], np.log(0.3)),
                pdd.hypothesis.Hypothesis([7], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([8], np.log(0.3)),
                pdd.hypothesis.Hypothesis([9], np.log(0.3)),
                pdd.hypothesis.Hypothesis([10], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([11], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([12], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 3, 3, 1, 6],
            [1, 0, 1, 0, 0, 1]
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        correct_mapping = {
            0: { 1 },
            1: { 1 },
            2: { 3, 4 },
            3: { 3, 4 },
            4: { 1 },

        }

        for (correct_cluster, correct_measurement_set), (cluster, measurement_set) in zip(correct_mapping.items(), cluster_links.cluster_to_linking_meas().items()):
            self.assertEqual(correct_cluster, cluster)
            self.assertEqual(correct_measurement_set, measurement_set)

        unmerged_clusters_correct = { 5 }
        self.assertEqual(unmerged_clusters_correct, cluster_links.unmerging_clusters())


    def test_measurement_to_cluster_map4(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,     1.0],
        ], order='F')

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([4], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.3)),
                pdd.hypothesis.Hypothesis([6], np.log(0.3)),
                pdd.hypothesis.Hypothesis([7], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([8], np.log(0.3)),
                pdd.hypothesis.Hypothesis([9], np.log(0.3)),
                pdd.hypothesis.Hypothesis([10], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([11], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([12], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 1, 1, 1, 6],
            [1, 0, 0, 0, 0, 1]
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        correct_mapping = {
            1: { 0, 1, 4 },
            3: { 2, 3 },
            4: { 2, 3 },
            5: { 3, 4 }
        }

        for (correct_measurement, correct_cluster_idx_set), (measurement, cluster_idx_set) in zip(correct_mapping.items(), cluster_links.meas_to_clusters().items()):
            self.assertEqual(correct_measurement, measurement)
            self.assertEqual(correct_cluster_idx_set, cluster_idx_set)

        unmerged_clusters_correct = { 5 }
        self.assertEqual(unmerged_clusters_correct, cluster_links.unmerging_clusters())
 
    def test_cluster_to_measurements_map4(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,     1.0],
        ], order='F')

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([4], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.3)),
                pdd.hypothesis.Hypothesis([6], np.log(0.3)),
                pdd.hypothesis.Hypothesis([7], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([8], np.log(0.3)),
                pdd.hypothesis.Hypothesis([9], np.log(0.3)),
                pdd.hypothesis.Hypothesis([10], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([11], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([12], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 1, 1, 1, 6],
            [1, 0, 0, 0, 0, 1]
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        correct_mapping = {
            0: { 1 },
            1: { 1 },
            2: { 3, 4 },
            3: { 3, 4, 5 },
            4: { 1, 5 },
        }

        for (correct_cluster, correct_measurement_set), (cluster, measurement_set) in zip(correct_mapping.items(), cluster_links.cluster_to_linking_meas().items()):
            self.assertEqual(correct_cluster, cluster)
            self.assertEqual(correct_measurement_set, measurement_set)

        unmerged_clusters_correct = { 5 }
        self.assertEqual(unmerged_clusters_correct, cluster_links.unmerging_clusters())


    def test_linking_mappings_per_merging_clusters(self):

        R_LC = np.array([
            [0.1,     1.0, -np.inf],
            [0.1,     1.0, -np.inf],
            [0.1,     1.0,     1.0],
            [0.1, -np.inf,     1.0],
            [0.1, -np.inf,     1.0],
            [0.1,     1.0, -np.inf],
        ], order='F')

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([4], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.5)),
                pdd.hypothesis.Hypothesis([6], np.log(0.5))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 1],
            [1, 0, 0]
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        correct_list = [
            LinkingMappings(cluster_to_linking_measurements={
                0: { 1 },
                1: { 1, 2 },
                2: { 1, 2 }
            }, linking_measurements_to_clusters={
                1: { 0, 1, 2},
                2: { 1, 2}
            }
            )
        ]

        for correct_mapping, mapping in zip(correct_list, cluster_links.linking_mappings_per_merging_clusters()):
            self.assertDictEqual(correct_mapping.cluster_to_linking_measurements, mapping.cluster_to_linking_measurements)
            self.assertDictEqual(correct_mapping.linking_measurements_to_clusters, mapping.linking_measurements_to_clusters)

    def test_linking_mappings_per_merging_clusters2(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf,     1.0, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf,     1.0,     1.0,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,     1.0],
        ], order='F')

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([4], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.3)),
                pdd.hypothesis.Hypothesis([6], np.log(0.3)),
                pdd.hypothesis.Hypothesis([7], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([8], np.log(0.3)),
                pdd.hypothesis.Hypothesis([9], np.log(0.3)),
                pdd.hypothesis.Hypothesis([10], np.log(0.4))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([11], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([12], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 3, 3, 1, 6],
            [1, 0, 1, 0, 0, 1]
        ])

        cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)

        correct_list = [
            
            LinkingMappings(cluster_to_linking_measurements={
                0: { 1 },
                1: { 1 },
                4: { 1 }
            }, linking_measurements_to_clusters={
                1: { 0, 1, 4 }            
            }),

            LinkingMappings(cluster_to_linking_measurements={
                2: { 3, 4 },
                3: { 3, 4 }
            }, linking_measurements_to_clusters={
                3: { 2, 3 },
                4: { 2, 3 }
            })
        ]

        for correct_mapping, mapping in zip(correct_list, cluster_links.linking_mappings_per_merging_clusters()):
            self.assertDictEqual(correct_mapping.cluster_to_linking_measurements, mapping.cluster_to_linking_measurements)
            self.assertDictEqual(correct_mapping.linking_measurements_to_clusters, mapping.linking_measurements_to_clusters)


class TestMHEHM2(unittest.TestCase):
    def test_mhehm2(self):
        R = np.array([
            [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')

        R = R[:3]

        R_LC = edmund_to_lc(R)

        assocLocal = np.array([
            [1],
            [1]
        ])

        prior_hypotheses_per_cluster: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
            py_dfg_da.hypothesis.Hypotheses([
                py_dfg_da.hypothesis.Hypothesis([1, 2], np.log(0.5)),
                py_dfg_da.hypothesis.Hypothesis([1, 3], np.log(0.5))
            ])
        ])

        margs, l = multihypothesis_ehm2(R_LC, prior_hypotheses_per_cluster[0])

        correct = MulticlusterExactEHM2()
        correct_out = correct.compute_marginals(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)

        self.assertAlmostEqual(l, correct_out.exact_normalization_constant)
        self.assertTrue(np.allclose(margs, correct_out.exact_marginals))

    def test_mhehm2_2(self):
        R = np.array([
            [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')

        R = R[3:]

        R_LC = edmund_to_lc(R)

        assocLocal = np.array([
            [1],
            [1]
        ])

        prior_hypotheses_per_cluster: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
            py_dfg_da.hypothesis.Hypotheses([
                py_dfg_da.hypothesis.Hypothesis([4], np.log(0.5)),
                py_dfg_da.hypothesis.Hypothesis([5], np.log(0.5)),
            ])
        ])

        margs, l = multihypothesis_ehm2(R_LC, prior_hypotheses_per_cluster[0])

        correct = MulticlusterExactEHM2()
        correct_out = correct.compute_marginals(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)

        self.assertAlmostEqual(l, correct_out.exact_normalization_constant)
        self.assertTrue(np.allclose(margs, correct_out.exact_marginals))


class TestMulticlusterEfficientMarginals(unittest.TestCase):
    def setUp(self):
        pass

    def test_marginals_computation(self):
        R = np.array([
            [    3.0, -0.60, -np.inf],
            [    3.2, -np.inf,   -0.56],
        ], order='F')


        R_LC = edmund_to_lc(R)

        assocLocal = np.array([
            [1, 1],
            [1, 0]
        ])

        prior_hypotheses_per_cluster: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
            py_dfg_da.hypothesis.Hypotheses([
                py_dfg_da.hypothesis.Hypothesis([1], np.log(0.5)),
                py_dfg_da.hypothesis.Hypothesis([], np.log(0.5))
            ]),
            py_dfg_da.hypothesis.Hypotheses([
                py_dfg_da.hypothesis.Hypothesis([2], np.log(0.5)),
                py_dfg_da.hypothesis.Hypothesis([], np.log(0.5)),
            ])
        ])
        
        exact_computer = MulticlusterExactEHM2()

        exact_output: MulticlusterExactOutput = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)
        
        efficient_cluster = MulticlusterEfficientMarginals(
            R_LC=R_LC,
            prior_hypotheses_per_cluster=prior_hypotheses_per_cluster,
            assocLocal=assocLocal
        )

        efficient_marginals, efficient_likelihood = efficient_cluster.compute_marginals_likelihood()

        print()
        print(f"Efficient:\n{efficient_marginals}\nExact:\n{exact_output.exact_marginals}")
        print(f"Efficient:\n{efficient_likelihood}\nExact:\n{exact_output.exact_normalization_constant}")

        self.assertAlmostEqual(efficient_likelihood, exact_output.exact_normalization_constant)
        self.assertTrue(np.allclose(exact_output.exact_marginals, efficient_marginals))


    def test_marginals_computation2(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf],
            [0.1,     1.0, -np.inf],
            [0.1,     1.0,     1.0],
            [0.1, -np.inf,     1.0],
            [0.1, -np.inf,     1.0],
            [0.1,     1.0, -np.inf],
        ], order='F')

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([4], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.5)),
                pdd.hypothesis.Hypothesis([6], np.log(0.5))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 1],
            [1, 0, 0]
        ])        
        exact_computer = MulticlusterExactEHM2()

        exact_output: MulticlusterExactOutput = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)
        
        efficient_cluster = MulticlusterEfficientMarginals(
            R_LC=R_LC,
            prior_hypotheses_per_cluster=prior_hypotheses_per_cluster,
            assocLocal=assocLocal
        )

        efficient_marginals, efficient_likelihood = efficient_cluster.compute_marginals_likelihood()

        print()
        print(f"Efficient:\n{efficient_marginals}\nExact:\n{exact_output.exact_marginals}")
        print(f"Efficient:\n{efficient_likelihood}\nExact:\n{exact_output.exact_normalization_constant}")

        self.assertAlmostEqual(efficient_likelihood, exact_output.exact_normalization_constant)
        self.assertTrue(np.allclose(exact_output.exact_marginals, efficient_marginals))


if __name__ == "__main__":
    unittest.main()
