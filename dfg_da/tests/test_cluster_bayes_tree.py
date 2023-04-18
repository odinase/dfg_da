import unittest
import py_dfg_da as pdd
from dfg_da.cluster_bayes_tree import *
from cluster_data_asso import edmund_to_lc, lc_to_edmund


class TestClusterLinks(unittest.TestCase):
    def setUp(self):
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

        self.cluster_links = ClusterLinks(R_LC, prior_hypotheses_per_cluster, assocLocal)


    def test_measurement_to_cluster_map(self):

        # cluster links should actually just be a dict over measurement indices with set over cluster
        # We create a dict over linking measurement indices with set over cluster indices, but the inverted dict, i.e a dict over cluster indices with set over linking measurements, is more useful implementation-wise

        # In this case, the "culprit" is measurement 2, as it's gated by tracks in both cluster 0 and 1
        correct_mapping = {
            2: {0, 1}
        }

        for (correct_measurement, correct_cluster_idx_set), (measurement, cluster_idx_set) in zip(correct_mapping.values(), self.cluster_links.meas_to_clusters().values()):
            self.assertEqual(correct_measurement, measurement)
            self.assertEqual(correct_cluster_idx_set, cluster_idx_set)


    def test_cluster_to_measurements_map(self):
        correct_mapping = {
            0: { 2 },
            1: { 2 }
        }

        for (correct_cluster, correct_measurement_set), (cluster, measurement_set) in zip(correct_mapping.values(), self.cluster_links.cluster_to_linking_meas().values()):
            self.assertEqual(correct_cluster, cluster)
            self.assertEqual(correct_measurement_set, measurement_set)



if __name__ == "__main__":
    unittest.main()
