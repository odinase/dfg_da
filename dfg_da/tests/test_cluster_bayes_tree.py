import unittest
import py_dfg_da as pdd
from dfg_da.cluster_bayes_tree import *
from cluster_data_asso import edmund_to_lc, lc_to_edmund



class TestClusterLinks(unittest.TestCase):
    def setUp(self):
        pass

    def test_forms_correct_links(self):

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

        assocLocal = np.array([
            [1, 1],
            [1, 0]
        ])
        
        cluster_links = ClusterLinks()
        
        cluster1s = [1, 2, 2]
        cluster2s = [2, 3, 1]

        R = np.array([
            [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')

        # What should cluster link be?
        # It might actually be more beneficial to assume we know the clusters that are merged (which we do), and keep track of the measurements together with the tracks of each cluster that gates that
        # In that sense, perhaps the best is to let cluster links denote all links, i.e >= 1, that connects >= 1 clusters into a supercluster
        # For the case above, perhaps the best way of containing the data is
        # (cluster_idxs...) -> [measurement_idx: (track_idxs, cluster_idxs...), ...] ?
        # Actually, it's probably more practical from an implementation perspective to loop over each linked cluster and get the measurements and gated tracks in that cluster
        # In that sense, the link is just a list that is number of clusters long with a dict over all measurement idxs that maps to the tracks that gates it. It might be redundant information to also include the tracks that gate the measurement, but it doesn't hurt
        # Maybe use a dict instead of a list as well, with cluster idx?

        correct_cluster_links = {
            0: {
                1: { 3 },
                2: { 3 }
            },
            1: {}
        }
        for cluster_idx, linking_measurements in cluster_links:
            # cluster idx is a integer
            # linking measurement is a dict over measurement idxs that are involved in the
            self.assertEqu
            for meas_idx, gated_tracks in linking_measurements.values():




if __name__ == "__main__":
    unittest.main()
