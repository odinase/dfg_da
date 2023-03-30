import unittest
import py_dfg_da as pdd
from dfg_da.marginals_computers import *

class TestClusterHypothesesPosterior(unittest.TestCase):
    def setUp(self):
        pass

    def test_correct_hypothesis_idx_mapping(self):
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

        chp = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster)
        true_mapping = [
            [
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 0)],
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 1)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 0)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 1)]
            ]
        ]

        # Assert we have equally many master clusters
        self.assertEqual(len(chp.hypothesis_index_map), len(true_mapping))

        for m1, m2 in zip(chp.hypothesis_index_map, true_mapping):
            # Assert that the number of hypotheses used to construct the posterior set are equal
            self.assertEqual(len(m1), len(m2))
            for label, true_label in zip(m1, m2):
                self.assertEqual(label, true_label)


    def test_correct_hypothesis_idx_mapping2(self):
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

        chp = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster)
        true_mapping = [
            [
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 0)],
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 1)],
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 0)],
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 1)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 0)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 1)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 0)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 1)]
            ]
        ]

        # Assert we have equally many master clusters
        self.assertEqual(len(chp.hypothesis_index_map), len(true_mapping))

        for m1, m2 in zip(chp.hypothesis_index_map, true_mapping):
            # Assert that the number of hypotheses used to construct the posterior set are equal
            self.assertEqual(len(m1), len(m2))
            for label, true_label in zip(m1, m2):
                self.assertEqual(label, true_label)


    def test_correct_hypothesis_idx_mapping3(self):
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
            [1, 1, 3],
            [1, 0, 1]
        ])

        chp = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster)
        true_mapping = [
            [
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 0)],
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 1)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 0)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 1)],
            ],
            [
                [ClusterHypothesisLabel(2, 0)],
                [ClusterHypothesisLabel(2, 1)],
            ]
        ]

        # Assert we have equally many master clusters
        self.assertEqual(len(chp.hypothesis_index_map), len(true_mapping))

        for m1, m2 in zip(chp.hypothesis_index_map, true_mapping):
            # Assert that the number of hypotheses used to construct the posterior set are equal
            self.assertEqual(len(m1), len(m2))
            for label, true_label in zip(m1, m2):
                self.assertEqual(label, true_label)



    def test_correct_hypothesis_idx_mapping4(self):
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
            [1, 2, 1],
            [1, 1, 0]
        ])

        chp = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster)
        true_mapping = [
            [
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(2, 0)],
                [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(2, 1)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(2, 0)],
                [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(2, 1)],
            ],
            [
                [ClusterHypothesisLabel(1, 0)],
                [ClusterHypothesisLabel(1, 1)],
            ]
        ]

        # Assert we have equally many master clusters
        self.assertEqual(len(chp.hypothesis_index_map), len(true_mapping))

        for m1, m2 in zip(chp.hypothesis_index_map, true_mapping):
            # Assert that the number of hypotheses used to construct the posterior set are equal
            self.assertEqual(len(m1), len(m2))
            for label, true_label in zip(m1, m2):
                self.assertEqual(label, true_label)


    def test_correct_posterior_hypothesis_probabilities1(self):
        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.2)),
                pdd.hypothesis.Hypothesis([2], np.log(0.8))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.3)),
                pdd.hypothesis.Hypothesis([4], np.log(0.7))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.5)),
                pdd.hypothesis.Hypothesis([6], np.log(0.5))
            ])
        ])

        assocLocal = np.array([
            [1, 2, 1],
            [1, 1, 0]
        ])

        chp = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster)
        true_probabilities_per_cluster = [
            [
                0.2*0.5,
                0.2*0.5,
                0.8*0.5,
                0.8*0.5,
            ],
            [
                0.3,
                0.7,
            ]
        ]

        for true_probabilities, ph in zip(true_probabilities_per_cluster, chp.prior_hypotheses_per_cluster_posterior):
            for true_prob, prob in zip(true_probabilities, ph.hypothesis_probabilites()):
                self.assertAlmostEqual(true_prob, prob, delta=1e-6)


    def test_correct_posterior_hypothesis_probabilities2(self):
        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1], np.log(0.2)),
                pdd.hypothesis.Hypothesis([2], np.log(0.8))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([3], np.log(0.3)),
                pdd.hypothesis.Hypothesis([4], np.log(0.7))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([5], np.log(0.5)),
                pdd.hypothesis.Hypothesis([6], np.log(0.5))
            ])
        ])

        assocLocal = np.array([
            [1, 1, 3],
            [1, 0, 1]
        ])

        chp = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster)
        true_probabilities_per_cluster = [
            [
                0.2*0.3,
                0.2*0.7,
                0.8*0.3,
                0.8*0.7,
            ],
            [
                0.5,
                0.5,
            ]
        ]

        for true_probabilities, ph in zip(true_probabilities_per_cluster, chp.prior_hypotheses_per_cluster_posterior):
            for true_prob, prob in zip(true_probabilities, ph.hypothesis_probabilites()):
                self.assertAlmostEqual(true_prob, prob, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
