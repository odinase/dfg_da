import unittest
import py_dfg_da as pdd
from dfg_da.marginals_computers import *
from cluster_data_asso import edmund_to_lc, lc_to_edmund


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


    def test_correct_hypothesis_idx_mapping5(self):
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
            [1, 2, 2],
            [1, 1, 0]
        ])

        chp = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster)
        true_mapping = [
            [
                [ClusterHypothesisLabel(0, 0)],
                [ClusterHypothesisLabel(0, 1)],
            ],
            [
                [ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 0)],
                [ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 1)],
                [ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 0)],
                [ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 1)],
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



    def test_prior_to_posterior_mapping1(self):
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

        label_prior = ClusterHypothesisLabel(2, 1)
        # true_mapping = [
        #     [
        #         [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 0)],
        #         [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 1)],
        #         [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 0)],
        #         [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 1)],
        #         [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 0)],
        #         [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 1)],
        #         [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 0)],
        #         [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 1)]
        #     ]
        # ]
        true_post_cidx = 0
        true_hyp_idxs = [1, 3, 5, 7]

        post_cidx, hyp_idxs = chp.map_prior_to_posteriors(label_prior)

        self.assertEqual(true_post_cidx, post_cidx)
        self.assertListEqual(true_hyp_idxs, hyp_idxs)


    def test_prior_to_posterior_mapping2(self):
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
            [1, 2, 2],
            [1, 1, 0]
        ])

        chp = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster)
        # true_mapping = [
        #     [
        #         [ClusterHypothesisLabel(0, 0)],
        #         [ClusterHypothesisLabel(0, 1)],
        #     ],
        #     [
        #         [ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 0)],
        #         [ClusterHypothesisLabel(1, 0), ClusterHypothesisLabel(2, 1)],
        #         [ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 0)],
        #         [ClusterHypothesisLabel(1, 1), ClusterHypothesisLabel(2, 1)],
        #     ]
        # ]

        label_prior = ClusterHypothesisLabel(2, 1)
        true_post_cidx = 1
        true_hyp_idxs = [1, 3]

        post_cidx, hyp_idxs = chp.map_prior_to_posteriors(label_prior)

        self.assertEqual(true_post_cidx, post_cidx)
        self.assertListEqual(true_hyp_idxs, hyp_idxs)


class TestMulticlusterExact(unittest.TestCase):
    def setUp(self):
        pass

    def test_correct_computation1(self):
        self.R = np.array([
            [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')

        self.R_LC = edmund_to_lc(self.R)

        self.assocLocal = np.array([
            [1, 1],
            [1, 0]
        ])

        self.prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
                pdd.hypothesis.Hypothesis([1, 3], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([4], np.log(0.5)),
                pdd.hypothesis.Hypothesis([5], np.log(0.5))
            ])
        ])
        exact_computer = MulticlusterExact()

        exact_output: MulticlusterExactOutput = exact_computer(self.R_LC, self.prior_hypotheses_per_cluster, assocLocal = self.assocLocal)

        correct_marginals = np.array([
            [0.34079521, 0.65920479, 0.,         0.        ],
            [0.28200107, 0.32212954, 0.,         0.39586939],
            [0.31165938, 0.00065374, 0.08355627, 0.60413061],
            [0.06285782, 0.,         0.84163842, 0.09550376],
            [0.06741553, 0.,         0.02808823, 0.90449624]
        ])

        self.assertTrue(np.allclose(correct_marginals, exact_output.exact_marginals))

        correct_multicluster_constant = 228.5276770589719
        exact_multicluster_constant = exact_output.exact_normalization_constant

        self.assertAlmostEqual(correct_multicluster_constant, exact_multicluster_constant, delta=1e-6)


    def test_correct_computation2(self):
        R = np.array([
            [    3.0, -np.inf,  -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,-np.inf],
            [    3.2, -np.inf,  -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf, -np.inf,-np.inf],
            [   -3.0,     1.2,  -np.inf, -np.inf, -np.inf,  -0.46, -np.inf,  -np.inf, -np.inf, -np.inf],
            [-np.inf, -np.inf,      1.7, -np.inf, -np.inf,-np.inf,  -0.46, -np.inf, -np.inf, -np.inf,],
            [-np.inf, -np.inf,      2.3, -np.inf, -np.inf,-np.inf,-np.inf,  -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0,  -np.inf, -np.inf, -np.inf,-np.inf,-np.inf,-np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4,  -np.inf, -np.inf, -np.inf, -np.inf,-np.inf,-np.inf,-np.inf,   -0.55],
        ], order='F')

        R_LC = edmund_to_lc(R)

        assocLocal = np.array([
            [1, 2, 1],
            [1, 1, 0]
        ])

        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
                pdd.hypothesis.Hypothesis([1, 3], np.log(0.5))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([4], np.log(0.5)),
                pdd.hypothesis.Hypothesis([5], np.log(0.5)),
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([6], np.log(0.2)),
                pdd.hypothesis.Hypothesis([7], np.log(0.8)),
            ])
        ])
        exact_computer = MulticlusterExact()

        exact_output: MulticlusterExactOutput = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal = assocLocal)

        correct_marginals = np.array([
            [0.31260581, 0.68739419, 0.        , 0.        , 0.        ],
            [0.25670035, 0.29322856, 0.        , 0.        , 0.45007109],
            [0.28369776, 0.00059509, 0.16577824, 0.        , 0.54992891],
            [0.03777722, 0.        , 0.        , 0.32757146, 0.63465133],
            [0.03777722, 0.        , 0.        , 0.59687411, 0.36534867],
            [0.04885913, 0.        , 0.65420213, 0.        , 0.29693873],
            [0.20960728, 0.        , 0.08733146, 0.        , 0.70306127]
        ])
        correct_multicluster_constant = 982.6004625179592

        self.assertTrue(np.allclose(correct_marginals, exact_output.exact_marginals))
        self.assertAlmostEqual(correct_multicluster_constant, exact_output.exact_normalization_constant, delta=1e-6)

    def test_correct_computation3(self):
        R = np.array([
            [    3.0, -np.inf,  -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf,  -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2,  -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf, -np.inf, -np.inf],
            [-np.inf,     3.0,  -np.inf, -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf, -np.inf],
            [-np.inf,    -0.4,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf, -np.inf,      1.7, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf, -np.inf,      2.3, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')

        R_LC = edmund_to_lc(R)

        assocLocal = np.array([
            [1, 1, 3],
            [1, 0, 1]
        ])
        prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1, 2], np.log(0.2)),
                pdd.hypothesis.Hypothesis([1, 3], np.log(0.8))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([4], np.log(0.3)),
                pdd.hypothesis.Hypothesis([5], np.log(0.7))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([6], np.log(0.5)),
                pdd.hypothesis.Hypothesis([7], np.log(0.5))
            ])
        ])
        exact_computer = MulticlusterExact()

        exact_output: MulticlusterExactOutput = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal = assocLocal)

        correct_marginals = np.array([
            [0.15645797 ,0.84354203, 0.        , 0.        , 0.        ],
            [0.1158497  ,0.132335  , 0.        , 0.        , 0.7518153 ],
            [0.51213489 ,0.00107426, 0.23860615, 0.        , 0.2481847 ],
            [0.09181665 ,0.        , 0.64381036, 0.        , 0.26437299],
            [0.21423885 ,0.        , 0.05013413, 0.        , 0.73562701],
            [0.03247864 ,0.        , 0.        , 0.33049203, 0.63702934],
            [0.0348336  ,0.        , 0.        , 0.60219573, 0.36297066],
        ])
        correct_multicluster_constant = 1200.8442065519412

        self.assertTrue(np.allclose(correct_marginals, exact_output.exact_marginals))
        self.assertAlmostEqual(correct_multicluster_constant, exact_output.exact_normalization_constant, delta=1e-6)
        

if __name__ == '__main__':
    unittest.main()
