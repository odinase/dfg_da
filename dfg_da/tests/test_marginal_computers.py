import unittest
import py_dfg_da as pdd
from dfg_da.marginals_computers import *
from cluster_data_asso import edmund_to_lc, lc_to_edmund


class TestClusterHypothesesPosterior(unittest.TestCase):
    def setUp(self):
        pass

    def test_prior_hypos_in_posterior_clusters1(self):
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
        # true_mapping = [
        #     [
        #         [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 0)],
        #         [ClusterHypothesisLabel(0, 0), ClusterHypothesisLabel(1, 1)],
        #         [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 0)],
        #         [ClusterHypothesisLabel(0, 1), ClusterHypothesisLabel(1, 1)],
        #     ],
        #     [
        #         [ClusterHypothesisLabel(2, 0)],
        #         [ClusterHypothesisLabel(2, 1)],
        #     ]
        # ]
        true_ph_dict_per_cluster = [
            {0: 2, 1: 2},
            {2: 2}
        ]

        ph_dict_per_cluster = chp.prior_hypos_in_posterior_clusters()


        self.assertEqual(len(true_ph_dict_per_cluster), len(ph_dict_per_cluster))

        for true_ph_dict, ph_dict in zip(true_ph_dict_per_cluster, ph_dict_per_cluster):
            self.assertEqual(len(true_ph_dict), len(ph_dict))

            for (true_c_idx, true_num_hypo), (c_idx, num_hypo) in zip(true_ph_dict.items(), ph_dict.items()):
                self.assertEqual(true_c_idx, c_idx)
                self.assertEqual(true_num_hypo, num_hypo)


    def test_prior_hypos_in_posterior_clusters2(self):
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

        true_ph_dict_per_cluster = [
            {0: 2, 1: 2, 2: 2}
        ]

        ph_dict_per_cluster = chp.prior_hypos_in_posterior_clusters()


        self.assertEqual(len(true_ph_dict_per_cluster), len(ph_dict_per_cluster))

        for true_ph_dict, ph_dict in zip(true_ph_dict_per_cluster, ph_dict_per_cluster):
            self.assertEqual(len(true_ph_dict), len(ph_dict))

            for (true_c_idx, true_num_hypo), (c_idx, num_hypo) in zip(true_ph_dict.items(), ph_dict.items()):
                self.assertEqual(true_c_idx, c_idx)
                self.assertEqual(true_num_hypo, num_hypo)

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
                pdd.hypothesis.Hypothesis([5], np.log(0.1)),
                pdd.hypothesis.Hypothesis([6], np.log(0.9))
            ])
        ])

        assocLocal = np.array([
            [1, 2, 1],
            [1, 1, 0]
        ])

        chp = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster)
        true_probabilities_per_cluster = [
            [
                0.2*0.1,
                0.2*0.9,
                0.8*0.1,
                0.8*0.9,
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
        
class TestMulticlusterExactOutput(unittest.TestCase):
    def setUp(self):
        self.R = np.array([
            [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')

        self.R_LC = np.asfortranarray(edmund_to_lc(self.R))

        assocLocal = np.array([
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
        self.mco: MulticlusterExactOutput = exact_computer(self.R_LC, self.prior_hypotheses_per_cluster, assocLocal=assocLocal)

    def test_computation_of_prior_hypothesis_posterior_distribution(self):
        theta_marginals = self.mco.compute_theta_posteriors()

        correct_marginals = [
            np.array([0.604131, 0.395869]),
            np.array([0.904496, 0.0955038])
        ]

        self.assertEqual(len(theta_marginals), len(correct_marginals))

        for correct_marginal, theta_marginal in zip(correct_marginals, theta_marginals):
            self.assertEqual(len(correct_marginal), len(theta_marginal))
            self.assertTrue(np.allclose(correct_marginal, theta_marginal))


class TestMulticlusterExactEHM2(unittest.TestCase):
    def setUp(self):
        self.exact_computer = MulticlusterExactEHM2()

    def test_compare_with_naive_exact1(self):
        R = np.array([
            [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')

        R_LC = np.asfortranarray(edmund_to_lc(R))

        assocLocal = np.array([
            [1, 1],
            [1, 0]
        ])

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

        mco: MulticlusterExactOutput = self.exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)
        mco_compare: MulticlusterExactOutput = self.exact_naive(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)

        self.assertTrue(np.allclose(mco.exact_marginals, mco_compare.exact_marginals))

        theta_posteriors = mco.compute_theta_posteriors()
        theta_posteriors_compare = mco_compare.compute_theta_posteriors()

        self.assertEqual(len(theta_posteriors), len(theta_posteriors_compare))

        for p, p_compare in zip(theta_posteriors, theta_posteriors_compare):
            self.assertTrue(np.allclose(p, p_compare))


    def test_compare_with_naive_exact1(self):
        R = np.array([
            [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
            [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
            [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
            [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
            [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
        ], order='F')

        R_LC = np.asfortranarray(edmund_to_lc(R))

        assocLocal = np.array([
            [1, 1],
            [1, 0]
        ])

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

        mco: MulticlusterExactOutput = self.exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal.copy())
        track_marginals, meas_marginals, theta_marginals, exact_normalization_constant = pdd.factor_graph.all_exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)


        self.assertTrue(np.allclose(mco.exact_marginals, track_marginals.T))

        theta_posteriors = mco.compute_theta_posteriors()

        self.assertEqual(len(theta_posteriors), len(theta_marginals))

        for p, p_compare in zip(theta_posteriors, theta_marginals.values()):
            self.assertTrue(np.allclose(p, p_compare))

        self.assertAlmostEqual(mco.exact_normalization_constant, exact_normalization_constant)
        for l, ph in zip(mco.hypo_cond_normalization_constants_per_cluster, mco.cluster_hypotheses_posterior.prior_hypotheses_per_cluster_posterior):
            tracks = np.sort(np.fromiter(ph.tracks(), dtype=int)) - 1
            R_sub = R_LC[tracks]
            R_sub = np.asfortranarray(lc_to_edmund(R_sub))
            old2new = dict()
            for i, t in enumerate(tracks):
                old2new[t+1] = i +1
            ph.reindex_tracks(old2new)
            l_compare = pdd.factor_graph.hypothesis_conditioned_likelihoods(R_sub, ph)
            self.assertTrue(np.allclose(l, l_compare))


    def test_compare_with_naive_exact2(self):
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

        mco: MulticlusterExactOutput = self.exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal.copy())
        track_marginals, meas_marginals, theta_marginals, exact_normalization_constant = pdd.factor_graph.all_exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)


        self.assertTrue(np.allclose(mco.exact_marginals, track_marginals.T))

        theta_posteriors = mco.compute_theta_posteriors()

        self.assertEqual(len(theta_posteriors), len(theta_marginals))

        for p, p_compare in zip(theta_posteriors, theta_marginals.values()):
            self.assertTrue(np.allclose(p, p_compare))

        self.assertAlmostEqual(mco.exact_normalization_constant, exact_normalization_constant)
        for l, ph in zip(mco.hypo_cond_normalization_constants_per_cluster, mco.cluster_hypotheses_posterior.prior_hypotheses_per_cluster_posterior):
            tracks = np.sort(np.fromiter(ph.tracks(), dtype=int)) - 1
            R_sub = R_LC[tracks]
            R_sub = np.asfortranarray(lc_to_edmund(R_sub))
            old2new = dict()
            for i, t in enumerate(tracks):
                old2new[t+1] = i +1
            ph.reindex_tracks(old2new)
            l_compare = pdd.factor_graph.hypothesis_conditioned_likelihoods(R_sub, ph)
            self.assertTrue(np.allclose(l, l_compare))


    def test_compare_with_naive_exact3(self):
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
                pdd.hypothesis.Hypothesis([4, 5], np.log(0.3)),
                pdd.hypothesis.Hypothesis([], np.log(0.7))
            ]),
            pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([], np.log(0.5)),
                pdd.hypothesis.Hypothesis([6, 7], np.log(0.5))
            ])
        ])

        mco: MulticlusterExactOutput = self.exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal.copy())

        track_marginals, meas_marginals, theta_marginals, exact_normalization_constant = pdd.factor_graph.all_exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)


        self.assertTrue(np.allclose(mco.exact_marginals, track_marginals.T))

        theta_posteriors = mco.compute_theta_posteriors()

        self.assertEqual(len(theta_posteriors), len(theta_marginals))

        for p, p_compare in zip(theta_posteriors, theta_marginals.values()):
            self.assertTrue(np.allclose(p, p_compare))

        self.assertAlmostEqual(mco.exact_normalization_constant, exact_normalization_constant)
        for l, ph in zip(mco.hypo_cond_normalization_constants_per_cluster, mco.cluster_hypotheses_posterior.prior_hypotheses_per_cluster_posterior):
            tracks = np.sort(np.fromiter(ph.tracks(), dtype=int)) - 1
            R_sub = R_LC[tracks]
            R_sub = np.asfortranarray(lc_to_edmund(R_sub))
            old2new = dict()
            for i, t in enumerate(tracks):
                old2new[t+1] = i +1
            ph.reindex_tracks(old2new)
            l_compare = pdd.factor_graph.hypothesis_conditioned_likelihoods(R_sub, ph)
            self.assertTrue(np.allclose(l, l_compare))


if __name__ == '__main__':
    unittest.main()
