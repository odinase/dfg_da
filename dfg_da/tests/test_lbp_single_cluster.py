import unittest
import py_dfg_da as pdd
import py_dfg_da
from dfg_da.marginals_computers import *
from cluster_data_asso import edmund_to_lc, lc_to_edmund


class TestCPPLBPvsPython(unittest.TestCase):
    def setUp(self):
        pass

    def test_marginals_computation(self):
        R = np.array([
            [    3.0, -0.60, -np.inf],
            [    3.2, -np.inf,   -0.56],
        ], order='F')

        R_LC = edmund_to_lc(R)

        prior_hypotheses = py_dfg_da.hypothesis.Hypotheses([
                py_dfg_da.hypothesis.Hypothesis([1, 2], np.log(0.5)),
                py_dfg_da.hypothesis.Hypothesis([ ], np.log(0.5))
            ])
        
        python_lbp = LBPMarginalsFullAssociation()

        marginals, theta_posterior, likelihood = python_lbp(R_LC, prior_hypotheses)
        cpp_output = pdd.lbp.lbp_single_cluster(R, prior_hypotheses)

        self.assertAlmostEqual(likelihood, cpp_output.bethe_pseudodual_normalization_constant())
        self.assertTrue(np.allclose(marginals, cpp_output.track_association_marginals().T))
        self.assertTrue(np.allclose(cpp_output.hypotheses_marginal(), theta_posterior))


    def test_marginals_computation2(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf],
            [0.1,     1.0, -np.inf],
            [0.1,     1.0,     1.0]
        ], order='F')


        prior_hypotheses = pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
                pdd.hypothesis.Hypothesis([1, 3], np.log(0.5))
            ])

        R = np.asfortranarray(lc_to_edmund(R_LC))
        python_lbp = LBPMarginalsFullAssociation()

        marginals, theta_posterior, likelihood = python_lbp(R_LC, prior_hypotheses)
        cpp_output = pdd.lbp.lbp_single_cluster(R, prior_hypotheses)

        self.assertAlmostEqual(likelihood, cpp_output.bethe_pseudodual_normalization_constant())
        self.assertTrue(np.allclose(marginals, cpp_output.track_association_marginals().T))
        self.assertTrue(np.allclose(cpp_output.hypotheses_marginal(), theta_posterior))

    def test_marginals_computation3(self):
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

        R_LC[:, 0] = np.log(R_LC[:, 0])

        prior_hypotheses = pdd.hypothesis.Hypotheses([
                pdd.hypothesis.Hypothesis([1, 2, 3], np.log(0.5)),
                pdd.hypothesis.Hypothesis([2, 4, 5], np.log(0.5)),
                pdd.hypothesis.Hypothesis([6, 7], np.log(0.3)),
                pdd.hypothesis.Hypothesis([7], np.log(0.4)),
                pdd.hypothesis.Hypothesis([8], np.log(0.3)),
                pdd.hypothesis.Hypothesis([9], np.log(0.3)),
                pdd.hypothesis.Hypothesis([10], np.log(0.4)),
                pdd.hypothesis.Hypothesis([11], np.log(0.9)),
                pdd.hypothesis.Hypothesis([12], np.log(0.9)),
                pdd.hypothesis.Hypothesis([  ], np.log(0.1)),
            ])

        R = np.asfortranarray(lc_to_edmund(R_LC))
        python_lbp = LBPMarginalsFullAssociation()

        marginals, theta_posterior, likelihood = python_lbp(R_LC, prior_hypotheses)
        cpp_output = pdd.lbp.lbp_single_cluster(R, prior_hypotheses)

        self.assertAlmostEqual(likelihood, cpp_output.bethe_pseudodual_normalization_constant())
        self.assertTrue(np.allclose(marginals, cpp_output.track_association_marginals().T))
        self.assertTrue(np.allclose(cpp_output.hypotheses_marginal(), theta_posterior))


    def test_marginals_computation4(self):
        R_LC = np.array([
            [0.1,     1.0, -np.inf, -np.inf,     1.0],
            [0.1,     1.0,     1.0,     1.0, -np.inf],
            [0.1,     1.0,     1.0,     1.0, -np.inf],
            [0.1, -np.inf,     1.0, -np.inf,     1.0],
        ], order='F')

        R_LC[:, 0] = np.log(R_LC[:, 0])

        prior_hypotheses = pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
            pdd.hypothesis.Hypothesis([2, 3], np.log(0.5)),
            pdd.hypothesis.Hypothesis([4], np.log(0.5))
        ])

        R = np.asfortranarray(lc_to_edmund(R_LC))
        python_lbp = LBPMarginalsFullAssociation()

        marginals, theta_posterior, likelihood = python_lbp(R_LC, prior_hypotheses)
        cpp_output = pdd.lbp.lbp_single_cluster(R, prior_hypotheses)

        self.assertAlmostEqual(likelihood, cpp_output.bethe_pseudodual_normalization_constant())
        self.assertTrue(np.allclose(marginals, cpp_output.track_association_marginals().T))
        self.assertTrue(np.allclose(cpp_output.hypotheses_marginal(), theta_posterior))
