import py_dfg_da
import numpy as np

if __name__ == "__main__":
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    n, mpn = R.shape
    m = mpn - n

    R_LC = np.hstack((np.diag(R[:, m:])[:,None], R[:,:m]))

    prior_hypotheses_per_cluster: py_dfg_da.HypothesesList = py_dfg_da.HypothesesList([
        py_dfg_da.Hypotheses([
            py_dfg_da.Hypothesis([1, 2], np.log(0.5)),
            py_dfg_da.Hypothesis([1, 3], np.log(0.5))
        ]),
        py_dfg_da.Hypotheses([
            py_dfg_da.Hypothesis([4], np.log(0.5)),
            py_dfg_da.Hypothesis([5], np.log(0.5))
        ])
    ])

    output = py_dfg_da.lbp_multicluster(R, prior_hypotheses_per_cluster)
    np.set_printoptions(suppress=True)
    print(output.track_association_marginals().T)