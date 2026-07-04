import numpy as np
# import factorgraph as fg
from dfg_da.marginal_association_Odin import  lbp_marginal
from time import time
import matplotlib.pyplot as plt
from cluster_data_asso import edmund_to_lc
import dfg_da_py as ddpy

if __name__ == "__main__":
    # # Make an empty graph
    # g = fg.Graph()

    # # Add discrete random variables (RVs)
    # g.rv('th', 2)
    # g.rv('a1', 3)
    # g.rv('a2', 3)
    # g.rv('a3', 3)
    # g.rv('b', 4)

    # g.factor(['th'], potential=np.array([0.5, 0.5]))

    # # Add factors between theta and tracks a
    # g.factor(['th', 'a1'], potential=np.array([
    #     [1., 1., 0.],
    #     [1., 1., 0.]
    # ]))

    # g.factor(['th', 'a2'], potential=np.array([
    #     [1., 1., 0.],
    #     [0., 0., 1.]
    # ]))

    # g.factor(['th', 'a3'], potential=np.array([
    #     [0., 0., 1.],
    #     [1., 1., 0.]
    # ]))

    # # Add factors between tracks a and measurement b
    # g.factor(['a1', 'b'], potential=np.array([
    #     [1., 0., 1., 1.],
    #     [0., 1., 0., 0.],
    #     [1., 0., 1., 1.],
    # ]))

    # g.factor(['a2', 'b'], potential=np.array([
    #     [1., 1., 0., 1.],
    #     [0., 0., 1., 0.],
    #     [1., 1., 0., 1.],
    # ]))

    # g.factor(['a3', 'b'], potential=np.array([
    #     [1., 1., 1., 0.],
    #     [0., 0., 0., 1.],
    #     [1., 1., 1., 0.],
    # ]))

    # Reward matrix
    # num_tracks = 10
    # num_meas = 3
    # R = np.full((num_tracks, num_tracks + num_meas), -np.inf)
    # R[:, :num_meas] = np.random.rand(num_tracks, num_meas)*15 - 5
    # d = np.diag_indices_from(R[:, num_meas:])
    # R[:, num_meas:][d] = np.random.rand(num_tracks) - 0.5

    R = np.array([
        [5.78, 3.78, -0.46, -np.inf, -np.inf],
        [6.37, 6.57, -np.inf, -0.52, -np.inf],
        [7.58, 2.58, -np.inf, -np.inf, -0.60]
    ])
    
    R_LC = edmund_to_lc(R)
    print(R_LC)

    # Python reference implementation.
    out = lbp_marginal(R_LC)
    probs_py = out["prob"]

    # Rust implementation (dfg_da_py).
    probs_rust, k_rust = ddpy.lbp_marginal(R_LC)

    # --- Verify the Rust implementation against the Python reference ---
    # 1. Marginal association probabilities must match.
    marginals_ok = np.allclose(probs_rust, probs_py)
    print("marginals match     :", marginals_ok,
          " (max abs diff:", np.max(np.abs(probs_rust - probs_py)), ")")

    # 2. Bethe loglikelihood. The Rust code returns the partition-function form
    #    (Z_t / Z_j / Z_tj); the reference's out["loglikelihood"] is the entropy form
    #    (U_B - H_B), which only coincides at an exact BP fixed point. So verify k_rust
    #    against the partition-function value recomputed from the reference's own messages.
    r = lbp_marginal(R_LC, return_mu_nu_w_nmd=True)
    w_nmd, a2b, b2a = r["w_nmd"], r["a2b_msg"], r["b2a_msg"]
    n, m = w_nmd.shape
    w_0 = np.ones((n, 1))
    wtm = w_nmd * b2a
    Z_t = w_0.ravel() + wtm.sum(axis=1)
    Z_j = 1.0 + a2b.sum(axis=0)
    Z_tj = (w_0 + (wtm.sum(axis=1, keepdims=True) - wtm)) \
        * (1 + (a2b.sum(axis=0, keepdims=True) - a2b)) + w_nmd
    F = (m - 1) * np.log(Z_t).sum() + (n - 1) * np.log(Z_j).sum() - np.log(Z_tj).sum()
    bethe_loglik_py = -F
    bethe_ok = np.isclose(k_rust, bethe_loglik_py)
    print("bethe loglik match  :", bethe_ok,
          " (rust:", k_rust, " python:", bethe_loglik_py, ")")
    print("reference entropy-form loglikelihood:", out["loglikelihood"])

    assert marginals_ok, "Rust marginals differ from the Python reference"
    assert bethe_ok, "Rust Bethe loglikelihood differs from the Python reference"
    print("\nOK: Rust implementation matches the Python reference.")