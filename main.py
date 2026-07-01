import numpy as np
# import factorgraph as fg
from dfg_da.marginal_association_Odin import exact_marginal, lbp_marginal, lbp_marginal_nonexistence
from time import time
import matplotlib.pyplot as plt
from cluster_data_asso import edmund_to_lc

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
    out = lbp_marginal(R_LC)
    print(out)
    print(out["prob"])
