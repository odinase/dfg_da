import factorgraph as fg
from dfg_da.marginal_association_Odin import exact_marginal, lbp_marginal, lbp_marginal_nonexistence
from time import time
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Make an empty graph
    g = fg.Graph()

    # Add discrete random variables (RVs)
    g.rv('th1', 2)
    g.rv('th2', 2)

    g.rv('a1', 4)
    g.rv('a2', 4)
    g.rv('a3', 4)
    g.rv('a4', 4)
    g.rv('a5', 4)
    
    g.rv('b1', 6)
    g.rv('b2', 6)

    g.factor(['th1'], potential=np.array([0.5, 0.5]))
    g.factor(['th2'], potential=np.array([0.5, 0.5]))

    # Cluster 1
    # Add factors between theta and tracks a
    g.factor(['th1', 'a1'], potential=np.array([
        [1., 1., 1., 0.],
        [1., 1., 1., 0.]
    ]))

    g.factor(['th1', 'a2'], potential=np.array([
        [1., 1., 1., 0.],
        [0., 0., 0., 1.]
    ]))

    g.factor(['th1', 'a3'], potential=np.array([
        [0., 0., 0., 1.],
        [1., 1., 1., 0.]
    ]))

    # Cluster 2
    g.factor(['th2', 'a4'], potential=np.array([
        [1., 1., 1., 0.],
        [0., 0., 0., 1.]
    ]))

    g.factor(['th2', 'a5'], potential=np.array([
        [0., 0., 0., 1.],
        [1., 1., 1., 0.]
    ]))

    # Add factors between tracks a and measurement b
    g.factor(['a1', 'b1'], potential=np.array([
        [1., 0., 1., 1., 1., 1.],
        [0., 1., 0., 0., 0., 0.],
        [1., 0., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1., 1.],
    ]))

    # Add factors between tracks a and measurement b
    g.factor(['a2', 'b1'], potential=np.array([
        [1., 1., 0., 1., 1., 1.],
        [0., 0., 1., 0., 0., 0.],
        [1., 1., 0., 1., 1., 1.],
        [1., 1., 0., 1., 1., 1.],
    ]))


    # Add factors between tracks a and measurement b
    g.factor(['a3', 'b1'], potential=np.array([
        [1., 1., 1., 0., 1., 1.],
        [0., 0., 0., 1., 0., 0.],
        [1., 1., 1., 0., 1., 1.],
        [1., 1., 1., 0., 1., 1.],
    ]))

    # Add factors between tracks a and measurement b
    g.factor(['a4', 'b1'], potential=np.array([
        [1., 1., 1., 1., 0., 1.],
        [0., 0., 0., 0., 1., 0.],
        [1., 1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 0., 1.],
    ]))

    # Add factors between tracks a and measurement b
    g.factor(['a5', 'b1'], potential=np.array([
        [1., 1., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 1.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 0.],
    ]))


    # Add factors between tracks a and measurement b
    g.factor(['a1', 'b2'], potential=np.array([
        [1., 0., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1., 1.],
        [0., 1., 0., 0., 0., 0.],
        [1., 0., 1., 1., 1., 1.],
    ]))

    # Add factors between tracks a and measurement b
    g.factor(['a2', 'b2'], potential=np.array([
        [1., 1., 0., 1., 1., 1.],
        [1., 1., 0., 1., 1., 1.],
        [0., 0., 1., 0., 0., 0.],
        [1., 1., 0., 1., 1., 1.],
    ]))


    # Add factors between tracks a and measurement b
    g.factor(['a3', 'b2'], potential=np.array([
        [1., 1., 1., 0., 1., 1.],
        [1., 1., 1., 0., 1., 1.],
        [0., 0., 0., 1., 0., 0.],
        [1., 1., 1., 0., 1., 1.],
    ]))

    # Add factors between tracks a and measurement b
    g.factor(['a4', 'b2'], potential=np.array([
        [1., 1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 0., 1.],
        [0., 0., 0., 0., 1., 0.],
        [1., 1., 1., 1., 0., 1.],
    ]))

    # Add factors between tracks a and measurement b
    g.factor(['a5', 'b2'], potential=np.array([
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0., 1.],
        [1., 1., 1., 1., 1., 0.],
    ]))

    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ])

    n, mpn = R.shape
    m = mpn - n

    R_LC = np.hstack((np.diag(R[:, m:])[:,None], R[:,:m]))

    ms = np.exp(R_LC[:,0])
    ls = np.exp(R_LC[:,1:])

    # Add prior factors on tracks
    n, m = ls.shape
    for i in range(n):
        print(np.array([ms[i], *ls[i], 1.0]))
        g.factor([f'a{i+1}'], potential=np.array([ms[i], *ls[i], 1.0]))

    # Run (loopy) belief propagation (LBP)
    iters, converged = g.lbp(normalize=True)
    print('LBP ran for %d iterations. Converged = %r' % (iters, converged))
    # print()

    # # Print out the final messages from LBP
    # g.print_messages()
    # print()

    # Print out the final marginals
    g.print_rv_marginals(normalize=True)
