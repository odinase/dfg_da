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
    
    g.rv('b', 6)

    g.factor(['th1'], potential=np.array([0.5, 0.5]))
    g.factor(['th2'], potential=np.array([0.5, 0.5]))

    # Cluster 1
    # Add factors between theta and tracks a
    g.factor(['th', 'a1'], potential=np.array([
        [1., 1., 0.],
        [1., 1., 0.]
    ]))

    g.factor(['th', 'a2'], potential=np.array([
        [1., 1., 0.],
        [0., 0., 1.]
    ]))

    g.factor(['th', 'a3'], potential=np.array([
        [0., 0., 1.],
        [1., 1., 0.]
    ]))

    # Add factors between tracks a and measurement b
    g.factor(['a1', 'b'], potential=np.array([
        [1., 0., 1., 1.],
        [0., 1., 0., 0.],
        [1., 0., 1., 1.],
    ]))

    g.factor(['a2', 'b'], potential=np.array([
        [1., 1., 0., 1.],
        [0., 0., 1., 0.],
        [1., 1., 0., 1.],
    ]))

    g.factor(['a3', 'b'], potential=np.array([
        [1., 1., 1., 0.],
        [0., 0., 0., 1.],
        [1., 1., 1., 0.],
    ]))

    # Assume two clusters

    # Add prior factors on tracks
    g.factor(['a1'], potential=np.array([m_1, l_11, 1.0]))
    g.factor(['a2'], potential=np.array([m_2, l_21, 1.0]))
    g.factor(['a3'], potential=np.array([m_3, l_31, 1.0]))

    # Run (loopy) belief propagation (LBP)
    iters, converged = g.lbp(normalize=True)
    print('LBP ran for %d iterations. Converged = %r' % (iters, converged))
    # print()

    # # Print out the final messages from LBP
    # g.print_messages()
    # print()

    # Print out the final marginals
    g.print_rv_marginals(normalize=True)
