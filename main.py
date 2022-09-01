import numpy as np
import factorgraph as fg

if __name__ == "__main__":
    # Make an empty graph
    g = fg.Graph()

    # Add discrete random variables (RVs)
    g.rv('th', 2)
    g.rv('a1', 3)
    g.rv('a2', 3)
    g.rv('a3', 3)
    g.rv('b', 4)

    g.factor(['th'], potential=np.array([0.5, 0.5]))

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

    # Reward matrix
    R = np.array([
        [4.78, -0.46, -np.inf, -np.inf],
        [5.37, -np.inf, -0.52, -np.inf],
        [6.58, -np.inf, -np.inf, -0.60]
    ])

    # exp to convert log into actual probabilities. Is this properly normalized?? Does it need to??
    l_11 = np.exp(R[0,0])
    l_21 = np.exp(R[1,0])
    l_31 = np.exp(R[2,0])

    m_1 = np.exp(R[0, 1])
    m_2 = np.exp(R[1, 2])
    m_3 = np.exp(R[2, 3])

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