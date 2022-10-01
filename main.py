import numpy as np
import factorgraph as fg
from marginal_association_Odin import exact_marginal, lbp_marginal
from time import time


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
    R = np.array([
        [4.78, -0.46, -np.inf, -np.inf],
        [5.37, -np.inf, -0.52, -np.inf],
        [6.58, -np.inf, -np.inf, -0.60]
    ])

    # # exp to convert log into actual probabilities. Is this properly normalized?? Does it need to??
    # l_11 = np.exp(R[0,0])
    # l_21 = np.exp(R[1,0])
    # l_31 = np.exp(R[2,0])

    # m_1 = np.exp(R[0, 1])
    # m_2 = np.exp(R[1, 2])
    # m_3 = np.exp(R[2, 3])

    # # Add prior factors on tracks
    # g.factor(['a1'], potential=np.array([m_1, l_11, 1.0]))
    # g.factor(['a2'], potential=np.array([m_2, l_21, 1.0]))
    # g.factor(['a3'], potential=np.array([m_3, l_31, 1.0]))

    # # Run (loopy) belief propagation (LBP)
    # iters, converged = g.lbp(normalize=True)
    # print('LBP ran for %d iterations. Converged = %r' % (iters, converged))
    # # print()

    # # # Print out the final messages from LBP
    # # g.print_messages()
    # # print()

    # # Print out the final marginals
    # g.print_rv_marginals(normalize=True)

    # def exact_marginal(llr: np.ndarray, do_cluster: bool = True, **kwargs) -> tuple[np.ndarray, np.ndarray, float]:

    # Reshape R into LC form
    n = R.shape[0]
    m = R.shape[1] - n
    R_LC = np.hstack((np.diag(R[:,m:])[:,None], R[:,:m]))

    prior_hypotheses = [
        (np.array([1, 2]), 0.5),
        (np.array([1, 3]), 0.5)
    ]
    start = time()
    stop = time()

    print(f"Spent {(stop - start)*1e3} ms")

    marginal_total = np.zeros((n, m + 1 + 1))

    all_tracks_idx = np.arange(n)

    conditioned_marginals = np.empty((n, m + 2))

    for tracks, p in prior_hypotheses:
        R_sub = R_LC[tracks-1, :]
        JPDAprobs, notTrackProb, loglikelihood = exact_marginal(R_sub, False)

        # We need to concatenate the JPDAprobs with all tracks and existence probs
        existing_tracks_idx = tracks - 1
        non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

        existing_probs = np.hstack((JPDAprobs, np.zeros((JPDAprobs.shape[0], 1))))
        nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

        conditioned_marginals[existing_tracks_idx] = existing_probs
        conditioned_marginals[non_existing_tracks_idx] =  nonexisting_probs

        marginal_total += conditioned_marginals*np.exp(loglikelihood)*p

    marginal_total = marginal_total / marginal_total.sum(axis=1).reshape(-1, 1)


    print(marginal_total)

    lbp_marginal_total = np.zeros((n, m + 1 + 1))

    conditioned_marginals = np.empty((n, m + 2))

    for tracks, p in prior_hypotheses:
        R_sub = R_LC[tracks-1, :]
        lbp_probs, notTrackProb = lbp_marginal(R_sub)

        # We need to concatenate the JPDAprobs with all tracks and existence probs
        existing_tracks_idx = tracks - 1
        non_existing_tracks_idx = np.delete(all_tracks_idx, existing_tracks_idx)

        existing_probs = np.hstack((lbp_probs, np.zeros((lbp_probs.shape[0], 1))))
        nonexisting_probs = np.hstack((np.zeros((len(non_existing_tracks_idx), m + 1)), np.ones((len(non_existing_tracks_idx), 1))))

        conditioned_marginals[existing_tracks_idx] = existing_probs
        conditioned_marginals[non_existing_tracks_idx] =  nonexisting_probs

        normalizing_constant = np.exp(R_sub[:, 1:]).sum(axis=0).prod()

        lbp_marginal_total += conditioned_marginals * normalizing_constant * p

    lbp_marginal_total = lbp_marginal_total / lbp_marginal_total.sum(axis=1).reshape(-1, 1)

    print(lbp_marginal_total)
