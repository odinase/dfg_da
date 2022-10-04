import factorgraph as fg
import numpy as np


def xnor(x, y): return not (x != y)


def dfg_from_reward_mat_hyp_prior(reward_mat, prior_hypotheses):
    dfg = fg.Graph()
    
    num_prior_hypotheses = len(prior_hypotheses)
    dfg.rv('th', num_prior_hypotheses)

    prior_probabilities = np.array([h[1] for h in prior_hypotheses])
    assert abs(prior_probabilities.sum() - 1) < 1e-6 and (0 <= prior_hypotheses).all() and (prior_probabilities <= 1.0).all()

    dfg.factor(['th'], potential=prior_probabilities)

    num_tracks, mpn = reward_mat.shape
    num_measurements = mpn - num_tracks

    ais = []

    for track in range(1, num_tracks + 1):
        ai = f'a{track}'
        dfg.rv(ai, 1 + num_measurements + 1)

        compatibility_table = []

        for prior_tracks, prior_prob in prior_hypotheses:
            contained_in_hypo = track in prior_tracks

            compatibility_table.append([xnor(meas < (num_measurements + 1), contained_in_hypo) for meas in range(num_measurements + 2)])


        dfg.factor(['th', ai], potential=np.array(compatibility_table))

        
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
    