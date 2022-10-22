import numpy as np
from scipy.special import logsumexp

if __name__ == "__main__":
    prior_hypotheses = [
        ([1, 2], 0.5),
        ([1, 3], 0.5)
    ]

    n = 3
    m = 1

    phi = np.array([h[1] for h in prior_hypotheses])
    num_hypotheses = len(prior_hypotheses)
    t2h_idx = np.array([
        [t+1 in hypo[0] for hypo in prior_hypotheses] for t in range(n)
    ])
    print(f"t2h:\n{t2h_idx}")
    h2t_idx = t2h_idx.T
    print(f"h2t:\n{h2t_idx}")

    rho = np.array([0.25, 0.3333, 1.4])

    sigma_true = np.empty(n)

    sigma_true[0] = 0
    sigma_true[1] = (
        (phi[1]*rho[0]*rho[2]) /
        (phi[0]*rho[0])
    )
    sigma_true[2] = (
        (phi[0]*rho[0]*rho[1]) /
        (phi[1]*rho[0])
    )

    rho_bc = np.broadcast_to(rho, (num_hypotheses, n))

    # For each element in sigma, which is for each track, gather the phis that the track doesnt appear in
    sigma_test = np.array([
        np.sum( np.multiply( phi[~t2h_idx[t]], [np.prod(rho_bc[h][h2t_idx[h]]) for h in ~t2h_idx[t]]) ) / # Sum over hypos without track
        np.sum( np.multiply( phi[t2h_idx[t]], [np.prod(rho_bc[h][h2t_idx[h]]) for h in t2h_idx[t]]) / rho[t] ) # Sum over hypos with track
        if (~t2h_idx[t]).any() else 0
        for t in range(n)
    ])

    sigma_test2 = np.empty(2)
    for t in range(n):
        print(f"Track {t+1} exists in hypotheses {t2h_idx[t]}")
        print(f"Track {t+1} does not exist in hypotheses {~t2h_idx[t]}")
        if (~t2h_idx[t]).any():
            rho_prod_numerator = 1.0
            for h in ~t2h_idx[t]:
                print(f"h2t_idx[h]: {h2t_idx[h]}")
                print(f"rho_bc[h]: {rho_bc[h]}")
                print(f"rho_bc[h][h2t_idx[h]]: {rho_bc[h][h2t_idx[h]]}")
        else:
            sigma_test2[t] = 0

    print(f"sigma true: {sigma_true}")
    print(f"sigma test: {sigma_test}")