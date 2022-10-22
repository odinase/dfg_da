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
    # print(f"t2h:\n{t2h_idx}")
    t2noth_idx = ~t2h_idx
    h2t_idx = t2h_idx.T
    # print(f"h2t:\n{h2t_idx}")

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
    # sigma_test = np.array([
    #     np.sum( np.multiply( phi[~t2h_idx[t]], [np.prod(rho[h]) for h in h2t_idx[~t2h_idx[t]]]) ) / # Sum over hypos without track
    #     np.sum( np.multiply( phi[t2h_idx[t]], [np.prod(rho[h]) / rho[t] for h in h2t_idx[t2h_idx[t]]]) ) # Sum over hypos with track
    #     if (~t2h_idx[t]).any() else 0
    #     for t in range(n)
    # ])

    
    (rho * h2t_idx + t2noth_idx.T).prod(axis=1)
    def compute_sigma(rho): return np.array([
        np.sum( np.multiply( phi[t2noth_idx[t]], [np.prod(rho[h]) for h in h2t_idx[t2noth_idx[t]]]) ) / # Sum over hypos without track
        np.sum( np.multiply( phi[t2h_idx[t]], [np.prod(rho[h]) / rho[t] for h in h2t_idx[t2h_idx[t]]]) ) # Sum over hypos with track
        if (t2noth_idx[t]).any() else 0 # Make sure the sum in the numerator is nonempty
        for t in range(n)
    ])

    # def compute_sigma(rho): return np.array([
    #     np.sum( np.multiply( phi[t2noth_idx[t]], [np.prod(rho[h]) for h in h2t_idx[t2noth_idx[t]]]) ) / # Sum over hypos without track
    #     np.sum( np.multiply( phi[t2h_idx[t]], [np.prod(rho[h]) / rho[t] for h in h2t_idx[t2h_idx[t]]]) ) # Sum over hypos with track
    #     if (t2noth_idx[t]).any() else 0 # Make sure the sum in the numerator is nonempty
    #     for t in range(n)
    # ])

    sigma_test = compute_sigma(rho)

    sigma_test2 = np.empty(n)
    for t in range(n):
        # print(f"Track {t+1} exists in hypotheses {np.where(t2h_idx[t])[0] + 1}")
        # print(f"Track {t+1} does not exist in hypotheses {np.where(~t2h_idx[t])[0] + 1}")
        i_in_th = t2h_idx[t]
        i_notin_th = ~i_in_th
        if (~t2h_idx[t]).any():
            # We have non-trivial numerator to compute. We can easily find all phis we need by indexing the hypotheses where the track does not exist
            phi_i_notin_th = phi[i_notin_th]
            # We need to compute a list of product for each term in the sum
            # First, find a list of lists containing what tracks are contained in the hypotheses not containing 
            # I.e., this is a list for each hypothesis that makes the sum, so use the information in each row to compute a single number (product) and put these numbers into a list
            tracks_in_hypotheses = h2t_idx[i_notin_th]
            print(tracks_in_hypotheses)
            list_of_prods = np.array([np.prod(rho[tracks]) for tracks in tracks_in_hypotheses])
            print(f"num: {list_of_prods}")
            numerator = np.sum(phi_i_notin_th * list_of_prods)

            # print(f"Numerator for track {t+1}: {numerator}")

            # Do the same for the denominator. Here we have use the trick of computing the product over all tracks contained in the hypothesis, which will include the track we are considering, so we divide by it afterwards
            phi_i_in_th = phi[i_in_th]
            tracks_in_hypotheses = h2t_idx[i_in_th]
            print(tracks_in_hypotheses)
            list_of_prods = np.array([np.prod(rho[tracks]) / rho[t] for tracks in tracks_in_hypotheses])
            print(f"denom: {list_of_prods}")

            denominator = np.sum(phi_i_notin_th * list_of_prods)
            sigma_test2[t] = numerator / denominator

            # print(f"Denominator for track {t+1}: {denominator}")
            # print(f"Sigma for track {t+1}: {sigma_test2[t]}")

        else:
            sigma_test2[t] = 0


    rho_c = rho * h2t_idx + t2noth_idx.T
    rho_nc = rho * ~h2t_idx + ~t2noth_idx.T
    print(rho_c)
    print(rho_c.prod(axis=1))
    print((rho_c.prod(axis=1) * phi * t2noth_idx).sum(axis=1))
    print(rho_nc)

    print(f"sigma true: {sigma_true}")
    print(f"sigma test: {sigma_test}")
    print(f"sigma test2: {sigma_test2}")
    
    a = (rho_c.prod(axis=1)*t2noth_idx*phi).sum(axis=1)
    b = (rho_c.prod(axis=1)*t2h_idx*phi).sum(axis=1) / rho
    