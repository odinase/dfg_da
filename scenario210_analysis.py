import numpy as np

from dfg_da.stats_logger import MatFileParser
import py_dfg_da
from ravens_parser_parallell_multicluster import merge_clusters

if __name__ == "__main__":
    path_many = "./data/scenario210/k210ic5Many_prior.mat"
    mat_file_many = MatFileParser(path_many, use_cpp=True)

    path_few = "./data/scenario210/k210ic5Few_prior.mat"
    mat_file_few = MatFileParser(path_few, use_cpp=True)

    # We want to compare the benefit of LBP between the few and many cases
    mat_file = mat_file_many
    R = np.asfortranarray(mat_file.reward_matrix_edmund)
    R_LC = np.asfortranarray(mat_file.reward_matrix_lc)
    prior_hypotheses_per_cluster_all = mat_file.prior_hypotheses_per_cluster
    prior_hypotheses_per_cluster = py_dfg_da.hypothesis.HypothesesList([ph for ph in prior_hypotheses_per_cluster_all[4:6]])
    tracks = set()
    for ph in prior_hypotheses_per_cluster:
        tracks |= ph.tracks()

    tracks = np.sort(np.fromiter(tracks, dtype=int))
    print(tracks)
    old2new = {old: new for old, new in zip(tracks, range(1, len(tracks) + 1))}
    for ph in prior_hypotheses_per_cluster:
        ph.reindex_tracks(old2new)
    
    # tracks = set()
    # for ph in prior_hypotheses_per_cluster:
    #     tracks |= ph.tracks()
    # tracks = np.sort(np.fromiter(tracks, dtype=int))
    # print(tracks)


    t_idx = tracks - 1
    n, mp1 = R_LC.shape
    m = mp1 - 1
    Rd = R[:, :m]
    Rmd = np.diag(R[:, m:])
    Rd = Rd[t_idx]
    Rmd_temp = np.full((len(t_idx), len(t_idx)), -1000.0)
    Rmd_temp[np.diag_indices(len(t_idx))] = Rmd[t_idx]
    print(Rmd[t_idx])
    gated_measurements = (Rd > -1000).any(axis=0)
    print(np.where(gated_measurements))
    R = np.hstack((Rd[:, gated_measurements], Rmd_temp))
    np.set_printoptions(suppress=True, linewidth=150, precision=6)
    R[R <= -1000] = -np.inf
    R = np.asfortranarray(R)
    print(R)

    mcmhlbp = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    mcmhlbp_marginals = mcmhlbp.track_association_marginals()  # Transpose to change marginals from column-wise to row-wise
    bethe_normalization_constant = mcmhlbp.bethe_pseudodual_normalization_constant()

    print(mcmhlbp_marginals.T)
    print(bethe_normalization_constant)

    assocLocal = mat_file_many.ws["assocLocal"]
    prior_hypotheses_per_cluster_posterior = merge_clusters(assocLocal, prior_hypotheses_per_cluster_all)
    prior_hypotheses_per_cluster_posterior = py_dfg_da.hypothesis.HypothesesList([prior_hypotheses_per_cluster_posterior[4]])
    for ph in prior_hypotheses_per_cluster_posterior:
        ph.reindex_tracks(old2new)
    
    exact_marginals, exact_normalization_constant = py_dfg_da.hypothesis.association_marginal_posteriors_normalization_constant_multicluster(R, prior_hypotheses_per_cluster_posterior)
    # exact_marginals, exact_normalization_constant = py_dfg_da.factor_graph.exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)

    print(exact_marginals.T)
    print(exact_normalization_constant)

    print(bethe_normalization_constant / exact_normalization_constant)