import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as phs
import dfg_da.stats_logger as sl
from dfg_da.marginal_association_Odin import ExplicitHypothesisEnumerationError

import matplotlib.pyplot as plt
from glob import glob
from typing import List
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool
import time
from pathlib import Path

import py_dfg_da

OUTPUT_PATH_BASE = "./ravens_output_multicluster"
PMBM_DATA_PATH = "./data/pmbm_output_files"


def loop_func(pmbm_file):
    mat_data: sl.MatFileParser = sl.MatFileParser(pmbm_file, use_cpp=True)

    R = np.asfortranarray(mat_data.reward_matrix_edmund)
    prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

    pmbm_file_path = Path(pmbm_file)
    pmbm_filename = pmbm_file_path.name[:-len(pmbm_file_path.suffix)]

    path = Path(f"{OUTPUT_PATH_BASE}")
    path.mkdir(parents=True, exist_ok=True)
    
    save_path = f"{path}/{pmbm_filename}_stats"

    num_clusters = len(prior_hypotheses_per_cluster)
    if num_clusters == 0:
        return

    print(R)
    print(prior_hypotheses_per_cluster)
    print(len(prior_hypotheses_per_cluster))
    mcmhlbp = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    print("LBP!")
    mcmhlbp_marginals = mcmhlbp.track_association_marginals()
    print("Marginals!")
    bethe_normalization_constant = mcmhlbp.bethe_pseudodual_normalization_constant()
    print("Bethe!")
    # exact_margs, Z = py_dfg_da.hypothesis.association_marginal_posteriors_normalization_constant(R, merged_hypos)
    exact_marginals, exact_normalization_constant = py_dfg_da.factor_graph.exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)
    print("Exact")

    cluster_data = sl.MulticlusterData(
        exact_marginals=exact_marginals.T,
        exact_normalization_constant=exact_normalization_constant,
        mhlbp_marginals=mcmhlbp_marginals.T,
        bethe_normalization_constant=bethe_normalization_constant
    )

    cluster_data.save_data(save_path)



if __name__ == "__main__":
    prior_hypotheses_per_cluster: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
        py_dfg_da.hypothesis.Hypotheses([
            py_dfg_da.hypothesis.Hypothesis([1, 2], np.log(0.5)),
            py_dfg_da.hypothesis.Hypothesis([1, 3], np.log(0.5))
        ]),
        py_dfg_da.hypothesis.Hypotheses([
            py_dfg_da.hypothesis.Hypothesis([4], np.log(0.5)),
            py_dfg_da.hypothesis.Hypothesis([5], np.log(0.5))
        ])
    ])

    assocLocal = np.array([
        [1, 1],
        [1, 0]
    ])

    num_posterior_clusters = np.sum(assocLocal[1])
    # First build master array
    prior_hypotheses_per_cluster_posterior: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
        h for k, h in enumerate(prior_hypotheses_per_cluster) if assocLocal[1, k]
    ])
    # master_idxs = np.where(assocLocal[1])[0]
    # print(master_idxs)

    # assocLocal =np.array([
    #  [1,     1,     3,     4,     4,     6,     7,     8,     1,     1,    11,     1,    13],
    #  [1,     0,     1,     1,     0,     1,     1,     1,     0,     0,     1,     0,     1]
    # ])
    master_idxs = np.cumsum(assocLocal[1]) - 1

    assert len(prior_hypotheses_per_cluster_posterior) == num_posterior_clusters
    
    for c, (master, is_master) in enumerate(assocLocal.T):
        if is_master:
            continue
            
        # We already have the masters, merge clusters
        hs = prior_hypotheses_per_cluster[c]
        prior_hypotheses_per_cluster_posterior[master_idxs[master]] = prior_hypotheses_per_cluster_posterior[master_idxs[master]].combine(hs)

    for k, hh in enumerate(prior_hypotheses_per_cluster_posterior):
        print(f"Cluster {k+1}")
        for cc in range(len(hh)):
            print(hh[cc].tracks())
            print(hh[cc].probability())

    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    exact_margs, Z = py_dfg_da.hypothesis.association_marginal_posteriors_normalization_constant(R, prior_hypotheses_per_cluster_posterior[0])
    print(exact_margs)
    print(Z)