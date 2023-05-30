import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as phs
import dfg_da.cluster_bayes_tree as cbt
import dfg_da.stats_logger as sl
from dfg_da.marginal_association_Odin import ExplicitHypothesisEnumerationError
from dfg_da.cluster_conditioning_lbp import MulticlusterEfficientMarginalsLBP, MulticlusterConditionendLBPOutput

import matplotlib.pyplot as plt
from glob import glob
from typing import List
import numpy as np
from tqdm import tqdm


from copy import deepcopy

from multiprocessing import Pool
import multiprocessing
import time
from pathlib import Path

import py_dfg_da

def warning_handler(pmbm_file, function_flags):
    with open(f'./warnings/{pmbm_file}.log', 'a') as f:
        for function_name, flag in function_flags:
            f.write(f"{function_name}: {flag}\n")


OUTPUT_PATH_BASE = "./ravens_output_multicluster"
PMBM_DATA_PATH = "./data/pmbm_output_files"

NUM_DONE = 0


def merge_clusters(assocLocal, prior_hypotheses_per_cluster):
    num_posterior_clusters = np.sum(assocLocal[1])
    # First build master array
    prior_hypotheses_per_cluster_posterior: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
        h for k, h in enumerate(prior_hypotheses_per_cluster) if assocLocal[1, k]
    ])
    master_idxs = np.cumsum(assocLocal[1]) - 1

    assert len(prior_hypotheses_per_cluster_posterior) == num_posterior_clusters
    
    for c, (master, is_master) in enumerate(assocLocal.T):
        if is_master:
            continue

        # We already have the masters, merge clusters
        hs = prior_hypotheses_per_cluster[c]
        prior_hypotheses_per_cluster_posterior[master_idxs[master]] = prior_hypotheses_per_cluster_posterior[master_idxs[master]].combine(hs)

    return prior_hypotheses_per_cluster_posterior


def loop_func(pmbm_file):
    mat_data: sl.MatFileParser = sl.MatFileParser(pmbm_file, use_cpp=True)

    R = np.asfortranarray(mat_data.reward_matrix_edmund)
    R_LC = np.asfortranarray(mat_data.reward_matrix_lc)
    prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

    pmbm_file_path = Path(pmbm_file)
    pmbm_filename = pmbm_file_path.name[:-len(pmbm_file_path.suffix)]

    path = Path(f"{OUTPUT_PATH_BASE}")
    path.mkdir(parents=True, exist_ok=True)

    save_path = f"{path}/{pmbm_filename}_stats"

    num_clusters = len(prior_hypotheses_per_cluster)
    if num_clusters == 0:
        return

    start = time.time()
    mcmhlbp = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    dur_mcmhlbp = time.time() - start
    assocLocal = mat_data.ws["assocLocal"].copy()
    explicit_hypothesis_enumeration_error = False
    exact_output = None

    try:
        start = time.time()
        exact_output: mc.MulticlusterExactOutput = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal.copy())
        dur_exact = time.time() - start
        exact_output.runtime = dur_exact
        exact_output.theta_posteriors = exact_output.compute_theta_posteriors()
    except ExplicitHypothesisEnumerationError:
        explicit_hypothesis_enumeration_error = True

    mc_bethe = MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=deepcopy(prior_hypotheses_per_cluster), assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbBethe())
    mc_bethe_output = mc_bethe.compute_marginals_likelihood()

    mc_mhlbp = MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=deepcopy(prior_hypotheses_per_cluster), assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsFullAssociationCPP())
    mc_mhlbp_output = mc_mhlbp.compute_marginals_likelihood()

    mc_phd = MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=deepcopy(prior_hypotheses_per_cluster), assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbPHD(), cluster_links=mc_bethe.cluster_links)
    mc_phd_output = mc_phd.compute_marginals_likelihood()


    cluster_data = sl.MulticlusterData(
        mcmhlbp_output=sl.MulticlusterApproximateOutput(
            approx_marginals=mcmhlbp.track_association_marginals().T,
            approx_normalization_constant=mcmhlbp.bethe_pseudodual_normalization_constant(),
            approx_theta_posteriors=mcmhlbp.hypotheses_marginals(),
            full_output=mcmhlbp,
            runtime=dur_mcmhlbp
        ),
        mc_phd_output=mc_phd_output,
        mc_bethe_output=mc_bethe_output,
        mc_mhlbp_output=mc_mhlbp_output,
        exact_output=exact_output,
        explicit_hypothesis_enumeration_error=explicit_hypothesis_enumeration_error
    )

    cluster_data.save_data(save_path)


if __name__ == "__main__":
    pmbm_files = glob(PMBM_DATA_PATH + "/*.mat")

    pmbm_files = sorted(pmbm_files)
    num_files = len(pmbm_files)
    exact_computer = mc.MulticlusterExactEHM2()

    if len(pmbm_files) != 10_000:
        raise ValueError()

    pmbm_files = pmbm_files[:1000]
    # pmbm_files = ["./data/pmbm_output_files/priorLikelihood_iMC10k100.mat"]

    print(f"Computing {len(pmbm_files)} files...")

    num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    print("Starting pool")
    start = time.time()
    with tqdm(total=len(pmbm_files)) as pbar:
        for i, result in enumerate(pool.imap_unordered(loop_func, pmbm_files)):
            pbar.update(1)
            pbar.set_description(f"Progress: {i+1}/{len(pmbm_files)}, {(i+1)/len(pmbm_files)*100.0:.2f}%")
    stop = time.time()
    print("Pools done")
    duration_s = stop - start
    duration_min = duration_s / 60.0
    duration_h = duration_min / 60.0
    print(f"Spent {duration_s:.3f} s = {duration_min:.3f} min = {duration_h:.3f} h")
