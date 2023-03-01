import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as phs
import dfg_da.stats_logger as sl
from dfg_da.marginal_association_Odin import ExplicitHypothesisEnumerationError

import matplotlib.pyplot as plt
from glob import glob
from typing import List
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool, Lock
import time
from pathlib import Path

import py_dfg_da

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
    n, mp1 = R_LC.shape
    m = mp1 - 1
    prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

    pmbm_file_path = Path(pmbm_file)
    pmbm_filename = pmbm_file_path.name[:-len(pmbm_file_path.suffix)]

    path = Path(f"{OUTPUT_PATH_BASE}")
    path.mkdir(parents=True, exist_ok=True)
    
    save_path = f"{path}/{pmbm_filename}_stats"

    num_clusters = len(prior_hypotheses_per_cluster)
    if num_clusters == 0:
        return

    mcmhlbp = py_dfg_da.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    mcmhlbp_marginals = mcmhlbp.track_association_marginals()
    bethe_normalization_constant = mcmhlbp.bethe_pseudodual_normalization_constant()
    assocLocal = mat_data.ws["assocLocal"]
    prior_hypotheses_per_cluster_posterior = merge_clusters(assocLocal, prior_hypotheses_per_cluster)
    
    exact_marginals = np.empty((0, 2 + m))
    exact_normalization_constant = 1.0
    exact_computation_error = False

    try:
        exact_marginals, exact_normalization_constant = py_dfg_da.hypothesis.association_marginal_posteriors_normalization_constant_multicluster(R, prior_hypotheses_per_cluster_posterior)
    except ValueError:
        exact_computation_error = True



    cluster_data = sl.MulticlusterData(
        exact_computation_error=exact_computation_error,
        exact_marginals=sl.Marginals(exact_marginals.T),
        exact_normalization_constant=exact_normalization_constant,
        mhlbp_marginals=sl.Marginals(mcmhlbp_marginals.T),
        bethe_normalization_constant=bethe_normalization_constant
    )

    cluster_data.save_data(save_path)


if __name__ == "__main__":
    # Super hacky way to hopefully avoid hypothesis enumeration explosion
    illegal_files = [
        'priorLikelihood_iMC5k312.mat',
        'priorLikelihood_iMC6k527.mat',
        'priorLikelihood_iMC5k299.mat',
        'priorLikelihood_iMC6k280.mat',
        'priorLikelihood_iMC6k130.mat',
        'priorLikelihood_iMC2k970.mat',
        'priorLikelihood_iMC3k867.mat',
        'priorLikelihood_iMC10k302.mat',
        'priorLikelihood_iMC3k621.mat',
        'priorLikelihood_iMC2k1090.mat',
        'priorLikelihood_iMC5k477.mat',
        'priorLikelihood_iMC3k58.mat',
        'priorLikelihood_iMC4k549.mat',
        'priorLikelihood_iMC4k866.mat',
        'priorLikelihood_iMC3k816.mat',
        'priorLikelihood_iMC6k208.mat',
        'priorLikelihood_iMC4k560.mat',
        'priorLikelihood_iMC1k769.mat',
        'priorLikelihood_iMC7k105.mat',
        'priorLikelihood_iMC5k532.mat',
        'priorLikelihood_iMC1k1113.mat',
        'priorLikelihood_iMC2k3.mat',
        'priorLikelihood_iMC6k138.mat',
        'priorLikelihood_iMC6k316.mat',
        'priorLikelihood_iMC5k294.mat',
        'priorLikelihood_iMC6k307.mat',
        'priorLikelihood_iMC4k770.mat',
        'priorLikelihood_iMC2k234.mat',
        'priorLikelihood_iMC6k151.mat'
    ]

    pmbm_files = [pmbm_file for pmbm_file in glob(PMBM_DATA_PATH + "/*.mat") if not Path(pmbm_file).name in illegal_files]
    # pmbm_files = glob(PMBM_DATA_PATH + "/*.mat")


    pmbm_files = sorted(pmbm_files)
    num_files = len(pmbm_files)
    # pmbm_files = ["/home/odinase/prog/cpp/dfg_da/data/at612/priorLikelihood612.mat"]

    exact_marginal_computer = mc.ExactMarginals()
    approx_marginal_computers = {
        "lbp_williams": mc.LBPMarginalsByTotalProb(),
        "lbp_mh": mc.LBPMarginalsFullAssociation(),
        "lbp_bethe": mc.LBPMarginalsByTotalProbBethe()
    }

    lock = Lock()

    print("Starting pool")
    start = time.time()
    # for pmbm_file in tqdm(pmbm_files):
    #     loop_func(pmbm_file)
    with Pool() as p:
        p.map(loop_func, pmbm_files)
    stop = time.time()
    print("Pools done")
    duration_s = stop - start
    duration_min = duration_s / 60.0
    duration_h = duration_min / 60.0
    print(f"Spent {duration_s:.3f} s = {duration_min:.3f} min = {duration_h:.3f} h")
