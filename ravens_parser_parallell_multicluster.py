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
    assocLocal = mat_data.ws["assocLocal"]    
    explicit_hypothesis_enumeration_error = False

    try:
        exact_output = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)
    except ExplicitHypothesisEnumerationError:
        explicit_hypothesis_enumeration_error = True


    cluster_data = sl.MulticlusterData(
        mhlbp_output=mcmhlbp,
        exact_output=exact_output,
        explicit_hypothesis_enumeration_error=explicit_hypothesis_enumeration_error
    )

    cluster_data.save_data(save_path)


if __name__ == "__main__":
    pmbm_files = glob(PMBM_DATA_PATH + "/*.mat")

    pmbm_files = sorted(pmbm_files)
    num_files = len(pmbm_files)
    exact_computer = mc.MulticlusterExact()

    if len(pmbm_files) != 10_000:
        raise ValueError()

    print(f"Computing {len(pmbm_files)} files...")

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
