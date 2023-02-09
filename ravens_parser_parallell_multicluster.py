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

    R = mat_data.reward_matrix_edmund
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
    exact_marginals, exact_normalization_constant = py_dfg_da.factor_graph.exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster)

    cluster_data = sl.MulticlusterData(
        exact_marginals=exact_marginals.T,
        exact_normalization_constant=exact_normalization_constant,
        mhlbp_marginals=mcmhlbp_marginals.T,
        bethe_normalization_constant=bethe_normalization_constant
    )

    cluster_data.save_data(save_path)



if __name__ == "__main__":
    # Make list over all files

    pmbm_files = glob(PMBM_DATA_PATH + "/*.mat")
    # pmbm_files = pmbm_files[:500]
    # pmbm_files = ["/home/odinase/prog/cpp/dfg_da/data/at612/priorLikelihood612.mat"]

    exact_marginal_computer = mc.ExactMarginals()
    approx_marginal_computers = {
        "lbp_williams": mc.LBPMarginalsByTotalProb(),
        "lbp_mh": mc.LBPMarginalsFullAssociation(),
        "lbp_bethe": mc.LBPMarginalsByTotalProbBethe()
    }

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
