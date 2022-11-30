import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as phs
import dfg_da.stats_logger as sl
from dfg_da.marginal_association_Odin import ExplicitHypothesisEnumerationError

import matplotlib.pyplot as plt
from glob import glob
from typing import List
import numpy as np


from multiprocessing import Pool
import itertools
import time
import tqdm
from pathlib import Path


OUTPUT_PATH_BASE = "./ravens_output"


def plot_survival_function(axes, max_errors, abs_errors, misdetection_errors, detection_errors, nonexistence_errors, label="_"):
    assert len(axes) == 5

    max_errors = np.sort(np.hstack(max_errors))
    abs_errors = np.sort(np.hstack(abs_errors))
    misdetection_errors = np.sort(np.hstack(misdetection_errors))
    detection_errors = np.sort(np.hstack(detection_errors))
    nonexistence_errors = np.sort(np.hstack(nonexistence_errors))

    errors = [max_errors,
        abs_errors,
        misdetection_errors,
        detection_errors,
        nonexistence_errors]

    titles = [
        "max_errors",
        "abs_errors",
        "misdetection_errors",
        "detection_errors",
        "nonexistence_errors"
    ]

    for ax, error, title in zip(axes, errors, titles):
        ax.set_title(title)
        steps = np.linspace(1.0, 0.0, len(error))
        ax.step(error, steps, label=label)
        # ax.set_yscale('symlog')
        ax.set_xscale('symlog', linthresh=1e-15)
        # ax.semilogx()
        ax.semilogy()
        # ax.loglog()
        if label != "_":
            ax.legend()


def loop_func(pmbm_file):
    mat_data: sl.MatFileParser = sl.MatFileParser(pmbm_file)

    R_LC = mat_data.reward_matrix_lc
    prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

    pmbm_file_path = Path(pmbm_file)
    pmbm_filename = pmbm_file_path.name[:-len(pmbm_file_path.suffix)]

    path = Path(f"{OUTPUT_PATH_BASE}/{pmbm_filename}")
    path.mkdir(parents=True, exist_ok=True)
    
    num_clusters = len(prior_hypotheses_per_cluster)

    for k, prior_hypotheses in enumerate(prior_hypotheses_per_cluster):
        cluster_stats = sl.ClusterData()
        save_path = f"{path}/cluster{str(k).zfill(int(np.ceil(np.log10(num_clusters))))}"
        try:
            exact_marginals, (exact_normalization_constants,) = exact_marginal_computer(R_LC, prior_hypotheses)
        except ExplicitHypothesisEnumerationError:
            cluster_stats.skipped = True
            cluster_stats.save_data(save_path)
            continue

        lbp_williams_marginals, (approx_normalization_constants, williams_iters, lbp_williams_marginals_exact_norm_const) = approx_marginal_computers["lbp_williams"](
            R_LC, prior_hypotheses, 
            own_normalizing_constants=exact_normalization_constants
        )
        lbp_mh_marginals, (msg_iters, tot_iters) = approx_marginal_computers["lbp_mh"](R_LC, prior_hypotheses)

        exact_stats = sl.ExactStats(marginals=sl.Marginals(exact_marginals), normalization_constants=exact_normalization_constants)
        lbp_stats = sl.LBPStats(msg_iters, tot_iters, sl.Marginals(lbp_mh_marginals))
        williams_stats = sl.WilliamsStats(
            lbp_iters=williams_iters,
            marginals=sl.Marginals(lbp_williams_marginals),
            marginals_exact_normalization_constant=sl.Marginals(lbp_williams_marginals_exact_norm_const),
            normalization_constants=approx_normalization_constants
        )

        cluster_stats.lbp_stats = lbp_stats
        cluster_stats.williams_stats = williams_stats
        cluster_stats.exact_stats = exact_stats

        cluster_stats.skipped = False

        tracks_in_cluster = frozenset(tt for t,_ in prior_hypotheses for tt in t)
        cluster_stats.cardinality = len(tracks_in_cluster)

        cluster_stats.tracks = tracks_in_cluster
        cluster_stats.num_hypotheses = len(prior_hypotheses)

        cluster_stats.save_data(save_path)


if __name__ == "__main__":
    # Make list over all files
    path = "./data/pmbm_output_files"

    pmbm_files = glob(path + "/*.mat")
    # pmbm_files = pmbm_files[:500]
    # pmbm_files = ["/home/odinase/prog/cpp/dfg_da/data/at612/priorLikelihood612.mat"]

    exact_marginal_computer = mc.ExactMarginals()
    approx_marginal_computers = {
        "lbp_williams": mc.LBPMarginalsByTotalProb(),
        "lbp_mh": mc.LBPMarginalsFullAssociation()
    }

    start = time.time()
    print("Starting pool")
    with Pool() as p:
        p.map(loop_func, pmbm_files)

    print("Pools done")
