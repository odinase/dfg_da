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

OUTPUT_PATH_BASE = "./ravens_output"
PMBM_DATA_PATH = "./data/pmbm_output_files"

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
    if num_clusters == 0:
        # There are no clusters in this timestep for some reason, but that is still valuable information
        cluster_stats = sl.ClusterData()
        save_path = f"{path}/empty_cluster"
        cluster_stats.cardinality = 0
        cluster_stats.tracks = frozenset()
        cluster_stats.num_hypotheses = 0

        cluster_stats.save_data(save_path)
        return

    digits = int(np.ceil(np.log10(num_clusters)))

    for k, prior_hypotheses in enumerate(prior_hypotheses_per_cluster):
        cluster_stats = sl.ClusterData()
        save_path = f"{path}/cluster{str(k).zfill(digits)}"
        tracks_in_cluster = frozenset(tt for t,_ in prior_hypotheses for tt in t)
        t_idx = np.sort(np.fromiter(tracks_in_cluster, dtype=int)) - 1

        try:
            exact_marginals, (exact_normalization_constants,) = exact_marginal_computer(R_LC, prior_hypotheses)
            exact_stats = sl.ExactStats(
                marginals=sl.Marginals(exact_marginals[t_idx, :]),
                normalization_constants=exact_normalization_constants
            )
            cluster_stats.exact_stats = exact_stats
            cluster_stats.explicit_hypothesis_enumeration_error = False
        except ExplicitHypothesisEnumerationError:
            cluster_stats.explicit_hypothesis_enumeration_error = True
            exact_normalization_constants = None

        lbp_williams_marginals, (approx_normalization_constants, williams_iters, williams_converged_list, lbp_williams_marginals_exact_norm_const) = approx_marginal_computers["lbp_williams"](
            R_LC, prior_hypotheses, 
            own_normalizing_constants=exact_normalization_constants
        )
        if lbp_williams_marginals_exact_norm_const is not None:
            lbp_williams_marginals_exact_norm_const = sl.Marginals(lbp_williams_marginals_exact_norm_const[t_idx, :])

        lbp_mh_marginals, (tot_iters, msg_iters, lbp_converged) = approx_marginal_computers["lbp_mh"](R_LC, prior_hypotheses)

        lbp_marginals_bethe, bethe_constants_odin, bethe_constants_lc = approx_marginal_computers["lbp_bethe"](R_LC, prior_hypotheses)

        lbp_stats = sl.LBPStats(
            num_iters_msg=msg_iters,
            num_iters=tot_iters,
            marginals=sl.Marginals(lbp_mh_marginals[t_idx, :]),
            converged=lbp_converged
        )
        williams_stats = sl.WilliamsStats(
            lbp_iters=williams_iters,
            marginals=sl.Marginals(lbp_williams_marginals[t_idx, :]),
            marginals_exact_normalization_constant=lbp_williams_marginals_exact_norm_const,
            normalization_constants=approx_normalization_constants,
            converged_list=williams_converged_list
        )

        bethe_stats = sl.BetheStats(
            marginals=sl.Marginals(lbp_marginals_bethe[t_idx, :]),
            normalization_constants_odin=bethe_constants_odin,
            normalization_constants_lc=bethe_constants_lc
        )

        cluster_stats.lbp_stats = lbp_stats
        cluster_stats.williams_stats = williams_stats
        cluster_stats.bethe_stats = bethe_stats

        cluster_stats.cardinality = len(tracks_in_cluster)

        cluster_stats.tracks = tracks_in_cluster
        cluster_stats.num_hypotheses = len(prior_hypotheses)

        cluster_stats.save_data(save_path)


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
