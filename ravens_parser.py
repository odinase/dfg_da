import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as phs
import dfg_da.stats_logger as sl

import matplotlib.pyplot as plt
from glob import glob
from typing import List
import numpy as np

import time


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



if __name__ == "__main__":
    # Make list over all files
    path = "./data/pmbm_output_files"

    pmbm_files = glob(path + "/*.mat")

    # pmbm_files = ["/home/odinase/prog/cpp/dfg_da/data/at612/priorLikelihood612.mat"]

    exact_marginal_computer = mc.ExactMarginalsWilliams()
    approx_marginal_computers = {
        "lbp_williams": mc.LBPMarginalsByTotalProb(),
        "lbp_mh": mc.LBPMarginalsFullAssociation()
    }

    lbp_mh_all_errors: List[sl.StatsLogger] = []
    lbp_williams_all_errors: List[sl.StatsLogger] = []

    approx_normalization_constants_all: List[np.ndarray] = []
    exact_normalization_constants_all: List[np.ndarray] = []

    start = time.time()
    for pmbm_file in pmbm_files[:200]:
        mat_data: sl.MatFileParser = sl.MatFileParser(pmbm_file)

        R_LC = mat_data.reward_matrix_lc
        prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

        for prior_hypotheses in prior_hypotheses_per_cluster:
            lbp_williams_marginals, approx_normalization_constants = approx_marginal_computers["lbp_williams"](R_LC, prior_hypotheses)
            lbp_mh_marginals, _ = approx_marginal_computers["lbp_mh"](R_LC, prior_hypotheses)
            exact_marginals, exact_normalization_constants = exact_marginal_computer(R_LC, prior_hypotheses)

            lbp_williams_errors = sl.MarginalsErrors(exact_marginals, lbp_williams_marginals)
            lbp_mh_errors = sl.MarginalsErrors(exact_marginals, lbp_mh_marginals)

            lbp_mh_all_errors.append(lbp_mh_errors)
            lbp_williams_all_errors.append(lbp_williams_errors)

            exact_normalization_constants_all.append(exact_normalization_constants)
            approx_normalization_constants_all.append(approx_normalization_constants)

    lbp_mh_all_errors: sl.MarginalsErrors = sl.MarginalsErrors.concatenate(lbp_mh_all_errors)
    lbp_williams_all_errors: sl.MarginalsErrors = sl.MarginalsErrors.concatenate(lbp_williams_all_errors)

    stop = time.time()
    print(f"Spent {stop - start} s")

    # fig, ax = plt.subplots()

    # ax.hist(lbp_mh_all_errors.max_errors)
    # ax.loglog()

    # fig_sf, axes_sf = plt.subplots(nrows=5, sharex=True)

    # fig_sf.suptitle("Survival function")

    # plot_survival_function(axes_sf, lbp_williams_all_errors.max_errors, lbp_williams_all_errors.abs_errors, lbp_williams_all_errors.misdetection_errors, lbp_williams_all_errors.detection_errors, lbp_williams_all_errors.nonexistence_errors, "Williams LBP with estimated normalization constant")
    # plot_survival_function(axes_sf, lbp_mh_all_errors.max_errors, lbp_mh_all_errors.abs_errors, lbp_mh_all_errors.misdetection_errors, lbp_mh_all_errors.detection_errors, lbp_mh_all_errors.nonexistence_errors, "LBP on full problem")

    # plt.show()

    path = "./pmbm_analysis_output"
    stats_logger = sl.StatsLogger()
    stats_logger.save_errors(path + "/lbp", lbp_mh_all_errors)
    stats_logger.save_errors(path + "/williams", lbp_williams_all_errors)