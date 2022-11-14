import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as phs
import dfg_da.stats_logger as sl

import matplotlib.pyplot as plt
from glob import glob


if __name__ == "__main__":
    # Make list over all files
    path = "./data/pmbm_output_files"

    pmbm_files = glob(path + "/*.mat")

    exact_marginal_computer = mc.ExactMarginalsWilliams()
    approx_marginal_computers = {
        "lbp_williams": mc.LBPMarginalsByTotalProb(),
        "lbp_mh": mc.LBPMarginalsFullAssociation()
    }

    lbp_mh_all_errors = []
    lbp_williams_all_errors = []

    for pmbm_file in pmbm_files[:3]:
        mat_data = sl.MatFileParser(pmbm_file)

        R_LC = mat_data.reward_matrix_lc
        prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

        for prior_hypotheses in prior_hypotheses_per_cluster:
            lbp_williams_marginals, _ = approx_marginal_computers["lbp_williams"](R_LC, prior_hypotheses)
            lbp_mh_marginals, _ = approx_marginal_computers["lbp_mh"](R_LC, prior_hypotheses)
            exact_marginals, _ = exact_marginal_computer(R_LC, prior_hypotheses)

            lbp_williams_errors = sl.StatsLogger.MarginalsErrors(exact_marginals, lbp_williams_marginals)
            lbp_mh_errors = sl.StatsLogger.MarginalsErrors(exact_marginals, lbp_mh_marginals)

            lbp_mh_all_errors.append(lbp_mh_errors)
            lbp_williams_all_errors.append(lbp_williams_errors)

    lbp_mh_all_errors: sl.StatsLogger.MarginalsErrors = sl.StatsLogger.MarginalsErrors.concatenate(lbp_mh_all_errors)
    lbp_williams_all_errors: sl.StatsLogger.MarginalsErrors = sl.StatsLogger.MarginalsErrors.concatenate(lbp_williams_all_errors)

    fig, ax = plt.subplots()

    ax.hist(lbp_mh_all_errors.max_errors)
    ax.loglog()

    plt.show()
