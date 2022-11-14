import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as phs
import dfg_da.stats_logger as sl
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

    for pmbm_file in pmbm_files[:3]:
        mat_data = sl.MatFileParser(pmbm_file)

        R_LC = mat_data.reward_matrix_lc
        prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

        for prior_hypotheses in prior_hypotheses_per_cluster:
            lbp_williams_marginals, _ = approx_marginal_computers["lbp_williams"](R_LC, prior_hypotheses)
            lbp_mh_marginals, _ = approx_marginal_computers["lbp_mh"](R_LC, prior_hypotheses)
            exact_marginals, _ = exact_marginal_computer(R_LC, prior_hypotheses)