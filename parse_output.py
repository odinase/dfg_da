import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


from ravens_parser_parallell import OUTPUT_PATH_BASE, PMBM_DATA_PATH
import dfg_da.stats_logger as sl
import dfg_da.marginals_computers as mc


if __name__ == "__main__":
    load_dirs = Path(OUTPUT_PATH_BASE).glob("*/*")

    exact_marginals_list = []
    lbp_mh_marginals_list = []
    lbp_williams_marginals_list = []

    load_dirs = list(load_dirs)
    num_files = len(load_dirs)

    # num_empty_clusters = [l for l in load_dirs if Path(l).name == "empty_cluster"]
    # cluster_stats = []
    # for cluster_file in tqdm(load_dirs, total=num_files):
    #     cluster_stats.append(
    #         (sl.ClusterData.from_data(cluster_file), cluster_file)
    #     )

    # num_lbp_converged = sum(cluster_stat.lbp_stats.converged for cluster_stat, _ in cluster_stats if cluster_stat.lbp_stats is not None)
    # num_lbp_not_converged = sum(not cluster_stat.lbp_stats.converged for cluster_stat, _ in cluster_stats if cluster_stat.lbp_stats is not None)
    # # lbp_not_converged_files = [cluster_file for cluster_stat, cluster_file in cluster_stats if cluster_stat.lbp_stats is not None and not cluster_stat.lbp_stats.converged]
    # print(num_lbp_converged)
    # print(num_lbp_converged / (num_lbp_converged + num_lbp_not_converged))

    lbp_computer = mc.LBPMarginalsFullAssociation()

    for cluster_file in tqdm(load_dirs, total=num_files):
        cluster_stat = sl.ClusterData.from_data(cluster_file)
        if cluster_stat.lbp_stats is not None and not cluster_stat.lbp_stats.converged:
            mat_file = sl.MatFileParser(f"{PMBM_DATA_PATH}/{cluster_file.parent.name}.mat")
            break

    R_LC = mat_file.reward_matrix_lc
    cluster_idx = int("".join(d for d in cluster_file.name if d.isdigit()))
    prior_hypotheses = mat_file.prior_hypotheses_per_cluster[cluster_idx]
    t = np.sort(np.array([t for t in cluster_stat.tracks])) - 1
    out = lbp_computer(R_LC, prior_hypotheses, full_output=True)
    ds = out[2]
    prev_b_avg, b_avg, s_avg, max_norm_probs = out[3:]

    fig, ax = plt.subplots(nrows=3)

    # ax.plot(ds)
    ax[0].plot(prev_b_avg[-15:], label="prev b")
    ax[0].plot(b_avg[-15:], label="b")
    ax[1].plot(s_avg[-15:], label="s")
    ax[2].plot(max_norm_probs[-15:], label="max norm")

    for ax in ax:
        ax.legend()
    plt.show()

    # for cluster_file in tqdm(load_dirs, total=len(load_dirs)):
    #     cluster_stat: sl.ClusterData = sl.ClusterData.from_data(cluster_file)
    #     cluster_stats[cluster_file] = cluster_stat
    #     if cluster_stat.skipped:
    #         continue


    #     lbp_iterations_total[k] = cluster_stat.lbp_stats.num_iters
    #     lbp_iterations_msg[k] = cluster_stat.lbp_stats.num_iters_msg
    #     k += 1

    #     # exact_marginals_list.append(cluster_stat.exact_stats.marginals)
    #     # lbp_mh_marginals_list.append(cluster_stat.lbp_stats.marginals)
    #     # lbp_williams_marginals_list.append(cluster_stat.williams_stats.marginals)
        

    # fig, ax = plt.subplots(ncols=3)

    # lbp_iterations = np.vstack([lbp_iterations_msg[:k], lbp_iterations_total[:k]])

    # ax.set_title(f"Mean: {lbp_iterations_total.mean()}, median: {np.median(lbp_iterations_total)}, max: {lbp_iterations_total.max()}")
    # labels = ["Iterations until messages converged", "Total number of iterations"]
    # m1 = np.argmax(lbp_iterations, axis=1)
    # maxes = lbp_iterations[:, m1]
    # print(m1)
    # print(maxes)
    # num_converged = np.sum(lbp_iterations[1] < 10_000)
    # num_not_converged = lbp_iterations.shape[1] - num_converged
    # fig.suptitle(f"Num not converged: {num_not_converged} ({num_not_converged / (num_not_converged + num_converged) * 100.0:.3f}%), num converged: {num_converged}, total: {lbp_iterations.shape[1]}")
    # ax[0].boxplot(lbp_iterations.T, labels=labels)
    # ax[1].hist(lbp_iterations[1], bins=50)
    # ax[2].plot(lbp_iterations[1], 'x')

    # exact_marginals: sl.Marginals = sl.Marginals.concatenate(exact_marginals_list)
    # lbp_mh_marginals: sl.Marginals = sl.Marginals.concatenate(lbp_mh_marginals_list)
    # lbp_williams_marginals: sl.Marginals = sl.Marginals.concatenate(lbp_williams_marginals_list)

    # ax.plot(lbp_mh_marginals.detection_marginals, exact_marginals.detection_marginals, 'ro', label='Detection')
    # ax.plot(lbp_mh_marginals.misdetection_marginals, exact_marginals.misdetection_marginals, 'bs', label='Misdetection')
    # ax.plot(lbp_mh_marginals.nonexistence_marginals, exact_marginals.nonexistence_marginals, 'gD', label='Nonexistence')

    # ax.set_xlabel("Approximate probability")
    # ax.set_ylabel("Exact probability")
    # ax.set_title("Correlation plot")

    # ax.legend()
    # plt.show()