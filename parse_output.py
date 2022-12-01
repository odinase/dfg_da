import numpy as np
import matplotlib.pyplot as plt
from ravens_parser_parallell import OUTPUT_PATH_BASE
from pathlib import Path
import dfg_da.stats_logger as sl
from tqdm import tqdm

if __name__ == "__main__":
    load_dirs = Path(OUTPUT_PATH_BASE).glob("*/*")

    exact_marginals_list = []
    lbp_mh_marginals_list = []
    lbp_williams_marginals_list = []


    load_dirs = list(load_dirs)
    num_files = len(load_dirs)

    lbp_iterations_total = np.empty(num_files, dtype=int)
    lbp_iterations_msg = np.empty(num_files, dtype=int)

    cluster_stats = dict()

    num_empty_clusters = [l for l in load_dirs if Path(l).name == "empty_cluster"]
    cluster_stats = [sl.ClusterData.from_data(cluster_file) for cluster_file in load_dirs]

    num_lbp_converged = sum(cluster_stat.lbp_stats.converged for cluster_stat in cluster_stats if cluster_stat.lbp_stats is not None)
    print(num_lbp_converged)

    print(num_empty_clusters)

    k = 0
    for cluster_file in tqdm(load_dirs, total=len(load_dirs)):
        cluster_stat: sl.ClusterData = sl.ClusterData.from_data(cluster_file)
        cluster_stats[cluster_file] = cluster_stat
        if cluster_stat.skipped:
            continue



        lbp_iterations_total[k] = cluster_stat.lbp_stats.num_iters
        lbp_iterations_msg[k] = cluster_stat.lbp_stats.num_iters_msg
        k += 1

        # exact_marginals_list.append(cluster_stat.exact_stats.marginals)
        # lbp_mh_marginals_list.append(cluster_stat.lbp_stats.marginals)
        # lbp_williams_marginals_list.append(cluster_stat.williams_stats.marginals)
        

    fig, ax = plt.subplots(ncols=3)

    lbp_iterations = np.vstack([lbp_iterations_msg[:k], lbp_iterations_total[:k]])

    # ax.set_title(f"Mean: {lbp_iterations_total.mean()}, median: {np.median(lbp_iterations_total)}, max: {lbp_iterations_total.max()}")
    labels = ["Iterations until messages converged", "Total number of iterations"]
    m1 = np.argmax(lbp_iterations, axis=1)
    maxes = lbp_iterations[:, m1]
    print(m1)
    print(maxes)
    num_converged = np.sum(lbp_iterations[1] < 10_000)
    num_not_converged = lbp_iterations.shape[1] - num_converged
    fig.suptitle(f"Num not converged: {num_not_converged} ({num_not_converged / (num_not_converged + num_converged) * 100.0:.3f}%), num converged: {num_converged}, total: {lbp_iterations.shape[1]}")
    ax[0].boxplot(lbp_iterations.T, labels=labels)
    ax[1].hist(lbp_iterations[1], bins=50)
    ax[2].plot(lbp_iterations[1], 'x')

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
    plt.show()