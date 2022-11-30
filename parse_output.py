import numpy as np
import matplotlib.pyplot as plt
from ravens_parser_parallell import OUTPUT_PATH_BASE
from pathlib import Path
import dfg_da.stats_logger as sl


if __name__ == "__main__":
    load_dirs = Path(OUTPUT_PATH_BASE).glob("*/*")

    exact_marginals_list = []
    lbp_mh_marginals_list = []
    lbp_williams_marginals_list = []

    for cluster_file in load_dirs:
        cluster_stat: sl.ClusterData = sl.ClusterData.from_data(cluster_file)

        exact_marginals_list.append(cluster_stat.exact_stats.marginals)
        lbp_mh_marginals_list.append(cluster_stat.lbp_stats.marginals)
        lbp_williams_marginals_list.append(cluster_stat.williams_stats.marginals)
        

    fig, ax = plt.subplots()

    exact_marginals: sl.Marginals = sl.Marginals.concatenate(exact_marginals_list)
    lbp_mh_marginals: sl.Marginals = sl.Marginals.concatenate(lbp_mh_marginals_list)
    lbp_williams_marginals: sl.Marginals = sl.Marginals.concatenate(lbp_williams_marginals_list)

    ax.plot(lbp_mh_marginals.detection_marginals, exact_marginals.detection_marginals, 'ro', label='Detection')
    ax.plot(lbp_mh_marginals.misdetection_marginals, exact_marginals.misdetection_marginals, 'bs', label='Misdetection')
    ax.plot(lbp_mh_marginals.nonexistence_marginals, exact_marginals.nonexistence_marginals, 'gD', label='Nonexistence')

    ax.set_xlabel("Approximate probability")
    ax.set_ylabel("Exact probability")
    ax.set_title("Correlation plot")

    ax.legend()
    plt.show()