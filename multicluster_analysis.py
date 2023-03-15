from plotting_multicluster import load_cluster_stats
import dfg_da.stats_logger as sl
import dfg_da.cluster_visualizations as cv
from typing import List, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from ravens_parser_parallell_multicluster import OUTPUT_PATH_BASE, PMBM_DATA_PATH


def find_large_errors(cluster_stats: List[Tuple[sl.MulticlusterData, Path]], max_error_threshold: float = 0.4) -> List[Tuple[sl.MarginalsErrors, Path]]:
    large_errors: List[Tuple[sl.MarginalsErrors, Path]] = []

    for cluster_stat, cluster_file in cluster_stats:
        if cluster_stat.exact_computation_error:
            continue

        marg_error: sl.MarginalsErrors = sl.MarginalsErrors(
            cluster_stat.exact_marginals,
            cluster_stat.mhlbp_marginals
        )

        if marg_error.max_errors.max() >= max_error_threshold:
            large_errors.append((marg_error, cluster_file))

    return large_errors


def find_larger_bethe(cluster_stats: List[Tuple[sl.MulticlusterData, Path]]) -> List[Tuple[sl.MulticlusterData, Path]]:
    return [(cluster_stat, cluster_path) for cluster_stat, cluster_path in cluster_stats if cluster_stat.bethe_normalization_constant > cluster_stat.exact_normalization_constant]


def large_norm_const_error(cluster_stats: List[Tuple[sl.MulticlusterData, Path]], log_diff_threshold: float = 2.0) -> List[Tuple[sl.MulticlusterData, Path]]:
    large_norm_const_errors: List[Tuple[sl.MulticlusterData, Path]] = []
    
    for cluster_stat, cluster_path in cluster_stats:
        if np.abs(np.log(cluster_stat.bethe_normalization_constant) - np.log(cluster_stat.exact_normalization_constant)) > 2.0:
            large_norm_const_errors.append((cluster_stat, cluster_path))

    return large_norm_const_errors


def result_path_to_mat_file_string(result_path: Path, data_path: str = PMBM_DATA_PATH, suffix_to_remove: str = "_stats") -> str:
    mat_file = result_path.name[:-len(suffix_to_remove)]
    return data_path + "/" + mat_file + ".mat"


def number_of_tracks_in_cluster():
    pass




if __name__ == "__main__":
    cluster_stats = load_cluster_stats(path=OUTPUT_PATH_BASE)

    # large_errors = find_large_errors(cluster_stats, max_error_threshold=0.7)

    # print(len(large_errors))

    larger_bethe: List[Tuple[sl.MulticlusterData, Path]] = find_larger_bethe(cluster_stats)

    # print(len(larger_bethe))

    # large_norm_errs = large_norm_const_error(cluster_stats)

    # print(f"{len(large_norm_errs) / len(cluster_stats)*100.0:.2f}%")

    # print(result_path_to_mat_file_string(large_norm_errs[0][1]))


    #file = "./data/pmbm_output_files/priorLikelihood_iMC2k65.mat"

    ## Plotting
    
    # file = larger_bethe[0][1]
    # matfile = sl.MatFileParser(result_path_to_mat_file_string(file), use_cpp=True)

    # predicted_meas: List[sl.PredictedMeasurement] = matfile.predicted_track_measurement()
    # ph_per_c = matfile.prior_hypotheses_per_cluster
    # # tracks: List[sl.TrackEstimate] = [tracks[t] for t in t_idx]

    # # Use two prior hypotheses from cluster 1
    # cluster_sizes = [len(phs) for phs in ph_per_c]
    # biggest_cluster = np.argmax(cluster_sizes)
    # phs = ph_per_c[biggest_cluster]
    # print(len(phs))
    # phs = [phs[k] for k in range(10)]

    # fig, ax = plt.subplots()
    # color_map = mpl.cm.get_cmap("tab20", len(phs))
    # colors = color_map(np.linspace(0.0, 1.0, len(phs)))
    # for j, (ph, c) in enumerate(zip(phs, colors)):
    #     t_idx = np.array(ph.tracks()) - 1
    #     for k, track in enumerate(t_idx):
    #         label = "_"
    #         if k == 0:
    #             label = f"Hypothesis {j+1}"
    #         cv.draw_estimate(predicted_meas[track].measurement, predicted_meas[track].covariance, ax, c, label)

    # ax.legend()
    # plt.show()


    # Let's try 

    for k, (cluster_stat, cluster_path) in enumerate(larger_bethe[:2]):
        print(f"Scenario {k+1}\n")
        print(f"Bethe: {cluster_stat.bethe_normalization_constant}\nExact: {cluster_stat.exact_normalization_constant}")
        matfile: sl.MatFileParser = sl.MatFileParser(result_path_to_mat_file_string(cluster_path), use_cpp=True)

        prior_hypos_per_cluster = matfile.prior_hypotheses_per_cluster

        for c, prior_hypos in enumerate(prior_hypos_per_cluster):
            print(f"Cluster {c+1}")
            for h, hypo in enumerate(prior_hypos):
                print(f"\tHypothesis {h+1}: {hypo.tracks()}")
        
        print()