import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'

import numpy as np
from typing import Optional, Tuple, List
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import datashader as ds
import pandas as pd
import colorcet as cc
import numpy as np
import matplotlib.pyplot as plt


from dfg_da.stats_logger import MarginalsErrors, Marginals, ClusterData
import dfg_da.stats_logger as sl
from dfg_da.marginals_computers import LBPMarginalsFullAssociation
from ravens_parser_parallell import OUTPUT_PATH_BASE, PMBM_DATA_PATH


FIGURES_PATH = "./figures"


def save_fig_to_pdf(fig, fig_name):
    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/{fig_name}.pdf", bbox_inches='tight')


def cluster_stats_to_errors(cluster_stats: List[Tuple[ClusterData, Path]], add_williams_exact: bool = False):
    lbp_errors = []
    williams_errors = []
    if add_williams_exact:
        williams_exact_errors = []

    for cluster_stat, _ in cluster_stats:
        if cluster_stat.explicit_hypothesis_enumeration_error:
            continue
        lbp_errors.append(MarginalsErrors(cluster_stat.exact_stats.marginals, cluster_stat.lbp_stats.marginals))
        williams_errors.append(MarginalsErrors(cluster_stat.exact_stats.marginals, cluster_stat.williams_stats.marginals))
        if add_williams_exact:
            williams_exact_errors.append(MarginalsErrors(cluster_stat.exact_stats.marginals, cluster_stat.williams_stats.marginals_exact_normalization_constant))


    lbp_errors: MarginalsErrors = MarginalsErrors.concatenate(lbp_errors)
    williams_errors: MarginalsErrors = MarginalsErrors.concatenate(williams_errors)
    if add_williams_exact:
        williams_exact_errors: MarginalsErrors = MarginalsErrors.concatenate(williams_exact_errors)

    return (lbp_errors, williams_errors) if not add_williams_exact else (lbp_errors, williams_errors, williams_exact_errors)


def plot_iterations_not_converged(cluster_stats: List[ClusterData]):
    cluster_size_table_lbp_converge = defaultdict(lambda: 0)
    cluster_size_table_lbp_converge_num = defaultdict(lambda: 0)
    clusters_not_converged = list()

    print(f"Num empty clusters: {len(empty_clusters)}, {len(empty_clusters) / (len(empty_clusters) + len(cluster_stats))*100.0:.3f}%")

    for cluster_stat, cluster_file in cluster_stats:
        c = cluster_stat.cardinality
        cluster_size_table_lbp_converge[c] += cluster_stat.lbp_stats.num_iters
        cluster_size_table_lbp_converge_num[c] += 1
        if not cluster_stat.lbp_stats.converged:
            clusters_not_converged.append(c)

    table = np.array([t for t in cluster_size_table_lbp_converge.items()])
    table = table[table[:,0].argsort()]

    nums = np.array([t for t in cluster_size_table_lbp_converge_num.items()])
    nums = nums[nums[:,0].argsort()]

    table[:,1] = table[:,1] / nums[:,1]

    print(f"Num not converged: {len(clusters_not_converged)}")
    clusters_not_converged = np.sort(np.array([c for c in set(clusters_not_converged)]))
    print(clusters_not_converged)

    fig, ax = plt.subplots(nrows=3)

    ax[0].plot(*table.T)
    ax[0].set_xlabel("Cluster cardinality")
    ax[0].set_ylabel("LBP convergence average iterations")
    for cnc in clusters_not_converged:
        ax[0].axvline(cnc, color="red")

    ax[1].plot(*nums.T)

    ax[2].hist(clusters_not_converged, bins=25)
    ax[2].set_title("Clusters not converged")



def make_survival_function_plots(cluster_stats: List[ClusterData]):
    fig, ax = plt.subplots(nrows=5, figsize=(7, 12), sharex=True)

    lbp_errors, williams_errors = cluster_stats_to_errors(cluster_stats)

    plot_survival_function(ax, lbp_errors, "Multihypothesis LBP")
    plot_survival_function(ax, williams_errors, "Hypothesis-conditioned LBP with PHD approximation")

    save_fig_to_pdf(fig, "sf")


def make_raw_error_plot(cluster_stats: List[Tuple[ClusterData, Path]]):

    lbp_errors, williams_errors, williams_exact_errors = cluster_stats_to_errors(cluster_stats, add_williams_exact=True)

    p = 0.1
    lbp_errors_sample = np.random.choice(lbp_errors.raw_errors, int(p*len(lbp_errors.raw_errors)), replace=False)
    # williams_errors_sample = np.random.choice(williams_errors.raw_errors, int(p*len(williams_errors.raw_errors)), replace=False)
    # williams_errors_exact_sample = np.random.choice(williams_exact_errors.raw_errors, int(p*len(williams_exact_errors.raw_errors)), replace=False)

    mh_lbp_label = "Multihypothesis LBP"
    williams_label = "Hypothesis-conditioned LBP with PHD approximation"
    williams_exact_label = "Hypothesis-conditioned LBP with exact normalization constant"

    fig, ax = plt.subplots()

    ax.plot(lbp_errors_sample, 'bo', label=mh_lbp_label, ms=2, alpha=0.2)
    ax.axhline(lbp_errors.raw_errors.mean(), color="blue")
    # ax.plot(williams_errors_sample, 'go', label=williams_label, ms=2, alpha=0.2)
    # ax.axhline(williams_errors.raw_errors.mean(), color="green")

    # leg = ax.legend(frameon = True)
    # frame = leg.get_frame()
    # frame.set_facecolor('white')
    # frame.set_edgecolor('black')
    # for lh in leg.legendHandles: 
    #     lh.set_alpha(1)

    ax.set_title("Scatter plot over signed error for MH-LBP")

    bins = 100
    hist_figsize = (10, 7)

    fig2, ax2 = plt.subplots(figsize=hist_figsize)

    x = williams_errors.raw_errors
    ax2.hist(x, bins=bins, label=williams_label, alpha=0.5)

    x = lbp_errors.raw_errors
    ax2.hist(x, bins=bins, label=mh_lbp_label, alpha=0.5)

    ax2.set_title("Histogram over signed marginal errors")
    ax2.semilogy()
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=hist_figsize)

    x = williams_exact_errors.raw_errors
    ax3.hist(x, bins=bins, label=williams_exact_label, alpha=0.5)

    x = lbp_errors.raw_errors
    ax3.hist(x, bins=bins, label=mh_lbp_label, alpha=0.5)

    ax3.set_title("Histogram over signed marginal errors")
    ax3.semilogy()
    ax3.legend()


    save_fig_to_pdf(fig, "signed_error")
    save_fig_to_pdf(fig2, "signed_error_histogram")
    save_fig_to_pdf(fig3, "signed_error_histogram_exact")


def make_correlation_plot(cluster_stats: List[ClusterData]):
    pass


def make_divergence_comparison_plot(cluster_stats: List[Tuple[ClusterData, Path]]):
    lbp_max_errors = []
    lbp_not_converged_idx = []
    k = 0
    for cluster_stat, cluster_file in cluster_stats:
        if not cluster_stat.explicit_hypothesis_enumeration_error:
            lbp_max_error = MarginalsErrors(cluster_stat.exact_stats.marginals, cluster_stat.lbp_stats.marginals).max_errors.max()
            lbp_max_errors.append(lbp_max_error)
            if not cluster_stat.lbp_stats.converged:
                lbp_not_converged_idx.append(k)
            
            k += 1

    lbp_max_errors = np.array(lbp_max_errors)
    lbp_not_converged_idx = np.array(lbp_not_converged_idx)
    
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(lbp_max_errors, 'b', label="max error per cluster")
    lbp_max_errors_not_converged: np.ndarray = lbp_max_errors[lbp_not_converged_idx]
    ax.plot(lbp_not_converged_idx, lbp_max_errors_not_converged, 'ro', label="Not converged clusters")
    leg = ax.legend(frameon = True)
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')

    ax.set_title(rf"Mean max error not converged: {lbp_max_errors_not_converged.mean()}$\pm${lbp_max_errors_not_converged.std()}")

    save_fig_to_pdf(fig, "divergence_plot")


def make_scatter_compare_plot(cluster_stats: List[Tuple[ClusterData, Path]]):
    # lbp_computer = LBPMarginalsFullAssociation()
    
    # Compare num tracks, num hypotheses, max error
    max_errors = defaultdict(list)
    num_tracks = defaultdict(list)
    num_hypos = defaultdict(list)
    num_gated_measurements = defaultdict(list)
    max_competing_tracks_for_measurement = defaultdict(list)

    # def copmute_number_of

    for cluster_stat, cluster_file in tqdm(cluster_stats):
        if not cluster_stat.explicit_hypothesis_enumeration_error:
            mat_file_path = PMBM_DATA_PATH + "/" + cluster_file.parent.name + ".mat"
            mat_file: sl.MatFileParser = sl.MatFileParser(mat_file_path)
            R_LC = mat_file.reward_matrix_lc
            gated_measurements = np.isfinite(R_LC[:, 1:])
            num_gated_measurements_c = gated_measurements.any(axis=0).sum()
            cluster_idx = int("".join(d for d in cluster_file.name if d.isdigit()))
            max_competing_tracks_for_measurement_c = -np.inf
            for tracks, _ in mat_file.prior_hypotheses_per_cluster[cluster_idx]:
                max_competing_tracks_for_measurement_hc = gated_measurements[tracks-1, :].sum(axis=0).max()
                max_competing_tracks_for_measurement_c = max(max_competing_tracks_for_measurement_c, max_competing_tracks_for_measurement_hc)

            max_competing_tracks_for_measurement_c = int(max_competing_tracks_for_measurement_c)               


            def save_stat(result):
                max_error = MarginalsErrors(cluster_stat.exact_stats.marginals, cluster_stat.lbp_stats.marginals).max_errors.max()
                max_errors[result].append(max_error)
                num_tracks_c = cluster_stat.cardinality
                num_hypos_c = cluster_stat.num_hypotheses

                num_tracks[result].append(num_tracks_c)
                num_hypos[result].append(num_hypos_c)
                num_gated_measurements[result].append(num_gated_measurements_c)
                max_competing_tracks_for_measurement[result].append(max_competing_tracks_for_measurement_c)
            
            if not cluster_stat.lbp_stats.converged:
                save_stat("divergent")
            else:
                save_stat("convergent")


    def make_df(result):
        return pd.DataFrame({
            "Max marginal error": max_errors[result],
            "Number of tracks": num_tracks[result],
            "Number of hypotheses": num_hypos[result],
            r"Number of gate\\measurements": num_gated_measurements[result],
            r"Highest number of tracks\\competing for measurement": max_competing_tracks_for_measurement[result]
        })

    figsize = (9, 8)


    df_divergent = make_df("divergent")
    num_data = len(df_divergent.columns)
    fig_divergent, ax_divergent = plt.subplots(figsize=figsize, nrows=num_data, ncols=num_data)
    pd.plotting.scatter_matrix(df_divergent, alpha=0.6, ax=ax_divergent)
    fig_divergent.suptitle("Statistics for divergent LBP")
    save_fig_to_pdf(fig_divergent, "scatter_matrix_diverged_clusters")


    df_convergent = make_df("convergent")
    num_data = len(df_convergent.columns)
    fig_convergent, ax_convergent = plt.subplots(figsize=figsize, nrows=num_data, ncols=num_data)
    pd.plotting.scatter_matrix(df_convergent, alpha=0.6, ax=ax_convergent)
    fig_convergent.suptitle("Statistics for convergent LBP")
    save_fig_to_pdf(fig_convergent, "scatter_matrix_converged_clusters")


    df_total = pd.concat((df_divergent, df_convergent))
    num_data = len(df_total.columns)
    fig, ax = plt.subplots(figsize=figsize, nrows=num_data, ncols=num_data)
    pd.plotting.scatter_matrix(df_total, alpha=0.6, ax=ax)
    fig.suptitle("Statistics for all LBP")
    save_fig_to_pdf(fig, "scatter_matrix_clusters")


def make_heatmap_correlation(cluster_stats: List[Tuple[ClusterData, Path]]):
    exact_marginals = []
    lbp_marginals = []

    for cluster_stat, _ in cluster_stats:
        if not cluster_stat.explicit_hypothesis_enumeration_error:
            exact_marginals.append(cluster_stat.exact_stats.marginals)
            lbp_marginals.append(cluster_stat.lbp_stats.marginals)

    exact_marginals: Marginals = Marginals.concatenate(exact_marginals)
    lbp_marginals: Marginals = Marginals.concatenate(lbp_marginals)

    num_bins = 100
    xedges = np.linspace(0, 1, num_bins)
    yedges = xedges
    bins = (xedges, yedges)

    heatmap = np.histogram2d(lbp_marginals.marginals, exact_marginals.marginals, bins=bins)[0]

    fig, ax = plt.subplots()

    log_heatmap = np.log(heatmap)
    log_heatmap[~np.isfinite(log_heatmap)] = np.nan
    i = ax.imshow(log_heatmap)
    ax.invert_yaxis()
    ticks = np.arange(num_bins) - 0.5
    ticks_step = num_bins // 5
    ax.set_xticks(ticks[ticks_step - 1::ticks_step], labels=[f"{i:.2f}" for i in xedges[ticks_step - 1::ticks_step]], rotation=70)
    ax.set_yticks(ticks[ticks_step - 1::ticks_step], labels=[f"{i:.2f}" for i in yedges[ticks_step - 1::ticks_step]])
    ax.set_xlabel("MH-LBP marginals")
    ax.set_ylabel("Exact marginals")
    c=fig.colorbar(i)
    c.ax.set_yticklabels(["$10^{" + str(int(cc)) + "}$" for cc in c.get_ticks()])
    plt.show()
    save_fig_to_pdf(fig, "heatmap_correlation")


def subsample(a: np.ndarray, inc: float) -> np.ndarray:
    """
    Returns the indices that subsamples the array.
    """
    idxs = [0]
    for k in range(1, len(a)):
        if a[k] - a[idxs[-1]] > inc:
            idxs.append(k)

    return idxs



def plot_survival_function(axes: plt.Axes, marginals_errors: MarginalsErrors, label: str = "_"):
    max_errors = np.sort(marginals_errors.max_errors)
    abs_errors = np.sort(marginals_errors.abs_errors)
    # raw_errors = np.sort(marginals_errors.abs_errors)
    misdetection_errors = np.sort(marginals_errors.misdetection_errors)
    detection_errors = np.sort(marginals_errors.detection_errors)
    nonexistence_errors = np.sort(marginals_errors.nonexistence_errors)

    errors = [
        max_errors,
        abs_errors,
        # raw_errors,
        misdetection_errors,
        detection_errors,
        nonexistence_errors
    ]

    titles = [
        "max_errors",
        "abs_errors",
        # "raw_errors",
        "misdetection_errors",
        "detection_errors",
        "nonexistence_errors"
    ]

    titles = [title.capitalize().replace("_", " ") for title in titles]

    for ax, error, title in zip(axes, errors, titles):
        ax.set_title(title)
        steps = np.linspace(1.0, 0.0, len(error))
        idxs = subsample(np.log(error), 1e-7)
        print(f"Subsampling {title} for {label} reduced data to {len(idxs)/len(error)*100.0:.3f}%")
        error = error[idxs]
        steps = steps[idxs]
        ax.step(error, steps, label=label)
        ax.set_xscale('symlog', linthresh=error[1])
        log_err = np.linspace(np.log10(error[1]), np.log10(error[-1]*1.1), 9).astype(int)
        xticks = np.array([0, *((10.0)**log_err)])
        print(f"Using xticks {xticks}")
        ax.set_xticks(xticks)
        xticks_labels = [0] + [rf'$10^{{{l}}}$' for l in log_err]
        ax.set_xticklabels(xticks_labels)
        ax.semilogy()
        ax.grid(True, alpha=0.2)
        if label != "_":
            ax.legend()


if __name__ == "__main__":
    load_dirs = Path(OUTPUT_PATH_BASE).glob("*/*")

    load_dirs = list(load_dirs)
    num_files = len(load_dirs)


    cluster_stats: List[Tuple[ClusterData, Path]] = []
    empty_clusters: List[Tuple[ClusterData, Path]] = []
    for cluster_file in tqdm(load_dirs, total=num_files):
        if cluster_file.name == "empty_cluster":
            empty_clusters.append(
                (ClusterData.from_data(cluster_file), cluster_file)
            )
        else:
            cluster_stats.append(
                (ClusterData.from_data(cluster_file), cluster_file)
            )


    # make_raw_error_plot(cluster_stats)
    # make_divergence_comparison_plot(cluster_stats)
    # make_scatter_compare_plot(cluster_stats)
    make_heatmap_correlation(cluster_stats)