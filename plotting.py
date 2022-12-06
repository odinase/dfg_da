import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'
plt.style.use("seaborn-v0_8-whitegrid")

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
from ravens_parser_parallell import OUTPUT_PATH_BASE, PMBM_DATA_PATH


FIGURES_PATH = "./figures"


def cluster_stats_to_errors(cluster_stats: List[ClusterData]):
    lbp_errors = []
    williams_errors = []

    for cluster_stat, _ in cluster_stats: 
        if cluster_stat.explicit_hypothesis_enumeration_error:
            continue
        lbp_errors.append(MarginalsErrors(cluster_stat.exact_stats.marginals, cluster_stat.lbp_stats.marginals))
        williams_errors.append(MarginalsErrors(cluster_stat.exact_stats.marginals, cluster_stat.williams_stats.marginals))

    lbp_errors: MarginalsErrors = MarginalsErrors.concatenate(lbp_errors)
    williams_errors: MarginalsErrors = MarginalsErrors.concatenate(williams_errors)

    return lbp_errors, williams_errors


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

    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/sf.pdf", bbox_inches='tight')


def make_raw_error_plot(cluster_stats: List[ClusterData]):
    fig, ax = plt.subplots()

    lbp_errors, williams_errors = cluster_stats_to_errors(cluster_stats)

    p = 0.01
    lbp_errors_sample = np.random.choice(lbp_errors.raw_errors, int(p*len(lbp_errors.raw_errors)), replace=False)
    williams_errors_sample = np.random.choice(williams_errors.raw_errors, int(p*len(williams_errors.raw_errors)), replace=False)

    ax.plot(lbp_errors_sample, 'bo', label="Multihypothesis LBP", ms=2, alpha=0.2)
    ax.axhline(lbp_errors.raw_errors.mean(), color="blue")
    ax.plot(williams_errors_sample, 'go', label="Hypothesis-conditioned LBP with PHD approximation", ms=2, alpha=0.2)
    ax.axhline(williams_errors.raw_errors.mean(), color="green")

    leg = ax.legend(frameon = True)
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    ax.set_title(rf"Raw error of marginal error. MH-LBP mean: {lbp_errors.raw_errors.mean():.4e}$\pm${lbp_errors.raw_errors.std():.4e}, LBP-PHD mean: {williams_errors.raw_errors.mean():.4e}$\pm${williams_errors.raw_errors.std():.4e}")
    #Disable opacity for legend
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    bins = 75

    x = lbp_errors.raw_errors
    fig2, ax2 = plt.subplots(ncols=2, figsize=(10, 7), sharey=True)

    ax2[0].hist(x, bins=bins)
    ax2[0].semilogy()
    ax2[0].set_title(f"MH-LBP error histogram. Min: {x.min():.3e}, max: {x.max():.3e}")

    x = williams_errors.raw_errors
    ax2[1].hist(x, bins=bins)
    ax2[1].semilogy()
    ax2[1].set_title(f"LBP-PHD error histogram. Min: {x.min():.3e}, max: {x.max():.3e}")

    fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/raw_error.pdf", bbox_inches='tight')

    fig2.tight_layout()
    fig2.savefig(f"{FIGURES_PATH}/raw_error_histogram.pdf", bbox_inches='tight')


def make_correlation_plot(cluster_stats: List[ClusterData]):
    pass


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


    make_raw_error_plot(cluster_stats)