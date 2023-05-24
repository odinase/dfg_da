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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from copy import deepcopy
from dataclasses import dataclass

import seaborn as sns
sns.set_theme(style="ticks")
import asyncio


from dfg_da.stats_logger import MarginalsErrors, Marginals, ClusterData, MulticlusterData
import dfg_da.stats_logger as sl
from dfg_da.marginals_computers import LBPMarginalsFullAssociation, ClusterHypothesesPosterior
from ravens_parser_parallell import OUTPUT_PATH_BASE, PMBM_DATA_PATH
import dfg_da.cluster_bayes_tree as cbt

from plotting_multicluster_memory_efficient import load_cluster_stats_batch, make_batch_intervals, save_fig

from glob import glob
from multiprocessing import Pool
import multiprocessing
from dataclasses import dataclass


@dataclass
class MerginingClustersStats:
    numbers_of_prior_hypotheses_apriori: List[np.ndarray]
    numbers_of_competing_tracks_per_meas_apriori: List[np.ndarray]
    numbers_of_prior_hypotheses_aposteriori: np.ndarray
    numbers_of_competing_tracks_per_meas_aposteriori: List[np.ndarray]
    number_of_superclusters: int
    number_of_linking_measurements: List[np.ndarray]
    number_of_clusters_merging: np.ndarray

@dataclass
class ScenarioStats:
    number_of_prior_clusters: int
    number_of_posterior_clusters: int
    number_avg_tracks_and_prior_hypos_posterior: np.ndarray
    number_of_competing_tracks: List[np.ndarray]
    number_gated_measurements: int
    merging_clusters_stats: MerginingClustersStats



def loop_func(pmbm_file):
    mat_file_path = pmbm_file
    mat_file: sl.MatFileParser = sl.MatFileParser(mat_file_path, use_cpp=True)
    R_LC = mat_file.reward_matrix_lc
    prior_hypotheses_per_cluster = mat_file.prior_hypotheses_per_cluster
    assocLocal = mat_file.ws["assocLocal"].copy()
        # def __init__(self, R_LC: np.ndarray, prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList, assocLocal: np.ndarray):
    cluster_links = cbt.ClusterLinks(R_LC=R_LC, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster, assocLocal=assocLocal.copy())

    number_of_prior_clusters = len(prior_hypotheses_per_cluster)

    # Perhaps better to here look at average number of tracks per hypothesis over all hypotheses?

    # We should strictly speaking be looking closer at merging clusters more as well, as that will be interesting to justify the cluster conditioning.

    cluster_hypotheses_posterior = ClusterHypothesesPosterior(assocLocal=assocLocal, prior_hypotheses_per_cluster = prior_hypotheses_per_cluster)
    prior_hypotheses_per_cluster_posterior = cluster_hypotheses_posterior.prior_hypotheses_per_cluster_posterior
    number_of_posterior_clusters = len(prior_hypotheses_per_cluster_posterior)
    number_avg_tracks_and_prior_hypos_posterior = []
    number_of_competing_tracks = []
    gated_detections = np.isfinite(R_LC[:, 1:])
    for ph_post in prior_hypotheses_per_cluster_posterior:
        avg_num_tracks = np.mean([len(h.tracks()) for h in ph_post])
        num_hypos = len(ph_post)
        number_avg_tracks_and_prior_hypos_posterior.append((avg_num_tracks, num_hypos))

        for h in ph_post:
            t_idxs = (np.sort(h.tracks()) - 1).astype(int)
            gated_detections_c_hypo = gated_detections[t_idxs]
            # Sum downwards sums up number of tracks that gated measurement
            num_competing_tracks_c_hypo = gated_detections_c_hypo.sum(0) # This will give a bunch of zeros down the line, but maybe ok? Just throw out?
            number_of_competing_tracks.append(num_competing_tracks_c_hypo)

    number_gated_measurements = gated_detections.any(0).sum()


    # Here we gather cluster merging stats
    numbers_of_prior_hypotheses_apriori = []
    numbers_of_prior_hypotheses_aposteriori = []
    number_of_superclusters = len(cluster_links.merging_clusters)
    number_of_linking_measurements = []
    numbers_of_competing_tracks_per_meas_apriori: List[np.ndarray] = []
    numbers_of_competing_tracks_per_meas_aposteriori: List[np.ndarray] = []


    number_of_clusters_merging = np.empty(number_of_superclusters, dtype=int)

    # We know that the lowest cluster number conventionally is the master in a cluster merge

    c2lms = cluster_links.cluster_to_linking_meas()
    for k, clusters in enumerate(cluster_links.merging_clusters):
        num_hypos_prior_cluster = np.array([len(prior_hypotheses_per_cluster[c]) for c in clusters])
        numbers_of_prior_hypotheses_apriori.append(num_hypos_prior_cluster)
        numbers_of_prior_hypotheses_aposteriori.append(np.prod(num_hypos_prior_cluster))
        number_of_linking_measurements.append(np.array([len(c2lms[c]) for c in clusters]))
        number_of_clusters_merging[k] = len(clusters)

        master = min(clusters)
        post_idx = assocLocal[1, master].sum() - 1
        ph_supercluster = prior_hypotheses_per_cluster_posterior[post_idx]
        assert len(ph_supercluster) == np.prod(num_hypos_prior_cluster) # Should always be equal if logic is correct
        for h in ph_supercluster:
            # TODO(odin): fix here
            pass


    # @dataclass
    # class MerginingClustersStats:
    #     numbers_of_prior_hypotheses_apriori: List[List[int]]
    #     numbers_of_prior_hypotheses_aposteriori: List[int]
    #     number_of_superclusters: int
    #     number_of_linking_measurements: List[List[int]]

    merging_clusters_stats = MerginingClustersStats(
        number_of_clusters_merging=number_of_clusters_merging,
        numbers_of_prior_hypotheses_apriori=numbers_of_prior_hypotheses_apriori,
        numbers_of_prior_hypotheses_aposteriori=np.array(numbers_of_prior_hypotheses_aposteriori),
        number_of_superclusters=number_of_superclusters,
        number_of_linking_measurements=number_of_linking_measurements
    )

    # @dataclass
    # class ScenarioStats:
    #     number_of_clusters: int
    #     number_avg_tracks_and_prior_hypos_posterior: List[Tuple[float, int]]
    #     number_of_competing_tracks: List[np.ndarray]
    #     number_gated_measurements: int
    #     merging_clusters_stats: MerginingClustersStats

    if len(number_of_competing_tracks) > 0:
        number_of_competing_tracks=np.hstack(number_of_competing_tracks)
    else:
        number_of_competing_tracks=np.empty(0, dtype=int)

    scenario_stats = ScenarioStats(
        number_of_prior_clusters=number_of_prior_clusters,
        number_of_posterior_clusters=number_of_posterior_clusters,
        number_avg_tracks_and_prior_hypos_posterior=np.array(number_avg_tracks_and_prior_hypos_posterior),
        number_of_competing_tracks=number_of_competing_tracks,
        number_gated_measurements=number_gated_measurements,
        merging_clusters_stats=merging_clusters_stats
    )

    return scenario_stats

def plot_number_of_hypos_vs_avg_number_tracks(results: List[ScenarioStats]):
    avg_tracks_vs_num_hypos = np.vstack([stat.number_avg_tracks_and_prior_hypos_posterior for stat in results if len(stat.number_avg_tracks_and_prior_hypos_posterior) > 0])

    fig, ax = plt.subplots()

    ax.plot(*avg_tracks_vs_num_hypos.T, 'o', alpha=0.02)
    pruning_limit = 150
    ax.axhline(pruning_limit, color='black', linestyle='--', label=f"Pruning limit of {pruning_limit} hypotheses")
    ax.set_xlabel("Average number of tracks per hypothesis")
    ax.set_ylabel("Number of hypotheses")
    ax.semilogy()
    ax.legend()

    # heatmap_lbp, xedges, yedges = np.histogram2d(*avg_tracks_vs_num_hypos.T, bins=20)
    # X_lbp, Y_lbp = np.meshgrid(xedges[:-1], yedges[:-1])
    # df_approx = pd.DataFrame({
    #     "Average number of tracks per hypothesis": np.around(X_lbp.ravel(), decimals=3),
    #     "Number of hypotheses": np.around(Y_lbp.ravel(), decimals=3),
    #     "hist": heatmap_lbp.T.ravel()
    # })
    # df_approx = df_approx.pivot(index="Number of hypotheses", columns="Average number of tracks per hypothesis", values="hist")

    # sns.heatmap(df_approx, square=True, norm=LogNorm(), cmap="Reds", ax=ax)
    # # sns.heatmap(df_approx, square=True, cmap="Reds", ax=ax)

    save_fig(fig, "num_hypos_vs_avg_num_tracks_posterior")


def histogram_linking_measurements(results: List[ScenarioStats]):
    num_linking_meas = np.hstack([np.hstack(stat.merging_clusters_stats.number_of_linking_measurements) for stat in results if len(stat.merging_clusters_stats.number_of_linking_measurements) > 0])

    fig, ax = plt.subplots()

    ax.hist(num_linking_meas)
    ax.set_xlabel("Number of linking measurements")

    save_fig(fig, "histogram_linking_measurements")

def histogram_number_prior_clusters_in_supercluster(results: List[ScenarioStats]):
    num_clusters_merging = np.hstack([stat.merging_clusters_stats.number_of_clusters_merging for stat in results])

    fig, ax = plt.subplots()

    bin_edges = ax.hist(num_clusters_merging)[1]
    ax.set_xticks(bin_edges[:-1])
    ax.set_xlabel("Number of prior clusters in supercluster")

    save_fig(fig, "histogram_clusters_merging")


def print_merging_clusters_stats(results: List[ScenarioStats]):
    tot_number_of_posterior_clusters = 0
    tot_number_of_superclusters = 0
    n = 0
    sum_linking_meas = 0
    n_clus = 0
    sum_clusters_merging = 0
    n_ph_post = 0
    sum_ph_post = 0
    n_ph_prior = 0
    sum_ph_prior = 0

    for stat in results:
        tot_number_of_posterior_clusters += stat.number_of_posterior_clusters
        tot_number_of_superclusters += stat.merging_clusters_stats.number_of_superclusters
        linkings = np.hstack(stat.merging_clusters_stats.number_of_linking_measurements) if len(stat.merging_clusters_stats.number_of_linking_measurements) > 0 else np.empty(0, dtype=int)

        n += len(linkings)
        sum_linking_meas += linkings.sum()

        n_clus += len(stat.merging_clusters_stats.number_of_clusters_merging)
        sum_clusters_merging += stat.merging_clusters_stats.number_of_clusters_merging.sum()

        num_ph_post = stat.merging_clusters_stats.numbers_of_prior_hypotheses_aposteriori
        n_ph_post += len(num_ph_post)
        sum_ph_post += num_ph_post.sum()

        if len(stat.merging_clusters_stats.numbers_of_prior_hypotheses_apriori) > 0:
            num_ph_prior = np.hstack([num_phs for num_phs in stat.merging_clusters_stats.numbers_of_prior_hypotheses_apriori])
            n_ph_prior += len(num_ph_prior)
            sum_ph_prior += num_ph_prior.sum()


    print(f"Total number of posterior clusters & ${tot_number_of_posterior_clusters}$ \\\\")
    print(f"Total number of superclusters & ${tot_number_of_superclusters}$ (${tot_number_of_superclusters / tot_number_of_posterior_clusters * 100.0:.3f}\\%$ of clusters) \\\\")
    print(f"Average number of linking measurements & ${sum_linking_meas / n:.3f}$ \\\\")
    print(f"Average number of prior clusters in supercluster & ${sum_clusters_merging / n_clus:.3f}$ \\\\")
    print(f"Average number of prior hypotheses in superclusters & ${sum_ph_post / n_ph_post:.3f}$ \\\\")
    print(f"Average number of prior hypotheses in prior cluster in supercluster & ${sum_ph_prior / n_ph_prior:.3f}$")


def histogram_competing_tracks(results: List[ScenarioStats]):
    number_of_competing_tracks = np.hstack([stat.number_of_competing_tracks for stat in results if len(stat.number_of_competing_tracks) > 0])

    fig2, ax2 = plt.subplots()
    ax2.hist(number_of_competing_tracks)
    ax2.set_xlabel("Number of competing tracks")
    # ax2.set_ylabel("Number of occurences")
    ax2.semilogy()

    save_fig(fig2, "num_competing_tracks")


if __name__ == "__main__":
    pmbm_files = glob(PMBM_DATA_PATH + "/*.mat")
    pmbm_files = sorted(pmbm_files)

    # pmbm_files = pmbm_files[:50]

    num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    with tqdm(total=len(pmbm_files)) as pbar:
        results: List[ScenarioStats] = []
        for i, result in enumerate(pool.imap_unordered(loop_func, pmbm_files)):
            results.append(result)
            pbar.update(1)
            pbar.set_description(f"Progress: {i+1}/{len(pmbm_files)}, {(i+1)/len(pmbm_files)*100.0:.2f}%")

    plot_number_of_hypos_vs_avg_number_tracks(results)
    histogram_competing_tracks(results)
    histogram_linking_measurements(results)
    print_merging_clusters_stats(results)
    histogram_number_prior_clusters_in_supercluster(results)
