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
from dfg_da.marginals_computers import LBPMarginalsFullAssociation
from ravens_parser_parallell import OUTPUT_PATH_BASE, PMBM_DATA_PATH


FIGURES_PATH = "./figures"

def save_fig_to_pdf(fig, fig_name, tight_layout=True):
    if tight_layout:
        fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/{fig_name}.pdf", bbox_inches='tight')
    print(f"Saved {FIGURES_PATH}/{fig_name}.pdf")


def save_fig_to_png(fig, fig_name, tight_layout=True):
    if tight_layout:
        fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/{fig_name}.png", bbox_inches='tight', dpi=600)
    print(f"Saved {FIGURES_PATH}/{fig_name}.png")


def save_fig(fig, fig_name, tight_layout=True):
    save_fig_to_pdf(fig, fig_name, tight_layout=tight_layout)
    save_fig_to_png(fig, fig_name, tight_layout=tight_layout)


def cluster_file_to_mat_file(cluster_file):
    return PMBM_DATA_PATH + "/" + cluster_file.parent.name + ".mat"

def cluster_file_to_cluster_idx(cluster_file):
    return int("".join(d for d in cluster_file.name if d.isdigit()))

def split_cluster_stats_converged(cluster_stats: List[Tuple[ClusterData, Path]]) -> Tuple[List[Tuple[ClusterData, Path]], List[Tuple[ClusterData, Path]]]:
    cluster_stats_converged = []
    cluster_stats_diverged = []

    for cluster_stat, cluster_file in cluster_stats:
        if cluster_stat.lbp_stats.converged:
            cluster_stats_converged.append((cluster_stat, cluster_file))
        else:
            cluster_stats_diverged.append((cluster_stat, cluster_file))

    return cluster_stats_converged, cluster_stats_diverged

def cluster_stats_to_errors(cluster_stats: List[Tuple[MulticlusterData, Path]]) -> MarginalsErrors:
    lbp_errors = []

    for cluster_stat, _ in cluster_stats:
        if cluster_stat.exact_computation_error:
            continue
        lbp_errors.append(MarginalsErrors(cluster_stat.exact_marginals, cluster_stat.mhlbp_marginals))


    lbp_errors: MarginalsErrors = MarginalsErrors.concatenate(lbp_errors)

    return lbp_errors

@dataclass
class NormConstTuple:
    bethe_constant: float
    exact_constant:float


def cluster_stats_to_norm_consts(cluster_stats: List[Tuple[MulticlusterData, Path]]) -> List[NormConstTuple]:
    norm_consts = []

    for cluster_stat, _ in cluster_stats:
        if cluster_stat.exact_computation_error:
            continue
        norm_consts.append(NormConstTuple(bethe_constant=cluster_stat.bethe_normalization_constant, exact_constant=cluster_stat.exact_marginals))

    return norm_consts



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

    fig.suptitle("Survival functions MCMH LBP")
    lbp_errors  = cluster_stats_to_errors(cluster_stats)

    plot_survival_function(ax, lbp_errors)
    save_fig(fig, "sf")



def make_conditioned_survival_function_plots(cluster_stats: List[Tuple[ClusterData, Path]]):
    cluster_stats_converged, cluster_stats_diverged = split_cluster_stats_converged(cluster_stats)
    cluster_stats_list = [cluster_stats_converged, cluster_stats_diverged]

    fig = plt.figure(figsize=(16, 12))
    subfigs = fig.subfigures(ncols=2)
    fig_titles = ["MH-LBP converged", "MH-LBP diverged"]
    for cluster_stats, subfig, figtitle in zip(cluster_stats_list, subfigs, fig_titles):
        subfig.suptitle(figtitle)
        ax = subfig.subplots(nrows=5, sharey=True, sharex=True)
        lbp_errors, williams_errors = cluster_stats_to_errors(cluster_stats)

        plot_survival_function(ax, lbp_errors, "Multihypothesis LBP")
        plot_survival_function(ax, williams_errors, "Hypothesis-conditioned LBP with PHD approximation")

    save_fig(fig, "sf_conditioned", tight_layout=False)


def make_raw_error_plot(cluster_stats: List[Tuple[ClusterData, Path]]):

    lbp_errors, williams_errors, williams_exact_errors = cluster_stats_to_errors(cluster_stats, add_williams_exact=True)

    # p = 0.1
    # lbp_errors_sample = np.random.choice(lbp_errors.raw_errors, int(p*len(lbp_errors.raw_errors)), replace=False)
    # williams_errors_sample = np.random.choice(williams_errors.raw_errors, int(p*len(williams_errors.raw_errors)), replace=False)
    # williams_errors_exact_sample = np.random.choice(williams_exact_errors.raw_errors, int(p*len(williams_exact_errors.raw_errors)), replace=False)

    mh_lbp_label = "Multihypothesis LBP"
    williams_label = "Hypothesis-conditioned LBP with PHD approximation"
    williams_exact_label = "Hypothesis-conditioned LBP with exact normalization constant"

    # fig, ax = plt.subplots()

    # ax.plot(lbp_errors_sample, 'bo', label=mh_lbp_label, ms=2, alpha=0.2)
    # ax.axhline(lbp_errors.raw_errors.mean(), color="blue")
    # ax.plot(williams_errors_sample, 'go', label=williams_label, ms=2, alpha=0.2)
    # ax.axhline(williams_errors.raw_errors.mean(), color="green")

    # leg = ax.legend(frameon = True)
    # frame = leg.get_frame()
    # frame.set_facecolor('white')
    # frame.set_edgecolor('black')
    # for lh in leg.legendHandles: 
    #     lh.set_alpha(1)

    # ax.set_title("Scatter plot over signed error for MH-LBP")

    bins = 100
    hist_figsize = (10, 7)

    fig2, ax2 = plt.subplots(figsize=hist_figsize)

    x = williams_errors.raw_errors
    ax2.hist(x, bins=bins, label=williams_label, alpha=0.33)

    x = lbp_errors.raw_errors
    ax2.hist(x, bins=bins, label=mh_lbp_label, alpha=0.33)

    x = williams_exact_errors.raw_errors
    ax2.hist(x, bins=bins, label=williams_exact_label, alpha=0.34)

    ax2.set_title("Histogram over signed marginal errors", fontsize=20)
    ax2.semilogy()
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='minor', labelsize=18)


    # fig3, ax3 = plt.subplots(figsize=hist_figsize)

    # x = lbp_errors.raw_errors
    # ax3.hist(x, bins=bins, label=mh_lbp_label, alpha=0.5)

    # ax3.set_title("Histogram over signed marginal errors")
    # ax3.semilogy()
    # ax3.legend()


    # save_fig_to_pdf(fig, "signed_error")
    save_fig(fig2, "signed_error_histogram")
    # save_fig_to_pdf(fig3, "signed_error_histogram_exact")

    # cluster_stats_lbp_converged = [(cluster_stat, cluster_file) for (cluster_stat, cluster_file) in cluster_stats if cluster_stat.lbp_stats.converged]
    # cluster_stats_lbp_not_converged = [(cluster_stat, cluster_file) for (cluster_stat, cluster_file) in cluster_stats if not cluster_stat.lbp_stats.converged]
    # lbp_errors_converged, williams_errors_converged = cluster_stats_to_errors(cluster_stats_lbp_converged)
    # lbp_errors_not_converged, williams_errors_not_converged = cluster_stats_to_errors(cluster_stats_lbp_not_converged)


    # fig4, ax4 = plt.subplots(figsize=hist_figsize)

    # x = williams_errors_converged.raw_errors
    # ax4.hist(x, bins=bins, label=williams_label, alpha=0.5)

    # x = lbp_errors_converged.raw_errors
    # ax4.hist(x, bins=bins, label=mh_lbp_label, alpha=0.5)

    # ax4.set_title("Histogram over signed marginal errors, MH-LBP converged")
    # ax4.semilogy()
    # ax4.legend()

    # save_fig_to_pdf(fig4, "signed_error_histogram_lbp_converged")


    # fig5, ax5 = plt.subplots(figsize=hist_figsize)

    # x = williams_errors_not_converged.raw_errors
    # ax5.hist(x, bins=bins, label=williams_label, alpha=0.5)

    # x = lbp_errors_not_converged.raw_errors
    # ax5.hist(x, bins=bins, label=mh_lbp_label, alpha=0.5)

    # ax5.set_title("Histogram over signed marginal errors, MH-LBP did not converge")
    # ax5.semilogy()
    # ax5.legend()

    # save_fig_to_pdf(fig5, "signed_error_histogram_lbp_not_converged")




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

    save_fig(fig, "divergence_plot")


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
            
            # if not cluster_stat.lbp_stats.converged:
            #     save_stat("divergent")
            # else:
            #     save_stat("convergent")

            save_stat("all")


    def make_df(result):
        return pd.DataFrame({
            "Convergence": result.capitalize(),
            "Max marginal error": max_errors[result],
            "Number of tracks": num_tracks[result],
            "Number of hypotheses": num_hypos[result],
            "Number of gated measurements": num_gated_measurements[result],
            "Highest number of tracks competing for measurement": max_competing_tracks_for_measurement[result]
        })

    figsize = (16, 16)

    # df_divergent = make_df("divergent")
    # df_convergent = make_df("convergent")
    # df = pd.concat((df_convergent, df_divergent))
    df = make_df("all").drop(columns=["Convergence", "Max marginal error"])
    num_data = len(df.columns)
    fig, ax = plt.subplots(figsize=figsize, nrows=2, ncols=2)

    for axx, d_name in zip(ax.ravel(), df.columns):
        axx.hist(df[d_name])
        axx.set_title(d_name, fontsize=20)
        axx.tick_params(axis='both', which='major', labelsize=18)
        axx.tick_params(axis='both', which='minor', labelsize=18)

    save_fig(fig, "hist_stats", tight_layout=False)


    # pd.plotting.scatter_matrix(df, alpha=0.3, ax=ax)
    # g = sns.pairplot(df, diag_kind="hist", plot_kws={"alpha": 0.2})#, diag_kws={"stat": "density"})
    # g.fig.set_size_inches(*figsize)
    # # # plt.show()
    # save_fig(fig, "scatter_matrix_we_max_marginal_error", tight_layout=False)


    # df_convergent = make_df("convergent")
    # num_data = len(df_convergent.columns)
    # fig_convergent, ax_convergent = plt.subplots(figsize=figsize, nrows=num_data, ncols=num_data)
    # pd.plotting.scatter_matrix(df_convergent, alpha=0.6, ax=ax_convergent)
    # fig_convergent.suptitle("Statistics for convergent LBP")
    # save_fig_to_pdf(fig_convergent, "scatter_matrix_converged_clusters")


    # df_total = pd.concat((df_divergent, df_convergent))
    # num_data = len(df_total.columns)
    # fig, ax = plt.subplots(figsize=figsize, nrows=num_data, ncols=num_data)
    # pd.plotting.scatter_matrix(df_total, alpha=0.6, ax=ax)
    # fig.suptitle("Statistics for all LBP")
    # save_fig_to_pdf(fig, "scatter_matrix_clusters")


def compare_mhlbp_lbpphd(cluster_stats: List[Tuple[ClusterData, Path]]):
    runtimes_mhlbp = []
    runtimes_wlbp = []

    for cluster_stat, _ in cluster_stats:
        runtimes_mhlbp.append(cluster_stat.lbp_stats.num_iters)
        runtimes_wlbp.append(cluster_stat.williams_stats.lbp_iters.sum())

    num_plots = 1
    fig, ax = plt.subplots(nrows=num_plots)
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    ax[0].hist(runtimes_mhlbp, alpha=0.7, label="MH-LBP")
    ax[0].hist(runtimes_wlbp, alpha=0.7, label="LBP-PHD")
    ax[0].set_ylabel("Runtimes")
    leg = ax[0].legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    ax[0].semilogy()

    # plt.show/(


    save_fig(fig, "mhlbp_lbpphd_compare")


def make_heatmap_correlation(cluster_stats: List[Tuple[MulticlusterData, Path]]):
    exact_marginals = []
    lbp_marginals = []

    for cluster_stat, _ in cluster_stats:
        exact_marginals.append(cluster_stat.exact_marginals)
        lbp_marginals.append(cluster_stat.mhlbp_marginals)

    exact_marginals: Marginals = Marginals.concatenate(exact_marginals)
    lbp_marginals: Marginals = Marginals.concatenate(lbp_marginals)

    num_bins = 200
    xedges = np.linspace(0, 1, num_bins)
    yedges = xedges
    bins = (xedges, yedges)

    heatmap_lbp, xedges, yedges = np.histogram2d(lbp_marginals.marginals, exact_marginals.marginals, bins=bins)
    X_lbp, Y_lbp = np.meshgrid(xedges[:-1], yedges[:-1])

    df_lbp = pd.DataFrame({
        "MH-LBP marginals": np.around(X_lbp.ravel(), decimals=3),
        "Exact marginals": np.around(Y_lbp.ravel(), decimals=3),
        "hist": heatmap_lbp.ravel()
    })
    df_lbp = df_lbp.pivot(index="Exact marginals", columns="MH-LBP marginals", values="hist")

    dfs = [df_lbp]
    figsize = (8, 8)

    nrows = len(dfs)
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, sharex=True)
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    num_ticks = 5
    depth_list = np.linspace(0, 1, num_ticks)
    # the index of the position of yticks
    # yticks = np.arange(0, num_bins, num_bins // num_ticks)
    # xticks = yticks
    # # # the content of labels of these yticks
    # yticklabes = np.linspace()
    # xticklabes = yticklabes
    for k, (axx, df) in enumerate(zip(ax, dfs)):
        # sns.heatmap(df, square=True, cmap="Reds", ax=axx)
        sns.heatmap(df, square=True, norm=LogNorm(), cmap="Reds", ax=axx)

        # axx.set_xticks(xticks)
        # axx.set_yticks(yticks)
        axx.tick_params(axis='both', which='major', labelsize=16)
        axx.tick_params(axis='both', which='minor', labelsize=16)
        cbar = axx.collections[0].colorbar
        axx.set_ylabel(df.index.name, fontsize=18)
        axx.set_xlabel(df.columns.name, fontsize=16)
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=18)
        axx.invert_yaxis()
        if k < nrows - 1:
            axx.tick_params(bottom=False)

    # params = {
    #         # 'legend.fontsize': 'x-large',
    #         # 'figure.figsize': (15, 5),
    #         # 'axes.labelsize': 30,
    #         # 'axes.titlesize':'x-large',
    #         # 'xtick.labelsize':'x-large',
    #         # 'ytick.labelsize':'x-large'
    #         }
    # plt.rcParams.update(params)


    save_fig(fig, "heatmap_correlation")


def correlation_plot_theta_posterior(cluster_stats: List[Tuple[MulticlusterData, Path]]):
    exact_marginals = []
    lbp_marginals = []

    for cluster_stat, _ in cluster_stats:
        exact_marginals.extend(cluster_stat.exact_output.compute_theta_posteriors())
        lbp_marginals.extend(cluster_stat.mhlbp_output.hypotheses_marginals())

    exact_marginals: np.ndarray = np.hstack(exact_marginals)
    lbp_marginals: np.ndarray = np.hstack(lbp_marginals)

    num_bins = 200
    xedges = np.linspace(0, 1, num_bins)
    yedges = xedges
    bins = (xedges, yedges)

    heatmap_lbp, xedges, yedges = np.histogram2d(lbp_marginals, exact_marginals, bins=bins)
    X_lbp, Y_lbp = np.meshgrid(xedges[:-1], yedges[:-1])

    df_lbp = pd.DataFrame({
        "MH-LBP marginals": np.around(X_lbp.ravel(), decimals=3),
        "Exact marginals": np.around(Y_lbp.ravel(), decimals=3),
        "hist": heatmap_lbp.ravel()
    })
    df_lbp = df_lbp.pivot(index="Exact marginals", columns="MH-LBP marginals", values="hist")

    dfs = [df_lbp]
    figsize = (8, 8)

    nrows = len(dfs)
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, sharex=True)
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    num_ticks = 5
    depth_list = np.linspace(0, 1, num_ticks)
    # the index of the position of yticks
    # yticks = np.arange(0, num_bins, num_bins // num_ticks)
    # xticks = yticks
    # # # the content of labels of these yticks
    # yticklabes = np.linspace()
    # xticklabes = yticklabes
    for k, (axx, df) in enumerate(zip(ax, dfs)):
        # sns.heatmap(df, square=True, cmap="Reds", ax=axx)
        sns.heatmap(df, square=True, norm=LogNorm(), cmap="Reds", ax=axx)

        # axx.set_xticks(xticks)
        # axx.set_yticks(yticks)
        axx.tick_params(axis='both', which='major', labelsize=16)
        axx.tick_params(axis='both', which='minor', labelsize=16)
        cbar = axx.collections[0].colorbar
        axx.set_ylabel(df.index.name, fontsize=18)
        axx.set_xlabel(df.columns.name, fontsize=16)
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=18)
        axx.invert_yaxis()
        if k < nrows - 1:
            axx.tick_params(bottom=False)

    # params = {
    #         # 'legend.fontsize': 'x-large',
    #         # 'figure.figsize': (15, 5),
    #         # 'axes.labelsize': 30,
    #         # 'axes.titlesize':'x-large',
    #         # 'xtick.labelsize':'x-large',
    #         # 'ytick.labelsize':'x-large'
    #         }
    # plt.rcParams.update(params)


    save_fig(fig, "heatmap_correlation_theta_posteriors")



def make_heatmap_correlation_distinct_errors(cluster_stats: List[Tuple[MulticlusterData, Path]], remove_nonexistence: bool = False):
    exact_marginals = []
    lbp_marginals = []

    for cluster_stat, _ in cluster_stats:
        if not cluster_stat.exact_computation_error:
            if remove_nonexistence:
                cluster_stat.exact_marginals = cluster_stat.exact_marginals.nonexistence_removed()
                cluster_stat.mhlbp_marginals = cluster_stat.mhlbp_marginals.nonexistence_removed()

            exact_marginals.append(cluster_stat.exact_marginals)
            lbp_marginals.append(cluster_stat.mhlbp_marginals)


    exact_marginals: Marginals = Marginals.concatenate(exact_marginals)
    lbp_marginals: Marginals = Marginals.concatenate(lbp_marginals)

    marginals_to_compare = [
        (exact_marginals.misdetection_marginals, lbp_marginals.misdetection_marginals),
        (exact_marginals.detection_marginals, lbp_marginals.detection_marginals),
        (exact_marginals.nonexistence_marginals, lbp_marginals.nonexistence_marginals)
    ]

    marginal_names = ["Misdetection", "Detection", "Nonexistence"]

    figsize = (8, 16)

    nrows = len(marginals_to_compare)
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, sharex=True)
    for k, (axx, (exact_marg, approx_marg), marginal_name) in enumerate(zip(ax, marginals_to_compare, marginal_names)):
        num_bins = 200
        xedges = np.linspace(0, 1, num_bins)
        yedges = xedges
        bins = (xedges, yedges)
        heatmap_w, xedges, yedges = np.histogram2d(approx_marg, exact_marg, bins=bins)
        X_w, Y_w = np.meshgrid(xedges[:-1], yedges[:-1])

        df = pd.DataFrame({
            "MCMH LBP": np.around(X_w.ravel(), decimals=3),
            "Exact marginals": np.around(Y_w.ravel(), decimals=3),
            "hist": heatmap_w.ravel()
        })
        df = df.pivot(index="Exact marginals", columns="MCMH LBP", values="hist")

        sns.heatmap(df, square=True, norm=LogNorm(), cmap="Oranges", ax=axx)
        axx.invert_yaxis()
        axx.set_title(marginal_name)
        if k < nrows - 1:
            axx.tick_params(bottom=False)

    fig_name = "heatmap_correlation_distinct_margs"
    if remove_nonexistence:
        fig_name += "_nonexistence_removed"

    save_fig(fig, fig_name)


def subsample(a: np.ndarray, inc: float, first_val=0) -> np.ndarray:
    """
    Returns the indices that subsamples the array.
    """
    idxs = [0]
    k = 0
    while a[k] < first_val:
        k += 1
    idxs.append(k)

    for k in range(idxs[0], len(a)):
        if a[k] - a[idxs[-1]] > inc:
            idxs.append(k)

    return idxs



def plot_survival_function(axes: plt.Axes, marginals_errors: MarginalsErrors, label: str = "_"):
    max_errors = np.sort(marginals_errors.max_errors)
    abs_errors = np.sort(marginals_errors.abs_errors)
    misdetection_errors = np.sort(marginals_errors.misdetection_errors)
    detection_errors = np.sort(marginals_errors.detection_errors)
    nonexistence_errors = np.sort(marginals_errors.nonexistence_errors)

    errors = [
        max_errors,
        abs_errors,
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

    for k, (ax, error, title) in enumerate(zip(axes, errors, titles)):
        ax.set_title(title, fontsize=20)
        steps = np.linspace(1.0, 0.0, len(error))
        first_nonzero = np.where(error > 0)[0][0]
        first_val = error[first_nonzero]
        idxs = subsample(np.log10(error), 1e-10, first_val=np.log10(first_val)) # Use -5 as first val since we are logarithmic
        print(f"Subsampling {title} for {label} reduced data to {len(idxs)/len(error)*100.0:.3f}%")
        error = error[idxs]
        steps = steps[idxs]
        ax.step(error, steps, label=label)
        print(error[1])
        ax.set_xscale('symlog', linthresh=first_val)
        log_err = np.linspace(np.log10(error[1]), np.log10(error[-1]*1.1), 9).astype(int)
        xticks = np.array([0, *((10.0)**log_err)])
        print(f"Using xticks {xticks}")
        ax.set_xticks(xticks)
        xticks_labels = [0] + [rf'$10^{{{l}}}$' for l in log_err]
        ax.set_xticklabels(xticks_labels)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        # ax.set_yscale('symlog', linthresh=steps[1])
        ax.semilogy()
        ax.grid(True, alpha=0.2)
        if label != "_" and k == 0:
            ax.legend(fontsize=13)


def condense_stats(cluster_stats: List[Tuple[ClusterData, Path]]):
    stats = defaultdict(list)

    # IoU of hypotheses?
    for cluster_stat, cluster_file in tqdm(cluster_stats):
        # intersection_tracks = set()
        # union_tracks = set()
        # mat_file = sl.MatFileParser(cluster_file_to_mat_file(cluster_file))
        # c_idx = cluster_file_to_cluster_idx(cluster_file)
        # for tracks, p in mat_file.prior_hypotheses_per_cluster[c_idx]:
        #     s_tracks = set(tracks)
        #     intersection_tracks = intersection_tracks & s_tracks
        #     union_tracks = union_tracks | s_tracks

        # IoU = len(intersection_tracks) / len(union_tracks)
        # stats["IoU"].append(IoU)
        pass

    
    return stats


def compare_converge_not_converge(cluster_stats: List[Tuple[ClusterData, Path]]):
    cluster_stats_converged = []
    cluster_stats_diverged = []

    mh_lbp_solver: LBPMarginalsFullAssociation = LBPMarginalsFullAssociation()

    for cluster_stat, cluster_file in cluster_stats:
        if cluster_stat.lbp_stats.converged:
            cluster_stats_converged.append((cluster_stat, cluster_file))
        else:
            mat_file = sl.MatFileParser(cluster_file_to_mat_file(cluster_file))
            c_idx = cluster_file_to_cluster_idx(cluster_file)
            R_LC = mat_file.reward_matrix_lc
            (asso_prob, (it, msg_it, converged)) = mh_lbp_solver.compute_marginals(R_LC, mat_file.prior_hypotheses_per_cluster[c_idx])
            print(converged)
            if converged:
                print(f"Converged!!")
                break
            cluster_stats_diverged.append((cluster_stat, cluster_file))

    # converged_stats = condense_stats(cluster_stats_converged)
    # diverged_stats = condense_stats(cluster_stats_diverged)

    # # print(np.mean(converged_stats["IoU"]))
    # print(np.mean(diverged_stats["IoU"]))

@dataclass
class BethePlotData:
    constants: np.ndarray
    label: str


def normalization_constant_scatter_plot(approx_normalization_constants: List[BethePlotData], exact_normalization_constants: np.ndarray):
    fig, ax = plt.subplots()
    # ax.plot(phd_normalization_constants, exact_normalization_constants, 'o', alpha=0.2, label="PHD")

    for approx_consts in approx_normalization_constants:
        ax.plot(approx_consts.constants, exact_normalization_constants, 'o', alpha=0.4, label=approx_consts.label, ms=10)

    ax.plot(exact_normalization_constants, exact_normalization_constants, '--', label="Perfect correlation")
    ax.set_xlabel("Approximate normalization constant", fontsize=18)
    ax.set_ylabel("Exact normalization constant", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)
    ax.loglog()

    # x = [l.constants.min() for l in approx_normalization_constants]
    # y = exact_normalization_constants
    # Determine tick values based on data range
    # data_min = min(np.min(x), np.min(y))
    # data_max = max(np.max(x), np.max(y))
    # ticks = np.logspace(np.floor(np.log10(data_min)), np.ceil(np.log10(data_max)), 5)

    # # Set equal tick labels on both axes
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(['$10^{%d}$' % np.log10(v) for v in ticks], fontsize=18)
    # ax.set_yticklabels(['$10^{%d}$' % np.log10(v) for v in ticks], fontsize=18)
    

    save_fig(fig, "normalization_constant")

def print_raw_error_stats(cluster_stats: List[Tuple[ClusterData, Path]]):
    lbp_errors, williams_errors, williams_errors_exact = cluster_stats_to_errors(cluster_stats, add_williams_exact=True)

    print(f"lbp mean error:\n{lbp_errors.raw_errors.mean():.4e} {lbp_errors.raw_errors.std():.4f}")
    print(f"williams mean error:\n{williams_errors.raw_errors.mean():.4e} {williams_errors.raw_errors.std():.4f}")
    print(f"williams exact mean error\n{williams_errors_exact.raw_errors.mean():.4e} {williams_errors_exact.raw_errors.std():.4f}")


def load_cluster_stats(path: str = OUTPUT_PATH_BASE, return_empty_clusters: bool = False, num_files_process: Optional[int] = None):
    load_dirs = Path(path).glob("**/*")

    load_dirs = list(load_dirs)
    num_files = len(load_dirs)

    if not num_files_process is None:
        if num_files_process > num_files:
            print(f"Asked to process {num_files_process}, but only {num_files} available! Processing {num_files}")
            num_files_process = num_files

        load_dirs = load_dirs[:num_files_process]
        assert len(load_dirs) == num_files_process
        num_files = num_files_process

    cluster_stats: List[Tuple[MulticlusterData, Path]] = []
    if return_empty_clusters:
        empty_clusters: List[Tuple[MulticlusterData, Path]] = []
    print(f"Looping over {len(load_dirs)} files!")
    for cluster_file in tqdm(load_dirs, total=num_files):
        if cluster_file.name != "empty_cluster":
            cluster_stats.append(
                (MulticlusterData.from_data(cluster_file), cluster_file)
            )
        if return_empty_clusters and cluster_file.name == "empty_cluster":
            empty_clusters.append(
                (MulticlusterData.from_data(cluster_file), cluster_file)
            )

    return cluster_stats


def load_cluster_stats_batch(path: str, batch_start: int, batch_stop: int):
    load_dirs = Path(path).glob("**/*")

    load_dirs = list(load_dirs)
    load_dirs = load_dirs[batch_start:batch_stop]
    num_files = len(load_dirs)

    cluster_stats: List[Tuple[MulticlusterData, Path]] = []
    for cluster_file in tqdm(load_dirs, total=num_files):
        if cluster_file.name != "empty_cluster":
            cluster_stats.append(
                (MulticlusterData.from_data(cluster_file), cluster_file)
            )

    return cluster_stats

async def load_cluster_stats_async(path: str = OUTPUT_PATH_BASE, num_files_process: Optional[int] = None):
    load_dirs = Path(path).glob("**/*")

    load_dirs = list(load_dirs)
    num_files = len(load_dirs)

    if not (num_files_process is None):
        if num_files_process > num_files:
            print(f"Asked to process {num_files_process}, but only {num_files} available! Processing {num_files}")
            num_files_process = num_files

        load_dirs = load_dirs[:num_files_process]
        num_files = num_files_process

    tasks = []
    async def task(cluster_file):
        return (MulticlusterData.from_data(cluster_file), cluster_file)

    for cluster_file in load_dirs:
        if cluster_file.name != "empty_cluster":
            tasks.append(asyncio.create_task(task(cluster_file)))

    cluster_stats: List[Tuple[MulticlusterData, Path]] = []
    # cluster_stats: List[Tuple[MulticlusterData, Path]] = await asyncio.gather(*tasks)

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        cluster_stats.append(result)

    return cluster_stats


def make_batch_intervals(batch_size: int):
    start = 0
    stop = batch_size - 1
    batch_intervals = []
    while stop <= 10_000:
        batch_intervals.append((start, stop))
        start = stop + 1
        stop = start + batch_size - 1

    stop = batch_intervals[-1][-1]
    if stop <= 9999:
        start = stop + 1
        stop = 10_000
        batch_intervals.append((start, stop))

    return batch_intervals

if __name__ == "__main__":
    from ravens_parser_parallell_multicluster import OUTPUT_PATH_BASE as path
    # path += "_last"
    print(f"Plotting data in {path}")
    batch_intervals = make_batch_intervals(2000)
    williams_approx_normalization_constants = np.empty(10_000, dtype=np.float32)
    phd_approx_normalization_constants = np.empty(10_000, dtype=np.float32)
    mc_eff_mhlbp_approx_normalization_constants = np.empty(10_000, dtype=np.float32)

    mcmhlbp_approx_normalization_constants = np.empty(10_000, dtype=np.float32)

    exact_normalization_constants = np.empty(10_000, dtype=np.float32)
    k = 0
    for batch_start, batch_stop in tqdm(batch_intervals):
        cluster_stats_batch = load_cluster_stats_batch(path, batch_start, batch_stop)
        for (cluster_stat, _) in tqdm(cluster_stats_batch):
            exact_normalization_constants[k] = cluster_stat.exact_output.exact_normalization_constant
            # mcmhlbp_approx_normalization_constants[k] = cluster_stat.mhlbp_output.bethe_pseudodual_normalization_constant()
            # williams_approx_normalization_constants[k] = cluster_stat.mc_bethe_output.likelihood
            # mc_eff_mhlbp_approx_normalization_constants[k] = cluster_stat.mc_mhlbp_output.likelihood
            phd_approx_normalization_constants[k] = cluster_stat.mc_phd_output.likelihood
            k += 1
        
        del cluster_stats_batch

    num_files = k

    mcmhlbp_approx_normalization_constants = BethePlotData(constants=mcmhlbp_approx_normalization_constants[:num_files], label="MCMH-LBP")
    williams_approx_normalization_constants = BethePlotData(constants=williams_approx_normalization_constants[:num_files], label="Approx Efficient Bethe")
    phd_approx_normalization_constants = BethePlotData(constants=phd_approx_normalization_constants[:num_files], label="Approx Efficient PHD")
    mc_eff_mhlbp_approx_normalization_constants = BethePlotData(constants=mc_eff_mhlbp_approx_normalization_constants[:num_files], label="Efficient MHLBP")

    exact_normalization_constants = exact_normalization_constants[:num_files]

    approx_normalization_constants = [mcmhlbp_approx_normalization_constants, williams_approx_normalization_constants, phd_approx_normalization_constants, mc_eff_mhlbp_approx_normalization_constants]
    approx_normalization_constants = [phd_approx_normalization_constants]

    # print(illegal_files)
    # make_raw_error_plot(cluster_stats)
    # make_divergence_comparison_plot(cluster_stats)
    # make_scatter_compare_plot(cluster_stats)
    # make_heatmap_correlation(cluster_stats)
    # compare_mhlbp_lbpphd(cluster_stats)
    # compare_converge_not_converge(cluster_stats)
    normalization_constant_scatter_plot(approx_normalization_constants, exact_normalization_constants)
    # correlation_plot_theta_posterior(cluster_stats)
    # make_conditioned_survival_function_plots(cluster_stats)
    # print_raw_error_stats(cluster_stats)
    # make_survival_function_plots(cluster_stats)
    # make_heatmap_correlation_distinct_errors(cluster_stats)
