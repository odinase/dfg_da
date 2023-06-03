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
from multicluster_analysis import result_path_to_mat_file_string


from dfg_da.stats_logger import MarginalsErrors, Marginals, ClusterData, MulticlusterData
import dfg_da.stats_logger as sl
from dfg_da.marginals_computers import LBPMarginalsFullAssociation
from ravens_parser_parallell import OUTPUT_PATH_BASE, PMBM_DATA_PATH



MURTY_OUTPUT = "./murty_output"
FIGURES_PATH = "./figures"




@dataclass
class BethePlotData:
    constants: np.ndarray
    label: str






def save_fig_to_pdf(fig, fig_name, tight_layout=True):
    if tight_layout:
        fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/results/{fig_name}.pdf", bbox_inches='tight')
    print(f"Saved {FIGURES_PATH}/results/{fig_name}.pdf")


def save_fig_to_png(fig, fig_name, tight_layout=True):
    if tight_layout:
        fig.tight_layout()
    fig.savefig(f"{FIGURES_PATH}/results/{fig_name}.png", bbox_inches='tight', dpi=600)
    print(f"Saved {FIGURES_PATH}/results/{fig_name}.png")


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



def make_survival_function_plots(williams_approx_marginals: List[sl.Marginals], phd_approx_marginals: List[sl.Marginals], mcmhlbp_marginals: List[sl.Marginals], mc_eff_mhlbp_marginals: List[sl.Marginals], exact_marginals: List[sl.Marginals], murty_marginals: List[sl.Marginals]):
    fig, ax = plt.subplots(nrows=5, figsize=(7, 12), sharex=True)

    # fig.suptitle("Survival functions")
    # williams_approx_marginals: sl.Marginals = sl.Marginals.concatenate(williams_approx_marginals)
    # phd_approx_marginals: sl.Marginals = sl.Marginals.concatenate(phd_approx_marginals)
    # mcmhlbp_marginals: sl.Marginals = sl.Marginals.concatenate(mcmhlbp_marginals)
    # mc_eff_mhlbp_marginals: sl.Marginals = sl.Marginals.concatenate(mc_eff_mhlbp_marginals)
    # exact_marginals: sl.Marginals = sl.Marginals.concatenate(exact_marginals)

    approximate_data: List[Tuple[sl.Marginals, str]] = [
        (williams_approx_marginals, "Approximate Efficient Bethe"),
        (phd_approx_marginals, "Approximate Efficient PHD"),
        (mcmhlbp_marginals, "MCMH-LBP"),
        (mc_eff_mhlbp_marginals, "Efficient MHLBP"),
        (murty_marginals, "Murty")
    ]
    for approx_margs, approx_name in approximate_data:
        approx_errors = sl.MarginalsErrors.concatenate([sl.MarginalsErrors(exact_marginals=e, approx_marginals=m) for e, m in zip(exact_marginals, approx_margs)])
        plot_survival_function(ax, approx_errors, label=approx_name)
    
    save_fig(fig, "sf")

@dataclass
class SurvivalPlotData:
    marginals: List[sl.Marginals]
    label: str


def make_survival_function_plots_generic(exact_marginals: List[sl.Marginals], approximate_data: List[SurvivalPlotData]):
    fig, ax = plt.subplots(nrows=5, figsize=(7, 12), sharex=True)

    for ap in approximate_data:
        approx_errors = sl.MarginalsErrors.concatenate([sl.MarginalsErrors(exact_marginals=e, approx_marginals=m) for e, m in zip(exact_marginals, ap.marginals)])
        plot_survival_function(ax, approx_errors, label=ap.label)
    
    save_fig(fig, "sf_generic")


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


def make_histogram_normconsts_rel_error(exact_normalization_constants: np.ndarray, approx_normalization_constants: List[BethePlotData], figure_name: str = "rel_error_normconsts_hist"):    
    alphas = np.ones(len(approx_normalization_constants)) / 4.0
    
    bins = 100
    hist_figsize = (10, 7)

    fig2, ax2 = plt.subplots(figsize=hist_figsize)

    for data, alpha in zip(approx_normalization_constants, alphas):
        err = (exact_normalization_constants - data.constants) / exact_normalization_constants
        ax2.hist(err, bins=bins, label=data.label, alpha=alpha)


    ax2.set_title("Histogram over relative normalization constants errors, (exact - approximate) / exact", fontsize=20)
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
    save_fig(fig2, figure_name)


def make_boxplot_normconsts_rel_error(exact_normalization_constants: np.ndarray, approx_normalization_constants: List[BethePlotData], figure_name: str = "rel_error_normconsts_boxplot"):    
    relative_errors = []

    for data in approx_normalization_constants:
        relative_error = (exact_normalization_constants - data.constants) / exact_normalization_constants
        relative_errors.append(relative_error)

    fig2, ax2 = plt.subplots()
    ax2.boxplot(relative_errors)
    ax2.set_title("Boxplot over relative normalization constants errors, (exact - approximate) / exact", fontsize=20)
    ax2.set_xlabel("Approximation", fontsize=18)
    ax2.set_ylabel("Relative Error", fontsize=18)
    ax2.set_title("Relative Error of Normalization Constants", fontsize=18)
    ax2.set_xticks(range(1, len(approx_normalization_constants) + 1))
    ax2.set_xticklabels([data.label for data in approx_normalization_constants], rotation=30)
    ax2.grid(True)
    ax2.semilogy()
    # ax2.boxplot(relative_errors)
    # ax2.set_xlabel("Approximation", fontsize=18)
    # ax2.set_ylabel("Relative Error", fontsize=18)
    # ax2.set_title("Relative Error of Normalization Constants", fontsize=18)
    # ax2.xticks(range(1, len(approx_normalization_constants) + 1), [data.label for data in approx_normalization_constants])
    # ax2.grid(True)


    # for data, alpha in zip(approx_normalization_constants, alphas):
    #     err = (exact_normalization_constants - data.constants) / exact_normalization_constants
    #     ax2.hist(err, bins=bins, label=data.label, alpha=alpha)


    # ax2.set_title("Histogram over relative normalization constants errors, (exact - approximate) / exact", fontsize=20)
    # ax2.semilogy()
    # ax2.legend(fontsize=10)
    # ax2.tick_params(axis='both', which='major', labelsize=18)
    # ax2.tick_params(axis='both', which='minor', labelsize=18)


    # fig3, ax3 = plt.subplots(figsize=hist_figsize)

    # x = lbp_errors.raw_errors
    # ax3.hist(x, bins=bins, label=mh_lbp_label, alpha=0.5)

    # ax3.set_title("Histogram over signed marginal errors")
    # ax3.semilogy()
    # ax3.legend()


    # save_fig_to_pdf(fig, "signed_error")
    save_fig(fig2, figure_name)



def make_raw_error_plot_hypotheses_posterior(williams_approx_theta_posteriors,
        phd_approx_theta_posteriors,
        mc_eff_mhlbp_theta_posteriors,
        mcmhlbp_theta_posteriors,
        exact_theta_posteriors):

    williams_approx_theta_posteriors = np.hstack(williams_approx_theta_posteriors)
    phd_approx_theta_posteriors = np.hstack(phd_approx_theta_posteriors)
    mc_eff_mhlbp_theta_posteriors = np.hstack(mc_eff_mhlbp_theta_posteriors)
    mcmhlbp_theta_posteriors = np.hstack(mcmhlbp_theta_posteriors)
    exact_theta_posteriors = np.hstack(exact_theta_posteriors)

    approximate_data: List[Tuple[sl.Marginals, str]] = [
        (williams_approx_theta_posteriors, "Approximate Efficient Bethe"),
        (phd_approx_theta_posteriors, "Approximate Efficient PHD"),
        (mc_eff_mhlbp_theta_posteriors, "MCMH-LBP"),
        (mcmhlbp_theta_posteriors, "Efficient MHLBP")
    ]
    
    alphas = np.ones(4) / 4.0
    alphas[-1] = 1 - alphas[:-1].sum() # Ensure it sums to one
    
    bins = 100
    hist_figsize = (10, 7)

    fig2, ax2 = plt.subplots(figsize=hist_figsize)

    for (data, label), alpha in zip(approximate_data, alphas):
        err = exact_theta_posteriors - data
        ax2.hist(err, bins=bins, label=label, alpha=alpha)


    ax2.set_title("Histogram over signed hypotheses posterior errors, exact - approximate", fontsize=20)
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
    save_fig(fig2, "signed_error_histogram_hypotheses_posterior")
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


def make_heatmap_correlation(williams_approx_marginals: List[sl.Marginals], phd_approx_marginals: List[sl.Marginals], mcmhlbp_marginals: List[sl.Marginals], mc_eff_mhlbp_marginals: List[sl.Marginals], exact_marginals: List[sl.Marginals], murty_marginals: List[sl.Marginals]):

    williams_approx_marginals: sl.Marginals = sl.Marginals.concatenate(williams_approx_marginals)
    phd_approx_marginals: sl.Marginals = sl.Marginals.concatenate(phd_approx_marginals)
    mcmhlbp_marginals: sl.Marginals = sl.Marginals.concatenate(mcmhlbp_marginals)
    mc_eff_mhlbp_marginals: sl.Marginals = sl.Marginals.concatenate(mc_eff_mhlbp_marginals)
    exact_marginals: sl.Marginals = sl.Marginals.concatenate(exact_marginals)
    murty_marginals: sl.Marginals = sl.Marginals.concatenate(murty_marginals)

    approximate_data: List[Tuple[sl.Marginals, str]] = [
        (williams_approx_marginals, "Approximate Efficient Bethe"),
        (phd_approx_marginals, "Approximate Efficient PHD"),
        (mcmhlbp_marginals, "MCMH-LBP"),
        (mc_eff_mhlbp_marginals, "Efficient MHLBP"),
        (murty_marginals, "Murty")
    ]

    num_bins = 200
    xedges = np.linspace(0, 1, num_bins)
    yedges = xedges
    bins = (xedges, yedges)

    dfs = []

    for approx_margs, approx_name in approximate_data:
        heatmap_lbp, xedges, yedges = np.histogram2d(approx_margs.marginals, exact_marginals.marginals, bins=bins)
        X_lbp, Y_lbp = np.meshgrid(xedges[:-1], yedges[:-1])

        df_approx = pd.DataFrame({
            approx_name: np.around(X_lbp.ravel(), decimals=3),
            "Exact marginals": np.around(Y_lbp.ravel(), decimals=3),
            "hist": heatmap_lbp.T.ravel()
        })
        df_approx = df_approx.pivot(index="Exact marginals", columns=approx_name, values="hist")
        dfs.append(df_approx)

    dfs.insert(1, None)

    figsize = (8, 8)

    nrows = 3
    ncols = 2
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    num_ticks = 5
    for k, (axx, df) in enumerate(zip(ax.ravel(), dfs)):
        if k == 1:
            fig.delaxes(axx)
            continue

        sns.heatmap(df, square=True, norm=LogNorm(), cmap="Reds", ax=axx)

        axx.tick_params(axis='both', which='major', labelsize=16)
        axx.tick_params(axis='both', which='minor', labelsize=16)
        cbar = axx.collections[0].colorbar
        axx.set_ylabel(df.index.name, fontsize=18)
        axx.set_xlabel(df.columns.name, fontsize=16)
        cbar.ax.tick_params(labelsize=18)
        axx.invert_yaxis()
        r, c = np.unravel_index(k, (nrows, ncols))
        if r < nrows - 1:
            axx.tick_params(bottom=False)

        # if (c == 1) and (r == nrows - 2):
        #     axx.xaxis.set_tick_params(labelbottom=True)
        #     axx.tick_params(axis='x', rotation=90)

    
    # Remove the last axis from the grid
    # fig.delaxes(ax.ravel()[-1])

    save_fig(fig, "heatmap_correlation")


def make_heatmap_correlation_theta(
        williams_approx_theta_posteriors: List[np.ndarray],
        phd_approx_theta_posteriors: List[np.ndarray],
        mc_eff_mhlbp_theta_posteriors: List[np.ndarray],
        mcmhlbp_theta_posteriors: List[np.ndarray],
        exact_theta_posteriors: List[np.ndarray]
    ):

    williams_approx_marginals: np.ndarray = np.hstack(williams_approx_theta_posteriors)
    phd_approx_marginals: np.ndarray = np.hstack(phd_approx_theta_posteriors)
    mcmhlbp_marginals: np.ndarray = np.hstack(mcmhlbp_theta_posteriors)
    mc_eff_mhlbp_marginals: np.ndarray = np.hstack(mc_eff_mhlbp_theta_posteriors)
    exact_marginals: np.ndarray = np.hstack(exact_theta_posteriors)

    approximate_data: List[Tuple[sl.Marginals, str]] = [
        (williams_approx_marginals, "Approximate Efficient Bethe"),
        (phd_approx_marginals, "Approximate Efficient PHD"),
        (mcmhlbp_marginals, "MCMH-LBP"),
        (mc_eff_mhlbp_marginals, "Efficient MHLBP")
    ]

    num_bins = 100
    xedges = np.linspace(0, 1, num_bins)
    yedges = xedges
    bins = (xedges, yedges)

    dfs = []

    for approx_margs, approx_name in approximate_data:
        heatmap_lbp, xedges, yedges = np.histogram2d(approx_margs, exact_marginals, bins=bins)
        X_lbp, Y_lbp = np.meshgrid(xedges[:-1], yedges[:-1])

        df_approx = pd.DataFrame({
            approx_name: np.around(X_lbp.ravel(), decimals=3),
            "Exact marginals": np.around(Y_lbp.ravel(), decimals=3),
            "hist": heatmap_lbp.T.ravel()
        })
        df_approx = df_approx.pivot(index="Exact marginals", columns=approx_name, values="hist")
        dfs.append(df_approx)

    figsize = (8, 8)

    nrows = 2
    ncols = 2
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    num_ticks = 5
    depth_list = np.linspace(0, 1, num_ticks)
    # the index of the position of yticks
    # yticks = np.arange(0, num_bins, num_bins // num_ticks)
    # xticks = yticks
    # # # the content of labels of these yticks
    # yticklabes = np.linspace()
    # xticklabes = yticklabes
    for k, (axx, df) in enumerate(zip(ax.ravel(), dfs)):
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


    save_fig(fig, "heatmap_correlation_theta_posterior")


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
        # first_nonzero = np.where(error > 0)[0][0]
        first_val = 1e-6#error[first_nonzero]
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
        ax.set_yscale('symlog', linthresh=1e-2)
        # ax.semilogy()
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


def normalization_constant_scatter_plot(approx_normalization_constants: List[BethePlotData], exact_normalization_constants: np.ndarray, figure_name: str = "normalization_constant"):
    fig, ax = plt.subplots()
    # ax.plot(phd_normalization_constants, exact_normalization_constants, 'o', alpha=0.2, label="PHD")

# plt.hexbin(x, y, gridsize=20, cmap='Blues', alpha=0.8)

    for approx_consts in approx_normalization_constants:
        ax.plot(approx_consts.constants, exact_normalization_constants, 'o', alpha=0.05, label=approx_consts.label)
        # hb = ax.hexbin(approx_consts.constants, exact_normalization_constants, alpha=0.2, label=approx_consts.label)

    # cbar = fig.colorbar(hb)


    ax.plot(exact_normalization_constants, exact_normalization_constants, '--', label="Perfect correlation")
    ax.set_xlabel("Approximate normalization constant", fontsize=18)
    ax.set_ylabel("Exact normalization constant", fontsize=18)
    ax.grid(True, alpha=0.3)
    leg = ax.legend(fontsize=14)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    ax.loglog()

    x = np.array([(l.constants.min(), l.constants.max()) for l in approx_normalization_constants])
    y = exact_normalization_constants
    # Determine tick values based on data range
    data_min = min(np.min(x), np.min(y))
    data_max = max(np.max(x), np.max(y))
    ticks = np.logspace(np.floor(np.log10(data_min)), np.ceil(np.log10(data_max)), 5)

    # Set equal tick labels on both axes
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(['$10^{%d}$' % np.log10(v) for v in ticks], fontsize=18)
    ax.set_yticklabels(['$10^{%d}$' % np.log10(v) for v in ticks], fontsize=18)
    

    save_fig(fig, figure_name)



def plot_iterations(bethe_msg_norm_iters: np.ndarray, bethe_msg_norm_errors: np.ndarray):
    assert bethe_msg_norm_iters.shape[0] == 2, f"Needs to be (2, N), is now {bethe_msg_norm_iters.shape}"
    assert bethe_msg_norm_errors.shape[0] == 2, f"Needs to be (2, N), is now {bethe_msg_norm_iters.shape}"

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2)

    axes[0].plot(bethe_msg_norm_iters[0], "g", alpha=0.7, label="Bethe iters")
    axes[0].plot(bethe_msg_norm_iters[1], "b", alpha=0.7, label="Message norm iters")

    axes[0].semilogy()
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Number of iterations")

    axes[0].legend()

    axes[1].plot(bethe_msg_norm_errors[0], "g", alpha=0.7, label="Bethe error at convergence")
    axes[1].plot(bethe_msg_norm_errors[1], "b", alpha=0.7, label="Message norm error at convergence")

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Error")

    axes[1].legend()


    save_fig(fig, "iterations_errrs_mcmhlbp")

    failed_converge = (bethe_msg_norm_iters == 10_000).any(0)
    number_of_failed_convergences = failed_converge.sum()
    if number_of_failed_convergences > 0:
        fig2, ax2 = plt.subplots()
        ax2.plot(bethe_msg_norm_iters[0, failed_converge], "g", label="Bethe iters")
        ax2.plot(bethe_msg_norm_iters[1, failed_converge], "b", label="Message norm iters")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Number of iterations")
        ax2.set_title(f"Number of failed convergence cases: {number_of_failed_convergences}")
        print(f"Errors at failed convergence: {bethe_msg_norm_errors[:, failed_converge]}")

        save_fig(fig2, "failed_convergence_iterations")
        


def overestimated_bethe(bethe_constants: np.ndarray, exact_constants: np.ndarray):
    large_bethe = bethe_constants > exact_constants
    
    bethe_constants = bethe_constants[large_bethe]
    exact_constants = exact_constants[large_bethe]
    
    fig, ax = plt.subplots()

    relative_largeness = ((bethe_constants - exact_constants) / exact_constants)*100.0

    ax.hist(relative_largeness)
    ax.set_title(f"Number of overestimated Bethe: {large_bethe.sum()}")

    save_fig(fig, "overestimated_bethe")



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


def load_cluster_stats_batch(load_dirs: List[str], batch_start: int, batch_stop: int):
    load_dirs_batch = load_dirs[batch_start:batch_stop]

    cluster_stats: List[Tuple[MulticlusterData, Path]] = []
    for cluster_file in load_dirs_batch:
        if cluster_file.name != "empty_cluster":
            try:
                cluster_stats.append(
                    (MulticlusterData.from_data(cluster_file), cluster_file)
                )
            except EOFError:
                print(f"Encountered trouble with {cluster_file}, skipping...")
                continue

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
    stop = batch_size
    batch_intervals = []
    while stop <= 10_000:
        batch_intervals.append((start, stop))
        start = stop
        stop = start + batch_size

    stop = batch_intervals[-1][-1]
    if stop <= 9999:
        start = stop + 1
        stop = 10_000 + 1
        batch_intervals.append((start, stop))

    return batch_intervals


def murty_results_from_cluster_path(cluster_path: Path, K: int = 150) -> Tuple[np.ndarray, float]:
    from scipy.io import loadmat
    output_file_path = f"{MURTY_OUTPUT}/nHypoTotalMax_{K}/{Path(result_path_to_mat_file_string(cluster_path)).name}"
    murty_ws = loadmat(output_file_path)
    murty_norm = murty_ws["Z"][0,0]
    murty_margs = murty_ws["a"]

    return murty_margs, murty_norm


if __name__ == "__main__":
    from ravens_parser_parallell_multicluster import OUTPUT_PATH_BASE as path
    # path += " (7th copy)"
    print(f"Plotting data in {path}")
    batch_intervals = make_batch_intervals(500)
    williams_approx_normalization_constants = np.empty(10_000, dtype=np.float32)
    phd_approx_normalization_constants = np.empty(10_000, dtype=np.float32)
    mc_eff_mhlbp_approx_normalization_constants = np.empty(10_000, dtype=np.float32)
    mcmhlbp_approx_normalization_constants = np.empty(10_000, dtype=np.float32)
    murty_normalization_constants = np.empty(10_000, dtype=np.float32)

    exact_normalization_constants = np.empty(10_000, dtype=np.float32)

    williams_approx_marginals = []
    phd_approx_marginals = []
    mc_eff_mhlbp_marginals = []
    mcmhlbp_marginals = []
    exact_marginals = []
    murty_marginals = []

    williams_approx_theta_posteriors = []
    phd_approx_theta_posteriors = []
    mc_eff_mhlbp_theta_posteriors = []
    mcmhlbp_theta_posteriors = []
    exact_theta_posteriors = []

    larger_bethe_constant = []

    mcmhlbp_runtimes = np.empty(10_000, dtype=np.float32)
    exact_runtimes = np.empty(10_000, dtype=np.float32)

    load_dirs = Path(path).glob("**/*")
    load_dirs = sorted(list(load_dirs))
    # load_dirs = load_dirs[:100]

    cluster_paths = []
    # batch_intervals = [batch_intervals[0]]

    k = 0
    for batch_start, batch_stop in tqdm(batch_intervals):
        cluster_stats_batch = load_cluster_stats_batch(load_dirs, batch_start, batch_stop)
        for (cluster_stat, cluster_path) in tqdm(cluster_stats_batch):
            try:
                murty_margs, murty_norm = murty_results_from_cluster_path(cluster_path)
            except FileNotFoundError:
                print(f"File {cluster_path} not computed by Murty, skipping...")
                # This happens because Murty is run on a smaller set of the data, fix later
                continue


            exact_normalization_constants[k] = cluster_stat.exact_output.exact_normalization_constant
            mcmhlbp_approx_normalization_constants[k] = cluster_stat.mcmhlbp_output.approx_normalization_constant
            williams_approx_normalization_constants[k] = cluster_stat.mc_bethe_output.likelihood
            mc_eff_mhlbp_approx_normalization_constants[k] = cluster_stat.mc_mhlbp_output.likelihood
            phd_approx_normalization_constants[k] = cluster_stat.mc_phd_output.likelihood
            murty_normalization_constants[k] = murty_norm

            if cluster_stat.mcmhlbp_output.approx_normalization_constant > cluster_stat.exact_output.exact_normalization_constant:
                path = result_path_to_mat_file_string(cluster_path)
                print(f"File {path} contains too large Bethe!")
                larger_bethe_constant.append((cluster_stat, cluster_path))

            williams_approx_marginals.append(sl.Marginals(cluster_stat.mc_bethe_output.marginals))
            phd_approx_marginals.append(sl.Marginals(cluster_stat.mc_phd_output.marginals))
            mcmhlbp_marginals.append(sl.Marginals(cluster_stat.mcmhlbp_output.approx_marginals))
            mc_eff_mhlbp_marginals.append(sl.Marginals(cluster_stat.mc_mhlbp_output.marginals))
            exact_marginals.append(sl.Marginals(cluster_stat.exact_output.exact_marginals))
            murty_marginals.append(sl.Marginals(murty_margs))

            williams_approx_theta_posteriors.append(np.hstack(cluster_stat.mc_bethe_output.theta_posteriors))
            phd_approx_theta_posteriors.append(np.hstack(cluster_stat.mc_phd_output.theta_posteriors))
            mcmhlbp_theta_posteriors.append(np.hstack(cluster_stat.mcmhlbp_output.approx_theta_posteriors))
            mc_eff_mhlbp_theta_posteriors.append(np.hstack(cluster_stat.mc_mhlbp_output.theta_posteriors))
            exact_theta_posteriors.append(np.hstack(cluster_stat.exact_output.compute_theta_posteriors()))

            mcmhlbp_runtimes[k] = cluster_stat.mcmhlbp_output.runtime
            exact_runtimes[k] = cluster_stat.exact_output.runtime

            if (
                cluster_stat.mc_bethe_output.raised_warning or
                cluster_stat.mc_mhlbp_output.raised_warning or
                cluster_stat.mc_phd_output.raised_warning
            ):
                print("Found raised warning!")
                if cluster_stat.mc_bethe_output.raised_warning:
                    print("mc_bethe_output!")
                if cluster_stat.mc_mhlbp_output.raised_warning:
                    print("mc_mhlbp_output")
                if cluster_stat.mc_phd_output.raised_warning:
                    print("mc_mhlbp_output")

            cluster_paths.append(cluster_path)
            k += 1

        del cluster_stats_batch

    num_files = k

    print(f"Num cases too large Bethe: {len(larger_bethe_constant)}")
    print(f"Num files read: {num_files}")

    mcmhlbp_approx_normalization_constants: BethePlotData = BethePlotData(constants=mcmhlbp_approx_normalization_constants[:num_files], label="MCMH-LBP")
    williams_approx_normalization_constants: BethePlotData = BethePlotData(constants=williams_approx_normalization_constants[:num_files], label="Approx Efficient Bethe")
    phd_approx_normalization_constants = BethePlotData(constants=phd_approx_normalization_constants[:num_files], label="Approx Efficient PHD")
    mc_eff_mhlbp_approx_normalization_constants = BethePlotData(constants=mc_eff_mhlbp_approx_normalization_constants[:num_files], label="Efficient MHLBP")
    murty_normalization_constants = BethePlotData(constants=murty_normalization_constants[:num_files], label="Murty")

    exact_normalization_constants: np.ndarray = exact_normalization_constants[:num_files]

    # mcmhlbp_runtimes = mcmhlbp_runtimes[:num_files]
    # exact_runtimes = exact_runtimes[:num_files]

    # # williams_approx_normalization_constants.constants = williams_approx_normalization_constants.constants * exact_normalization_constants.min()/williams_approx_normalization_constants.constants.min()

    approx_normalization_constants = [
        phd_approx_normalization_constants,
        mc_eff_mhlbp_approx_normalization_constants,
        williams_approx_normalization_constants,
        mcmhlbp_approx_normalization_constants,
        murty_normalization_constants,
    ]

    approx_hypotheses_posteriors = [
        williams_approx_theta_posteriors,
        phd_approx_theta_posteriors,
        mc_eff_mhlbp_theta_posteriors,
        mcmhlbp_theta_posteriors,
    ]

    normalization_constant_scatter_plot(approx_normalization_constants, exact_normalization_constants)
    make_histogram_normconsts_rel_error(exact_normalization_constants, approx_normalization_constants)
    make_boxplot_normconsts_rel_error(exact_normalization_constants, approx_normalization_constants)
    make_heatmap_correlation(williams_approx_marginals, phd_approx_marginals, mcmhlbp_marginals, mc_eff_mhlbp_marginals, exact_marginals, murty_marginals)
    make_heatmap_correlation_theta(
        williams_approx_theta_posteriors,
        phd_approx_theta_posteriors,
        mc_eff_mhlbp_theta_posteriors,
        mcmhlbp_theta_posteriors,
        exact_theta_posteriors
    )
    make_survival_function_plots(williams_approx_marginals, phd_approx_marginals, mcmhlbp_marginals, mc_eff_mhlbp_marginals, exact_marginals, murty_marginals)
    
    make_raw_error_plot_hypotheses_posterior(williams_approx_theta_posteriors,
        phd_approx_theta_posteriors,
        mc_eff_mhlbp_theta_posteriors,
        mcmhlbp_theta_posteriors,
        exact_theta_posteriors
    )

    # fig, ax = plt.subplots()

    # print(f"MCMH-LBP runtime average: {mcmhlbp_runtimes.mean()}\u00B1{mcmhlbp_runtimes.std()} s\nMax: {mcmhlbp_runtimes.max()} s\nMin: {mcmhlbp_runtimes.min()} s")
    # print(f"Exact runtime average: {exact_runtimes.mean()}\u00B1{exact_runtimes.std()} s\nMax: {exact_runtimes.max()} s\nMin: {exact_runtimes.min()} s")
    # ax.plot(mcmhlbp_runtimes, "*--", label="MCMH-LBP runtime")
    # ax.plot(exact_runtimes, "o--", label="Exact runtime")
    # ax.semilogy()

    # ax.plot(mcmhlbp_iters[:,0], "--", label="Bethe iters")
    # ax.plot(mcmhlbp_iters[:,1], "--", label="Msg norm iters")
    # failed_converge = np.where(mcmhlbp_iters[:,0] == 10_000)[0]
    # if len(failed_converge) > 0:
    #     ax.plot(mcmhlbp_iters[failed_converge,0], "x")

    # ax.semilogy()

    # ax.legend()

    # plt.show()


    # print(illegal_files)
    # make_raw_error_plot(cluster_stats)
    # make_divergence_comparison_plot(cluster_stats)
    # make_scatter_compare_plot(cluster_stats)
    # compare_mhlbp_lbpphd(cluster_stats)
    # compare_converge_not_converge(cluster_stats)
    # correlation_plot_theta_posterior(cluster_stats)
    # make_conditioned_survival_function_plots(cluster_stats)
    # print_raw_error_stats(cluster_stats)
    # make_heatmap_correlation_distinct_errors(cluster_stats)


    N = len(cluster_paths)

    murty_output = {
        10:  ([], np.empty(N, dtype=np.float32)),
        20:  ([], np.empty(N, dtype=np.float32)),
        50:  ([], np.empty(N, dtype=np.float32)),
        100: ([], np.empty(N, dtype=np.float32)),
        150: ([], np.empty(N, dtype=np.float32)),
    }

    for k, cluster_path in enumerate(cluster_paths):
        for K in murty_output:
            murty_margs, murty_norm = murty_results_from_cluster_path(cluster_path, K=K)
            murty_output[K][0].append(sl.Marginals(murty_margs))
            murty_output[K][1][k] = murty_norm
    
    # def make_survival_function_plots_generic(exact_marginals: List[sl.Marginals], approximate_data: List[SurvivalPlotData]):

    sf_plot_data = []

    efficient_bethe_sf_data = SurvivalPlotData(marginals=williams_approx_marginals, label="Efficient Approx Bethe")

    sf_plot_data.append(efficient_bethe_sf_data)

    for K in murty_output:
        marginals = murty_output[K][0]
        label = f"{K}-Murty"
        sf_plot_data.append(SurvivalPlotData(marginals=marginals, label=label))

    make_survival_function_plots_generic(exact_marginals=exact_marginals, approximate_data=sf_plot_data)
    # normalization_constant_scatter_plot(approx_normalization_constants: List[BethePlotData], exact_normalization_constants: np.ndarray):
    norm_data = []

    for K in murty_output:
        constants = murty_output[K][1]
        label = f"{K}-Murty"
        norm_data.append(BethePlotData(constants=constants, label=label))

    norm_data.append(mcmhlbp_approx_normalization_constants)

    normalization_constant_scatter_plot(norm_data, exact_normalization_constants, figure_name="norm_consts_murtys")
# def make_histogram_normconsts_rel_error(exact_normalization_constants: np.ndarray, approx_normalization_constants: List[BethePlotData]):    
    make_histogram_normconsts_rel_error(exact_normalization_constants, norm_data, figure_name="rel_error_normconsts_hist_murtys")
