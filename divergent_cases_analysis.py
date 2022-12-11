import numpy as np
import matplotlib.pyplot as plt

from plotting import load_cluster_stats, split_cluster_stats_converged, cluster_file_to_mat_file, cluster_file_to_cluster_idx
from typing import List, Tuple
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from dfg_da.stats_logger import ClusterData, MatFileParser
import dfg_da.stats_logger as sl
import dfg_da.marginals_computers as mc
import dfg_da.prior_hypothesis as ph


class ClustersSummary:
    def __init__(self, cluster_stats: List[Tuple[ClusterData, Path]]) -> None:
        self.num_hypotheses = []
        self.num_failed_exact = 0
        self.num_data = len(cluster_stats)
        self.num_tracks = []

        for cluster_stat, cluster_file in cluster_stats:
            if cluster_stat.explicit_hypothesis_enumeration_error:
                self.num_failed_exact += 1
            
            self.num_hypotheses.append(cluster_stat.num_hypotheses)
            self.num_tracks.append(cluster_stat.cardinality)
        


def msg_plot_lbp(cluster_file: Path):
    mat_file = MatFileParser(cluster_file_to_mat_file(cluster_file))
    print([len(ph) for ph in mat_file.prior_hypotheses_per_cluster])
    lbp_solve = mc.LBPMarginalsFullAssociation()
    exact_solve = mc.ExactMarginals()
    c_idx = cluster_file_to_cluster_idx(cluster_file)
    prior_hypotheses = mat_file.prior_hypotheses_per_cluster[c_idx]

    asso_prob, (it, msg_it, converged), lbp_output = lbp_solve(mat_file.reward_matrix_lc, prior_hypotheses, lbp_output=True)
    exact_marginals, (exact_normalization_constants,) = exact_solve(mat_file.reward_matrix_lc, prior_hypotheses)

    fig, axes = plt.subplots(ncols=4)

    msg_names = ["track -- > hyp (rho)", "hyp -- > track (sigma)", "track -- > meas (mu)", "meas -- > track (nu)"]

    msg_minmaxs = lbp_output.minmaxs()

    for msg_minmax, msg_name, ax in zip(msg_minmaxs, msg_names, axes):
        mins, maxs = msg_minmax.T
        ax.plot(mins, label='min')
        ax.plot(maxs, label='max')
        ax.set_title(msg_name)
        ax.legend()

    # phs: ph.PriorHypotheses = mat_file.prior_hypotheses_per_cluster[c_idx]
    print(converged)

    num_hyps_to_see = 15
    # prior_hypotheses = mat_file.prior_hypotheses_per_cluster[c_idx]
    num_hyps_to_see = min(num_hyps_to_see, len(prior_hypotheses))
    print(len(prior_hypotheses))
    fig2, ax2 = plt.subplots(ncols=num_hyps_to_see)
    for (tracks, p), ax in zip(prior_hypotheses, ax2):
        t = np.sort(tracks)
        print(f"{t}: {p}")
        I = ax.imshow(np.exp(mat_file.reward_matrix_lc[t-1, :]))
        fig2.colorbar(I, ax=ax)

    fig3, ax3 = plt.subplots()
    ax3.plot(prior_hypotheses.hypothesis_probabilities())

    lbp_marginals: sl.Marginals = sl.Marginals(asso_prob)
    exact_marginals: sl.Marginals = sl.Marginals(exact_marginals)

    fig4, ax4 = plt.subplots()
    ax4.plot(lbp_marginals.misdetection_marginals, exact_marginals.misdetection_marginals, 'rx', label="Misdetection")
    ax4.plot(lbp_marginals.detection_marginals, exact_marginals.detection_marginals, 'gx', label="Detection")
    ax4.plot(lbp_marginals.nonexistence_marginals, exact_marginals.nonexistence_marginals, 'bx', label="Nonexistence")
    ax4.legend()

    plt.show()


def hypothesis_distributions(cluster_stats: List[Tuple[ClusterData, Path]]):
    hypothesis_dist = []
    for cluster_stat, cluster_file in cluster_stats:
        mat_file = sl.MatFileParser(cluster_file_to_mat_file(cluster_file))
        c_idx = cluster_file_to_cluster_idx(cluster_file)
        hypothesis_dist.append(mat_file.prior_hypotheses_per_cluster[c_idx].hypothesis_probabilities())

    return hypothesis_dist

def hypo_dist_loop_func(cluster_tuple):
    cluster_stat, cluster_file = cluster_tuple
    mat_file = sl.MatFileParser(cluster_file_to_mat_file(cluster_file))
    c_idx = cluster_file_to_cluster_idx(cluster_file)
    return mat_file.prior_hypotheses_per_cluster[c_idx].hypothesis_probabilities()

def hypothesis_distributions_async(cluster_stats: List[Tuple[ClusterData, Path]]):
    with Pool() as p:
        result = p.map(hypo_dist_loop_func, cluster_stats)

    return list(result)

def plot_hypotheses_distributions(cluster_stats: List[Tuple[ClusterData, Path]]):
    cluster_stats_converged, cluster_stats_diverged = split_cluster_stats_converged(cluster_stats)
    cluster_stats_lists = [cluster_stats_converged, cluster_stats_diverged]
    fig, axes = plt.subplots(nrows=2)
    labels = ["convergent", "divergent"]
    colors = ['b', 'r']
    hypos = None
    for cluster_st, label, c, ax in zip(cluster_stats_lists, labels, colors, axes):
        if len(cluster_st) > 1000:
            hypos = hypothesis_distributions_async(cluster_st)
        else:
            hypos = hypothesis_distributions(cluster_st)

        max_card = max(len(hypo) for hypo in hypos)
        num_hypo_distr = len(hypos)
        hypos_mat = np.zeros((num_hypo_distr, max_card))
        for k, dist in enumerate(hypos):
            i = len(dist)
            hypos_mat[k, :i] = np.sort(np.asarray(dist))[::-1]

        mean_prob = hypos_mat.mean(axis=0)
        std_prob = hypos_mat.std(axis=0)

        ax.plot(mean_prob, c, label=label)
        ax.plot(mean_prob + std_prob, c + "--")
        ax.plot(mean_prob - std_prob, c + "--")
        ax.fill_between(np.arange(len(mean_prob)), mean_prob - std_prob, mean_prob + std_prob, alpha=0.2)
        
    
        ax.legend()

    plt.show()


def plot_samples_reward_matices(cluster_stats_converged: List[Tuple[ClusterData, Path]], cluster_stats_diverged: List[Tuple[ClusterData, Path]], num_samples=5, seed=None):
    rng = np.random.default_rng(seed=seed)
    cluster_stats_converged_sample = rng.choice(cluster_stats_converged, size=num_samples, replace=False)
    cluster_stats_diverged_sample = rng.choice(cluster_stats_diverged, size=num_samples, replace=False)

    fig, axes = plt.subplots(nrows=num_samples, ncols=2)
    def imshow_reward_mat(cluster_stat, cluster_file, ax):
        mat_file: sl.MatFileParser = sl.MatFileParser(cluster_file_to_mat_file(cluster_file))
        tracks = np.sort(np.fromiter(cluster_stat.tracks, dtype=int))
        R = np.exp(mat_file.reward_matrix_lc)[tracks - 1, :]
        return ax.imshow(R, vmin=0, vmax=100)

    for ax, (cluster_stat_diverged, cluster_file_diverged), (cluster_stat_converged, cluster_file_converged) in zip(axes, cluster_stats_diverged_sample, cluster_stats_converged_sample):
        I1 = imshow_reward_mat(cluster_stat_diverged, cluster_file_diverged, ax[0])
        I2 = imshow_reward_mat(cluster_stat_converged, cluster_file_converged, ax[1])
        fig.colorbar(I1, ax=ax[0])
        fig.colorbar(I2, ax=ax[1])

    plt.show()


def plot_num_meas_num_tracks(cluster_stats_converged: List[Tuple[ClusterData, Path]], cluster_stats_diverged: List[Tuple[ClusterData, Path]]):
    def compute_num_tracks_meas(cluster_stats: List[Tuple[ClusterData, Path]]):
        num_tracks_meas = []
        for cluster_stat, cluster_file in tqdm(cluster_stats):
            n, mp2 = cluster_stat.lbp_stats.marginals.marginals_raw.shape
            m = mp2 - 2
            num_tracks_meas.append((n, m))
    
        return np.array(num_tracks_meas)
    
    num_tracks_meas_converged = compute_num_tracks_meas(cluster_stats_converged)
    num_tracks_meas_diverged = compute_num_tracks_meas(cluster_stats_diverged)

    fig, ax = plt.subplots(ncols=2, sharey=True)
    num_tms = [num_tracks_meas_converged, num_tracks_meas_diverged]
    titles = ["converged", "diverged"]
    for axx, num_tm, title in zip(ax, num_tms, titles):
        axx.plot(*num_tm.T, 'x')
        axx.set_xlabel("num tracks")
        axx.set_ylabel("num meas")
        axx.set_title(title)

    plt.show()


def test_all_divergent_cases(cluster_files_diverged: List[Path]):
    convergd_list = np.empty(len(cluster_files_diverged), dtype=bool)
    for k, cluster_file in tqdm(enumerate(cluster_files_diverged), total=len(cluster_files_diverged)):
        lbp_solve = mc.LBPMarginalsFullAssociation()
        mat_file = sl.MatFileParser(cluster_file_to_mat_file(cluster_file))
        c_idx = cluster_file_to_cluster_idx(cluster_file)
        prior_hypotheses = mat_file.prior_hypotheses_per_cluster[c_idx]
        asso_prob, (it, msg_it, converged) = lbp_solve(mat_file.reward_matrix_lc, prior_hypotheses)
        convergd_list[k] = converged

    num_converged = convergd_list.sum()
    print(f"Num converged: {num_converged}, {num_converged / len(cluster_files_diverged) * 100.0:.3f}%")

    

if __name__ == "__main__":
    cluster_stats = load_cluster_stats()

    cluster_stats_converged, cluster_stats_diverged = split_cluster_stats_converged(cluster_stats)

    # print(len(cluster_stats_converged))
    # print(len(cluster_stats_diverged))
    # print(cluster_stats_diverged[0])


    # cluster_summary_conv: ClustersSummary = ClustersSummary(cluster_stats_converged)
    # cluster_summary_div: ClustersSummary = ClustersSummary(cluster_stats_diverged)

    # fig, ax = plt.subplots()

    # ax.hist(cluster_summary_div.num_hypotheses, density=True, bins=100, alpha=0.5, label="Divergent")
    # ax.hist(cluster_summary_conv.num_hypotheses, density=True, bins=100, alpha=0.5, label="Convergent")

    # ax.set_title("Number of hypotheses")
    # ax.legend()

    # print(f"Num hypotheses:\n\tMax convergent: {np.max(cluster_summary_conv.num_hypotheses)}\n\tMin convergent: {np.min(cluster_summary_conv.num_hypotheses)}\n\tMean convergent: {np.mean(cluster_summary_conv.num_hypotheses)}\n\tMedian convergent: {np.median(cluster_summary_conv.num_hypotheses)}\n\n\tMax divergent: {np.max(cluster_summary_div.num_hypotheses)}\n\tMin divergent: {np.min(cluster_summary_div.num_hypotheses)}\n\tMean divergent: {np.mean(cluster_summary_div.num_hypotheses)}\n\tMedian divergent: {np.median(cluster_summary_div.num_hypotheses)}")

    # fig2, ax2 = plt.subplots()

    # ax2.hist(cluster_summary_div.num_tracks, density=True, bins=100, alpha=0.5, label="Divergent")
    # ax2.hist(cluster_summary_conv.num_tracks, density=True, bins=100, alpha=0.5, label="Convergent")
    # ax2.set_title("Number of tracks")

    # print(f"Num tracks:\n\tMax convergent: {np.max(cluster_summary_conv.num_tracks)}\n\tMin convergent: {np.min(cluster_summary_conv.num_tracks)}\n\tMean convergent: {np.mean(cluster_summary_conv.num_tracks)}\n\tMedian convergent: {np.median(cluster_summary_conv.num_tracks)}\n\n\tMax divergent: {np.max(cluster_summary_div.num_tracks)}\n\tMin divergent: {np.min(cluster_summary_div.num_tracks)}\n\tMean divergent: {np.mean(cluster_summary_div.num_tracks)}\n\tMedian convergent: {np.median(cluster_summary_div.num_tracks)}")

    # ax2.legend()

    # print(f"Percent failed exact marginals convergent: {cluster_summary_conv.num_failed_exact / cluster_summary_conv.num_data * 100.0}%")
    # print(f"Percent failed exact marginals divergent: {cluster_summary_div.num_failed_exact / cluster_summary_div.num_data * 100.0}%")


    # plt.show()

    cluster_stat, cluster_file = cluster_stats_diverged[0]

    print(cluster_file)
    msg_plot_lbp(cluster_file)

    # cluster_stats_converge_many_hypotheses = [(cluster_stat, cluster_file) for (cluster_stat, cluster_file) in cluster_stats_converged if cluster_stat.num_hypotheses > 50 and cluster_stat.cardinality > 15]

    # i = np.random.randint(0, len(cluster_stats_converge_many_hypotheses))
    # print(i)

    # cluster_stat, cluster_file = cluster_stats_converge_many_hypotheses[i]
    # print(cluster_file)

    # msg_plot_lbp(cluster_file)

    # plot_hypotheses_distributions(cluster_stats)
    # seed = None
    # plot_samples_reward_matices(cluster_stats_converged, cluster_stats_diverged, seed=seed, num_samples=8)
    # plot_num_meas_num_tracks(cluster_stats_converged, cluster_stats_diverged)

    # cluster_files_diverged = [cluster_file for (cluster_stat, cluster_file) in cluster_stats_diverged]
    # print(len(cluster_files_diverged))

    # test_all_divergent_cases(cluster_files_diverged)
