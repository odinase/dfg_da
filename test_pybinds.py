import py_dfg_da
import py_dfg_da as pdd
import numpy as np
import dfg_da.marginal_association_Odin as ma
import dfg_da.marginals_computers as mc
import dfg_da.stats_logger as sl
from dfg_da.cluster_conditioning_lbp import MulticlusterEfficientMarginalsLBP
import pickle
import dfg_da as dd
from typing import List, Optional
import matplotlib.pyplot as plt


from ravens_parser_parallell_multicluster import merge_clusters
from cluster_data_asso import edmund_to_lc, lc_to_edmund
from plotting_multicluster import save_fig

def make_clusters(llr):
        g = (llr > -np.inf)
        pr_a = np.zeros_like(llr)
        pr_new = np.zeros(llr.shape[1] - 1, float)
        usedtr = np.zeros(llr.shape[0], bool)
        usedm = np.zeros(llr.shape[1] - 1, bool)
        gm = g[:, 1:]
        c = []
        for i in range(len(llr)):
            if not usedtr[i]:
                ctr = np.zeros_like(usedtr)
                ca = np.zeros(llr.shape[1], bool)
                ca[0] = True
                cm = ca[1:]
                ctr[i] = True
                m_in_same = gm[i]
                cm[m_in_same] = True
                changed = True
                while changed:
                    tr_in_same = np.any(gm[:, cm], axis=1)
                    changed &= ~np.all(ctr[tr_in_same])
                    ctr[tr_in_same] = True
                    m_in_same = np.any(gm[tr_in_same, :], axis=0)
                    changed &= ~np.all(cm[m_in_same])
                    cm[m_in_same] = True

                print(ctr, ca)
        #         pr_a[np.ix_(ctr, ca)], pr_new[cm] = exact_marginal(
        #             llr[np.ix_(ctr, ca)], False)
        #         usedtr[ctr] = True
        #         usedm[cm] = True
        # pr_new[~usedm] = 1
        # return pr_a, pr_new


def theta_posterior_correlation(true_posteriors: List[np.ndarray], lbp_posteriors: List[np.ndarray], path: str, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()

    true_posteriors = np.hstack(true_posteriors)
    lbp_posteriors = np.hstack(lbp_posteriors)

    ax.plot(lbp_posteriors, true_posteriors, 'bx')
    ax.plot(lbp_posteriors, lbp_posteriors, 'y--', label="Perfect correlation", alpha=0.6)
    ax.set_xlabel("Approximate probabilities")
    ax.set_ylabel("Exact probabilities")
    ax.set_title("Correlation plot prior hypothesis posterior")

    # Set the x and y axis ticks to range between 0 and 1
    step = 2
    xticks = np.arange(0, 10 + step, step) / 10.0
    yticks = np.arange(0, 10 + step, step) / 10.0
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Set the x and y axis tick labels to display 1 decimal place
    ax.set_xticklabels([f"{tick:.1f}" for tick in xticks])
    ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])

    save_fig(fig, "correlation_plot_theta_posteriors", path=path, tight_layout=True)


def theta_posterior_correlation_per_cluster(true_posteriors: List[np.ndarray], lbp_posteriors: List[np.ndarray], path: str, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()

    for c, (true, lbp) in enumerate(zip(true_posteriors, lbp_posteriors)):
        ax.plot(lbp, true, 'o', label=f"Cluster {c+1}", alpha=0.7)

    x = np.linspace(0, 1, 10)
    ax.plot(x, x, '--', label="Perfect correlation", alpha=0.7)
    ax.legend()
    ax.set_xlabel("Approximate probabilities")
    ax.set_ylabel("Exact probabilities")
    ax.set_title("Correlation plot prior hypothesis posterior")

    # Set the x and y axis ticks to range between 0 and 1
    step = 2
    xticks = np.arange(0, 10 + step, step) / 10.0
    yticks = np.arange(0, 10 + step, step) / 10.0
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Set the x and y axis tick labels to display 1 decimal place
    ax.set_xticklabels([f"{tick:.1f}" for tick in xticks])
    ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])

    save_fig(fig, "correlation_plot_per_cluster_theta_posteriors", path=path, tight_layout=True)



def marginals_correlation(exact_margs: np.ndarray, approx_margs: np.ndarray, path: str, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()

    exact_margs: sl.Marginals = sl.Marginals(marginals=exact_margs)
    approx_margs: sl.Marginals = sl.Marginals(marginals=approx_margs)

    ax.plot(approx_margs.misdetection_marginals, exact_margs.misdetection_marginals, 'ro', label="Misdetection", alpha=0.6)
    ax.plot(approx_margs.detection_marginals, exact_margs.detection_marginals, 'go', label="Detection", alpha=0.6)
    ax.plot(approx_margs.nonexistence_marginals, exact_margs.nonexistence_marginals, 'bo', label="Nonexistence", alpha=0.6)

    x = np.linspace(0, 1, 10)
    ax.plot(x, x, 'y--', label="Perfect correlation", alpha=0.6)
    

    ax.legend()
    ax.set_xlabel("Approximate probabilities")
    ax.set_ylabel("Exact probabilities")
    ax.set_title("Correlation plot association marginals")

    # Set the x and y axis ticks to range between 0 and 1
    step = 2
    xticks = np.arange(0, 10 + step, step) / 10.0
    yticks = np.arange(0, 10 + step, step) / 10.0
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Set the x and y axis tick labels to display 1 decimal place
    ax.set_xticklabels([f"{tick:.1f}" for tick in xticks])
    ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])

    save_fig(fig, "correlation_plot_marginals", path=path, tight_layout=True)

def marginals_correlation_per_cluster(exact_margs: np.ndarray, approx_margs: np.ndarray, t_idxs_per_cluster: List[np.ndarray], path: str, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()

    for c, t_idxs in enumerate(t_idxs_per_cluster):
        approx = approx_margs[t_idxs].ravel()
        exact = exact_margs[t_idxs].ravel()

        ax.plot(approx, exact, 'o', label=f"Cluster {c+1}", alpha=0.7)
    
    x = np.linspace(0, 1, 10)
    ax.plot(x, x, '--', label="Perfect correlation", alpha=0.7)

    ax.legend()
    ax.set_xlabel("Approximate probabilities")
    ax.set_ylabel("Exact probabilities")
    ax.set_title("Correlation plot association marginals")

    # Set the x and y axis ticks to range between 0 and 1
    step = 2
    xticks = np.arange(0, 10 + step, step) / 10.0
    yticks = np.arange(0, 10 + step, step) / 10.0
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Set the x and y axis tick labels to display 1 decimal place
    ax.set_xticklabels([f"{tick:.1f}" for tick in xticks])
    ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])

    save_fig(fig, "correlation_plot_per_cluster_marginals", path=path, tight_layout=True)


def numpy_to_latex(a):
    m, n = a.shape
    rows = []
    for i in range(m):
        row = []
        for j in range(n):
            if np.isfinite(a[i, j]):
                row.append(f"{a[i, j]:.3f}")
            elif a[i, j] == np.inf:
                row.append(r"\infty")
            else:
                row.append(r"-\infty")
        rows.append(" & ".join(row))
    latex = "\\begin{bmatrix}\n"
    latex += " \\\\ \n".join(rows)
    latex += "\n\\end{bmatrix}"
    print(latex)


def numpy_array_to_latex_table(arr):
    # Get the dimensions of the array
    rows, cols = arr.shape
    
    # Create the first row of the table
    table_header = [''] + [f"${i}$" for i in range(cols-1)] + ['$N$']
    
    # Create the body of the table
    table_body = []
    for i in range(rows):
        row = ['$a^{}$'.format(i+1)]
        for j in range(cols):
            row.append('${:.3f}$'.format(arr[i,j]))
        table_body.append(row)
    
    # Combine the header and body of the table
    table = [table_header] + table_body
    
    # Convert the table to LateX code
    latex = '\\begin{minipage}{\\linewidth}\n\\vspace{3ex}\n\\centering\n\\begin{tabular}{c|' + ' '.join(['c']*cols) + '}\n'
    latex += ' & '.join(table_header) + ' \\\\\\midrule\n'
    for row in table_body:
        latex += ' & '.join(row) + ' \\\\\n'
    latex += '\\end{tabular}\n\\vspace{3ex}\n\\end{minipage}'
    
    return latex


def test_case_1():
    R = np.array([
        [    3.0, -np.inf,   -0.60, -np.inf, -np.inf, -np.inf, -np.inf],
        [    3.2, -np.inf, -np.inf,   -0.56, -np.inf, -np.inf, -np.inf],
        [   -3.0,     1.2, -np.inf, -np.inf,   -0.46, -np.inf, -np.inf],
        [-np.inf,     3.0, -np.inf, -np.inf, -np.inf,   -0.62, -np.inf],
        [-np.inf,    -0.4, -np.inf, -np.inf, -np.inf, -np.inf,   -0.55],
    ], order='F')

    n, mpn = R.shape
    m = mpn - n

    R_LC = np.hstack((np.diag(R[:, m:])[:,None], R[:,:m]))

    assocLocal = np.array([
        [1, 1],
        [1, 0]
    ])

    prior_hypotheses_per_cluster: py_dfg_da.hypothesis.HypothesesList = py_dfg_da.hypothesis.HypothesesList([
        py_dfg_da.hypothesis.Hypotheses([
            py_dfg_da.hypothesis.Hypothesis([1, 2], np.log(0.5)),
            py_dfg_da.hypothesis.Hypothesis([1, 3], np.log(0.5))
        ]),
        py_dfg_da.hypothesis.Hypotheses([
            py_dfg_da.hypothesis.Hypothesis([4, 5], np.log(0.5)),
            py_dfg_da.hypothesis.Hypothesis([], np.log(0.5)),
        ])
    ])

    return R, R_LC, prior_hypotheses_per_cluster, assocLocal

def test_case_2():
    R_LC = np.array([
        [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1,     1.0,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf,     1.0, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf,     1.0,     1.0, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf,     1.0, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf,     1.0,     1.0,     1.0, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,     1.0],
    ], order='F')
    R_LC[:, 0] = np.log(R_LC[:, 0])

    R = np.asfortranarray(lc_to_edmund(R_LC))

    prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
        # Cluster 1
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([1], np.log(0.5)),
            pdd.hypothesis.Hypothesis([2], np.log(0.5))
        ]),
        # Cluster 2
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([3], np.log(0.5)),
            pdd.hypothesis.Hypothesis([4], np.log(0.5))
        ]),
        # Cluster 3
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([5], np.log(0.3)),
            pdd.hypothesis.Hypothesis([6], np.log(0.3)),
            pdd.hypothesis.Hypothesis([7], np.log(0.4))
        ]),
        # Cluster 4
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([8], np.log(0.3)),
            pdd.hypothesis.Hypothesis([9], np.log(0.3)),
            pdd.hypothesis.Hypothesis([10], np.log(0.4))
        ]),
        # Cluster 5
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([11], np.log(0.9)),
            pdd.hypothesis.Hypothesis([  ], np.log(0.1))
        ]),
        # Cluster 6
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([12], np.log(0.9)),
            pdd.hypothesis.Hypothesis([  ], np.log(0.1))
        ])
    ])

    assocLocal = np.array([
        [1, 1, 3, 3, 1, 6],
        [1, 0, 1, 0, 0, 1]
    ])

    return R, R_LC, prior_hypotheses_per_cluster, assocLocal


def test_case_3():
    R_LC = np.array([
        [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1,     1.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1,     1.0,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf,     1.0, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf,     1.0, -np.inf, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf,     1.0,     1.0, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf,     1.0, -np.inf, -np.inf],
        [0.1, -np.inf, -np.inf,     1.0,     1.0,     1.0, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [0.1,     1.0, -np.inf, -np.inf, -np.inf,     1.0, -np.inf],
        [0.1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,     1.0],
    ], order='F')

    R = np.asfortranarray(lc_to_edmund(R_LC))

    prior_hypotheses_per_cluster: pdd.hypothesis.HypothesesList = pdd.hypothesis.HypothesesList([
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([1, 2], np.log(0.5)),
            pdd.hypothesis.Hypothesis([], np.log(0.5))
        ]),
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([3, 4], np.log(0.5)),
            pdd.hypothesis.Hypothesis([], np.log(0.5))
        ]),
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([5, 6, 7], np.log(0.3)),
            pdd.hypothesis.Hypothesis([], np.log(0.3)),
            # pdd.hypothesis.Hypothesis([7], np.log(0.4))
        ]),
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([8, 9, 10], np.log(0.3)),
            pdd.hypothesis.Hypothesis([], np.log(0.3)),
            # pdd.hypothesis.Hypothesis([10], np.log(0.4))
        ]),
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([11], np.log(0.9)),
            # pdd.hypothesis.Hypothesis([  ], np.log(0.1))
        ]),
        pdd.hypothesis.Hypotheses([
            pdd.hypothesis.Hypothesis([12], np.log(0.9)),
            # pdd.hypothesis.Hypothesis([  ], np.log(0.1))
        ])
    ])

    assocLocal = np.array([
        [1, 1, 1, 1, 1, 6],
        [1, 0, 0, 0, 0, 1]
    ])

    correct_mapping = {
        1: { 0, 1, 4 },
        3: { 2, 3 },
        4: { 2, 3 },
        5: { 3, 4 }
    }

    return R, R_LC, prior_hypotheses_per_cluster, assocLocal



if __name__ == "__main__":
    from plotting_multicluster import FIGURES_PATH
    path = FIGURES_PATH + "/test_case3"
    R, R_LC, prior_hypotheses_per_cluster, assocLocal = test_case_3()

    t_idxs_per_cluster = [ph.t_idxs() for ph in prior_hypotheses_per_cluster]

    exact_computer: mc.MulticlusterExact = mc.MulticlusterExactEHM2()
    exact_output: mc.MulticlusterExactOutput = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal)
    np.set_printoptions(suppress=True, linewidth=150)
    print(exact_output.exact_marginals)
    numpy_to_latex(exact_output.exact_marginals)
    print(numpy_array_to_latex_table(exact_output.exact_marginals))
    print()
    print(exact_output.exact_marginals)
    print()
    true_theta_posteriors = exact_output.compute_theta_posteriors()
    print(true_theta_posteriors)
    print(f"{exact_output.exact_normalization_constant:.3f}")

    mcmhlbp: pdd.lbp.MHLBPMulticlusterOutput = pdd.lbp.lbp_multicluster(R, prior_hypotheses_per_cluster)
    lbp_margs = mcmhlbp.track_association_marginals().T
    print(numpy_array_to_latex_table(lbp_margs))
    print()
    print(lbp_margs)
    print()
    hp = mcmhlbp.hypotheses_marginals()
    print(hp)
    print(f"{mcmhlbp.bethe_pseudodual_normalization_constant():.3f}")

    print("Efficient Bethe!")
    efficient_mc_williams = MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbBethe())
    efficient_mc_williams_output: sl.MulticlusterConditionendLBPOutput = efficient_mc_williams.compute_marginals_likelihood()

    print(efficient_mc_williams_output.marginals)
    print(efficient_mc_williams_output.likelihood)
    print(efficient_mc_williams_output.theta_posteriors)

    print("Efficient PHD!")
    efficient_mc_williams = MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsByTotalProbPHD())
    efficient_mc_williams_output: sl.MulticlusterConditionendLBPOutput = efficient_mc_williams.compute_marginals_likelihood()

    print(efficient_mc_williams_output.marginals)
    print(efficient_mc_williams_output.likelihood)
    print(efficient_mc_williams_output.theta_posteriors)

    print("Efficient MHLBP!")
    efficient_mc_williams = MulticlusterEfficientMarginalsLBP(R_LC=R_LC, prior_hypotheses_per_cluster=prior_hypotheses_per_cluster, assocLocal=assocLocal.copy(), lbp_solver=mc.LBPMarginalsFullAssociationCPP())
    efficient_mc_williams_output: sl.MulticlusterConditionendLBPOutput = efficient_mc_williams.compute_marginals_likelihood()

    print(efficient_mc_williams_output.marginals)
    print(efficient_mc_williams_output.likelihood)
    print(efficient_mc_williams_output.theta_posteriors)

    theta_posterior_correlation(true_posteriors=true_theta_posteriors, lbp_posteriors=hp, path=path)
    theta_posterior_correlation_per_cluster(true_posteriors=true_theta_posteriors, lbp_posteriors=hp, path=path)

    marginals_correlation(exact_output.exact_marginals, lbp_margs, path=path)
    marginals_correlation_per_cluster(exact_output.exact_marginals, lbp_margs, t_idxs_per_cluster, path=path)