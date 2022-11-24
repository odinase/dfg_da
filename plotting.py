import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm

from dfg_da.stats_logger import MarginalsErrors, Marginals



def subsample(a: np.ndarray, inc: float) -> np.ndarray:
    """
    Returns the indices that subsamples the array.
    """
    idxs = [0]
    for k in range(1, len(a)):
        if a[k] > a[idxs[-1]]:
            idxs.append(k)
            break
    for k in range(idxs[-1] + 1, len(a)):
        if a[k] - a[idxs[-1]] > inc:
            idxs.append(k)

    return idxs



def plot_survival_function(ax: plt.Axes, marginals_errors: MarginalsErrors, label: str = "_"):
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

    for ax, error, title in zip(axes, errors, titles):
        ax.set_title(title)
        steps = np.linspace(1.0, 0.0, len(error))
        idxs = subsample(error, 1e-20)
        # emin = error[error > 0].min()
        print(f"Subsampling {title} for {label} reduced data to {len(idxs)/len(error)*100.0:.3f}%")
        print(f"Num zeros: {idxs[1]}")
        error = error[idxs]
        steps = steps[idxs]
        ax.step(error, steps, label=label)
        # ax.set_yscale('symlog')
        ax.set_xscale('symlog', linthresh=error[1])
        log_err = np.linspace(np.log10(error[1]), np.log10(error[-1]*1.1), 9).astype(int)
        xticks = np.array([0, *((10.0)**log_err)])
        print(f"Using xticks {xticks}")
        ax.set_xticks(xticks)
        xticks_labels = [0] + [rf'$10^{{{l}}}$' for l in log_err]
        ax.set_xticklabels(xticks_labels)
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter("10**{x:d}"))
        # ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        # ax.semilogx()
        ax.semilogy()
        # ax.loglog()
        if label != "_":
            ax.legend()


if __name__ == "__main__":
    lbp_mh_errors: MarginalsErrors = MarginalsErrors.from_path("./pmbm_analysis_output copy/lbp/errors")
    lbp_williams_errors: MarginalsErrors = MarginalsErrors.from_path("./pmbm_analysis_output copy/williams/errors")

    errors = [
        lbp_williams_errors,
        lbp_mh_errors
    ]

    labels = [
        "Williams LBP with estimated normalization constant",
        "LBP on full problem"
    ]

    fig, axes = plt.subplots(nrows=5, figsize=(7, 12), sharex=True)
    axes[-1].set_xlabel("Probability error 1-norm")


    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for error, label in tqdm(zip(errors, labels), total=len(errors)):
        plot_survival_function(axes, error, label)

    # fig2, ax2 = plt.subplots()
    # x = lbp_mh_errors.abs_errors

    # q25, q75 = np.percentile(x, [25, 75])
    # bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    # bins = round((x.max() - x.min()) / bin_width)

    # ax2.hist(x)
    # ax2.semilogy()
    # print("Showing plots...")
    
    # fig.savefig("plot.png")

    # plt.show()
    
    fig.tight_layout()
    fig.savefig("sf.pdf", bbox_inches = 'tight')
