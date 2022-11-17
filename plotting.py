# import matplotlib
# import matplotlib.pyplot as plt

from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Step, Glyph

import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm

from dfg_da.stats_logger import MarginalsErrors, Marginals


def plot_survival_function(marginals_errors: MarginalsErrors, label: str = "_") -> Glyph:
    max_errors = np.sort(marginals_errors.max_errors)
    abs_errors = np.sort(marginals_errors.abs_errors)
    # raw_errors = np.sort(marginals_errors.abs_errors)
    misdetection_errors = np.sort(marginals_errors.misdetection_errors)
    detection_errors = np.sort(marginals_errors.detection_errors)
    nonexistence_errors = np.sort(marginals_errors.nonexistence_errors)

    steps = np.linspace(1.0, 0.0, len(max_errors))

    

    # errors = [max_errors,
    #     abs_errors,
    #     # raw_errors,
    #     misdetection_errors,
    #     detection_errors,
    #     nonexistence_errors]

    # titles = [
    #     "max_errors",
    #     "abs_errors",
    #     # "raw_errors",
    #     "misdetection_errors",
    #     "detection_errors",
    #     "nonexistence_errors"
    # ]

    # for ax, error, title in zip(axes, errors, titles):
    #     ax.set_title(title)
    #     steps = np.linspace(1.0, 0.0, len(error))
    #     ax.step(error, steps, label=label)
    #     # ax.set_yscale('symlog')
    #     ax.set_xscale('symlog', linthresh=1e-15)
    #     # ax.semilogx()
    #     ax.semilogy()
    #     # ax.loglog()
    #     if label != "_":
    #         ax.legend()
            


if __name__ == "__main__":
    lbp_mh_errors = MarginalsErrors.from_path("./pmbm_analysis_output copy/lbp/errors")
    lbp_williams_errors = MarginalsErrors.from_path("./pmbm_analysis_output copy/williams/errors")

    errors = [
        lbp_williams_errors,
        lbp_mh_errors
    ]
    labels = [
        "Williams LBP with estimated normalization constant",
        "LBP on full problem"
    ]

    fig, axes = plt.subplots(nrows=6)

    for error, label in tqdm(zip(errors, labels), total=len(errors)):
        plot_survival_function(axes, error, label)

    # print("Showing plots...")

    # fig2, ax = plt.subplots()

    # lbp_mh_marginals = Marginals.from_path("./pmbm_analysis_output/lbp/marginals")
    # exact_marginals = Marginals.from_path("./pmbm_analysis_output/exact/marginals")

    # ax.plot(lbp_mh_marginals.detection_marginals, exact_marginals.detection_marginals, 'ro', label='Detection')
    # ax.plot(lbp_mh_marginals.misdetection_marginals, exact_marginals.misdetection_marginals, 'bs', label='Misdetection')
    # ax.plot(lbp_mh_marginals.nonexistence_marginals, exact_marginals.nonexistence_marginals, 'gD', label='Nonexistence')

    # ax.set_xlabel("Approximate probability")
    # ax.set_ylabel("Exact probability")
    # ax.set_title("Correlation plot")

    # ax.legend()

    fig.savefig("corr_plot.png")
    # fig2.savefig("sf.png")