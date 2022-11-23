# import matplotlib
# import matplotlib.pyplot as plt

from bokeh.io import curdoc, show, export_png
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Step, Glyph

import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm

from dfg_da.stats_logger import MarginalsErrors, Marginals


def draw_survival_function_glyph(marginals_errors: MarginalsErrors, label: str = "_") -> Glyph:
    data = ColumnDataSource({
        "max_errors": np.sort(marginals_errors.max_errors),
        # "abs_errors": np.sort(marginals_errors.abs_errors),
        # "misdetection_errors": np.sort(marginals_errors.misdetection_errors),
        # "detection_errors": np.sort(marginals_errors.detection_errors),
        # "nonexistence_errors": np.sort(marginals_errors.nonexistence_errors),
        "steps_max": np.linspace(1.0, 0.0, len(marginals_errors.max_errors))
    })

    step_glyph = Step(x="steps_max", y="max_errors", line_color="#f46d43", mode="before")
    return data, step_glyph


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

    plot = Plot(title=None, width=300, height=300, min_border=0, toolbar_location=None)

    for error, label in tqdm(zip(errors, labels), total=len(errors)):
        plot.add_glyph(*draw_survival_function_glyph(error, label))

    xaxis = LinearAxis()
    plot.add_layout(xaxis, 'below')

    yaxis = LinearAxis()
    plot.add_layout(yaxis, 'left')

    curdoc().add_root(plot)
    
    export_png(plot, filename="plot.png")

    # show(plot)
