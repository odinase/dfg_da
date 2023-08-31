import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from plotting import save_fig


if __name__ == "__main__":
    good_murty = np.array([149, 148.9, 148.7, 0.01, 0.01, 0.0001])
    good_area = np.sum(good_murty)
    bad_murty = np.ones(len(good_murty))
    bad_area = np.sum(bad_murty)
    enumerated_by_murtys = 3
    x = np.arange(enumerated_by_murtys)

    murtys = [(good_murty, good_area, "good_murty"), (bad_murty, bad_area, "bad_murty")]

    for m, a, n in murtys:
        fig, ax = plt.subplots(figsize=(8, 5))

        m_enumerated= m[:enumerated_by_murtys]
        area_enumerated = np.sum(m_enumerated)

        ax.plot(m, label='Hypothesis distribution')
        ax.plot(m_enumerated, 'o', label="Enumerated by Murty's")
        ax.set_ylabel("Hypothesis score", fontsize=24)
        ax.set_xlabel("Hypothesis index", fontsize=24)
        # ax.fill_between(x, m[:enumerated_by_murtys], color='blue', alpha=0.3)

        # shade_legend = mpatches.Patch(color='blue', alpha=0.3, label="Enumerated by Murty's")
        ax.set_title(rf"Mass enumerated: {area_enumerated/a*100.0:.3f}\%", fontsize=20)
        print(rf"Mass enumerated: {area_enumerated/a*100.0:.3f}\%")
        ax.legend(fontsize=20)

        # Remove the axes
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.xaxis.set_ticks_position('none')
        # ax.yaxis.set_ticks_position('none')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        save_fig(fig, n)
