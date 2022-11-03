import yaml
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm


def dict2numpy(d):
    return np.array(d["data"]).reshape((d["rows"], d["cols"]))

def append_statistics(max_errors, abs_errors, marginal_abs_error):
    max_error = marginal_abs_error.max(axis=1)
    abs_error = marginal_abs_error.ravel()

    max_errors += max_error.tolist()
    abs_errors += abs_error.tolist()


if __name__ == "__main__":
    path = "/workspaces/ros_ws/logged_data"

    log_files = glob(f"{path}/*.yaml")

    num_files = len(log_files)
    max_errors_ipda = []
    abs_errors_ipda = []

    max_errors_lbp = []
    abs_errors_lbp = []

    # compute max error and 1-norm

    for file in tqdm(log_files):
        with open(file) as f:
            data = yaml.full_load(f)

        ipda_marginals = dict2numpy(data["ipda"])
        exact_marginals = dict2numpy(data["exact"])
        lbp_marginals = dict2numpy(data["lbp"])

        ipda_abs_errors = np.abs(ipda_marginals - exact_marginals)
        append_statistics(max_errors_ipda, abs_errors_ipda, ipda_abs_errors)

        lbp_abs_errors = np.abs(lbp_marginals - exact_marginals)
        append_statistics(max_errors_lbp, abs_errors_lbp, lbp_abs_errors)

    suptitle = f"Based on {num_files} association marginals"

    fig_lbp, ax_lbp = plt.subplots(nrows=2)
    fig_lbp.suptitle(suptitle)
    ax_lbp[0].hist(max_errors_lbp)
    ax_lbp[0].set_title(f"Max errors LBP: average {np.mean(max_errors_lbp)}")
    ax_lbp[1].hist(abs_errors_lbp)
    ax_lbp[1].set_title("Abs errors LBP")
    # ax_lbp[0].set_xscale('log')
    # ax_lbp[1].set_xscale('log')
    ax_lbp[0].loglog()
    ax_lbp[1].loglog()
    
    fig_ipda, ax_ipda = plt.subplots(nrows=2)
    fig_ipda.suptitle(suptitle)
    ax_ipda[0].hist(max_errors_ipda)
    ax_ipda[0].set_title(f"Max errors IPDA: average {np.mean(max_errors_ipda)}")
    ax_ipda[1].hist(abs_errors_ipda)
    ax_ipda[1].set_title("Abs errors IPDA")
    # ax_ipda[0].set_xscale('log')
    # ax_ipda[1].set_xscale('log')
    ax_ipda[0].loglog()
    ax_ipda[1].loglog()

    plt.show()