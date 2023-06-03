from multicluster_analysis import result_path_to_mat_file_string
from pathlib import Path
import numpy as np
import dfg_da.stats_logger as sl
import dfg_da.marginals_computers as mc
from ravens_parser_parallell_multicluster import OUTPUT_PATH_BASE, PMBM_DATA_PATH
from scipy.io import loadmat


def test_matlab(load_dirs):
    for pmbm_file in load_dirs:
        mat_data: sl.MatFileParser = sl.MatFileParser(pmbm_file, use_cpp=True)

        R_LC = np.asfortranarray(mat_data.reward_matrix_lc)
        prior_hypotheses_per_cluster = mat_data.prior_hypotheses_per_cluster

        num_clusters = len(prior_hypotheses_per_cluster)
        if num_clusters == 0:
            return

        assocLocal = mat_data.ws["assocLocal"].copy()
        exact_output = None
        exact_computer = mc.MulticlusterExactEHM2()

        exact_output: mc.MulticlusterExactOutput = exact_computer(R_LC, prior_hypotheses_per_cluster, assocLocal=assocLocal.copy())
        exact_output.theta_posteriors = exact_output.compute_theta_posteriors()

        murty_ws = loadmat(f"./murty_output/{pmbm_file.name}")

        murty_norm = murty_ws["Z"][0,0]
        murty_margs = murty_ws["a"]

        print(f"File: {pmbm_file.name}")
        print(f"Exact: {exact_output.exact_normalization_constant}\nMurty: {murty_norm}")
        
        np.set_printoptions(suppress=True, linewidth=250)
        print(f"Exact:\n{exact_output.exact_marginals[:3]}\nMurty:\n{murty_margs[:3]}")


def write_paths_to_file(load_dirs):
    # Concatenate the strings with newline separator
    concatenated_string = '\n'.join([result_path_to_mat_file_string(p) for p in load_dirs])

    # Specify the path to the output file
    file_path = './files_used.txt'

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the concatenated string to the file
        file.write(concatenated_string)



if __name__ == "__main__":
    from ravens_parser_parallell_multicluster import OUTPUT_PATH_BASE as path
    # path += " (7th copy)"

    load_dirs = Path(path).glob("**/*")
    load_dirs = sorted(list(load_dirs))

    # load_dirs = load_dirs[:5]
    print(f"load dirs: {len(load_dirs)} from {path}")
    write_paths_to_file(load_dirs)
    # org_files = [Path(result_path_to_mat_file_string(p)) for p in load_dirs]

    # test_matlab(org_files)
