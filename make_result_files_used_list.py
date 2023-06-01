from multicluster_analysis import result_path_to_mat_file_string
from pathlib import Path
import numpy as np
import dfg_da.stats_logger as sl
import dfg_da.marginals_computers as mc
from ravens_parser_parallell_multicluster import OUTPUT_PATH_BASE, PMBM_DATA_PATH



def test_matlab(pmbm_file):
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

    x=1    


if __name__ == "__main__":
    from ravens_parser_parallell_multicluster import OUTPUT_PATH_BASE as path
    path += " (7th copy)"


    load_dirs = Path(path).glob("**/*")
    load_dirs = sorted(list(load_dirs))

    load_dirs = load_dirs[:2]

    # Concatenate the strings with newline separator
    concatenated_string = '\n'.join([result_path_to_mat_file_string(p) for p in load_dirs])

    # Specify the path to the output file
    file_path = './files_used.txt'

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the concatenated string to the file
        file.write(concatenated_string)


    test_matlab(result_path_to_mat_file_string(load_dirs[1]))