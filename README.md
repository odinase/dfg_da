# Installation of code

This project is split into C++ code and Python code. The C++ code is built with pybind11. This means that when cloning this repo, you should do `git submodule update --init --recursive` afterwards. Then the C++ code can be built and installed as a Python package with `pip install .`.

Additionally, if you want to use the exact solver that finishes in reasonable time, i.e. multi-hypothesis EHM2, you'll need to clone [this fork](https://github.com/odinase/pyehm) of PyEHM2, change branch to `feature/add_return_likelihood` and install it with `python -m pip install -e .`. The only difference between the fork and the official repo is that the fork also returns the hypothesis-conditioned normalization constant.

If you at any point run into an error along the lines `could not find library libmetis-gtsam.so`, return to the GTSAM build folder and do
```
sudo make install
sudo ldconfig
``` 
(probably best solution) or add
```
export LD_LIBRARY_PATH=/usr/local/lib/:$HOME/local/lib/:$LD_LIBRARY_PATH
```
to your `~/.bashrc`/`~/.zshrc` or potentially
```
RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib/:$HOME/local/lib/:$LD_LIBRARY_PATH' >> ~/.zshrc
```
to your Dockerfile.

# Overview of most important code

## The scripts running collecting data - `ravens_parser_parallell.py` and `ravens_parser_parallell_multicluster.py`

The scripts `ravens_parser_parallell.py` and `ravens_parser_parallell_multicluster.py` are used to generate the results from Edmund's PMBM data. In particular, these scripts are good examples for how to parse data into data structures accepted by the inference methods implemented, and also how to use the "library" in general.

## The C++ code

All the C++ code can be found in `/src` and `/include`. The most relevant code is in `lbp.cpp` where single-cluster and multi-cluster multiple hypothesis LBP are implemented. Both take in an Eigen MatrixXd/numpy array of the reward matrix, where the left-most block are the detection loglikelihoods and the right-most block is diagonal with the misdetection logprobabilities (the same format as in the Sensor Fusion book of Brekke, except potentally transposed such that tracks are along the rows and the measurements the columns). __Note that when passing the reward matrix from Python into C++, you must use `order='C'`__. The code will fail if you try otherwise. This is to ensure that we don't copy the matrix across boundaries.

In `hypothesis.h`, two important datatypes `Hypotheses` and `Hypothesis` are defined. __These are used both in C++ and Python!__ Note that, when passing a `std::vector<Hypotheses>` for multicluster data association from Python into C++, you have to use the Python binding `HypothesesList`. Using `List[Hypotheses]` will not work, this is to avoid unnecessary copying of data across boundaries.

A good place to go to see examples for how to construct and manipulate these types is in `test_pybinds.py` and also `ravens_parser_parallell.py` and `ravens_parser_parallell_multicluster.py`.

## The Python native code

The most important Python native code can be found in the folder `dfg_da`. The file `marginal_association_Odin.py` contains `lbp_marginal`, the original Williams LBP algorithm for single-cluster, single-hypothesis data association, and the generalization `lbp_marginal_nonexistence` for multi-hypothesis. The returned values from these functions are a mess, so make sure you understand what is returned and in what order. It is probably the best to only use `lbp_marginal` and instead use the multi-hypothesis and multi-cluster generalized version from C++.

### The implementations in `marginal_computers.py`

Inside `marginal_computers.py` are classes that intend to expose a common interface with different implementations underneath.

For single-cluster, multi-hypothesis association, see the classes:

- `LBPMarginalsFullAssociation`: Does full MH-LBP on the single-cluster, multi-hypothesis factor graph.
- `LBPMarginalsByTotalProb`: Marginalizes over prior hypotheses where the hypothesis-conditioned likelihood is approximated with PHD.
- `LBPMarginalsByTotalProbBethe`: Marginalizes over prior hypotheses where the hypothesis-conditioned likelihood is approximated with Bethe constant/pseudodual.

Meanwhile, multi-cluster methods are:

- `MulticlusterExactEHM2`: Full, multi-cluster, multi-hypothesis exact solver to compute likelihoods and marginals to compute with. Returns the type `MulticlusterExactOutput` which should contain just about anything you can be curious to know about the output.
- The multi-cluster, multi-hypothesis LBP is implemented in C++. A Python implementation can be found in `marginal_association_Odin.py` in the function `lbp_marginal_nonexistence_multicluster`. This is deprecated, so use on your own risk.
- The fancy cluster-conditioning methods are in `cluster_bayes_tree.py` and `cluster_conditioning_lbp.py`, see below.

There are also a helper implementation in `ClusterHypothesesPosterior`. It computes the prior hypotheses of the posterior clusters when doing multi-cluster associations and is necessary for computing prior hypothesis posteriors. 


### The implementations in `cluster_bayes_tree.py` and `cluster_conditioning_lbp.py`

Firstly, note that the implementations in `cluster_bayes_tree.py` and `cluster_conditioning_lbp.py` are very similar. Merging the two implementations into one should be possible, but was done this way due to simplicity and time constraints.

In `cluster_bayes_tree.py` is the exact, multi-cluster, multi-hypothesis solver that is described in my (Odin) master's thesis. Its interface is slightly different from what is used in `marginal_computers.py` for no apparent reason. The top level is `MulticlusterEfficientMarginals` which uses `assocLocal` and the reward matrix `R_LC`, which is in LC format. It uses this information to create `ClusterLinks`, which contains information about what clusters have interacted with each other and the index of the measurements that causes this interaction. It then initializes the super clusters based on this information in its own `ConditionalSuperclusterMarginals` class. When this class is initialized, it further initializes `ConditionedCluster` instances for each prior cluster in the super cluster. It is in the very depths of `ConditionedCluster` the magic happens, as it keeps a mapping internally called `linking_mappings_mapping_mat` that it unpacks into `self.reindex_meas` and `self.actual_meas_idxs` that converts between the local measurement indices over linking measurements in the super cluster and the global measurement indices that is used to index into the reward matrix to disallow or allow measurement associations.

The code in `cluster_conditioning_lbp.py` is very similar to `cluster_bayes_tree.py`, where the major difference is that the top level is called `MulticlusterEfficientMarginalsLBP` which also accepts a `lbp_solver` "functor" that determines what LBP variant to use internally. See `ravens_parser_parallell_multicluster.py` for examples on how to configure it.

### Saving output to file

The result output from running the methods through the PMBM data to Edmund is saved to file for each MAT-file with the implementations in `stats_logger.py`, in particular `MulticlusterData` and `ClusterData`. In addition, this file contains the helper classes `Marginals` and `MarginalsErrors` that makes accessing the different marginal types we estimate more available. Last, but not least, this file contains the parser class `MatFileParser` that converts an Edmund PMBM MAT-file to data structures we can use, e.g. reward matrices and prior hypotheses. __Note that in the constructor you must use the input argument `use_cpp = True` for the prior hypotheses data structure to be `HypothesesList`, which is important if you want to do multi-cluster marginalization__.

# Pitfalls

## Index conventions

We assume that track and measurement indices are 1-indexed, reserving 0 for misdetection and false alarm, respectively. For cluster and hypothesis indices, however, we use 0-indexing.

## Two types of reward matrices - Edmund and Lars-Christian format {#reward-matrices}

For no particular reason, two types of reward matrices are used in the code: Edmund format and LC (Lars-Christian) format. Edmund format is the Sensor fusion format, where misdetections are in a diagonal block in the right-most part of the reward matrix. LC format stacks the misdetection probabilities ontop of each other in the first column. To convert between the two, use the fuctions `edmund_to_lc` and `lc_to_edmund` in `cluster_data_asso.py`. In general, the C++ code assumes Edmund format while Python native code assumes LC format. 

## Track indices and reindexing

The LBP code in C++ assumes the union of track indices over prior hypotheses to be the same as the row indices of the passed in reward matrix, 1-indexed. As an example, for a reward matrix with 5 rows, using the two prior hypotheses with tracks [1, 13, 4], [2, 44, 5] will not work, but tracks [1, 2, 3], [4, 5] will. There is a function implemented on `Hypotheses` that either takes in a `std::map<size_t, size_t>` or `Dict[int, int]` containing the mapping from old track to new track index, or no argument, in which case it collects the N track indices that exist and maps them in increasing order to the set {1, ..., N}. This is used several times in `cluster_conditioning_lbp.py` since we slice out parts of the global reward matrix when we factorize into conditioned cluster distributions.
