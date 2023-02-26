#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include <vector>
#include "dfg_da/hypothesis.h"
PYBIND11_MAKE_OPAQUE(std::vector<dfg_da::hypothesis::Hypotheses>);


#include <iostream>
#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteDistribution.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <limits>
#include <fstream>



#ifdef GLOG_AVAILABLE
#include <glog/logging.h>
#endif // GLOG_AVAILABLE
#include <cmath>

#include "dfg_da/factor_graph.h"
#include "dfg_da/lbp.h"

using gtsam::symbol_shorthand::A;



void gtsam_test() {
    gtsam::DiscreteKeys all_keys;
    gtsam::DiscreteFactorGraph dfg = dfg_da::factor_graph::build_test_factor_graph();

    gtsam::DiscreteFactor::Values solution = dfg.optimize();
    gtsam::DiscreteMarginals marginals(dfg);

    for (const auto& key : all_keys) {
        gtsam::Vector marginal = marginals.marginalProbabilities(key);
        if (gtsam::symbolChr(key.first) == 'a') {
        std::cout << "Marginals for " << gtsam::Symbol(key.first) << ": " << marginal.transpose() << "\n";
        }
    }

    constexpr double inf = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd R(3, 4);
    R << 4.78, -0.46, -inf, -inf,
         5.37, -inf, -0.52, -inf,
         6.58, -inf, -inf, -0.60;

    dfg_da::hypothesis::Hypothesis h1({1, 2}, log(0.5));
    dfg_da::hypothesis::Hypothesis h2({1, 3}, log(0.5));

    dfg_da::hypothesis::Hypotheses h{{h1, h2}};

    Eigen::MatrixXd probs = dfg_da::hypothesis::association_marginal_posteriors(R, h);

    std::cout << probs << "\n";
}


namespace py = pybind11;
using namespace pybind11::literals;
using namespace dfg_da;

int add(int i, int j) {
    return i + j;
}


PYBIND11_MODULE(py_dfg_da, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("gtsam_test", &gtsam_test);

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");


// std::tuple<Eigen::ArrayXXd, double> exact_marginals_and_normalization_constant(const Eigen::Ref<const Eigen::MatrixXd> &R, const std::vector<dfg_da::hypothesis::Hypotheses> &prior_hypotheses_per_cluster);
    py::module_ factor_graph = m.def_submodule("factor_graph");
    factor_graph.def("exact_marginals_and_normalization_constant", factor_graph::exact_marginals_and_normalization_constant, "R"_a.noconvert(), "prior_hypotheses_per_cluster"_a.noconvert());

    py::module_ hypothesis = m.def_submodule("hypothesis");

    hypothesis.def("association_marginal_posteriors_normalization_constant", hypothesis::association_marginal_posteriors_normalization_constant, "reward_matrix"_a.noconvert(), "prior_hypotheses"_a.noconvert());

    py::bind_vector<std::vector<dfg_da::hypothesis::Hypotheses>>(hypothesis, "HypothesesList");
    py::class_<hypothesis::Hypothesis>(hypothesis, "Hypothesis")
    .def(py::init<const std::vector<size_t>&, double>())
    .def("probability", &hypothesis::Hypothesis::probability)
    .def("tracks", &hypothesis::Hypothesis::tracks);

    py::class_<hypothesis::Hypotheses>(hypothesis, "Hypotheses")
    .def(py::init<const std::vector<hypothesis::Hypothesis>&>())
    .def("combine", &hypothesis::Hypotheses::combine)
    .def("__getitem__", &hypothesis::Hypotheses::operator[])
    .def("__len__", &hypothesis::Hypotheses::num_hypotheses)
    .def("tracks", &hypothesis::Hypotheses::tracks);


    py::module_ lbp = m.def_submodule("lbp");

    py::class_<lbp::MHLBPMulticlusterOutput>(lbp, "MHLBPMultilusterOutput")
    .def("track_association_marginals",  &lbp::MHLBPMulticlusterOutput::track_association_marginals)
    .def("bethe_pseudodual_loglikelihood",  &lbp::MHLBPMulticlusterOutput::bethe_pseudodual_loglikelihood)
    .def("bethe_pseudodual_normalization_constant",  &lbp::MHLBPMulticlusterOutput::bethe_pseudodual_normalization_constant);
    // MHLBPMultilusterOutput lbp_multicluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const std::vector<hypothesis::Hypotheses> &prior_hypotheses_per_cluster, size_t max_num_iters = 300);
    lbp.def("lbp_multicluster", &lbp::lbp_multicluster, "reward_matrix"_a.noconvert(), "prior_hypotheses_per_cluster"_a.noconvert(), "max_num_iters"_a = 300);
}



// PYBIND11_MODULE(dfg_da, m)
// {
//     // m.doc() = "pybind11 example plugin"; // optional module docstring
//     py::class_<Test>(m, "Test")
//         .def(py::init<>());
//     m.def("test_optional", &test_optional);
//     // m.def("test_shared_ptr", &test_shared_ptr);
//     m.def("test_ptr", &test_ptr);
//     m.def("test_vec", &test_vec);


//     // .def("setName", &Pet::setName)
//     // .def("getName", &Pet::getName);
//     py::module_ targets = m.def_submodule("targets");
//     py::class_<Target>(targets, "Target")
//         .def(py::init<const VectorConstRef, std::shared_ptr<DynamicModel>, int>(), py::arg("init_state"), py::arg("dyn_model"), py::arg("len_story") = 1);

//     py::module_ dynamics = m.def_submodule("dynamics");
//     py::class_<DynamicModel, std::shared_ptr<DynamicModel>, PyDynamicModel>(dynamics, "DynamicModel")
//         .def(py::init<>())
//         .def("state_space_dim", &DynamicModel::state_space_dim)
//         .def("pos_idx", &DynamicModel::pos_idx)
//         .def("input_vec", &DynamicModel::input_vec)
//         .def("gain_mat", &DynamicModel::gain_mat)
//         .def("cov_mat", &DynamicModel::cov_mat)
//         .def("dyn_mat", &DynamicModel::dyn_mat)
//         .def("propagate", &DynamicModel::propagate);
// }