#include <pybind11/pybind11.h>

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

#include "dfg_da/hypothesis.h"
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

    Eigen::MatrixXd probs = dfg_da::hypothesis::association_marginal_posteriors(h, R);

    std::cout << probs << "\n";
}


namespace py = pybind11;

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