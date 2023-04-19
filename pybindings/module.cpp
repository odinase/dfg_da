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

    m.def("throw_test", []() { throw std::invalid_argument("Test"); });


// std::tuple<Eigen::ArrayXXd, double> exact_marginals_and_normalization_constant(const Eigen::Ref<const Eigen::MatrixXd> &R, const std::vector<dfg_da::hypothesis::Hypotheses> &prior_hypotheses_per_cluster);
    py::module_ factor_graph = m.def_submodule("factor_graph");
    factor_graph.def("exact_marginals_and_normalization_constant", factor_graph::exact_marginals_and_normalization_constant, "R"_a.noconvert(), "prior_hypotheses_per_cluster"_a.noconvert())
    .def("all_exact_marginals_and_normalization_constant", factor_graph::all_exact_marginals_and_normalization_constant, "R"_a.noconvert(), "prior_hypotheses_per_cluster"_a.noconvert())
    // Eigen::ArrayXd hypothesis_conditioned_likelihoods(const Eigen::Ref<const Eigen::MatrixXd> &R, const dfg_da::hypothesis::Hypotheses &prior_hypotheses)
    .def("hypothesis_conditioned_likelihoods", factor_graph::hypothesis_conditioned_likelihoods, "R"_a.noconvert(), "prior_hypotheses"_a.noconvert());


    py::module_ hypothesis = m.def_submodule("hypothesis");

    hypothesis.def("association_marginal_posteriors_normalization_constant", hypothesis::association_marginal_posteriors_normalization_constant, "reward_matrix"_a.noconvert(), "prior_hypotheses"_a.noconvert());
    hypothesis.def("association_marginal_posteriors_normalization_constant_multicluster", hypothesis::association_marginal_posteriors_normalization_constant_multicluster, "reward_matrix"_a.noconvert(), "prior_hypotheses_per_cluster_posterior"_a.noconvert());
    hypothesis.def("hypothesis_enumeration", hypothesis::hypothesis_enumeration, "reward_matrix"_a.noconvert(), "prior_hypothesis"_a.noconvert());
    hypothesis.def("mo_to_to_hypothesis", hypothesis::mo_to_to_hypothesis, "mo_hypothesis"_a, "num_tracks"_a);
    hypothesis.def("prior_hypothesis_conditional_association_probability", hypothesis::prior_hypothesis_conditional_association_probability, "to_hypothesis"_a, "prior_hypothesis"_a.noconvert(), "reward_matrix"_a.noconvert());

// std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, std::map<std::string, Eigen::ArrayXd>, double> all_exact_marginals_and_normalization_constant

    py::bind_vector<std::vector<dfg_da::hypothesis::Hypotheses>>(hypothesis, "HypothesesList")
    .def(py::pickle(
        [](const std::vector<dfg_da::hypothesis::Hypotheses> &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */

            auto tuple = py::tuple(p.size());
            for (int i = 0 ; i < p.size() ; i++)
            {
                tuple[i] = p.at(i);
            }

            return tuple;
        },
        [](py::tuple t) { // __setstate__
            if (t.size() <= 0)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            std::vector<dfg_da::hypothesis::Hypotheses> p;
            for (int i = 0 ; i < t.size() ; i++)
            {
                p.push_back(t[i].cast<dfg_da::hypothesis::Hypotheses>());
            }
            return p;
        }
    ));

    py::class_<hypothesis::Hypothesis>(hypothesis, "Hypothesis")
    .def(py::init<const std::vector<size_t>&, double>())
    .def("probability", &hypothesis::Hypothesis::probability)
    .def("tracks", &hypothesis::Hypothesis::tracks)
    .def("reindex_tracks", py::overload_cast<const std::map<size_t, size_t>&>(&hypothesis::Hypothesis::reindex_tracks))
    .def("reindex_tracks", py::overload_cast<>(&hypothesis::Hypothesis::reindex_tracks))
    .def("contains", &hypothesis::Hypothesis::contains)
    .def("log_prob", &hypothesis::Hypothesis::log_prob)
    .def(py::pickle(
        [](const hypothesis::Hypothesis &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.log_prob(), p.tracks());
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 2)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            double log_prob = t[0].cast<double>();
            dfg_da::hypothesis::Hypothesis h(
                std::move(t[1].cast<std::vector<size_t>>()),
                log_prob
            );

            return h;
        }
    ));

    py::class_<hypothesis::Hypotheses>(hypothesis, "Hypotheses")
    .def(py::init<const std::vector<hypothesis::Hypothesis>&>())
    .def("combine", &hypothesis::Hypotheses::combine)
    .def("__getitem__", &hypothesis::Hypotheses::operator[])
    .def("__len__", &hypothesis::Hypotheses::num_hypotheses)
    .def("tracks", &hypothesis::Hypotheses::tracks)
    .def("reindex_tracks", py::overload_cast<const std::map<size_t, size_t>&>(&hypothesis::Hypotheses::reindex_tracks))
    .def("reindex_tracks", py::overload_cast<>(&hypothesis::Hypotheses::reindex_tracks))
    .def("hypothesis_probabilites", &hypothesis::Hypotheses::hypothesis_probabilites)
    .def("num_hypotheses", &hypothesis::Hypotheses::num_hypotheses)
    .def("t_idxs", &hypothesis::Hypotheses::t_idxs)
    .def("__iter__", [](hypothesis::Hypotheses &h) { return py::make_iterator(h.begin(), h.end()); },
                         py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
    .def(py::pickle(
        [](const hypothesis::Hypotheses &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            std::vector<hypothesis::Hypothesis> h(p.cbegin(), p.cend());
            return py::make_tuple(h);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 1)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            dfg_da::hypothesis::Hypotheses hh(std::move(t[0].cast<std::vector<hypothesis::Hypothesis>>()));

            return hh;
        }
    ));


    py::module_ lbp = m.def_submodule("lbp");


    py::class_<lbp::ClusterData>(lbp, "ClusterData")
    .def("phi",  &lbp::ClusterData::phi)
    .def_readwrite("phi_vec", &lbp::ClusterData::phi_vec)
    .def_readwrite("t_idx", &lbp::ClusterData::t_idx)
    .def_readwrite("t2h_not", &lbp::ClusterData::t2h_not)
    .def_readwrite("t2h", &lbp::ClusterData::t2h)
    .def(py::pickle(
        [](const lbp::ClusterData &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */

            // We need to undo the 0-indexing of tracks that is done in the constructor before saving to file
            std::vector<size_t> tracks;
            std::transform(p.t_idx.begin(), p.t_idx.end(), std::back_inserter(tracks), [](const auto& t) { return t + 1; });
            return py::make_tuple(p.phi_vec, tracks, p.t2h_not, p.t2h);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 4)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            lbp::ClusterData p(
                t[0].cast<std::vector<double>>(),
                t[1].cast<std::vector<size_t>>(),
                t[2].cast<Eigen::ArrayXXd>(),
                t[3].cast<Eigen::ArrayXXd>()
            );

            return p;
        }
    ));


    py::class_<lbp::MHLBPMulticlusterOutput>(lbp, "MHLBPMulticlusterOutput")
    .def("track_association_marginals",  &lbp::MHLBPMulticlusterOutput::track_association_marginals)
    .def("measurement_association_marginals",  &lbp::MHLBPMulticlusterOutput::measurement_association_marginals)
    .def("hypotheses_marginals",  &lbp::MHLBPMulticlusterOutput::hypotheses_marginals)
    .def("bethe_pseudodual_loglikelihood",  &lbp::MHLBPMulticlusterOutput::bethe_pseudodual_loglikelihood)
    .def("bethe_pseudodual_normalization_constant",  &lbp::MHLBPMulticlusterOutput::bethe_pseudodual_normalization_constant)
    .def_readonly("mu", &lbp::MHLBPMulticlusterOutput::mu)
    .def_readonly("nu", &lbp::MHLBPMulticlusterOutput::nu)
    .def_readonly("rho", &lbp::MHLBPMulticlusterOutput::rho)
    .def_readonly("sigma", &lbp::MHLBPMulticlusterOutput::sigma)
    .def_readonly("w_nmd", &lbp::MHLBPMulticlusterOutput::w_nmd)
    .def_readonly("w_0", &lbp::MHLBPMulticlusterOutput::w_0)
    .def_readonly("cluster_data", &lbp::MHLBPMulticlusterOutput::cluster_data)
    .def_readonly("num_iters", &lbp::MHLBPMulticlusterOutput::num_iters)
    .def_readonly("num_tracks", &lbp::MHLBPMulticlusterOutput::num_tracks)
    .def_readonly("num_measurements", &lbp::MHLBPMulticlusterOutput::num_measurements)
    .def_readonly("num_clusters", &lbp::MHLBPMulticlusterOutput::num_clusters)
    .def(py::pickle(
        [](const lbp::MHLBPMulticlusterOutput &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(
                p.mu,
                p.nu,
                p.rho,
                p.sigma,
                p.w_nmd,
                p.w_0,
                p.cluster_data,
                p.num_iters
            );
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 8)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            lbp::MHLBPMulticlusterOutput p(
                t[0].cast<Eigen::ArrayXXd>(),
                t[1].cast<Eigen::ArrayXXd>(),
                t[2].cast<Eigen::ArrayXd>(),
                t[3].cast<Eigen::ArrayXd>(),
                t[4].cast<Eigen::ArrayXXd>(),
                t[5].cast<Eigen::ArrayXd>(),
                t[6].cast<std::vector<lbp::ClusterData>>(),
                t[7].cast<size_t>()
            );

            return p;
        }
    ));

    // MHLBPMultilusterOutput lbp_multicluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const std::vector<hypothesis::Hypotheses> &prior_hypotheses_per_cluster, size_t max_num_iters = 300);
    lbp.def("lbp_multicluster", &lbp::lbp_multicluster, "reward_matrix"_a.noconvert(), "prior_hypotheses_per_cluster"_a.noconvert(), "max_num_iters"_a = 10'000);

    py::class_<lbp::MHLBPSingleClusterOutput>(lbp, "MHLBPSingleClusterOutput")
    .def("track_association_marginals",  &lbp::MHLBPSingleClusterOutput::track_association_marginals)
    .def("bethe_pseudodual_loglikelihood",  &lbp::MHLBPSingleClusterOutput::bethe_pseudodual_loglikelihood)
    .def("bethe_pseudodual_normalization_constant",  &lbp::MHLBPSingleClusterOutput::bethe_pseudodual_normalization_constant);
    // MHLBPMultilusterOutput lbp_multicluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const std::vector<hypothesis::Hypotheses> &prior_hypotheses_per_cluster, size_t max_num_iters = 300);
    lbp.def("lbp_single_cluster", &lbp::lbp_single_cluster, "reward_matrix"_a.noconvert(), "prior_hypotheses"_a.noconvert(), "max_num_iters"_a = 300);
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