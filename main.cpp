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
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::T;

constexpr bool xnor(const bool x, const bool y) { return !(x != y); }

/**
 *     gtsam::DiscreteKey x(X(0), 2), y(Y(0), 2);
    gtsam::DiscreteKeys keys {y, x};
    gtsam::KeyVector k{keys[0].first, keys[1].first};
    gtsam::DiscreteFactorGraph dfg;

    std::vector<double> x_table = {2.0, 1.0};
    gtsam::DiscreteKeys xk = {x};
    gtsam::DecisionTreeFactor x_prior(xk, x_table);
    dfg.push_back(x_prior);

    std::vector<double> xy_table = {
        1.0, 2.0,
        3.0, 4.0
    };
    gtsam::DecisionTreeFactor xy_factor(keys, xy_table);
    dfg.push_back(xy_factor);

    // auto [bayes_net, fg] = dfg.eliminatePartialSequential(k);
    // fg->print();
    auto fac = dfg.product();
    auto ff = fac.sum(2);

    double val = (*ff)({});
    std::cout << val << "\n";
*/

// gtsam::DiscreteFactorGraph dfg_from_reward_mat_hyp_prior(const Eigen::Ref<const Eigen::MatrixXd> &R, const std::vector<dfg_da::hypothesis::Hypotheses> &prior_hypotheses_per_cluster)
// {
//     gtsam::DiscreteFactorGraph dfg;

//     // Build left side of graph: Connect tracks to hypothesis variable for each cluster
//     const size_t num_clusters = prior_hypotheses_per_cluster.size();
//     // Add all track variables.
//     // The reward matrix should nt x (m + nt), ie, one row for each track
//     // and one column for each measurement plus columns for misdetection
//     assert(R.cols() >= R.rows());
//     const size_t num_tracks = R.rows();
//     const size_t num_measurements = R.cols() - num_tracks;
//     gtsam::DiscreteKeys ais; // Track variables and measurement variables
//     ais.reserve(num_tracks);

//     for (size_t c = 0; c < num_clusters; c++)
//     {
//         const dfg_da::hypothesis::Hypotheses& prior_hypotheses = prior_hypotheses_per_cluster[c];
//         // First construct hypothesis prior factor and variable
//         const size_t num_prior_hypotheses = prior_hypotheses.num_hypotheses();
//         std::set<size_t> tracks_in_cluster = prior_hypotheses.tracks();

//         gtsam::DiscreteKey th{T(c), num_prior_hypotheses};
//         std::vector<double> theta_table = prior_hypotheses.hypothesis_probabilites();
//         gtsam::DiscreteDistribution th_factor(th, theta_table);
//         dfg.push_back(th_factor);

//         // Hard compatability constraints are basically: 1 everywhere except nonexistence if it exists in the prior hypothesis
//         for (const size_t track : tracks_in_cluster)
//         {
//             gtsam::DiscreteKey ai(A(track), 1 + num_measurements + 1); // misdetection + num measurements + non-existence
//             ais.push_back(ai);
//             gtsam::DiscreteKeys keys{th, ai};

//             // Add factor between hypothesis variable and track variable
//             std::vector<double> compatibility_table;
//             for (size_t hypo = 0; hypo < num_prior_hypotheses; hypo++)
//             {
//                 bool contained_in_hypo = prior_hypotheses[hypo].contains(track);
//                 for (size_t meas = 0; meas <= num_measurements + 1; meas++)
//                 {
//                     bool exists = meas < (num_measurements + 1);
//                     compatibility_table.push_back(xnor(contained_in_hypo, exists)); // If contained in hypothesis and less than non-existence id, use 1
//                 }
//             }

//             gtsam::DecisionTreeFactor hyp_to_track_factor(keys, compatibility_table);
//             dfg.push_back(hyp_to_track_factor);

//             // Add prior factors
//             // We assume that the reward matrix is the logarithm of probabilities, as this is common to use
//             std::vector<double> prior_table;

//             size_t t_idx = track - 1; // The rows are 0-indexed, so we need to offset the track index.
//             size_t misdetection_idx = num_measurements + t_idx;
//             // Add misdetection
//             double m = exp(R(t_idx, misdetection_idx));
//             prior_table.push_back(m);

//             // Add all likelihoods
//             for (size_t meas_idx = 0; meas_idx < num_measurements; meas_idx++)
//             {
//                 double l = exp(R(t_idx, meas_idx));
//                 prior_table.push_back(l);
//             }

//             // Lastly, add non-existence
//             prior_table.push_back(1.0);

//             gtsam::DiscreteKeys aik = {ai};
//             gtsam::DecisionTreeFactor prior_factor(aik, prior_table);
//             dfg.push_back(prior_factor);
//         }
//     }

//     // Final stretch, add measurement variables
//     // Lets 1-index measurements as well for consistency
//     for (size_t meas = 1; meas <= num_measurements; meas++)
//     {
//         gtsam::DiscreteKey bj_dk(B(meas), 1 + num_tracks); // clutter or associated to a track

//         uint64_t bj = gtsam::symbolIndex(bj_dk.first);

//         // Add factor from measurement to all track variables
//         for (const auto &ai_dk : ais)
//         {
//             std::vector<double> compatibility_table;

//             size_t track_cardinality = ai_dk.second;
//             uint64_t ai = gtsam::symbolIndex(ai_dk.first);

//             // We need to do this row-major, so fix the row. Along one row we vary what association is compatible for this measurement
//             for (size_t j = 0; j < track_cardinality; j++)
//             {
//                 for (size_t i = 0; i <= num_tracks; i++)
//                 {
//                     // The compatibility here is the fact that we must assign bt to at for at == bt and no other ats, or otherwise the opposite
//                     // Here we see the benefit of using 1-indexed measurements: Since the tracks assume that measurement 0 is misdetection, c will automatically point to correct measurement

//                     double compatibility = xnor(i == ai, j == bj);
//                     compatibility_table.push_back(compatibility);
//                 }
//             }
//             gtsam::DiscreteKeys keys{ai_dk, bj_dk};
//             gtsam::DecisionTreeFactor meas_to_track_factor(keys, compatibility_table);
//             dfg.push_back(meas_to_track_factor);
//         }
//     }

//     return dfg;
// }

int main(int argc, char **argv)
{
#ifdef GLOG_AVAILABLE
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif // GLOG_AVAILABLE

    // gtsam::DiscreteFactorGraph dfg = dfg_da::factor_graph::build_test_factor_graph();

    // gtsam::DiscreteFactor::Values solution = dfg.optimize();
    // gtsam::DiscreteMarginals marginals(dfg);

    // auto dks = dfg.discreteKeys();
    // std::set<gtsam::DiscreteKey> all_keys(dks.begin(), dks.end());
    // for (const auto& key : all_keys) {
    //     gtsam::Vector marginal = marginals.marginalProbabilities(key);
    //     if (gtsam::symbolChr(key.first) == 'a') {
    //     std::cout << "Marginals for " << gtsam::Symbol(key.first) << ": " << marginal.transpose() << "\n";
    //     }
    // }

    constexpr double inf = std::numeric_limits<double>::infinity();
    // Eigen::MatrixXd R(3, 4);
    // R << 4.78, -0.46, -inf, -inf,
    //      5.37, -inf, -0.52, -inf,
    //      6.58, -inf, -inf, -0.60;

    // dfg_da::hypothesis::Hypothesis h1({1, 2}, log(0.5));
    // dfg_da::hypothesis::Hypothesis h2({1, 3}, log(0.5));

    // dfg_da::hypothesis::Hypotheses h{{h1, h2}};

    // Eigen::MatrixXd probs = dfg_da::hypothesis::association_marginal_posteriors(R, h);

    // std::cout << probs << "\n";

    // dfg_da::factor_graph::FactorGraph fg(dfg);
    // dfg_da::factor_graph::Marginals lbp_marginals = fg.lbp();
    // for (const auto& [k, m] : lbp_marginals) {
    //     std::cout << gtsam::Symbol(k) << ": ";
    //     for (const auto p : m) {
    //         std::cout << p << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Eigen::ArrayXXd asso_probs = dfg_da::lbp::lbp(R, h);
    // std::cout << asso_probs.transpose() << "\n";

    // gtsam::Ordering order(gtsam::KeyVector{{gtsam::Symbol('b', 1), gtsam::Symbol('a', 1), gtsam::Symbol('a', 2), gtsam::Symbol('a', 3), gtsam::Symbol('T', 0)}});

    // auto keys = dfg.keys();
    // for (const auto& k: keys) {
    //     std::cout << gtsam::Symbol(k) << "\n";
    // }

    // auto dbn = dfg.eliminateMultifrontal();

    // dbn->saveGraph("graph.txt");

    constexpr size_t num_tracks = 5;
    constexpr size_t num_measurements = 2;

    Eigen::MatrixXd R(num_tracks, num_measurements + num_tracks);
    R << 3.0, -inf, -0.60, -inf, -inf, -inf, -inf,
        3.2, -inf, -inf, -0.56, -inf, -inf, -inf,
        -3.0, 1.2, -inf, -inf, -0.46, -inf, -inf,
        -inf, 3.0, -inf, -inf, -inf, -0.62, -inf,
        -inf, -0.4, -inf, -inf, -inf, -inf, -0.55;

    std::vector<dfg_da::hypothesis::Hypotheses> prior_hypotheses_per_cluster;
    // Cluster 1
    std::vector<size_t> tracks1 = {1, 2};
    double logprob1 = log(0.5);
    dfg_da::hypothesis::Hypothesis h1(std::move(tracks1), logprob1);

    std::vector<size_t> tracks2 = {1, 3};
    double logprob2 = log(0.5);
    dfg_da::hypothesis::Hypothesis h2(std::move(tracks2), logprob2);

    std::vector<dfg_da::hypothesis::Hypothesis> hypos1({h1, h2});
    prior_hypotheses_per_cluster.emplace_back(dfg_da::hypothesis::Hypotheses(std::move(hypos1)));

    // Cluster 2
    std::vector<size_t> tracks3 = {4};
    double logprob3 = log(0.5);
    dfg_da::hypothesis::Hypothesis h3(std::move(tracks3), logprob3);

    std::vector<size_t> tracks4 = {5};
    double logprob4 = log(0.5);
    dfg_da::hypothesis::Hypothesis h4(std::move(tracks4), logprob4);

    std::vector<dfg_da::hypothesis::Hypothesis> hypos2({h3, h4});
    prior_hypotheses_per_cluster.emplace_back(dfg_da::hypothesis::Hypotheses(std::move(hypos2)));

    auto mhlbp = dfg_da::lbp::lbp_multicluster(R, prior_hypotheses_per_cluster);
    Eigen::ArrayXXd marginals = mhlbp.track_association_marginals();
    double Z_bethe = mhlbp.bethe_pseudodual_normalization_constant();
    std::cout << Z_bethe << "\n";

    std::cout << marginals << "\n";

    auto [exact_marginals, exact_normalization_constant] = dfg_da::factor_graph::exact_marginals_and_normalization_constant(R, prior_hypotheses_per_cluster);
    std::cout << exact_marginals << "\n" << exact_normalization_constant << "\n";


    // Test merging of clusters

    std::vector<size_t> hh1 = {1, 2, 4};
    double log_p_hh1 = 2.0*log(0.5);

    dfg_da::hypothesis::Hypothesis hv1(std::move(hh1), log_p_hh1);

    std::vector<size_t> hh2 = {1, 2, 5};
    double log_p_hh2 = 2.0*log(0.5);

    dfg_da::hypothesis::Hypothesis hv2(std::move(hh2), log_p_hh2);

    std::vector<size_t> hh3 = {1, 3, 4};
    double log_p_hh3 = 2.0*log(0.5);

    dfg_da::hypothesis::Hypothesis hv3(std::move(hh3), log_p_hh3);

    std::vector<size_t> hh4 = {1, 3, 5};
    double log_p_hh4 = 2.0*log(0.5);

    dfg_da::hypothesis::Hypothesis hv4(std::move(hh4), log_p_hh4);

    std::vector<dfg_da::hypothesis::Hypothesis> hhv = {hv1, hv2, hv3, hv4};
    // dfg_da::hypothesis::Hypotheses hh(std::move(hhv));

    size_t i = 0;
    double Z = 0.0;
    std::vector<std::pair<std::vector<size_t>, double>> all_hypos;
    std::cout << "hhv\n";
    for (auto& h : hhv) {
        for (const auto& t : h.tracks()) {
            std::cout << t << " ";
        }
        std::cout << std::endl;
        // std::vector<std::vector<size_t>> hypo_enumerations = dfg_da::hypothesis::hypothesis_enumeration(R, h);
        // for (auto& asso : hypo_enumerations) {
        //     double log_Z = 0.0;
        //     std::vector<size_t> to_cond_posterior_hypothesis = dfg_da::hypothesis::mo_to_to_hypothesis(asso, num_tracks);
        //     double log_p = prior_hypothesis_conditional_association_probability(to_cond_posterior_hypothesis, h, R);
        //     log_Z += log_p;
        //     log_Z += h.log_prob();
        //     all_hypos.push_back({to_cond_posterior_hypothesis, log_Z});
        //     Z += exp(log_Z);
        // }
    }
    std::cout << "h_combined\n";
    dfg_da::hypothesis::Hypotheses h_combined = prior_hypotheses_per_cluster[0].combine(prior_hypotheses_per_cluster[1]);
    for (auto& h : h_combined) {
        for (const auto& t : h.tracks()) {
            std::cout << t << " ";
        }
        std::cout << std::endl;
        // std::vector<std::vector<size_t>> hypo_enumerations = dfg_da::hypothesis::hypothesis_enumeration(R, h);
        // for (auto& asso : hypo_enumerations) {
        //     double log_Z = 0.0;
        //     std::vector<size_t> to_cond_posterior_hypothesis = dfg_da::hypothesis::mo_to_to_hypothesis(asso, num_tracks);
        //     double log_p = prior_hypothesis_conditional_association_probability(to_cond_posterior_hypothesis, h, R);
        //     log_Z += log_p;
        //     log_Z += h.log_prob();
        //     all_hypos.push_back({to_cond_posterior_hypothesis, log_Z});
        //     Z += exp(log_Z);
        // }
    }
    // std::sort(all_hypos.begin(), all_hypos.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
    // for (const auto& [h, r] : all_hypos) {
    //     std::cout << r << ", " << exp(r - log(Z)) << ": ";
    //     for (const auto& t : h) {
    //         std::cout << t << " ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << Z << "\n";
}