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
    R << 3.0, -inf,   -0.60, -inf, -inf, -inf, -inf,
            3.2, -inf, -inf,   -0.56, -inf, -inf, -inf,
           -3.0,     1.2, -inf, -inf,   -0.46, -inf, -inf,
        -inf,     3.0, -inf, -inf, -inf,   -0.62, -inf,
        -inf,    -0.4, -inf, -inf, -inf, -inf,   -0.55;


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

   Eigen::ArrayXXd marginals = dfg_da::lbp::lbp_multicluster(R, prior_hypotheses_per_cluster);

   std::cout << marginals.transpose() << "\n";
}