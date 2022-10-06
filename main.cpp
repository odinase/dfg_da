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

using gtsam::symbol_shorthand::A;



int main(int argc, char **argv)
{
    #ifdef GLOG_AVAILABLE
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    #endif // GLOG_AVAILABLE

    gtsam::DiscreteFactorGraph dfg = dfg_da::factor_graph::build_test_factor_graph();


    gtsam::DiscreteFactor::Values solution = dfg.optimize();
    gtsam::DiscreteMarginals marginals(dfg);

    auto dks = dfg.discreteKeys();
    std::set<gtsam::DiscreteKey> all_keys(dks.begin(), dks.end());
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
    

    dfg.saveGraph("graph.txt");
}