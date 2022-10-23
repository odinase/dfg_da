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
    
    dfg_da::factor_graph::FactorGraph fg(dfg);
    dfg_da::factor_graph::Marginals lbp_marginals = fg.lbp();
    for (const auto& [k, m] : lbp_marginals) {
        std::cout << gtsam::Symbol(k) << ": ";
        for (const auto p : m) {
            std::cout << p << " ";
        }
        std::cout << std::endl;
    }

    Eigen::ArrayXXd asso_probs = dfg_da::lbp::lbp(R, h);
    std::cout << asso_probs.transpose() << "\n";

    gtsam::Ordering order(gtsam::KeyVector{{gtsam::Symbol('b', 1), gtsam::Symbol('a', 1), gtsam::Symbol('a', 2), gtsam::Symbol('a', 3), gtsam::Symbol('T', 0)}});

    auto keys = dfg.keys();
    for (const auto& k: keys) {
        std::cout << gtsam::Symbol(k) << "\n";
    }

    auto dbn = dfg.eliminateMultifrontal();

    dbn->saveGraph("graph.txt");
}