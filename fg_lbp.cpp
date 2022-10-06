#include "dfg_da/factor_graph.h"
#include <iostream>
#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteDistribution.h>
#include <gtsam/inference/Symbol.h>

#ifdef GLOG_AVAILABLE
#include <glog/logging.h>
#endif // GLOG_AVAILABLE

#include <cmath>


using gtsam::symbol_shorthand::A;


int main(int argc, char **argv)
{
    #ifdef GLOG_AVAILABLE
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    #endif // GLOG_AVAILABLE 

    gtsam::DiscreteFactorGraph dfg = dfg_da::factor_graph::build_test_factor_graph();

    dfg_da::factor_graph::FactorGraph fg(dfg);

    dfg_da::factor_graph::Marginals marginals = fg.lbp(32);

    std::cout << "Marginals:\n";
    for (const auto &[k, marginal] : marginals) {
        std::cout << gtsam::Symbol(k) << ": ";
        for (auto p : marginal) {
            std::cout << p << " ";
        }
        std::cout << std::endl;
    }
}