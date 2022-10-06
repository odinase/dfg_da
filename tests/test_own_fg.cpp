// Include for test suite
#include <gtest/gtest.h>

#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteDistribution.h>

#include <gtsam/inference/Symbol.h>

#include "dfg_da/factor_graph.h"
#include "dfg_da/factor_graph/nodes.h"


using gtsam::symbol_shorthand::A;



TEST(TestSuite, test_lbp_own)
{
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

    EXPECT_EQ(1, 1);
}