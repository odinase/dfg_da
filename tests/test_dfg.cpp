// Include for test suite
#include <gtest/gtest.h>

#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteDistribution.h>

#include <gtsam/inference/Symbol.h>


#include "dfg_da/hypothesis.h"
#include "dfg_da/factor_graph.h"

#ifdef GLOG_AVAILABLE
#include <glog/logging.h>
#endif // GLOG_AVAILABLE


using gtsam::symbol_shorthand::A;


TEST(TestSuite, test_lbp)
{
    #ifdef GLOG_AVAILABLE
    google::InstallFailureSignalHandler();
    #endif // GLOG_AVAILABLE

    Eigen::MatrixXd R(3, 3 + 1);
    constexpr double inf = std::numeric_limits<double>::infinity();
    R << 4.78, -0.46, -inf, -inf,
         5.37, -inf, -0.52, -inf,
         6.58, -inf, -inf, -0.60;

    dfg_da::hypothesis::Hypothesis h1({1, 2}, log(0.5));
    dfg_da::hypothesis::Hypothesis h2({1, 3}, log(0.5));

    dfg_da::hypothesis::Hypotheses h{{h1, h2}};

    gtsam::DiscreteFactorGraph dfg_test = dfg_da::factor_graph::dfg_from_reward_mat_hyp_prior(R, h);
    gtsam::DiscreteFactorGraph dfg_correct = dfg_da::factor_graph::build_test_factor_graph();

    // GTSAM checks the order as well, so we sort first
    std::sort(dfg_test.begin(), dfg_test.end(), [](const auto& lhs, const auto& rhs) { return lhs->size() < rhs->size(); });
    std::sort(dfg_correct.begin(), dfg_correct.end(), [](const auto& lhs, const auto& rhs) { return lhs->size() < rhs->size(); });

    EXPECT_TRUE(dfg_test.equals(dfg_correct));
}


TEST(TestSuite, test_hypothesis_tree)
{
    #ifdef GLOG_AVAILABLE
    google::InstallFailureSignalHandler();
    #endif // GLOG_AVAILABLE

    Eigen::MatrixXd R(3, 3 + 1);
    constexpr double inf = std::numeric_limits<double>::infinity();
    R << 4.78, -0.46, -inf, -inf,
         5.37, -inf, -0.52, -inf,
         6.58, -inf, -inf, -0.60;

    dfg_da::hypothesis::Hypothesis h({1, 2, 3}, log(1.0));

    std::vector<std::vector<size_t>> hypotheses = dfg_da::hypothesis::hypothesis_enumeration(R, h);

    std::cout << "Length hypotheses: " << hypotheses.size() << "\n";
    for (const auto& hypothesis : hypotheses) {
        for (const auto& association : hypothesis) {
            std::cout << association << " ";
        }
        std::cout << "\n";
    }

    EXPECT_TRUE(true);
}