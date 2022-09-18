// Include for test suite
#include <gtest/gtest.h>

#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteDistribution.h>

#include <gtsam/inference/Symbol.h>

// #include <dcsam/DCSAM_types.h>
// #include <dcsam/DiscretePriorFactor.h>

// #include "discrete_factor_graph/lbp.h"
#include "discrete_factor_graph/hypothesis.h"
#include "discrete_factor_graph/factor_graph.h"


#include <glog/logging.h>



using gtsam::symbol_shorthand::A;


gtsam::DiscreteFactorGraph build_test_factor_graph() {
    // Initialize discrete prior hypothesis variable theta
    gtsam::DiscreteKey theta(gtsam::symbol('t', 0), 2);

    double w_a = 0.5;
    double w_b = 1.0 - w_a;
    std::vector<std::vector<int>> prior_hypotheses = {
        {1, 2},
        {1, 3}}; // Tracks
    std::vector<double> theta_prior_probs{w_a, w_b};
    gtsam::DiscreteDistribution phi_H(theta, theta_prior_probs);

    gtsam::DiscreteFactorGraph dfg{};
    dfg.push_back(phi_H);

    // Add Bernoulli components (tracks??)
    gtsam::DiscreteKeys as;
    for (int i = 1; i <= 3; i++)
    {
        as.emplace_back(gtsam::DiscreteKey(A(i), 3)); // Cardinality is 3 because one measurement => misdetection, measurement, non-existence
    }

    // Add factors between theta and the tracks
    // a1
    gtsam::DiscreteKeys phi_A_keys = {theta, as[0]};
    std::vector<double> phi_A_table = {
        1, 1, 0,
        1, 1, 0
    };
    gtsam::DecisionTreeFactor phi_A(phi_A_keys, phi_A_table);
    dfg.push_back(phi_A);

    // a2
    gtsam::DiscreteKeys phi_B_keys = {theta, as[1]};
    std::vector<double> phi_B_table = {
        1, 1, 0,
        0, 0, 1
    };
    gtsam::DecisionTreeFactor phi_B(phi_B_keys, phi_B_table);
    dfg.push_back(phi_B);

    // a3
    gtsam::DiscreteKeys phi_C_keys = {theta, as[2]};
    std::vector<double> phi_C_table = {
        0, 0, 1,
        1, 1, 0
    };
    gtsam::DecisionTreeFactor phi_C(phi_C_keys, phi_C_table);
    dfg.push_back(phi_C);


    // track-to-measurement factors
    // Define b variable
    gtsam::DiscreteKey b(gtsam::symbol('b', 1), 4); // Cardinality 4 because three different tracks or misdetection??

    // a1
    gtsam::DiscreteKeys phi_X_keys = {as[0], b};
    std::vector<double> phi_X_table = {
        1, 0, 1, 1,
        0, 1, 0, 0,
        1, 0, 1, 1,
    };
    gtsam::DecisionTreeFactor phi_X(phi_X_keys, phi_X_table);
    dfg.push_back(phi_X);

    // a2
    gtsam::DiscreteKeys phi_Y_keys = {as[1], b};
    std::vector<double> phi_Y_table = {
        1, 1, 0, 1,
        0, 0, 1, 0,
        1, 1, 0, 1,
    };
    gtsam::DecisionTreeFactor phi_Y(phi_Y_keys, phi_Y_table);
    dfg.push_back(phi_Y);

    // a3
    gtsam::DiscreteKeys phi_Z_keys = {as[2], b};
    std::vector<double> phi_Z_table = {
        1, 1, 1, 0,
        0, 0, 0, 1,
        1, 1, 1, 0,
    };
    gtsam::DecisionTreeFactor phi_Z(phi_Z_keys, phi_Z_table);
    dfg.push_back(phi_Z);

    // Add unary track factors
    // Reward matrix
    // R = [ vertcat( l^{11}, l^{21}, l^{31} , [m^1, -infty, -infty ; -infty, m^2 , -infty ; -infty, -infty, m^3].
    // Three tracks and one measurement plus three misdetections
    constexpr double inf = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd R(3, 4);
    R << 4.78, -0.46, -inf, -inf,
         5.37, -inf, -0.52, -inf,
         6.58, -inf, -inf, -0.60;

    // exp to convert log into actual probabilities. Is this properly normalized?? Does it need to??
    double l_11 = exp(R(0,0));
    double l_21 = exp(R(1,0));
    double l_31 = exp(R(2,0));

    double m_1 = exp(R(0, 1));
    double m_2 = exp(R(1, 2));
    double m_3 = exp(R(2, 3));

    // phi D
    std::vector<double> phi_D_table{m_1, l_11, 1};
    gtsam::DiscreteDistribution phi_D(as[0], phi_D_table);
    dfg.push_back(phi_D);

    // phi E
    std::vector<double> phi_E_table{m_2, l_21, 1};
    gtsam::DiscreteDistribution phi_E(as[1], phi_E_table);
    dfg.push_back(phi_E);

    // phi F
    std::vector<double> phi_F_table{m_3, l_31, 1};
    gtsam::DiscreteDistribution phi_F(as[2], phi_F_table);
    dfg.push_back(phi_F);


    return dfg;
}



// TEST(TestSuite, test_compile)
// {
//     EXPECT_EQ(1, 1);
// }

// // TEST(TestSuite, test_loop_factors)
// // {
// //     gtsam::DiscreteFactorGraph dfg = build_test_factor_graph();

// //     for (auto&& df : dfg) {
// //         df->print();
// //     }

// //     EXPECT_EQ(1, 1);
// // }

// TEST(TestSuite, test_lbp)
// {
//     gtsam::DiscreteFactorGraph dfg = build_test_factor_graph();

//     auto marginals = lbp(dfg, 32);

//     std::cout << "Marginals:\n";
//     for (const auto &[k, marginal] : marginals) {
//         std::cout << gtsam::Symbol(k) << ": ";
//         for (auto p : marginal) {
//             std::cout << p << " ";
//         }
//         std::cout << std::endl;
//     }

//     EXPECT_EQ(1, 1);
// }


TEST(TestSuite, test_lbp)
{
    google::InstallFailureSignalHandler();


    Eigen::MatrixXd R(3, 3 + 1);
    constexpr double inf = std::numeric_limits<double>::infinity();
    R << 4.78, -0.46, -inf, -inf,
         5.37, -inf, -0.52, -inf,
         6.58, -inf, -inf, -0.60;

    Hypothesis h1({1, 2}, log(0.5));
    Hypothesis h2({1, 3}, log(0.5));

    Hypotheses h{{h1, h2}};

    gtsam::DiscreteFactorGraph dfg_test = dfg_from_reward_mat_hyp_prior(R, h);
    gtsam::DiscreteFactorGraph dfg_correct = build_test_factor_graph();

    // GTSAM checks the order as well, so we sort first
    std::sort(dfg_test.begin(), dfg_test.end(), [](const auto& lhs, const auto& rhs) { return lhs->size() < rhs->size(); });
    std::sort(dfg_correct.begin(), dfg_correct.end(), [](const auto& lhs, const auto& rhs) { return lhs->size() < rhs->size(); });

    EXPECT_TRUE(dfg_test.equals(dfg_correct));
}


TEST(TestSuite, test_hypothesis_tree)
{
    google::InstallFailureSignalHandler();

    Eigen::MatrixXd R(3, 3 + 1);
    constexpr double inf = std::numeric_limits<double>::infinity();
    R << 4.78, -0.46, -inf, -inf,
         5.37, -inf, -0.52, -inf,
         6.58, -inf, -inf, -0.60;


    std::vector<std::vector<size_t>> hypotheses = hypothesis_enumeration(R);

    std::cout << "Length hypotheses: " << hypotheses.size() << "\n";
    for (const auto& hypothesis : hypotheses) {
        for (const auto& association : hypothesis) {
            std::cout << association << " ";
        }
        std::cout << "\n";
    }

    EXPECT_TRUE(true);
}