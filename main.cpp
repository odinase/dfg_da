#include <iostream>
#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteDistribution.h>
#include <gtsam/inference/Symbol.h>

// #include <dcsam/DCSAM_types.h>
// #include <dcsam/DiscretePriorFactor.h>

// #include <idbt/iDBT.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <limits>
#include <fstream>

#include <glog/logging.h>
#include <cmath>

#include "discrete_factor_graph/hypothesis.h"
#include "discrete_factor_graph/factor_graph.h"


using gtsam::symbol_shorthand::A;



int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    gtsam::DiscreteFactorGraph dfg{};

  // Initialize discrete prior hypothesis variable theta
    gtsam::DiscreteKey theta1(gtsam::symbol('T', 1), 2);
    gtsam::DiscreteKey theta2(gtsam::symbol('T', 2), 2);

    double p1 = 0.5;
    double p2 = 0.5;
    std::vector<std::vector<int>> prior_hypotheses = {
        {1, 2},
        {1, 3}}; // Tracks
    std::vector<double> theta1_probs{1.0 - p1, p1};
    gtsam::DiscreteDistribution phi_H1(theta1, theta1_probs);

    std::vector<double> theta2_probs{1.0 - p2, p2};
    gtsam::DiscreteDistribution phi_H2(theta2, theta2_probs);

    dfg.push_back(phi_H1);
    dfg.push_back(phi_H2);


    // Add Bernoulli components (tracks??)
    gtsam::DiscreteKeys as;
    for (int i = 1; i <= 3; i++)
    {
        as.emplace_back(gtsam::DiscreteKey(A(i), 3)); // Cardinality is 3 because one measurement => misdetection, measurement, non-existence
    }


    // Add factors between theta and the tracks
    // a1
    gtsam::DiscreteKeys phi_ths_keys = {theta1, theta2};
    std::vector<double> phi_ths_table = {
        0, 1,
        1, 0
    };

    gtsam::DecisionTreeFactor phi_ths(phi_ths_keys, phi_ths_table);
    dfg.push_back(phi_ths);

    // Add factors between theta and the tracks
    // a1
    gtsam::DiscreteKeys phi_A1_keys = {theta1, as[0]};
    std::vector<double> phi_A1_table = {
        1, 1, 1,
        1, 1, 0
    };
    gtsam::DecisionTreeFactor phi_A1(phi_A1_keys, phi_A1_table);
    dfg.push_back(phi_A1);

    gtsam::DiscreteKeys phi_A2_keys = {theta2, as[0]};
    std::vector<double> phi_A2_table = {
        1, 1, 1,
        1, 1, 0
    };
    gtsam::DecisionTreeFactor phi_A2(phi_A2_keys, phi_A2_table);
    dfg.push_back(phi_A2);

    // a2
    gtsam::DiscreteKeys phi_B1_keys = {theta1, as[1]};
    std::vector<double> phi_B1_table = {
        1, 1, 1,
        1, 1, 0
    };
    gtsam::DecisionTreeFactor phi_B1(phi_B1_keys, phi_B1_table);
    dfg.push_back(phi_B1);

    // a2
    gtsam::DiscreteKeys phi_B2_keys = {theta2, as[1]};
    std::vector<double> phi_B2_table = {
        1, 1, 1,
        0, 0, 1
    };
    gtsam::DecisionTreeFactor phi_B2(phi_B2_keys, phi_B2_table);
    dfg.push_back(phi_B2);

    // a3
    gtsam::DiscreteKeys phi_C1_keys = {theta1, as[2]};
    std::vector<double> phi_C1_table = {
        1, 1, 1,
        0, 0, 1
    };
    gtsam::DecisionTreeFactor phi_C1(phi_C1_keys, phi_C1_table);
    dfg.push_back(phi_C1);

    // a3
    gtsam::DiscreteKeys phi_C2_keys = {theta2, as[2]};
    std::vector<double> phi_C2_table = {
        1, 1, 1,
        1, 1, 0
    };
    gtsam::DecisionTreeFactor phi_C2(phi_C2_keys, phi_C2_table);
    dfg.push_back(phi_C2);


    // track-to-measurement factors
    // Define b variable
    gtsam::DiscreteKey b(gtsam::symbol('b', 0), 4); // Cardinality 4 because three different tracks or misdetection??

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

}