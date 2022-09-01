#include <iostream>
#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/inference/Symbol.h>
#include <dcsam/DCSAM_types.h>
#include <dcsam/DiscretePriorFactor.h>
#include <idbt/iDBT.h>

using gtsam::symbol_shorthand::A;

int main()
{
    // Initialize discrete prior hypothesis variable theta
    gtsam::DiscreteKey theta;
    double w_a = 0.5;
    double w_b = 1.0 - w_a;
    std::vector<std::vector<int>> prior_hypotheses = {
        {1, 2},
        {1, 3}}; // Tracks
    std::vector<double> theta_prior_probs{w_a, w_b};
    dcsam::DiscretePriorFactor phi_H(theta, theta_prior_probs);

    gtsam::DiscreteFactorGraph dfg{};
    dfg.push_back(phi_H);

    // Add Bernoulli components (tracks??)
    std::vector<gtsam::DiscreteKey> as;
    for (int i = 1; i <= 3; i++) {
        as.emplace_back(gtsam::DiscreteKey(A(i), 3)); // Cardinality is 3 because one measurement => misdetection, measurement, non-existence
    }

    // Add factors between theta and the tracks
    // a1
    gtsam::DecisionTreeFactor phi_A(
        gtsam::DiscreteKeys{{theta, as[0]}},
            "1 1 0  "
            "1 1 0"
    );

    // a2

    // a3

    // Add prio
}