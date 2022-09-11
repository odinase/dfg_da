#pragma once

#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/inference/Symbol.h>

// #include <dcsam/DCSAM_types.h>
// #include <dcsam/DiscretePriorFactor.h>

#include <unordered_map>
#include <vector>


// struct Messages {
    
// };

std::unordered_map<gtsam::Key, std::vector<double>> lbp(
    const gtsam::DiscreteFactorGraph& dfg,
    const size_t max_iters = 100,
    const double kl_threshold = 1.0
);