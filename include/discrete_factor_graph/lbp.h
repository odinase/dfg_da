#pragma once

#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/inference/Symbol.h>

#include <dcsam/DCSAM_types.h>
#include <dcsam/DiscretePriorFactor.h>

#include <unordered_map>
#include <vector>


// struct Messages {
    
// };

using Messages = std::unordered_map<gtsam::DiscreteKey, std::vector<double>>;
using Marginals = std::unordered_map<gtsam::DiscreteKey, std::vector<double>>;

std::unordered_map<gtsam::DiscreteKey, std::vector<double>> lbp(const gtsam::DiscreteFactorGraph& dfg);