#include "discrete_factor_graph/lbp.h"
#include <iostream>


std::unordered_map<gtsam::DiscreteKey, std::vector<double>> lbp(const gtsam::DiscreteFactorGraph& dfg) {
    for (auto&& df : dfg) {
        std::cout << "Found factor containing keys\n";
        df->print();
    }
}