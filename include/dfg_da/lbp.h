#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "dfg_da/hypothesis.h"

namespace dfg_da {

namespace lbp {

Eigen::ArrayXXd lbp(const Eigen::MatrixXd& reward_matrix, const hypothesis::Hypotheses& prior_hypotheses, size_t max_num_iters = 300);

} // namespace lbp
} // namespace dfg_da