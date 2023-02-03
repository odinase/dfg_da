#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "dfg_da/hypothesis.h"

namespace dfg_da {

namespace lbp {

Eigen::ArrayXXd lbp_single_cluster(const Eigen::Ref<const Eigen::MatrixXd>& reward_matrix, const hypothesis::Hypotheses& prior_hypotheses, size_t max_num_iters = 300);
Eigen::ArrayXXd lbp_multicluster(const Eigen::Ref<const Eigen::MatrixXd>& reward_matrix, const std::vector<hypothesis::Hypotheses>& prior_hypotheses_per_cluster, size_t max_num_iters = 300);

} // namespace lbp
} // namespace dfg_da