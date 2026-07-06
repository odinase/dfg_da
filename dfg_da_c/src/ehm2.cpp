// C-API implementation over the pyehm fork's exact EHM2 solver.
// Marshals flat row-major arrays <-> Eigen; every entry point is exception-safe.

#include "dfg_da_c/ehm2.h"

#include <Eigen/Dense>
#include <string>
#include <tuple>

#include "core/EHM2.h"  // pyehm: third_party/pyehm/src/core/EHM2.h

namespace {
thread_local std::string g_ehm2_last_error;

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace

extern "C" {

const char* dfg_ehm2_last_error(void) { return g_ehm2_last_error.c_str(); }

int dfg_ehm2_run_and_likelihood(const int32_t* validation_matrix,
                                const double* likelihood_matrix, size_t rows,
                                size_t cols, double* out_assoc,
                                double* out_loglik) {
  if (validation_matrix == nullptr || likelihood_matrix == nullptr ||
      out_assoc == nullptr || out_loglik == nullptr) {
    g_ehm2_last_error = "null argument to dfg_ehm2_run_and_likelihood";
    return 1;
  }
  try {
    const auto r = static_cast<Eigen::Index>(rows);
    const auto c = static_cast<Eigen::Index>(cols);
    // Map the row-major inputs, then convert into EHM2's (column-major) Eigen
    // types via assignment.
    Eigen::MatrixXi validation =
        Eigen::Map<const RowMatrixXi>(reinterpret_cast<const int*>(validation_matrix), r, c);
    Eigen::MatrixXd likelihood =
        Eigen::Map<const RowMatrixXd>(likelihood_matrix, r, c);

    Eigen::MatrixXd assoc;
    double loglik;
    std::tie(assoc, loglik) =
        ehm::core::EHM2::runAndLikelihood(validation, likelihood);

    if (assoc.rows() != r || assoc.cols() != c) {
      g_ehm2_last_error = "EHM2 association matrix shape != input shape";
      return 3;
    }
    Eigen::Map<RowMatrixXd>(out_assoc, r, c) = assoc;
    *out_loglik = loglik;
    return 0;
  } catch (const std::exception& e) {
    g_ehm2_last_error = e.what();
    return 2;
  }
}

}  // extern "C"
