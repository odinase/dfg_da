// C-API implementation over dfg_da's single-cluster LBP.
// Casts opaque handles <-> C++ types; every entry point is exception-safe.

#include "dfg_da_c/lbp.h"

#include <Eigen/Core>
#include <cmath>
#include <string>
#include <vector>

#include "dfg_da/hypothesis.h"
#include "dfg_da/lbp.h"

namespace {
thread_local std::string g_last_error;

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace

// Opaque backing structs. Names match the `typedef struct <Name>_s*` in the header
// (bindgen treats them as opaque via the build.rs `.opaque_type("Dfg.*_s")` rule).
struct DfgHypotheses_s {
  std::vector<dfg_da::hypothesis::Hypothesis> hypos;
};

struct DfgLbpOutput_s {
  dfg_da::lbp::MHLBPSingleClusterOutput out;
};

extern "C" {

const char* dfg_last_error(void) { return g_last_error.c_str(); }

DfgHypotheses dfg_hypotheses_create(void) {
  try {
    return new DfgHypotheses_s{};
  } catch (const std::exception& e) {
    g_last_error = e.what();
    return nullptr;
  }
}

void dfg_hypotheses_add(DfgHypotheses h, const size_t* tracks, size_t n,
                        double log_prob) {
  if (h == nullptr) return;
  try {
    std::vector<dfg_da::hypothesis::Track> t(tracks, tracks + n);
    h->hypos.emplace_back(std::move(t), log_prob);
  } catch (const std::exception& e) {
    g_last_error = e.what();
  }
}

size_t dfg_hypotheses_len(DfgHypotheses h) {
  return h == nullptr ? 0 : h->hypos.size();
}

void dfg_hypotheses_destroy(DfgHypotheses h) { delete h; }

DfgLbpOutput dfg_lbp_single_cluster(const double* reward_matrix, size_t rows,
                                    size_t cols, DfgHypotheses hyps,
                                    size_t max_iters) {
  if (reward_matrix == nullptr || hyps == nullptr) {
    g_last_error = "null argument to dfg_lbp_single_cluster";
    return nullptr;
  }
  try {
    Eigen::Map<const RowMatrixXd> R(reward_matrix, static_cast<Eigen::Index>(rows),
                                    static_cast<Eigen::Index>(cols));
    // Hypotheses' constructor copies + log-normalizes the prior list.
    dfg_da::hypothesis::Hypotheses prior(hyps->hypos);
    auto out = dfg_da::lbp::lbp_single_cluster(R, prior, max_iters);
    return new DfgLbpOutput_s{std::move(out)};
  } catch (const std::exception& e) {
    g_last_error = e.what();
    return nullptr;
  }
}

double dfg_lbp_output_bethe_norm_const(DfgLbpOutput out) {
  if (out == nullptr) return std::nan("");
  try {
    return out->out.bethe_pseudodual_normalization_constant();
  } catch (const std::exception& e) {
    g_last_error = e.what();
    return std::nan("");
  }
}

void dfg_lbp_output_destroy(DfgLbpOutput out) { delete out; }

}  // extern "C"
