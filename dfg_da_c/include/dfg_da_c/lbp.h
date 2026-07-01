#ifndef DFG_DA_C_LBP_H
#define DFG_DA_C_LBP_H

/*
 * Minimal C API over dfg_da's single-cluster LBP + Bethe pseudodual.
 *
 * This is the "C bridge" the Rust `dfg-da-sys` crate binds to via bindgen.
 * Handles are opaque pointers to the underlying C++ objects; every create/*
 * has a matching destroy. All functions are exception-safe (a C++ exception
 * is caught and reported as a NULL handle / NaN result + dfg_last_error()).
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle -> a builder holding std::vector<dfg_da::hypothesis::Hypothesis>. */
typedef struct DfgHypotheses_s* DfgHypotheses;

/* Opaque handle -> dfg_da::lbp::MHLBPSingleClusterOutput. */
typedef struct DfgLbpOutput_s* DfgLbpOutput;

/* Last error message from a failed call (thread-local, empty if none). */
const char* dfg_last_error(void);

/* --- Hypotheses builder --- */
DfgHypotheses dfg_hypotheses_create(void);

/* Append one prior hypothesis: `tracks` = `n` global track ids, with `log_prob`. */
void dfg_hypotheses_add(DfgHypotheses h, const size_t* tracks, size_t n, double log_prob);

size_t dfg_hypotheses_len(DfgHypotheses h);

void dfg_hypotheses_destroy(DfgHypotheses h);

/* --- Single-cluster LBP --- */
/* reward_matrix: row-major, rows x cols (the "edmund" layout). Returns NULL on error. */
DfgLbpOutput dfg_lbp_single_cluster(const double* reward_matrix, size_t rows, size_t cols,
                                    DfgHypotheses hyps, size_t max_iters);

/* Bethe pseudodual normalization constant. Returns NaN if `out` is NULL. */
double dfg_lbp_output_bethe_norm_const(DfgLbpOutput out);

void dfg_lbp_output_destroy(DfgLbpOutput out);

#ifdef __cplusplus
}
#endif

#endif /* DFG_DA_C_LBP_H */
