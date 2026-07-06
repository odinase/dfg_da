#ifndef DFG_DA_C_EHM2
#define DFG_DA_C_EHM2

/*
 * Minimal C API over the pyehm fork's exact EHM2 solver
 * (ehm::core::EHM2::runAndLikelihood). This is the "C bridge" the Rust
 * `dfg-da-sys` crate can bind to via bindgen (names use the `dfg_`/`Dfg`
 * prefixes its allowlists pick up). EHM2's entry points are stateless static
 * methods, so no opaque handle is needed. Every function is exception-safe:
 * a C++ exception is caught and reported via dfg_ehm2_last_error().
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Last error message from a failed EHM2 call (thread-local, empty if none). */
const char* dfg_ehm2_last_error(void);

/*
 * Exact EHM2 marginal association probabilities + loglikelihood.
 *
 *   validation_matrix: row-major int32, rows*cols (0/1 gating mask; column 0 is
 *                      the missed-detection hypothesis).
 *   likelihood_matrix: row-major f64, rows*cols (association likelihoods).
 *   out_assoc:  caller-allocated row-major f64 of rows*cols; receives the
 *               association-probability matrix.
 *   out_loglik: receives the cluster loglikelihood.
 *
 * Returns 0 on success, nonzero on error (message via dfg_ehm2_last_error()).
 */
int dfg_ehm2_run_and_likelihood(const int32_t* validation_matrix,
                                const double* likelihood_matrix,
                                size_t rows, size_t cols,
                                double* out_assoc, double* out_loglik);

#ifdef __cplusplus
}
#endif

#endif  // DFG_DA_C_EHM2
