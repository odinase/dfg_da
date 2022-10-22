#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>

#include "dfg_da/hypothesis.h"
#include "dfg_da/lbp.h"

namespace dfg_da {

namespace lbp {

Eigen::ArrayXXd lbp(const Eigen::MatrixXd& reward_matrix, const hypothesis::Hypotheses& prior_hypotheses, size_t max_num_iters) {
    const size_t n = reward_matrix.rows();
    const size_t m = reward_matrix.cols() - n;

    auto w_nmd = reward_matrix.leftCols(m).array().exp();
    auto w_0 = reward_matrix.rightCols(n).diagonal().array().exp();

    double w_N = 1.0;

    Eigen::ArrayXXd mu = w_nmd / (w_0 + ((-w_nmd).colwise() + w_nmd.rowwise().sum()) + w_N);
    
    Eigen::ArrayXXd nu = 1 / (1 + ((-mu).rowwise() + mu.colwise().sum()));

    std::vector<double> prior_probs = prior_hypotheses.hypothesis_probabilites();
    const size_t num_hypotheses = prior_hypotheses.num_hypotheses();
    Eigen::Map<Eigen::ArrayXd> phi(prior_probs.data(), num_hypotheses);

    Eigen::ArrayXd rho = w_0 + (w_nmd * nu).rowwise().sum();

    Eigen::ArrayXXd t2h(n, num_hypotheses), t2h_not(n, num_hypotheses);

    for (size_t t = 1, i = 0; t <= n; t++, i++) {
        for (size_t h = 0; h < num_hypotheses; h++) {
            t2h(i,h) = prior_hypotheses[h].contains(t);
            t2h_not(i,h) = !t2h(i,h);
        }
    }

    auto rho_prods = (t2h.colwise() * rho + t2h_not).colwise().prod();
    auto sigma_n = (t2h_not.rowwise()* (rho_prods * phi.transpose())).rowwise().sum();
    auto sigma_d = (t2h.rowwise()*(rho_prods*phi.transpose())).rowwise().sum() / rho;
    Eigen::ArrayXd sigma = sigma_n / sigma_d;


    size_t iter = 0;
    Eigen::ArrayXXd w_times_msg(n, m);
    std::vector<double> sigma_compute;

    while (iter < max_num_iters) {
        w_times_msg = w_nmd * nu;


        mu = w_nmd / (w_0 + ((-w_times_msg).colwise() + w_times_msg.rowwise().sum()) + w_N*sigma);
        nu = 1 / (1 + ((-mu).rowwise() + mu.colwise().sum()));

        rho = w_0 + (w_nmd * nu).rowwise().sum();

        auto start = std::chrono::high_resolution_clock::now();
        auto rho_prods = (t2h.colwise() * rho + t2h_not).colwise().prod();
        auto sigma_n = (t2h_not.rowwise() * (rho_prods * phi.transpose())).rowwise().sum();
        auto sigma_d = (t2h.rowwise()*(rho_prods*phi.transpose())).rowwise().sum() / rho;
        sigma = sigma_n / sigma_d;
        auto stop = std::chrono::high_resolution_clock::now();
        sigma_compute.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1e-3);

        iter += 1;
    }

    // Column-major, so each column is a marginal distribution
    Eigen::ArrayXXd asso_probs(2 + m, n);
    asso_probs.row(0) = w_0;
    asso_probs.block(1, 0, m, n) = (w_nmd * nu).transpose();
    asso_probs.row(m + 1) = w_N * sigma;

    asso_probs.rowwise() /= asso_probs.colwise().sum();

    double sum = std::accumulate(sigma_compute.begin(), sigma_compute.end(), 0.0);
    double mean = sum / sigma_compute.size();

    double sq_sum = std::inner_product(sigma_compute.begin(), sigma_compute.end(), sigma_compute.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / sigma_compute.size() - mean * mean);

    std::cout << "Mean: " << mean << " +- " << stdev << "\n";

    return asso_probs;
}

} // namespace lbp
} // namespace dfg_da