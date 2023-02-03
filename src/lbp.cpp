#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <set>

#include "dfg_da/hypothesis.h"
#include "dfg_da/lbp.h"

namespace dfg_da
{

    namespace lbp
    {

        Eigen::ArrayXXd lbp_single_cluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const hypothesis::Hypotheses &prior_hypotheses, size_t max_num_iters)
        {
            const size_t n = reward_matrix.rows();
            const size_t m = reward_matrix.cols() - n;

            Eigen::ArrayXXd w_nmd = reward_matrix.leftCols(m).array().exp();
            Eigen::ArrayXd w_0 = reward_matrix.rightCols(n).diagonal().array().exp();

            Eigen::ArrayXXd mu = w_nmd / (((-w_nmd).colwise() + (w_nmd.rowwise().sum() + w_0)) + 1.0);

            Eigen::ArrayXXd nu = 1.0 / (1.0 + ((-mu).rowwise() + mu.colwise().sum()));

            std::vector<double> prior_probs = prior_hypotheses.hypothesis_probabilites();
            const size_t num_hypotheses = prior_hypotheses.num_hypotheses();
            Eigen::Map<Eigen::ArrayXd> phi(prior_probs.data(), num_hypotheses);

            Eigen::ArrayXd rho = w_0 + (w_nmd * nu).rowwise().sum();

            Eigen::ArrayXXd t2h(n, num_hypotheses), t2h_not(n, num_hypotheses);

            for (size_t t = 1, i = 0; t <= n; t++, i++)
            {
                for (size_t h = 0; h < num_hypotheses; h++)
                {
                    t2h(i, h) = prior_hypotheses[h].contains(t);
                    t2h_not(i, h) = !t2h(i, h);
                }
            }

            Eigen::ArrayXd rho_prods = (t2h.colwise() * rho + t2h_not).colwise().prod().transpose();
            Eigen::ArrayXd sigma_n = (t2h_not.rowwise() * (rho_prods * phi).transpose()).rowwise().sum();
            Eigen::ArrayXd sigma_d = (t2h.rowwise() * (rho_prods * phi).transpose()).rowwise().sum() / rho + 1e-16; // Add small value to avoid divide by zero
            Eigen::ArrayXd sigma = sigma_n / sigma_d;

            size_t iter = 0;
            Eigen::ArrayXXd w_times_msg(n, m);

            while (iter < max_num_iters)
            {
                w_times_msg = w_nmd * nu;

                mu = w_nmd / (((-w_times_msg).colwise() + (w_times_msg.rowwise().sum() + w_0 + sigma)));
                nu = 1.0 / (1.0 + ((-mu).rowwise() + mu.colwise().sum()));

                rho = w_0 + (w_nmd * nu).rowwise().sum();

                rho_prods = (t2h.colwise() * rho + t2h_not).colwise().prod().transpose();
                sigma_n = (t2h_not.rowwise() * (rho_prods * phi).transpose()).rowwise().sum();
                sigma_d = (t2h.rowwise() * (rho_prods * phi).transpose()).rowwise().sum() / rho + 1e-16;
                sigma = sigma_n / sigma_d;

                iter += 1;
            }

            // Column-major, so each column is a marginal distribution
            Eigen::ArrayXXd asso_probs(2 + m, n), meas_probs(1 + n, m);
            asso_probs.topRows<1>() = w_0;
            asso_probs.block(1, 0, m, n) = (w_nmd * nu).transpose();
            asso_probs.bottomRows<1>() = sigma;

            asso_probs.rowwise() /= asso_probs.colwise().sum();

            // meas_probs.topRows<1>() = 1;
            // meas_probs.block(1, 0, n, m) = mu;
            // meas_probs.rowwise() /= meas_probs.colwise().sum();
            // std::cout << meas_probs << "\n";

            // rho_prods = (t2h.colwise() * rho + t2h_not).colwise().prod().transpose();
            // Eigen::ArrayXd hypo_probs(num_hypotheses);
            // hypo_probs = phi * rho_prods;
            // hypo_probs /= hypo_probs.sum();
            // std::cout << hypo_probs << "\n";

            return asso_probs;
        }

        struct ClusterData
        {
            std::vector<double> phi__;
            std::vector<size_t> t_idx;
            Eigen::ArrayXXd t2h_not;
            Eigen::ArrayXXd t2h;

            ClusterData(
                std::vector<double> &&phi_,
                std::vector<size_t> &&tracks_,
                Eigen::ArrayXXd &&t2hnot_idx_,
                Eigen::ArrayXXd &&t2h_idx_)
                : phi__(std::move(phi_)),
                  t_idx(std::move(tracks_)),
                  t2h_not(std::move(t2hnot_idx_)),
                  t2h(std::move(t2h_idx_))
            {
                size_t nt = t_idx.size();
                for (size_t t = 0; t < nt; t++) {
                    t_idx[t] -= 1;
                }
            }
            inline Eigen::Map<const Eigen::ArrayXd> phi() const {
                return Eigen::Map<const Eigen::ArrayXd>(phi__.data(), phi__.size());
            }
        };

        void update_sigma(Eigen::Ref<Eigen::ArrayXd> sigma, const Eigen::Ref<const Eigen::ArrayXd> &rho, const std::vector<ClusterData> &cluster_data)
        {
            for (const auto& d : cluster_data) {
                auto rho_c = rho(d.t_idx);
                auto rho_prods = (d.t2h.colwise() * rho_c + d.t2h_not).colwise().prod().transpose();
                auto sigma_n = (d.t2h_not.rowwise() * (rho_prods * d.phi()).transpose()).rowwise().sum();
                auto sigma_d = (d.t2h.rowwise() * (rho_prods * d.phi()).transpose()).rowwise().sum();
                sigma(d.t_idx) = rho_c * sigma_n / sigma_d;
            }
        }

        Eigen::ArrayXXd lbp_multicluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const std::vector<hypothesis::Hypotheses> &prior_hypotheses_per_cluster, size_t max_num_iters)
        {
            const size_t n = reward_matrix.rows();
            const size_t m = reward_matrix.cols() - n;

            Eigen::ArrayXXd w_nmd = reward_matrix.leftCols(m).array().exp();
            Eigen::ArrayXd w_0 = reward_matrix.rightCols(n).diagonal().array().exp();

            Eigen::ArrayXXd mu = w_nmd / (((-w_nmd).colwise() + (w_nmd.rowwise().sum() + w_0)) + 1.0);

            Eigen::ArrayXXd nu = 1.0 / (1.0 + ((-mu).rowwise() + mu.colwise().sum()));

            std::vector<ClusterData> cluster_data;
            const size_t num_clusters = prior_hypotheses_per_cluster.size();
            cluster_data.reserve(num_clusters);

            for (const auto &prior_hypotheses : prior_hypotheses_per_cluster)
            {
                std::set<size_t> tracks_set;
                for (auto h = prior_hypotheses.cbegin(); h != prior_hypotheses.cend(); h++) {
                    for (const size_t t : h->tracks()) {
                        tracks_set.insert(t);
                    }
                }
                std::vector<size_t> tracks(tracks_set.begin(), tracks_set.end());
                const size_t num_tracks_in_cluster= tracks.size();
                std::vector<double> prior_probs = prior_hypotheses.hypothesis_probabilites();
                const size_t num_hypotheses = prior_hypotheses.num_hypotheses();
                Eigen::ArrayXXd t2h(num_tracks_in_cluster, num_hypotheses), t2h_not(num_tracks_in_cluster, num_hypotheses);
                for (size_t i = 0; i < num_tracks_in_cluster; i++)
                {
                    for (size_t h = 0; h < num_hypotheses; h++)
                    {
                        t2h(i, h) = prior_hypotheses[h].contains(tracks[i]);
                        t2h_not(i, h) = !t2h(i, h);
                    }
                }
                cluster_data.emplace_back(ClusterData(
                    std::move(prior_probs),
                    std::move(tracks),
                    std::move(t2h_not),
                    std::move(t2h)
                ));
            }

            Eigen::ArrayXd rho = w_0 + (w_nmd * nu).rowwise().sum();

            Eigen::ArrayXd sigma(n);

            update_sigma(sigma, rho, cluster_data);

            size_t iter = 0;
            Eigen::ArrayXXd w_times_msg(n, m);

            while (iter < max_num_iters)
            {
                w_times_msg = w_nmd * nu;

                mu = w_nmd / (((-w_times_msg).colwise() + (w_times_msg.rowwise().sum() + w_0 + sigma)));
                nu = 1.0 / (1.0 + ((-mu).rowwise() + mu.colwise().sum()));

                rho = w_0 + (w_nmd * nu).rowwise().sum();

                update_sigma(sigma, rho, cluster_data);

                iter += 1;
            }

            // Column-major, so each column is a marginal distribution
            Eigen::ArrayXXd asso_probs(2 + m, n), meas_probs(1 + n, m);
            asso_probs.topRows<1>() = w_0;
            asso_probs.block(1, 0, m, n) = (w_nmd * nu).transpose();
            asso_probs.bottomRows<1>() = sigma;

            asso_probs.rowwise() /= asso_probs.colwise().sum();

            // meas_probs.topRows<1>() = 1;
            // meas_probs.block(1, 0, n, m) = mu;
            // meas_probs.rowwise() /= meas_probs.colwise().sum();
            // std::cout << meas_probs << "\n";

            // rho_prods = (t2h.colwise() * rho + t2h_not).colwise().prod().transpose();
            // Eigen::ArrayXd hypo_probs(num_hypotheses);
            // hypo_probs = phi * rho_prods;
            // hypo_probs /= hypo_probs.sum();
            // std::cout << hypo_probs << "\n";

            return asso_probs;
        }

    } // namespace lbp
} // namespace dfg_da