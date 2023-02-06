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

        Eigen::ArrayXXd MHLBPSingleClusterOutput::track_association_marginals() const
        {
            Eigen::ArrayXXd asso_probs(2 + num_measurements, num_tracks);
            asso_probs.topRows<1>() = w_0;
            asso_probs.block(1, 0, num_measurements, num_tracks) = (w_nmd * nu).transpose();
            asso_probs.bottomRows<1>() = sigma;

            asso_probs.rowwise() /= asso_probs.colwise().sum();

            return asso_probs;
        }
        Eigen::ArrayXXd MHLBPSingleClusterOutput::measurement_association_marginals() const
        {
            Eigen::ArrayXXd meas_probs(1 + num_tracks, num_measurements);
            meas_probs.topRows<1>() = 1;
            meas_probs.block(1, 0, num_tracks, num_measurements) = mu;
            meas_probs.rowwise() /= meas_probs.colwise().sum();

            return meas_probs;
        }
        Eigen::ArrayXd MHLBPSingleClusterOutput::hypotheses_marginal() const
        {
            auto rho_prods = (t2h.colwise() * rho + t2h_not).colwise().prod().transpose();
            Eigen::ArrayXd hypo_probs(num_hypotheses);
            hypo_probs = phi * rho_prods;
            hypo_probs /= hypo_probs.sum();

            return hypo_probs;
        }

        double MHLBPSingleClusterOutput::bethe_pseudodual() const
        {
        }

        MHLBPSingleClusterOutput lbp_single_cluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const hypothesis::Hypotheses &prior_hypotheses, size_t max_num_iters)
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

            return MHLBPSingleClusterOutput(
                std::move(mu),
                std::move(nu),
                std::move(rho),
                std::move(sigma),
                std::move(w_nmd),
                std::move(w_0),
                std::move(t2h),
                std::move(t2h_not),
                std::move(phi));
        }

        Eigen::ArrayXXd MHLBPMultilusterOutput::track_association_marginals() const
        {
            Eigen::ArrayXXd asso_probs(2 + num_measurements, num_tracks);
            asso_probs.topRows<1>() = w_0;
            asso_probs.block(1, 0, num_measurements, num_tracks) = (w_nmd * nu).transpose();
            asso_probs.bottomRows<1>() = sigma;

            asso_probs.rowwise() /= asso_probs.colwise().sum();

            return asso_probs;
        }
        Eigen::ArrayXXd MHLBPMultilusterOutput::measurement_association_marginals() const
        {
            Eigen::ArrayXXd meas_probs(1 + num_tracks, num_measurements);
            meas_probs.topRows<1>() = 1;
            meas_probs.block(1, 0, num_tracks, num_measurements) = mu;
            meas_probs.rowwise() /= meas_probs.colwise().sum();

            return meas_probs;
        }

        MHLBPMultilusterOutput lbp_multicluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const std::vector<hypothesis::Hypotheses> &prior_hypotheses_per_cluster, size_t max_num_iters)
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
                for (auto h = prior_hypotheses.cbegin(); h != prior_hypotheses.cend(); h++)
                {
                    for (const size_t t : h->tracks())
                    {
                        tracks_set.insert(t);
                    }
                }
                std::vector<size_t> tracks(tracks_set.begin(), tracks_set.end());
                const size_t num_tracks_in_cluster = tracks.size();
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
                    std::move(t2h)));
            }

            Eigen::ArrayXd rho = w_0 + (w_nmd * nu).rowwise().sum();

            Eigen::ArrayXd sigma(n), rho_c, rho_prods, sigma_n, sigma_d;

            for (const auto &d : cluster_data)
            {
                rho_c = rho(d.t_idx);
                rho_prods = (d.t2h.colwise() * rho_c + d.t2h_not).colwise().prod().transpose();
                sigma_n = (d.t2h_not.rowwise() * (rho_prods * d.phi()).transpose()).rowwise().sum();
                sigma_d = (d.t2h.rowwise() * (rho_prods * d.phi()).transpose()).rowwise().sum();
                sigma(d.t_idx) = rho_c * sigma_n / sigma_d;
            }

            size_t iter = 0;
            Eigen::ArrayXXd w_times_msg(n, m);

            while (iter < max_num_iters)
            {
                w_times_msg = w_nmd * nu;

                mu = w_nmd / (((-w_times_msg).colwise() + (w_times_msg.rowwise().sum() + w_0 + sigma)));
                nu = 1.0 / (1.0 + ((-mu).rowwise() + mu.colwise().sum()));

                rho = w_0 + (w_nmd * nu).rowwise().sum();

                for (const auto &d : cluster_data)
                {
                    rho_c = rho(d.t_idx);
                    rho_prods = (d.t2h.colwise() * rho_c + d.t2h_not).colwise().prod().transpose();
                    sigma_n = (d.t2h_not.rowwise() * (rho_prods * d.phi()).transpose()).rowwise().sum();
                    sigma_d = (d.t2h.rowwise() * (rho_prods * d.phi()).transpose()).rowwise().sum();
                    sigma(d.t_idx) = rho_c * sigma_n / sigma_d;
                }

                iter += 1;
            }

            return MHLBPMultilusterOutput(
                std::move(mu),
                std::move(nu),
                std::move(rho),
                std::move(sigma),
                std::move(w_nmd),
                std::move(w_0),
                std::move(cluster_data));
        }

    } // namespace lbp
} // namespace dfg_da