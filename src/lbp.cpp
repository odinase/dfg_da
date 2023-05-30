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

        double MHLBPSingleClusterOutput::bethe_pseudodual_loglikelihood() const
        {
            double Z_theta = hypotheses_normalization_constant();
            Eigen::ArrayXd Z_tths = track_hypo_normalization_constants();
            Eigen::ArrayXd Z_ts = track_normalization_constants();
            Eigen::ArrayXd Z_js = meas_normalization_constants();
            Eigen::ArrayXXd Z_tjs = track_meas_normalization_constants();

            double F_theta = (num_tracks - 1) * log(Z_theta);
            double F_ts = num_measurements * Z_ts.log().sum();
            double F_js = (num_tracks - 1) * Z_js.log().sum();
            double F_tths = Z_tths.log().sum();
            double F_tjs = Z_tjs.log().sum();

            double F_bethe_pseudo = F_theta + F_ts + F_js - F_tjs - F_tths;

            return -F_bethe_pseudo;
        }

        Eigen::ArrayXd MHLBPSingleClusterOutput::track_normalization_constants() const
        {
            return w_0 + (w_nmd * nu).rowwise().sum() + sigma;
        }
        Eigen::ArrayXd MHLBPSingleClusterOutput::meas_normalization_constants() const
        {
            return mu.colwise().sum().transpose() + 1.0;
        }
        double MHLBPSingleClusterOutput::hypotheses_normalization_constant() const
        {
            double Z_theta = ((t2h.colwise() * rho + t2h_not).colwise().prod().transpose() * phi).sum();
            
            return Z_theta;
        }
        Eigen::ArrayXXd MHLBPSingleClusterOutput::track_meas_normalization_constants() const
        {
            Eigen::ArrayXXd w_times_msg = w_nmd * nu;
            return (1.0 + ((-mu).rowwise() + mu.colwise().sum())) * ((-w_times_msg).colwise() + (w_times_msg.rowwise().sum() + w_0 + sigma)) + w_nmd;
        }
        Eigen::ArrayXd MHLBPSingleClusterOutput::track_hypo_normalization_constants() const
        {
            Eigen::ArrayXd w_sum_times_msg = (w_nmd * nu).rowwise().sum() + w_0;
            Eigen::ArrayXd phi_rho_prods = phi * (t2h.colwise() * rho + t2h_not).colwise().prod().transpose();
            Eigen::ArrayXd Z_tth = w_sum_times_msg / rho * (t2h.rowwise() * phi_rho_prods.transpose()).rowwise().sum() + (t2h_not.rowwise() * phi_rho_prods.transpose()).rowwise().sum();

            return Z_tth;
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
                std::move(Eigen::ArrayXd{phi}));
        }

        Eigen::ArrayXXd MHLBPMulticlusterOutput::track_association_marginals() const
        {
            Eigen::ArrayXXd asso_probs(2 + num_measurements, num_tracks);
            track_association_marginals_inplace(asso_probs.data());

            return asso_probs;
        }
        void MHLBPMulticlusterOutput::track_association_marginals_inplace(double *data) const
        {
            Eigen::Map<Eigen::ArrayXXd> asso_probs(data, 2 + num_measurements, num_tracks);
            asso_probs.topRows<1>() = w_0.transpose();
            asso_probs.block(1, 0, num_measurements, num_tracks) = (w_nmd * nu).transpose();
            asso_probs.bottomRows<1>() = sigma;

            asso_probs.rowwise() /= asso_probs.colwise().sum();
        }

        Eigen::ArrayXXd MHLBPMulticlusterOutput::measurement_association_marginals() const
        {
            Eigen::ArrayXXd meas_probs(1 + num_tracks, num_measurements);
            meas_probs.topRows<1>() = 1.0;
            meas_probs.block(1, 0, num_tracks, num_measurements) = mu;
            meas_probs.rowwise() /= meas_probs.colwise().sum();

            return meas_probs;
        }

        std::vector<Eigen::ArrayXd> MHLBPMulticlusterOutput::hypotheses_marginals() const {
            std::vector<Eigen::ArrayXd> marginals;
            Eigen::ArrayXd rho_c, rho_prods;
            std::transform(
                cluster_data.begin(), cluster_data.end(),
                std::back_inserter(marginals),
                [&](const auto &d)
                {
                    rho_c = rho(d.t_idx);
                    rho_prods = (d.t2h.colwise() * rho_c + d.t2h_not).colwise().prod().transpose();
                    Eigen::ArrayXd hypo_probs = d.phi() * rho_prods;
                    hypo_probs /= hypo_probs.sum();
                    return hypo_probs;
                });

            return marginals;
        }

        double MHLBPMulticlusterOutput::bethe_pseudodual_loglikelihood() const
        {
            Eigen::ArrayXd Z_thetas = hypotheses_normalization_constants();
            std::vector<Eigen::ArrayXd> Z_tths = track_hypos_normalization_constants();
            Eigen::ArrayXd Z_ts = track_normalization_constants();
            Eigen::ArrayXd Z_js = meas_normalization_constants();
            Eigen::ArrayXXd Z_tjs = track_meas_normalization_constants();

            Eigen::ArrayXd num_tracks_per_cluster(num_clusters);
            std::transform(cluster_data.begin(), cluster_data.end(), num_tracks_per_cluster.begin(), [](const ClusterData &d)
                           { return d.t_idx.size(); });

            double F_thetas = ((num_tracks_per_cluster - 1) * Z_thetas.log()).sum();
            double F_ts = num_measurements * Z_ts.log().sum();
            double F_js = (num_tracks - 1) * Z_js.log().sum();
            double F_tths = std::accumulate(Z_tths.begin(), Z_tths.end(), 0.0, [](const double prev, const Eigen::ArrayXXd &Z)
                                            { return prev + Z.log().sum(); });
            double F_tjs = Z_tjs.log().sum();

            double F_bethe_pseudo = F_thetas + F_ts + F_js - F_tjs - F_tths;

            return -F_bethe_pseudo;
        }

        Eigen::ArrayXd MHLBPMulticlusterOutput::track_normalization_constants() const
        {
            return w_0 + (w_nmd * nu).rowwise().sum() + sigma;
        }
        Eigen::ArrayXd MHLBPMulticlusterOutput::meas_normalization_constants() const
        {
            return mu.colwise().sum().transpose() + 1.0;
        }
        Eigen::ArrayXd MHLBPMulticlusterOutput::hypotheses_normalization_constants() const
        {
            Eigen::ArrayXd rho_c, rho_prods, Z_thetas(num_clusters);
            std::transform(
                cluster_data.begin(), cluster_data.end(),
                std::begin(Z_thetas),
                [&](const auto &d)
                {
                    rho_c = rho(d.t_idx);
                    rho_prods = (d.t2h.colwise() * rho_c + d.t2h_not).colwise().prod().transpose();
                    return (rho_prods * d.phi()).sum();
                });

            return Z_thetas;
        }
        Eigen::ArrayXXd MHLBPMulticlusterOutput::track_meas_normalization_constants() const
        {
            Eigen::ArrayXXd w_times_msg = w_nmd * nu;
            return (1.0 + ((-mu).rowwise() + mu.colwise().sum())) * ((-w_times_msg).colwise() + (w_times_msg.rowwise().sum() + w_0 + sigma)) + w_nmd;
        }
        std::vector<Eigen::ArrayXd> MHLBPMulticlusterOutput::track_hypos_normalization_constants() const
        {
            std::vector<Eigen::ArrayXd> Z_tth(num_clusters);
            Eigen::ArrayXd rho_c, phi_rho_prods, w_sum, w_sum_times_msg = (w_nmd * nu).rowwise().sum() + w_0;
            std::transform(
                cluster_data.begin(), cluster_data.end(),
                Z_tth.begin(),
                [&](const ClusterData &d)
                {
                    rho_c = rho(d.t_idx);
                    phi_rho_prods = d.phi() * (d.t2h.colwise() * rho_c + d.t2h_not).colwise().prod().transpose();
                    w_sum = w_sum_times_msg(d.t_idx);
                    return w_sum / rho_c * (d.t2h.rowwise() * phi_rho_prods.transpose()).rowwise().sum() + (d.t2h_not.rowwise() * phi_rho_prods.transpose()).rowwise().sum();
                });

            return Z_tth;
        }

        Eigen::ArrayXd track_normalization_constants(
            const Eigen::ArrayXXd &nu,
            const Eigen::ArrayXd &sigma,
            const Eigen::ArrayXXd &w_nmd,
            const Eigen::ArrayXd &w_0
        )
        {
            return w_0 + (w_nmd * nu).rowwise().sum() + sigma;
        }
        Eigen::ArrayXd meas_normalization_constants(
            const Eigen::ArrayXXd &mu
        )
        {
            return mu.colwise().sum().transpose() + 1.0;
        }
        Eigen::ArrayXd hypotheses_normalization_constants(
            const Eigen::ArrayXd &rho,
            const std::vector<ClusterData> &cluster_data)
        {
            const size_t num_clusters = cluster_data.size();
            Eigen::ArrayXd rho_c, rho_prods, Z_thetas(num_clusters);
            std::transform(
                cluster_data.begin(), cluster_data.end(),
                std::begin(Z_thetas),
                [&](const auto &d)
                {
                    rho_c = rho(d.t_idx);
                    rho_prods = (d.t2h.colwise() * rho_c + d.t2h_not).colwise().prod().transpose();
                    return (rho_prods * d.phi()).sum();
                });

            return Z_thetas;
        }
        Eigen::ArrayXXd track_meas_normalization_constants(
            const Eigen::ArrayXXd &mu,
            const Eigen::ArrayXXd &nu,
            const Eigen::ArrayXd &sigma,
            const Eigen::ArrayXXd &w_nmd,
            const Eigen::ArrayXd &w_0
        )
        {
            Eigen::ArrayXXd w_times_msg = w_nmd * nu;
            return (1.0 + ((-mu).rowwise() + mu.colwise().sum())) * ((-w_times_msg).colwise() + (w_times_msg.rowwise().sum() + w_0 + sigma)) + w_nmd;
        }
        std::vector<Eigen::ArrayXd> track_hypos_normalization_constants(
            const Eigen::ArrayXXd &nu,
            const Eigen::ArrayXd &rho,
            const Eigen::ArrayXXd &w_nmd,
            const Eigen::ArrayXd &w_0,
            const std::vector<ClusterData> &cluster_data)
        {
            const size_t num_clusters = cluster_data.size();
            std::vector<Eigen::ArrayXd> Z_tth(num_clusters);
            Eigen::ArrayXd rho_c, phi_rho_prods, w_sum, w_sum_times_msg = (w_nmd * nu).rowwise().sum() + w_0;
            std::transform(
                cluster_data.begin(), cluster_data.end(),
                Z_tth.begin(),
                [&](const ClusterData &d)
                {
                    rho_c = rho(d.t_idx);
                    phi_rho_prods = d.phi() * (d.t2h.colwise() * rho_c + d.t2h_not).colwise().prod().transpose();
                    w_sum = w_sum_times_msg(d.t_idx);
                    return w_sum / rho_c * (d.t2h.rowwise() * phi_rho_prods.transpose()).rowwise().sum() + (d.t2h_not.rowwise() * phi_rho_prods.transpose()).rowwise().sum();
                });

            return Z_tth;
        }

        double bethe_pseudodual_loglikelihood(const Eigen::ArrayXXd &mu,
                                              const Eigen::ArrayXXd &nu,
                                              const Eigen::ArrayXd &rho,
                                              const Eigen::ArrayXd &sigma,
                                              const Eigen::ArrayXXd &w_nmd,
                                              const Eigen::ArrayXd &w_0,
                                              const std::vector<ClusterData> &cluster_data)
        {
            const size_t num_tracks(w_nmd.rows());
            const size_t num_measurements(w_nmd.cols());
            const size_t num_clusters(cluster_data.size());

            Eigen::ArrayXd Z_thetas = hypotheses_normalization_constants(rho, cluster_data);
            std::vector<Eigen::ArrayXd> Z_tths = track_hypos_normalization_constants(nu, rho, w_nmd, w_0, cluster_data);
            Eigen::ArrayXd Z_ts = track_normalization_constants(nu, sigma, w_nmd, w_0);
            Eigen::ArrayXd Z_js = meas_normalization_constants(mu);
            Eigen::ArrayXXd Z_tjs = track_meas_normalization_constants(mu, nu, sigma, w_nmd, w_0);

            Eigen::ArrayXd num_tracks_per_cluster(num_clusters);
            std::transform(cluster_data.begin(), cluster_data.end(), num_tracks_per_cluster.begin(), [](const ClusterData &d)
                           { return d.t_idx.size(); });

            double F_thetas = ((num_tracks_per_cluster - 1) * Z_thetas.log()).sum();
            double F_ts = num_measurements * Z_ts.log().sum();
            double F_js = (num_tracks - 1) * Z_js.log().sum();
            double F_tths = std::accumulate(Z_tths.begin(), Z_tths.end(), 0.0, [](const double prev, const Eigen::ArrayXXd &Z)
                                            { return prev + Z.log().sum(); });
            double F_tjs = Z_tjs.log().sum();

            double F_bethe_pseudo = F_thetas + F_ts + F_js - F_tjs - F_tths;

            return -F_bethe_pseudo;
        }

        double message_norm(const Eigen::ArrayXXd &nu, const Eigen::ArrayXXd &nu_prev) {
            Eigen::ArrayXXd nu_ratio = nu / nu_prev;
            double max_ratio = nu_ratio.maxCoeff();
            double min_ratio = nu_ratio.minCoeff();
            double max_abs = std::max(max_ratio, 1.0 / min_ratio);
            return std::log(max_abs);
        }

        MHLBPMulticlusterOutput lbp_multicluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const std::vector<hypothesis::Hypotheses> &prior_hypotheses_per_cluster, size_t max_num_iters)
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
            size_t bethe_iter = 0;
            size_t msg_norm_iter = 0;

            Eigen::ArrayXXd w_times_msg(n, m);

            double prev_b = bethe_pseudodual_loglikelihood(
                mu,
                nu,
                rho,
                sigma,
                w_nmd,
                w_0,
                cluster_data);

            Eigen::ArrayXXd prev_nu = nu;

            double b;

            double tol_b = 1e-7;
            double tol_msg = 1e-5;
            double err_bethe = std::numeric_limits<double>::infinity();
            double err_msg = std::numeric_limits<double>::infinity();

            bool bethe_converged = false;
            bool msg_norm_converged = false;

            double bethe_error_converged = std::numeric_limits<double>::infinity();
            double msg_error_converged = std::numeric_limits<double>::infinity();

            while (iter < max_num_iters && !(bethe_converged && msg_norm_converged))
            {
                w_times_msg = w_nmd * nu;

                mu = w_nmd / ((-w_times_msg).colwise() + (w_times_msg.rowwise().sum() + w_0 + sigma));
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

                b = bethe_pseudodual_loglikelihood(
                    mu,
                    nu,
                    rho,
                    sigma,
                    w_nmd,
                    w_0,
                    cluster_data);
                err_bethe = fabs(b - prev_b);
                err_msg = message_norm(nu, prev_nu);
                prev_nu = nu;
                prev_b = b;

                if (err_bethe <= tol_b && !bethe_converged) {
                    bethe_converged = true;
                    bethe_error_converged = err_bethe;
                } else {
                    bethe_iter += 1;
                }
                if (err_msg <= tol_msg && !msg_norm_converged) {
                    msg_norm_converged = true;
                    msg_error_converged = err_msg;
                } else {
                    msg_norm_iter += 1;
                }
            }

            // We failed to converge, use errors at time of termination
            if (!bethe_converged) {
                bethe_error_converged = err_bethe;
            }
            if (!msg_norm_converged) {
                msg_error_converged = err_msg;
            }

            return MHLBPMulticlusterOutput(
                std::move(mu),
                std::move(nu),
                std::move(rho),
                std::move(sigma),
                std::move(w_nmd),
                std::move(w_0),
                std::move(cluster_data),
                MHLBPMulticlusterConvergenceResults{
                    // size_t total_number_iterations,
                    iter,
                    // size_t bethe_pseudodual_iterations,
                    bethe_iter,
                    // double bethe_pseudodual_error,
                    bethe_error_converged,
                    // const double bethe_pseudodual_tol,
                    tol_b,
                    // size_t msg_norm_iterations,
                    msg_norm_iter,
                    // double msg_norm_error,
                    msg_error_converged,
                    // const double msg_norm_tol
                    tol_msg
                }
            );
        }

    } // namespace lbp
} // namespace dfg_da