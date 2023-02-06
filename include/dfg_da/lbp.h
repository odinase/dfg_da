#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include "dfg_da/hypothesis.h"

namespace dfg_da
{

    namespace lbp
    {

        struct MHLBPSingleClusterOutput
        {
            public:
            const Eigen::ArrayXXd mu;
            const Eigen::ArrayXXd nu;
            const Eigen::ArrayXd rho;
            const Eigen::ArrayXd sigma;
            const Eigen::ArrayXXd w_nmd;
            const Eigen::ArrayXd w_0;
            const Eigen::ArrayXXd t2h;
            const Eigen::ArrayXXd t2h_not;
            const Eigen::ArrayXd phi;
            const size_t num_hypotheses;
            const size_t num_tracks;
            const size_t num_measurements;

            MHLBPSingleClusterOutput(
                Eigen::ArrayXd &&mu_,
                Eigen::ArrayXd &&nu_,
                Eigen::ArrayXd &&rho_,
                Eigen::ArrayXd &&sigma_,
                Eigen::ArrayXXd &&w_nmd_,
                Eigen::ArrayXd &&w_0_,
                Eigen::ArrayXXd &&t2h_,
                Eigen::ArrayXXd &&t2h_not_,
                Eigen::ArrayXd &&phi_
            )
            : mu(std::move(mu_)),
                    nu(std::move(nu_)),
                    rho(std::move(rho_)),
                    sigma(std::move(sigma_)),
                    w_nmd(std::move(w_nmd_)),
                    w_0(std::move(w_0_)),
                    t2h(std::move(t2h_)),
                    t2h_not(std::move(t2h_not_)),
                    phi(std::move(phi_)),
                    num_hypotheses(phi.size()),
                    num_tracks(mu.rows()),
                    num_measurements(mu.cols()) {}
            
            Eigen::ArrayXXd track_association_marginals() const;
            Eigen::ArrayXXd measurement_association_marginals() const;
            Eigen::ArrayXd hypotheses_marginal() const;
            double bethe_pseudodual() const;

            private:
            // track_normalizing_constant(self, w_nmd: np.ndarray, nu: np.ndarray)
            Eigen::ArrayXd track_normalization_constant() const;
            // meas_normalizing_constant(self, mu: np.ndarray)
            Eigen::ArrayXd meas_normalization_constant() const;
            // edge_normalizing_constant(self, w_nmd: np.ndarray, mu: np.ndarray, nu: np.ndarray)
            Eigen::ArrayXXd edge_normalization_constant() const;
            double hypotheses_normalization_constant() const;
        };
        MHLBPSingleClusterOutput lbp_single_cluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const hypothesis::Hypotheses &prior_hypotheses, size_t max_num_iters = 300);

        struct ClusterData
        {
            private:
            std::vector<double> phi__;
            public:
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
                for (size_t t = 0; t < nt; t++)
                {
                    t_idx[t] -= 1;
                }
            }
            inline Eigen::Map<const Eigen::ArrayXd> phi() const
            {
                return Eigen::Map<const Eigen::ArrayXd>(phi__.data(), phi__.size());
            }
        };

        struct MHLBPMultilusterOutput
        {
            public:
            const Eigen::ArrayXXd mu;
            const Eigen::ArrayXXd nu;
            const Eigen::ArrayXd rho;
            const Eigen::ArrayXd sigma;
            const Eigen::ArrayXXd w_nmd;
            const Eigen::ArrayXd w_0;
            const std::vector<ClusterData> cluster_data;
            const size_t num_tracks;
            const size_t num_measurements;
            const size_t num_clusters;

            MHLBPMultilusterOutput(
                Eigen::ArrayXXd &&mu_,
                Eigen::ArrayXXd &&nu_,
                Eigen::ArrayXd &&rho_,
                Eigen::ArrayXd &&sigma_,
                Eigen::ArrayXXd &&w_nmd_,
                Eigen::ArrayXd &&w_0_,
                std::vector<ClusterData> &&cluster_data_
            )
            : mu(std::move(mu_)),
                    nu(std::move(nu_)),
                    rho(std::move(rho_)),
                    sigma(std::move(sigma_)),
                    w_nmd(std::move(w_nmd_)),
                    w_0(std::move(w_0_)),
                    cluster_data(std::move(cluster_data_)),
                    num_tracks(mu.rows()),
                    num_measurements(mu.cols()),
                    num_clusters(cluster_data.size()) {}
            
            Eigen::ArrayXXd track_association_marginals() const;
            void track_association_marginals_inplace(double* data) const;
            Eigen::ArrayXXd measurement_association_marginals() const;
            Eigen::ArrayXXd hypotheses_marginals() const;
            double bethe_pseudodual() const;
            private:
                        // track_normalizing_constant(self, w_nmd: np.ndarray, nu: np.ndarray)
            Eigen::ArrayXd track_normalization_constants() const;
            // meas_normalizing_constant(self, mu: np.ndarray)
            Eigen::ArrayXd meas_normalization_constants() const;
            // edge_normalizing_constant(self, w_nmd: np.ndarray, mu: np.ndarray, nu: np.ndarray)
            Eigen::ArrayXXd track_meas_normalization_constants() const;
            Eigen::ArrayXd hypotheses_normalization_constants() const;
            Eigen::ArrayXXd track_hypos_normalization_constants() const;
        };
        MHLBPMultilusterOutput lbp_multicluster(const Eigen::Ref<const Eigen::MatrixXd> &reward_matrix, const std::vector<hypothesis::Hypotheses> &prior_hypotheses_per_cluster, size_t max_num_iters = 300);

    } // namespace lbp
} // namespace dfg_da