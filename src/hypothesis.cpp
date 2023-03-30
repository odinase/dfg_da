#include "dfg_da/hypothesis.h"
#include "dfg_da/utils.h"

#include <numeric>
#include <functional>
#include <algorithm>
#include <set>
#include <iostream>
#include <unordered_map>
#include <stack>
#include <bitset>

namespace dfg_da
{

    namespace hypothesis
    {
        void Hypothesis::reindex_tracks(const std::map<size_t, size_t>& old2new_idx) {
            for (auto& track : tracks_) {
                track = old2new_idx.at(track);
            }
        }

        // Combining two hypotheses means to concatenate the tracks existing and adding the log probabilities together
        Hypothesis Hypothesis::combine(const Hypothesis &h_rhs) const
        {
            Hypothesis new_h(h_rhs);
            new_h.tracks_.insert(new_h.tracks_.end(), this->tracks_.begin(), this->tracks_.end());
            std::sort(new_h.tracks_.begin(), new_h.tracks_.end());
            new_h.log_prob_ += this->log_prob_;
            return new_h;
        }

        Hypotheses::Hypotheses(const std::vector<Hypothesis> &hypos) : hypos_(hypos)
        {
            log_normalize();
        }

        Hypotheses::Hypotheses(std::vector<Hypothesis> &&hypos) : hypos_(std::move(hypos))
        {
            log_normalize();
        }

        std::vector<double> Hypotheses::log_probs() const
        {
            std::vector<double> log_probs_;
            for (const auto &hypo : hypos_)
            {
                log_probs_.push_back(hypo.log_prob_);
            }

            return log_probs_;
        }

        void Hypotheses::log_normalize()
        {
            std::vector<double> log_probs_ = log_probs();

            std::vector<double> log_probs_normalized = utils::logNormalize(log_probs_);
            for (size_t i = 0; i < hypos_.size(); i++)
            {
                hypos_[i].log_prob_ = log_probs_normalized[i];
            }
        }

        Hypotheses Hypotheses::hypotheses_containing(const Track &track) const
        {
            std::vector<Hypothesis> hypos;

            for (const auto &hypo : hypos_)
            {
                if (hypo.contains(track))
                {
                    hypos.push_back(hypo);
                }
            }

            return Hypotheses(std::move(hypos));
        }

        std::vector<double> Hypotheses::hypothesis_probabilites() const
        {
            return utils::expNormalize(log_probs());
        }

        std::set<size_t> Hypotheses::tracks() const
        {
            std::set<size_t> all_tracks;
            for (const auto &h : hypos_)
            {
                all_tracks.insert(h.tracks_.begin(), h.tracks_.end());
            }
            return all_tracks;
        }

        void Hypotheses::reindex_tracks(const std::map<size_t, size_t>& old2new_idx) {
            for (auto& ph : hypos_) {
                ph.reindex_tracks(old2new_idx);
            }
        }

        // Return map over each measurement together with list of tracks gated by measurement
        std::unordered_map<size_t, std::vector<size_t>> gated_tracks(const Eigen::MatrixXd &reward_matrix)
        {
            std::unordered_map<size_t, std::vector<size_t>> gated_tracks_;

            size_t n = reward_matrix.rows();
            size_t m = reward_matrix.cols() - n;

            if (m > 0)
            {
                for (size_t s = 0, j = 1; s < m; s++, j++)
                {
                    // We need to initialize with empty vector in case we have a measurement gated by no tracks
                    gated_tracks_[j] = {};
                    for (size_t t = 0, i = 1; t < n; t++, i++)
                    {
                        if (std::isfinite(reward_matrix(t, s)))
                        {
                            gated_tracks_[j].push_back(i);
                        }
                    }
                }
            }

            return gated_tracks_;
        }

        std::vector<std::vector<size_t>> hypothesis_enumeration(const Eigen::MatrixXd &reward_matrix, const Hypothesis &prior_hypothesis)
        {
            size_t n = reward_matrix.rows();
            size_t m = reward_matrix.cols() - n;

            std::unordered_map<size_t, std::vector<size_t>> gated_tracks_ = gated_tracks(reward_matrix);
            assert(gated_tracks_.size() == m);

            std::vector<std::vector<size_t>> hypotheses;
            std::vector<size_t> parent_hypothesis;
            traverse_hypothesis_tree_recursive(hypotheses, parent_hypothesis, prior_hypothesis, gated_tracks_, 1, m);
            // traverse_hypothesis_tree(hypotheses, prior_hypothesis, gated_tracks_, m);

            //     void traverse_hypothesis_tree(
            // std::vector<std::vector<size_t>> &hypotheses,
            // const Hypothesis &prior_hypothesis,
            // const std::unordered_map<size_t, std::vector<size_t>> &gated_tracks_,
            // size_t M)

            return hypotheses;
        }

        void traverse_hypothesis_tree_recursive(
            std::vector<std::vector<size_t>> &hypotheses,
            std::vector<size_t> &parent_hypothesis,
            const Hypothesis &prior_hypothesis,
            const std::unordered_map<size_t, std::vector<size_t>> &gated_tracks_,
            size_t j,
            const size_t M)
        {
            if (hypotheses.size() > 1e9)
            {
                throw std::invalid_argument("Too many hypotheses, exceeds 1e9");
            }
            // We are currently considering measurement j \in {1, ..., M}.
            // If we have considered all measurements, return
            if (j > M)
            {
                assert(parent_hypothesis.size() == M);
                hypotheses.push_back(parent_hypothesis);
                return;
            }

            // First consider misdetection
            parent_hypothesis.push_back(0);
            traverse_hypothesis_tree_recursive(hypotheses, parent_hypothesis, prior_hypothesis, gated_tracks_, j + 1, M);
            parent_hypothesis.pop_back();

            // Loop over all tracks that can claim measurements
            for (const auto &track : gated_tracks_.at(j))
            {
                // Track is not claimed yet if it is not contained in parent hypothesis
                if (std::find(parent_hypothesis.begin(), parent_hypothesis.end(), track) == parent_hypothesis.end() && prior_hypothesis.contains(track))
                {
                    parent_hypothesis.push_back(track);
                    traverse_hypothesis_tree_recursive(hypotheses, parent_hypothesis, prior_hypothesis, gated_tracks_, j + 1, M);
                    parent_hypothesis.pop_back();
                }
            }
        }

        void traverse_hypothesis_tree(
            std::vector<std::vector<size_t>> &hypotheses,
            const Hypothesis &prior_hypothesis,
            const std::unordered_map<size_t, std::vector<size_t>> &gated_tracks_,
            size_t M)
        {
            std::stack<std::tuple<std::bitset<100>, size_t>> stack;
            stack.push(std::make_tuple(std::bitset<100>(), 1));

            while (!stack.empty())
            {
                auto [parent_hypothesis, j] = stack.top();
                stack.pop();

                if (hypotheses.size() > 1e9)
                {
                    throw std::invalid_argument("Too many hypotheses, exceeds 1e9");
                }

                if (j > M)
                {
                    std::vector<size_t> hypothesis;
                    for (size_t i = 0; i < M; ++i)
                    {
                        if (parent_hypothesis.test(i))
                        {
                            hypothesis.push_back(i + 1);
                        }
                    }
                    hypotheses.push_back(hypothesis);
                    continue;
                }

                // First consider misdetection
                auto md_parent_hypothesis = parent_hypothesis;
                md_parent_hypothesis.reset(j - 1);
                stack.push(std::make_tuple(md_parent_hypothesis, j + 1));

                // Loop over all tracks that can claim measurements
                for (const auto &track : gated_tracks_.at(j))
                {
                    // Track is not claimed yet if it is not contained in parent hypothesis
                    if (!parent_hypothesis.test(track - 1) && prior_hypothesis.contains(track))
                    {
                        auto claimed_parent_hypothesis = parent_hypothesis;
                        claimed_parent_hypothesis.set(track - 1);
                        stack.push(std::make_tuple(claimed_parent_hypothesis, j + 1));
                    }
                }
            }
        }

        Eigen::ArrayXXd association_marginal_posteriors(const Eigen::MatrixXd &reward_matrix, const Hypotheses &prior_hypotheses)
        {

            const size_t N = reward_matrix.rows();
            const size_t M = reward_matrix.cols() - N;

            Eigen::ArrayXXd association_marginals = Eigen::ArrayXXd::Zero(M + 2, N); // misdetection + num measurements + nonexistence
            const size_t nonexistence_idx = M + 1;

            // For each prior hypothesis, find all valid posterior hypotheses
            for (auto prior_hypothesis_iter = prior_hypotheses.cbegin(); prior_hypothesis_iter != prior_hypotheses.cend(); ++prior_hypothesis_iter)
            {
                // Compute new posterior hypotheses conditioned on the prior hypothesis
                std::vector<std::vector<size_t>> conditional_posterior_hypotheses = hypothesis_enumeration(reward_matrix, *prior_hypothesis_iter);
                double log_prior_prob = prior_hypothesis_iter->log_prob();

                for (const std::vector<size_t> &cond_posterior_hypothesis : conditional_posterior_hypotheses)
                {
                    // Convert hypothesis to be over all tracks we know of
                    std::vector<size_t> to_cond_posterior_hypothesis = mo_to_to_hypothesis(cond_posterior_hypothesis, N);

                    // Compute unnormalized probability for prior hypothesis conditional
                    double log_p = prior_hypothesis_conditional_association_probability(to_cond_posterior_hypothesis, *prior_hypothesis_iter, reward_matrix);
                    for (size_t i = 0, t = 1; i < to_cond_posterior_hypothesis.size(); i++, t++)
                    {
                        size_t idx = prior_hypothesis_iter->contains(t) ? to_cond_posterior_hypothesis[i] : nonexistence_idx;
                        association_marginals(idx, i) += exp(log_p + log_prior_prob);
                    }
                }
            }

            association_marginals.rowwise() /= association_marginals.colwise().sum();

            return association_marginals;
        }

        std::tuple<Eigen::ArrayXXd, double> association_marginal_posteriors_normalization_constant(const Eigen::MatrixXd &reward_matrix, const Hypotheses &prior_hypotheses)
        {

            const size_t N = reward_matrix.rows();
            const size_t M = reward_matrix.cols() - N;

            Eigen::ArrayXXd association_marginals = Eigen::ArrayXXd::Zero(M + 2, N); // misdetection + num measurements + nonexistence
            const size_t nonexistence_idx = M + 1;

            double Z = 0.0;
            // For each prior hypothesis, find all valid posterior hypotheses
            for (auto prior_hypothesis_iter = prior_hypotheses.cbegin(); prior_hypothesis_iter != prior_hypotheses.cend(); ++prior_hypothesis_iter)
            {
                // Compute new posterior hypotheses conditioned on the prior hypothesis
                std::vector<std::vector<size_t>> conditional_posterior_hypotheses = hypothesis_enumeration(reward_matrix, *prior_hypothesis_iter);
                double log_prior_prob = prior_hypothesis_iter->log_prob();

                for (const std::vector<size_t> &cond_posterior_hypothesis : conditional_posterior_hypotheses)
                {
                    double log_Z = 0.0;
                    // Convert hypothesis to be over all tracks we know of
                    std::vector<size_t> to_cond_posterior_hypothesis = mo_to_to_hypothesis(cond_posterior_hypothesis, N);

                    // Compute unnormalized probability for prior hypothesis conditional
                    double log_p = prior_hypothesis_conditional_association_probability(to_cond_posterior_hypothesis, *prior_hypothesis_iter, reward_matrix);
                    for (size_t i = 0, t = 1; i < to_cond_posterior_hypothesis.size(); i++, t++)
                    {
                        size_t idx = prior_hypothesis_iter->contains(t) ? to_cond_posterior_hypothesis[i] : nonexistence_idx;
                        association_marginals(idx, i) += exp(log_p + log_prior_prob);
                    }
                    log_Z += log_p + log_prior_prob;
                    // std::cout << "log_p: " << log_p << "\n";
                    // std::cout << "log_Z: " << log_Z << "\n";
                    Z += exp(log_Z);
                    std::cout << Z << "\n";
                }
                std::cout << "\n";
            }

            association_marginals.rowwise() /= association_marginals.colwise().sum();

            return {association_marginals, Z};
        }


        // We assume multicluster since
        std::tuple<Eigen::ArrayXXd, double> association_marginal_posteriors_normalization_constant_multicluster(const Eigen::MatrixXd &reward_matrix, const std::vector<Hypotheses> &prior_hypotheses_per_cluster_posterior)
        {

            const size_t N = reward_matrix.rows();
            const size_t M = reward_matrix.cols() - N;

            Eigen::ArrayXXd association_marginals = Eigen::ArrayXXd::Zero(M + 2, N); // misdetection + num measurements + nonexistence
            const size_t nonexistence_idx = M + 1;

            // We need the total multicluster normalization constant, which should be just the product of normalization constants for each cluster.
            double Z_tot = 1.0;
            // Loop over each cluster
            for (const auto &prior_hypotheses : prior_hypotheses_per_cluster_posterior)
            {
                std::set<size_t> tracks = prior_hypotheses.tracks();
                double Z_cluster = 0.0;
                // For each prior hypothesis, find all valid posterior hypotheses
                for (auto prior_hypothesis_iter = prior_hypotheses.cbegin(); prior_hypothesis_iter != prior_hypotheses.cend(); ++prior_hypothesis_iter)
                {
                    // Compute new posterior hypotheses conditioned on the prior hypothesis
                    std::vector<std::vector<size_t>> conditional_posterior_hypotheses = hypothesis_enumeration(reward_matrix, *prior_hypothesis_iter);
                    double log_prior_prob = prior_hypothesis_iter->log_prob();

                    for (const std::vector<size_t> &cond_posterior_hypothesis : conditional_posterior_hypotheses)
                    {
                        double log_Z = 0.0;
                        // Convert hypothesis to be over all tracks we know of
                        std::vector<size_t> to_cond_posterior_hypothesis = mo_to_to_hypothesis(cond_posterior_hypothesis, N);

                        // Compute unnormalized probability for prior hypothesis conditional
                        double log_p = prior_hypothesis_conditional_association_probability(to_cond_posterior_hypothesis, *prior_hypothesis_iter, reward_matrix);
                        for (size_t i = 0, t = 1; i < to_cond_posterior_hypothesis.size(); i++, t++)
                        {
                            // The track must exist in the cluster.
                            if (std::find(tracks.begin(), tracks.end(), t) != tracks.end())
                            {
                                size_t idx = prior_hypothesis_iter->contains(t) ? to_cond_posterior_hypothesis[i] : nonexistence_idx;
                                association_marginals(idx, i) += exp(log_p + log_prior_prob);
                            }
                        }
                        log_Z += log_p + log_prior_prob;
                        std::cout << log_p << ", " << log_prior_prob << "\n";
                        Z_cluster += exp(log_Z);
                    }
                }
                std::cout << Z_cluster << "\n";
                Z_tot *= Z_cluster;
            }

            association_marginals.rowwise() /= association_marginals.colwise().sum();

            return {association_marginals, Z_tot};
        }


        // We assume multicluster since
        std::tuple<Eigen::ArrayXXd, double> association_marginal_posteriors_normalization_constant_multicluster_efficient(const Eigen::MatrixXd &reward_matrix, const std::vector<Hypotheses> &prior_hypotheses_per_cluster_posterior)
        {

            return {Eigen::ArrayXd(), 0.0};

            const size_t N = reward_matrix.rows();
            const size_t M = reward_matrix.cols() - N;

            Eigen::ArrayXXd association_marginals = Eigen::ArrayXXd::Zero(M + 2, N); // misdetection + num measurements + nonexistence
            const size_t nonexistence_idx = M + 1;

            // Make convenience variable for later
            std::unordered_map<size_t, std::vector<size_t>> gated_tracks_ = gated_tracks(reward_matrix);
            std::vector<size_t> hypothesis_branch;
            hypothesis_branch.reserve(N);

            // We need the total multicluster normalization constant, which should be just the product of normalization constants for each cluster.
            double Z_tot = 1.0;
            // Loop over each cluster
            for (const auto &prior_hypotheses : prior_hypotheses_per_cluster_posterior)
            {
                // Set over all tracks existing in the cluster. We should only compute marginals for these
                std::set<size_t> tracks_set = prior_hypotheses.tracks();
                std::vector<size_t> tracks(tracks_set.begin(), tracks_set.end());
                size_t i = 0;

                double Z_cluster = 0.0;

                // For each prior hypothesis, find all valid posterior hypotheses
                for (auto prior_hypothesis_iter = prior_hypotheses.cbegin(); prior_hypothesis_iter != prior_hypotheses.cend(); ++prior_hypothesis_iter)
                {
                    std::stack<size_t> tracks_stack;
                    // We use index 0 here, but will map these track indices with the vector above
                    tracks_stack.push(0);

                    while (!tracks_stack.empty())
                    {
                        size_t track = tracks_stack.top();
                        tracks_stack.pop();

                        // Associate track with misdetection
                        hypothesis_branch.push_back(0);
                        tracks_stack.push(track + 1);
                    }

                    double log_prior_prob = prior_hypothesis_iter->log_prob();

                    // for (const std::vector<size_t> &cond_posterior_hypothesis : conditional_posterior_hypotheses)
                    // {
                    //     double log_Z = 0.0;
                    //     // Convert hypothesis to be over all tracks we know of
                    //     std::vector<size_t> to_cond_posterior_hypothesis = mo_to_to_hypothesis(cond_posterior_hypothesis, N);

                    //     // Compute unnormalized probability for prior hypothesis conditional
                    //     double log_p = prior_hypothesis_conditional_association_probability(to_cond_posterior_hypothesis, *prior_hypothesis_iter, reward_matrix);
                    //     for (size_t i = 0, t = 1; i < to_cond_posterior_hypothesis.size(); i++, t++)
                    //     {
                    //         // The track must exist in the cluster.
                    //         if (std::find(tracks.begin(), tracks.end(), t) != tracks.end())
                    //         {
                    //             size_t idx = prior_hypothesis_iter->contains(t) ? to_cond_posterior_hypothesis[i] : nonexistence_idx;
                    //             association_marginals(idx, i) += exp(log_p + log_prior_prob);
                    //         }
                    //     }
                    //     log_Z += log_p + log_prior_prob;
                    //     Z_cluster += exp(log_Z);
                    // }
                }
                Z_tot *= Z_cluster;
            }

            association_marginals.rowwise() /= association_marginals.colwise().sum();

            return {association_marginals, Z_tot};
        }

        std::vector<size_t> mo_to_to_hypothesis(const std::vector<size_t> &mo_hypothesis, const size_t num_tracks)
        {
            std::vector<size_t> to_hypothesis;

            for (size_t i = 0; i < num_tracks; i++)
            {
                const auto asso_iter = std::find(mo_hypothesis.begin(), mo_hypothesis.end(), i + 1);
                // Unassociated track => misdetection
                if (asso_iter == mo_hypothesis.end())
                {
                    to_hypothesis.push_back(0);
                }
                else
                {
                    size_t meas_idx = std::distance(mo_hypothesis.begin(), asso_iter) + 1;
                    to_hypothesis.push_back(meas_idx);
                }
            }

            return to_hypothesis;
        }

        double prior_hypothesis_conditional_association_probability(const std::vector<size_t> &to_hypothesis, const Hypothesis &prior_hypothesis, const Eigen::MatrixXd &reward_matrix)
        {
            double log_prob = 0.0;
            size_t N = reward_matrix.rows();
            size_t M = reward_matrix.cols() - N;

            for (size_t i = 0, track = 1; i < to_hypothesis.size(); i++, track++)
            {
                // If track does not exist in prior hypothesis it does not contribute to the probability
                if (!prior_hypothesis.contains(track))
                {
                    continue;
                }

                // Detection, use likelihood
                if (to_hypothesis[i] > 0)
                {
                    log_prob += reward_matrix(i, to_hypothesis[i] - 1);
                }
                // Misdetection
                else
                {
                    log_prob += reward_matrix(i, M + i);
                }
            }

            return log_prob;
        }

        void Hypotheses::append(const Hypothesis &new_hypothesis)
        {
            hypos_.push_back(new_hypothesis);
            log_normalize();
        }

        Hypotheses Hypotheses::combine(const Hypotheses &h_rhs) const
        {
            std::vector<Hypothesis> hh;
            for (const Hypothesis &h1 : hypos_)
            {
                for (const Hypothesis &h2 : h_rhs.hypos_)
                {
                    hh.push_back(h1.combine(h2));
                }
            }
            return Hypotheses(std::move(hh));
        }
    } // namespace hypothesis
} // namespace dfg_da
