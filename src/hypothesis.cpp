#include "discrete_factor_graph/hypothesis.h"
#include "discrete_factor_graph/utils.h"

#include <numeric>
#include <functional>
#include <set>
#include <iostream>
#include <unordered_map>

Hypotheses::Hypotheses(std::vector<Hypothesis> &&hypos) : hypos_(hypos)
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

    std::vector<double> log_probs_normalized = logNormalize(log_probs_);
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
    return expNormalize(log_probs());
}

// Return map over each measurement together with list of tracks gated by measurement
std::unordered_map<size_t, std::vector<size_t>> gated_tracks(const Eigen::MatrixXd &reward_matrix)
{
    std::unordered_map<size_t, std::vector<size_t>> gated_tracks_;

    size_t n = reward_matrix.rows();
    size_t m = reward_matrix.cols() - n;

    if (m > 0)
    {
        for (size_t t = 0, i = 1; t < n; t++, i++)
        {
            for (size_t s = 0, j = 1; s < m; s++, j++)
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
    traverse_hypothesis_tree(hypotheses, parent_hypothesis, prior_hypothesis, gated_tracks_, 1, m);

    return hypotheses;
}

void traverse_hypothesis_tree(
    std::vector<std::vector<size_t>> &hypotheses,
    std::vector<size_t> &parent_hypothesis,
    const Hypothesis &prior_hypothesis,
    const std::unordered_map<size_t, std::vector<size_t>> &gated_tracks_,
    size_t j,
    const size_t M)
{
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
    traverse_hypothesis_tree(hypotheses, parent_hypothesis, prior_hypothesis, gated_tracks_, j + 1, M);
    parent_hypothesis.pop_back();

    // Loop over all tracks that can claim measurements
    for (const auto &track : gated_tracks_.at(j))
    {
        // Track is not claimed yet if it is not contained in parent hypothesis
        if (std::find(parent_hypothesis.begin(), parent_hypothesis.end(), track) == parent_hypothesis.end() && prior_hypothesis.contains(track))
        {
            parent_hypothesis.push_back(track);
            traverse_hypothesis_tree(hypotheses, parent_hypothesis, prior_hypothesis, gated_tracks_, j + 1, M);
            parent_hypothesis.pop_back();
        }
    }
}

Eigen::MatrixXd association_marginal_posteriors(const Hypotheses &prior_hypotheses, const Eigen::MatrixXd &reward_matrix)
{

    const size_t N = reward_matrix.rows();
    const size_t M = reward_matrix.cols() - N;

    Eigen::MatrixXd association_marginals = Eigen::MatrixXd::Zero(N, M + 2); // misdetection + num measurements + nonexistence
    std::cout << "We have " << N << " x " << M + 2 << "\n";
    const size_t misdetection_idx = 0;
    const size_t nonexistence_idx = M + 1;

    // For each prior hypothesis, find all valid posterior hypotheses
    for (auto prior_hypothesis_iter = prior_hypotheses.cbegin(); prior_hypothesis_iter != prior_hypotheses.cend(); ++prior_hypothesis_iter)
    {
        std::cout << "Looking at prior hypothesis\n";
        for (const auto track : prior_hypothesis_iter->tracks()) {
            std::cout << track << " ";
        }
        std::cout << "\n";
        // Compute new posterior hypotheses conditioned on the prior hypothesis
        std::vector<std::vector<size_t>> conditional_posterior_hypotheses = hypothesis_enumeration(reward_matrix, *prior_hypothesis_iter);
        double log_prior_prob = prior_hypothesis_iter->log_prob();
        for (const std::vector<size_t> &cond_posterior_hypothesis : conditional_posterior_hypotheses)
        {
            std::cout << "conditional posterior hypothesis:\n";
            for (size_t i = 0, j = 1; i < cond_posterior_hypothesis.size(); i++, j++) {
                std::cout << "Measurement " << j << " associated to track " << cond_posterior_hypothesis[i] << "\n";
            }
            // Convert hypothesis to be over all tracks we know of
            std::vector<size_t> to_cond_posterior_hypothesis = mo_to_to_hypothesis(cond_posterior_hypothesis, N);

            // Compute unnormalized probability for prior hypothesis conditional
            double log_p = prior_hypothesis_conditional_association_probability(to_cond_posterior_hypothesis, *prior_hypothesis_iter, reward_matrix);
            for (size_t i = 0, t = 1; i < to_cond_posterior_hypothesis.size(); i++, t++)
            {
                // Track does not exist
                if (!prior_hypothesis_iter->contains(t))
                {
                    std::cout << "NONEXISTENCE: Accessing index " << i << ", " << nonexistence_idx << "\n"; 
                    association_marginals(i, nonexistence_idx) += log_prior_prob; // log(1.0 * prior_prob) = log_prior_prob
                }
                else
                {
                    if (to_cond_posterior_hypothesis[i] == misdetection_idx)
                    {
                        std::cout << "MISDETECTION: Accessing index " << i << ", " << misdetection_idx << "\n"; 
                        association_marginals(i, misdetection_idx) += log_p + log_prior_prob;
                    }
                    else {
                        std::cout << "DETECTION: Accessing index " << i << ", " << to_cond_posterior_hypothesis[i] << "\n"; 
                        association_marginals(i, to_cond_posterior_hypothesis[i]) += log_p + log_prior_prob;
                    }
                }
            }
        }
    }

    Eigen::VectorXd normalizing_constant = logsumexp(association_marginals);
    association_marginals = (association_marginals.colwise() - normalizing_constant).array().exp();

    return association_marginals;
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
            log_prob += reward_matrix(i, to_hypothesis[i]);
        }
        // Misdetection
        else
        {
            log_prob += reward_matrix(i, M + i);
        }
    }

    return log_prob;
}