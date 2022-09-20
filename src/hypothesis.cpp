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

std::vector<std::vector<size_t>> hypothesis_enumeration(const Eigen::MatrixXd &reward_matrix, const Hypothesis& prior_hypothesis)
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
    std::vector<std::vector<size_t>>& hypotheses,
    std::vector<size_t> &parent_hypothesis,
    const Hypothesis& prior_hypothesis,
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
    for (const auto& track : gated_tracks_.at(j)) {
        // Track is not claimed yet if it is not contained in parent hypothesis
        if (std::find(parent_hypothesis.begin(), parent_hypothesis.end(), track) == parent_hypothesis.end() && prior_hypothesis.contains(track)) {
            parent_hypothesis.push_back(track);
            traverse_hypothesis_tree(hypotheses, parent_hypothesis, prior_hypothesis, gated_tracks_, j + 1, M);
            parent_hypothesis.pop_back();
        }
    }
}



std::unordered_map<size_t, std::vector<double>> association_marginal_posteriors(const Hypotheses& prior_hypotheses, const Eigen::MatrixXd& reward_matrix,
const double PD,
const double clutter_intensity,
const double arrival_intensity
)
    {

    const size_t N = reward_matrix.rows();
    const size_t M = reward_matrix.cols() - N;

    std::vector<std::pair<std::vector<size_t>, double>> log_joint_distribution; // List over pairs with associations and corresponding probability

    // These are inside the reward matrix
    // double log_intensity = log(clutter_intensity + PD*arrival_intensity);
    // double log_PD = log(PD);
    // double log_PND = log(1.0 - PD);

    // For each prior hypothesis, find all valid posterior hypotheses
    for (auto prior_hypothesis_iter = prior_hypotheses.cbegin(); prior_hypothesis_iter != prior_hypotheses.cend(); ++prior_hypothesis_iter) {
        std::vector<std::vector<size_t>> conditional_posterior_hypotheses = hypothesis_enumeration(reward_matrix, *prior_hypothesis_iter);
        double log_prior_prob = prior_hypothesis_iter->log_prob();
        double log_prob = log_prior_prob;
        for (const std::vector<size_t>& cond_posterior_hypothesis : conditional_posterior_hypotheses) {
            std::vector<size_t> to_cond_posterior_hypothesis = mo_to_to_hypothesis(cond_posterior_hypothesis, N);
            for (size_t i = 0, t = 1; i < to_cond_posterior_hypothesis.size(); i++, t++) {
                // Unassociated tracks
                if (to_cond_posterior_hypothesis[i] == 0) {
                    log_prob += reward_matrix(i, M + i);
                } else {
                    // Index of associated measurement
                    size_t j = to_cond_posterior_hypothesis[i] - 1;
                    log_prob += reward_matrix(i, j);
                }
            }
            log_joint_distribution.push_back({to_cond_posterior_hypothesis, log_prob});
        }
    }
    // Then, use the probability 
}

std::vector<size_t> mo_to_to_hypothesis(const std::vector<size_t>& mo_hypothesis, const size_t num_tracks) {
    std::vector<size_t> to_hypothesis(num_tracks);

    for (size_t i = 0; i < num_tracks; i++) {
        const auto asso_iter = std::find(mo_hypothesis.begin(), mo_hypothesis.end(), i + 1);
        // Unassociated track => misdetection
        if (asso_iter == mo_hypothesis.end()) {
            to_hypothesis.push_back(0);
        } else {
            size_t meas_idx = std::distance(mo_hypothesis.begin(), asso_iter) + 1;
            to_hypothesis.push_back(meas_idx);
        }
    }

    return to_hypothesis;
}