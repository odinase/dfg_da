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

std::vector<std::vector<size_t>> hypothesis_enumeration(const Eigen::MatrixXd &reward_matrix)
{
    size_t n = reward_matrix.rows();
    size_t m = reward_matrix.cols() - n;

    std::unordered_map<size_t, std::vector<size_t>> gated_tracks_ = gated_tracks(reward_matrix);
    assert(gated_tracks_.size() == m);

    std::vector<std::vector<size_t>> hypotheses;
    std::vector<size_t> parent_hypothesis;
    traverse_hypothesis_tree(hypotheses, parent_hypothesis, gated_tracks_, 1, m);

    return hypotheses;
}

void traverse_hypothesis_tree(
    std::vector<std::vector<size_t>>& hypotheses,
    std::vector<size_t> &parent_hypothesis,
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
    traverse_hypothesis_tree(hypotheses, parent_hypothesis, gated_tracks_, j + 1, M);
    parent_hypothesis.pop_back();

    // Loop over all tracks that can claim measurements
    for (const auto& track : gated_tracks_.at(j)) {
        // Track is not claimed yet if it is not contained in parent hypothesis
        if (std::find(parent_hypothesis.begin(), parent_hypothesis.end(), track) == parent_hypothesis.end()) {
            parent_hypothesis.push_back(track);
            traverse_hypothesis_tree(hypotheses, parent_hypothesis, gated_tracks_, j + 1, M);
            parent_hypothesis.pop_back();
        }
    }
}
