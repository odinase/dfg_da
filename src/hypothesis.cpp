#include "discrete_factor_graph/hypothesis.h"
#include "discrete_factor_graph/utils.h"

#include <numeric>
#include <functional>
#include <set>

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
    size_t m = reward_matrix.cols() - m;

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

std::vector<std::unordered_map<size_t, size_t>> hypothesis_enumeration(const Eigen::MatrixXd &reward_matrix)
{
    // We wish to traverse the hypothesis tree
    // Make set of all unclaimed measurements

    size_t n = reward_matrix.rows();
    size_t m = reward_matrix.cols() - n;

    std::set<size_t> unclaimed_measurements;
    for (size_t j = 1; j <= m; j++)
    {
        unclaimed_measurements.insert(j);
    }

    std::vector<std::unordered_map<size_t, size_t>> hypotheses;
    std::unordered_map<size_t, std::vector<size_t>> gated_tracks_ = gated_tracks(reward_matrix);

    traverse_hypothesis_tree(hypotheses, unclaimed_measurements, gated_tracks_, 1, m);
}

void traverse_hypothesis_tree(
    std::vector<std::unordered_map<size_t, size_t>> &hypotheses,
    std::set<size_t> &unclaimed_measurements,
    const std::unordered_map<size_t, std::vector<size_t>> &gated_tracks_,
    size_t j,
    const size_t M)
{
    // We are currently considering measurement j \in {1, ..., M}.
    // If we have considered all measurements, return
    if (j > M)
    {
        return;
    }

    // First consider misdetection
    

    // Loop over all tracks that can claim measurements
    for (const auto& track : gated_tracks_[j]) {

    }
}

// std::vector<std::vector<size_t>> hypothesis_enumeration(const Eigen::MatrixXd& reward_matrix) {
//     // We first need containers for the possible associations for each track and also the number of possible associations for each track
//     std::vector<std::vector<size_t>> tracks_possible_associations;
//     std::vector<size_t> track_num_possible_associations;

//     size_t N_tracks = reward_matrix.rows();
//     size_t N_measurements = reward_matrix.cols() - N_tracks;

//     for (size_t i = 0; i < N_tracks; i++) {
//         size_t num_possible_associations = 0;

//         tracks_possible_associations.emplace_back();
//         std::vector<size_t>& track_possible_associations = tracks_possible_associations.back();

//         track_possible_associations.push_back(0); // Misdetection is always possible

//         // Find gated measurements. This would mean sweep all measurements in reward matrix for each track and find the finite elements
//         for (size_t j = 0; j < N_measurements; j++) {
//             double loglikelihood = std::isfinite(reward_matrix(i, j));
//             if (std::isfinite(loglikelihood)) {
//                 track_possible_associations.push_back(j);
//                 num_possible_associations += 1;
//             }
//         }

//         track_num_possible_associations.push_back(num_possible_associations);

//     }
// }