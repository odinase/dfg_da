#pragma once

#include <vector>
#include <set>
#include <cmath>
#include <iterator>
#include <unordered_map>
#include <Eigen/Core>


namespace dfg_da {

namespace hypothesis {


class Hypotheses;

using Track = size_t;

struct Hypothesis
{
    friend class Hypotheses;

private:
    double log_prob_;
    std::vector<Track> tracks_;

public:
    Hypothesis(std::vector<Track> &&tracks, double log_prob) : log_prob_(log_prob), tracks_(std::move(tracks))
    {
        for (const auto t : tracks_)
        {
            assert(t > 0); // We don't accept tracks that use ID 0 as this is reserved for misdetection
        }
    }

    Hypothesis(const std::vector<Track> &tracks, double log_prob) : log_prob_(log_prob), tracks_(tracks)
    {
        for (const auto t : tracks_)
        {
            assert(t > 0); // We don't accept tracks that use ID 0 as this is reserved for misdetection
        }
    }

    inline bool contains(const Track track_id) const { return std::find(tracks_.begin(), tracks_.end(), track_id) != tracks_.end(); }
    inline double log_prob() const { return log_prob_; }
    inline double probability() const { return exp(log_prob()); }
    inline size_t size() const { return tracks_.size(); }
    const std::vector<Track>& tracks() const { return tracks_; }
};

class Hypotheses
{
private:
    std::vector<Hypothesis> hypos_;
    void log_normalize();
    std::vector<double> log_probs() const;

public:
    explicit Hypotheses(std::vector<Hypothesis> &&hypos);
    explicit Hypotheses(const std::vector<Hypothesis> &hypos);

    const Hypothesis &operator[](size_t i) const { return hypos_[i]; }
    inline size_t num_hypotheses() const { return hypos_.size(); }

    template <class TrackIterator>
    Hypotheses hypotheses_containing(TrackIterator tracks_iter_begin, TrackIterator tracks_iter_end) const
    {
        static_assert(std::is_same<typename std::iterator_traits<TrackIterator>::value_type, Track>::value);

        std::vector<Hypothesis> hypos;

        for (TrackIterator track_iter = tracks_iter_begin; track_iter != tracks_iter_end; track_iter++)
        {
            for (const auto &hypo : hypos_)
            {
                if (hypo.contains(*track_iter))
                {
                    hypos.push_back(hypo);
                }
            }
        }

        return Hypotheses(std::move(hypos));
    }
    Hypotheses hypotheses_containing(const Track &track) const;
    std::vector<double> hypothesis_probabilites() const;
    std::set<size_t> tracks() const;

    auto begin() { return hypos_.begin(); }
    auto end() { return hypos_.end(); }

    auto cbegin() const { return hypos_.cbegin(); }
    auto cend() const { return hypos_.cend(); }

    void append(const Hypothesis& new_hypothesis);
};


std::vector<std::vector<size_t>> hypothesis_enumeration(const Eigen::MatrixXd &reward_matrix, const Hypothesis& prior_hypothesis);
void traverse_hypothesis_tree(
    std::vector<std::vector<size_t>>& hypotheses,
    std::vector<size_t> &parent_hypothesis,
    const Hypothesis& prior_hypothesis,
    const std::unordered_map<size_t, std::vector<size_t>> &gated_tracks_,
    size_t j,
    const size_t M
);

Eigen::ArrayXXd association_marginal_posteriors(const Eigen::MatrixXd &reward_matrix, const Hypotheses &prior_hypotheses);

std::vector<size_t> mo_to_to_hypothesis(const std::vector<size_t>& mo_hypothesis, const size_t num_tracks);

double prior_hypothesis_conditional_association_probability(const std::vector<size_t>& to_hypothesis, const Hypothesis& prior_hypothesis, const Eigen::MatrixXd& reward_matrix);

} // namespace hypothesis
} // namespace dfg_da
