#pragma once

#include <vector>
#include <cmath>
#include <iterator>
#include <Eigen/Core>

class Hypotheses;

using Track = size_t;

struct Hypothesis
{
    friend class Hypotheses;

private:
    double log_prob_;
    std::vector<Track> tracks_;

public:
    Hypothesis(std::vector<Track> &&tracks, double log_prob) : log_prob_(log_prob), tracks_(tracks)
    {
        for (const auto t : tracks_)
        {
            assert(t > 0); // We don't accept tracks that use ID 0 as this is reserved for misdetection
        }
    }

    inline bool contains(const Track track_id) const { return std::find(tracks_.begin(), tracks_.end(), track_id) != tracks_.end(); }
    inline double probability() const { return exp(log_prob_); }
};

class Hypotheses
{
private:
    std::vector<Hypothesis> hypos_;
    void log_normalize();
    std::vector<double> log_probs() const;

public:
    explicit Hypotheses(std::vector<Hypothesis> &&hypos);

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

    auto begin() { return hypos_.begin(); }
    auto end() { return hypos_.end(); }

    auto cbegin() const { return hypos_.cbegin(); }
    auto cend() const { return hypos_.cend(); }
};