#include "discrete_factor_graph/hypothesis.h"
#include "discrete_factor_graph/utils.h"

Hypotheses::Hypotheses(std::vector<Hypothesis> &&hypos) : hypos_(hypos)
{
    log_normalize();
}

std::vector<double> Hypotheses::log_probs() const {
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

std::vector<double> Hypotheses::hypothesis_probabilites() const {
    return expNormalize(log_probs());
}
