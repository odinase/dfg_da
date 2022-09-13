#pragma once

#include <gtsam/base/Vector.h>
#include <math.h>

#include <limits>
#include <string>
#include <vector>

// Odin: Shamelessly stolen from https://github.com/MarineRoboticsGroup/dcsam/blob/main/include/dcsam/DCSAM_utils.h

inline std::vector<double> logNormalize(const std::vector<double> &logProbs)
{
    std::vector<double> cleanLogProbs;
    double maxLogProb = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < logProbs.size(); i++)
    {
        double logProb = (!std::isnan(logProbs[i]))
                             ? logProbs[i]
                             : -std::numeric_limits<double>::infinity();
        if ((logProb != std::numeric_limits<double>::infinity()) &&
            logProb > maxLogProb)
        {
            maxLogProb = logProb;
        }
        cleanLogProbs.push_back(logProb);
    }

    // After computing the max = "Z" of the log probabilities L_i, we compute
    // the log of the normalizing constant, log S, where S = sum_j exp(L_j - Z).
    double total = 0.0;
    for (size_t i = 0; i < cleanLogProbs.size(); i++)
    {
        double probPrime = exp(cleanLogProbs[i] - maxLogProb);
        total += probPrime;
    }
    double logTotal = log(total);

    // Now we compute the (normalized) probability (for each i):
    // p_i = exp(L_i - Z - log S)
    double checkNormalization = 0.0;
    std::vector<double> log_probs_normalized;
    for (size_t i = 0; i < cleanLogProbs.size(); i++)
    {
        double log_prob = cleanLogProbs[i] - maxLogProb - logTotal;
        log_probs_normalized.push_back(log_prob);
        checkNormalization += exp(log_prob);
    }
    // Numerical tolerance for floating point comparisons
    double tol = 1e-9;

    if (!gtsam::fpEqual(checkNormalization, 1.0, tol))
    {
        std::string errMsg =
            std::string("expNormalize failed to normalize probabilities. ") +
            std::string("Expected normalization constant = 1.0. Got value: ") +
            std::to_string(checkNormalization) +
            std::string(
                "\n This could have resulted from numerical overflow/underflow.");
        throw std::logic_error(errMsg);
    }

    return log_probs_normalized;
}

inline std::vector<double> expNormalize(const std::vector<double> &logProbs)
{
    std::vector<double> log_probs_normalized = logNormalize(logProbs);

    std::vector<double> probs;

    for (const auto &log_prob : log_probs_normalized)
    {
        probs.push_back(exp(log_prob));
    }

    return probs;
}
