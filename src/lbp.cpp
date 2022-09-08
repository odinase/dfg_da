#include "discrete_factor_graph/lbp.h"
#include <gtsam/inference/Ordering.h>

#include <iostream>


void normalize_message(gtsam::DecisionTreeFactor& message) {
    assert(message.discreteKeys().size() == 1); // There should be only one key in this list  
    message = message / *message.sum(1);
}


std::unordered_map<gtsam::Key, std::vector<double>> lbp(
    const gtsam::DiscreteFactorGraph &dfg,
    const size_t max_iters,
    const double kl_threshold)
{
    // Main difficulity here is that we need keys for factors as well
    // Strictly speaking, computing all messages out and into a factor must necessarily compute all messages we need, and so we shouldn't need to explicitly know what factor we are in(?)

    // Apparently shared_ptrs are hashable, nice
    std::unordered_map<gtsam::DiscreteFactor::shared_ptr, std::unordered_map<gtsam::Key, gtsam::DecisionTreeFactor>> outgoing_messages;
    std::unordered_map<gtsam::DiscreteFactor::shared_ptr, std::unordered_map<gtsam::Key, gtsam::DecisionTreeFactor>> ingoing_messages;

    // Since looping factor is so easy and also gives all adjacent variables,
    // it would be useful to first construct a "neighbor data structure"
    // by looping over all factor, make a map for each variable and append the pointer to the variable as its neighbor.
    // This is probably very useful for later?
    std::unordered_map<gtsam::Key, std::vector<gtsam::DiscreteFactor::shared_ptr>> neighbors;

    // We should also at the same time initialize all messages

    for (auto &&fac : dfg)
    {
        // All factors in the dfg should be castable to DecisionTreeFactor, if not, something is seriously wrong
        gtsam::DecisionTreeFactor df = fac->toDecisionTreeFactor();

        for (const auto& dk : df.discreteKeys())
        {
            neighbors[dk.first].push_back(fac);
            outgoing_messages[fac][dk.first] = gtsam::DecisionTreeFactor{{dk}, std::vector<double>(dk.second, 1.0)};
            ingoing_messages[fac][dk.first] = gtsam::DecisionTreeFactor{{dk}, std::vector<double>(dk.second, 1.0)};
        }
    }

    bool converged = false;
    size_t iter = 0;

    std::unordered_map<gtsam::DiscreteFactor::shared_ptr, std::unordered_map<gtsam::Key, gtsam::DecisionTreeFactor>> prev_outgoing_messages;
    std::unordered_map<gtsam::DiscreteFactor::shared_ptr, std::unordered_map<gtsam::Key, gtsam::DecisionTreeFactor>> prev_ingoing_messages;

    while (!converged && iter < max_iters)
    {
        // Initilize previous iteration of messages
        prev_outgoing_messages = outgoing_messages;
        prev_ingoing_messages = ingoing_messages;

        // Store all messages m_i for this iteration
        std::unordered_map<gtsam::Key, gtsam::DecisionTreeFactor> m_is;

        // Loop over all factors
        for (auto &&df : dfg)
        {
            // Recompute all outgoing messages, a -> i
            // There is really no way around avoiding the double loop, so prepare fa here for loop below
            gtsam::DecisionTreeFactor fa = df->toDecisionTreeFactor();

            gtsam::KeySet key_set;
            for (auto k : df->keys())
            {
                fa = fa * prev_ingoing_messages[df][k];
                key_set.insert(k);
            }

            for (auto k : df->keys())
            {
                // Recompute all incoming messages, i -> a, for each i

                // Compute the m_i for all variables i around df as product of all messages into variable i
                gtsam::DecisionTreeFactor m_i;

                // We first check if we have the message already stored
                if (m_is.find(k) == m_is.end())
                {
                    // Haven't computed it yet, do it now
                    for (auto &f : neighbors[k])
                    {
                        m_i = m_i * prev_outgoing_messages[f][k];
                    }
                    m_is[k] = m_i;
                }
                else
                {
                    // Use stored value
                    m_i = m_is[k];
                }

                // We now have m_i, compute i -> a
                const auto &m_a_i = prev_outgoing_messages[df][k];
                ingoing_messages[df][k] = m_i / m_a_i;
                normalize_message(ingoing_messages[df][k]);

                // Use fa to compute

                // We need to indicate variables to sum out
                key_set.erase(k);
                gtsam::Ordering vars_to_sum(key_set.begin(), key_set.end());

                const auto &m_i_a = prev_ingoing_messages[df][k];
                outgoing_messages[df][k] = *(fa / m_i_a).sum(vars_to_sum);
                normalize_message(outgoing_messages[df][k]);

                key_set.insert(k); // Reinsert key for next iteration
            }
        }

        // TODO(odin): Fix this after testing
        // converged = true;
        iter++;
    }

    // All messages are computed, time to make marginals
    std::unordered_map<gtsam::Key, std::vector<double>> marginals;
    for (auto [k, c] : dfg.discreteKeys())
    {
        gtsam::DecisionTreeFactor mtf;
        for (auto &f : neighbors[k])
        {
            mtf = mtf * outgoing_messages[f][k];
        }

        normalize_message(mtf);
        gtsam::DiscreteDistribution m(mtf);
        marginals[k] = m.pmf();
    }

    return marginals;
}