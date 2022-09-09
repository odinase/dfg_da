#include "discrete_factor_graph/factor_graph.h"
#include "discrete_factor_graph/nodes.h"
#include <ranges>

Message::Message(const gtsam::DecisionTreeFactor& m) {
    assert(m.size() == 1);
    cardinality_ = m.discreteKeys()[0].second;
    m_ = m;
}


Message FactorGraph::Node::belief() const {
    Message m;
    for (const auto& n : neighbors_) {
        m *= n->incoming_message(n);
    }

    return m;
}


const Message& FactorGraph::Node::incoming_message(const Node::shared_ptr& node) const {
    auto message_iter = std::find(neighbors_.begin(), neighbors_.end(), node);
    // We don't accept input that is not a neighbor
    assert(message_iter != neighbors_.end());
    
    size_t message_index = std::distance(neighbors_.begin(), message_iter);
    return outgoing_messages_[message_index];
}


std::vector<Node::shared_ptr> FactorGraph::Node::neighbors() const { 
    std::vector<Node::shared_ptr> v;
    for (auto& p : outgoing_messages_) {
        v.push_back(p.first);
    }

    return v;
}



FactorGraph::FactorGraph(const gtsam::DiscreteFactorGraph &dfg)
{
    std::unordered_map<gtsam::Key, Variable::shared_ptr> seen_vars;

    for (auto &&fac : dfg)
    {
        // All factors in the dfg should be castable to DecisionTreeFactor, if not, something is seriously wrong
        if (gtsam::DecisionTreeFactor::shared_ptr df = boost::dynamic_pointer_cast<gtsam::DecisionTreeFactor>(fac))
        {
            Factor::shared_ptr factor = std::make_shared<Factor>(df);
            add_node(factor);

            for (const auto &dk : df->discreteKeys())
            {
                Variable::shared_ptr var;
                // Add variable if not seen before
                if (seen_vars.find(dk.first) == seen_vars.end())
                {
                    var = std::make_shared<Variable>(dk);
                    add_node(var);
                    seen_vars.insert({dk.first, var});
                }
                else
                {
                    var = seen_vars[dk.first];
                }

                var->add_neighbor(factor);
                factor->add_neighbor(var);
            }
        }
    }

    // Initialize for LBP
    for (const auto& node : nodes_) {
        node->init_messages();
    }
}

Marginals FactorGraph::lbp()
{

}

void FactorGraph::add_node(FactorGraph::Node::shared_ptr node) {}