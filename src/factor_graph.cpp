#include "discrete_factor_graph/factor_graph.h"
#include "discrete_factor_graph/nodes.h"


Message::Message(const gtsam::DecisionTreeFactor &m)
{
    assert(m.size() == 1);
    cardinality_ = m.discreteKeys()[0].second;
    m_ = m;
}

std::vector<double> Message::pmf() {
    normalize();
    gtsam::DiscreteDistribution m(m_);
    return m.pmf();
}


FactorGraph::Node::~Node() {}

const Message &FactorGraph::Node::incoming_message(const FactorGraph::Node::shared_ptr &node) const
{
    auto node_iter = std::find_if(node->outgoing_messages_.begin(), node->outgoing_messages_.end(), [this](const auto &n)
                                  { return n.first.get() == this; });

    // // We don't accept input that is not a neighbor
    assert(node_iter != outgoing_messages_.end());

    return node_iter->second;
}

std::unordered_map<FactorGraph::Node::shared_ptr, const Message *const> FactorGraph::Node::incoming_messages() const
{
    std::unordered_map<FactorGraph::Node::shared_ptr, const Message *const> m_in;
    for (const auto &n : outgoing_messages_)
    {
        m_in.insert({n.first, &incoming_message(n.first)});
    }

    return m_in;
}

std::vector<FactorGraph::Node::shared_ptr> FactorGraph::Node::neighbors() const
{
    std::vector<Node::shared_ptr> v;
    for (auto &p : outgoing_messages_)
    {
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
    for (const auto &node : nodes_)
    {
        node->init_messages();
    }
}

Marginals FactorGraph::lbp(const size_t max_iters, const double kl_threshold)
{
    sort_nodes();

    bool converged = false;
    size_t iter = 0;

    while (!converged && iter < max_iters)
    {
        for (auto &node : nodes_)
        {
            node->update_messages();
        }

        iter++;
    }

    Marginals marginals;
    for (const auto &node : nodes_)
    {
        if (Variable::shared_ptr v = std::dynamic_pointer_cast<Variable>(node))
        {
            Message b = v->belief();
            marginals[v->key()] = b.pmf();
        }
    }

    return marginals;
}
