#pragma once

#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <set>


struct Message
{    
    gtsam::DecisionTreeFactor m_;
    size_t cardinality_;

    Message() = default;
    explicit Message(const gtsam::DecisionTreeFactor& m);
    Message(gtsam::DiscreteKey dk) : m_({dk}, std::vector<double>(dk.second, 1.0)) {}

    inline void normalize() { m_ = m_ / *(m_.sum(1)); }

    Message operator*(const Message& m) const {
        return Message(m_ * m.m_);
    }

    Message& operator*=(const Message& m) {
        *this = (*this)*m;
        return *this;
    }

    Message operator/(const Message& m) const {
        return Message(m_ / m.m_);
    }

    enum class ConvergenceStatus {
        Converged,
        NotConverged
    };
};

using Marginals = std::unordered_map<gtsam::Key, std::vector<double>>;

class FactorGraph
{
public:
    class Node
    {
        friend class FactorGraph;

    public:
        typedef std::shared_ptr<Node> shared_ptr;

    protected:
        std::vector<Message> outgoing_messages_;
        std::vector<Node::shared_ptr> neighbors_;
        inline void add_neighbor(Node::shared_ptr node) { neighbors_.push_back(node); }

    public:
        const std::vector<Node::shared_ptr> &neighbors() const { return neighbors_; }
        inline size_t num_neighbors() const { return neighbors_.size(); }

        // Belief is product of incoming messages normalized
        Message belief() const;
        // Get message from this node into the node calling the method
        const Message& incoming_message(const Node::shared_ptr& node) const;

        virtual void init_messages() = 0;
        virtual Message::ConvergenceStatus update_messages() = 0;
        virtual ~Node() = 0;
    };

    FactorGraph() = default;
    explicit FactorGraph(const gtsam::DiscreteFactorGraph &dfg);

    Marginals lbp();

private:
    std::vector<Node::shared_ptr> nodes_;
    void add_node(Node::shared_ptr node);
    
    // Sort nodes from fewest to most edges
    inline void sort_nodes()
    {
        std::sort(nodes_.begin(), nodes_.end(), [](const auto &lhs, const auto &rhs)
                  { return lhs->num_neighbors() < rhs->num_neighbors(); });
    } 
};
