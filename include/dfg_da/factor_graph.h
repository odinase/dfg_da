#pragma once

#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <set>
#include "dfg_da/hypothesis.h"


namespace dfg_da {

namespace factor_graph {

gtsam::DiscreteFactorGraph dfg_from_reward_mat_hyp_prior_single_cluster(const Eigen::MatrixXd& R, const hypothesis::Hypotheses& prior_hypotheses);
gtsam::DiscreteFactorGraph dfg_from_reward_mat_hyp_prior_multicluster(const Eigen::Ref<const Eigen::MatrixXd> &R, const std::vector<dfg_da::hypothesis::Hypotheses> &prior_hypotheses_per_cluster);
std::tuple<Eigen::ArrayXXd, double> exact_marginals_and_normalization_constant(const Eigen::Ref<const Eigen::MatrixXd> &R, const std::vector<dfg_da::hypothesis::Hypotheses> &prior_hypotheses_per_cluster);


struct Message
{    
    gtsam::DecisionTreeFactor m_;
    size_t cardinality_;

    Message() = default;
    explicit Message(const gtsam::DecisionTreeFactor& m);
    // explicit Message(const Factor& f) : Message(f.factor()) {}
    
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

    std::vector<double> pmf();

    enum class ConvergenceStatus {
        Converged,
        NotConverged
    };
};

using Marginals = std::map<gtsam::Key, std::vector<double>>;

class FactorGraph
{
public:
    class Node
    {
        friend class FactorGraph;

    public:
        typedef std::shared_ptr<Node> shared_ptr;

    protected:
        std::unordered_map<Node::shared_ptr, Message> outgoing_messages_;
        inline void add_neighbor(Node::shared_ptr node) { outgoing_messages_.insert({node, {}}); }

    public:
        std::vector<Node::shared_ptr> neighbors() const;
        inline size_t num_neighbors() const { return outgoing_messages_.size(); }

        // Get message from this node into the node calling the method
        const Message& incoming_message(const Node::shared_ptr& node) const;
        std::unordered_map<Node::shared_ptr, const Message* const> incoming_messages() const;

        virtual void init_messages() = 0;
        virtual Message::ConvergenceStatus update_messages() = 0;

        virtual void print() {}
        virtual ~Node() = 0;
    };

private:
    std::vector<Node::shared_ptr> nodes_;
    inline void add_node(Node::shared_ptr node) { nodes_.push_back(node); }
    
    // Sort nodes from fewest to most edges
    inline void sort_nodes()
    {
        std::sort(nodes_.begin(), nodes_.end(), [](const auto &lhs, const auto &rhs)
                  { return lhs->num_neighbors() < rhs->num_neighbors(); });
    }

public:

    FactorGraph() = default;
    explicit FactorGraph(const gtsam::DiscreteFactorGraph &dfg);

    Marginals lbp(const size_t max_iters = 100, const double kl_threshold = 1.0);
    const std::vector<Node::shared_ptr>& nodes() const { return nodes_; }

};
    
gtsam::DiscreteFactorGraph build_test_factor_graph();

} // namespace factor_graph
} // namespace dfg_da 