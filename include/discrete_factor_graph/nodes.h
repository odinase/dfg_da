#pragma once

#include "discrete_factor_graph/factor_graph.h"


class Factor : public FactorGraph::Node
{
private:
    gtsam::DecisionTreeFactor::shared_ptr factor_;

public:
    typedef std::shared_ptr<Factor> shared_ptr;

    Factor(gtsam::DecisionTreeFactor::shared_ptr factor) : factor_(factor) {}

    Factor operator/(const Message& m) const {
        gtsam::DecisionTreeFactor::shared_ptr fac = boost::make_shared<gtsam::DecisionTreeFactor>(*factor_ / m.m_);
        return Factor(fac);
    }

    template<class Iterator>
    Factor sum(Iterator first, Iterator last) {
        gtsam::Ordering vars_to_sum(first, last);
        return *factor_->sum(vars_to_sum);
    }

    Factor sum(gtsam::Key key) {
        gtsam::KeyVector kv{key};
        return sum(kv.begin(), kv.end());
    }

    virtual void init_messages();
    virtual Message::ConvergenceStatus update_messages();
};

class Variable : public FactorGraph::Node
{
private:
    gtsam::DiscreteKey dk_;

public:
    typedef std::shared_ptr<Variable> shared_ptr;

    Variable(gtsam::DiscreteKey dk) : dk_(dk) {}

    inline gtsam::DiscreteKey discreteKey() const {return dk_; }
    inline gtsam::Key key() const { return dk_.first; }
    inline size_t cardinality() const { return dk_.second; }

    virtual void init_messages();
    virtual Message::ConvergenceStatus update_messages();
};