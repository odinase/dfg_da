#include "discrete_factor_graph/factor_graph.h"
#include "discrete_factor_graph/nodes.h"

#include <gtsam/inference/Symbol.h>


void Factor::init_messages() {
    // We initialize messages to the same cardinality and 1 for all neighboring Variables
    
    for (auto& p : outgoing_messages_) {
        Variable::shared_ptr v = std::dynamic_pointer_cast<Variable>(p.first);
        assert(v); // If v is not variable, something is wrong
        p.second = Message(v->discreteKey());
    }
}

Message::ConvergenceStatus Factor::update_messages() {
    Factor f = *this * belief();
    auto m_in = incoming_messages();

    gtsam::KeySet neighbor_keys;
    for (const auto& n : outgoing_messages_) {
        Variable::shared_ptr v = std::dynamic_pointer_cast<Variable>(n.first);
        assert(v);
        neighbor_keys.insert(v->key());
    }

    for (const auto& [node, message] : m_in) {
        Variable::shared_ptr v = std::dynamic_pointer_cast<Variable>(node);
        assert(v);

        neighbor_keys.erase(v->key());
        
        gtsam::DecisionTreeFactor dtf = (f / *message).sum(neighbor_keys.begin(), neighbor_keys.end()).factor();
        std::cout << dtf.size() << "\n";
        assert(dtf.size() == 1);
        outgoing_messages_[node] = Message(dtf);
        outgoing_messages_[node].normalize();

        neighbor_keys.insert(v->key());
    }

    // We don't check for convergence for now
    return Message::ConvergenceStatus::NotConverged;
}

void Factor::print() {
    factor_->print();
}


void Variable::init_messages() {
    for (auto& p: outgoing_messages_) {
        p.second = Message(dk_);
    }
}

Message::ConvergenceStatus Variable::update_messages() {
    Message b = belief();
    auto m_in = incoming_messages();

    for (const auto& [node, message] : m_in) {
        outgoing_messages_[node] = b / *message;
        outgoing_messages_[node].normalize();
    }

    // We don't check for convergence for now
    return Message::ConvergenceStatus::NotConverged;
}

void Variable::print() {
    std::cout << "Variable: " << gtsam::Symbol(dk_.first) << std::endl;
}
