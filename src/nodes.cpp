#include "discrete_factor_graph/factor_graph.h"
#include "discrete_factor_graph/nodes.h"



void Factor::init_messages() {
    // We initialize messages to the same cardinality and 1 for all neighboring Variables
    outgoing_messages_.clear();
    
    for (const auto& var : neighbors_) {
        Variable::shared_ptr v = std::dynamic_pointer_cast<Variable>(var);
        assert(v); // If v is not variable, something is wrong
        outgoing_messages_.push_back(Message(v->discreteKey()));
    }
}

Message::ConvergenceStatus Factor::update_messages() {
    
}


void Variable::init_messages() {
    outgoing_messages_.clear();
    
    size_t num_outgoing_messages = num_neighbors();
    for (size_t i  = 0; i < num_outgoing_messages; i++) {
        outgoing_messages_.push_back(Message(dk_));
    }
}

Message::ConvergenceStatus Variable::update_messages() {

}

