#include "dfg_da/factor_graph.h"
#include "dfg_da/factor_graph/nodes.h"

#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteDistribution.h>
#include <gtsam/inference/Symbol.h>



namespace dfg_da {

namespace factor_graph {


using gtsam::symbol_shorthand::A;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::T;


constexpr bool xnor(const bool x, const bool y) { return !(x != y);}




double approx_normalizing_constant(const Eigen::MatrixXd& R, const std::vector<size_t>& tracks) {
    // [333.96721775 839.64367929]
    // normalizing_constant = np.exp(R_sub[:, 1:]).sum(axis=0).prod()   
    const size_t n = R.rows();
    const size_t m = R.cols() - n;

    Eigen::Map<const Eigen::Array<size_t, Eigen::Dynamic, 1>> track_idxs(tracks.data(), tracks.size());
    double mu = (1.0 - R.rightCols(n).diagonal().array().exp()).sum();

    return exp(-mu) * R(track_idxs - 1, Eigen::seqN(0, m)).array().exp().colwise().sum().prod();
}

gtsam::DiscreteFactorGraph dfg_from_reward_mat_hyp_prior(const Eigen::MatrixXd &R, const hypothesis::Hypotheses &prior_hypotheses)
{
    gtsam::DiscreteFactorGraph dfg;

    // First construct hypothesis prior factor and variable
    const size_t num_prior_hypotheses = prior_hypotheses.num_hypotheses();
    gtsam::DiscreteKey th{T(0), num_prior_hypotheses};
    std::vector<double> theta_table = prior_hypotheses.hypothesis_probabilites();
    // std::vector<double> normalizing_constants;
    // for (size_t i = 0; i < prior_hypotheses.num_hypotheses(); i++) {
    //     double c = approx_normalizing_constant(R, prior_hypotheses[i].tracks());
    //     normalizing_constants.push_back(c);
    // }
    gtsam::DiscreteDistribution th_factor(th, theta_table);
    dfg.push_back(th_factor);

    // Add all track variables.
    // The reward matrix should nt x (m + nt), ie, one row for each track
    // and one column for each measurement plus columns for misdetection
    assert(R.cols() >= R.rows());
    const size_t num_tracks = R.rows();
    const size_t num_measurements = R.cols() - num_tracks;

    gtsam::DiscreteKeys ais; // Track variables and measurement variables
    ais.reserve(num_tracks);

    // Hard compatability constraints are basically: 1 everywhere except nonexistence if it exists in the prior hypothesis
    for (size_t track = 1; track <= num_tracks; track++)
    {
        gtsam::DiscreteKey ai(A(track), 1 + num_measurements + 1); // misdetection + num measurements + non-existence
        ais.push_back(ai);
        gtsam::DiscreteKeys keys{th, ai};

        // Add factor between hypothesis variable and track variable
        std::vector<double> compatibility_table;
        for (size_t hypo = 0; hypo < num_prior_hypotheses; hypo++)
        {
            bool contained_in_hypo = prior_hypotheses[hypo].contains(track);
            for (size_t meas = 0; meas <= num_measurements + 1; meas++)
            {
                bool exists = meas < (num_measurements + 1);
                compatibility_table.push_back(xnor(contained_in_hypo, exists)); // If contained in hypothesis and less than non-existence id, use 1
                // If not contained and also non-existence => true
                // If contained and existence => true
                // Otherwise, false
                // The above constraints should be NXOR (XNOR?)
            }
        }

        gtsam::DecisionTreeFactor hyp_to_track_factor(keys, compatibility_table);
        dfg.push_back(hyp_to_track_factor);


        // Add prior factors
        // We assume that the reward matrix is the logarithm of probabilities, as this is common to use
        std::vector<double> prior_table;

        size_t t_idx = track - 1; // The rows are 0-indexed, so we need to offset the track index.
        size_t misdetection_idx = num_measurements + t_idx;
        // Add misdetection
        double m = exp(R(t_idx, misdetection_idx));
        prior_table.push_back(m);

        // Add all likelihoods
        for (size_t meas_idx = 0; meas_idx < num_measurements; meas_idx++) {
            double l = exp(R(t_idx, meas_idx));
            prior_table.push_back(l);
        }

        // Lastly, add non-existence
        prior_table.push_back(1.0);

        gtsam::DiscreteDistribution prior_factor(ai, prior_table);
        dfg.push_back(prior_factor);
    }

    // Final stretch, add measurement variables
    // Lets 1-index measurements as well for consistency
    for (size_t meas = 1; meas <= num_measurements; meas++) {
        gtsam::DiscreteKey bj_dk(B(meas), 1 + num_tracks); // clutter or associated to a track

        uint64_t bj = gtsam::symbolIndex(bj_dk.first);

        // Add factor from measurement to all track variables
        for (const auto& ai_dk : ais) {
            std::vector<double> compatibility_table;

            size_t track_cardinality = ai_dk.second;
            uint64_t ai = gtsam::symbolIndex(ai_dk.first);

            // We need to do this row-major, so fix the row. Along one row we vary what association is compatible for this measurement
            for (size_t j = 0; j < track_cardinality; j++) {
                for (size_t i = 0; i <= num_tracks; i++) {
                    // The compatibility here is the fact that we must assign bt to at for at == bt and no other ats, or otherwise the opposite
                    // Here we see the benefit of using 1-indexed measurements: Since the tracks assume that measurement 0 is misdetection, c will automatically point to correct measurement
                    
                    double compatibility = xnor(i == ai, j == bj);
                    compatibility_table.push_back(compatibility);
                }
            }
            gtsam::DiscreteKeys keys{ai_dk, bj_dk};
            gtsam::DecisionTreeFactor meas_to_track_factor(keys, compatibility_table);
            dfg.push_back(meas_to_track_factor);
        }
    }

    return dfg;
}

Message::Message(const gtsam::DecisionTreeFactor &m)
{
    assert(m.size() == 1);
    cardinality_ = m.discreteKeys()[0].second;
    m_ = m;
}

std::vector<double> Message::pmf()
{
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
    std::unordered_map<gtsam::Key, nodes::Variable::shared_ptr> seen_vars;

    for (auto &&fac : dfg)
    {
        // All factors in the dfg should be castable to DecisionTreeFactor, if not, something is seriously wrong
        if (gtsam::DecisionTreeFactor::shared_ptr df = boost::dynamic_pointer_cast<gtsam::DecisionTreeFactor>(fac))
        {
            nodes::Factor::shared_ptr factor = std::make_shared<nodes::Factor>(df);
            add_node(factor);

            for (const auto &dk : df->discreteKeys())
            {
                nodes::Variable::shared_ptr var;
                // Add variable if not seen before
                if (seen_vars.find(dk.first) == seen_vars.end())
                {
                    var = std::make_shared<nodes::Variable>(dk);
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
        if (nodes::Variable::shared_ptr v = std::dynamic_pointer_cast<nodes::Variable>(node))
        {
            Message b = v->belief();
            marginals[v->key()] = b.pmf();
        }
    }

    return marginals;
}

gtsam::DiscreteFactorGraph build_test_factor_graph() {
    // Initialize discrete prior hypothesis variable theta
    gtsam::DiscreteKey theta(gtsam::symbol('T', 0), 2);

    double w_a = 0.5;
    double w_b = 1.0 - w_a;
    std::vector<std::vector<int>> prior_hypotheses = {
        {1, 2},
        {1, 3}}; // Tracks
    std::vector<double> theta_prior_probs{w_a, w_b};
    gtsam::DiscreteDistribution phi_H(theta, theta_prior_probs);

    gtsam::DiscreteFactorGraph dfg{};
    dfg.push_back(phi_H);

    // Add Bernoulli components (tracks??)
    gtsam::DiscreteKeys as;
    for (int i = 1; i <= 3; i++)
    {
        as.emplace_back(gtsam::DiscreteKey(A(i), 3)); // Cardinality is 3 because one measurement => misdetection, measurement, non-existence
    }

    // Add factors between theta and the tracks
    // a1
    gtsam::DiscreteKeys phi_A_keys = {theta, as[0]};
    std::vector<double> phi_A_table = {
        1, 1, 0,
        1, 1, 0
    };
    gtsam::DecisionTreeFactor phi_A(phi_A_keys, phi_A_table);
    dfg.push_back(phi_A);

    // a2
    gtsam::DiscreteKeys phi_B_keys = {theta, as[1]};
    std::vector<double> phi_B_table = {
        1, 1, 0,
        0, 0, 1
    };
    gtsam::DecisionTreeFactor phi_B(phi_B_keys, phi_B_table);
    dfg.push_back(phi_B);

    // a3
    gtsam::DiscreteKeys phi_C_keys = {theta, as[2]};
    std::vector<double> phi_C_table = {
        0, 0, 1,
        1, 1, 0
    };
    gtsam::DecisionTreeFactor phi_C(phi_C_keys, phi_C_table);
    dfg.push_back(phi_C);


    // track-to-measurement factors
    // Define b variable
    gtsam::DiscreteKey b(gtsam::symbol('b', 1), 4); // Cardinality 4 because three different tracks or misdetection??

    // a1
    gtsam::DiscreteKeys phi_X_keys = {as[0], b};
    std::vector<double> phi_X_table = {
        1, 0, 1, 1,
        0, 1, 0, 0,
        1, 0, 1, 1,
    };
    gtsam::DecisionTreeFactor phi_X(phi_X_keys, phi_X_table);
    dfg.push_back(phi_X);

    // a2
    gtsam::DiscreteKeys phi_Y_keys = {as[1], b};
    std::vector<double> phi_Y_table = {
        1, 1, 0, 1,
        0, 0, 1, 0,
        1, 1, 0, 1,
    };
    gtsam::DecisionTreeFactor phi_Y(phi_Y_keys, phi_Y_table);
    dfg.push_back(phi_Y);

    // a3
    gtsam::DiscreteKeys phi_Z_keys = {as[2], b};
    std::vector<double> phi_Z_table = {
        1, 1, 1, 0,
        0, 0, 0, 1,
        1, 1, 1, 0,
    };
    gtsam::DecisionTreeFactor phi_Z(phi_Z_keys, phi_Z_table);
    dfg.push_back(phi_Z);

    // Add unary track factors
    // Reward matrix
    // R = [ vertcat( l^{11}, l^{21}, l^{31} , [m^1, -infty, -infty ; -infty, m^2 , -infty ; -infty, -infty, m^3].
    // Three tracks and one measurement plus three misdetections
    constexpr double inf = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd R(3, 4);
    R << 4.78, -0.46, -inf, -inf,
         5.37, -inf, -0.52, -inf,
         6.58, -inf, -inf, -0.60;

    // exp to convert log into actual probabilities. Is this properly normalized?? Does it need to??
    double l_11 = exp(R(0,0));
    double l_21 = exp(R(1,0));
    double l_31 = exp(R(2,0));

    double m_1 = exp(R(0, 1));
    double m_2 = exp(R(1, 2));
    double m_3 = exp(R(2, 3));

    // phi D
    std::vector<double> phi_D_table{m_1, l_11, 1};
    gtsam::DiscreteDistribution phi_D(as[0], phi_D_table);
    dfg.push_back(phi_D);

    // phi E
    std::vector<double> phi_E_table{m_2, l_21, 1};
    gtsam::DiscreteDistribution phi_E(as[1], phi_E_table);
    dfg.push_back(phi_E);

    // phi F
    std::vector<double> phi_F_table{m_3, l_31, 1};
    gtsam::DiscreteDistribution phi_F(as[2], phi_F_table);
    dfg.push_back(phi_F);


    return dfg;
}


} // namespace factor_graph
} // namespace dfg_da
