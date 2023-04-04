#include "dfg_da/factor_graph.h"
#include "dfg_da/factor_graph/nodes.h"

#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteDistribution.h>
#include <gtsam/inference/Symbol.h>
#include <numeric>
#include <sstream>


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

gtsam::DiscreteFactorGraph dfg_from_reward_mat_hyp_prior_multicluster(const Eigen::Ref<const Eigen::MatrixXd> &R, const std::vector<dfg_da::hypothesis::Hypotheses> &prior_hypotheses_per_cluster)
{
    gtsam::DiscreteFactorGraph dfg;

    // Build left side of graph: Connect tracks to hypothesis variable for each cluster
    const size_t num_clusters = prior_hypotheses_per_cluster.size();
    // Add all track variables.
    // The reward matrix should nt x (m + nt), ie, one row for each track
    // and one column for each measurement plus columns for misdetection
    assert(R.cols() >= R.rows());
    const size_t num_tracks = R.rows();
    const size_t num_measurements = R.cols() - num_tracks;
    gtsam::DiscreteKeys ais; // Track variables and measurement variables
    ais.reserve(num_tracks);

    for (size_t c = 0; c < num_clusters; c++)
    {
        dfg_da::hypothesis::Hypotheses prior_hypotheses = prior_hypotheses_per_cluster[c];
        // First construct hypothesis prior factor and variable
        size_t num_prior_hypotheses = prior_hypotheses.num_hypotheses();
        if (num_prior_hypotheses == 1) {
            // GTSAM doesn't like variables with cardinality 1, so append dummy hypothesis
            hypothesis::Hypothesis dummy({}, -std::numeric_limits<double>::infinity());
            prior_hypotheses.append(dummy);
            num_prior_hypotheses += 1;
        }
        std::set<size_t> tracks_in_cluster = prior_hypotheses.tracks();

        gtsam::DiscreteKey th{T(c + 1), num_prior_hypotheses};
        std::vector<double> theta_table = prior_hypotheses.hypothesis_probabilites();
        gtsam::DiscreteDistribution th_factor(th, theta_table);
        dfg.push_back(th_factor);
        // Hard compatability constraints are basically: 1 everywhere except nonexistence if it exists in the prior hypothesis
        for (const size_t track : tracks_in_cluster)
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
            for (size_t meas_idx = 0; meas_idx < num_measurements; meas_idx++)
            {
                double l = exp(R(t_idx, meas_idx));
                prior_table.push_back(l);
            }

            // Lastly, add non-existence
            prior_table.push_back(1.0);

            gtsam::DiscreteKeys aik = {ai};
            gtsam::DecisionTreeFactor prior_factor(aik, prior_table);
            dfg.push_back(prior_factor);
        }
    }

    // Final stretch, add measurement variables
    // Lets 1-index measurements as well for consistency
    for (size_t meas = 1; meas <= num_measurements; meas++)
    {
        gtsam::DiscreteKey bj_dk(B(meas), 1 + num_tracks); // clutter or associated to a track

        uint64_t bj = gtsam::symbolIndex(bj_dk.first);

        // Add factor from measurement to all track variables
        for (const auto &ai_dk : ais)
        {
            size_t track_cardinality = ai_dk.second;
            uint64_t ai = gtsam::symbolIndex(ai_dk.first);
            
            // if (!std::isfinite(R(ai - 1, bj - 1))) {
            //     continue;
            // }
            std::vector<double> compatibility_table;


            // We need to do this row-major, so fix the row. Along one row we vary what association is compatible for this measurement
            for (size_t j = 0; j < track_cardinality; j++)
            {
                for (size_t i = 0; i <= num_tracks; i++)
                {
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

std::tuple<Eigen::ArrayXXd, double> exact_marginals_and_normalization_constant(const Eigen::Ref<const Eigen::MatrixXd> &R, const std::vector<dfg_da::hypothesis::Hypotheses> &prior_hypotheses_per_cluster) {
    gtsam::DiscreteFactorGraph dfg = dfg_from_reward_mat_hyp_prior_multicluster(R, prior_hypotheses_per_cluster);

    const size_t num_tracks = R.rows();
    const size_t num_measurements = R.cols() - num_tracks;

    auto fac = dfg.product();
    size_t num_thetas = prior_hypotheses_per_cluster.size();
    auto ff = fac.sum(num_thetas + num_tracks + num_measurements);

    double exact_normalization_constant = (*ff)({});
    
    gtsam::DiscreteMarginals dfg_marginals(dfg);

    auto dks = dfg.discreteKeys();
    std::set<gtsam::DiscreteKey> all_keys;
    for (const auto& dk : dks) {
        if (gtsam::symbolChr(dk.first) == 'a') {
            all_keys.insert(dk);
        }
    }

    Eigen::ArrayXXd exact_marginals(2 + num_measurements, num_tracks);
    size_t c = 0;
    for (const auto& key : all_keys) {
        exact_marginals.col(c) = dfg_marginals.marginalProbabilities(key);
        c += 1;
    }

    return {exact_marginals, exact_normalization_constant};
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, std::map<std::string, Eigen::ArrayXd>, double> all_exact_marginals_and_normalization_constant(const Eigen::Ref<const Eigen::MatrixXd> &R, const std::vector<dfg_da::hypothesis::Hypotheses> &prior_hypotheses_per_cluster) {
    gtsam::DiscreteFactorGraph dfg = dfg_from_reward_mat_hyp_prior_multicluster(R, prior_hypotheses_per_cluster);

    const size_t num_tracks = R.rows();
    const size_t num_measurements = R.cols() - num_tracks;

    auto fac = dfg.product();
    size_t num_thetas = prior_hypotheses_per_cluster.size();
    auto ff = fac.sum(num_thetas + num_tracks + num_measurements);

    double exact_normalization_constant = (*ff)({});
    
    gtsam::DiscreteMarginals dfg_marginals(dfg);

    auto dks = dfg.discreteKeys();
    std::set<gtsam::DiscreteKey> all_keys(dks.begin(), dks.end());
    gtsam::DiscreteKeys ais;
    gtsam::DiscreteKeys bjs;
    gtsam::DiscreteKeys ths;
    for (const auto& dk : all_keys) {
        if (gtsam::symbolChr(dk.first) == 'a') {
            ais.push_back(dk);
        } else if (gtsam::symbolChr(dk.first) == 'b') {
            bjs.push_back(dk);
        } else if (gtsam::symbolChr(dk.first) == 't') {
            ths.push_back(dk);
        } else {
            std::cerr << "Unknown discrete key: " << dk.first << std::endl;
        }
    }

    Eigen::ArrayXXd track_marginals(2 + num_measurements, num_tracks), meas_marginals(1 + num_tracks, num_measurements);
    std::map<std::string, Eigen::ArrayXd> theta_marginals;
    size_t c = 0;
    for (const auto& aik : ais) {
        track_marginals.col(c) = dfg_marginals.marginalProbabilities(aik);
        c += 1;
    }

    c = 0;
    for (const auto& bjk : bjs) {
        meas_marginals.col(c) = dfg_marginals.marginalProbabilities(bjk);
        c += 1;
    }

    std::stringstream ss;
    for (const auto& thk : ths) {
        ss << gtsam::Symbol(thk.first);
        theta_marginals.insert({
            ss.str(), dfg_marginals.marginalProbabilities(thk)
        });
        ss.str("");
    }

    return {track_marginals, meas_marginals, theta_marginals, exact_normalization_constant};
}



Eigen::ArrayXd hypothesis_conditioned_likelihoods(const Eigen::Ref<const Eigen::MatrixXd> &R, const dfg_da::hypothesis::Hypotheses &prior_hypotheses) {
    gtsam::DiscreteFactorGraph dfg = dfg_from_reward_mat_hyp_prior(R, prior_hypotheses);

    auto dks = dfg.discreteKeys();
    std::set<gtsam::DiscreteKey> all_keys(dks.begin(), dks.end());

    size_t num_hypos = prior_hypotheses.num_hypotheses();
    gtsam::DiscreteKey t{T(0), num_hypos};

    all_keys.erase(t);

    gtsam::Ordering vars_to_sum_out{};
    for (const auto& dk : all_keys) {
        vars_to_sum_out += dk.first;
    }

    auto fac = dfg.product();
    auto ff = fac.sum(vars_to_sum_out);
    Eigen::ArrayXd likelihoods(num_hypos);

    std::vector<double> hypo_probs = prior_hypotheses.hypothesis_probabilites();
    for (size_t val = 0; val < num_hypos; val++) {
        gtsam::DiscreteValues v;
        v.insert({t.first, val});
        double l = (*ff)(v);
        // Evaluating the factor graph in this point is just the joint p(Z, theta), so divide by p(theta) to get conditional
        likelihoods(val) = l / hypo_probs[val];
    }

    return likelihoods;
}



// std::tuple<Eigen::ArrayXXd, double> exact_marginals_and_normalization_constant(const Eigen::Ref<const Eigen::MatrixXd> &R, const std::vector<dfg_da::hypothesis::Hypotheses> &prior_hypotheses_per_cluster_posterior) {

//     const size_t num_tracks = R.rows();
//     const size_t num_measurements = R.cols() - num_tracks;

//     Eigen::ArrayXXd exact_marginals(2 + num_measurements, num_tracks);
//     double exact_normalization_constant = 1.0;

//     for (auto& h : prior_hypotheses_per_cluster_posterior) {
//         auto tracks = h.tracks();
//         Eigen::ArrayXi t_idx(tracks.begin(), tracks.end());
//         t_idx -= 1;
//         gtsam::DiscreteFactorGraph dfg = dfg_from_reward_mat_hyp_prior_single_cluster(R, h);
//         auto fac = dfg.product();
//         auto ff = fac.sum(1 + num_tracks + num_measurements);

//         double exact_normalization_constant = (*ff)({});
        
//         gtsam::DiscreteMarginals dfg_marginals(dfg);

//         auto dks = dfg.discreteKeys();
//         std::set<gtsam::DiscreteKey> all_keys;
//         for (const auto& dk : dks) {
//             if (gtsam::symbolChr(dk.first) == 'a') {
//                 all_keys.insert(dk);
//             }
//         }

//         size_t c = 0;
//         for (const auto& key : all_keys) {
//             exact_marginals.col(c) = dfg_marginals.marginalProbabilities(key);
//             c += 1;
//         }
//     }

//     return {exact_marginals, exact_normalization_constant};
// }


// std::tuple<Eigen::MatrixXd, gtsam::KeyVector> calculate_precision_matrix(const gtsam::DiscreteFactorGraph& dfg) {
//     // Let's do this super stupid to start of. We'll loop over all pairs of variables and compute the covariance for each

//     // First, get all the keys
//     auto ks = dfg.discreteKeys();
//     std::set<gtsam::DiscreteKey> dks(ks.begin(), ks.end());

//     // Now, we'll loop over all pairs of keys and compute the covariance
//     size_t num_keys = dks.size();
//     gtsam::KeyVector rest_keys;
//     std::transform(dks.begin(), dks.end(), std::back_inserter(rest_keys), [](const gtsam::DiscreteKey& dk) { return dk.first; });

//     Eigen::MatrixXd covariance_matrix(num_keys, num_keys);
//     for (size_t i = 0; i < num_keys; i++) {
//         auto k1_iter = std::find(rest_keys.begin(), rest_keys.end(), dks[i].first);
//         rest_keys.erase(k1_iter);
//         // Ok, we have to be smarter here.
//         // We marginalize out one variable, as we need it's marginal distribution anyway
//         // The remaining Bayes tree is probably cheaper to marginalize as well(??)
//         auto [bt, marg_fg_i] = dfg.eliminatePartialMultifrontal(rest_keys);
        
//         // Expected value
//         double Ei = 0.0;
//         for (size_t ki = 0; ki < k1_iter->second; ki++) {
//             gtsam::DiscreteValues dv;
//             dv.insert(dks[i].first, ki);
//             Ei += ki * (*marg_fg_i)(dv);
//         }
//         double Zi = (*(marg_fg_i->product().sum(1)))({});
//         Ei =/ Zi;

//         // Compute the variance
//         double Vi = 0.0;
//         for (size_t ki = 0; ki < k1_iter->second; ki++) {
//             gtsam::DiscreteValues dv;
//             dv.insert(dks[i].first, ki);
//             Vi += (ki - Ei) * (ki - Ei) * (*marg_fg_i)(dv);
//         }
//         Vi =/ Zi;
//         covariance_matrix(i, i) = Vi;

//         for (size_t j = i + 1; j < num_keys; j++) {
//             auto k2_iter = std::find(rest_keys.begin(), rest_keys.end(), dks[j].first);
//             rest_keys.erase(k2_iter);

//             size_t num_vars_left = dks.size() - rest_keys.size();
//             double Z = (*dfg.product().sum(num_vars_left))({});

//             double cov = 0.0;
//             for (size_t ki = 0; ki < k1_iter->second; ki++) {
//                 for (size_t kj = 0; kj < k2_iter->second; kj++) {
//                     gtsam::DiscreteValues dv;
//                     dv.insert(dks[i].first, ki);
//                     dv.insert(dks[j].first, kj);
//                     double f = (*fg)({dv});
//                     cov += f;
//                 }
//             }

//             precision_matrix(i, j) = cov_ij;

//             rest_keys.push_back(dks[j].first);
//         }
//         rest_keys.push_back(dks[i].first);
//     }
// }


} // namespace factor_graph
} // namespace dfg_da
