#include <iostream>
#include <gtsam/discrete/DiscreteConditional.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/discrete/DiscreteMarginals.h>
#include <gtsam/discrete/DecisionTreeFactor.h>
#include <gtsam/discrete/DiscreteDistribution.h>
#include <gtsam/inference/Symbol.h>

// #include <dcsam/DCSAM_types.h>
// #include <dcsam/DiscretePriorFactor.h>

// #include <idbt/iDBT.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <limits>
#include <fstream>

#include <glog/logging.h>
#include <cmath>

#include "discrete_factor_graph/hypothesis.h"
#include "discrete_factor_graph/factor_graph.h"


using gtsam::symbol_shorthand::A;


double approx_normalizing_constant(const Eigen::MatrixXd& R) {
    // [333.96721775 839.64367929]
    // normalizing_constant = np.exp(R_sub[:, 1:]).sum(axis=0).prod()   
    const size_t n = R.rows();
    const size_t m = R.cols() - n;
    
    return R(Eigen::all, Eigen::seqN(0, m)).array().exp().rowwise().sum().prod();
}



int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    constexpr double inf = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd R(3, 4);
    R << 4.78, -0.46, -inf, -inf,
         5.37, -inf, -0.52, -inf,
         6.58, -inf, -inf, -0.60;


    Hypothesis h1({1, 2}, log(0.5));
    Hypothesis h2({1, 3}, log(0.5));

    // Hypothesis h1({1, 2, 3}, log(1.0));
    // We need an empty dummy hypothesis with probability 0 to satisfy GTSAM
    // Hypothesis h2({}, log(0.0));

    Hypotheses h{{h1, h2}};

    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd probs = association_marginal_posteriors(h, R);
    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Spent " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1e-6 << " ms\n";

    std::cout << probs << "\n";
    
    gtsam::DiscreteFactorGraph dfg = dfg_from_reward_mat_hyp_prior(R, h);

    FactorGraph fg(dfg);

    Marginals lbp_marginals = fg.lbp(5);

    std::cout << "Marginals from LBP:\n";
    for (const auto &[k, marginal] : lbp_marginals) {
        std::cout << gtsam::Symbol(k) << ": ";
        for (auto p : marginal) {
            std::cout << p << " ";
        }
        std::cout << std::endl;
    }

    for (const auto& ph : h) {
        Eigen::Map<const Eigen::Array<size_t, Eigen::Dynamic, 1>> tracks(ph.tracks().data(), ph.tracks().size());
        std::cout << approx_normalizing_constant(R(tracks - 1, Eigen::all)) << " ";
    }


    dfg.saveGraph("graph.txt");
}