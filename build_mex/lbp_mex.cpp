/* lbp_mex
 * [marginals, bethe_constant] = lbp_mex(gainMatPostC, num_tracks, num_measurements, hypos, hyposCard, probLogHypos, clusters, clustersCard);
 * Takes in reward matrix and hypotheses for all clusters and runs LBP for approximate marginals
*/

#include "mex.hpp"
#include "mexAdapter.hpp"
#include "dfg_da/lbp.h"
#include "dfg_da/hypothesis.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <numeric>

// using namespace matlab::data;
// using matlab::mex::ArgumentList;

class MexFunction : public matlab::mex::Function {
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        // checkArguments(outputs, inputs);
        // const double offSet = inputs[0][0];
        // TypedArray<double> doubleArray = std::move(inputs[1]);
        // for (auto& elem : doubleArray) {
        //     elem += offSet;
        // }
        // outputs[0] = doubleArray;
        std::cout << "Received " << inputs.size() << " inputs\n";
        auto reward_matrix = reward_matrix_conversion(inputs);
        std::vector<dfg_da::hypothesis::Hypotheses> hypos_in_clusters = hypotheses_conversion(inputs);
        std::cout << "Completed!\n";
        for (size_t i = 0; i < hypos_in_clusters.size(); i++) {
            std::cout << "Cluster " << i + 1 << ":\n";
            const auto& hypos = hypos_in_clusters[i];
            for (size_t h = 0; h < hypos.num_hypotheses(); h++) {
                std::cout << "\tHypothesis " << h + 1 << "\n";
                std::cout << "\t\tProbability: " << hypos[h].probability();
                std::cout << "\t\tTracks: ";
                for (const auto& t : hypos[h].tracks()) {
                    std::cout << t << " ";
                }
                std::cout << "\n";
            }
        }

        std::cout << reward_matrix.rows() << ", " << reward_matrix.cols() << "\n";
        std::cout << reward_matrix << "\n";
        // std::cout << reward_matrix << "\n";
    }

    // Shamelessly stolen from https://se.mathworks.com/matlabcentral/answers/1573713-c-mex-data-api-how-can-i-get-the-raw-pointer-of-an-array-input-without-copying-it
    // Accessed 2. feb 2023
    template <typename T>
    const T* getDataPtr(matlab::data::Array arr) {
        const matlab::data::TypedArray<T> arr_t = arr;
        matlab::data::TypedIterator<const T> it(arr_t.begin());
        return it.operator->();
    }

    Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>> reward_matrix_conversion(matlab::mex::ArgumentList inputs) {
        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        const size_t num_tracks = inputs[1][0];
        const size_t num_measurements = inputs[2][0];
        const double* data = getDataPtr<double>(inputs[0]);

        return Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>>(data, num_tracks, num_tracks + num_measurements, Eigen::OuterStride<>(num_tracks + num_measurements));
    }

    std::vector<dfg_da::hypothesis::Hypotheses> hypotheses_conversion(matlab::mex::ArgumentList inputs) {
        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        matlab::data::TypedArray<double> hypos = inputs[3];
        matlab::data::TypedArray<double> hyposCard = inputs[4];
        matlab::data::TypedArray<double> probLogHypos = inputs[5];
        matlab::data::TypedArray<double> clusters = inputs[6];
        matlab::data::TypedArray<double> clustersCard = inputs[7];

        // Compute cumsum indices for getting tracks in hypotheses down the line
        size_t num_hypos = hyposCard.getNumberOfElements();
        std::vector<double> endInd;
        endInd.reserve(num_hypos);
        std::vector<double> beginInd;
        beginInd.reserve(num_hypos);
        size_t prev_endi = 0;
        for (size_t i = 0; i < num_hypos; i++) {
            size_t begini = prev_endi;
            size_t endi = begini + hyposCard[i];
            
            endInd.push_back(endi);
            beginInd.push_back(begini);

            prev_endi = endi;
        }

        std::vector<dfg_da::hypothesis::Hypotheses> hypos_in_clusters;
        hypos_in_clusters.reserve(clustersCard.getNumberOfElements());
        size_t start = 0, ci = 0;
        // We loop over each cluster, and collect the hypotheses
        for (const auto& cc : clustersCard) {
            size_t cluster_card = cc;
            std::vector<dfg_da::hypothesis::Hypothesis> hs;

            size_t stop = start + cluster_card;
            // Loop over all hypotheses in cluster and retrieve 
            for (size_t i = start; i < stop; i++) {
                size_t hypothesis_in_c = clusters[i] - 1; // We subtract 1 to make the hypothesis indexable
                double log_prob_hypothesis = probLogHypos[hypothesis_in_c];

                // Get tracks in hypothesis_in_c
                std::vector<size_t> tracks;
                for (size_t t_idx = beginInd[hypothesis_in_c]; t_idx < endInd[hypothesis_in_c]; t_idx++) {
                    size_t track = hypos[t_idx];
                    tracks.push_back(track);
                }
                hs.emplace_back(dfg_da::hypothesis::Hypothesis(std::move(tracks), log_prob_hypothesis));
            }

            hypos_in_clusters.emplace_back(dfg_da::hypothesis::Hypotheses(std::move(hs)));
            start += cluster_card;
            ci += 1;
        }

        return hypos_in_clusters;
    }

    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        // Get pointer to engine
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

        // Get array factory
        matlab::data::ArrayFactory factory;

        // Check offset argument: First input must be scalar double
        if (inputs[0].getType() != matlab::data::ArrayType::DOUBLE ||
            inputs[0].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE ||
            inputs[0].getNumberOfElements() != 1)
        {
            matlabPtr->feval(u"error",
                0,
                std::vector<matlab::data::Array>({ factory.createScalar("First input must be scalar double") }));
        }

        // Check array argument: Second input must be double array
        if (inputs[1].getType() != matlab::data::ArrayType::DOUBLE ||
            inputs[1].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE)
        {
            matlabPtr->feval(u"error",
                0,
                std::vector<matlab::data::Array>({ factory.createScalar("Input must be double array") }));
        }
        // Check number of outputs
        if (outputs.size() > 1) {
            matlabPtr->feval(u"error",
                0,
                std::vector<matlab::data::Array>({ factory.createScalar("Only one output is returned") }));
        }
    }
};