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

class MexFunction : public matlab::mex::Function
{
public:
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs)
    {
        checkArguments(outputs, inputs);
        auto reward_matrix = reward_matrix_conversion(inputs);
        std::vector<dfg_da::hypothesis::Hypotheses> hypos_in_clusters = hypotheses_conversion(inputs);

        // Eigen::ArrayXXd
        matlab::data::ArrayFactory f;

        size_t num_tracks = reward_matrix.rows();
        size_t num_measurements = reward_matrix.cols() - num_tracks;
        matlab::data::buffer_ptr_t<double> data = f.createBuffer<double>(num_tracks * (2 + num_measurements));
        Eigen::Map<Eigen::ArrayXXd> marginals(data.get(), 2 + num_measurements, num_tracks);

        marginals = dfg_da::lbp::lbp_multicluster(reward_matrix, hypos_in_clusters);

        size_t num_outputs = outputs.size();
        if (num_outputs >= 1)
        {
            uint64_t r = marginals.rows();
            uint64_t c = marginals.cols();
            outputs[0] = f.createArrayFromBuffer<double>({r, c}, std::move(data));
        }
        if (num_outputs >= 2)
        {
            outputs[1] = f.createArray<double>({2, 2}, {1.2, 2.2, 3.2, 4.2});
        }
    }

    // Shamelessly stolen from https://se.mathworks.com/matlabcentral/answers/1573713-c-mex-data-api-how-can-i-get-the-raw-pointer-of-an-array-input-without-copying-it
    // Accessed 2. feb 2023
    template <typename T>
    const T *getDataPtr(matlab::data::Array arr)
    {
        const matlab::data::TypedArray<T> arr_t = arr;
        matlab::data::TypedIterator<const T> it(arr_t.begin());
        return it.operator->();
    }

    Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>> reward_matrix_conversion(matlab::mex::ArgumentList inputs)
    {
        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        const size_t num_tracks = inputs[1][0];
        const size_t num_measurements = inputs[2][0];
        const double *data = getDataPtr<double>(inputs[0]);

        return Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>>(data, num_tracks, num_tracks + num_measurements, Eigen::OuterStride<>(num_tracks + num_measurements));
    }

    std::vector<dfg_da::hypothesis::Hypotheses> hypotheses_conversion(matlab::mex::ArgumentList inputs)
    {
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
        for (size_t i = 0; i < num_hypos; i++)
        {
            size_t begini = prev_endi;
            size_t endi = begini + hyposCard[i];

            endInd.push_back(endi);
            beginInd.push_back(begini);

            prev_endi = endi;
        }

        std::vector<dfg_da::hypothesis::Hypotheses> hypos_in_clusters;
        hypos_in_clusters.reserve(clustersCard.getNumberOfElements());
        size_t start = 0;
        // We loop over each cluster, and collect the hypotheses
        for (const auto &cc : clustersCard)
        {
            size_t cluster_card = cc;
            std::vector<dfg_da::hypothesis::Hypothesis> hs;

            size_t stop = start + cluster_card;
            // Loop over all hypotheses in cluster and retrieve
            for (size_t i = start; i < stop; i++)
            {
                size_t hypothesis_in_c = clusters[i] - 1; // We subtract 1 to make the hypothesis indexable
                double log_prob_hypothesis = probLogHypos[hypothesis_in_c];

                // Get tracks in hypothesis_in_c
                std::vector<size_t> tracks;
                for (size_t t_idx = beginInd[hypothesis_in_c]; t_idx < endInd[hypothesis_in_c]; t_idx++)
                {
                    size_t track = hypos[t_idx];
                    tracks.push_back(track);
                }
                hs.emplace_back(dfg_da::hypothesis::Hypothesis(std::move(tracks), log_prob_hypothesis));
            }

            hypos_in_clusters.emplace_back(dfg_da::hypothesis::Hypotheses(std::move(hs)));
            start += cluster_card;
        }

        return hypos_in_clusters;
    }

    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs)
    {
        // Get pointer to engine
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

        // Get array factory
        matlab::data::ArrayFactory factory;

        if (inputs.size() != 8)
        {
            matlabPtr->feval(u"error",
                             0,
                             std::vector<matlab::data::Array>({factory.createScalar("Needs to accept 8 input arguments: lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard)")}));
        }

        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        // Check reward matrix
        if (inputs[1].getType() != matlab::data::ArrayType::DOUBLE ||
            inputs[1].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE ||
            inputs[1].getNumberOfElements() > 1)
        {
            matlabPtr->feval(u"error",
                             0,
                             std::vector<matlab::data::Array>({factory.createScalar("Second argument, num_tracks, must be a scalar")}));
        }

        if (inputs[2].getType() != matlab::data::ArrayType::DOUBLE ||
            inputs[2].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE ||
            inputs[2].getNumberOfElements() > 1)
        {
            matlabPtr->feval(u"error",
                             0,
                             std::vector<matlab::data::Array>({factory.createScalar("Third argument, num_measurements, must be a scalar")}));
        }

        size_t num_tracks = inputs[1][0];
        size_t num_measurements = inputs[2][0];

        {
            auto dims = inputs[0].getDimensions();
            if (inputs[0].getType() != matlab::data::ArrayType::DOUBLE ||
                inputs[0].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE ||
                dims.size() != 2 ||
                dims[0] != num_tracks + num_measurements ||
                dims[1] != num_tracks + num_measurements ||
                dims[0] != dims[1] ||
                inputs[0].getMemoryLayout() != matlab::data::MemoryLayout::COLUMN_MAJOR)
            {
                matlabPtr->feval(u"error",
                                 0,
                                 std::vector<matlab::data::Array>({factory.createScalar("First input, gainMatPostC, must be square column-major matrix with double elements and rows and columns both equal to num_tracks + num_measurements")}));
            }
        }

        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        {
            auto dims = inputs[3].getDimensions();
            if ((dims.size() != 1 && dims.size() != 2) ||
                (dims.size() == 2 && dims[0] != 1) ||
                inputs[3].getType() != matlab::data::ArrayType::DOUBLE ||
                inputs[3].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE)
            {
                matlabPtr->feval(u"error",
                                 0,
                                 std::vector<matlab::data::Array>({factory.createScalar("Fourth input, hypos, must be vector or row vector of doubles")}));
            }
        }

        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        {
            auto dims = inputs[4].getDimensions();
            if ((dims.size() != 1 && dims.size() != 2) ||
                (dims.size() == 2 && dims[0] != 1) ||
                inputs[4].getType() != matlab::data::ArrayType::DOUBLE ||
                inputs[4].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE)
            {
                matlabPtr->feval(u"error",
                                 0,
                                 std::vector<matlab::data::Array>({factory.createScalar("Fifth input, hyposCard, must be vector or row vector of doubles")}));
            }
        }

        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        {
            auto dims = inputs[5].getDimensions();
            if ((dims.size() != 1 && dims.size() != 2) ||
                (dims.size() == 2 && dims[0] != 1) ||
                inputs[5].getType() != matlab::data::ArrayType::DOUBLE ||
                inputs[5].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE)
            {
                matlabPtr->feval(u"error",
                                 0,
                                 std::vector<matlab::data::Array>({factory.createScalar("Sixth input, probLogHypos, must be vector or row vector of doubles")}));
            }
        }

        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        {
            auto dims = inputs[6].getDimensions();
            if ((dims.size() != 1 && dims.size() != 2) ||
                (dims.size() == 2 && dims[0] != 1) ||
                inputs[6].getType() != matlab::data::ArrayType::DOUBLE ||
                inputs[6].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE)
            {
                matlabPtr->feval(u"error",
                                 0,
                                 std::vector<matlab::data::Array>({factory.createScalar("Seventh input, clusters, must be vector or row vector of doubles")}));
            }
        }

        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        {
            auto dims = inputs[7].getDimensions();
            if ((dims.size() != 1 && dims.size() != 2) ||
                (dims.size() == 2 && dims[0] != 1) ||
                inputs[7].getType() != matlab::data::ArrayType::DOUBLE ||
                inputs[7].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE)
            {
                matlabPtr->feval(u"error",
                                 0,
                                 std::vector<matlab::data::Array>({factory.createScalar("Eighth input, clustersCard, must be vector or row vector of doubles")}));
            }
        }

        // lbp_mex(0: gainMatPostC, 1: num_tracks, 2: num_measurements, 3: hypos, 4: hyposCard, 5: probLogHypos, 6: clusters, 7: clustersCard);
        if ((inputs[4].getNumberOfElements() != inputs[5].getNumberOfElements()) ||
            (inputs[5].getNumberOfElements() != inputs[6].getNumberOfElements()) ||
            (inputs[4].getNumberOfElements() != inputs[6].getNumberOfElements()))
        {
            matlabPtr->feval(u"error",
                             0,
                             std::vector<matlab::data::Array>({factory.createScalar("Inputs hyposCard, probLogHypos and clusters must have equal lengths (the total number of hypotheses over all clusters)")}));
        }
    }
};