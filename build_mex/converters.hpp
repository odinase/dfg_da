#ifndef CONVERTERS_H
#define CONVERTERS_H


#include "mex.hpp"
#include "dfg_da/hypothesis.h"
#include <Eigen/Core>
#include <Eigen/Dense>

// using namespace matlab::data;
// using matlab::mex::ArgumentList;

namespace dfg_da
{
    namespace mex
    {
        
        hypothesis::Hypotheses hypotheses_conversion(matlab::mex::ArgumentList inputs);
        Eigen::MatrixXd reward_matrix_conversion(matlab::mex::ArgumentList inputs);

    } // namespace mex
} // namespace dfg_da

#endif // CONVERTERS_H