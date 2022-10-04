#include <matioCpp/matioCpp.h>
#include <Eigen/Core>
#include <Eigen/Dense>


int main()
{
    matioCpp::File input("./data/at612/priorLikelihood612.mat");
    // matioCpp::Vector<double> vector = input.read("hypos").asVector<double>();
    // Eigen::VectorXd eigenVec = matioCpp::to_eigen(vector);
    // std::cout << eigenVec << "\n";

    Eigen::MatrixXd m(3, 2);
    m.reshaped() = Eigen::VectorXd::LinSpaced(3*2, 1, 3*2);

    std::cout << m << "\n";

    auto matioMatrix = matioCpp::make_variable("mat", m);

    matioCpp::File file = matioCpp::File::Create("test.mat"); //If the file already exists, use the same cnstructor as the example above
    file.write(matioMatrix);

    auto matioMatrix2 = matioCpp::make_variable("mat2", m);
    file.write(matioMatrix2);
}