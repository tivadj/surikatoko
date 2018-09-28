#pragma once
#include <Eigen/Dense>
#include "suriko/rt-config.h"

namespace suriko
{
class GaussRandomVar
{
    using EigenDynVec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using EigenDynMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    EigenDynVec mean_;
    EigenDynMat transform_;
    std::normal_distribution<> dist_;
    EigenDynVec rand_basis_;
public:
    GaussRandomVar(const EigenDynMat& covar)
        : GaussRandomVar(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    GaussRandomVar(const EigenDynMat& covar, Eigen::VectorXd const& mean)
        : mean_(mean)
    {
        rand_basis_.resize(mean.size(), Eigen::NoChange);

        Eigen::SelfAdjointEigenSolver<EigenDynMat> eigenSolver(covar);
        SRK_ASSERT(eigenSolver.info() == Eigen::Success);
        transform_ = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    template <typename OutMat>
    void NewSample(std::mt19937* gen, OutMat* result)
    {
        RandomizeBasis(gen);
        *result = mean_ + transform_ * rand_basis_;
    }
private:
    void RandomizeBasis(std::mt19937* gen)
    {
        for (EigenDynVec::Index i = 0; i < rand_basis_.size(); ++i)
            rand_basis_[i] = dist_(*gen);
    }
};

template <typename _Scalar, int Dim>
void CalcCovarMat(const std::vector <Eigen::Matrix<_Scalar, Dim, 1>>& samples, Eigen::Matrix<_Scalar, Dim, Dim>* covar_mat)
{
    if (samples.empty())
        return;
    size_t size = samples[0].rows();

    Eigen::Matrix<_Scalar, Dim, 1> samples_mean;
    samples_mean.setZero(size);

    Eigen::Matrix<_Scalar, Dim, Dim> xy_mean;
    xy_mean.setZero(size, size);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const auto& s = samples[i];
        samples_mean += s;

        // store only upper triangle of a symmetric matrix
        for (int row = 0; row < covar_mat->rows(); ++row)
            for (int col = row; col < covar_mat->cols(); ++col)
            {
                xy_mean(row, col) += s[row] * s[col];
            }
    }
    samples_mean *= (1.0 / samples.size());
    xy_mean *= (1.0 / samples.size());

    //
    covar_mat->setZero();
    for (int row = 0; row < covar_mat->rows(); ++row)
        for (int col = row; col < covar_mat->cols(); ++col)
        {
            Scalar var = xy_mean(row, col) - samples_mean[row] * samples_mean[col];
            (*covar_mat)(row, col) = var;
            (*covar_mat)(col, row) = var; // store symmetric part
        }
}

/// For y=f(x), finds SigY, propagating the uncertainty SigX.
/// F :: XMean->YMean
/// XSize is the size of X.
template <typename XMean, typename XCovar, typename F, typename YCovar>
void PropagateUncertaintyUsingSimulation(const XMean& x_mean, const XCovar& x_covar,
    F propag_fun,
    size_t gen_samples_count, std::mt19937* gen,
    YCovar* y_covar)
{
    suriko::GaussRandomVar x_rand(x_covar, x_mean);

    static constexpr size_t kYSize = YCovar::RowsAtCompileTime;

    typedef Eigen::Matrix<Scalar, kYSize, 1> YMean;
    std::vector<YMean> samples(gen_samples_count);

    std::generate_n(samples.begin(), gen_samples_count, [&x_rand, &propag_fun, gen]()
    {
        XMean samp;
        x_rand.NewSample(gen, &samp);

        YMean out_y;
        propag_fun(samp, &out_y);

        return out_y;
    });

    CalcCovarMat<Scalar, kYSize>(samples, y_covar);
}
}