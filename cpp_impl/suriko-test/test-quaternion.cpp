#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "suriko/obs-geom.h"
#include "suriko/quat.h"
#include "suriko/rt-config.h"

namespace suriko_test
{
using namespace suriko;

class QuaternionTest : public testing::Test
{
public:
	Scalar atol = (Scalar)1e-5;
};

TEST_F(QuaternionTest, AxisAngleToQuat)
{
    Eigen::Matrix<Scalar, 3, 1> v(1, 2, 3);

    std::array<Scalar, 4> q{};
    QuatFromAxisAngle(gsl::make_span(v.data(), 3), q);

    Scalar ang = -1;
    Eigen::Matrix<Scalar, 3, 1> v_back;
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<double>, const Eigen::Matrix<double, 3, 1, 0>, const Eigen::Matrix<double, 3, 1, 0>> ff = v_back - v_back;
    AxisPlusAngleFromQuat(q, gsl::make_span(v_back.data(), 3), &ang);
    v_back *= ang;

    Scalar diff_value = (v_back - v).norm();
    EXPECT_NEAR(0, diff_value, atol);
}

template <typename _Scalar>
struct OpBinaryDifference
{
    typedef _Scalar Scalar;
    Scalar operator()(Scalar a, Scalar b) const
    {
        return a - b;
    }
};

template <typename BinOpT>
class GslSpanBinaryOp
{
    using Scalar = typename BinOpT::Scalar;
    
    gsl::span<const Scalar> data1_;
    gsl::span<const Scalar> data2_;
public:
    GslSpanBinaryOp(gsl::span<const Scalar> data1, gsl::span<const Scalar> data2) : data1_(data1), data2_(data2) {}

    Scalar Norm() const
    {
        Scalar sum = 0;
        BinOpT bin_op{};
        for (typename gsl::span<const Scalar>::index_type i = 0; i < data1_.size(); ++i)
        {
            Scalar value = bin_op.operator()(data1_[i], data2_[i]);
            sum += value;
        }
        return std::sqrt(sum);
    }
};

template <typename _Scalar>
GslSpanBinaryOp<OpBinaryDifference<_Scalar>> operator -(gsl::span<const _Scalar> a, gsl::span<const _Scalar> b)
{
    return GslSpanBinaryOp<OpBinaryDifference<_Scalar>>(a, b);
}

TEST_F(QuaternionTest, ZeroAxisAngleToQuatSpan)
{
    std::array<Scalar, 3> v = { 0, 0, 0 };

    std::array<Scalar, 4> q{};
    QuatFromAxisAngle(v, q);

    std::array<Scalar, 3> v_back = {999, 999, 999};
    AxisAngleFromQuat(q, v_back);

    //GslSpanBinaryOp<BinaryOpDifference<Scalar>> diff_value = (v_back - v).Norm();
    GslSpanBinaryOp<OpBinaryDifference<Scalar>> diff_value_op = operator-(gsl::make_span<const Scalar>(v_back.data(),3), gsl::make_span<const Scalar>(v.data(),3));
    Scalar diff_value = diff_value_op.Norm();
    EXPECT_NEAR(0, diff_value, atol);
}
    
TEST_F(QuaternionTest, ZeroAxisAngleToQuat)
{
    Eigen::Matrix<Scalar, 3, 1> v = { 0, 0, 0 };

    Eigen::Matrix<Scalar, 4, 1> q{};
    QuatFromAxisAngle(v, &q);

    Eigen::Matrix<Scalar, 3, 1> v_back = {999, 999, 999};
    AxisAngleFromQuat(q, &v_back);

    Scalar diff_value = (v_back - v).norm();
    EXPECT_NEAR(0, diff_value, atol);
}

TEST_F(QuaternionTest, RotMatToQuat)
{
	Eigen::Matrix<Scalar, 3, 1> v(1,2,3);
	Eigen::Matrix<Scalar, 3, 3> R;
    bool op = RotMatFromAxisAngle(v, &R);
    ASSERT_TRUE(op);

    std::array<Scalar, 4> q{};
    op = QuatFromRotationMat(R, q);
    EXPECT_TRUE(op);

    Eigen::Matrix<Scalar, 3, 3> R_back;
    RotMatFromQuat(q, &R_back);
    Scalar diff_value = (R_back - R).norm();
    EXPECT_NEAR(0, diff_value, atol);
}
}
