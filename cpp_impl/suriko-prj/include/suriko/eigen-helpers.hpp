#pragma once
#include <Eigen/Dense>

namespace suriko {

template <typename Scalar>
suriko::Point3<Scalar> operator*(const Eigen::Matrix<Scalar, 3, 3>& m, const suriko::Point3<Scalar>& p)
{
    Eigen::Matrix<Scalar, 3, 1> vec =  m.col(0) * p[0] + m.col(1) * p[1] + m.col(2) * p[2];
    return suriko::Point3<Scalar>(vec(0), vec(1), vec(2));
}

}