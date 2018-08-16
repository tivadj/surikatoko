#include <string>
#include <cmath> // std::sqrt
#include <algorithm> // std::clamp
#include <glog/logging.h>
#include <Eigen/Cholesky>
#include <gsl/span>
#include "suriko/approx-alg.h"
#include "suriko/obs-geom.h"
#include "suriko/quat.h"

namespace suriko
{
auto QuatFromRotationMatNoRChecks(const Eigen::Matrix<Scalar, 3, 3>& R, gsl::span<Scalar> quat) -> void
{
    // source : "A Recipe on the Parameterization of Rotation Matrices", Terzakis, 2012
    // formula 24
    //quat = np.zeros(4, dtype = type(R(0, 0]))
    if (R(1, 1) > -R(2, 2) && R(0, 0) > -R(1, 1) && R(0, 0) > -R(2, 2))
    {
        Scalar sum = 1 + R(0, 0) + R(1, 1) + R(2, 2);
        SRK_ASSERT(sum >= 0);
        Scalar root = std::sqrt(sum);
        quat[0] = 0.5 * root;
        quat[1] = 0.5 * (R(2, 1) - R(1, 2)) / root;
        quat[2] = 0.5 * (R(0, 2) - R(2, 0)) / root;
        quat[3] = 0.5 * (R(1, 0) - R(0, 1)) / root;
    }
    else if (R(1, 1) < -R(2, 2) && R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2))
    {
        Scalar sum = 1 + R(0, 0) - R(1, 1) - R(2, 2);
        SRK_ASSERT(sum >= 0);
        Scalar root = std::sqrt(sum);
        quat[0] = 0.5 * (R(2, 1) - R(1, 2)) / root;
        quat[1] = 0.5 * root;
        quat[2] = 0.5 * (R(1, 0) + R(0, 1)) / root;
        quat[3] = 0.5 * (R(2, 0) + R(0, 2)) / root;
    }
    else if (R(1, 1) > R(2, 2) && R(0, 0) < R(1, 1) && R(0, 0) < -R(2, 2))
    {
        Scalar sum = 1 - R(0, 0) + R(1, 1) - R(2, 2);
        SRK_ASSERT(sum >= 0);
        Scalar root = std::sqrt(sum);
        quat[0] = 0.5 * (R(0, 2) - R(2, 0)) / root;
        quat[1] = 0.5 * (R(1, 0) + R(0, 1)) / root;
        quat[2] = 0.5 * root;
        quat[3] = 0.5 * (R(2, 1) + R(1, 2)) / root;
    }
    else if (R(1, 1) < R(2, 2) && R(0, 0) < -R(1, 1) && R(0, 0) < -R(2, 2))
    {
        Scalar sum = 1 - R(0, 0) - R(1, 1) + R(2, 2);
        SRK_ASSERT(sum >= 0);
        Scalar root = std::sqrt(sum);
        quat[0] = 0.5 * (R(1, 0) - R(0, 1)) / root;
        quat[1] = 0.5 * (R(2, 0) + R(0, 2)) / root;
        quat[2] = 0.5 * (R(2, 1) + R(1, 2)) / root;
        quat[3] = 0.5 * root;
    }
    else AssertFalse();
}

auto QuatFromRotationMat(const Eigen::Matrix<Scalar, 3, 3>& R, gsl::span<Scalar> quat, std::string* err_msg) -> bool
{
    bool op = IsSpecialOrthogonal(R, err_msg);
    if (!op)
        return false;
    
    QuatFromRotationMatNoRChecks(R, quat);
    
    return true;
}

auto RotMatFromQuat(gsl::span<const Scalar> q, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> rot_mat) -> void
{
    Eigen::Matrix<Scalar, 3, 3>& R = *rot_mat;
    // source : "A Recipe on the Parameterization of Rotation Matrices", Terzakis, 2012
    // formula 9
    R(0, 0) = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];
    R(0, 1) = 2 * (q[1] * q[2] - q[0] * q[3]);
    R(0, 2) = 2 * (q[1] * q[3] + q[0] * q[2]);

    R(1, 0) = 2 * (q[1] * q[2] + q[0] * q[3]);
    R(1, 1) = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3];
    R(1, 2) = 2 * (q[2] * q[3] - q[0] * q[1]);

    R(2, 0) = 2 * (q[1] * q[3] - q[0] * q[2]);
    R(2, 1) = 2 * (q[2] * q[3] + q[0] * q[1]);
    R(2, 2) = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];
}

auto RotMat(const Eigen::Matrix<Scalar, 4, 1>& quat)->Eigen::Matrix<Scalar, 3, 3>
{
    Eigen::Matrix<Scalar, 3, 3> result;
    RotMatFromQuat(gsl::make_span(quat.data(), 4), &result);
    return result;
}

auto QuatFromAxisAngle(gsl::span<const Scalar> axis_ang, gsl::span<Scalar> quat) -> void
{
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> axis(axis_ang.data());
    Scalar ang = axis.norm();
    if (IsClose(0, ang))
    {
        quat[0] = 1;
        quat[1] = 0;
        quat[2] = 0;
        quat[3] = 0;
    }
    else
    {
        quat[0] = std::cos(ang / 2);

        Scalar sin_ang2 = std::sin(ang / 2);
        quat[1] = sin_ang2 * axis_ang[0] / ang;
        quat[2] = sin_ang2 * axis_ang[1] / ang;
        quat[3] = sin_ang2 * axis_ang[2] / ang;
    }
}

auto QuatFromAxisAngle(const Eigen::Matrix<Scalar, 3, 1>& axis_ang, Eigen::Matrix<Scalar, 4, 1>* quat) -> void
{
    QuatFromAxisAngle(gsl::make_span<const Scalar>(axis_ang.data(), 3), gsl::make_span<Scalar>(quat->data(), 4));
}

auto AxisPlusAngleFromQuat(gsl::span<const Scalar> q, gsl::span<Scalar> dir, Scalar* angle) -> void
{
    bool zero_ang = IsClose(1.0, q[0]);
    if (zero_ang)
    {
        *angle = 0;
        dir[0] = 0;
        dir[1] = 0;
        dir[2] = 0;
    }
    else
    {
        Scalar ang = 2 * std::acos(q[0]);
        *angle = ang;

        Scalar sin_ang2 = std::sin(ang / 2);
        dir[0] = q[1] / sin_ang2;
        dir[1] = q[2] / sin_ang2;
        dir[2] = q[3] / sin_ang2;
    }
}

auto AxisAngleFromQuat(gsl::span<const Scalar> q, gsl::span<Scalar> axis_angle) -> void
{
    Scalar ang;
    AxisPlusAngleFromQuat(q, axis_angle, &ang);
    axis_angle[0] *= ang;
    axis_angle[1] *= ang;
    axis_angle[2] *= ang;
}

auto AxisAngleFromQuat(const Eigen::Matrix<Scalar, 4, 1>& q, Eigen::Matrix<Scalar, 3, 1>* axis_angle) -> void
{
    AxisAngleFromQuat(gsl::make_span<const Scalar>(q.data(),4), gsl::make_span<Scalar>(axis_angle->data(),3));
}

auto QuatMult(const Eigen::Matrix<Scalar, 4, 1>& a, const Eigen::Matrix<Scalar, 4, 1>& b, Eigen::Matrix<Scalar, 4, 1>* result) -> void
{
    Eigen::Matrix<Scalar, 4, 4> a_mat;
    a_mat <<
        a[0], -a[1], -a[2], -a[3],
        a[1],  a[0], -a[3],  a[2],
        a[2],  a[3],  a[0], -a[1],
        a[3], -a[2],  a[1],  a[0];

    *result = a_mat * b;
}

auto QuatInverse(const Eigen::Matrix<Scalar, 4, 1>& a)->Eigen::Matrix<Scalar, 4, 1>
{
    return Eigen::Matrix<Scalar, 4, 1>(a[0], -a[1], -a[2], -a[3]);
}
}