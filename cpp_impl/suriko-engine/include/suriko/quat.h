﻿#pragma once
#include <string>
#include <gsl/span>
#include <Eigen/Dense> // Eigen::Matrix
#include "suriko/rt-config.h"

namespace suriko
{
inline auto NewQuat(Scalar qx, Scalar qy, Scalar qz, Scalar qw)->Eigen::Matrix<Scalar, 4, 1> {
    return Eigen::Matrix<Scalar, 4, 1> {qw, qx, qy, qz};
}

/// Specifies the order in which to put quaternion's components into sequence.
enum class QuatLayout
{
    XyzW,  // [x y z w], used by TUM's datasets
    WXyz   // [w x y z]
};

inline auto NewQuatFrom(gsl::span<Scalar> q, QuatLayout layout)->std::array<Scalar, 4> {
    auto qx = q[layout == QuatLayout::XyzW ? 0 : 1];
    auto qy = q[layout == QuatLayout::XyzW ? 1 : 2];
    auto qz = q[layout == QuatLayout::XyzW ? 2 : 3];
    auto qw = q[layout == QuatLayout::XyzW ? 3 : 0];
    return std::array<Scalar, 4> {qw, qx, qy, qz};
}

/// Converts from rotation matrix (SO3) to quaternion. Doesn't check R.
auto QuatFromRotationMatNoRChecks(const Eigen::Matrix<Scalar, 3, 3>& R, gsl::span<Scalar> q) -> void;
    
/// Converts from rotation matrix (SO3) to quaternion.
/// param R : [3x3] rotation matrix
/// return : quaternion corresponding to a given rotation matrix
[[nodiscard]]
auto QuatFromRotationMat(const Eigen::Matrix<Scalar, 3, 3>& R, gsl::span<Scalar> q, std::string* err_msg = nullptr) -> bool;

/// Constructs rotation matrix (SO3) corresponding to given quaternion.
/// param q : quaternion, 4 - element vector
/// return : rotation matrix, [3x3]
auto RotMatFromQuat(gsl::span<const Scalar> q, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> R) -> void;

auto RotMat(const Eigen::Matrix<Scalar, 4, 1>& quat)->Eigen::Matrix<Scalar, 3, 3>;
auto RotMat(gsl::span<Scalar> quat)->Eigen::Matrix<Scalar, 3, 3>;

/// Converts from axis-angle representation of a rotation (SO3) to quaternion.
/// param axis_ang : 3 - element vector of angle*rot_axis
/// return : quaternion corresponding to a given axis - angle
auto QuatFromAxisAngle(gsl::span<const Scalar> axis_ang, gsl::span<Scalar> quat) -> void;
auto QuatFromAxisAngle(const Eigen::Matrix<Scalar, 3, 1>& axis_ang, Eigen::Matrix<Scalar, 4, 1>* quat) -> void;

/// Converts from quaternion to (axis,angle)
auto AxisPlusAngleFromQuat(gsl::span<const Scalar> q, gsl::span<Scalar> dir, Scalar* angle) -> void;

/// axis-angle [w1,w2,w3] -> quaternion [q0,q1,q2,q3]
auto AxisAngleFromQuat(gsl::span<const Scalar> q, gsl::span<Scalar> axis_angle) -> void;
auto AxisAngleFromQuat(const Eigen::Matrix<Scalar, 4, 1>& q, Eigen::Matrix<Scalar, 3, 1>* axis_angle) -> void;

/// Multiply two quaternions.
auto QuatMult(const Eigen::Matrix<Scalar, 4, 1>& a, const Eigen::Matrix<Scalar, 4, 1>& b, Eigen::Matrix<Scalar, 4, 1>* result) -> void;

auto QuatInverse(const Eigen::Matrix<Scalar, 4, 1>& a) -> Eigen::Matrix<Scalar, 4, 1>;
}