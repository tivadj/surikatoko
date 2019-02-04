#include <Eigen/Dense> // Eigen::Matrix
#include "suriko/obs-geom.h"
#include "suriko/rt-config.h"

namespace suriko
{
namespace
{
Scalar Power(Scalar x, int n) { return std::pow(x, n); }
}

void ProjectEllipsoidOnCamera_FillEllipseOutlineDirectionsMat3x3(
    const RotatedEllipsoid3D& rot_ellip,
    const Eigen::Matrix<Scalar, 3, 1>& eye,
    Eigen::Matrix<Scalar,3,3>* M)
{
    auto a = rot_ellip.semi_axes[0];
    auto b = rot_ellip.semi_axes[1];
    auto c = rot_ellip.semi_axes[2];
    auto r11 = rot_ellip.world_from_ellipse.R(0,0);
    auto r12 = rot_ellip.world_from_ellipse.R(0,1);
    auto r13 = rot_ellip.world_from_ellipse.R(0,2);
    auto r21 = rot_ellip.world_from_ellipse.R(1,0);
    auto r22 = rot_ellip.world_from_ellipse.R(1,1);
    auto r23 = rot_ellip.world_from_ellipse.R(1,2);
    auto r31 = rot_ellip.world_from_ellipse.R(2,0);
    auto r32 = rot_ellip.world_from_ellipse.R(2,1);
    auto r33 = rot_ellip.world_from_ellipse.R(2,2);
    auto t1 = rot_ellip.world_from_ellipse.T[0];
    auto t2 = rot_ellip.world_from_ellipse.T[1];
    auto t3 = rot_ellip.world_from_ellipse.T[2];
    auto e1 = eye[0];
    auto e2 = eye[1];
    auto e3 = eye[2];

// the expressions below are extracted from Mathematica output, see EllipseCurvedOutlineOfRotatedEllipseProjectedOnCamera.nb

Scalar coef_d0d0 = 
(4 * (-(Power(c, 2)*Power(e2*r12*r21 - e2 * r11*r22 + e3 * r12*r31 - e3 * r11*r32 -
r12 * r21*t2 + r11 * r22*t2 - r12 * r31*t3 + r11 * r32*t3, 2)) +
Power(b, 2)*(Power(c, 2)*Power(r11, 2) + Power(a, 2)*Power(r13, 2) -
Power(e2*r13*r21 - e2 * r11*r23 + e3 * r13*r31 - e3 * r11*r33 - r13 * r21*t2 +
r11 * r23*t2 - r13 * r31*t3 + r11 * r33*t3, 2)) +
Power(a, 2)*(Power(c, 2)*Power(r12, 2) -
Power(e2*r13*r22 - e2 * r12*r23 + e3 * r13*r32 - e3 * r12*r33 - r13 * r22*t2 +
r12 * r23*t2 - r13 * r32*t3 + r12 * r33*t3, 2)))) /
(Power(a, 2)*Power(b, 2)*Power(c, 2));

Scalar coef_d0d1 =
(8 * (Power(a, 2)*(e2*r13*r22 - e2 * r12*r23 + e3 * r13*r32 - e3 * r12*r33 - r13 * r22*t2 +
r12 * r23*t2 - r13 * r32*t3 + r12 * r33*t3)*
(e1*r13*r22 - e1 * r12*r23 - e3 * r23*r32 + e3 * r22*r33 - r13 * r22*t1 +
r12 * r23*t1 + r23 * r32*t3 - r22 * r33*t3) +
Power(c, 2)*(Power(a, 2)*r12*r22 +
e1 * (r12*r21 - r11 * r22)*(e2*r12*r21 - e2 * r11*r22 + e3 * r12*r31 -
e3 * r11*r32 - r12 * r21*t2 + r11 * r22*t2 - r12 * r31*t3 + r11 * r32*t3) +
(e2*r12*r21 - e2 * r11*r22 + e3 * r12*r31 - e3 * r11*r32 - r12 * r21*t2 +
r11 * r22*t2 - r12 * r31*t3 + r11 * r32*t3)*
(-(e3*r22*r31) + e3 * r21*r32 - r12 * r21*t1 + r11 * r22*t1 + r22 * r31*t3 -
r21 * r32*t3)) + Power(b, 2)*
(Power(c, 2)*r11*r21 + Power(a, 2)*r13*r23 - e2 * e3*r13*r21*r23*r31 +
e2 * e3*r11*Power(r23, 2)*r31 - Power(e3, 2)*r13*r23*Power(r31, 2) +
e2 * e3*r13*Power(r21, 2)*r33 - e2 * e3*r11*r21*r23*r33 +
Power(e3, 2)*r13*r21*r31*r33 + Power(e3, 2)*r11*r23*r31*r33 -
Power(e3, 2)*r11*r21*Power(r33, 2) - e2 * Power(r13, 2)*Power(r21, 2)*t1 +
2 * e2*r11*r13*r21*r23*t1 - e2 * Power(r11, 2)*Power(r23, 2)*t1 -
e3 * Power(r13, 2)*r21*r31*t1 + e3 * r11*r13*r23*r31*t1 +
e3 * r11*r13*r21*r33*t1 - e3 * Power(r11, 2)*r23*r33*t1 +
e3 * r13*r21*r23*r31*t2 - e3 * r11*Power(r23, 2)*r31*t2 -
e3 * r13*Power(r21, 2)*r33*t2 + e3 * r11*r21*r23*r33*t2 +
Power(r13, 2)*Power(r21, 2)*t1*t2 - 2 * r11*r13*r21*r23*t1*t2 +
Power(r11, 2)*Power(r23, 2)*t1*t2 + e2 * r13*r21*r23*r31*t3 -
e2 * r11*Power(r23, 2)*r31*t3 + 2 * e3*r13*r23*Power(r31, 2)*t3 -
e2 * r13*Power(r21, 2)*r33*t3 + e2 * r11*r21*r23*r33*t3 -
2 * e3*r13*r21*r31*r33*t3 - 2 * e3*r11*r23*r31*r33*t3 +
2 * e3*r11*r21*Power(r33, 2)*t3 + Power(r13, 2)*r21*r31*t1*t3 -
r11 * r13*r23*r31*t1*t3 - r11 * r13*r21*r33*t1*t3 +
Power(r11, 2)*r23*r33*t1*t3 - r13 * r21*r23*r31*t2*t3 +
r11 * Power(r23, 2)*r31*t2*t3 + r13 * Power(r21, 2)*r33*t2*t3 -
r11 * r21*r23*r33*t2*t3 - r13 * r23*Power(r31, 2)*Power(t3, 2) +
r13 * r21*r31*r33*Power(t3, 2) + r11 * r23*r31*r33*Power(t3, 2) -
r11 * r21*Power(r33, 2)*Power(t3, 2) +
e1 * (r13*r21 - r11 * r23)*(e2*r13*r21 - e2 * r11*r23 + e3 * r13*r31 -
e3 * r11*r33 - r13 * r21*t2 + r11 * r23*t2 - r13 * r31*t3 + r11 * r33*t3)))) /
(Power(a, 2)*Power(b, 2)*Power(c, 2));

Scalar coef_d0d2 =
(8 * (Power(a, 2)*(e1*r13*r32 + e2 * r23*r32 - e1 * r12*r33 - e2 * r22*r33 - r13 * r32*t1 +
r12 * r33*t1 - r23 * r32*t2 + r22 * r33*t2)*
(e2*r13*r22 - e2 * r12*r23 + e3 * r13*r32 - e3 * r12*r33 - r13 * r22*t2 +
r12 * r23*t2 - r13 * r32*t3 + r12 * r33*t3) +
Power(b, 2)*(Power(c, 2)*r11*r31 + Power(e2, 2)*r13*r21*r23*r31 -
Power(e2, 2)*r11*Power(r23, 2)*r31 + e2 * e3*r13*r23*Power(r31, 2) +
Power(a, 2)*r13*r33 - Power(e2, 2)*r13*Power(r21, 2)*r33 +
Power(e2, 2)*r11*r21*r23*r33 - e2 * e3*r13*r21*r31*r33 -
e2 * e3*r11*r23*r31*r33 + e2 * e3*r11*r21*Power(r33, 2) -
e2 * Power(r13, 2)*r21*r31*t1 + e2 * r11*r13*r23*r31*t1 -
e3 * Power(r13, 2)*Power(r31, 2)*t1 + e2 * r11*r13*r21*r33*t1 -
e2 * Power(r11, 2)*r23*r33*t1 + 2 * e3*r11*r13*r31*r33*t1 -
e3 * Power(r11, 2)*Power(r33, 2)*t1 - 2 * e2*r13*r21*r23*r31*t2 +
2 * e2*r11*Power(r23, 2)*r31*t2 - e3 * r13*r23*Power(r31, 2)*t2 +
2 * e2*r13*Power(r21, 2)*r33*t2 - 2 * e2*r11*r21*r23*r33*t2 +
e3 * r13*r21*r31*r33*t2 + e3 * r11*r23*r31*r33*t2 -
e3 * r11*r21*Power(r33, 2)*t2 + Power(r13, 2)*r21*r31*t1*t2 -
r11 * r13*r23*r31*t1*t2 - r11 * r13*r21*r33*t1*t2 +
Power(r11, 2)*r23*r33*t1*t2 + r13 * r21*r23*r31*Power(t2, 2) -
r11 * Power(r23, 2)*r31*Power(t2, 2) - r13 * Power(r21, 2)*r33*Power(t2, 2) +
r11 * r21*r23*r33*Power(t2, 2) - e2 * r13*r23*Power(r31, 2)*t3 +
e2 * r13*r21*r31*r33*t3 + e2 * r11*r23*r31*r33*t3 -
e2 * r11*r21*Power(r33, 2)*t3 + Power(r13, 2)*Power(r31, 2)*t1*t3 -
2 * r11*r13*r31*r33*t1*t3 + Power(r11, 2)*Power(r33, 2)*t1*t3 +
r13 * r23*Power(r31, 2)*t2*t3 - r13 * r21*r31*r33*t2*t3 -
r11 * r23*r31*r33*t2*t3 + r11 * r21*Power(r33, 2)*t2*t3 +
e1 * (r13*r31 - r11 * r33)*(e2*r13*r21 - e2 * r11*r23 + e3 * r13*r31 -
e3 * r11*r33 - r13 * r21*t2 + r11 * r23*t2 - r13 * r31*t3 + r11 * r33*t3)) +
Power(c, 2)*(Power(a, 2)*r12*r32 -
Power(e2, 2)*(r12*r21 - r11 * r22)*(-(r22*r31) + r21 * r32) -
e3 * Power(r12, 2)*Power(r31, 2)*t1 + 2 * e3*r11*r12*r31*r32*t1 -
e3 * Power(r11, 2)*Power(r32, 2)*t1 - e3 * r12*r22*Power(r31, 2)*t2 +
e3 * r12*r21*r31*r32*t2 + e3 * r11*r22*r31*r32*t2 -
e3 * r11*r21*Power(r32, 2)*t2 + Power(r12, 2)*r21*r31*t1*t2 -
r11 * r12*r22*r31*t1*t2 - r11 * r12*r21*r32*t1*t2 +
Power(r11, 2)*r22*r32*t1*t2 + r12 * r21*r22*r31*Power(t2, 2) -
r11 * Power(r22, 2)*r31*Power(t2, 2) - r12 * Power(r21, 2)*r32*Power(t2, 2) +
r11 * r21*r22*r32*Power(t2, 2) + Power(r12, 2)*Power(r31, 2)*t1*t3 -
2 * r11*r12*r31*r32*t1*t3 + Power(r11, 2)*Power(r32, 2)*t1*t3 +
r12 * r22*Power(r31, 2)*t2*t3 - r12 * r21*r31*r32*t2*t3 -
r11 * r22*r31*r32*t2*t3 + r11 * r21*Power(r32, 2)*t2*t3 +
e1 * (r12*r31 - r11 * r32)*(e2*r12*r21 - e2 * r11*r22 + e3 * r12*r31 -
e3 * r11*r32 - r12 * r21*t2 + r11 * r22*t2 - r12 * r31*t3 + r11 * r32*t3) +
e2 * (e3*(r12*r31 - r11 * r32)*(r22*r31 - r21 * r32) -
Power(r12, 2)*r21*r31*t1 - Power(r11, 2)*r22*r32*t1 +
r11 * r12*(r22*r31 + r21 * r32)*t1 +
r12 * (-(r22*r31) + r21 * r32)*(2 * r21*t2 + r31 * t3) +
r11 * (r22*r31 - r21 * r32)*(2 * r22*t2 + r32 * t3))))) /
(Power(a, 2)*Power(b, 2)*Power(c, 2));

Scalar coef_d1d1 =
(4 * (Power(b, 2)*Power(c, 2)*Power(r21, 2) -
Power(c, 2)*Power(e1, 2)*Power(r12, 2)*Power(r21, 2) -
Power(b, 2)*Power(e1, 2)*Power(r13, 2)*Power(r21, 2) +
2 * Power(c, 2)*Power(e1, 2)*r11*r12*r21*r22 +
Power(a, 2)*Power(c, 2)*Power(r22, 2) -
Power(c, 2)*Power(e1, 2)*Power(r11, 2)*Power(r22, 2) -
Power(a, 2)*Power(e1, 2)*Power(r13, 2)*Power(r22, 2) +
2 * Power(b, 2)*Power(e1, 2)*r11*r13*r21*r23 +
2 * Power(a, 2)*Power(e1, 2)*r12*r13*r22*r23 +
Power(a, 2)*Power(b, 2)*Power(r23, 2) -
Power(b, 2)*Power(e1, 2)*Power(r11, 2)*Power(r23, 2) -
Power(a, 2)*Power(e1, 2)*Power(r12, 2)*Power(r23, 2) +
2 * Power(c, 2)*e1*e3*r12*r21*r22*r31 -
2 * Power(c, 2)*e1*e3*r11*Power(r22, 2)*r31 +
2 * Power(b, 2)*e1*e3*r13*r21*r23*r31 -
2 * Power(b, 2)*e1*e3*r11*Power(r23, 2)*r31 -
Power(c, 2)*Power(e3, 2)*Power(r22, 2)*Power(r31, 2) -
Power(b, 2)*Power(e3, 2)*Power(r23, 2)*Power(r31, 2) -
2 * Power(c, 2)*e1*e3*r12*Power(r21, 2)*r32 +
2 * Power(c, 2)*e1*e3*r11*r21*r22*r32 + 2 * Power(a, 2)*e1*e3*r13*r22*r23*r32 -
2 * Power(a, 2)*e1*e3*r12*Power(r23, 2)*r32 +
2 * Power(c, 2)*Power(e3, 2)*r21*r22*r31*r32 -
Power(c, 2)*Power(e3, 2)*Power(r21, 2)*Power(r32, 2) -
Power(a, 2)*Power(e3, 2)*Power(r23, 2)*Power(r32, 2) -
2 * Power(b, 2)*e1*e3*r13*Power(r21, 2)*r33 -
2 * Power(a, 2)*e1*e3*r13*Power(r22, 2)*r33 +
2 * Power(b, 2)*e1*e3*r11*r21*r23*r33 + 2 * Power(a, 2)*e1*e3*r12*r22*r23*r33 +
2 * Power(b, 2)*Power(e3, 2)*r21*r23*r31*r33 +
2 * Power(a, 2)*Power(e3, 2)*r22*r23*r32*r33 -
Power(b, 2)*Power(e3, 2)*Power(r21, 2)*Power(r33, 2) -
Power(a, 2)*Power(e3, 2)*Power(r22, 2)*Power(r33, 2) +
2 * Power(c, 2)*e1*Power(r12, 2)*Power(r21, 2)*t1 +
2 * Power(b, 2)*e1*Power(r13, 2)*Power(r21, 2)*t1 -
4 * Power(c, 2)*e1*r11*r12*r21*r22*t1 +
2 * Power(c, 2)*e1*Power(r11, 2)*Power(r22, 2)*t1 +
2 * Power(a, 2)*e1*Power(r13, 2)*Power(r22, 2)*t1 -
4 * Power(b, 2)*e1*r11*r13*r21*r23*t1 - 4 * Power(a, 2)*e1*r12*r13*r22*r23*t1 +
2 * Power(b, 2)*e1*Power(r11, 2)*Power(r23, 2)*t1 +
2 * Power(a, 2)*e1*Power(r12, 2)*Power(r23, 2)*t1 -
2 * Power(c, 2)*e3*r12*r21*r22*r31*t1 +
2 * Power(c, 2)*e3*r11*Power(r22, 2)*r31*t1 -
2 * Power(b, 2)*e3*r13*r21*r23*r31*t1 +
2 * Power(b, 2)*e3*r11*Power(r23, 2)*r31*t1 +
2 * Power(c, 2)*e3*r12*Power(r21, 2)*r32*t1 -
2 * Power(c, 2)*e3*r11*r21*r22*r32*t1 - 2 * Power(a, 2)*e3*r13*r22*r23*r32*t1 +
2 * Power(a, 2)*e3*r12*Power(r23, 2)*r32*t1 +
2 * Power(b, 2)*e3*r13*Power(r21, 2)*r33*t1 +
2 * Power(a, 2)*e3*r13*Power(r22, 2)*r33*t1 -
2 * Power(b, 2)*e3*r11*r21*r23*r33*t1 - 2 * Power(a, 2)*e3*r12*r22*r23*r33*t1 -
Power(c, 2)*Power(r12, 2)*Power(r21, 2)*Power(t1, 2) -
Power(b, 2)*Power(r13, 2)*Power(r21, 2)*Power(t1, 2) +
2 * Power(c, 2)*r11*r12*r21*r22*Power(t1, 2) -
Power(c, 2)*Power(r11, 2)*Power(r22, 2)*Power(t1, 2) -
Power(a, 2)*Power(r13, 2)*Power(r22, 2)*Power(t1, 2) +
2 * Power(b, 2)*r11*r13*r21*r23*Power(t1, 2) +
2 * Power(a, 2)*r12*r13*r22*r23*Power(t1, 2) -
Power(b, 2)*Power(r11, 2)*Power(r23, 2)*Power(t1, 2) -
Power(a, 2)*Power(r12, 2)*Power(r23, 2)*Power(t1, 2) -
2 * Power(c, 2)*e1*r12*r21*r22*r31*t3 +
2 * Power(c, 2)*e1*r11*Power(r22, 2)*r31*t3 -
2 * Power(b, 2)*e1*r13*r21*r23*r31*t3 +
2 * Power(b, 2)*e1*r11*Power(r23, 2)*r31*t3 +
2 * Power(c, 2)*e3*Power(r22, 2)*Power(r31, 2)*t3 +
2 * Power(b, 2)*e3*Power(r23, 2)*Power(r31, 2)*t3 +
2 * Power(c, 2)*e1*r12*Power(r21, 2)*r32*t3 -
2 * Power(c, 2)*e1*r11*r21*r22*r32*t3 - 2 * Power(a, 2)*e1*r13*r22*r23*r32*t3 +
2 * Power(a, 2)*e1*r12*Power(r23, 2)*r32*t3 -
4 * Power(c, 2)*e3*r21*r22*r31*r32*t3 +
2 * Power(c, 2)*e3*Power(r21, 2)*Power(r32, 2)*t3 +
2 * Power(a, 2)*e3*Power(r23, 2)*Power(r32, 2)*t3 +
2 * Power(b, 2)*e1*r13*Power(r21, 2)*r33*t3 +
2 * Power(a, 2)*e1*r13*Power(r22, 2)*r33*t3 -
2 * Power(b, 2)*e1*r11*r21*r23*r33*t3 - 2 * Power(a, 2)*e1*r12*r22*r23*r33*t3 -
4 * Power(b, 2)*e3*r21*r23*r31*r33*t3 - 4 * Power(a, 2)*e3*r22*r23*r32*r33*t3 +
2 * Power(b, 2)*e3*Power(r21, 2)*Power(r33, 2)*t3 +
2 * Power(a, 2)*e3*Power(r22, 2)*Power(r33, 2)*t3 +
2 * Power(c, 2)*r12*r21*r22*r31*t1*t3 -
2 * Power(c, 2)*r11*Power(r22, 2)*r31*t1*t3 +
2 * Power(b, 2)*r13*r21*r23*r31*t1*t3 -
2 * Power(b, 2)*r11*Power(r23, 2)*r31*t1*t3 -
2 * Power(c, 2)*r12*Power(r21, 2)*r32*t1*t3 +
2 * Power(c, 2)*r11*r21*r22*r32*t1*t3 + 2 * Power(a, 2)*r13*r22*r23*r32*t1*t3 -
2 * Power(a, 2)*r12*Power(r23, 2)*r32*t1*t3 -
2 * Power(b, 2)*r13*Power(r21, 2)*r33*t1*t3 -
2 * Power(a, 2)*r13*Power(r22, 2)*r33*t1*t3 +
2 * Power(b, 2)*r11*r21*r23*r33*t1*t3 + 2 * Power(a, 2)*r12*r22*r23*r33*t1*t3 -
Power(c, 2)*Power(r22, 2)*Power(r31, 2)*Power(t3, 2) -
Power(b, 2)*Power(r23, 2)*Power(r31, 2)*Power(t3, 2) +
2 * Power(c, 2)*r21*r22*r31*r32*Power(t3, 2) -
Power(c, 2)*Power(r21, 2)*Power(r32, 2)*Power(t3, 2) -
Power(a, 2)*Power(r23, 2)*Power(r32, 2)*Power(t3, 2) +
2 * Power(b, 2)*r21*r23*r31*r33*Power(t3, 2) +
2 * Power(a, 2)*r22*r23*r32*r33*Power(t3, 2) -
Power(b, 2)*Power(r21, 2)*Power(r33, 2)*Power(t3, 2) -
Power(a, 2)*Power(r22, 2)*Power(r33, 2)*Power(t3, 2))) /
(Power(a, 2)*Power(b, 2)*Power(c, 2));

Scalar coef_d1d2 =
(8 * (-(Power(a, 2)*(e1*r13*r32 + e2 * r23*r32 - e1 * r12*r33 - e2 * r22*r33 - r13 * r32*t1 +
r12 * r33*t1 - r23 * r32*t2 + r22 * r33*t2)*
(e1*r13*r22 - e1 * r12*r23 - e3 * r23*r32 + e3 * r22*r33 - r13 * r22*t1 +
r12 * r23*t1 + r23 * r32*t3 - r22 * r33*t3)) +
Power(c, 2)*(Power(a, 2)*r22*r32 -
Power(e1, 2)*(r12*r21 - r11 * r22)*(r12*r31 - r11 * r32) -
e3 * r12*r22*Power(r31, 2)*t1 + e3 * r12*r21*r31*r32*t1 +
e3 * r11*r22*r31*r32*t1 - e3 * r11*r21*Power(r32, 2)*t1 -
Power(r12, 2)*r21*r31*Power(t1, 2) + r11 * r12*r22*r31*Power(t1, 2) +
r11 * r12*r21*r32*Power(t1, 2) - Power(r11, 2)*r22*r32*Power(t1, 2) -
e3 * Power(r22, 2)*Power(r31, 2)*t2 + 2 * e3*r21*r22*r31*r32*t2 -
e3 * Power(r21, 2)*Power(r32, 2)*t2 - r12 * r21*r22*r31*t1*t2 +
r11 * Power(r22, 2)*r31*t1*t2 + r12 * Power(r21, 2)*r32*t1*t2 -
r11 * r21*r22*r32*t1*t2 + r12 * r22*Power(r31, 2)*t1*t3 -
r12 * r21*r31*r32*t1*t3 - r11 * r22*r31*r32*t1*t3 +
r11 * r21*Power(r32, 2)*t1*t3 + Power(r22, 2)*Power(r31, 2)*t2*t3 -
2 * r21*r22*r31*r32*t2*t3 + Power(r21, 2)*Power(r32, 2)*t2*t3 +
e2 * (r22*r31 - r21 * r32)*(e3*r22*r31 - e3 * r21*r32 + r12 * r21*t1 -
r11 * r22*t1 - r22 * r31*t3 + r21 * r32*t3) +
e1 * (e3*(r12*r31 - r11 * r32)*(r22*r31 - r21 * r32) +
e2 * (r12*r21 - r11 * r22)*(-(r22*r31) + r21 * r32) +
2 * Power(r12, 2)*r21*r31*t1 - 2 * r11*r12*r22*r31*t1 -
2 * r11*r12*r21*r32*t1 + 2 * Power(r11, 2)*r22*r32*t1 +
r12 * r21*r22*r31*t2 - r11 * Power(r22, 2)*r31*t2 -
r12 * Power(r21, 2)*r32*t2 + r11 * r21*r22*r32*t2 -
r12 * r22*Power(r31, 2)*t3 + r12 * r21*r31*r32*t3 + r11 * r22*r31*r32*t3 -
r11 * r21*Power(r32, 2)*t3)) +
Power(b, 2)*(Power(c, 2)*r21*r31 + e2 * e3*Power(r23, 2)*Power(r31, 2) +
Power(a, 2)*r23*r33 - 2 * e2*e3*r21*r23*r31*r33 +
e2 * e3*Power(r21, 2)*Power(r33, 2) -
Power(e1, 2)*(r13*r21 - r11 * r23)*(r13*r31 - r11 * r33) +
e2 * r13*r21*r23*r31*t1 - e2 * r11*Power(r23, 2)*r31*t1 -
e3 * r13*r23*Power(r31, 2)*t1 - e2 * r13*Power(r21, 2)*r33*t1 +
e2 * r11*r21*r23*r33*t1 + e3 * r13*r21*r31*r33*t1 + e3 * r11*r23*r31*r33*t1 -
e3 * r11*r21*Power(r33, 2)*t1 - Power(r13, 2)*r21*r31*Power(t1, 2) +
r11 * r13*r23*r31*Power(t1, 2) + r11 * r13*r21*r33*Power(t1, 2) -
Power(r11, 2)*r23*r33*Power(t1, 2) - e3 * Power(r23, 2)*Power(r31, 2)*t2 +
2 * e3*r21*r23*r31*r33*t2 - e3 * Power(r21, 2)*Power(r33, 2)*t2 -
r13 * r21*r23*r31*t1*t2 + r11 * Power(r23, 2)*r31*t1*t2 +
r13 * Power(r21, 2)*r33*t1*t2 - r11 * r21*r23*r33*t1*t2 -
e2 * Power(r23, 2)*Power(r31, 2)*t3 + 2 * e2*r21*r23*r31*r33*t3 -
e2 * Power(r21, 2)*Power(r33, 2)*t3 + r13 * r23*Power(r31, 2)*t1*t3 -
r13 * r21*r31*r33*t1*t3 - r11 * r23*r31*r33*t1*t3 +
r11 * r21*Power(r33, 2)*t1*t3 + Power(r23, 2)*Power(r31, 2)*t2*t3 -
2 * r21*r23*r31*r33*t2*t3 + Power(r21, 2)*Power(r33, 2)*t2*t3 +
e1 * (e3*(r13*r31 - r11 * r33)*(r23*r31 - r21 * r33) +
e2 * (r13*r21 - r11 * r23)*(-(r23*r31) + r21 * r33) +
2 * Power(r13, 2)*r21*r31*t1 - 2 * r11*r13*r23*r31*t1 -
2 * r11*r13*r21*r33*t1 + 2 * Power(r11, 2)*r23*r33*t1 +
r13 * r21*r23*r31*t2 - r11 * Power(r23, 2)*r31*t2 -
r13 * Power(r21, 2)*r33*t2 + r11 * r21*r23*r33*t2 -
r13 * r23*Power(r31, 2)*t3 + r13 * r21*r31*r33*t3 + r11 * r23*r31*r33*t3 -
r11 * r21*Power(r33, 2)*t3)))) / (Power(a, 2)*Power(b, 2)*Power(c, 2));

Scalar coef_d2d2 =
(-4 * Power(a, 2)*Power(e1*r13*r32 + e2 * r23*r32 - e1 * r12*r33 - e2 * r22*r33 -
r13 * r32*t1 + r12 * r33*t1 - r23 * r32*t2 + r22 * r33*t2, 2) -
4 * Power(c, 2)*(-(Power(a, 2)*Power(r32, 2)) +
Power(e1, 2)*Power(r12*r31 - r11 * r32, 2) +
Power(e2, 2)*Power(r22*r31 - r21 * r32, 2) +
Power(r12, 2)*Power(r31, 2)*Power(t1, 2) - 2 * r11*r12*r31*r32*Power(t1, 2) +
Power(r11, 2)*Power(r32, 2)*Power(t1, 2) + 2 * r12*r22*Power(r31, 2)*t1*t2 -
2 * r12*r21*r31*r32*t1*t2 - 2 * r11*r22*r31*r32*t1*t2 +
2 * r11*r21*Power(r32, 2)*t1*t2 + Power(r22, 2)*Power(r31, 2)*Power(t2, 2) -
2 * r21*r22*r31*r32*Power(t2, 2) + Power(r21, 2)*Power(r32, 2)*Power(t2, 2) -
2 * e2*(r22*r31 - r21 * r32)*(r12*r31*t1 - r11 * r32*t1 + r22 * r31*t2 -
r21 * r32*t2) + 2 * e1*(r12*r31 - r11 * r32)*
(e2*r22*r31 - e2 * r21*r32 - r12 * r31*t1 + r11 * r32*t1 - r22 * r31*t2 +
r21 * r32*t2)) + 4 * Power(b, 2)*
(Power(c, 2)*Power(r31, 2) - Power(e2, 2)*Power(r23, 2)*Power(r31, 2) +
2 * Power(e2, 2)*r21*r23*r31*r33 + Power(a, 2)*Power(r33, 2) -
Power(e2, 2)*Power(r21, 2)*Power(r33, 2) -
Power(e1, 2)*Power(r13*r31 - r11 * r33, 2) + 2 * e2*r13*r23*Power(r31, 2)*t1 -
2 * e2*r13*r21*r31*r33*t1 - 2 * e2*r11*r23*r31*r33*t1 +
2 * e2*r11*r21*Power(r33, 2)*t1 - Power(r13, 2)*Power(r31, 2)*Power(t1, 2) +
2 * r11*r13*r31*r33*Power(t1, 2) - Power(r11, 2)*Power(r33, 2)*Power(t1, 2) +
2 * e2*Power(r23, 2)*Power(r31, 2)*t2 - 4 * e2*r21*r23*r31*r33*t2 +
2 * e2*Power(r21, 2)*Power(r33, 2)*t2 - 2 * r13*r23*Power(r31, 2)*t1*t2 +
2 * r13*r21*r31*r33*t1*t2 + 2 * r11*r23*r31*r33*t1*t2 -
2 * r11*r21*Power(r33, 2)*t1*t2 - Power(r23, 2)*Power(r31, 2)*Power(t2, 2) +
2 * r21*r23*r31*r33*Power(t2, 2) - Power(r21, 2)*Power(r33, 2)*Power(t2, 2) +
2 * e1*(r13*r31 - r11 * r33)*(-(e2*r23*r31) + e2 * r21*r33 + r13 * r31*t1 -
r11 * r33*t1 + r23 * r31*t2 - r21 * r33*t2))) /
(Power(a, 2)*Power(b, 2)*Power(c, 2));

    auto& m = *M;
    // above the diagonal
    m(0, 0) = coef_d0d0;
    m(0, 1) = coef_d0d1 * 0.5;
    m(0, 2) = coef_d0d2 * 0.5;
    m(1, 1) = coef_d1d1;
    m(1, 2) = coef_d1d2 * 0.5;
    m(2, 2) = coef_d2d2;
    m.triangularView<Eigen::StrictlyLower>() = m.triangularView<Eigen::StrictlyUpper>().transpose();
}
}
