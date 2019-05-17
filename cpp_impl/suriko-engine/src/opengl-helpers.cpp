#include <gsl/span>
#include <Eigen/Dense>
#include <glog/logging.h>
#include "suriko/opengl-helpers.h"

namespace suriko { namespace internals {

void LoadSE3TransformIntoOpengGLMat(const SE3Transform& cam_wfc, gsl::span<double> opengl_mat_by_col)
{
    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::ColMajor>> opengl_mat(opengl_mat_by_col.data());
    opengl_mat.topLeftCorner<3, 3>() = cam_wfc.R.cast<double>();
    opengl_mat.topRightCorner<3, 1>() = cam_wfc.T.cast<double>();
    opengl_mat.bottomLeftCorner<1, 3>().setZero();
    opengl_mat(3, 3) = 1;
}

void ConvertAxesHartleyZissermanToOpenGL(gsl::span<Scalar> m4x4_by_col)
{
    // the conversion goes like this:
    // m_out = rot(OY,pi) * m_in
    // rot(OY,pi)=
    // [-1 0  0 0]
    // [ 0 1  0 0]
    // [ 0 0 -1 0]
    // [ 0 0  0 1]
    SRK_ASSERT(m4x4_by_col.size() == 16);
    // 0-row
    m4x4_by_col[0] *= -1;
    m4x4_by_col[4] *= -1;
    m4x4_by_col[8] *= -1;
    m4x4_by_col[12] *= -1;
    // 2-row
    m4x4_by_col[2] *= -1;
    m4x4_by_col[6] *= -1;
    m4x4_by_col[10] *= -1;
    m4x4_by_col[14] *= -1;
}

void GetOpenGLFromHartleyZissermanMat(gsl::span<Scalar> m4x4_by_col)
{
    // the conversion goes like this:
    // m_out = rot(OY,pi) * m_in
    // rot(OY,pi)=
    // [-1 0  0 0]
    // [ 0 1  0 0]
    // [ 0 0 -1 0]
    // [ 0 0  0 1]
    SRK_ASSERT(m4x4_by_col.size() == 16);
    
    Eigen::Map<Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>> opengl_mat(m4x4_by_col.data());
    opengl_mat.setIdentity();
    opengl_mat(0, 0) = -1; // OX=-OX
    opengl_mat(2, 2) = -1; // OY=-OY
}
}}