#pragma once
#include <gsl/span>
#include "suriko/rt-config.h" // Scalar
#include "suriko/obs-geom.h"

namespace suriko { namespace internals
{
void LoadSE3TransformIntoOpengGLMat(const SE3Transform& cam_wfc, gsl::span<double> opengl_mat_by_col);

/// Convert axes from Hartley and Zisserman format (XYZ=left-up-forward) into OpenGL format (XYZ=right-up-back).
void ConvertAxesHartleyZissermanToOpenGL(gsl::span<Scalar> m4x4_by_col);

/// Gets 4x4 matrix to convert axes from Hartley and Zisserman format (XYZ=left-up-forward) into OpenGL format (XYZ=right-up-back).
void GetOpenGLFromHartleyZissermanMat(gsl::span<Scalar> m4x4_by_col);
}}
