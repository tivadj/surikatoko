#pragma once
#include "suriko/rt-config.h"
#include "suriko/obs-geom.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core.hpp> // cv::Mat
#endif

namespace suriko_demos
{
#if defined(SRK_HAS_OPENCV)
void Draw2DProjectedAxes(suriko::Scalar f0, std::function<Eigen::Matrix<suriko::Scalar, 3, 1>(const suriko::Point3&)> projector, cv::Mat* camera_image_rgb);
#endif
}