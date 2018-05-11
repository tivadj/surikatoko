#include "visualize-helpers.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/imgproc.hpp> // CV_RGB, cv::line
#endif

namespace suriko_demos
{
using namespace suriko;

#if defined(SRK_HAS_OPENCV)
void Draw2DProjectedAxes(suriko::Scalar f0, std::function<Eigen::Matrix<suriko::Scalar, 3, 1>(const suriko::Point3&)> projector, cv::Mat* camera_image_rgb)
{
    // show center of coordinates as red dot
    std::vector<suriko::Point3> axes_pnts = {
        suriko::Point3(0, 0, 0),
        suriko::Point3(1, 0, 0),
        suriko::Point3(0, 1, 0),
        suriko::Point3(0, 0, 1)
    };
    std::vector<cv::Scalar> axes_colors = {
        CV_RGB(255, 255, 255),
        CV_RGB(255,   0,   0),
        CV_RGB(0, 255,   0),
        CV_RGB(0,   0, 255)
    };
    std::vector<cv::Point2i> axes_pnts2D(axes_pnts.size());
    for (size_t i = 0; i < axes_pnts.size(); ++i)
    {
        //Eigen::Matrix<Scalar, 3, 1> p = ProjectPnt(K, cam_inverse_orient, axes_pnts[i]);
        Eigen::Matrix<Scalar, 3, 1> p = projector(axes_pnts[i]);
        axes_pnts2D[i] = cv::Point2i(
            static_cast<int>(p[0] / p[2] * f0),
            static_cast<int>(p[1] / p[2] * f0));
    }
    for (size_t i = 1; i < axes_pnts.size(); ++i)
    {
        cv::line(*camera_image_rgb, axes_pnts2D[0], axes_pnts2D[i], axes_colors[i]); // OX, OZ, OZ segments
        //cv::circle(camera_image_rgb, axes_pnts2D[i], 3, axes_colors[i]);
    }
}
#endif
}