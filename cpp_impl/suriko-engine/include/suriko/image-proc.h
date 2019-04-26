#pragma once

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core/core.hpp> // cv::Mat
#endif

namespace suriko {
/// Helper class to transfer around a gray and BGR image from camera. The BGR image is used for debugging purposes.
struct Picture
{
    cv::Mat gray;
#if defined(SRK_DEBUG)
    cv::Mat bgr_debug;
#endif
};

void CopyBgr(const Picture& image, cv::Mat* out_image_bgr);
}
