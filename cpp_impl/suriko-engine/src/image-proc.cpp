#include "suriko/image-proc.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/imgproc.hpp> // cv::cvtColor
#endif

namespace suriko
{
void CopyBgr(const Picture& image, cv::Mat* out_image_bgr)
{
#if defined(SRK_DEBUG)
    image.bgr_debug.copyTo(*out_image_bgr);
#else
    cv::cvtColor(image.gray, *out_image_bgr, CV_GRAY2BGR);
#endif
}
}
