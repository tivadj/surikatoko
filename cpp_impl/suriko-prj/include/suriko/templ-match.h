#pragma once
#include "suriko/rt-config.h"
#include "suriko/image-proc.h"
#include "suriko/obs-geom.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core/core.hpp> // cv::Mat
#endif

namespace suriko {
struct CorrelationCoeffData
{
    Scalar corr_prod_sum;
    Scalar image_diff_sqr_sum;
};

Scalar GetGrayPatchMean(const cv::Mat& gray_image, suriko::Pointi patch_top_left, int patch_width, int patch_height);
Scalar GetGrayPatchDiffSqrSum(const cv::Mat& gray_image, suriko::Pointi patch_top_left, int patch_width, int patch_height, Scalar patch_mean);

CorrelationCoeffData CalcCorrCoeff(const suriko::Picture& pic,
    suriko::Pointi pic_roi_top_left,
    Scalar pic_roi_mean,
    const cv::Mat& templ_gray,
    Scalar templ_mean);
}
