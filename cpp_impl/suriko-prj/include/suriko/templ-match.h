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

/// Computes mean(img).
Scalar GetGrayImageMean(const cv::Mat& gray_image, suriko::Recti roi);

/// Computes sqr(X-mean(X)).
Scalar GetGrayImageSumSqrDiff(const cv::Mat& gray_image, suriko::Recti roi, Scalar patch_mean);

CorrelationCoeffData CalcCorrCoeffComponents(const suriko::Picture& pic,
    suriko::Recti pic_roi,
    Scalar pic_roi_mean,
    const cv::Mat& templ_gray,
    Scalar templ_mean);

/// Returns null if corr coef is undefined (when variance=0, eg. entire image is filled with a single color)
std::optional<Scalar> CalcCorrCoeff(const Picture& pic,
    Recti pic_roi,
    const cv::Mat& templ_gray,
    Scalar templ_mean,
    Scalar templ_sqrt_sum_sqr_diff);
} // ns
