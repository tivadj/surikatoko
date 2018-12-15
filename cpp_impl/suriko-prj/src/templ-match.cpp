#include "suriko/templ-match.h"
#include "suriko/approx-alg.h"

namespace suriko
{
Scalar GetGrayPatchMean(const cv::Mat& gray_image, Pointi patch_top_left, int patch_width, int patch_height)
{
    Scalar sum{ 0 };
    for (int row = 0; row < patch_height; ++row)
    {
        for (int col = 0; col < patch_width; ++col)
        {
            auto v = gray_image.at<unsigned char>(patch_top_left.y + row, patch_top_left.x + col);
            auto v_float = static_cast<Scalar>(v);
            sum += v_float;
        }
    }
    const Scalar mean = sum / (patch_width*patch_height);
    return mean;
}

Scalar GetGrayPatchDiffSqrSum(const cv::Mat& gray_image, Pointi patch_top_left, int patch_width, int patch_height, Scalar patch_mean)
{
    Scalar sum{ 0 };
    for (int row = 0; row < patch_height; ++row)
    {
        for (int col = 0; col < patch_width; ++col)
        {
            auto v = gray_image.at<unsigned char>(patch_top_left.y + row, patch_top_left.x + col);
            auto v_float = static_cast<Scalar>(v);

            Scalar v_diff = suriko::Sqr(v_float - patch_mean);
            sum += v_diff;
        }
    }
    return sum;
}

CorrelationCoeffData CalcCorrCoeff(const Picture& pic,
    Pointi pic_roi_top_left,
    Scalar pic_roi_mean,
    const cv::Mat& templ_gray,
    Scalar templ_mean)
{
    CorrelationCoeffData corr{};
    for (int row = 0; row < templ_gray.rows; ++row)
    {
        for (int col = 0; col < templ_gray.cols; ++col)
        {
            auto frame_value = pic.gray.at<unsigned char>(pic_roi_top_left.y + row, pic_roi_top_left.x + col);
            auto templ_value = templ_gray.at<unsigned char>(row, col);

            auto f = static_cast<Scalar>(frame_value);
            auto t = static_cast<Scalar>(templ_value);

            Scalar f_diff = f - pic_roi_mean;
            Scalar t_diff = t - templ_mean;

            Scalar prod = f_diff * t_diff;
            corr.corr_prod_sum += prod;

            Scalar f_diff2 = suriko::Sqr(f_diff);
            corr.image_diff_sqr_sum += f_diff2;
        }
    }
    return corr;
}
}
