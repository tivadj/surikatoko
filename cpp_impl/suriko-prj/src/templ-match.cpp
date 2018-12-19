#include "suriko/templ-match.h"
#include "suriko/approx-alg.h"
#include <opencv2/imgproc.hpp>

namespace suriko
{
Scalar GetGrayImageMean(const cv::Mat& gray_image, suriko::Recti roi)
{
    Scalar s{ 0 };
    for (int row = 0; row < roi.height; ++row)
    {
        auto src_image_row_ptr = gray_image.ptr<unsigned char>(roi.y + row);
        auto src_image_cell_ptr = &src_image_row_ptr[roi.x];

        for (int col = 0; col < roi.width; ++col)
        {
            // NOTE: Mat.at(x,y) is a hot-spot (called multitude of times, bounds checking)
            auto v = *src_image_cell_ptr;
            src_image_cell_ptr++;

            auto v_float = static_cast<Scalar>(v);
            s += v_float;
        }
    }
    const Scalar mean = s / (roi.width*roi.height);
    return mean;
}

Scalar GetGrayImageSumSqrDiff(const cv::Mat& gray_image, suriko::Recti roi, Scalar roi_mean)
{
    Scalar sum{ 0 };
    for (int row = 0; row < roi.height; ++row)
    {
        auto src_image_row_ptr = gray_image.ptr<unsigned char>(roi.y + row);
        auto src_image_cell_ptr = &src_image_row_ptr[roi.x];

        for (int col = 0; col < roi.width; ++col)
        {
            // NOTE: Mat.at(x,y) is a hot-spot (called multitude of times, bounds checking)
            auto v = *src_image_cell_ptr;
            src_image_cell_ptr++;

            auto v_float = static_cast<Scalar>(v);

            Scalar v_diff = suriko::Sqr(v_float - roi_mean);
            sum += v_diff;
        }
    }
    return sum;
}

CorrelationCoeffData CalcCorrCoeffComponents(const Picture& pic,
    Recti pic_roi,
    Scalar pic_roi_mean,
    const cv::Mat& templ_gray,
    Scalar templ_mean)
{
    CorrelationCoeffData corr{};
    for (int row = 0; row < templ_gray.rows; ++row)
    {
        auto src_image_row_ptr = pic.gray.ptr<unsigned char>(pic_roi.y + row);
        auto src_image_cell_ptr = &src_image_row_ptr[pic_roi.x];

        auto templ_image_cell_ptr = templ_gray.ptr<unsigned char>(row);

        for (int col = 0; col < templ_gray.cols; ++col)
        {
            // NOTE: Mat.at(x,y) is a hot-spot (called multitude of times, bounds checking)
            auto frame_value = *src_image_cell_ptr;
            auto templ_value = *templ_image_cell_ptr;

            src_image_cell_ptr++;
            templ_image_cell_ptr++;

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

std::optional<Scalar> CalcCorrCoeff(const Picture& pic,
    Recti pic_roi,
    const cv::Mat& templ_gray,
    Scalar templ_mean,
    Scalar templ_sqrt_sum_sqr_diff)
{
    SRK_ASSERT(templ_sqrt_sum_sqr_diff != 0);
    Scalar pic_roi_mean = GetGrayImageMean(pic.gray, pic_roi);

    CorrelationCoeffData corr_data = CalcCorrCoeffComponents(pic, pic_roi, pic_roi_mean, templ_gray, templ_mean);

    // corr coef is undefined when variance=0 (image is filled with a single color)
    if (IsClose(0, corr_data.image_diff_sqr_sum))
        return std::nullopt;
    
    Scalar corr_coeff = corr_data.corr_prod_sum / (std::sqrt(corr_data.image_diff_sqr_sum) * templ_sqrt_sum_sqr_diff);
    SRK_ASSERT(std::isfinite(corr_coeff));

    return corr_coeff;
}

}
