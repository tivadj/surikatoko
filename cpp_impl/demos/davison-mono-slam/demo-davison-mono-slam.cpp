#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <functional>
#include <numeric>
#include <utility>
#include <cassert>
#include <cmath>
#include <corecrt_math_defines.h>
#include <random>
#include <tuple>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <filesystem>
#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "suriko/rt-config.h"
#include "suriko/bundle-adj-kanatani.h"
#include "suriko/obs-geom.h"
#include "suriko/mat-serialization.h"
#include "suriko/templ-match.h"
#include "suriko/approx-alg.h"
#include "suriko/davison-mono-slam.h"
#include "suriko/virt-world/scene-generator.h"
#include "suriko/quat.h"
#include "../stat-helpers.h"
#include "../visualize-helpers.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/imgproc.hpp> // cv::circle
#include <opencv2/highgui.hpp> // cv::imshow
#include <opencv2/features2d.hpp> // cv::ORB
#endif

#include "demo-davison-mono-slam-ui.h"

// PROVIDE DATA SOURCE FOR DEMO
#define DEMO_DATA_SOURCE_TYPE kVirtualScene

namespace suriko_demos_davison_mono_slam
{
using namespace std;
using namespace boost::filesystem;
using namespace suriko; 
using namespace suriko::internals;
using namespace suriko::virt_world;

// Specify data source for demo: virtual scene data or sequence of images in a directory.
#define kVirtualScene 0
#define kImageSeqDir 1
//static constexpr int kVirtualScene = 0;
//static constexpr int kImageSeqDir = 1;
//enum class DemoDataSource { kVirtualScene, kImageSeqDir };

auto DemoGetSalPntId(const SalientPointFragment& fragment) -> DavisonMonoSlam::SalPntId
{
    auto sal_pnt_id = DavisonMonoSlam::SalPntId::Null();
    if (fragment.user_obj != nullptr)
    {
        // got salient point which hits the current frame and had been seen before
        // auto sal_pnt_id = reinterpret_cast<DavisonMonoSlam::SalPntId>(fragment.UserObj);
        std::memcpy(&sal_pnt_id, &fragment.user_obj, sizeof(decltype(fragment.user_obj)));
        static_assert(sizeof fragment.user_obj >= sizeof sal_pnt_id);
    }
    return sal_pnt_id;
}

auto DemoGetSalPntFramgmentId(const FragmentMap& entire_map, DavisonMonoSlam::SalPntId sal_pnt_id) -> std::optional<size_t>
{
    std::vector<size_t> salient_points_ids;
    entire_map.GetSalientPointsIds(&salient_points_ids);

    for (size_t frag_id : salient_points_ids)
    {
        const SalientPointFragment& fragment = entire_map.GetSalientPointNew(frag_id);
        DavisonMonoSlam::SalPntId sal_pnt_id_in_fragment = DemoGetSalPntId(fragment);

        if (sal_pnt_id_in_fragment == sal_pnt_id)
            return frag_id;
    }
    return std::nullopt;
}

void GenerateCameraShotsAlongRectangularPath(const WorldBounds& wb, size_t steps_per_side_x, size_t steps_per_side_y,
    suriko::Point3 eye_offset, 
    suriko::Point3 center_offset,
    const Eigen::Matrix<Scalar, 3, 1>& up,
    std::vector<SE3Transform>* inverse_orient_cams)
{
    std::array<suriko::Point3,5> look_at_base_points = {
        suriko::Point3(wb.x_min, wb.y_min, wb.z_min),
        suriko::Point3(wb.x_max, wb.y_min, wb.z_min),
        suriko::Point3(wb.x_max, wb.y_max, wb.z_min),
        suriko::Point3(wb.x_min, wb.y_max, wb.z_min),
        suriko::Point3(wb.x_min, wb.y_min, wb.z_min),
    };

    // number of viewer steps per each side is variable
    std::array<size_t, 4> viewer_steps_per_side = {
        steps_per_side_x,
        steps_per_side_y,
        steps_per_side_x,
        steps_per_side_y
    };

    for (size_t base_point_ind = 0; base_point_ind < look_at_base_points.size()-1; ++base_point_ind)
    {
        suriko::Point3 base1 = look_at_base_points[base_point_ind];
        suriko::Point3 base2 = look_at_base_points[base_point_ind+1];
        size_t steps_per_side = viewer_steps_per_side[base_point_ind];

        Eigen::Matrix<Scalar, 3, 1> step = (base2.Mat() - base1.Mat()) / steps_per_side;

        // to avoid repeating the adjacent point of two consecutive segments, for each segment,
        // the last point is not included because
        // it will be included as the first point of the next segment
        for (size_t step_ind = 0; step_ind < steps_per_side; ++step_ind)
        {
            suriko::Point3 cur_point = suriko::Point3(base1.Mat() + step * step_ind);

            auto wfc = LookAtLufWfc(
                cur_point.Mat() + eye_offset.Mat(),
                cur_point.Mat() + center_offset.Mat(),
                up);

            SE3Transform RT = SE3Inv(wfc);

            inverse_orient_cams->push_back(RT);
        }
    }
}

void GenerateWorldPoints(WorldBounds wb, const std::array<Scalar, 3>& cell_size,
    bool corrupt_salient_points_with_noise,
    std::mt19937* gen,
    std::normal_distribution<Scalar>* x3D_noise_dis, FragmentMap* entire_map)
{
    size_t next_virtual_point_id = 6000'000 + 1;
    constexpr Scalar inclusive_gap = 1e-8; // small value to make iteration inclusive


    Scalar xmid = (wb.x_min + wb.x_max) / 2;
    Scalar xlen = wb.x_max - wb.x_min;
    Scalar zlen = wb.z_max - wb.z_min;
    for (Scalar grid_x = wb.x_min; grid_x < wb.x_max + inclusive_gap; grid_x += cell_size[0])
    {
        for (Scalar grid_y = wb.y_min; grid_y < wb.y_max + inclusive_gap; grid_y += cell_size[1])
        {
            Scalar x = grid_x;
            Scalar y = grid_y;

            Scalar z_perc = std::cos((x - xmid) / xlen * M_PI);
            Scalar z = wb.z_min + z_perc * zlen;

            // jit x and y so the points can be distinguished during movement
            if (corrupt_salient_points_with_noise)
            {
                x += (*x3D_noise_dis)(*gen);
                y += (*x3D_noise_dis)(*gen);
                z += (*x3D_noise_dis)(*gen);
            }

            SalientPointFragment& frag = entire_map->AddSalientPointPatch(Point3(x, y, z));
            frag.synthetic_virtual_point_id = next_virtual_point_id++;
        }
    }
}

/// Gets the transformation from world into camera in given frame.
SE3Transform CurCamFromTrackerOrigin(const std::vector<SE3Transform>& gt_cam_orient_cfw, size_t frame_ind, const SE3Transform& tracker_from_world)
{
    const SE3Transform& cur_cam_cfw = gt_cam_orient_cfw[frame_ind];
    SE3Transform rt_cft = SE3AFromB(cur_cam_cfw, tracker_from_world);  // current camera in the coordinates of the first camera
    return rt_cft;
}

suriko::Point3 PosTrackerOriginFromWorld(const std::vector<SE3Transform>& gt_cam_orient_cfw, suriko::Point3 p_world,
    const SE3Transform& tracker_from_world)
{
    suriko::Point3 p_tracker = SE3Apply(tracker_from_world, p_world);
    return p_tracker;
}

struct BlobInfo
{
    suriko::Point2 Coord;
    DavisonMonoSlam::SalPntId SalPntIdInTracker;
    std::optional<Scalar> GTInvDepth; // inversed ground truth depth to salient point in virtual environments
    size_t FragmentId; // the id of a fragment in entire map
};

class DemoCornersMatcher : public CornersMatcherBase
{
    const std::vector<SE3Transform>& gt_cam_orient_cfw_;
    FragmentMap& entire_map_;
    suriko::Sizei img_size_;
    const DavisonMonoSlam* mono_slam_;
    bool suppress_observations_ = false; // true to make camera magically don't detect any salient points
    std::vector<BlobInfo> detected_blobs_; // blobs detected in current frame
public:
    SE3Transform tracker_origin_from_world_;
    std::optional<size_t> max_new_blobs_per_frame_;
    std::optional<size_t> max_new_blobs_in_first_frame_;
    std::optional<float> match_blob_prob_ = 1; // [0,1] portion of blobs which are matched with ones in the previous frame; 1=all matched, 0=none matched;
    std::mt19937 gen_{292};
    std::uniform_real_distribution<float> uniform_distr_{};
public:
    DemoCornersMatcher(const DavisonMonoSlam* mono_slam, const std::vector<SE3Transform>& gt_cam_orient_cfw, FragmentMap& entire_map,
        const suriko::Sizei& img_size)
        : mono_slam_(mono_slam),
        gt_cam_orient_cfw_(gt_cam_orient_cfw),
        entire_map_(entire_map),
        img_size_(img_size)
    {
        tracker_origin_from_world_.T.setZero();
        tracker_origin_from_world_.R.setIdentity();
    }

    void AnalyzeFrame(size_t frame_ind, const Picture& image) override
    {
        detected_blobs_.clear();

        if (suppress_observations_)
            return;

        // determine current camerra's orientation using the ground truth
        const SE3Transform& cami_from_tracker = CurCamFromTrackerOrigin(gt_cam_orient_cfw_, frame_ind, tracker_origin_from_world_);

        std::vector<size_t> salient_points_ids;
        entire_map_.GetSalientPointsIds(&salient_points_ids);

        for (size_t frag_id : salient_points_ids)
        {
            const SalientPointFragment& fragment = entire_map_.GetSalientPointNew(frag_id);

            const Point3& salient_point_world = fragment.coord.value();
            suriko::Point3 pnt_tracker = PosTrackerOriginFromWorld(gt_cam_orient_cfw_, salient_point_world, tracker_origin_from_world_);

            suriko::Point3 pnt_camera = SE3Apply(cami_from_tracker, pnt_tracker);
            suriko::Point2 pnt_pix = mono_slam_->ProjectCameraPoint(pnt_camera);
            Scalar pix_x = pnt_pix[0];
            Scalar pix_y = pnt_pix[1];
            bool hit_wnd =
                pix_x >= 0 && pix_x < (Scalar)img_size_.width &&
                pix_y >= 0 && pix_y < (Scalar)img_size_.height;
            if (!hit_wnd)
                continue;

            DavisonMonoSlam::SalPntId sal_pnt_id = DemoGetSalPntId(fragment);

            const Scalar depth = pnt_camera.Mat().norm();
            SRK_ASSERT(!IsClose(0, depth)) << "salient points with zero depth are prohibited";

            BlobInfo blob_info;
            blob_info.Coord = pnt_pix;
            blob_info.SalPntIdInTracker = sal_pnt_id;
            blob_info.GTInvDepth = 1 / depth;
            blob_info.FragmentId = frag_id;
            detected_blobs_.push_back(blob_info);
        }
    }

    void MatchSalientPoints(size_t frame_ind,
        const Picture& image,
        const std::set<SalPntId>& tracking_sal_pnts,
        std::vector<std::pair<DavisonMonoSlam::SalPntId, CornersMatcherBlobId>>* matched_sal_pnts) override
    {
        if (suppress_observations_)
            return;
        
        for (size_t i=0; i < detected_blobs_.size(); ++i)
        {
            bool allow_match = true;
            if (match_blob_prob_.has_value())
            {
                float rv = uniform_distr_(gen_);
                allow_match = rv < match_blob_prob_.value();
            }
            if (!allow_match)
                continue;

            BlobInfo blob_info = detected_blobs_[i];

            DavisonMonoSlam::SalPntId sal_pnt_id = blob_info.SalPntIdInTracker;
            if (auto it = tracking_sal_pnts.find(sal_pnt_id); it == tracking_sal_pnts.end())
            {
                // match only salient points which have been tracked earlier
                continue;
            }

            // the tracker is interested in this blob
            size_t blob_id = i;
            auto sal_pnt_to_coord = std::make_pair(sal_pnt_id, CornersMatcherBlobId{ blob_id });
            matched_sal_pnts->push_back(sal_pnt_to_coord);
        }
    }

    void RecruitNewSalientPoints(size_t frame_ind,
        const Picture& image,
        const std::set<SalPntId>& tracking_sal_pnts,
        const std::vector<std::pair<DavisonMonoSlam::SalPntId, CornersMatcherBlobId>>& matched_sal_pnts,
        std::vector<CornersMatcherBlobId>* new_blob_ids) override
    {
        if (suppress_observations_)
            return;

        std::optional<size_t> max_blobs = std::nullopt;
        if (frame_ind == 0)
        {
            if (max_new_blobs_in_first_frame_.has_value())
                max_blobs = max_new_blobs_in_first_frame_;
        }
        else
        {
            if (max_new_blobs_per_frame_.has_value())
                max_blobs = max_new_blobs_per_frame_;
        }
        
        for (size_t i = 0; i < detected_blobs_.size(); ++i)
        {
            if (max_blobs.has_value() && new_blob_ids->size() >= max_blobs)
                break;

            BlobInfo blob_info = detected_blobs_[i];

            DavisonMonoSlam::SalPntId sal_pnt_id = blob_info.SalPntIdInTracker;
            if (auto it = tracking_sal_pnts.find(sal_pnt_id); it != tracking_sal_pnts.end())
            {
                // this salient point is already matched and can't be treated as new
                continue;
            }

            // if salient point is associated with a blob then prevent associating new salient point with such a blob
            if (blob_info.SalPntIdInTracker.HasId())
                continue;

            // the tracker is interested in this blob
            size_t blob_id = i;
            new_blob_ids->push_back(CornersMatcherBlobId {blob_id});
        }
    }

    void OnSalientPointIsAssignedToBlobId(SalPntId sal_pnt_id, CornersMatcherBlobId blob_id, const Picture& image) override
    {
        size_t frag_id = detected_blobs_[blob_id.Ind].FragmentId;
        SalientPointFragment& frag = entire_map_.GetSalientPointNew(frag_id);
        
        static_assert(sizeof sal_pnt_id <= sizeof frag.user_obj, "SalPntId must fit into UserObject");
        std::memcpy(&frag.user_obj, &sal_pnt_id, sizeof(sal_pnt_id));
    }

    suriko::Point2 GetBlobCoord(CornersMatcherBlobId blob_id) override
    {
        return detected_blobs_[blob_id.Ind].Coord;
    }
    
    std::optional<Scalar> GetSalientPointGroundTruthInvDepth(CornersMatcherBlobId blob_id) override
    {
        return detected_blobs_[blob_id.Ind].GTInvDepth;
    }

    void SetSuppressObservations(bool value) { suppress_observations_ = value; }

    const std::vector<BlobInfo>& DetectedBlobs() const { return detected_blobs_; }
};

#if defined(SRK_HAS_OPENCV)

class ImagePatchCornersMatcher : public CornersMatcherBase
{
    bool suppress_observations_ = false; // true to make camera magically don't detect any salient points
    cv::Ptr<cv::ORB> detector_;
    DavisonMonoSlam* kalman_tracker_;
    std::vector<cv::KeyPoint> new_keypoints_;
public:
    bool stop_on_sal_pnt_moved_too_far_ = false;
    Scalar ellisoid_cut_thr_;
    std::function<void(DavisonMonoSlam&, SalPntId, cv::Mat*)> draw_sal_pnt_fun_;
    std::function<void(std::string_view, cv::Mat)> show_image_fun_;
    std::optional<suriko::Sizei> min_search_rect_size_;
    std::optional<Scalar> min_templ_corr_coeff_;
public:
    ImagePatchCornersMatcher(DavisonMonoSlam* kalman_tracker)
        :kalman_tracker_(kalman_tracker)
    {
        int nfeatures = 50;
        detector_ = cv::ORB::create(nfeatures);
    }

    struct TemplateMatchResult
    {
        bool success;
        suriko::Point2 center;
        Scalar corr_coef;

#if defined(SRK_DEBUG)
        suriko::Pointi top_left;
        int executed_match_templ_calls; // specify the number of calls to match-template routine to find this specific match-result
#endif
    };

    TemplateMatchResult MatchSalientPointTemplCenterInRect(const SalPntPatch& sal_pnt, const Picture& pic, Recti search_rect)
    {
        Pointi search_center{ search_rect.x + search_rect.width / 2, search_rect.y + search_rect.height / 2 };
        const int search_radius_left = search_rect.width / 2;
        const int search_radius_up = search_rect.height / 2;

        // -1 to make up for center pixel
        const int search_radius_right = search_rect.width - search_radius_left - 1;
        const int search_radius_down = search_rect.height - search_radius_up - 1;
        const int search_common_rad = std::min({ search_radius_left , search_radius_right , search_radius_up, search_radius_down });

        struct PosAndErr
        {
            Pointi templ_top_left;
            Scalar corr_coef;
            int executed_match_templ_calls = 0;
        };
        
        // choose template-candidate with the maximum correlation coefficient
        // TODO: do we need to handle the case of multiple equal corr coefs? (eg when all pixels of a cadidate are equal)
        auto max_corr_coeff = Scalar{ -1 - 0.001 };
        PosAndErr best_match_info;
        int match_templ_call_order = 0;  // specify the order of calls to template match routine

        Scalar templ_mean = sal_pnt.templ_stats.templ_mean_;
        Scalar templ_sqrt_sum_sqr_diff = sal_pnt.templ_stats.templ_sqrt_sum_sqr_diff_;
        const auto& templ_gray = sal_pnt.initial_templ_gray_;

        auto match_templ_at = [this, &templ_gray, &pic,
            &max_corr_coeff, &best_match_info, templ_mean, templ_sqrt_sum_sqr_diff, &match_templ_call_order](Pointi search_center)
        {
#if defined(SRK_DEBUG)
            match_templ_call_order++;
#endif
            Pointi pic_roi_top_left = kalman_tracker_->TemplateTopLeftInt(suriko::Point2{ search_center.x, search_center.y });

            auto pic_roi = suriko::Recti{ pic_roi_top_left.x, pic_roi_top_left.y,
                kalman_tracker_->sal_pnt_patch_size_.width,
                kalman_tracker_->sal_pnt_patch_size_.height
            };
            
            std::optional<Scalar> corr_coeff_opt = CalcCorrCoeff(pic, pic_roi, templ_gray, templ_mean, templ_sqrt_sum_sqr_diff);
            if (!corr_coeff_opt.has_value())
                return;  // roi is filled with a single color
            
            Scalar corr_coeff = corr_coeff_opt.value();
            if (corr_coeff > max_corr_coeff)
            {
                best_match_info = PosAndErr{ };
                best_match_info.templ_top_left = pic_roi_top_left;
                best_match_info.corr_coef = corr_coeff;
#if defined(SRK_DEBUG)
                best_match_info.executed_match_templ_calls = match_templ_call_order;
#endif
                max_corr_coeff = corr_coeff;
            }
        };

        // process central pixel
        match_templ_at(search_center);  // rad=0

        // The probability of matching is the greatest in the center.
        // Hence iterate around the center pixel in circles.
        
        Recti border{ search_center.x, search_center.y, 1, 1 };
        for (int rad = 1; rad < search_common_rad; ++rad)
        {
            // inflate by one line from each side
            border.x -= 1;
            border.y -= 1;
            border.width += 2;
            border.height += 2;

            // each pixel at the end of a segment is not included
            // thus the corner pixels are iterated exactly once

            // top-right to top-left
            for (int x = border.Right() - 1; x > border.x; --x)
                match_templ_at(Pointi{ x, border.y });

            // top-left to bottom-left
            for (int y = border.y; y < border.Bottom() - 1; ++y)
                match_templ_at(Pointi{ border.x, y });

            // bottom-left to bottom-right
            for (int x = border.x; x < border.Right() - 1; ++x)
                match_templ_at(Pointi{ x, border.Bottom() - 1 });

            // bottom-right to top-right
            for (int y = border.Bottom() - 1; y > border.y; --y)
                match_templ_at(Pointi{ border.Right() - 1, y });
        }
        
        // iterate through the remainder rectangles at each side of a search rectangle
        {
            // iterate top-left, top-middle and top-right rectangular areas at one pass
            for (int x = search_rect.x; x < search_rect.Right(); ++x)
                for (int y = search_rect.y; y < search_center.y - search_common_rad; ++y)
                    match_templ_at(Pointi{ x, y });

            // iterate bottom-left, bottom-middle and bottom-right rectangular areas at one pass
            for (int x = search_rect.x; x < search_rect.Right(); ++x)
                for (int y = search_center.y + search_common_rad; y < search_rect.Bottom(); ++y)
                    match_templ_at(Pointi{ x, y });

            // iterate left-middle rectangular area
            for (int x = search_rect.x; x < search_center.x - search_common_rad; ++x)
                for (int y = search_center.y - search_common_rad; y < search_center.y + search_common_rad; ++y)
                    match_templ_at(Pointi{ x, y });

            // iterate right-middle rectangular area
            for (int x = search_center.x + search_common_rad; x < search_rect.Right(); ++x)
                for (int y = search_center.y - search_common_rad; y < search_center.y + search_common_rad; ++y)
                    match_templ_at(Pointi{ x, y });
        }

        // correlation coefficient can't be calculated when entire roi is filled with a single color
        TemplateMatchResult result {false};

        if (max_corr_coeff >= -1)
        {
            // preserve fractional coordinates of central pixel
            suriko::Pointi best_match_top_left = best_match_info.templ_top_left;

            const auto& center_offset = sal_pnt.OffsetFromTopLeft();
            suriko::Point2 center{ best_match_top_left.x + center_offset.X(), best_match_top_left.y + center_offset.Y() };

            result.success = true;
            result.center = center;
            result.corr_coef = best_match_info.corr_coef;
#ifdef SRK_DEBUG
            result.top_left = best_match_top_left;
            result.executed_match_templ_calls = best_match_info.executed_match_templ_calls;
#endif
        }
        return result;
    }

    Recti PredictSalientPointSearchRect(DavisonMonoSlam& mono_slam, SalPntId sal_pnt_id, Scalar ellisoid_cut_thr)
    {
        MeanAndCov2D corner = mono_slam.GetSalientPointProjected2DPosWithUncertainty(FilterStageType::Predicted, sal_pnt_id);

        RotatedEllipse2D corner_ellipse = Get2DRotatedEllipseFromCovMat(corner.cov, corner.mean, ellisoid_cut_thr);
        Rect corner_bounds = GetEllipseBounds2(corner_ellipse);
        Recti corner_bounds_i = TruncateRect(corner_bounds);
        return corner_bounds_i;
    }

    // Ensures the size of the rectangle is at least of a given value, keeping the center intact.
    static Recti ClampRectWhenFixedCenter(const Recti& r, suriko::Sizei min_size)
    {
        Recti result = r;
        if (result.width < min_size.width)
        {
            int expand_x = min_size.width - result.width;
            int expand_left_x = expand_x / 2;
            result.x -= expand_left_x;
            result.width = min_size.width;
        }

        if (result.height < min_size.height)
        {
            int expand_y = min_size.height - result.height;
            int expand_up_y = expand_y / 2;
            result.y -= expand_up_y;
            result.height = min_size.height;
        }
        return result;
    }

    std::optional<suriko::Point2> MatchSalientPatch(DavisonMonoSlam& mono_slam, SalPntId sal_pnt_id, const Picture& pic,
        Scalar ellisoid_cut_thr)
    {
        Recti search_rect_unbounded = PredictSalientPointSearchRect(mono_slam, sal_pnt_id, ellisoid_cut_thr);
        
        if (min_search_rect_size_.has_value())
            search_rect_unbounded = ClampRectWhenFixedCenter(search_rect_unbounded, min_search_rect_size_.value());

        Recti image_bounds = { 0, 0, pic.gray.cols, pic.gray.rows };
        
        int radx = mono_slam.sal_pnt_patch_size_.width / 2;
        int rady = mono_slam.sal_pnt_patch_size_.height / 2;
        Recti image_sensitive_portion = DeflateRect(image_bounds, radx, rady, radx, rady);
        
        std::optional<Recti> search_rect_opt = IntersectRects(search_rect_unbounded, image_sensitive_portion);
        if (!search_rect_opt.has_value())
            return std::nullopt; // lost

        const Recti search_rect = search_rect_opt.value();

        static bool debug_template_bounds = false;
        if (debug_template_bounds)
            LOG(INFO) << "patch_bnds=[" << search_rect.x << "," << search_rect.y << "," << search_rect.width << "," << search_rect.height
                << " (" << search_rect.x + search_rect.width/2 <<"," << search_rect.y + search_rect.height/2 << ")";

        const SalPntPatch& sal_pnt = mono_slam.GetSalientPoint(sal_pnt_id);

        TemplateMatchResult match_result = MatchSalientPointTemplCenterInRect(sal_pnt, pic, search_rect);
        if (!match_result.success)
            return std::nullopt;

        static cv::Mat image_with_match_bgr;
        static bool debug_ui = false;
        if (debug_ui)
        {
            CopyBgr(pic, &image_with_match_bgr);

            //if (this->draw_sal_pnt_fun_ != nullptr)
            //    draw_sal_pnt_fun_(mono_slam, sal_pnt, &image_with_match_bgr);

            cv::Rect search_rect_cv{ search_rect.x, search_rect.y, search_rect.width, search_rect.height };
            cv::rectangle(image_with_match_bgr, search_rect_cv, cv::Scalar::all(255));

            // patch bounds in new frame
            suriko::Pointi new_top_left = mono_slam.TemplateTopLeftInt(match_result.center);
            cv::Rect patch_rect{
                new_top_left.x, new_top_left.y,
                mono_slam.sal_pnt_patch_size_.width, mono_slam.sal_pnt_patch_size_.height };
            cv::rectangle(image_with_match_bgr, patch_rect, cv::Scalar(172,172,0));

            if (show_image_fun_ != nullptr)
                show_image_fun_("Match.search_rect", image_with_match_bgr);
        }

        // skip a match with low correlation coefficient
        if (min_templ_corr_coeff_.has_value() && match_result.corr_coef < min_templ_corr_coeff_.value())
        {
            static bool debug_matching = false;
            if (debug_matching)
            {
                MeanAndCov2D predicted_center = kalman_tracker_->GetSalientPointProjected2DPosWithUncertainty(FilterStageType::Predicted, sal_pnt_id);
                VLOG(5) << "Treating sal_pnt(ind=" << sal_pnt.sal_pnt_ind << ")"
                    << " as undetected because corr_coef=" << match_result.corr_coef
                    << " is less than thr=" << min_templ_corr_coeff_.value() << ","
                    << " predicted center_pix=[" << predicted_center.mean[0] << "," << predicted_center.mean[1] << "]";
            }
            return std::nullopt;
        }

#if defined(SRK_DEBUG)
        static bool debug_calls = false;
        if (debug_calls)
        {
            int max_core_match_calls = search_rect.width * search_rect.height;
            LOG(INFO) << "match_err_per_pixel=" << match_result.corr_coef
                << " match_calls=" << match_result.executed_match_templ_calls << "/" << max_core_match_calls
                << "(" << match_result.executed_match_templ_calls / (float)max_core_match_calls << ")";
        }
#endif

        return match_result.center;
    }

    void MatchSalientPoints(size_t frame_ind,
        const Picture& image,
        const std::set<SalPntId>& tracking_sal_pnts,
        std::vector<std::pair<DavisonMonoSlam::SalPntId, CornersMatcherBlobId>>* matched_sal_pnts) override
    {
        for (auto sal_pnt_id : tracking_sal_pnts)
        {
            const SalPntPatch& sal_pnt = kalman_tracker_->GetSalientPoint(sal_pnt_id);

            std::optional<suriko::Point2> match_pnt_center = MatchSalientPatch(*kalman_tracker_, sal_pnt_id, image, ellisoid_cut_thr_);
            bool is_lost = !match_pnt_center.has_value();
            if (is_lost)
                continue;

            const auto& new_center = match_pnt_center.value();

#if defined(SRK_DEBUG)
            bool is_cosecutive_detection = sal_pnt.prev_detection_frame_ind_debug_ + 1 == frame_ind;
            if (is_cosecutive_detection)
            {
                // check that template doesn't jump far away in the consecutive frames
                static float max_shift_per_frame = 30;
                auto diffC = (sal_pnt.prev_detection_templ_center_pix_debug_.Mat() - new_center.Mat()).norm();
                if (diffC > max_shift_per_frame)
                {
                    if (stop_on_sal_pnt_moved_too_far_)
                        SRK_ASSERT(false) << "sal pnt moved to far away";
                    else
                        continue;
                }
            }
#endif
            size_t blob_ind = new_keypoints_.size();
            cv::KeyPoint kp{};
            kp.pt = cv::Point2f{ static_cast<float>(new_center.Mat()[0]), static_cast<float>(new_center.Mat()[1]) };
            new_keypoints_.push_back(kp);

            matched_sal_pnts->push_back(std::make_pair(sal_pnt_id, CornersMatcherBlobId{ blob_ind }));
        }
    }

    void RecruitNewSalientPoints(size_t frame_ind,
        const Picture& image,
        const std::set<SalPntId>& tracking_sal_pnts,
        const std::vector<std::pair<DavisonMonoSlam::SalPntId, CornersMatcherBlobId>>& matched_sal_pnts,
        std::vector<CornersMatcherBlobId>* new_blob_ids) override
    {
        std::vector<cv::KeyPoint> keypoints;
        detector_->detect(image.gray, keypoints);

        cv::Mat descr_per_row;
        detector_->compute(image.gray, keypoints, descr_per_row);

        std::vector<cv::KeyPoint> sparse_keypoints;
        const float rad_diag = std::sqrt(suriko::Sqr(kalman_tracker_->sal_pnt_patch_size_.width) + suriko::Sqr(kalman_tracker_->sal_pnt_patch_size_.height)) / 2;
        FilterOutClosest(keypoints, rad_diag, &sparse_keypoints);

        cv::Mat sparse_img;
        static bool debug_keypoints = false;
        if (debug_keypoints)
            cv::drawKeypoints(image.gray, sparse_keypoints, sparse_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // remove keypoints which are close to 'matched' salient points
        auto filter_out_close_to_existing = [this,&matched_sal_pnts](const std::vector<cv::KeyPoint>& keypoints, float exclude_radius,
            std::vector<cv::KeyPoint>* result)
        {
            for (size_t cand_ind = 0; cand_ind < keypoints.size(); ++cand_ind)
            {
                const auto& cand = keypoints[cand_ind];

                bool has_close_blob = false;
                for (auto[sal_pnt_id, blob_id] : matched_sal_pnts)
                {
                    std::optional<suriko::Point2> exist_pix_opt = kalman_tracker_->GetDetectedSalientPatchCenter(sal_pnt_id);
                    suriko::Point2 exist_pix = exist_pix_opt.value();
                    float dist = std::sqrt(suriko::Sqr(cand.pt.x - exist_pix[0]) + suriko::Sqr(cand.pt.y - exist_pix[1]));
                    if (dist < exclude_radius)
                    {
                        has_close_blob = true;
                        break;
                    }
                }
                if (!has_close_blob)
                    result->push_back(cand);
            }
        };

        new_keypoints_.clear();
        filter_out_close_to_existing(sparse_keypoints, rad_diag, &new_keypoints_);

        cv::Mat img_no_closest;
        if (debug_keypoints)
            cv::drawKeypoints(image.gray, new_keypoints_, img_no_closest, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        for (size_t i=0; i< new_keypoints_.size(); ++i)
        {
            new_blob_ids->push_back(CornersMatcherBlobId{i});
        }
    }

    static void FilterOutClosest(const std::vector<cv::KeyPoint>& keypoints, float exclude_radius, std::vector<cv::KeyPoint>* sparse_keypoints)
    {
        std::vector<char> processed(keypoints.size(), (char)false);
        for (size_t stage_ind = 0; stage_ind < keypoints.size(); ++stage_ind)
        {
            if (processed[stage_ind]) continue;

            const auto& stage = keypoints[stage_ind];
            sparse_keypoints->push_back(stage);
            processed[stage_ind] = (char)true;

            for (size_t i = stage_ind + 1; i < keypoints.size(); ++i)
            {
                if (processed[i]) continue;

                const auto& cand = keypoints[i];
                float dist = std::sqrt(suriko::Sqr(cand.pt.x - stage.pt.x) + suriko::Sqr(cand.pt.y - stage.pt.y));
                if (dist < exclude_radius)
                    processed[i] = (char)true;
            }
        }
    }

    suriko::Point2 GetBlobCoord(CornersMatcherBlobId blob_id) override
    {
        const cv::KeyPoint& kp = new_keypoints_[blob_id.Ind];
        return suriko::Point2{ kp.pt.x, kp.pt.y };
    }

    Picture GetBlobPatchTemplate(CornersMatcherBlobId blob_id, const Picture& image) override
    {
        const cv::KeyPoint& kp = new_keypoints_[blob_id.Ind];

        int rad_x_int = kalman_tracker_->sal_pnt_patch_size_.width / 2;
        int rad_y_int = kalman_tracker_->sal_pnt_patch_size_.height / 2;

        int center_x = (int)kp.pt.x;
        int center_y = (int)kp.pt.y;
        cv::Rect patch_bounds{ center_x - rad_x_int, center_y - rad_y_int, kalman_tracker_->sal_pnt_patch_size_.width, kalman_tracker_->sal_pnt_patch_size_.height };

        cv::Mat patch_gray;
        image.gray(patch_bounds).copyTo(patch_gray);
        SRK_ASSERT(patch_gray.rows == kalman_tracker_->sal_pnt_patch_size_.height);
        SRK_ASSERT(patch_gray.cols == kalman_tracker_->sal_pnt_patch_size_.width);

        Picture patch_template{};
        patch_template.gray = patch_gray;

#if defined(SRK_DEBUG)
        cv::Mat patch;
        image.bgr_debug(patch_bounds).copyTo(patch);
        patch_template.bgr_debug = patch;
#endif
        return patch_template;
    }

    void SetSuppressObservations(bool value) { suppress_observations_ = value; }
};
#endif

#if defined(SRK_HAS_OPENCV)
template <typename EigenMat>
void WriteMatElements(cv::FileStorage& fs, const EigenMat& m)
{
    for (decltype(m.size()) i = 0; i < m.size(); ++i)
        fs << m.data()[i];
}
#endif

bool WriteTrackerInternalsToFile(std::string_view file_name, const DavisonMonoSlamTrackerInternalsHist& hist)
{
#if !defined(SRK_HAS_OPENCV)
    return false;
#else
    cv::FileStorage fs;
    if (!fs.open(file_name.data(), cv::FileStorage::WRITE, "utf8"))
        return false;

    fs << "FramesCount" << static_cast<int>(hist.state_samples.size());
    fs << "AvgFrameProcessingDur" << static_cast<float>(hist.avg_frame_processing_dur.count()); // seconds
    fs << "Frames" <<"[";

    for (const auto& item : hist.state_samples)
    {
        fs << "{";

        cv::write(fs, "CurReprojErr", item.cur_reproj_err);
        cv::write(fs, "EstimatedSalPnts", static_cast<int>(item.estimated_sal_pnts));
        cv::write(fs, "NewSalPnts", static_cast<int>(item.new_sal_pnts));
        cv::write(fs, "CommonSalPnts", static_cast<int>(item.common_sal_pnts));
        cv::write(fs, "DeletedSalPnts", static_cast<int>(item.deleted_sal_pnts));
        cv::write(fs, "FrameProcessingDur", item.frame_processing_dur.count()); // seconds

        Eigen::Map<const Eigen::Matrix<Scalar, 9, 1>> cam_pos_uncert(item.cam_pos_uncert.data());
        fs << "CamPosUnc_s" <<"[:";
        WriteMatElements(fs, cam_pos_uncert);
        fs << "]";

        if (item.sal_pnts_uncert_median.has_value())
        {
            fs << "SalPntUncMedian_s" << "[:";
            WriteMatElements(fs, item.sal_pnts_uncert_median.value());
            fs << "]";
        }

        fs << "}";
    }
    fs << "]";
#endif
    return true;
}

static bool ValidateDirectoryExists(const char *flagname, const std::string &value)
{
    auto test_data_path = std::filesystem::absolute(value);
    if (std::filesystem::is_directory(test_data_path))
        return true;
    std::cout << "directory " << test_data_path.string() << " doesn't exist" << std::endl;
    return false;
}

//static constexpr char* kVirtualCStr = "virtual";
//static constexpr char* kImageSeqDirCStr = "imageseqdir";
//DEFINE_string(scene_source, kVirtualCStr, "{virtual,imageseqdir}");
DEFINE_string(scene_imageseq_dir, "", "Path to directory with image files");
DEFINE_validator(scene_imageseq_dir, &ValidateDirectoryExists);
DEFINE_int32(virtual_scenario, 1, "");
DEFINE_double(world_xmin, -1.5, "world xmin");
DEFINE_double(world_xmax, 1.5, "world xmax");
DEFINE_double(world_ymin, -1.5, "world ymin");
DEFINE_double(world_ymax, 1.5, "world ymax");
DEFINE_double(world_zmin, 0, "world zmin");
DEFINE_double(world_zmax, 1, "world zmax");
DEFINE_double(world_cell_size_x, 0.5, "cell size x");
DEFINE_double(world_cell_size_y, 0.5, "cell size y");
DEFINE_double(world_cell_size_z, 0.5, "cell size z");
DEFINE_double(world_noise_R_std, 0.005, "Standard deviation of noise distribution for R, 0=no noise (eg: 0.01)");
DEFINE_double(world_noise_x3D_std, 0.005, "Standard deviation of noise distribution for salient points, 0=no noise (eg: 0.1)");
DEFINE_double(viewer_eye_offset_x, 4, "");
DEFINE_double(viewer_eye_offset_y, -2.5, "");
DEFINE_double(viewer_eye_offset_z, 7, "");
DEFINE_double(viewer_center_offset_x, 0, "");
DEFINE_double(viewer_center_offset_y, 0, "");
DEFINE_double(viewer_center_offset_z, 0, "");
DEFINE_double(viewer_up_x, 0, "");
DEFINE_double(viewer_up_y, 0, "");
DEFINE_double(viewer_up_z, 1, "");
// virtual scenario2
DEFINE_double(s2_max_deviation, 0.1, "");
DEFINE_int32(s2_num_steps, 100, "");
// virtual scenario3
DEFINE_double(s3_max_deviation, 0.1, "");
DEFINE_double(s3_periods_count, 1.0, "");
DEFINE_int32(s3_shots_per_period, 4, "");
// virtual scenario4
DEFINE_int32(s4_periods_count, 0, "");
DEFINE_double(s4_0_eye_x, 0.0, "");
DEFINE_double(s4_0_eye_y, 0.0, "");
DEFINE_double(s4_0_eye_z, 0.0, "");
DEFINE_double(s4_0_center_x, 0.0, "");
DEFINE_double(s4_0_center_y, 0.0, "");
DEFINE_double(s4_0_center_z, 0.0, "");
DEFINE_double(s4_0_up_x, 0.0, "");
DEFINE_double(s4_0_up_y, 0.0, "");
DEFINE_double(s4_0_up_z, 0.0, "");
DEFINE_double(s4_1_eye_x, 0.0, "");
DEFINE_double(s4_1_eye_y, 0.0, "");
DEFINE_double(s4_1_eye_z, 0.0, "");
DEFINE_double(s4_1_center_x, 0.0, "");
DEFINE_double(s4_1_center_y, 0.0, "");
DEFINE_double(s4_1_center_z, 0.0, "");
DEFINE_double(s4_1_up_x, 0.0, "");
DEFINE_double(s4_1_up_y, 0.0, "");
DEFINE_double(s4_1_up_z, 0.0, "");

DEFINE_int32(viewer_steps_per_side_x, 20, "number of viewer's steps at each side of the rectangle");
DEFINE_int32(viewer_steps_per_side_y, 10, "number of viewer's steps at each side of the rectangle");
DEFINE_double(kalman_estim_var_init_std, 0.001, "");
DEFINE_double(kalman_input_noise_std, 0.08, "");
DEFINE_double(kalman_sal_pnt_init_inv_dist, 1, "");
DEFINE_double(kalman_sal_pnt_init_inv_dist_std, 1, "");
DEFINE_double(kalman_measurm_noise_std_pix, 1, "");
DEFINE_int32(kalman_update_impl, 1, "");
DEFINE_double(kalman_max_new_blobs_in_first_frame, 7, "");
DEFINE_double(kalman_max_new_blobs_per_frame, 1, "");
DEFINE_double(kalman_match_blob_prob, 1, "[0,1] portion of blobs which are matched with ones in the previous frame; 1=all matched, 0=none matched");
DEFINE_int32(kalman_templ_width, 15, "width of patch template");
DEFINE_int32(kalman_templ_min_search_rect_width, 7, "the min width of a rectangle when searching for tempplate in the next frame");
DEFINE_int32(kalman_templ_min_search_rect_height, 7, "");
DEFINE_double(kalman_templ_min_corr_coeff, -1, "");
DEFINE_bool(kalman_stop_on_sal_pnt_moved_too_far, false, "width of patch template");
DEFINE_bool(kalman_fix_estim_vars_covar_symmetry, true, "");
DEFINE_bool(kalman_debug_estim_vars_cov, false, "");
DEFINE_bool(kalman_debug_predicted_vars_cov, false, "");
DEFINE_int32(kalman_debug_max_sal_pnt_count, -1, "[default=-1(none)] number of salient points won't be greater than this value");
DEFINE_bool(kalman_fake_sal_pnt_init_inv_dist, false, "");
DEFINE_double(kalman_ellipsoid_cut_thr, 0.04, "probability cut threshold for uncertainty ellipsoid");
DEFINE_bool(ui_swallow_exc, true, "true to ignore (swallow) exceptions in UI");
DEFINE_int32(ui_loop_prolong_period_ms, 3000, "");
DEFINE_int32(ui_tight_loop_relaxing_delay_ms, 100, "");
DEFINE_int32(ui_dots_per_uncert_ellipse, 12, "Number of dots to split uncertainty ellipse (4=rectangle)");
DEFINE_bool(ctrl_multi_threaded_mode, false, "true for UI to work in a separated dedicated thread; false for UI to work inside worker's thread");
DEFINE_bool(ctrl_wait_after_each_frame, false, "true to wait for keypress after each iteration");
DEFINE_bool(ctrl_debug_skim_over, false, "overview the synthetic world without reconstruction");
DEFINE_bool(ctrl_visualize_during_processing, true, "");
DEFINE_bool(ctrl_visualize_after_processing, true, "");
DEFINE_bool(ctrl_collect_tracker_internals, false, "");
DEFINE_int32(camera_image_width, 320, "");
DEFINE_int32(camera_image_height, 240, "");
DEFINE_double(camera_princip_point_x, 162.0, "");
DEFINE_double(camera_princip_point_y, 125.0, "");
DEFINE_double(camera_focal_length_pix_x, 195.0, "");
DEFINE_double(camera_focal_length_pix_y, 195.0, "");
DEFINE_double(camera_look_from_x, 0.0, "");
DEFINE_double(camera_look_from_y, 0.0, "");
DEFINE_double(camera_look_from_z, 0.0, "");
DEFINE_double(camera_look_to_x, 0.0, "");
DEFINE_double(camera_look_to_y, 0.0, "");
DEFINE_double(camera_look_to_z, 1.0, "");
DEFINE_double(camera_up_x, 0.0, "");
DEFINE_double(camera_up_y, 1.0, "");
DEFINE_double(camera_up_z, 0.0, "");

int DavisonMonoSlamDemo(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);

#if defined(SRK_HAS_OPENCV)
    cv::theRNG().state = 123; // specify seed for OpenCV randomness, so that debugging always goes the same execution path
#endif

#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    LOG(INFO) << "world_noise_x3D_std=" << FLAGS_world_noise_x3D_std;
    LOG(INFO) << "world_noise_R_std=" << FLAGS_world_noise_R_std;

    //
    bool corrupt_salient_points_with_noise = FLAGS_world_noise_x3D_std > 0;
    bool corrupt_cam_orient_with_noise = FLAGS_world_noise_R_std > 0;
    std::vector<SE3Transform> gt_cam_orient_cfw; // ground truth camera orientation transforming into camera from world

    WorldBounds wb{};
    wb.x_min = FLAGS_world_xmin;
    wb.x_max = FLAGS_world_xmax;
    wb.y_min = FLAGS_world_ymin;
    wb.y_max = FLAGS_world_ymax;
    wb.z_min = FLAGS_world_zmin;
    wb.z_max = FLAGS_world_zmax;
    std::array<Scalar, 3> cell_size = { FLAGS_world_cell_size_x, FLAGS_world_cell_size_y, FLAGS_world_cell_size_z };

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    gen.seed(1234);

    std::unique_ptr<std::normal_distribution<Scalar>> x3D_noise_dis;
    if (corrupt_salient_points_with_noise)
        x3D_noise_dis = std::make_unique<std::normal_distribution<Scalar>>(0, FLAGS_world_noise_x3D_std);

    FragmentMap entire_map;
    entire_map.SetFragmentIdOffsetInternal(1000'000);
    GenerateWorldPoints(wb, cell_size, corrupt_salient_points_with_noise, &gen, x3D_noise_dis.get(), &entire_map);
    LOG(INFO) << "points_count=" << entire_map.SalientPointsCount();

    suriko::Point3 viewer_eye_offset(FLAGS_viewer_eye_offset_x, FLAGS_viewer_eye_offset_y, FLAGS_viewer_eye_offset_z);
    suriko::Point3 viewer_center_offset(FLAGS_viewer_center_offset_x, FLAGS_viewer_center_offset_y, FLAGS_viewer_center_offset_z);
    Eigen::Matrix<Scalar, 3, 1> up(FLAGS_viewer_up_x, FLAGS_viewer_up_y, FLAGS_viewer_up_z);

    if (FLAGS_virtual_scenario == 1)
        GenerateCameraShotsAlongRectangularPath(wb, FLAGS_viewer_steps_per_side_x, FLAGS_viewer_steps_per_side_y,
            viewer_eye_offset, viewer_center_offset, up, &gt_cam_orient_cfw);
    else if (FLAGS_virtual_scenario == 2)
        GenerateCameraShotsRightAndLeft(wb, viewer_eye_offset, viewer_center_offset, up,
            FLAGS_s2_max_deviation,
            FLAGS_s2_num_steps,
            &gt_cam_orient_cfw);
    else if (FLAGS_virtual_scenario == 3)
    {
        auto viewer_eye = viewer_eye_offset;
        auto center = viewer_center_offset;
        GenerateCameraShotsOscilateRightAndLeft(wb, viewer_eye, center, up,
            FLAGS_s3_max_deviation,
            FLAGS_s3_periods_count,
            FLAGS_s3_shots_per_period,
            &gt_cam_orient_cfw);
    }
    else if (FLAGS_virtual_scenario == 4)
    {
        std::vector<LookAtComponents> cam_poses;
        cam_poses.push_back(LookAtComponents{
            suriko::Point3{FLAGS_s4_0_eye_x,FLAGS_s4_0_eye_y,FLAGS_s4_0_eye_z},
            suriko::Point3{FLAGS_s4_0_center_x,FLAGS_s4_0_center_y,FLAGS_s4_0_center_z},
            suriko::Point3{FLAGS_s4_0_up_x,FLAGS_s4_0_up_y,FLAGS_s4_0_up_z}
        });
        cam_poses.push_back(LookAtComponents{
            suriko::Point3{FLAGS_s4_1_eye_x,FLAGS_s4_1_eye_y,FLAGS_s4_1_eye_z},
            suriko::Point3{FLAGS_s4_1_center_x,FLAGS_s4_1_center_y,FLAGS_s4_1_center_z},
            suriko::Point3{FLAGS_s4_1_up_x,FLAGS_s4_1_up_y,FLAGS_s4_1_up_z}
        });
        GenerateCameraShots3DPath(wb, cam_poses, FLAGS_s4_periods_count, &gt_cam_orient_cfw);
    }

    std::vector<SE3Transform> gt_cam_orient_wfc;
    std::transform(gt_cam_orient_cfw.begin(), gt_cam_orient_cfw.end(), std::back_inserter(gt_cam_orient_wfc), [](auto& t) { return SE3Inv(t); });

    if (corrupt_cam_orient_with_noise)
    {
        std::normal_distribution<Scalar> cam_orient_noise_dis(0, FLAGS_world_noise_R_std);
        for (SE3Transform& cam_orient : gt_cam_orient_cfw)
        {
            Eigen::Matrix<Scalar, 3, 1> dir;
            if (AxisAngleFromRotMat(cam_orient.R, &dir))
            {
                Scalar d1 = cam_orient_noise_dis(gen);
                Scalar d2 = cam_orient_noise_dis(gen);
                Scalar d3 = cam_orient_noise_dis(gen);
                dir[0] += d1;
                dir[1] += d2;
                dir[2] += d3;

                Eigen::Matrix<Scalar, 3, 3> newR;
                if (RotMatFromAxisAngle(dir, &newR))
                    cam_orient.R = newR;
            }
        }
    }

    size_t frames_count = gt_cam_orient_cfw.size();
    LOG(INFO) << "frames_count=" << frames_count;
#endif

    // focal_len_pix = focal_len_mm / pixel_size_mm
    std::array<Scalar, 2> foc_len_pix = { FLAGS_camera_focal_length_pix_x, FLAGS_camera_focal_length_pix_y };

    // assume dy=PixelSizeMm[1]=some constant
    const float pix_size_y = 0.001f;

    const float focal_length_mm = foc_len_pix[1] * pix_size_y;
    float pix_size_x = focal_length_mm / foc_len_pix[0];

    CameraIntrinsicParams cam_intrinsics;
    cam_intrinsics.image_size = { FLAGS_camera_image_width, FLAGS_camera_image_height };
    cam_intrinsics.principal_point_pix = { (Scalar)FLAGS_camera_princip_point_x, (Scalar)FLAGS_camera_princip_point_y };
    cam_intrinsics.focal_length_mm = focal_length_mm;
    cam_intrinsics.pixel_size_mm = { pix_size_x , pix_size_y };

    LOG(INFO) << "img_size=[" << cam_intrinsics.image_size.width << "," << cam_intrinsics.image_size.height << "] pix";
    LOG(INFO) << "foc_len="
        << cam_intrinsics.focal_length_mm << " mm" << " PixelSize[dx,dy]=[" 
        << cam_intrinsics.pixel_size_mm[0] << ","
        << cam_intrinsics.pixel_size_mm[1] << "] mm";
    LOG(INFO) << " PrincipPoint[Cx,Cy]=["
        << cam_intrinsics.principal_point_pix[0] << "," 
        << cam_intrinsics.principal_point_pix[1] << "] pix";

    std::array<Scalar, 2> f_pix = cam_intrinsics.FocalLengthPix();
    LOG(INFO) << "foc_len[alphax,alphay]=[" << f_pix[0] << "," << f_pix[1] << "] pix";

    RadialDistortionParams cam_distort_params;
    cam_distort_params.k1 = 0;
    cam_distort_params.k2 = 0;

    //
    DavisonMonoSlam2DDrawer drawer;
    drawer.dots_per_uncert_ellipse_ = FLAGS_ui_dots_per_uncert_ellipse;
    drawer.ellipse_cut_thr_ = FLAGS_kalman_ellipsoid_cut_thr;

    // the origin of a tracker (sometimes cam0)
    SE3Transform tracker_origin_from_world;
#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    // tracker coordinate system = cam0
    tracker_origin_from_world = gt_cam_orient_cfw[0];
#elif DEMO_DATA_SOURCE_TYPE == kImageSeqDir
    // tracker coordinates system = world coordinate system
    tracker_origin_from_world.R.setIdentity();
    tracker_origin_from_world.T.setZero();
#endif

    DavisonMonoSlam::DebugPathEnum debug_path = DavisonMonoSlam::DebugPathEnum::DebugNone;
    if (FLAGS_kalman_debug_estim_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugEstimVarsCov;
    if (FLAGS_kalman_debug_predicted_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugPredictedVarsCov;
    DavisonMonoSlam::SetDebugPath(debug_path);

    DavisonMonoSlam mono_slam{ };
    mono_slam.in_multi_threaded_mode_ = FLAGS_ctrl_multi_threaded_mode;
    mono_slam.between_frames_period_ = 1;
    mono_slam.cam_intrinsics_ = cam_intrinsics;
    mono_slam.cam_distort_params_ = cam_distort_params;
    mono_slam.sal_pnt_init_inv_dist_ = FLAGS_kalman_sal_pnt_init_inv_dist;
    mono_slam.sal_pnt_init_inv_dist_std_ = FLAGS_kalman_sal_pnt_init_inv_dist_std;
    mono_slam.SetInputNoiseStd(FLAGS_kalman_input_noise_std);
    mono_slam.measurm_noise_std_pix_ = FLAGS_kalman_measurm_noise_std_pix;
    mono_slam.sal_pnt_patch_size_ = { FLAGS_kalman_templ_width, FLAGS_kalman_templ_width };

#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    mono_slam.sal_pnt_small_std_ = FLAGS_kalman_estim_var_init_std;
    mono_slam.SetCamera(SE3Transform::NoTransform(), FLAGS_kalman_estim_var_init_std);
#elif DEMO_DATA_SOURCE_TYPE == kImageSeqDir
    SE3Transform cam_cfw = SE3Inv(LookAtLufWfc(
        { (Scalar)FLAGS_camera_look_from_x, (Scalar)FLAGS_camera_look_from_y, (Scalar)FLAGS_camera_look_from_z },
        { (Scalar)FLAGS_camera_look_to_x, (Scalar)FLAGS_camera_look_to_y, (Scalar)FLAGS_camera_look_to_z },
        { (Scalar)FLAGS_camera_up_x, (Scalar)FLAGS_camera_up_y, (Scalar)FLAGS_camera_up_z }));
    mono_slam.SetCamera(cam_cfw, FLAGS_kalman_estim_var_init_std);
#endif

    mono_slam.kalman_update_impl_ = FLAGS_kalman_update_impl;
    mono_slam.fix_estim_vars_covar_symmetry_ = FLAGS_kalman_fix_estim_vars_covar_symmetry;
    mono_slam.debug_ellipsoid_cut_thr_ = FLAGS_kalman_ellipsoid_cut_thr;
    if (FLAGS_kalman_debug_max_sal_pnt_count != -1)
        mono_slam.debug_max_sal_pnt_coun_ = FLAGS_kalman_debug_max_sal_pnt_count;
#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    mono_slam.fake_sal_pnt_initial_inv_dist_ = FLAGS_kalman_fake_sal_pnt_init_inv_dist;
    mono_slam.gt_cami_from_world_fun_ = [&gt_cam_orient_cfw](size_t frame_ind) -> SE3Transform
    {
        SE3Transform c = gt_cam_orient_cfw[frame_ind];
        return c;
    };    
    mono_slam.gt_cami_from_tracker_new_ = [&gt_cam_orient_cfw](SE3Transform tracker_from_world, size_t frame_ind) -> SE3Transform
    {
        SE3Transform cami_from_world = gt_cam_orient_cfw[frame_ind];
        SE3Transform cami_from_tracker = SE3AFromB(cami_from_world, tracker_from_world);
        return cami_from_tracker;
    };    
    mono_slam.gt_cami_from_tracker_fun_ = [&gt_cam_orient_cfw, tracker_origin_from_world](size_t frame_ind) -> SE3Transform
    {
        SE3Transform c = CurCamFromTrackerOrigin(gt_cam_orient_cfw, frame_ind, tracker_origin_from_world);
        return c;
    };    
    mono_slam.gt_sal_pnt_in_camera_fun_ = [&entire_map]
        (SE3Transform tracker_from_world, SE3Transform camera_from_tracker, DavisonMonoSlam::SalPntId sal_pnt_id) -> Dir3DAndDistance
    {
        std::optional<size_t> frag_id = DemoGetSalPntFramgmentId(entire_map, sal_pnt_id);
        const SalientPointFragment& fragment = entire_map.GetSalientPointNew(frag_id.value());
        suriko::Point3 pnt_world = fragment.coord.value();

        SE3Transform camera_from_world = SE3Compose(camera_from_tracker, tracker_from_world);
        suriko::Point3 pnt_camera = SE3Apply(camera_from_world, pnt_world);
        
        Eigen::Matrix<Scalar, 3, 1> pnt_mat = pnt_camera.Mat(); 
        Dir3DAndDistance p;
        p.unity_dir = pnt_mat.normalized();
        p.dist = pnt_mat.norm();
        return p;
    };
#endif
    mono_slam.PredictEstimVarsHelper();
    LOG(INFO) << "kalman_update_impl=" << FLAGS_kalman_update_impl;

#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    {
        auto corners_matcher = std::make_unique<DemoCornersMatcher>(&mono_slam, gt_cam_orient_cfw, entire_map, cam_intrinsics.image_size);
        corners_matcher->tracker_origin_from_world_ = tracker_origin_from_world;

        if (FLAGS_kalman_max_new_blobs_in_first_frame > 0)
            corners_matcher->max_new_blobs_in_first_frame_ = FLAGS_kalman_max_new_blobs_in_first_frame;
        if (FLAGS_kalman_max_new_blobs_per_frame > 0)
            corners_matcher->max_new_blobs_per_frame_ = FLAGS_kalman_max_new_blobs_per_frame;
        if (FLAGS_kalman_match_blob_prob > 0)
            corners_matcher->match_blob_prob_ = FLAGS_kalman_match_blob_prob;

        mono_slam.SetCornersMatcher(std::move(corners_matcher));
    }
#elif DEMO_DATA_SOURCE_TYPE == kImageSeqDir
    {
        auto corners_matcher = std::make_unique<ImagePatchCornersMatcher>(&mono_slam);
        corners_matcher->stop_on_sal_pnt_moved_too_far_ = FLAGS_kalman_stop_on_sal_pnt_moved_too_far;
        corners_matcher->ellisoid_cut_thr_ = FLAGS_kalman_ellipsoid_cut_thr;
        corners_matcher->min_search_rect_size_ = suriko::Sizei{ FLAGS_kalman_templ_min_search_rect_width, FLAGS_kalman_templ_min_search_rect_height };
        if (FLAGS_kalman_templ_min_corr_coeff > -1)
            corners_matcher->min_templ_corr_coeff_ = FLAGS_kalman_templ_min_corr_coeff;
        corners_matcher->draw_sal_pnt_fun_ = [&drawer](DavisonMonoSlam& mono_slam, SalPntId sal_pnt_id, cv::Mat* out_image_bgr)
        {
            drawer.DrawEstimatedSalientPoint(mono_slam, sal_pnt_id, out_image_bgr);
        };
        corners_matcher->show_image_fun_ = [](std::string_view wnd_name, const cv::Mat& image_bgr)
        {
#if defined(SRK_HAS_OPENCV)
            cv::imshow(wnd_name.data(), image_bgr);
#endif
        };
        mono_slam.SetCornersMatcher(std::move(corners_matcher));
    }
#endif

    if (FLAGS_ctrl_collect_tracker_internals)
    {
        mono_slam.SetStatsLogger(std::make_unique<DavisonMonoSlamInternalsLogger>(&mono_slam));
    }

#if defined(SRK_HAS_OPENCV)
    cv::Mat camera_image_bgr = cv::Mat::zeros(cam_intrinsics.image_size.height, (int)cam_intrinsics.image_size.width, CV_8UC3);
#endif
#if defined(SRK_HAS_PANGOLIN)
    // across threads shared data
    auto worker_chat = std::make_shared<WorkerChatSharedState>();
    ptrdiff_t observable_frame_ind = -1; // this is visualized by UI, it is one frame less than current frame
    std::vector<SE3Transform> cam_orient_cfw_history; // the actual trajectory of the tracker

    UIThreadParams ui_params {};
    ui_params.wait_for_user_input_after_each_frame = FLAGS_ctrl_wait_after_each_frame;
    ui_params.mono_slam = &mono_slam;
    ui_params.tracker_origin_from_world = tracker_origin_from_world;
    ui_params.ellipsoid_cut_thr = FLAGS_kalman_ellipsoid_cut_thr;
    ui_params.cam_orient_cfw_history = &cam_orient_cfw_history;
    ui_params.get_observable_frame_ind_fun = [&observable_frame_ind]() { return observable_frame_ind; };
    ui_params.worker_chat = worker_chat;
    ui_params.ui_swallow_exc = FLAGS_ui_swallow_exc;
    ui_params.ui_tight_loop_relaxing_delay = std::chrono::milliseconds(FLAGS_ui_tight_loop_relaxing_delay_ms);
#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    ui_params.entire_map = &entire_map;
    ui_params.gt_cam_orient_cfw = &gt_cam_orient_cfw;
#endif

    static constexpr int kKeyForward = static_cast<int>('f'); // 'Forward'
    static constexpr int kKeyIgnoreDetection = static_cast<int>('s'); // 'Skip'
    static constexpr int kKeySetToGroundTruth = static_cast<int>('u');
    static constexpr int kKeyDumpInfo = static_cast<int>('i');

    std::thread ui_thread;
    const bool defer_ui_construction = true;
    auto pangolin_gui = SceneVisualizationPangolinGui::New(defer_ui_construction);  // used in single threaded mode
    if (FLAGS_ctrl_visualize_during_processing)
    {
        if (FLAGS_ctrl_multi_threaded_mode)
            ui_thread = std::thread(SceneVisualizationThread, ui_params);
        else
        {
            SceneVisualizationPangolinGui::s_ui_params_ = ui_params;
            pangolin_gui->ui_loop_prolong_period_ = std::chrono::milliseconds(FLAGS_ui_loop_prolong_period_ms);
            pangolin_gui->ui_tight_loop_relaxing_delay_ = std::chrono::milliseconds(FLAGS_ui_tight_loop_relaxing_delay_ms);
            pangolin_gui->dots_per_uncert_ellipse_ = FLAGS_ui_dots_per_uncert_ellipse;
            pangolin_gui->cam_instrinsics_ = cam_intrinsics;
            pangolin_gui->allowed_key_pressed_codes_ = { kKeyForward, kKeyIgnoreDetection, 
                kKeySetToGroundTruth, kKeyDumpInfo };
            pangolin_gui->key_pressed_handler_ = nullptr; // initialized lazily later
            pangolin_gui->InitUI();
            int back_dist = 5;
            pangolin_gui->SetCameraBehindTrackerOnce(tracker_origin_from_world, back_dist);
        }
    }
#endif

    Picture image;
#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
#elif DEMO_DATA_SOURCE_TYPE == kImageSeqDir
    LOG(INFO) << "imageseq_dir=" << FLAGS_scene_imageseq_dir;

    size_t frame_ind = -1;
    auto dir = std::filesystem::directory_iterator(FLAGS_scene_imageseq_dir);
    for (const auto& dir_entry : dir)
#endif
    {
#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
#elif DEMO_DATA_SOURCE_TYPE == kImageSeqDir
        ++frame_ind;
        auto image_file_path = dir_entry.path();
        auto path_str = image_file_path.string();
        LOG(INFO) << path_str;
        
        cv::Mat image_bgr = cv::imread(image_file_path.string());
        bool match_size =
            image_bgr.cols == cam_intrinsics.image_size.width &&
            image_bgr.rows == cam_intrinsics.image_size.height;
        if (!match_size)
        {
            LOG(ERROR)
                << "got image of sizeWxH=[" << image_bgr.cols << "," << image_bgr.rows << "] "
                << "but expected sizeWxH=[" << cam_intrinsics.image_size.width << "," << cam_intrinsics.image_size.height << "]";
            break;
        }

        cv::Mat image_gray;
        cv::cvtColor(image_bgr, image_gray, CV_BGR2GRAY);
        image.gray = image_gray;
#if defined(SRK_DEBUG)
        image.bgr_debug = image_bgr;
#endif
#endif
        auto& corners_matcher = mono_slam.CornersMatcher();
        corners_matcher.AnalyzeFrame(frame_ind, image);

        std::optional<std::chrono::duration<double>> frame_process_time; // time it took to process current frame by tracker

        // process the frame
        if (!FLAGS_ctrl_debug_skim_over)
        {
            auto t1 = std::chrono::high_resolution_clock::now();

            mono_slam.ProcessFrame(frame_ind, image);

            auto t2 = std::chrono::high_resolution_clock::now();
            frame_process_time = t2 - t1;

            CameraStateVars cam_state = mono_slam.GetCameraEstimatedVars();
            SE3Transform actual_cam_wfc = CamWfc(cam_state);
            SE3Transform actual_cam_cfw = SE3Inv(actual_cam_wfc);
#if defined(SRK_HAS_PANGOLIN)
            cam_orient_cfw_history.push_back(actual_cam_cfw);
#endif

#if defined(SRK_HAS_PANGOLIN)
            observable_frame_ind = frame_ind;
#endif
        }

        std::optional<std::chrono::duration<double>> frame_OpenCV_gui_time;

#if defined(SRK_HAS_OPENCV)
        if (FLAGS_ctrl_visualize_during_processing)
        {
#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
            camera_image_bgr.setTo(0);
            const SE3Transform& rt_cfw = gt_cam_orient_cfw[frame_ind];
            auto project_fun = [&rt_cfw, &mono_slam](const suriko::Point3& sal_pnt_world) -> Eigen::Matrix<suriko::Scalar, 3, 1>
            {
                suriko::Point3 pnt_cam = SE3Apply(rt_cfw, sal_pnt_world);
                suriko::Point2 pnt_pix = mono_slam.ProjectCameraPoint(pnt_cam);
                return Eigen::Matrix<suriko::Scalar, 3, 1>(pnt_pix[0], pnt_pix[1], 1);
            };
            constexpr Scalar f0 = 1;
            suriko_demos::Draw2DProjectedAxes(f0, project_fun, &camera_image_bgr);

            //
            auto a_corners_matcher = dynamic_cast<DemoCornersMatcher*>(&corners_matcher);
            if (a_corners_matcher != nullptr)
            {
                for (const BlobInfo& blob_info : a_corners_matcher->DetectedBlobs())
                {
                    Scalar pix_x = blob_info.Coord[0];
                    Scalar pix_y = blob_info.Coord[1];
                    camera_image_bgr.at<cv::Vec3b>((int)pix_y, (int)pix_x) = cv::Vec3b(0xFF, 0xFF, 0xFF);
                }
            }

            std::stringstream strbuf;
            strbuf << "f=" << frame_ind;
            cv::putText(camera_image_bgr, cv::String(strbuf.str()), cv::Point(10, (int)cam_intrinsics.image_size.height - 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
#elif DEMO_DATA_SOURCE_TYPE == kImageSeqDir
            auto t1 = std::chrono::high_resolution_clock::now();

            drawer.DrawScene(mono_slam, image_bgr, &camera_image_bgr);

            auto t2 = std::chrono::high_resolution_clock::now();
            frame_OpenCV_gui_time = t2 - t1;
#endif
            cv::imshow("front-camera", camera_image_bgr);
            cv::waitKey(1); // allow to refresh an opencv view
        }
#endif
        std::optional<std::chrono::duration<double>> frame_Pangolin_gui_time;

#if defined(SRK_HAS_PANGOLIN)
        if (FLAGS_ctrl_multi_threaded_mode)
        {
            // check if UI requests the exit
            std::lock_guard<std::mutex> lk(worker_chat->the_mutex);
            std::optional<WorkerChatMessage> msg = PopMsgUnderLock(&worker_chat->worker_message);
            if (msg == WorkerChatMessage::WorkerExit)
                break;
        }

        auto pangolin_key_handler = [&mono_slam, &frame_ind](int key) -> bool
        {
            switch (key)
            {
            case kKeySetToGroundTruth:
                mono_slam.SetStateToGroundTruth(frame_ind);
                return true;
            case kKeyDumpInfo:
                std::ostringstream os;
                mono_slam.DumpTrackerState(os);
                LOG(INFO) << os.str();
                return true;
            }
            return false;
        };

        // update UI
        if (FLAGS_ctrl_visualize_during_processing)
        {
            auto key_of_interest = [](int key)
            {
                return key == kKeyIgnoreDetection || key == kKeyForward || key == pangolin::PANGO_KEY_ESCAPE;
            };

            if (FLAGS_ctrl_wait_after_each_frame)
            {
                // let a user observe the UI and signal back when to continue

                int key = -1;
                if (FLAGS_ctrl_multi_threaded_mode)
                {
                    std::unique_lock<std::mutex> ulk(worker_chat->the_mutex);
                    worker_chat->ui_message = UIChatMessage::UIWaitKey; // reset the waiting flag
                    worker_chat->ui_wait_key_predicate_ = key_of_interest;

                    // wait till UI requests to resume processing
                    worker_chat->worker_got_new_message_cv.wait(ulk, [&worker_chat] {return worker_chat->worker_message == WorkerChatMessage::WorkerKeyPressed; });
                    key = worker_chat->ui_pressed_key.value_or(-1);
                }
                else
                {
                    // initialize GUI lazily here, because the handler depends on frame_ind which is not known during initialization
                    if (pangolin_gui->key_pressed_handler_ == nullptr)
                        pangolin_gui->key_pressed_handler_ = pangolin_key_handler;

                    key = pangolin_gui->WaitKey(key_of_interest);
                }

                if (key == pangolin::PANGO_KEY_ESCAPE)
                    break;
                
                const bool suppress_observations = (key == kKeyIgnoreDetection);

                auto a_corners_matcher = dynamic_cast<DemoCornersMatcher*>(&mono_slam.CornersMatcher());
                if (a_corners_matcher != nullptr)
                    a_corners_matcher->SetSuppressObservations(suppress_observations);
            }
            else
            {
                if (FLAGS_ctrl_multi_threaded_mode) {}
                else
                {
                    // in single thread mode the controller executes the tracker and gui code sequentially in the same thread
                    auto break_on = [](int key) { return key == pangolin::PANGO_KEY_ESCAPE; };;

                    // initialize GUI lazily here, because the handler depends on frame_ind which is not known during initialization
                    if (pangolin_gui->key_pressed_handler_ == nullptr)
                        pangolin_gui->key_pressed_handler_ = pangolin_key_handler;

                    auto t1 = std::chrono::high_resolution_clock::now();

                    std::optional<int> key = pangolin_gui->RenderFrameAndProlongUILoopOnUserInput(break_on);

                    auto t2 = std::chrono::high_resolution_clock::now();
                    frame_Pangolin_gui_time = t2 - t1;

                    if (key == pangolin::PANGO_KEY_ESCAPE)
                        break;
                }
            }
        }
#endif
#if defined(SRK_HAS_OPENCV)
        if (FLAGS_ctrl_visualize_during_processing)
        {
            cv::waitKey(1); // wait for a moment to allow OpenCV to redraw the image
        }
#endif
        auto zero_time = std::chrono::seconds{ 0 };
        auto total_time = 
            frame_process_time.value_or(zero_time) +
            frame_OpenCV_gui_time.value_or(zero_time) +
            frame_Pangolin_gui_time.value_or(zero_time);
        VLOG(4) << "done f=" << frame_ind
            << " fps=" << (frame_process_time.has_value() ? 1 / frame_process_time.value().count() : 0.0f)
            << "(core+gui=" << 1 / total_time.count() << "fps)"
            << " t=" << std::chrono::duration_cast<std::chrono::milliseconds>(frame_process_time.value_or(zero_time)).count() <<"ms"
            << "(core+gui=" << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() <<"ms)"
            << " #SP=" << mono_slam.SalientPointsCount();
    } // for each frame

    VLOG(4) << "Finished processing all the frames";

#if defined(SRK_HAS_PANGOLIN)
    if (FLAGS_ctrl_visualize_after_processing)
    {
        if (FLAGS_ctrl_multi_threaded_mode)
        {
            if (!ui_thread.joinable())  // don't create thread second time
            {
                ui_thread = std::thread(SceneVisualizationThread, ui_params);
            }

            if (ui_thread.joinable())
            {
                VLOG(4) << "Waiting for UI to request the exit";
                {
                    // wait for Pangolin UI to request the exit
                    std::unique_lock<std::mutex> ulk(worker_chat->the_mutex);
                    worker_chat->ui_message = UIChatMessage::UIWaitKey;
                    worker_chat->ui_wait_key_predicate_ = [](int key) { return key == pangolin::PANGO_KEY_ESCAPE; };
                    worker_chat->worker_got_new_message_cv.wait(ulk, [&worker_chat] {return worker_chat->worker_message == WorkerChatMessage::WorkerKeyPressed; });
                }
                VLOG(4) << "Got UI notification to exit working thread";
                {
                    // notify Pangolin UI to finish visualization thread
                    std::lock_guard<std::mutex> lk(worker_chat->the_mutex);
                    worker_chat->ui_message = UIChatMessage::UIExit;
                }
                VLOG(4) << "Waiting for UI to perform the exit";
                ui_thread.join();
                VLOG(4) << "UI thread has been shut down";
            }
        }
        else
        {
            if (FLAGS_ctrl_wait_after_each_frame)
            {
                // if there was pause after each frame then we won't wait again
            }
            else
            {
                pangolin_gui->WaitKey();
            }
        }
    }
#elif defined(SRK_HAS_OPENCV)
    cv::waitKey(0); // 0=wait forever
#endif

        //
    const DavisonMonoSlamTrackerInternalsHist& internal_stats = mono_slam.StatsLogger()->BuildStats();

    bool dump_op = WriteTrackerInternalsToFile("davison_tracker_internals.json", internal_stats);
    if (FLAGS_ctrl_collect_tracker_internals && !dump_op)
        LOG(ERROR) << "Can't dump the tracker's internals";

    return 0;
}
}

int main(int argc, char* argv[])
{
    int result = 0;
    result = suriko_demos_davison_mono_slam::DavisonMonoSlamDemo(argc, argv);
    return result;
}