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
#include <random>
#include <tuple>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <sstream>
#include <filesystem>
#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "suriko/adapt/tum-dataset.h"
#include "suriko/rt-config.h"
#include "suriko/bundle-adj-kanatani.h"
#include "suriko/obs-geom.h"
#include "suriko/mat-serialization.h"
#include "suriko/templ-match.h"
#include "suriko/approx-alg.h"
#include "suriko/config-reader.h"
#include "suriko/davison-mono-slam.h"
#include "suriko/virt-world/scene-generator.h"
#include "suriko/quat.h"
#include "suriko/stat-helpers.h"
#include "../visualize-helpers.h"

#if defined(SRK_PARALLEL_ENGINE)
#include <execution>
#endif

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/imgcodecs.hpp> // cv::imread
#include <opencv2/imgproc.hpp> // cv::circle, cv::cvtColor
#include <opencv2/highgui.hpp> // cv::imshow
#include <opencv2/features2d.hpp> // cv::ORB
#include <opencv2/calib3d.hpp>  // cv::undistort
#endif

#include "demo-davison-mono-slam-ui.h"

namespace suriko_demos_davison_mono_slam
{
using namespace std;
using namespace suriko; 
using namespace suriko::adapt::tum;
using namespace suriko::internals;
using namespace suriko::virt_world;
using namespace suriko::config;

// Specify data source for demo: virtual scene data or sequence of images in a directory.
enum class DemoDataSource { kVirtualScene, kImageSeqDir };

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
    const Point3& up,
    std::vector<std::optional<SE3Transform>>* cam_orient_cfw)
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

        Point3 step = (base2 - base1) / steps_per_side;

        // to avoid repeating the adjacent point of two consecutive segments, for each segment,
        // the last point is not included because
        // it will be included as the first point of the next segment
        for (size_t step_ind = 0; step_ind < steps_per_side; ++step_ind)
        {
            suriko::Point3 cur_point = base1 + step * step_ind;

            auto wfc = LookAtLufWfc(
                cur_point + eye_offset,
                cur_point + center_offset,
                up);

            SE3Transform RT = SE3Inv(wfc);

            cam_orient_cfw->push_back(RT);
        }
    }
}

void GenerateWorldPoints(WorldBounds wb, const std::array<Scalar, 3>& cell_size, Scalar z_ascent,
    bool corrupt_salient_points_with_noise,
    std::mt19937* gen,
    std::normal_distribution<Scalar>* x3D_noise_dis, FragmentMap* entire_map)
{
    size_t next_virtual_point_id = 6000'000 + 1;
    constexpr Scalar inclusive_gap = 1e-8f; // small value to make iteration inclusive


    Scalar xmid = (wb.x_min + wb.x_max) / 2;
    Scalar xlen = wb.x_max - wb.x_min;
    for (Scalar grid_z = wb.z_min; grid_z < wb.z_max + inclusive_gap; grid_z += cell_size[2])
    {
        for (Scalar grid_y = wb.y_min; grid_y < wb.y_max + inclusive_gap; grid_y += cell_size[1])
        {
            for (Scalar grid_x = wb.x_min; grid_x < wb.x_max + inclusive_gap; grid_x += cell_size[0])
            {
                Scalar x = grid_x;
                Scalar y = grid_y;

                Scalar z_perc = std::cos((x - xmid) / xlen * Pi<Scalar>());
                Scalar z = grid_z + z_perc * z_ascent;

                // jit x and y so the points can be distinguished during movement
                if (corrupt_salient_points_with_noise)
                {
                    x += (*x3D_noise_dis)(*gen);
                    y += (*x3D_noise_dis)(*gen);
                    z += (*x3D_noise_dis)(*gen);
                }

                SalientPointFragment& frag = entire_map->AddSalientPointTempl(Point3(x, y, z));
                frag.synthetic_virtual_point_id = next_virtual_point_id++;
            }
        }
    }
}

bool GetSyntheticCameraInitialMovement(const std::vector<std::optional<SE3Transform>>& gt_cam_orient_cfw,
    suriko::Point3* cam_vel_tracker,
    suriko::Point3* cam_ang_vel_c)
{
    if (gt_cam_orient_cfw.size() < 2)
        return false;
    if (!gt_cam_orient_cfw[0].has_value() || !gt_cam_orient_cfw[1].has_value())
        return false;
    SE3Transform c0_from_world = gt_cam_orient_cfw[0].value();
    SE3Transform c1_from_world = gt_cam_orient_cfw[1].value();
    SE3Transform world_from_c0 = SE3Inv(c0_from_world);
    SE3Transform world_from_c1 = SE3Inv(c1_from_world);

    // In a synthetic scenario we can perfectly foresee the movement of camera (from frame 0 to 1).
    // When processing frame 1, the residual should be zero.
    // camera's velocity
    // Tw1=Tw0+v01_w
    // v01_w=velocity from camera-0 to camera-1 in world coordinates
    auto init_shift_world = world_from_c1.T - world_from_c0.T;
    auto init_shift_tracker = c0_from_world.R * init_shift_world;
    *cam_vel_tracker = init_shift_tracker;

    // camera's angular velocity
    // Rw1=Rw0*R01, R01=delta, which rotates from camera-1 to camera-0.
    SE3Transform c0_from_c1 = SE3AFromB(c0_from_world, c1_from_world);

    Point3 axisangle_c0_from_c1;
    bool op = AxisAngleFromRotMat(c0_from_c1.R, &axisangle_c0_from_c1);
    if (!op)
        Fill(0, &axisangle_c0_from_c1);
    *cam_ang_vel_c = suriko::Point3{ axisangle_c0_from_c1[0], axisangle_c0_from_c1[1], axisangle_c0_from_c1[2] };
    return true;
}

/// Gets the transformation from world into camera in given frame.
SE3Transform CurCamFromTrackerOrigin(const std::vector<std::optional<SE3Transform>>& gt_cam_orient_cfw, size_t frame_ind, const SE3Transform& tracker_from_world)
{
    const SE3Transform& cur_cam_cfw = gt_cam_orient_cfw[frame_ind].value();
    SE3Transform rt_cft = SE3AFromB(cur_cam_cfw, tracker_from_world);  // current camera in the coordinates of the first camera
    return rt_cft;
}

suriko::Point3 PosTrackerOriginFromWorld(const std::vector<std::optional<SE3Transform>>& gt_cam_orient_cfw, suriko::Point3 p_world,
    const SE3Transform& tracker_from_world)
{
    suriko::Point3 p_tracker = SE3Apply(tracker_from_world, p_world);
    return p_tracker;
}

struct BlobInfo
{
    suriko::Point2f Coord;
    DavisonMonoSlam::SalPntId SalPntIdInTracker;
    std::optional<Scalar> GTInvDepth; // inversed ground truth depth to salient point in virtual environments
    size_t FragmentId; // the id of a fragment in entire map
};

class DemoCornersMatcher : public CornersMatcherBase
{
    const std::vector<std::optional<SE3Transform>>& gt_cam_orient_cfw_;
    FragmentMap& entire_map_;
    suriko::Sizei img_size_;
    const DavisonMonoSlam* mono_slam_;
    std::vector<BlobInfo> detected_blobs_; // blobs detected in current frame
public:
    SE3Transform tracker_origin_from_world_;
    std::optional<size_t> max_new_blobs_per_frame_;
    std::optional<size_t> max_new_blobs_in_first_frame_;
    std::optional<float> match_blob_prob_ = 1.0f; // [0,1] portion of blobs which are matched with ones in the previous frame; 1=all matched, 0=none matched;
    std::mt19937 gen_{292};
    std::uniform_real_distribution<float> uniform_distr_{};
    std::normal_distribution<float> templ_center_detection_noise_distr_{};
    float templ_center_detection_noise_std_ = 0;  // 0=no noise; added to blob's center to mimic measurement noise of a filter
public:
    DemoCornersMatcher(const DavisonMonoSlam* mono_slam, const std::vector<std::optional<SE3Transform>>& gt_cam_orient_cfw, FragmentMap& entire_map,
        const suriko::Sizei& img_size)
        : mono_slam_(mono_slam),
        gt_cam_orient_cfw_(gt_cam_orient_cfw),
        entire_map_(entire_map),
        img_size_(img_size)
    {
        Fill(0, &tracker_origin_from_world_.T);
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
            suriko::Point2f pnt_pix = mono_slam_->ProjectCameraPoint(pnt_camera);

            // note: pixel's coordinate is fractional, eg. [252.345,100.273]

            Scalar pix_x = pnt_pix.X();
            Scalar pix_y = pnt_pix.Y();
            bool hit_wnd =
                pix_x >= 0 && pix_x < (Scalar)img_size_.width &&
                pix_y >= 0 && pix_y < (Scalar)img_size_.height;
            if (!hit_wnd)
                continue;

            if (templ_center_detection_noise_std_ > 0)
            {
                gen_.seed(static_cast<unsigned int>(frame_ind ^ frag_id));  // make noise unique for given salient point in given frame
                const float center_noise_x = templ_center_detection_noise_distr_(gen_);
                const float center_noise_y = templ_center_detection_noise_distr_(gen_);

                constexpr auto Pad = 1e-6f; // keep int(coord) of a pixel inside the picture
                auto in_img_x = std::clamp<Scalar>(pnt_pix.X() + center_noise_x, 0, img_size_.width - Pad);
                auto in_img_y = std::clamp<Scalar>(pnt_pix.Y() + center_noise_y, 0, img_size_.height - Pad);
                pnt_pix = suriko::Point2f{ in_img_x, in_img_y };
            }

            DavisonMonoSlam::SalPntId sal_pnt_id = DemoGetSalPntId(fragment);

            const Scalar depth = Norm(pnt_camera);
            SRK_ASSERT(!IsClose(0, depth)) << "salient points with zero depth are prohibited";

            BlobInfo blob_info;
            blob_info.Coord = pnt_pix;
            blob_info.SalPntIdInTracker = sal_pnt_id;
            blob_info.GTInvDepth = 1 / depth;
            blob_info.FragmentId = frag_id;
            detected_blobs_.push_back(blob_info);
        }
    }

    void MatchSalientPoints(
        const DavisonMonoSlam& mono_slam,
        const std::set<SalPntId>& tracking_sal_pnts,
        size_t frame_ind,
        const Picture& image,
        std::vector<std::pair<DavisonMonoSlam::SalPntId, CornerCorrespond>>* matched_sal_pnts) override
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

            auto sal_pnt_to_coord = std::make_pair(sal_pnt_id, CornerCorrespond{ blob_info.Coord });
            matched_sal_pnts->push_back(sal_pnt_to_coord);
        }
    }

    void RecruitNewCorners(
        const DavisonMonoSlam& mono_slam,
        const std::set<SalPntId>& tracking_sal_pnts,
        size_t frame_ind,
        const Picture& image,
        suriko::Sizei sal_pnt_templ_size,
        std::vector<CornerVicinity>* new_blob_ids) override
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

            CornerVicinity vic;
            vic.virtual_blob_ind = i;
            vic.coord = blob_info.Coord;
            vic.pnt_inv_dist_gt = blob_info.GTInvDepth;
            new_blob_ids->push_back(vic);
        }
    }

    void OnSalientPointIsAssignedToVicinity(CornerVicinity corner_vicinity, TrackedSalientPoint* sal_pnt) override
    {
        BlobInfo& blob_info = detected_blobs_[corner_vicinity.virtual_blob_ind.value()];

        // entire_map's fragment.user_obj point to the salien point id
        size_t frag_id = blob_info.FragmentId;
        SalientPointFragment& frag = entire_map_.GetSalientPointNew(frag_id);

        SalPntId sal_pnt_id{ sal_pnt };
        static_assert(sizeof sal_pnt_id <= sizeof frag.user_obj, "SalPntId must fit into UserObject");
        std::memcpy(&frag.user_obj, &sal_pnt_id, sizeof(sal_pnt_id));
    }

    void SetTemplCenterDetectionNoiseStd(float value)
    {
        templ_center_detection_noise_std_ = value;
        if (value > 0)
            templ_center_detection_noise_distr_ = std::normal_distribution<float>{ 0, value };
    }
    
    const std::vector<BlobInfo>& DetectedBlobs() const { return detected_blobs_; }
};

#if defined(SRK_HAS_OPENCV)

enum class MatchCornerImpl
{
    Templ,    // match patches of images (original Monoslam strategy)
    OrbDescr  // match ORB descriptors
};

class ImageTemplCornersMatcher : public CornersMatcherBase
{
    cv::Ptr<cv::ORB> detector_;
    std::vector<cv::KeyPoint> keypoints_on_got_frame_;
    cv::Mat keypoints_on_got_frame_debug_;  // debug, for visualization of detected ORB features
public:
    bool stop_on_sal_pnt_moved_too_far_ = false;
    std::optional<suriko::Sizei> min_search_rect_size_;

    // this allow to register a new salient point only if the distance between corners is greater than this value;
    // this prevents overlapping of templates of tracked salient points
    std::optional<Scalar> closest_corner_min_dist_pix_;

    MatchCornerImpl match_corners_impl_ = MatchCornerImpl::Templ;

    // impl=match templates
    Scalar min_templ_corr_coeff_ = 0.65f;

    // impl=match ORB descriptors
    int detect_orb_corners_per_frame_ = 500;

    // Small value (<32) will lead to mismatch of slightly changed corners.
    // Corners quickly become unobserved and later deleted.
    // Large values (>96) will lead to corners be matched with wrong corners.
    // Corners become unstable and drift from frame to frame.
    size_t match_orb_descr_max_hamming_dist_ = 64;  //!< [0;256] orb features with signature's distance in [0; value] range are matched

    bool force_sequential_execution_ = false;

    //
    std::function<void(DavisonMonoSlam&, SalPntId, cv::Mat*)> draw_sal_pnt_fun_;
    std::function<void(std::string_view, cv::Mat)> show_image_fun_;
public:
    ImageTemplCornersMatcher(int nfeatures = 50)
    : detect_orb_corners_per_frame_(nfeatures)
    {
        detector_ = cv::ORB::create(nfeatures);
    }


    void AnalyzeFrame(size_t frame_ind, const Picture& image) override
    {
        keypoints_on_got_frame_.clear();

        if (suppress_observations_) return;
    }

    struct TemplateMatchResult
    {
        bool success;
        suriko::Point2f center;
        std::optional<Scalar> corr_coef;  // used only in impl of matching by template

#if defined(SRK_DEBUG)
        suriko::Point2i top_left;
        int executed_match_templ_calls; // specify the number of calls to match-template routine to find this specific match-result
#endif
    };

    TemplateMatchResult FindMatchedTemplateCenterInRect(const DavisonMonoSlam& mono_slam, const TrackedSalientPoint& sal_pnt, const Picture& pic, Recti search_rect) const
    {
        Point2i search_center{ search_rect.x + search_rect.width / 2, search_rect.y + search_rect.height / 2 };
        const int search_radius_left = search_rect.width / 2;
        const int search_radius_up = search_rect.height / 2;

        // -1 to make up for center pixel
        const int search_radius_right = search_rect.width - search_radius_left - 1;
        const int search_radius_down = search_rect.height - search_radius_up - 1;
        const int search_common_rad = std::min({ search_radius_left , search_radius_right , search_radius_up, search_radius_down });

        struct PosAndErr
        {
            Point2i templ_top_left;
            Scalar corr_coef;
            int executed_match_templ_calls = 0;
        };
        
        // choose template-candidate with the maximum correlation coefficient
        // TODO: do we need to handle the case of multiple equal corr coefs? (eg when all pixels of a cadidate are equal)
        Scalar max_corr_coeff = -1 - 0.001f;
        PosAndErr best_match_info;
        int match_templ_call_order = 0;  // specify the order of calls to template match routine

        Scalar templ_mean = sal_pnt.templ_stats.templ_mean_;
        Scalar templ_sqrt_sum_sqr_diff = sal_pnt.templ_stats.templ_sqrt_sum_sqr_diff_;
        const auto& templ_gray = sal_pnt.initial_templ_gray_;

        auto match_templ_at = [this, &templ_gray, &pic, &mono_slam,
            &max_corr_coeff, &best_match_info, templ_mean, templ_sqrt_sum_sqr_diff, &match_templ_call_order](Point2i search_center)
        {
#if defined(SRK_DEBUG)
            match_templ_call_order++;
#endif
            Point2i pic_roi_top_left = mono_slam.TemplateTopLeftInt(suriko::Point2f{ search_center.x, search_center.y });

            auto pic_roi = suriko::Recti{ pic_roi_top_left.x, pic_roi_top_left.y,
                mono_slam.sal_pnt_templ_size_.width,
                mono_slam.sal_pnt_templ_size_.height
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
                match_templ_at(Point2i{ x, border.y });

            // top-left to bottom-left
            for (int y = border.y; y < border.Bottom() - 1; ++y)
                match_templ_at(Point2i{ border.x, y });

            // bottom-left to bottom-right
            for (int x = border.x; x < border.Right() - 1; ++x)
                match_templ_at(Point2i{ x, border.Bottom() - 1 });

            // bottom-right to top-right
            for (int y = border.Bottom() - 1; y > border.y; --y)
                match_templ_at(Point2i{ border.Right() - 1, y });
        }
        
        // iterate through the remainder rectangles at each side of a search rectangle
        {
            // iterate top-left, top-middle and top-right rectangular areas at one pass
            for (int x = search_rect.x; x < search_rect.Right(); ++x)
                for (int y = search_rect.y; y < search_center.y - search_common_rad; ++y)
                    match_templ_at(Point2i{ x, y });

            // iterate bottom-left, bottom-middle and bottom-right rectangular areas at one pass
            for (int x = search_rect.x; x < search_rect.Right(); ++x)
                for (int y = search_center.y + search_common_rad; y < search_rect.Bottom(); ++y)
                    match_templ_at(Point2i{ x, y });

            // iterate left-middle rectangular area
            for (int x = search_rect.x; x < search_center.x - search_common_rad; ++x)
                for (int y = search_center.y - search_common_rad; y < search_center.y + search_common_rad; ++y)
                    match_templ_at(Point2i{ x, y });

            // iterate right-middle rectangular area
            for (int x = search_center.x + search_common_rad; x < search_rect.Right(); ++x)
                for (int y = search_center.y - search_common_rad; y < search_center.y + search_common_rad; ++y)
                    match_templ_at(Point2i{ x, y });
        }

        // correlation coefficient can't be calculated when entire roi is filled with a single color
        TemplateMatchResult result {false};

        if (max_corr_coeff >= -1)
        {
            // preserve fractional coordinates of central pixel
            suriko::Point2i best_match_top_left = best_match_info.templ_top_left;

            const auto& center_offset = sal_pnt.OffsetFromTopLeft();
            suriko::Point2f center{ best_match_top_left.x + center_offset.X(), best_match_top_left.y + center_offset.Y() };

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

    std::tuple<bool,Recti> PredictSalientPointSearchRect(const DavisonMonoSlam& mono_slam, SalPntId sal_pnt_id) const
    {
        auto [op_2D_ellip, corner_ellipse] = mono_slam.GetPredictedSalientPointProjectedUncertEllipse(sal_pnt_id);
        static_assert(std::is_same_v<decltype(corner_ellipse), RotatedEllipse2D>);
        SRK_ASSERT(op_2D_ellip);
        if (!op_2D_ellip) return std::make_tuple(false, Recti{});

        Rect corner_bounds = GetEllipseBounds2(corner_ellipse);
        Recti corner_bounds_i = EncompassRect(corner_bounds);
        return std::make_tuple(true, corner_bounds_i);
    }

    TemplateMatchResult FindCorrespOrbDescrInRect(const TrackedSalientPoint& sal_pnt, const Picture& pic, Recti search_rect) const
    {
        SRK_ASSERT(match_corners_impl_ == MatchCornerImpl::OrbDescr);

        std::vector<cv::KeyPoint> close_keypoints;
        for (const cv::KeyPoint& kp : keypoints_on_got_frame_)
        {
            bool hit_search_rect =
                search_rect.x <= kp.pt.x && kp.pt.x <= search_rect.Right() &&
                search_rect.y <= kp.pt.y && kp.pt.y <= search_rect.Bottom();
            if (hit_search_rect)
                close_keypoints.push_back(kp);
        }

        cv::Mat close_keyp_img;
        static bool debug_keypoints = false;
        if (debug_keypoints)
            cv::drawKeypoints(pic.gray, close_keypoints, close_keyp_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // HOTSPOT: computing ORB descriptor is very slow
        // Also, we can't take the keypoint, closest to the center of a search rect, because the closest corner
        // may actually be a wrong match. So the correspondence must be based on the comparison of descriptors.
        cv::Mat descr_per_row;
        detector_->compute(pic.gray, close_keypoints, descr_per_row);

        struct IndInfo { size_t close_ind; size_t dist; };
        IndInfo closest_neighbour{ (size_t)-1, std::numeric_limits<size_t>::max() };
        for (size_t i = 0; i < close_keypoints.size(); ++i)
        {
            auto pdescr = reinterpret_cast<std::byte*>(descr_per_row.row((int)i).data);
            auto neigh = gsl::make_span<const std::byte>(pdescr, (size_t)descr_per_row.cols);
            size_t descr_dist = BitsHammingDistance(sal_pnt.corner_orb_descr_, neigh);

            static bool print_each = false;
            if (print_each) VLOG(4) << "descr_dist=" << descr_dist;

            if (descr_dist < closest_neighbour.dist)
                closest_neighbour = IndInfo{ i, descr_dist };
        }

        static bool print_best = false;
        if (print_best) VLOG(4) << "best descr_dist=" << closest_neighbour.dist;

        TemplateMatchResult match_result{ false };
        if (closest_neighbour.dist < match_orb_descr_max_hamming_dist_)
        {
            const cv::KeyPoint& match_kp = close_keypoints[closest_neighbour.close_ind];
            match_result.success = true;
            match_result.center = suriko::Point2f{ (Scalar)match_kp.pt.x, (Scalar)match_kp.pt.y };
        }
        return match_result;
    }

    std::optional<suriko::Point2f> FindMatchedCornerInFrame(const DavisonMonoSlam& mono_slam, SalPntId sal_pnt_id, const Picture& pic) const
    {
        auto [op, search_rect_unbounded] = PredictSalientPointSearchRect(mono_slam, sal_pnt_id);
        if (!op)
            return std::nullopt; // broken covariance matrix

        static_assert(std::is_same_v<decltype(search_rect_unbounded), Recti>);

        if (min_search_rect_size_.has_value())
            search_rect_unbounded = ClampRectWhenFixedCenter(search_rect_unbounded, min_search_rect_size_.value());

        Recti image_bounds = { 0, 0, pic.gray.cols, pic.gray.rows };
        
        int radx = mono_slam.sal_pnt_templ_size_.width / 2;
        int rady = mono_slam.sal_pnt_templ_size_.height / 2;
        Recti image_sensitive_portion = DeflateRect(image_bounds, radx, rady, radx, rady);
        
        std::optional<Recti> search_rect_opt = IntersectRects(search_rect_unbounded, image_sensitive_portion);
        if (!search_rect_opt.has_value())
            return std::nullopt; // lost

        const Recti search_rect = search_rect_opt.value();

        static bool debug_template_bounds = false;
        if (debug_template_bounds)
            LOG(INFO) << "templ_bnds=[" << search_rect.x << "," << search_rect.y << "," << search_rect.width << "," << search_rect.height
                << " (" << search_rect.x + search_rect.width/2 <<"," << search_rect.y + search_rect.height/2 << ")";

        const TrackedSalientPoint& sal_pnt = mono_slam.GetSalientPoint(sal_pnt_id);

        TemplateMatchResult match_result;
        if (match_corners_impl_ == MatchCornerImpl::OrbDescr)
            match_result = FindCorrespOrbDescrInRect(sal_pnt, pic, search_rect);
        else
            match_result = FindMatchedTemplateCenterInRect(mono_slam, sal_pnt, pic, search_rect);

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

            // template bounds in new frame
            suriko::Point2i new_top_left = mono_slam.TemplateTopLeftInt(match_result.center);
            cv::Rect templ_rect{
                new_top_left.x, new_top_left.y,
                mono_slam.sal_pnt_templ_size_.width, mono_slam.sal_pnt_templ_size_.height };
            cv::rectangle(image_with_match_bgr, templ_rect, cv::Scalar(172,172,0));

            if (show_image_fun_ != nullptr)
                show_image_fun_("Match.search_rect", image_with_match_bgr);
        }

        // skip a match with low correlation coefficient
        if (min_templ_corr_coeff_ &&
            match_result.corr_coef.has_value() &&
            match_result.corr_coef.value() < min_templ_corr_coeff_)
        {
            static bool debug_matching = false;
            if (debug_matching)
            {
                auto [op, predicted_center] = mono_slam.GetSalientPointProjected2DPosWithUncertainty(FilterStageType::Predicted, sal_pnt_id);
                static_assert(std::is_same_v<decltype(predicted_center), MeanAndCov2D>);
                SRK_ASSERT(op);
                VLOG(5) << "Treating sal_pnt(ind=" << sal_pnt.sal_pnt_ind << ")"
                    << " as undetected because corr_coef=" << match_result.corr_coef.value()
                    << " is less than thr=" << min_templ_corr_coeff_ << ","
                    << " predicted center_pix=[" << predicted_center.mean[0] << "," << predicted_center.mean[1] << "]";
            }
            return std::nullopt;
        }

#if defined(SRK_DEBUG)
        static bool debug_calls = false;
        if (debug_calls)
        {
            int max_core_match_calls = search_rect.width * search_rect.height;
            LOG(INFO) << "match_err_per_pixel=" << match_result.corr_coef.value_or(-1.0f)
                << " match_calls=" << match_result.executed_match_templ_calls << "/" << max_core_match_calls
                << "(" << match_result.executed_match_templ_calls / (float)max_core_match_calls << ")";
        }
#endif

        return match_result.center;
    }

    void MatchSalientPoints(
        const DavisonMonoSlam& mono_slam,
        const std::set<SalPntId>& tracking_sal_pnts,
        size_t frame_ind,
        const Picture& image,
        std::vector<std::pair<DavisonMonoSlam::SalPntId, CornerCorrespond>>* matched_sal_pnts) override
    {
        if (suppress_observations_) return;

        if (match_corners_impl_ == MatchCornerImpl::OrbDescr)
        {
            detector_->detect(image.gray, keypoints_on_got_frame_);  // keypoints are sorted by ascending size [W,H]

            static bool debug_keypoints = false;
            if (debug_keypoints)
                cv::drawKeypoints(image.gray, keypoints_on_got_frame_, keypoints_on_got_frame_debug_, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }

        // HOTSPOT: finding correspondences between corners is slow
        std::mutex match_corners_mutex;
        auto match_fun = [this,&mono_slam,frame_ind,&image,&matched_sal_pnts,&match_corners_mutex](SalPntId sal_pnt_id) -> void
        {
            const TrackedSalientPoint& sal_pnt = mono_slam.GetSalientPoint(sal_pnt_id);

            std::optional<suriko::Point2f> match_pnt_center = FindMatchedCornerInFrame(mono_slam, sal_pnt_id, image);
            bool is_lost = !match_pnt_center.has_value();
            if (is_lost)
                return;

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
                        return;
                }
            }
#endif

            std::lock_guard<std::mutex> lk(match_corners_mutex);
            matched_sal_pnts->push_back(std::make_pair(sal_pnt_id, CornerCorrespond{ new_center }));
        };

#if defined(SRK_PARALLEL_ENGINE)
            if (force_sequential_execution_)
                std::for_each(std::execution::seq, tracking_sal_pnts.begin(), tracking_sal_pnts.end(), match_fun);
            else
                std::for_each(std::execution::par, tracking_sal_pnts.begin(), tracking_sal_pnts.end(), match_fun);
#else
            std::for_each(tracking_sal_pnts.begin(), tracking_sal_pnts.end(), match_fun);
#endif
    }

    void RecruitNewCorners(
        const DavisonMonoSlam& mono_slam,
        const std::set<SalPntId>& tracking_sal_pnts,
        size_t frame_ind,
        const Picture& image,
        suriko::Sizei sal_pnt_templ_size,
        std::vector<CornerVicinity>* new_blob_ids) override
    {
        std::vector<cv::KeyPoint> keypoints;
        if (match_corners_impl_ == MatchCornerImpl::OrbDescr)
        {
            // in ORB impl we detect keypoints on every frame and the keypoints are already found
            keypoints = keypoints_on_got_frame_;
        }
        else
        {
            detector_->detect(image.gray, keypoints);  // keypoints are sorted by ascending size [W,H]
        }

        // reorder the features from high quality to low
        // this will lead to deterministic creation and matching of image features
        // otherwise different features may be selected for the same picture for different program's executions
        std::sort(keypoints.begin(), keypoints.end(), [](auto& a, auto& b) { return a.response > b.response; });

        cv::Mat keyp_img;
        static bool debug_keypoints = false;
        if (debug_keypoints)
            cv::drawKeypoints(image.gray, keypoints, keyp_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        std::vector<cv::KeyPoint> sparse_keypoints;
        Scalar closest_corner_min_dist = ClosestCornerMinDistance(sal_pnt_templ_size);
        FilterOutClosest(keypoints, closest_corner_min_dist, &sparse_keypoints);

        cv::Mat sparse_img;
        if (debug_keypoints)
            cv::drawKeypoints(image.gray, sparse_keypoints, sparse_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // remove keypoints which are close to 'matched' salient points
        auto filter_out_close_to_existing = [&mono_slam, &tracking_sal_pnts](const std::vector<cv::KeyPoint>& keypoints, Scalar exclude_radius,
            std::vector<cv::KeyPoint>* result)
        {
            for (size_t cand_ind = 0; cand_ind < keypoints.size(); ++cand_ind)
            {
                const auto& cand = keypoints[cand_ind];

                bool has_close_blob = false;
                for (SalPntId sal_pnt_id : tracking_sal_pnts)
                {
                    const auto& sal_pnt = mono_slam.GetSalientPoint(sal_pnt_id);
                    bool ok =
                        sal_pnt.track_status == SalPntTrackStatus::New || 
                        sal_pnt.track_status == SalPntTrackStatus::Matched ||
                        sal_pnt.track_status == SalPntTrackStatus::Unobserved;
                    if (!ok) continue;

                    std::optional<suriko::Point2f> exist_pix_opt = sal_pnt.templ_center_pix_;
                    if (!exist_pix_opt.has_value())
                        continue;

                    suriko::Point2f exist_pix = exist_pix_opt.value();
                    Scalar dist = (Scalar)std::sqrt(suriko::Sqr(cand.pt.x - exist_pix[0]) + suriko::Sqr(cand.pt.y - exist_pix[1]));
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

        std::vector<cv::KeyPoint> new_keypoints;
        filter_out_close_to_existing(sparse_keypoints, closest_corner_min_dist, &new_keypoints);

        cv::Mat img_no_closest;
        if (debug_keypoints)
            cv::drawKeypoints(image.gray, new_keypoints, img_no_closest, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::Mat descr_per_row;
        if (match_corners_impl_ == MatchCornerImpl::OrbDescr)
            detector_->compute(image.gray, new_keypoints, descr_per_row);

        for (size_t i=0; i< new_keypoints.size(); ++i)
        {
            cv::KeyPoint kp = new_keypoints[i];

            TemplMatchStats templ_stats{};
            Picture templ_img = GetBlobTemplate(kp, image, sal_pnt_templ_size);
            if (templ_img.gray.empty())
                continue;

            CornerVicinity vic;
            vic.coord = GetBlobCoord(kp);
            vic.templ_img = templ_img;

            if (match_corners_impl_ == MatchCornerImpl::OrbDescr)
            {
                auto src = reinterpret_cast<std::byte*>(descr_per_row.row((int)i).data);
                std::copy_n(src, vic.orb_descr.size(), vic.orb_descr.data());
            }
            else
            {
                // calculate the statistics of this template (mean and variance), used for matching templates
                auto templ_roi = Recti{ 0, 0, sal_pnt_templ_size.width, sal_pnt_templ_size.height };
                Scalar templ_mean = GetGrayImageMean(templ_img.gray, templ_roi);
                Scalar templ_sum_sqr_diff = GetGrayImageSumSqrDiff(templ_img.gray, templ_roi, templ_mean);

                // correlation coefficient is undefined for templates with zero variance (because variance goes into the denominator of corr coef)
                if (IsClose(0, templ_sum_sqr_diff))
                    continue;

                templ_stats.templ_mean_ = templ_mean;
                templ_stats.templ_sqrt_sum_sqr_diff_ = std::sqrt(templ_sum_sqr_diff);
                vic.templ_stats = templ_stats;
            }

            new_blob_ids->push_back(vic);
        }
    }

    static void FilterOutClosest(const std::vector<cv::KeyPoint>& keypoints, Scalar exclude_radius, std::vector<cv::KeyPoint>* sparse_keypoints)
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
                auto dist = (Scalar)std::sqrt(suriko::Sqr(cand.pt.x - stage.pt.x) + suriko::Sqr(cand.pt.y - stage.pt.y));
                if (dist < exclude_radius)
                    processed[i] = (char)true;
            }
        }
    }

    suriko::Point2f GetBlobCoord(const cv::KeyPoint& kp)
    {
        // some pixels already have fractional X or Y coordinate, like 213.2, so return it without changes
        return suriko::Point2f{ kp.pt.x, kp.pt.y };
    }

    /// Gets rectangle around the given blob of requested size.
    /// @param templ_size the size of requested image portion
    Picture GetBlobTemplate(const cv::KeyPoint& kp, const Picture& image, suriko::Sizei templ_size)
    {
        int rad_x_int = templ_size.width / 2;
        int rad_y_int = templ_size.height / 2;

        int center_x = (int)kp.pt.x;
        int center_y = (int)kp.pt.y;
        cv::Rect templ_bounds{ center_x - rad_x_int, center_y - rad_y_int, templ_size.width, templ_size.height };

        cv::Mat templ_gray;
        image.gray(templ_bounds).copyTo(templ_gray);
        SRK_ASSERT(templ_gray.rows == templ_size.height);
        SRK_ASSERT(templ_gray.cols == templ_size.width);

        Picture templ{};
        templ.gray = templ_gray;

#if defined(SRK_DEBUG)
        cv::Mat templ_mat;
        image.bgr_debug(templ_bounds).copyTo(templ_mat);
        templ.bgr_debug = templ_mat;
#endif
        return templ;
    }

    void OnSalientPointIsAssignedToVicinity(CornerVicinity corner_context, TrackedSalientPoint* sal_pnt) override
    {
        // matcher saves its payload in the salient point itself
        if (match_corners_impl_ == MatchCornerImpl::OrbDescr)
        {
            std::copy_n(corner_context.orb_descr.begin(), corner_context.orb_descr.size(), sal_pnt->corner_orb_descr_.data());
        }
        else
        {
            sal_pnt->templ_stats = corner_context.templ_stats;
        }
    }

    // Keeps corners away from each other to prevent overlapping.
    Scalar ClosestCornerMinDistance(suriko::Sizei sal_pnt_templ_size) const
    {
        if (closest_corner_min_dist_pix_.has_value())
            return closest_corner_min_dist_pix_.value();

        // when two salient points touch each other, the distance between them is 2R, R='radius of a template'
        const Scalar touch_dist = std::sqrt(
            suriko::Sqr(static_cast<Scalar>(sal_pnt_templ_size.width)) +
            suriko::Sqr(static_cast<Scalar>(sal_pnt_templ_size.height)));
        return touch_dist;
    }
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

        cv::write(fs, "CurReprojErrMeas", item.cur_reproj_err_meas);
        cv::write(fs, "CurReprojErrPred", item.cur_reproj_err_pred);
        cv::write(fs, "EstimatedSalPnts", static_cast<int>(item.estimated_sal_pnts));
        cv::write(fs, "NewSalPnts", static_cast<int>(item.new_sal_pnts));
        cv::write(fs, "CommonSalPnts", static_cast<int>(item.common_sal_pnts));
        cv::write(fs, "DeletedSalPnts", static_cast<int>(item.deleted_sal_pnts));
        cv::write(fs, "OptimalEstimMulErr", static_cast<float>(item.optimal_estim_mul_err));
        cv::write(fs, "FrameProcessingDur", item.frame_processing_dur.count()); // seconds

        fs << "CamState" <<"[:";
        WriteMatElements(fs, item.cam_state);
        fs << "]";

        if (item.cam_state_gt.has_value())
        {
            fs << "CamStateGT" << "[:";
            WriteMatElements(fs, item.cam_state_gt.value());
            fs << "]";
        }

        if (item.sal_pnts_uncert_median.has_value())
        {
            fs << "SalPntUncMedian_s" << "[:";
            WriteMatElements(fs, item.sal_pnts_uncert_median.value());
            fs << "]";
        }

        // estimation error is available only when ground truth is available
        if (item.estim_err.has_value())
        {
            fs << "EstimErr" << "[:";
            WriteMatElements(fs, item.estim_err.value());
            fs << "]";
        }

        // std of estimated state
        fs << "EstimErrStd" << "[:";
        WriteMatElements(fs, item.estim_err_std);
        fs << "]";

        // residuals
        fs << "MeasResidual" << "[:";
        WriteMatElements(fs, item.meas_residual);
        fs << "]";
        fs << "MeasResidualStd" << "[:";
        WriteMatElements(fs, item.meas_residual_std);
        fs << "]";

        fs << "}";
    }
    fs << "]";
#endif
    return true;
}

std::string FormatImageFileNameNoExt(size_t frame_ind)
{
    constexpr int frame_ind_field_width = 6;  // file names in KITTI dataset have the length = 6
    std::stringstream img_file_name;
    img_file_name.fill('0');
    img_file_name << std::setw(frame_ind_field_width) << frame_ind;
    return img_file_name.str();
}

void EnsureAllSubdirsExist(const std::filesystem::path& dir)
{
    if (!std::filesystem::exists(dir))
    {
        bool create_dir_op = std::filesystem::create_directories(dir);
        SRK_ASSERT(create_dir_op) << "Can't create dir (" << dir << ")";
    }
}

void DumpOpenCVImage(const std::filesystem::path& root_dump_dir, size_t frame_ind, std::string_view cam_name, const cv::Mat& camera_image_bgr)
{
    std::string img_file_name  = FormatImageFileNameNoExt(frame_ind);

    std::filesystem::path cam0_dump_dir = root_dump_dir / cam_name;
    EnsureAllSubdirsExist(cam0_dump_dir);

    std::filesystem::path img_path = cam0_dump_dir / (img_file_name + ".png");
    bool op = cv::imwrite(img_path.string(), camera_image_bgr);
    SRK_ASSERT(op) << "Can't write to file (" << img_path << ")";
}

std::optional<Scalar> GetMaxCamShift(const std::vector<std::optional<SE3Transform>>& gt_cam_orient_cfw)
{
    std::optional<SE3Transform> prev_cam_wfc;
    std::optional<Scalar> between_frames_max_cam_shift;
    for (const auto& cfw_opt : gt_cam_orient_cfw)
    {
        std::optional<SE3Transform> wfc = std::nullopt;
        if (cfw_opt.has_value())
        {
            wfc = SE3Inv(cfw_opt.value());
            if (prev_cam_wfc.has_value())
            {
                Point3 cam_shift = wfc.value().T - prev_cam_wfc.value().T;
                auto dist = Norm(cam_shift);
                if (!between_frames_max_cam_shift.has_value() ||
                    dist > between_frames_max_cam_shift.value())
                    between_frames_max_cam_shift = dist;
            }
        }
        prev_cam_wfc = wfc;
    }
    return between_frames_max_cam_shift;
}

void CheckDavisonMonoSlamConfigurationAndDump(const DavisonMonoSlam& mono_slam, 
    const std::vector<std::optional<SE3Transform>>& gt_cam_orient_cfw)
{
    // check max shift of the camera is expected by tracker
    std::optional<Scalar> between_frames_max_cam_shift = GetMaxCamShift(gt_cam_orient_cfw);
    if (between_frames_max_cam_shift.has_value())
    {
        auto max_expected_cam_shift = mono_slam.process_noise_linear_velocity_std_ * 3;
        if (between_frames_max_cam_shift.value() > max_expected_cam_shift)
            LOG(INFO)
                << "Note: max_cam_shift=" << between_frames_max_cam_shift.value()
                << " is too big compared to input (process) noise 3sig=" << max_expected_cam_shift;
    }
}

bool LoadTumGtTraj(const std::filesystem::path& tum_dataset_dirpath,
    std::vector<std::optional<SE3Transform>>* gt_cam_orient_cfw,
    std::vector<TumTimestamp>* gt_rgb_timestamps)
{
    if (tum_dataset_dirpath.empty() || !is_directory(tum_dataset_dirpath))
        return false;

    // TUM dataset has ground truth trajectory in oversampled frequency (4ms), while images are collected every 33ms.

    // images
    std::filesystem::path rgb_filepath = tum_dataset_dirpath / "rgb.txt"sv;
    std::vector<TumTimestampFilename> filename_stamps;
    filename_stamps.reserve(1024);
    std::string err_msg;
    bool op = ReadTumDatasetTimedRgb(rgb_filepath, &filename_stamps, &err_msg);
    if (!op)
    {
        LOG(ERROR) << "Can't read TUM 'rgb' file: " << rgb_filepath << ", error: " << err_msg;
        return false;
    }
    auto& rgb_stamps = *gt_rgb_timestamps;
    rgb_stamps.resize(filename_stamps.size());
    std::transform(filename_stamps.begin(), filename_stamps.end(), rgb_stamps.begin(), [](const TumTimestampFilename& i) { return i.timestamp; });

    // ground truth
    std::filesystem::path gt_filepath = tum_dataset_dirpath / "groundtruth.txt"sv;
    std::vector<TumTimestampPose> oversampled_poses_gt;
    oversampled_poses_gt.reserve(filename_stamps.size());  // there may be more samples due to oversampling
    op = ReadTumDatasetGroundTruth(gt_filepath, &oversampled_poses_gt, &err_msg);
    if (!op)
    {
        LOG(ERROR) << "Can't read TUM groundtruth file: " << gt_filepath << ", error: " << err_msg;
        return false;
    }

    // maximal allowed difference between ground truth and rgb image timestamp
    auto max_time_diff = MaxMatchTimeDifference(oversampled_poses_gt);
    if (!max_time_diff.has_value()) return false;

    std::vector<ptrdiff_t> rgb_poses_gt_inds;
    size_t found_gt_count = AssignCloseIndsByTimestamp(oversampled_poses_gt, rgb_stamps, max_time_diff.value(), &rgb_poses_gt_inds);
    LOG(INFO) << found_gt_count << " out of " << rgb_stamps.size() << " camera frames has ground truth";

    // construct subset of ground truth trajectory, corresponding to rgb images
    std::vector<std::optional<SE3Transform>> rgb_poses_gt{ rgb_stamps.size() };
    for (size_t i = 0; i < rgb_poses_gt.size(); ++i)
    {
        ptrdiff_t gt_ind = rgb_poses_gt_inds[i];
        if (gt_ind != -1)
        {
            const TumTimestampPose& p = oversampled_poses_gt[gt_ind];
            rgb_poses_gt[i] = TimestampPoseToSE3(p);
        }
    }
    Scalar rgb_poses_gt_traj_len = CalcTrajectoryLength(&rgb_poses_gt, nullptr);
    LOG(INFO) << "gt trajectory length: " << rgb_poses_gt_traj_len;

    // ground truth corresponding to the first camera frame
    ptrdiff_t first_cam_gt_ind = rgb_poses_gt_inds[0];
    const TumTimestampPose& first_cam_gt = oversampled_poses_gt[first_cam_gt_ind];
    std::optional<SE3Transform> first_cam_gt_se3 = TimestampPoseToSE3(first_cam_gt);
    if (!first_cam_gt_se3.has_value()) return false;

    // transform the entire ground truth sequence, so that a ground truth frame, corresponding to the first camera's frame, coincides with the origin

    SE3Transform first_cam_gt_inv = SE3Inv(first_cam_gt_se3.value());
    //for (const TumTimestampPose& t : oversampled_poses_gt)
    //{
    //    std::optional<SE3Transform> cfw = TimestampPoseToSE3(t);
    //    if (cfw.has_value())
    //        cfw = SE3Compose(first_cam_gt_inv, cfw.value());
    //    gt_cam_orient_cfw->push_back(cfw);
    //}
    for (const std::optional<SE3Transform>& t : rgb_poses_gt)
    {
        std::optional<SE3Transform> cfw = t;
        if (cfw.has_value())
            cfw = SE3Compose(first_cam_gt_inv, cfw.value());
        gt_cam_orient_cfw->push_back(cfw);
    }

    return true;
}

bool ValidateDirectoryEmptyOrExists(const std::string &value)
{
    const std::filesystem::path test_data_path = std::filesystem::absolute(value);
    
    // allow empty directory in case of virtual scenario
    return std::filesystem::is_directory(test_data_path);
}

/// The workspace data for the cv::undistort routine.
struct UndistMaps
{
#if defined(SRK_HAS_OPENCV)
    cv::Matx<double, 3, 3> cam_mat;
    cv::Mat dist_coeffs;

    cv::Mat map1;  // the maps as the output of cv::initUndistortRectifyMap
    cv::Mat map2;
#endif
};

static constexpr auto kVirtualSceneCStr = "virtscene";
static constexpr auto kImageSeqDirCStr = "imageseqdir";

DEFINE_string(demo_params, "", "path to json file to read parameters for demo");
DEFINE_bool(monoslam_cam_perfect_init_vel, false, "");
DEFINE_bool(monoslam_cam_perfect_init_ang_vel, false, "");
DEFINE_double(monoslam_cam_pos_x_std_m, 0, "");
DEFINE_double(monoslam_cam_pos_y_std_m, 0, "");
DEFINE_double(monoslam_cam_pos_z_std_m, 0, "");
DEFINE_double(monoslam_cam_orient_q_comp_std, 0, "");
DEFINE_double(monoslam_cam_vel_std, 0, "");
DEFINE_double(monoslam_cam_ang_vel_std, 0, "");

DEFINE_double(monoslam_sal_pnt_pos_x_std_if_gt, 0, "");
DEFINE_double(monoslam_sal_pnt_pos_y_std_if_gt, 0, "");
DEFINE_double(monoslam_sal_pnt_pos_z_std_if_gt, 0, "");
DEFINE_double(monoslam_sal_pnt_first_cam_pos_std_if_gt, 0, "");
DEFINE_double(monoslam_sal_pnt_azimuth_std_if_gt, 0, "");
DEFINE_double(monoslam_sal_pnt_elevation_std_if_gt, 0, "");
DEFINE_double(monoslam_sal_pnt_inv_dist_std_if_gt, 0, "");

DEFINE_bool(monoslam_force_xyz_sal_pnt_pos_diagonal_uncert, false, "false to derive XYZ sal pnt uncertainty from spherical sal pnt; true to set diagonal covariance values");
DEFINE_double(monoslam_sal_pnt_negative_inv_rho_substitute, -1, "");
DEFINE_bool(monoslam_force_sequential_engine, false, "True to perform sequential matching");
DEFINE_int32(monoslam_update_impl, 0, "");
DEFINE_int32(monoslam_max_new_blobs_in_first_frame, 7, "");
DEFINE_int32(monoslam_max_new_blobs_per_frame, 1, "");
DEFINE_double(monoslam_match_blob_prob, 1, "[0,1] portion of blobs which are matched with ones in the previous frame; 1=all matched, 0=none matched");
DEFINE_int32(monoslam_match_corners_impl, 1, "1 to match image patches, 2 to match ORB features");
DEFINE_int32(monoslam_match_detect_corners_per_frame, 500, "# of ORB features to create per frame");
DEFINE_int32(monoslam_orb_descr_max_hamming_distance, 64, "ORB descriptors with Hamming distance in [0;value] are matched");
DEFINE_int32(monoslam_templ_width, 15, "width of template");
DEFINE_int32(monoslam_templ_min_search_rect_width, 7, "the min width of a rectangle when searching for tempplate in the next frame");
DEFINE_int32(monoslam_templ_min_search_rect_height, 7, "");
DEFINE_double(monoslam_templ_center_detection_noise_std_pix, 0, "std of measurement noise(=sqrt(R), 0=no noise");
DEFINE_double(monoslam_match_closest_corner_min_dist_pix, 0, "");
DEFINE_bool(monoslam_match_stop_on_sal_pnt_moved_too_far, false, "width of template");
DEFINE_bool(monoslam_fix_estim_vars_covar_symmetry, true, "");
DEFINE_bool(monoslam_debug_estim_vars_cov, false, "");
DEFINE_bool(monoslam_debug_predicted_vars_cov, false, "");
DEFINE_int32(monoslam_debug_max_sal_pnt_count, -1, "[default=-1(none)] number of salient points won't be greater than this value");
DEFINE_bool(monoslam_sal_pnt_perfect_init_inv_dist, false, "");
DEFINE_int32(monoslam_set_estim_state_covar_to_gt_impl, 2, "1=ignore correlations, 2=set correlations as if 'AddNewSalientPoint' is called on each salient point");
DEFINE_double(monoslam_covar2D_to_ellipse_confidence, 0.95f, "");

DEFINE_bool(ui_swallow_exc, true, "true to ignore (swallow) exceptions in UI");
DEFINE_int32(ui_loop_prolong_period_ms, 3000, "");
DEFINE_int32(ui_tight_loop_relaxing_delay_ms, 100, "");
DEFINE_int32(ui_dots_per_uncert_ellipse, 12, "Number of dots to split uncertainty ellipse (4=rectangle)");
DEFINE_double(ui_covar3D_to_ellipsoid_chi_square, 7.814f, "{confidence interval,chi-square}={68%,3.505},{95%,7.814},{99%,11.344}");
DEFINE_string(FLAGS_adorn_traj_filepath, "", "Loads some trajectory from a file in TUM dataset format and renders it");

constexpr bool FLAGS_ctrl_ui_in_separate_thread = false;  // TODO: to be removed, locking of predicted/updated state of a tracker is not implemented
DEFINE_bool(ctrl_wait_after_each_frame, false, "true to wait for keypress after each iteration");
DEFINE_bool(ctrl_debug_skim_over, false, "overview the synthetic world without reconstruction");
DEFINE_bool(ctrl_visualize_during_processing, true, "");
DEFINE_bool(ctrl_visualize_after_processing, true, "");
DEFINE_bool(ctrl_collect_tracker_internals, false, "");
DEFINE_bool(ctrl_log_slam_images_cam0, false, "Whether to write images of camera to filesystem");
DEFINE_bool(ctrl_log_slam_images_scene3D, false, "Whether to write images of 3D scene to filesystem");
DEFINE_string(ctrl_log_slam_images_dir, "", "The directory where to output the images");
DEFINE_bool(ctrl_log_slam_cam_traj, false, "Whether to output estimated camera's trajectory in TUM dataset format");

bool ApplyParamsFromConfigFile(DavisonMonoSlam* mono_slam, ConfigReader* config_reader)
{
    auto& cr = *config_reader;
    auto& ms = *mono_slam;

    auto opt_set = [](std::optional<Scalar> opt_value, gsl::not_null<Scalar*> dst)
    {
        if (opt_value.has_value())
            * dst = opt_value.value();
    };

    auto process_noise_linear_velocity_std = FloatParam<Scalar>(&cr, "monoslam_process_noise_cam_lin_veloc_std_mm").value_or(-1);
    if (process_noise_linear_velocity_std == -1)
    {
        LOG(ERROR) << "no mandatory parameter monoslam_process_noise_cam_lin_veloc_std_mm";
        return false;
    }
    auto process_noise_angular_velocity_std = FloatParam<Scalar>(&cr, "monoslam_process_noise_cam_ang_veloc_std_rad").value_or(-1);
    if (process_noise_angular_velocity_std == -1)
    {
        LOG(ERROR) << "no mandatory parameter monoslam_process_noise_cam_ang_veloc_std_rad";
        return false;
    }
    ms.SetProcessNoiseStd(process_noise_linear_velocity_std, process_noise_angular_velocity_std);

    opt_set(FloatParam<Scalar>(&cr, "monoslam_measurm_noise_std_pix"), &ms.measurm_noise_std_pix_);
    mono_slam->one_point_ransac_corner_max_divergence_pix_ = FloatParam<Scalar>(&cr, "monoslam_1pransac_corner_max_divergence_pix");
    mono_slam->one_point_ransac_high_innov_chi_square_thresh_pix2_ = FloatParam<Scalar>(&cr, "monoslam_1pransac_high_innov_chisq_thr_pix2");
    opt_set(FloatParam<Scalar>(&cr, "monoslam_sal_pnt_init_inv_dist"), &ms.sal_pnt_init_inv_dist_);
    opt_set(FloatParam<Scalar>(&cr, "monoslam_sal_pnt_init_inv_dist_std"), &ms.sal_pnt_init_inv_dist_std_);

    opt_set(FloatParam<Scalar>(&cr, "monoslam_seconds_per_frame"), &ms.seconds_per_frame_);
    
    auto sal_pnt_max_undetected_frames_count = cr.GetValue<int>("monoslam_sal_pnt_max_undetected_frames_count");
    if (sal_pnt_max_undetected_frames_count.has_value())
        ms.sal_pnt_max_undetected_frames_count_ = static_cast<size_t>(sal_pnt_max_undetected_frames_count.value());

    return true;
}

int DavisonMonoSlamDemo(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);

#if defined(SRK_HAS_OPENCV)
    cv::theRNG().state = 123; // specify seed for OpenCV randomness, so that debugging always goes the same execution path
#endif

    auto log_absent_mandatory_flag = [](std::string_view param_name)
    {
        LOG(ERROR) << "mandatory flag '" << param_name << "' is not provided";
    };

    std::filesystem::path demo_params = FLAGS_demo_params;
    if (demo_params.empty())
    {
        log_absent_mandatory_flag("demo_params");
        return 1;
    }

    ConfigReader config_reader{ demo_params };
    auto demo_params_dev = demo_params.replace_filename(demo_params.filename().stem().string() + "-DEV" + demo_params.extension().string());
    if (std::filesystem::exists(demo_params_dev))
        config_reader.ReadConfig(demo_params_dev);  // allow dev to override some params
    if (config_reader.HasErrors())
    {
        LOG(ERROR) << config_reader.Error();
        return 1;
    }

    DemoDataSource demo_data_source = DemoDataSource::kVirtualScene;
    std::string scene_imageseq_dir;

    std::vector<std::optional<SE3Transform>> gt_cam_orient_cfw;  // entire sequence of ground truth camera orientation, transforming into camera from world
    std::vector<TumTimestamp> estim_rgb_timestamps;              // timestamps of input images
    std::vector<SE3Transform> estim_cam_orient_cfw;              // estimated trajectory of the camera

    std::vector<std::optional<SE3Transform>> external_cam_orient_cfw;  // some trajectory to render

    // load ground truth from TUM dataset
    std::filesystem::path tum_dataset_dirpath;
    if (auto tum_dataset_dirpath_opt = config_reader.GetValue<std::string>("tum_dataset_dir"); tum_dataset_dirpath_opt.has_value())
    {
        tum_dataset_dirpath = std::filesystem::path{ tum_dataset_dirpath_opt.value() };
        LOG(INFO) << "tum_dataset_dir=" << tum_dataset_dirpath;

        demo_data_source = DemoDataSource::kImageSeqDir;
        scene_imageseq_dir = (tum_dataset_dirpath / "rgb/").string();
    }
    else
    {
        std::string scene_source = config_reader.GetValue<std::string>("scene_source").value_or("");
        if (scene_source.empty())
        {
            log_absent_mandatory_flag("scene_source");
            return 1;
        }

        demo_data_source = scene_source == std::string(kImageSeqDirCStr) ? DemoDataSource::kImageSeqDir : DemoDataSource::kVirtualScene;
        scene_imageseq_dir = config_reader.GetValue<std::string>("scene_imageseq_dir").value_or("");
    }

    auto check_sal_pnt_representation = [&]() -> bool
    {
        // 6x1 salient point's representation is generic and works everywhere
        if (DavisonMonoSlam::kSalPntRepres == SalPntComps::kSphericalFirstCamInvDist)
            return true;

        // 3x1 Euclidean representation of a salient point is allowed only in virtual scenarios with turned on fake initialization of inverse depth.
        if (DavisonMonoSlam::kSalPntRepres == SalPntComps::kXyz)
        {
            if (demo_data_source == DemoDataSource::kVirtualScene && FLAGS_monoslam_sal_pnt_perfect_init_inv_dist)
                return true;
        }
        return false;
    };
    
    if (!check_sal_pnt_representation())
    {
        LOG(ERROR)
            << "XYZ [3x1] representation of a salient point is allowed only in virtual scenarios, "
            << "because in real-world scenario the depth of a salient point is unknown and can't be initialized."
            << "Use spherical [6x1] representation of a salient point (use c++ flag: SAL_PNT_REPRES=2).";

        // TODO: how to merge two messages
        if (DavisonMonoSlam::kSalPntRepres == SalPntComps::kXyz &&
            demo_data_source == DemoDataSource::kVirtualScene)
        {
            LOG(ERROR) << "Set run-time flag monoslam_fake_sal_pnt_init_inv_dist=true to initialize a salient point's initial distance to ground truth.";
        }
        return 1;
    }

    if (!scene_imageseq_dir.empty())
    {
        // validate directory only if it is the source of images for demo
        if (!ValidateDirectoryEmptyOrExists(scene_imageseq_dir)) {
            LOG(ERROR) << "directory [" << scene_imageseq_dir << "] doesn't exist";
            return 2;
        }
    }

    //
    FragmentMap entire_map;
    if (demo_data_source == DemoDataSource::kVirtualScene)
    {
        auto world_noise_x3D_std = FloatParam<Scalar>(&config_reader, "world_noise_x3D_std").value_or(0.0f);
        auto world_noise_R_std = FloatParam<Scalar>(&config_reader, "world_noise_R_std").value_or(0.0f);

        LOG(INFO) << "world_noise_x3D_std=" << world_noise_x3D_std;
        LOG(INFO) << "world_noise_R_std=" << world_noise_R_std;

        //
        bool corrupt_salient_points_with_noise = world_noise_x3D_std > 0;
        bool corrupt_cam_orient_with_noise = world_noise_R_std > 0;

        auto world_x_limits = FloatSeq<Scalar>(&config_reader, "world_x_limits").value_or(std::vector<Scalar>{-1.5f, 1.5f});
        auto world_y_limits = FloatSeq<Scalar>(&config_reader, "world_y_limits").value_or(std::vector<Scalar>{-1.5f, 1.5f});
        auto world_z_limits = FloatSeq<Scalar>(&config_reader, "world_z_limits").value_or(std::vector<Scalar>{-1.5f, 1.5f});
        if (world_x_limits.size() != 2 || world_y_limits.size() != 2 || world_z_limits.size() != 2)
        {
            LOG(ERROR) << "require type(world_xyz_limits): array<double,2>";
            return 1;
        }

        auto world_z_ascent = FloatParam<Scalar>(&config_reader, "world_z_ascent").value_or(0.2f);

        auto world_cell_size = FloatSeq<Scalar>(&config_reader, "world_cell_size").value_or(std::vector<Scalar>{0.5f, 0.5f, 0.5f});
        if (world_cell_size.size() != 3)
        {
            LOG(ERROR) << "require type(world_cell_size): array<double,3>";
            return 1;
        }

        WorldBounds wb{};
        wb.x_min = world_x_limits[0];
        wb.x_max = world_x_limits[1];
        wb.y_min = world_y_limits[0];
        wb.y_max = world_y_limits[1];
        wb.z_min = world_z_limits[0];
        wb.z_max = world_z_limits[1];
        std::array<Scalar, 3> cell_size = {
            world_cell_size[0],
            world_cell_size[1],
            world_cell_size[2]
        };

        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        gen.seed(1234);

        std::unique_ptr<std::normal_distribution<Scalar>> x3D_noise_dis;
        if (corrupt_salient_points_with_noise)
            x3D_noise_dis = std::make_unique<std::normal_distribution<Scalar>>(0.0f, world_noise_x3D_std);

        //
        entire_map.SetFragmentIdOffsetInternal(1000'000);
        GenerateWorldPoints(wb, cell_size, world_z_ascent, corrupt_salient_points_with_noise, &gen, x3D_noise_dis.get(), &entire_map);
        LOG(INFO) << "points_count=" << entire_map.SalientPointsCount();

        auto viewer_eye_offset_a = FloatSeq<Scalar>(&config_reader, "viewer_eye_offset").value_or(std::vector<Scalar>{4, -2.5f, 7});
        auto viewer_center_offset_a = FloatSeq<Scalar>(&config_reader, "viewer_center_offset").value_or(std::vector<Scalar>{0, 0, 0});
        auto viewer_up_a = FloatSeq<Scalar>(&config_reader, "viewer_up").value_or(std::vector<Scalar>{0, 0, 1});
        if (viewer_eye_offset_a.size() != 3 || viewer_center_offset_a.size() != 3 || viewer_up_a.size() != 3)
        {
            LOG(ERROR) << "require type(viewer_eye_offset): array<double,3>";
            LOG(ERROR) << "require type(viewer_center_offset): array<double,3>";
            LOG(ERROR) << "require type(viewer_up): array<double,3>";
            return 1;
        }

        suriko::Point3 viewer_eye_offset{ viewer_eye_offset_a[0], viewer_eye_offset_a[1], viewer_eye_offset_a[2] };
        suriko::Point3 viewer_center_offset{ viewer_center_offset_a[0], viewer_center_offset_a[1], viewer_center_offset_a[2] };
        Point3 viewer_up{ viewer_up_a[0], viewer_up_a[1], viewer_up_a[2] };

        std::string virtual_scenario = config_reader.GetValue<std::string>("virtual_scenario").value_or("");
        if (virtual_scenario.empty())
        {
            log_absent_mandatory_flag("virtual_scenario");
            return 1;
        }

        if (virtual_scenario == "RectangularPath")
        {
            int viewer_steps_per_side_x = config_reader.GetValue<int>("viewer_steps_per_side_x").value_or(20);
            int viewer_steps_per_side_y = config_reader.GetValue<int>("viewer_steps_per_side_y").value_or(10);
            GenerateCameraShotsAlongRectangularPath(wb, viewer_steps_per_side_x, viewer_steps_per_side_y,
                viewer_eye_offset, viewer_center_offset, viewer_up, &gt_cam_orient_cfw);
        }
        else if (virtual_scenario == "RightAndLeft")
        {
            auto max_deviation = FloatParam<Scalar>(&config_reader, "viewer_max_deviation").value_or(1.5f);
            auto num_steps = config_reader.GetValue<int>("viewer_max_deviation").value_or(100);
            GenerateCameraShotsRightAndLeft(wb, viewer_eye_offset, viewer_center_offset, viewer_up,
                max_deviation,
                num_steps,
                &gt_cam_orient_cfw);
        }
        else if (virtual_scenario == "OscilateRightAndLeft")
        {
            auto max_deviation = FloatParam<Scalar>(&config_reader, "viewer_max_deviation").value_or(0.6f);
            int shots_per_period = config_reader.GetValue<int>("viewer_shots_per_period").value_or(160);
            int periods_count = config_reader.GetValue<int>("viewer_periods_count").value_or(100);
            bool const_view_dir = config_reader.GetValue<bool>("viewer_const_view_dir").value_or(false);
            auto viewer_eye = viewer_eye_offset;
            auto center = viewer_center_offset;
            GenerateCameraShotsOscilateRightAndLeft(wb, viewer_eye, center, viewer_up,
                max_deviation,
                periods_count,
                shots_per_period,
                const_view_dir,
                &gt_cam_orient_cfw);
        }
        else if (virtual_scenario == "Custom3DPath")
        {
            int periods_count = config_reader.GetValue<int>("viewer_periods_count").value_or(100);
            auto float_seq = FloatSeq<Scalar>(&config_reader, "viewer_eye_center_up").value_or(std::vector<Scalar>{});
            if (float_seq.size() % 9 != 0)
            {
                LOG(INFO) << "Expect sequence of N camera orientations, formatted 9*N=[eye center up...] where eye, center and up are 3D position [X Y Z]";
                return 1;
            }
            size_t cams_count = float_seq.size() / 9;
            std::vector<LookAtComponents> cam_poses;
            for (size_t i = 0; i < cams_count; ++i)
            {
                size_t off = i * 9;
                suriko::Point3 eye{ float_seq[off + 0], float_seq[off + 1], float_seq[off + 2] };
                suriko::Point3 cnt{ float_seq[off + 3], float_seq[off + 4], float_seq[off + 5] };
                suriko::Point3 upp{ float_seq[off + 6], float_seq[off + 7], float_seq[off + 8] };
                cam_poses.push_back(LookAtComponents{ eye, cnt, upp });
            }
            GenerateCameraShots3DPath(wb, cam_poses, periods_count, &gt_cam_orient_cfw);
        }
        else if (virtual_scenario == "RotateLeftAndRight")
        {
            auto viewer_min_ang = FloatParam<Scalar>(&config_reader, "viewer_min_ang").value_or(0.95f);
            auto viewer_max_ang = FloatParam<Scalar>(&config_reader, "viewer_max_ang").value_or(1.39f);
            auto shots_per_period = config_reader.GetValue<int>("shots_per_period").value_or(32);
            auto periods_count = config_reader.GetValue<int>("periods_count").value_or(100);
            auto viewer_eye = viewer_eye_offset;
            GenerateCameraShotsRotateLeftAndRight(wb, viewer_eye, viewer_up,
                viewer_min_ang,
                viewer_max_ang,
                periods_count,
                shots_per_period,
                &gt_cam_orient_cfw);
        }
        else
        {
            LOG(ERROR) << "Unsupported virtual scenario: '" << virtual_scenario 
                << "'. Use one of [RectangularPath,RightAndLeft,OscilateRightAndLeft,Custom3DPath,RotateLeftAndRight]";
            return 1;
        }

        if (corrupt_cam_orient_with_noise)
        {
            std::normal_distribution<Scalar> cam_orient_noise_dis(0, world_noise_R_std);
            for (std::optional<SE3Transform>& cam_orient : gt_cam_orient_cfw)
            {
                if (!cam_orient.has_value()) continue;
                Point3 dir;
                if (AxisAngleFromRotMat(cam_orient.value().R, &dir))
                {
                    Scalar d1 = cam_orient_noise_dis(gen);
                    Scalar d2 = cam_orient_noise_dis(gen);
                    Scalar d3 = cam_orient_noise_dis(gen);
                    dir[0] += d1;
                    dir[1] += d2;
                    dir[2] += d3;

                    Eigen::Matrix<Scalar, 3, 3> newR;
                    if (RotMatFromAxisAngle(dir, &newR))
                        cam_orient.value().R = newR;
                }
            }
        }
    }
    else // sequence of images mode
    {
        if (!tum_dataset_dirpath.empty())
        {
            LOG(INFO) << "Loading TUM dataset";

            // ground truth
            bool oploadgt = LoadTumGtTraj(tum_dataset_dirpath, &gt_cam_orient_cfw, &estim_rgb_timestamps);
            if (!oploadgt)
                LOG(INFO) << "Can't load ground truth from TUM dataset";
        }

        if (auto external_traj_filepath = config_reader.GetValue<std::string>("external_traj_filepath").value_or(""); !external_traj_filepath.empty())
        {
            std::string err_msg;
            std::vector<TumTimestampPose> poses_cfw;
            bool op = ReadTumDatasetGroundTruth(external_traj_filepath, &poses_cfw, &err_msg);
            if (!op)
            {
                LOG(ERROR) << "Can't read external trajectory from path: " << external_traj_filepath;
                return 1;
            }
            LOG(INFO) << "Loaded external trajectory, count:" << poses_cfw.size() <<", filepath:" << external_traj_filepath;

            std::transform(poses_cfw.begin(), poses_cfw.end(), std::back_inserter(external_cam_orient_cfw), 
                [](const TumTimestampPose& t) { return TimestampPoseToSE3(t); });
        }
    }

    std::optional<Scalar> between_frames_max_cam_shift = GetMaxCamShift(gt_cam_orient_cfw);
    if (between_frames_max_cam_shift.has_value())
        LOG(INFO) << "max_cam_shift=" << between_frames_max_cam_shift.value();

    size_t frames_count = gt_cam_orient_cfw.size();
    LOG(INFO) << "frames_count=" << frames_count;

    auto camera_image_size = config_reader.GetSeq<int>("camera_image_size").value_or(std::vector<int>{0, 0});
    auto camera_princip_point = FloatSeq<Scalar>(&config_reader, "camera_princip_point").value_or(std::vector<Scalar>{0, 0});
    auto camera_pixel_size_mm = FloatSeq<Scalar>(&config_reader, "camera_pixel_size_mm").value_or(std::vector<Scalar>{0, 0});
    auto camera_focal_length_mm = FloatParam<Scalar>(&config_reader, "camera_focal_length_mm").value_or(0);

    if (camera_focal_length_mm == 0 || camera_pixel_size_mm[0] == 0)
    {
        LOG(ERROR) << "camera_focal_length_mm: f is mandatory";
        LOG(ERROR) << "camera_pixel_size_mm: [dx,dy] is mandatory";
        return 1;
    }

    CameraIntrinsicParams cam_intrinsics;
    cam_intrinsics.image_size = { camera_image_size[0], camera_image_size[1] };
    cam_intrinsics.principal_point_pix = { camera_princip_point[0], camera_princip_point[1] };
    cam_intrinsics.focal_length_mm = camera_focal_length_mm;
    cam_intrinsics.pixel_size_mm = { camera_pixel_size_mm[0] , camera_pixel_size_mm[1] };

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

    MikhailRadialDistortionParams cam_distort_params{};
    if (auto camera_distort_mikhail_k1k2 = FloatSeq<Scalar>(&config_reader, "camera_distort_mikhail_k1k2"); camera_distort_mikhail_k1k2.has_value())
    {
        cam_distort_params.k1 = camera_distort_mikhail_k1k2.value()[0];
        cam_distort_params.k2 = camera_distort_mikhail_k1k2.value()[1];
    }

    //
    DavisonMonoSlam2DDrawer drawer;
    drawer.dots_per_uncert_ellipse_ = FLAGS_ui_dots_per_uncert_ellipse;
    drawer.covar3D_to_ellipsoid_chi_square_ = static_cast<Scalar>(FLAGS_ui_covar3D_to_ellipsoid_chi_square);
    drawer.ui_swallow_exc_ = FLAGS_ui_swallow_exc;

    // the origin of a tracker (sometimes cam0)
    SE3Transform tracker_origin_from_world;
    if (demo_data_source == DemoDataSource::kVirtualScene)
    {
        // tracker coordinate system = cam0
        tracker_origin_from_world = gt_cam_orient_cfw[0].value();
    }
    else if (demo_data_source == DemoDataSource::kImageSeqDir)
    {
        // tracker coordinates system = world coordinate system
        tracker_origin_from_world.R.setIdentity();
        Fill(0, &tracker_origin_from_world.T);
    }

    DavisonMonoSlam::DebugPathEnum debug_path = DavisonMonoSlam::DebugPathEnum::DebugNone;
    if (FLAGS_monoslam_debug_estim_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugEstimVarsCov;
    if (FLAGS_monoslam_debug_predicted_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugPredictedVarsCov;
    DavisonMonoSlam::SetDebugPath(debug_path);

    DavisonMonoSlam mono_slam{ };
    if (!ApplyParamsFromConfigFile(&mono_slam, &config_reader))
        return 1;
    mono_slam.ui_in_separate_thread_ = FLAGS_ctrl_ui_in_separate_thread;
    mono_slam.cam_intrinsics_ = cam_intrinsics;
    mono_slam.cam_distort_params_ = cam_distort_params;
    mono_slam.cam_enable_distortion_ = config_reader.GetValue<bool>("camera_enable_distortion").value_or(true);
    mono_slam.force_xyz_sal_pnt_pos_diagonal_uncert_ = FLAGS_monoslam_force_xyz_sal_pnt_pos_diagonal_uncert;
    mono_slam.sal_pnt_templ_size_ = { FLAGS_monoslam_templ_width, FLAGS_monoslam_templ_width };
    if (FLAGS_monoslam_sal_pnt_negative_inv_rho_substitute >= 0)
        mono_slam.sal_pnt_negative_inv_rho_substitute_ = static_cast<Scalar>(FLAGS_monoslam_sal_pnt_negative_inv_rho_substitute);
    mono_slam.covar2D_to_ellipse_confidence_ = static_cast<Scalar>(FLAGS_monoslam_covar2D_to_ellipse_confidence);

    if (FLAGS_monoslam_update_impl != 0)
        mono_slam.mono_slam_update_impl_ = FLAGS_monoslam_update_impl;
    mono_slam.fix_estim_vars_covar_symmetry_ = FLAGS_monoslam_fix_estim_vars_covar_symmetry;
    if (FLAGS_monoslam_debug_max_sal_pnt_count != -1)
        mono_slam.debug_max_sal_pnt_count_ = FLAGS_monoslam_debug_max_sal_pnt_count;
    if (demo_data_source == DemoDataSource::kVirtualScene)
    {
        std::optional<suriko::Point3> cam_vel_tracker;
        std::optional<suriko::Point3> cam_ang_vel_c;
        if (suriko::Point3 cam_vel_tracker_tmp, cam_ang_vel_c_tmp;
            GetSyntheticCameraInitialMovement(gt_cam_orient_cfw, &cam_vel_tracker_tmp, &cam_ang_vel_c_tmp))
        {
            if (FLAGS_monoslam_cam_perfect_init_vel) cam_vel_tracker = cam_vel_tracker_tmp;
            if (FLAGS_monoslam_cam_perfect_init_ang_vel) cam_ang_vel_c = cam_ang_vel_c_tmp;
        }
        mono_slam.SetCameraVelocity(cam_vel_tracker, cam_ang_vel_c);

        //
        mono_slam.sal_pnt_perfect_init_inv_dist_ = FLAGS_monoslam_sal_pnt_perfect_init_inv_dist;
        mono_slam.set_estim_state_covar_to_gt_impl_ = FLAGS_monoslam_set_estim_state_covar_to_gt_impl;

        // covariances used together with ground truth state
        mono_slam.sal_pnt_pos_x_std_if_gt_ = static_cast<Scalar>(FLAGS_monoslam_sal_pnt_pos_x_std_if_gt);
        mono_slam.sal_pnt_pos_y_std_if_gt_ = static_cast<Scalar>(FLAGS_monoslam_sal_pnt_pos_y_std_if_gt);
        mono_slam.sal_pnt_pos_z_std_if_gt_ = static_cast<Scalar>(FLAGS_monoslam_sal_pnt_pos_z_std_if_gt);
        mono_slam.sal_pnt_first_cam_pos_std_if_gt_ = static_cast<Scalar>(FLAGS_monoslam_sal_pnt_first_cam_pos_std_if_gt);
        mono_slam.sal_pnt_azimuth_std_if_gt_ = static_cast<Scalar>(FLAGS_monoslam_sal_pnt_azimuth_std_if_gt);
        mono_slam.sal_pnt_elevation_std_if_gt_ = static_cast<Scalar>(FLAGS_monoslam_sal_pnt_elevation_std_if_gt);
        mono_slam.sal_pnt_inv_dist_std_if_gt_ = static_cast<Scalar>(FLAGS_monoslam_sal_pnt_inv_dist_std_if_gt);

        mono_slam.gt_cami_from_world_fun_ = [&gt_cam_orient_cfw](size_t frame_ind) -> SE3Transform
        {
            SE3Transform c = gt_cam_orient_cfw[frame_ind].value();
            return c;
        };
        mono_slam.gt_cami_from_tracker_new_ = [&gt_cam_orient_cfw](SE3Transform tracker_from_world, size_t frame_ind) -> std::optional<SE3Transform>
        {
            if (frame_ind >= gt_cam_orient_cfw.size())
                return std::nullopt;

            std::optional<SE3Transform> cami_from_world = gt_cam_orient_cfw[frame_ind];
            if (!cami_from_world.has_value()) return std::nullopt;

            SE3Transform cami_from_tracker = SE3AFromB(cami_from_world.value(), tracker_from_world);
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

            Dir3DAndDistance p;
            p.unity_dir = Normalized(pnt_camera);
            p.dist = Norm(pnt_camera);
            return p;
        };
    }

    // perhaps these values should be just constants
    mono_slam.cam_pos_x_std_m_ = static_cast<Scalar>(FLAGS_monoslam_cam_pos_x_std_m);
    mono_slam.cam_pos_y_std_m_ = static_cast<Scalar>(FLAGS_monoslam_cam_pos_y_std_m);
    mono_slam.cam_pos_z_std_m_ = static_cast<Scalar>(FLAGS_monoslam_cam_pos_z_std_m);
    mono_slam.cam_orient_q_comp_std_ = static_cast<Scalar>(FLAGS_monoslam_cam_orient_q_comp_std);
    mono_slam.cam_vel_std_ = static_cast<Scalar>(FLAGS_monoslam_cam_vel_std);
    mono_slam.cam_ang_vel_std_ = static_cast<Scalar>(FLAGS_monoslam_cam_ang_vel_std);
    mono_slam.SetCameraStateCovarHelper();

    LOG(INFO) << "mono_slam_dT=" << mono_slam.seconds_per_frame_;
    LOG(INFO) << "mono_slam_process_noise_lin_veloc_std=" << mono_slam.process_noise_linear_velocity_std_;
    LOG(INFO) << "mono_slam_process_noise_ang_veloc_std=" << mono_slam.process_noise_angular_velocity_std_;
    LOG(INFO) << "mono_slam_measurm_noise_std_pix=" << mono_slam.measurm_noise_std_pix_;
    LOG(INFO) << "mono_slam_update_impl=" << FLAGS_monoslam_update_impl;
    LOG(INFO) << "mono_slam_sal_pnt_vars=" << DavisonMonoSlam::kSalientPointComps;
    LOG(INFO) << "mono_slam_templ_center_detection_noise_std_pix=" << FLAGS_monoslam_templ_center_detection_noise_std_pix;
    LOG(INFO) << "mono_slam_sal_pnt_negative_inv_rho_substitute=" << mono_slam.sal_pnt_negative_inv_rho_substitute_.value_or(static_cast<Scalar>(-1));

    if (demo_data_source == DemoDataSource::kVirtualScene)
    {
        auto corners_matcher = std::make_shared<DemoCornersMatcher>(&mono_slam, gt_cam_orient_cfw, entire_map, cam_intrinsics.image_size);
        corners_matcher->SetTemplCenterDetectionNoiseStd(static_cast<float>(FLAGS_monoslam_templ_center_detection_noise_std_pix));
        corners_matcher->tracker_origin_from_world_ = tracker_origin_from_world;

        if (FLAGS_monoslam_max_new_blobs_in_first_frame > 0)
            corners_matcher->max_new_blobs_in_first_frame_ = FLAGS_monoslam_max_new_blobs_in_first_frame;
        if (FLAGS_monoslam_max_new_blobs_per_frame > 0)
            corners_matcher->max_new_blobs_per_frame_ = FLAGS_monoslam_max_new_blobs_per_frame;
        if (FLAGS_monoslam_match_blob_prob > 0)
            corners_matcher->match_blob_prob_ = (float)FLAGS_monoslam_match_blob_prob;

        mono_slam.SetCornersMatcher(corners_matcher);
    }
    else if (demo_data_source == DemoDataSource::kImageSeqDir)
    {
        int detect_orb_corners_per_frame = FLAGS_monoslam_match_detect_corners_per_frame;

        auto corners_matcher = std::make_shared<ImageTemplCornersMatcher>(detect_orb_corners_per_frame);
        corners_matcher->closest_corner_min_dist_pix_ = static_cast<Scalar>(FLAGS_monoslam_match_closest_corner_min_dist_pix);
        corners_matcher->match_corners_impl_ = FLAGS_monoslam_match_corners_impl == 2 ? MatchCornerImpl::OrbDescr : MatchCornerImpl::Templ;
        corners_matcher->match_corners_impl_ = FLAGS_monoslam_match_corners_impl == 2 ? MatchCornerImpl::OrbDescr : MatchCornerImpl::Templ;
        corners_matcher->match_orb_descr_max_hamming_dist_ = FLAGS_monoslam_orb_descr_max_hamming_distance;
        corners_matcher->stop_on_sal_pnt_moved_too_far_ = FLAGS_monoslam_match_stop_on_sal_pnt_moved_too_far;
        corners_matcher->force_sequential_execution_ = FLAGS_monoslam_force_sequential_engine;
        corners_matcher->min_search_rect_size_ = suriko::Sizei{ FLAGS_monoslam_templ_min_search_rect_width, FLAGS_monoslam_templ_min_search_rect_height };
        corners_matcher->min_templ_corr_coeff_ = config_reader.GetValue<Scalar>("monoslam_templ_min_corr_coeff").value_or(corners_matcher->min_templ_corr_coeff_);
        
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
        mono_slam.SetCornersMatcher(corners_matcher);
    }

    if (FLAGS_ctrl_collect_tracker_internals)
    {
        mono_slam.SetStatsLogger(std::make_shared<DavisonMonoSlamInternalsLogger>());
    }

    auto unused_params = config_reader.GetUnusedParams();
    for (auto param : unused_params)
        LOG(INFO) << "Unused param=" << param;


#if defined(SRK_HAS_OPENCV)
    cv::Mat camera_image_bgr = cv::Mat::zeros(cam_intrinsics.image_size.height, (int)cam_intrinsics.image_size.width, CV_8UC3);
#endif
#if defined(SRK_HAS_PANGOLIN)
    // across threads shared data
    auto worker_chat = std::make_shared<WorkerChatSharedState>();
    ptrdiff_t observable_frame_ind = -1; // this is visualized by UI, it is one frame less than current frame

    UIThreadParams ui_params {};
    ui_params.wait_for_user_input_after_each_frame = FLAGS_ctrl_wait_after_each_frame;
    ui_params.mono_slam = &mono_slam;
    ui_params.tracker_origin_from_world = tracker_origin_from_world;
    ui_params.covar3D_to_ellipsoid_chi_square = static_cast<Scalar>(FLAGS_ui_covar3D_to_ellipsoid_chi_square);
    ui_params.estim_cam_orient_cfw = &estim_cam_orient_cfw;
    ui_params.get_observable_frame_ind_fun = [&observable_frame_ind]() { return observable_frame_ind; };
    ui_params.worker_chat = worker_chat;
    ui_params.ui_swallow_exc = FLAGS_ui_swallow_exc;
    ui_params.ui_tight_loop_relaxing_delay = std::chrono::milliseconds(FLAGS_ui_tight_loop_relaxing_delay_ms);
    ui_params.entire_map = &entire_map;
    ui_params.gt_cam_orient_cfw = &gt_cam_orient_cfw;
    ui_params.external_cam_orient_cfw = &external_cam_orient_cfw;

    static constexpr int kKeyForward = static_cast<int>('f'); // 'Forward'
    static constexpr int kKeyIgnoreDetection = static_cast<int>('s'); // 'Skip'
    static constexpr int kKeySetToGroundTruth = static_cast<int>('u');
    static constexpr int kKeyDumpInfo = static_cast<int>('i');

    std::thread ui_thread;
    const bool defer_ui_construction = true;
    auto pangolin_gui = SceneVisualizationPangolinGui::New(defer_ui_construction);  // used in single threaded mode
    if (FLAGS_ctrl_visualize_during_processing)
    {
        if (FLAGS_ctrl_ui_in_separate_thread)
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
            pangolin_gui->SetCameraBehindTracker();
        }
    }
#endif

    CheckDavisonMonoSlamConfigurationAndDump(mono_slam, gt_cam_orient_cfw);

    // undistort images after reading them, so that the tracking algorithm gets undistorted images as input
    std::optional<UndistMaps> undistort_img_maps;
    auto undistort_img_cam_mat_by_rows = FloatSeq<double>(&config_reader, "undistort_img_cam_mat_by_rows");
    if (undistort_img_cam_mat_by_rows.has_value())
    {
        std::vector<double> A_vec = undistort_img_cam_mat_by_rows.value();
        if (A_vec.size() != 3 * 3)
        {
            LOG(ERROR) << "Undistort.CameraMatrix must be vector of 9 elements, which is 3x3 by rows";
            return 1;
        }

        auto undistort_img_dist_coeffs = FloatSeq<double>(&config_reader, "undistort_img_dist_coeffs").value_or(std::vector<Scalar>{0, 0, 0, 0});

        UndistMaps m;
#if defined(SRK_HAS_OPENCV)
        m.cam_mat = cv::Matx<double, 3, 3> { undistort_img_cam_mat_by_rows.value().data() };
        m.dist_coeffs = cv::Mat{ undistort_img_dist_coeffs, false };

        cv::Size cv_img_size{ cam_intrinsics.image_size.width, cam_intrinsics.image_size.height };
        auto I = cv::Mat_<double>::eye(3, 3);
        cv::initUndistortRectifyMap(m.cam_mat, m.dist_coeffs, I, m.cam_mat, cv_img_size, CV_16SC2, m.map1, m.map2);
#endif
        undistort_img_maps = m;
    }

    std::filesystem::path root_dump_dir = std::filesystem::absolute(FLAGS_ctrl_log_slam_images_dir);

    //
    size_t frame_ind = -1;
    std::filesystem::directory_iterator dir_it;

    if (demo_data_source == DemoDataSource::kImageSeqDir)
    {
        LOG(INFO) << "imageseq_dir=" << scene_imageseq_dir;
        dir_it = std::filesystem::directory_iterator(scene_imageseq_dir);
    }
    bool iterate_frames = true;
    while(iterate_frames)  // for each frame
    {
        ++frame_ind;

        Picture image;
        cv::Mat image_bgr;

        if (demo_data_source == DemoDataSource::kVirtualScene)
        {
            if (frame_ind >= frames_count) break;

            image_bgr.create(cv::Size(cam_intrinsics.image_size.width, cam_intrinsics.image_size.height), CV_8UC3);
        }
        else if (demo_data_source == DemoDataSource::kImageSeqDir)
        {
            if (dir_it == std::filesystem::directory_iterator()) break;

            std::filesystem::directory_entry dir_entry = *dir_it;
            dir_it++;

            auto image_file_path = dir_entry.path();
            auto path_str = image_file_path.string();
            LOG(INFO) << path_str;

            image_bgr = cv::imread(image_file_path.string());
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

            if (undistort_img_maps.has_value())  // cv::undistort is requested by user
            {
                UndistMaps& m = undistort_img_maps.value();
#if defined(SRK_HAS_OPENCV)
                cv::Mat image_bgr_undist;
                cv::remap(image_bgr, image_bgr_undist, m.map1, m.map2, cv::INTER_LINEAR);
                image_bgr = image_bgr_undist;
#endif
            }

            cv::Mat image_gray;
            cv::cvtColor(image_bgr, image_gray, cv::COLOR_BGR2GRAY);

            image.gray = image_gray;
#if defined(SRK_DEBUG)
            image.bgr_debug = image_bgr;
#endif
        }

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
            estim_cam_orient_cfw.push_back(actual_cam_cfw);

#if defined(SRK_HAS_PANGOLIN)
            observable_frame_ind = frame_ind;
#endif
        }

        std::optional<std::chrono::duration<double>> frame_OpenCV_gui_time;
        std::optional<std::chrono::duration<double>> frame_Pangolin_gui_time;

        // Draw / Render loop (usually run once, but a user may request to redraw current frame for some reason)
        for (int redraw_times = 1; redraw_times > 0 && iterate_frames; redraw_times--)
        {
#if defined(SRK_HAS_OPENCV)
            if (FLAGS_ctrl_visualize_during_processing)
            {
                if (demo_data_source == DemoDataSource::kVirtualScene)
                {
                    auto draw_virtual_scene = [&mono_slam, &gt_cam_orient_cfw, frame_ind](const DemoCornersMatcher& corners_matcher, cv::Mat* out_image_bgr)
                    {
                        out_image_bgr->setTo(0);

                        std::optional<SE3Transform> cfw_opt = gt_cam_orient_cfw[frame_ind];
                        if (!cfw_opt.has_value()) return;

                        // the world axes are drawn on the image to provide richer context about the structure of the scene
                        // (drawing just salient points would be vague)
                        const SE3Transform& rt_cfw = cfw_opt.value();
                        auto project_fun = [&rt_cfw, &mono_slam](const suriko::Point3& sal_pnt_world) -> Point3
                        {
                            suriko::Point3 pnt_cam = SE3Apply(rt_cfw, sal_pnt_world);
                            suriko::Point2f pnt_pix = mono_slam.ProjectCameraPoint(pnt_cam);
                            return AsHomog(pnt_pix);
                        };
                        constexpr Scalar f0 = 1;
                        suriko_demos::Draw2DProjectedAxes(f0, project_fun, out_image_bgr);

                        //
                        for (const BlobInfo& blob_info : corners_matcher.DetectedBlobs())
                        {
                            Scalar pix_x = blob_info.Coord[0];
                            Scalar pix_y = blob_info.Coord[1];
                            out_image_bgr->at<cv::Vec3b>((int)pix_y, (int)pix_x) = cv::Vec3b(0xFF, 0xFF, 0xFF);
                        }
                    };

                    auto& corners_matcher = mono_slam.CornersMatcher();
                    auto a_corners_matcher = dynamic_cast<DemoCornersMatcher*>(&corners_matcher);
                    if (a_corners_matcher != nullptr)
                    {
                        draw_virtual_scene(*a_corners_matcher, &image_bgr);
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();

                image_bgr.copyTo(camera_image_bgr);  // background
                drawer.DrawScene(mono_slam, &camera_image_bgr);

                std::stringstream strbuf;
                strbuf << "f=" << frame_ind;
                cv::putText(camera_image_bgr, cv::String(strbuf.str()), cv::Point(10, (int)cam_intrinsics.image_size.height - 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));

                auto t2 = std::chrono::high_resolution_clock::now();
                frame_OpenCV_gui_time = t2 - t1;

                if (FLAGS_ctrl_log_slam_images_cam0)
                    DumpOpenCVImage(root_dump_dir, frame_ind, "cam0", camera_image_bgr);

                cv::imshow("front-camera", camera_image_bgr);
                cv::waitKey(1); // allow to refresh an opencv view
            }
#endif
#if defined(SRK_HAS_PANGOLIN)
            if (FLAGS_ctrl_ui_in_separate_thread)
            {
                // check if UI requests the exit
                std::lock_guard<std::mutex> lk(worker_chat->the_mutex);
                std::optional<WorkerChatMessage> msg = PopMsgUnderLock(&worker_chat->worker_message);
                if (msg == WorkerChatMessage::WorkerExit)
                    break;
            }

            auto pangolin_key_handler = [&mono_slam, &frame_ind, &redraw_times](int key) -> KeyHandlerResult
            {
                KeyHandlerResult handler_result {false};
                switch (key)
                {
                case kKeySetToGroundTruth:
                {
                    mono_slam.SetEstimStateAndCovarToGroundTruth(frame_ind);
                    handler_result.handled = true;
                    handler_result.stop_wait_loop = true;
                    redraw_times++;  // request redrawing the OpenCV viewer
                    break;
                }
                case kKeyDumpInfo:
                    std::ostringstream os;
                    mono_slam.DumpTrackerState(os);
                    LOG(INFO) << os.str();
                    handler_result.handled = true;
                    break;
                }
                return handler_result;
            };

            // update UI
            if (FLAGS_ctrl_visualize_during_processing)
            {
                if (FLAGS_ctrl_log_slam_images_scene3D)
                {
                    std::filesystem::path scene_dump_dir = root_dump_dir / "scene3D";
                    EnsureAllSubdirsExist(scene_dump_dir);

                    std::string img_file_name_no_ext = FormatImageFileNameNoExt(frame_ind);
                    std::filesystem::path scene_out_file_path = scene_dump_dir / img_file_name_no_ext;
                    pangolin_gui->SetOnRenderOutputFilePath(scene_out_file_path.string());
                }

                auto stop_wait_on_key = [](int key)
                {
                    return key == kKeyIgnoreDetection || key == kKeyForward || key == pangolin::PANGO_KEY_ESCAPE;
                };

                if (FLAGS_ctrl_wait_after_each_frame)
                {
                    // let a user observe the UI and signal back when to continue

                    int key = -1;
                    if (FLAGS_ctrl_ui_in_separate_thread)
                    {
                        std::unique_lock<std::mutex> ulk(worker_chat->the_mutex);
                        worker_chat->ui_message = UIChatMessage::UIWaitKey; // reset the waiting flag
                        worker_chat->ui_wait_key_predicate_ = stop_wait_on_key;

                        // wait till UI requests to resume processing
                        worker_chat->worker_got_new_message_cv.wait(ulk, [&worker_chat] {return worker_chat->worker_message == WorkerChatMessage::WorkerKeyPressed; });
                        key = worker_chat->ui_pressed_key.value_or(-1);
                    }
                    else
                    {
                        // initialize GUI lazily here, because the handler depends on frame_ind which is not known during initialization
                        if (pangolin_gui->key_pressed_handler_ == nullptr)
                            pangolin_gui->key_pressed_handler_ = pangolin_key_handler;

                        key = pangolin_gui->WaitKey(stop_wait_on_key);
                    }

                    if (key == pangolin::PANGO_KEY_ESCAPE)
                    {
                        iterate_frames = false;
                        break;
                    }

                    const bool suppress_observations = (key == kKeyIgnoreDetection);
                    mono_slam.CornersMatcher().SetSuppressObservations(suppress_observations);
                }
                else
                {
                    if (FLAGS_ctrl_ui_in_separate_thread) {}
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
                        {
                            iterate_frames = false;
                            break;
                        }
                    }
                }
            }
#endif
        }
        auto zero_time = std::chrono::seconds{ 0 };
        auto total_time = 
            frame_process_time.value_or(zero_time) +
            frame_OpenCV_gui_time.value_or(zero_time) +
            frame_Pangolin_gui_time.value_or(zero_time);
        VLOG(4) << "done f=" << frame_ind
            << " track=" << std::chrono::duration_cast<std::chrono::milliseconds>(frame_process_time.value_or(zero_time)).count() << "ms"
            << "|" << (frame_process_time.has_value() ? 1 / frame_process_time.value().count() : 0.0f) << "fps"
            << " track+gui=" << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() << "ms"
            << "|" << 1 / total_time.count() << "fps"
            << " #SP=" << mono_slam.SalientPointsCount();
    } // for each frame

    VLOG(4) << "Finished processing all the frames";

#if defined(SRK_HAS_PANGOLIN)
    if (FLAGS_ctrl_visualize_after_processing)
    {
        if (FLAGS_ctrl_ui_in_separate_thread)
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
    if (FLAGS_ctrl_visualize_after_processing)
        cv::waitKey(0); // 0=wait forever
#endif

    // dump trajectory of a camera
    if (FLAGS_ctrl_log_slam_cam_traj)
    {
        std::vector<TumTimestampPose> cam_poses_cfw;
        cam_poses_cfw.resize(estim_cam_orient_cfw.size());
        for (size_t i = 0; i < estim_cam_orient_cfw.size(); ++i)
        {
            const SE3Transform& s = estim_cam_orient_cfw[i];

            TumTimestamp stamp = i * mono_slam.seconds_per_frame_;
            if (i < estim_rgb_timestamps.size())  // image timestamp is available
                stamp = estim_rgb_timestamps[i];

            TumTimestampPose pose_cfw;
            pose_cfw.timestamp = stamp;
            pose_cfw.pos = s.T;
            Eigen::Matrix<Scalar, 4, 1 > q;
            QuatFromRotationMatNoRChecks(s.R, gsl::make_span<Scalar>(q.data(), 4));
            pose_cfw.quat = q;
            cam_poses_cfw[i] = pose_cfw;
        }
        std::string err_mgs;
        bool dump_traj = SaveTumDatasetGroundTruth("cam_traj.txt", &cam_poses_cfw, QuatLayout::XyzW, &err_mgs);
        if (!dump_traj)
            LOG(ERROR) << "Can't dump camera's trajectory. " << err_mgs;
    }
    if (bool calc_traj_error = true)
    {
        std::vector<std::optional<SE3Transform>> estim_cfw;
        estim_cfw.reserve(estim_cam_orient_cfw.size());
        std::transform(estim_cam_orient_cfw.begin(), estim_cam_orient_cfw.end(), std::back_inserter(estim_cfw), [](const SE3Transform& t) { return std::make_optional(t); });

        // calculate average camera movement in estimated trajectory
        {
            MeanStdAlgo stat_calc;
            std::vector<Scalar> cam_shifts;
            for (size_t i = 1; i < estim_cfw.size(); ++i)
            {
                const auto& v1 = estim_cfw[i - 1];
                const auto& v2 = estim_cfw[i];
                if (!v1.has_value() || !v2.has_value()) continue;
                SE3Transform wfc1 = SE3Inv(v1.value());
                SE3Transform wfc2 = SE3Inv(v2.value());
                Scalar cam_shift = (wfc2.T - wfc1.T).norm();
                stat_calc.Next(cam_shift);
                cam_shifts.push_back(cam_shift);
            }
            if (!cam_shifts.empty())
            {
                Scalar median = LeftMedianInplace(&cam_shifts).value();
                LOG(INFO) << "Estim cam shift (m/frame): {"
                    << "median:" << median
                    << ",mean:" << stat_calc.Mean()
                    << ",std:" << stat_calc.Std()
                    << ",min:" << stat_calc.Min().value()
                    << ",max:" << stat_calc.Max().value() << "}";
            }
        }

        size_t poses_count = std::min(estim_cfw.size(), gt_cam_orient_cfw.size());
        auto gt_poses = gsl::make_span(gt_cam_orient_cfw).first(poses_count);

        std::vector<Scalar> transl_errs;
        bool op = CalcRelativePoseError(gt_poses, estim_cfw, &transl_errs);
        if (!op)
            LOG(ERROR) << "Can't calculate relative pose error.";
        else
        {
            std::optional<ErrWithMoments> rpe = CalcTrajectoryErrStats(transl_errs);
            if (rpe.has_value())
            {
                const ErrWithMoments& e = rpe.value();
                LOG(INFO) << "RPE (m/frame): {"
                    << "RMSE:" << e.rmse
                    << ",median:" << e.median
                    << ",mean:" << e.mean
                    << ",std:" << e.std
                    << ",min:" << e.min
                    << ",max:" << e.max << "}";
            }
        }
    }

    //
    if (mono_slam.StatsLogger() != nullptr)
    {
        const DavisonMonoSlamTrackerInternalsHist& internal_stats = mono_slam.StatsLogger()->BuildStats();

        bool dump_op = WriteTrackerInternalsToFile("davison_tracker_internals.json", internal_stats);
        if (FLAGS_ctrl_collect_tracker_internals && !dump_op)
            LOG(ERROR) << "Can't dump the tracker's internals";
    }

    return 0;
}
}

int main(int argc, char* argv[])
{
    auto t1 = std::chrono::high_resolution_clock::now();

    int result = suriko_demos_davison_mono_slam::DavisonMonoSlamDemo(argc, argv);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> exec_time = t2 - t1;
    VLOG(4) << "exec time=" << std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count() << "ms";

    return result;
}
