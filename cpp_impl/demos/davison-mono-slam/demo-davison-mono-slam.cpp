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
#include "suriko/config-reader.h"
#include "suriko/davison-mono-slam.h"
#include "suriko/virt-world/scene-generator.h"
#include "suriko/quat.h"
#include "../stat-helpers.h"
#include "../visualize-helpers.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/imgcodecs.hpp> // cv::imread
#include <opencv2/imgproc.hpp> // cv::circle, cv::cvtColor
#include <opencv2/highgui.hpp> // cv::imshow
#include <opencv2/features2d.hpp> // cv::ORB
#endif

#include "demo-davison-mono-slam-ui.h"

namespace suriko_demos_davison_mono_slam
{
using namespace std;
using namespace suriko; 
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

bool GetSyntheticCameraInitialMovement(const std::vector<SE3Transform>& gt_cam_orient_cfw,
    suriko::Point3* cam_vel_tracker,
    suriko::Point3* cam_ang_vel_c)
{
    if (gt_cam_orient_cfw.size() < 2)
        return false;
    SE3Transform c0_from_world = gt_cam_orient_cfw[0];
    SE3Transform c1_from_world = gt_cam_orient_cfw[1];
    SE3Transform world_from_c0 = SE3Inv(c0_from_world);
    SE3Transform world_from_c1 = SE3Inv(c1_from_world);

    // In a synthetic scenario we can perfectly foresee the movement of camera (from frame 0 to 1).
    // When processing frame 1, the residual should be zero.
    // camera's velocity
    // Tw1=Tw0+v01_w
    // v01_w=velocity from camera-0 to camera-1 in world coordinates
    auto init_shift_world = suriko::Point3{ world_from_c1.T - world_from_c0.T };
    auto init_shift_tracker = suriko::Point3{ c0_from_world.R * init_shift_world.Mat() };
    *cam_vel_tracker = init_shift_tracker;

    // camera's angular velocity
    // Rw1=Rw0*R01, R01=delta, which rotates from camera-1 to camera-0.
    SE3Transform c0_from_c1 = SE3AFromB(c0_from_world, c1_from_world);

    Eigen::Matrix<Scalar, 3, 1> axisangle_c0_from_c1;
    bool op = AxisAngleFromRotMat(c0_from_c1.R, &axisangle_c0_from_c1);
    if (!op)
        axisangle_c0_from_c1.fill(0);
    *cam_ang_vel_c = suriko::Point3{ axisangle_c0_from_c1[0], axisangle_c0_from_c1[1], axisangle_c0_from_c1[2] };
    return true;
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
    suriko::Point2f Coord;
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
    std::optional<float> match_blob_prob_ = 1.0f; // [0,1] portion of blobs which are matched with ones in the previous frame; 1=all matched, 0=none matched;
    std::mt19937 gen_{292};
    std::uniform_real_distribution<float> uniform_distr_{};
    std::normal_distribution<float> templ_center_detection_noise_distr_{};
    float templ_center_detection_noise_std_ = 0;  // 0=no noise; added to blob's center to mimic measurement noise of a filter
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

    suriko::Point2f GetBlobCoord(CornersMatcherBlobId blob_id) override
    {
        return detected_blobs_[blob_id.Ind].Coord;
    }

    void SetTemplCenterDetectionNoiseStd(float value)
    {
        templ_center_detection_noise_std_ = value;
        if (value > 0)
            templ_center_detection_noise_distr_ = std::normal_distribution<float>{ 0, value };
    }
    
    std::optional<Scalar> GetSalientPointGroundTruthInvDepth(CornersMatcherBlobId blob_id) override
    {
        return detected_blobs_[blob_id.Ind].GTInvDepth;
    }

    void SetSuppressObservations(bool value) { suppress_observations_ = value; }

    const std::vector<BlobInfo>& DetectedBlobs() const { return detected_blobs_; }
};

#if defined(SRK_HAS_OPENCV)

class ImageTemplCornersMatcher : public CornersMatcherBase
{
    bool suppress_observations_ = false; // true to make camera magically don't detect any salient points
    cv::Ptr<cv::ORB> detector_;
    DavisonMonoSlam* mono_slam_;
    std::vector<cv::KeyPoint> new_keypoints_;
public:
    bool stop_on_sal_pnt_moved_too_far_ = false;
    Scalar ellisoid_cut_thr_;
    std::function<void(DavisonMonoSlam&, SalPntId, cv::Mat*)> draw_sal_pnt_fun_;
    std::function<void(std::string_view, cv::Mat)> show_image_fun_;
    std::optional<suriko::Sizei> min_search_rect_size_;
    std::optional<Scalar> min_templ_corr_coeff_;
public:
    ImageTemplCornersMatcher(DavisonMonoSlam* mono_slam)
        :mono_slam_(mono_slam)
    {
        int nfeatures = 50;
        detector_ = cv::ORB::create(nfeatures);
    }

    struct TemplateMatchResult
    {
        bool success;
        suriko::Point2f center;
        Scalar corr_coef;

#if defined(SRK_DEBUG)
        suriko::Point2i top_left;
        int executed_match_templ_calls; // specify the number of calls to match-template routine to find this specific match-result
#endif
    };

    TemplateMatchResult MatchSalientPointTemplCenterInRect(const TrackedSalientPoint& sal_pnt, const Picture& pic, Recti search_rect)
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

        auto match_templ_at = [this, &templ_gray, &pic,
            &max_corr_coeff, &best_match_info, templ_mean, templ_sqrt_sum_sqr_diff, &match_templ_call_order](Point2i search_center)
        {
#if defined(SRK_DEBUG)
            match_templ_call_order++;
#endif
            Point2i pic_roi_top_left = mono_slam_->TemplateTopLeftInt(suriko::Point2f{ search_center.x, search_center.y });

            auto pic_roi = suriko::Recti{ pic_roi_top_left.x, pic_roi_top_left.y,
                mono_slam_->sal_pnt_templ_size_.width,
                mono_slam_->sal_pnt_templ_size_.height
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

    std::tuple<bool,Recti> PredictSalientPointSearchRect(DavisonMonoSlam& mono_slam, SalPntId sal_pnt_id, Scalar ellisoid_cut_thr)
    {
        auto [op_cov, corner] = mono_slam.GetSalientPointProjected2DPosWithUncertainty(FilterStageType::Predicted, sal_pnt_id);
        static_assert(std::is_same_v<decltype(corner), MeanAndCov2D>);
        SRK_ASSERT(op_cov);
        if (!op_cov) return std::make_tuple(false, Recti{});

        // an ellipse can always be extracted from 'good' covariance mat of error in position
        // but here we allow bad covariance matrix
        auto [op_2D_ellip, corner_ellipse] = Get2DRotatedEllipseFromCovMat(corner.cov, corner.mean, ellisoid_cut_thr);
        static_assert(std::is_same_v<decltype(corner_ellipse), RotatedEllipse2D>);
        SRK_ASSERT(op_2D_ellip);
        if (!op_2D_ellip) return std::make_tuple(false, Recti{});

        Rect corner_bounds = GetEllipseBounds2(corner_ellipse);
        Recti corner_bounds_i = TruncateRect(corner_bounds);
        return std::make_tuple(true, corner_bounds_i);
    }

    std::optional<suriko::Point2f> MatchSalientTempl(DavisonMonoSlam& mono_slam, SalPntId sal_pnt_id, const Picture& pic,
        Scalar ellisoid_cut_thr)
    {
        auto [op, search_rect_unbounded] = PredictSalientPointSearchRect(mono_slam, sal_pnt_id, ellisoid_cut_thr);
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
        if (min_templ_corr_coeff_.has_value() && match_result.corr_coef < min_templ_corr_coeff_.value())
        {
            static bool debug_matching = false;
            if (debug_matching)
            {
                auto [op, predicted_center] = mono_slam_->GetSalientPointProjected2DPosWithUncertainty(FilterStageType::Predicted, sal_pnt_id);
                static_assert(std::is_same_v<decltype(predicted_center), MeanAndCov2D>);
                SRK_ASSERT(op);
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
            const TrackedSalientPoint& sal_pnt = mono_slam_->GetSalientPoint(sal_pnt_id);

            std::optional<suriko::Point2f> match_pnt_center = MatchSalientTempl(*mono_slam_, sal_pnt_id, image, ellisoid_cut_thr_);
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
        detector_->detect(image.gray, keypoints);  // keypoints are sorted by ascending size [W,H]

        // reorder the features from high quality to low
        // this will lead to deterministic creation and matching of image features
        // otherwise different features may be selected for the same picture for different program's executions
        std::sort(keypoints.begin(), keypoints.end(), [](auto& a, auto& b) { return a.response > b.response; });

        cv::Mat keyp_img;
        static bool debug_keypoints = false;
        if (debug_keypoints)
            cv::drawKeypoints(image.gray, keypoints, keyp_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::Mat descr_per_row;
        detector_->compute(image.gray, keypoints, descr_per_row);

        std::vector<cv::KeyPoint> sparse_keypoints;
        Scalar closest_templ_min_dist = mono_slam_->ClosestSalientPointTemplateMinDistance();
        FilterOutClosest(keypoints, closest_templ_min_dist, &sparse_keypoints);

        cv::Mat sparse_img;
        if (debug_keypoints)
            cv::drawKeypoints(image.gray, sparse_keypoints, sparse_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // remove keypoints which are close to 'matched' salient points
        auto filter_out_close_to_existing = [this,&matched_sal_pnts](const std::vector<cv::KeyPoint>& keypoints, Scalar exclude_radius,
            std::vector<cv::KeyPoint>* result)
        {
            for (size_t cand_ind = 0; cand_ind < keypoints.size(); ++cand_ind)
            {
                const auto& cand = keypoints[cand_ind];

                bool has_close_blob = false;
                for (auto[sal_pnt_id, blob_id] : matched_sal_pnts)
                {
                    std::optional<suriko::Point2f> exist_pix_opt = mono_slam_->GetDetectedSalientTemplCenter(sal_pnt_id);
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

        new_keypoints_.clear();
        filter_out_close_to_existing(sparse_keypoints, closest_templ_min_dist, &new_keypoints_);

        cv::Mat img_no_closest;
        if (debug_keypoints)
            cv::drawKeypoints(image.gray, new_keypoints_, img_no_closest, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        for (size_t i=0; i< new_keypoints_.size(); ++i)
        {
            new_blob_ids->push_back(CornersMatcherBlobId{i});
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

    suriko::Point2f GetBlobCoord(CornersMatcherBlobId blob_id) override
    {
        const cv::KeyPoint& kp = new_keypoints_[blob_id.Ind];
        // some pixels already have fractional X or Y coordinate, like 213.2, so return it without changes
        return suriko::Point2f{ kp.pt.x, kp.pt.y };
    }

    Picture GetBlobTemplate(CornersMatcherBlobId blob_id, const Picture& image) override
    {
        const cv::KeyPoint& kp = new_keypoints_[blob_id.Ind];

        int rad_x_int = mono_slam_->sal_pnt_templ_size_.width / 2;
        int rad_y_int = mono_slam_->sal_pnt_templ_size_.height / 2;

        int center_x = (int)kp.pt.x;
        int center_y = (int)kp.pt.y;
        cv::Rect templ_bounds{ center_x - rad_x_int, center_y - rad_y_int, mono_slam_->sal_pnt_templ_size_.width, mono_slam_->sal_pnt_templ_size_.height };

        cv::Mat templ_gray;
        image.gray(templ_bounds).copyTo(templ_gray);
        SRK_ASSERT(templ_gray.rows == mono_slam_->sal_pnt_templ_size_.height);
        SRK_ASSERT(templ_gray.cols == mono_slam_->sal_pnt_templ_size_.width);

        Picture templ{};
        templ.gray = templ_gray;

#if defined(SRK_DEBUG)
        cv::Mat templ_mat;
        image.bgr_debug(templ_bounds).copyTo(templ_mat);
        templ.bgr_debug = templ_mat;
#endif
        return templ;
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

std::optional<Scalar> GetMaxCamShift(const std::vector<SE3Transform>& gt_cam_orient_cfw)
{
    std::optional<SE3Transform> prev_cam_wfc;
    std::optional<Scalar> between_frames_max_cam_shift;
    for (const auto& cfw : gt_cam_orient_cfw)
    {
        auto wfc = SE3Inv(cfw);
        if (prev_cam_wfc.has_value())
        {
            Eigen::Matrix<Scalar, 3, 1> cam_shift = wfc.T - prev_cam_wfc.value().T;
            auto dist = cam_shift.norm();
            if (!between_frames_max_cam_shift.has_value() ||
                dist > between_frames_max_cam_shift.value())
                between_frames_max_cam_shift = dist;
        }
        prev_cam_wfc = wfc;
    }
    return between_frames_max_cam_shift;
}

void CheckDavisonMonoSlamConfigurationAndDump(const DavisonMonoSlam& mono_slam, 
    const std::vector<SE3Transform>& gt_cam_orient_cfw)
{
    // check max shift of the camera is expected by tracker
    std::optional<Scalar> between_frames_max_cam_shift = GetMaxCamShift(gt_cam_orient_cfw);
    if (between_frames_max_cam_shift.has_value())
    {
        auto max_expected_cam_shift = mono_slam.process_noise_std_ * 3;
        if (between_frames_max_cam_shift.value() > max_expected_cam_shift)
            LOG(INFO)
                << "Note: max_cam_shift=" << between_frames_max_cam_shift.value()
                << " is too big compared to input (process) noise 3sig=" << max_expected_cam_shift;
    }
}

bool ValidateDirectoryEmptyOrExists(const std::string &value)
{
    const std::filesystem::path test_data_path = std::filesystem::absolute(value);
    
    // allow empty directory in case of virtual scenario
    return std::filesystem::is_directory(test_data_path);
}

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
DEFINE_int32(monoslam_sal_pnt_max_undetected_frames_count, 0, "");
DEFINE_double(monoslam_sal_pnt_negative_inv_rho_substitute, -1, "");
DEFINE_int32(monoslam_update_impl, 1, "");
DEFINE_int32(monoslam_max_new_blobs_in_first_frame, 7, "");
DEFINE_int32(monoslam_max_new_blobs_per_frame, 1, "");
DEFINE_double(monoslam_match_blob_prob, 1, "[0,1] portion of blobs which are matched with ones in the previous frame; 1=all matched, 0=none matched");
DEFINE_int32(monoslam_templ_width, 15, "width of template");
DEFINE_int32(monoslam_templ_min_search_rect_width, 7, "the min width of a rectangle when searching for tempplate in the next frame");
DEFINE_int32(monoslam_templ_min_search_rect_height, 7, "");
DEFINE_double(monoslam_templ_min_corr_coeff, -1, "");
DEFINE_double(monoslam_templ_center_detection_noise_std_pix, 0, "std of measurement noise(=sqrt(R), 0=no noise");
DEFINE_double(monoslam_templ_closest_templ_min_dist_pix, 0, "");
DEFINE_bool(monoslam_stop_on_sal_pnt_moved_too_far, false, "width of template");
DEFINE_bool(monoslam_fix_estim_vars_covar_symmetry, true, "");
DEFINE_bool(monoslam_debug_estim_vars_cov, false, "");
DEFINE_bool(monoslam_debug_predicted_vars_cov, false, "");
DEFINE_int32(monoslam_debug_max_sal_pnt_count, -1, "[default=-1(none)] number of salient points won't be greater than this value");
DEFINE_bool(monoslam_sal_pnt_perfect_init_inv_dist, false, "");
DEFINE_int32(monoslam_set_estim_state_covar_to_gt_impl, 2, "1=ignore correlations, 2=set correlations as if 'AddNewSalientPoint' is called on each salient point");
DEFINE_double(monoslam_ellipsoid_cut_thr, 0.04, "probability cut threshold for uncertainty ellipsoid");

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

void ApplyParamsFromConfigFile(DavisonMonoSlam* mono_slam, ConfigReader* config_reader)
{
    auto& cr = *config_reader;
    auto& ms = *mono_slam;

    auto opt_set = [](std::optional<double> opt_f64, gsl::not_null<Scalar*> dst)
    {
        if (opt_f64.has_value())
            * dst = static_cast<Scalar>(opt_f64.value());
    };

    auto process_noise_std = cr.GetValue<double>("monoslam_process_noise_std");
    if (process_noise_std.has_value())
        ms.SetProcessNoiseStd(static_cast<Scalar>(process_noise_std.value()));

    opt_set(cr.GetValue<double>("monoslam_measurm_noise_std_pix"), &ms.measurm_noise_std_pix_);
    opt_set(cr.GetValue<double>("monoslam_sal_pnt_init_inv_dist"), &ms.sal_pnt_init_inv_dist_);
    opt_set(cr.GetValue<double>("monoslam_sal_pnt_init_inv_dist_std"), &ms.sal_pnt_init_inv_dist_std_);
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

    std::string scene_source = config_reader.GetValue<std::string>("scene_source").value_or("");
    if (scene_source.empty())
    {
        log_absent_mandatory_flag("scene_source");
        return 1;
    }

    DemoDataSource demo_data_source = DemoDataSource::kVirtualScene;
    if (scene_source == std::string(kImageSeqDirCStr))
        demo_data_source = DemoDataSource::kImageSeqDir;

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

    std::string scene_imageseq_dir;
    if (demo_data_source == DemoDataSource::kImageSeqDir)
    {
        scene_imageseq_dir = config_reader.GetValue<std::string>("scene_imageseq_dir").value_or("");

        // validate directory only if it is the source of images for demo
        if (!ValidateDirectoryEmptyOrExists(scene_imageseq_dir)) {
            LOG(ERROR) << "directory [" << scene_imageseq_dir << "] doesn't exist";
            return 2;
        }
    }

    //
    FragmentMap entire_map;
    std::vector<SE3Transform> gt_cam_orient_cfw; // ground truth camera orientation transforming into camera from world

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
        Eigen::Matrix<Scalar, 3, 1> viewer_up{ viewer_up_a[0], viewer_up_a[1], viewer_up_a[2] };

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
            auto max_deviation = FloatParam<Scalar>(&config_reader, "scenario_max_deviation").value_or(0.6f);
            int shots_per_period = config_reader.GetValue<int>("scenario_shots_per_period").value_or(160);
            int periods_count = config_reader.GetValue<int>("scenario_periods_count").value_or(100);
            bool const_view_dir = config_reader.GetValue<bool>("scenario_const_view_dir").value_or(false);
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

        std::vector<SE3Transform> gt_cam_orient_wfc;
        std::transform(gt_cam_orient_cfw.begin(), gt_cam_orient_cfw.end(), std::back_inserter(gt_cam_orient_wfc), [](auto& t) { return SE3Inv(t); });

        if (corrupt_cam_orient_with_noise)
        {
            std::normal_distribution<Scalar> cam_orient_noise_dis(0, world_noise_R_std);
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

        std::optional<Scalar> between_frames_max_cam_shift = GetMaxCamShift(gt_cam_orient_cfw);
        LOG(INFO) << "max_cam_shift=" << between_frames_max_cam_shift.value();
    }

    size_t frames_count = gt_cam_orient_cfw.size();
    LOG(INFO) << "frames_count=" << frames_count;

    auto camera_image_size = config_reader.GetSeq<int>("camera_image_size").value_or(std::vector<int>{0, 0});
    auto camera_princip_point = config_reader.GetSeq<double>("camera_princip_point").value_or(std::vector<double>{0, 0});
    auto camera_focal_length_pix = config_reader.GetSeq<double>("camera_focal_length_pix").value_or(std::vector<double>{0, 0});

    // focal_len_pix = focal_len_mm / pixel_size_mm
    std::array<Scalar, 2> foc_len_pix = {
        static_cast<Scalar>(camera_focal_length_pix[0]),
        static_cast<Scalar>(camera_focal_length_pix[1]) };

    // assume dy=PixelSizeMm[1]=some constant
    const Scalar pix_size_y = 0.001f;

    const Scalar focal_length_mm = foc_len_pix[1] * pix_size_y;
    Scalar pix_size_x = focal_length_mm / foc_len_pix[0];

    CameraIntrinsicParams cam_intrinsics;
    cam_intrinsics.image_size = { camera_image_size[0], camera_image_size[1] };
    cam_intrinsics.principal_point_pix = { static_cast<Scalar>(camera_princip_point[0]), static_cast<Scalar>(camera_princip_point[1]) };
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
    drawer.ellipse_cut_thr_ = static_cast<Scalar>(FLAGS_monoslam_ellipsoid_cut_thr);
    drawer.ui_swallow_exc_ = FLAGS_ui_swallow_exc;

    // the origin of a tracker (sometimes cam0)
    SE3Transform tracker_origin_from_world;
    if (demo_data_source == DemoDataSource::kVirtualScene)
    {
        // tracker coordinate system = cam0
        tracker_origin_from_world = gt_cam_orient_cfw[0];
    }
    else if (demo_data_source == DemoDataSource::kImageSeqDir)
    {
        // tracker coordinates system = world coordinate system
        tracker_origin_from_world.R.setIdentity();
        tracker_origin_from_world.T.setZero();
    }

    DavisonMonoSlam::DebugPathEnum debug_path = DavisonMonoSlam::DebugPathEnum::DebugNone;
    if (FLAGS_monoslam_debug_estim_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugEstimVarsCov;
    if (FLAGS_monoslam_debug_predicted_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugPredictedVarsCov;
    DavisonMonoSlam::SetDebugPath(debug_path);

    DavisonMonoSlam mono_slam{ };
    ApplyParamsFromConfigFile(&mono_slam, &config_reader);
    mono_slam.in_multi_threaded_mode_ = FLAGS_ctrl_multi_threaded_mode;
    mono_slam.between_frames_period_ = 1;
    mono_slam.cam_intrinsics_ = cam_intrinsics;
    mono_slam.cam_distort_params_ = cam_distort_params;
    mono_slam.force_xyz_sal_pnt_pos_diagonal_uncert_ = FLAGS_monoslam_force_xyz_sal_pnt_pos_diagonal_uncert;
    mono_slam.sal_pnt_templ_size_ = { FLAGS_monoslam_templ_width, FLAGS_monoslam_templ_width };
    if (FLAGS_monoslam_templ_closest_templ_min_dist_pix > 0)
        mono_slam.closest_sal_pnt_templ_min_dist_pix_ = static_cast<Scalar>(FLAGS_monoslam_templ_closest_templ_min_dist_pix);
    if (FLAGS_monoslam_sal_pnt_max_undetected_frames_count > 0)
        mono_slam.sal_pnt_max_undetected_frames_count_ = FLAGS_monoslam_sal_pnt_max_undetected_frames_count;
    if (FLAGS_monoslam_sal_pnt_negative_inv_rho_substitute >= 0)
        mono_slam.sal_pnt_negative_inv_rho_substitute_ = static_cast<Scalar>(FLAGS_monoslam_sal_pnt_negative_inv_rho_substitute);

    mono_slam.mono_slam_update_impl_ = FLAGS_monoslam_update_impl;
    mono_slam.fix_estim_vars_covar_symmetry_ = FLAGS_monoslam_fix_estim_vars_covar_symmetry;
    mono_slam.debug_ellipsoid_cut_thr_ = static_cast<Scalar>(FLAGS_monoslam_ellipsoid_cut_thr);
    if (FLAGS_monoslam_debug_max_sal_pnt_count != -1)
        mono_slam.debug_max_sal_pnt_coun_ = FLAGS_monoslam_debug_max_sal_pnt_count;
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
            SE3Transform c = gt_cam_orient_cfw[frame_ind];
            return c;
        };
        mono_slam.gt_cami_from_tracker_new_ = [&gt_cam_orient_cfw](SE3Transform tracker_from_world, size_t frame_ind) -> std::optional<SE3Transform>
        {
            if (frame_ind >= gt_cam_orient_cfw.size())
                return std::nullopt;
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
    }

    // perhaps these values should be just constants
    mono_slam.cam_pos_x_std_m_ = static_cast<Scalar>(FLAGS_monoslam_cam_pos_x_std_m);
    mono_slam.cam_pos_y_std_m_ = static_cast<Scalar>(FLAGS_monoslam_cam_pos_y_std_m);
    mono_slam.cam_pos_z_std_m_ = static_cast<Scalar>(FLAGS_monoslam_cam_pos_z_std_m);
    mono_slam.cam_orient_q_comp_std_ = static_cast<Scalar>(FLAGS_monoslam_cam_orient_q_comp_std);
    mono_slam.cam_vel_std_ = static_cast<Scalar>(FLAGS_monoslam_cam_vel_std);
    mono_slam.cam_ang_vel_std_ = static_cast<Scalar>(FLAGS_monoslam_cam_ang_vel_std);
    mono_slam.SetCameraStateCovarHelper();

    LOG(INFO) << "mono_slam_process_noise_std=" << mono_slam.process_noise_std_;
    LOG(INFO) << "mono_slam_measurm_noise_std_pix=" << mono_slam.measurm_noise_std_pix_;
    LOG(INFO) << "mono_slam_update_impl=" << FLAGS_monoslam_update_impl;
    LOG(INFO) << "mono_slam_sal_pnt_vars=" << DavisonMonoSlam::kSalientPointComps;
    LOG(INFO) << "mono_slam_templ_min_dist=" << mono_slam.ClosestSalientPointTemplateMinDistance();
    LOG(INFO) << "mono_slam_templ_center_detection_noise_std_pix=" << FLAGS_monoslam_templ_center_detection_noise_std_pix;
    LOG(INFO) << "mono_slam_sal_pnt_negative_inv_rho_substitute=" << mono_slam.sal_pnt_negative_inv_rho_substitute_.value_or(static_cast<Scalar>(-1));

    if (demo_data_source == DemoDataSource::kVirtualScene)
    {
        auto corners_matcher = std::make_unique<DemoCornersMatcher>(&mono_slam, gt_cam_orient_cfw, entire_map, cam_intrinsics.image_size);
        corners_matcher->SetTemplCenterDetectionNoiseStd(static_cast<float>(FLAGS_monoslam_templ_center_detection_noise_std_pix));
        corners_matcher->tracker_origin_from_world_ = tracker_origin_from_world;

        if (FLAGS_monoslam_max_new_blobs_in_first_frame > 0)
            corners_matcher->max_new_blobs_in_first_frame_ = FLAGS_monoslam_max_new_blobs_in_first_frame;
        if (FLAGS_monoslam_max_new_blobs_per_frame > 0)
            corners_matcher->max_new_blobs_per_frame_ = FLAGS_monoslam_max_new_blobs_per_frame;
        if (FLAGS_monoslam_match_blob_prob > 0)
            corners_matcher->match_blob_prob_ = (float)FLAGS_monoslam_match_blob_prob;

        mono_slam.SetCornersMatcher(std::move(corners_matcher));
    }
    else if (demo_data_source == DemoDataSource::kImageSeqDir)
    {
        auto corners_matcher = std::make_unique<ImageTemplCornersMatcher>(&mono_slam);
        corners_matcher->stop_on_sal_pnt_moved_too_far_ = FLAGS_monoslam_stop_on_sal_pnt_moved_too_far;
        corners_matcher->ellisoid_cut_thr_ = static_cast<Scalar>(FLAGS_monoslam_ellipsoid_cut_thr);
        corners_matcher->min_search_rect_size_ = suriko::Sizei{ FLAGS_monoslam_templ_min_search_rect_width, FLAGS_monoslam_templ_min_search_rect_height };
        if (FLAGS_monoslam_templ_min_corr_coeff > -1)
            corners_matcher->min_templ_corr_coeff_ = static_cast<Scalar>(FLAGS_monoslam_templ_min_corr_coeff);
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

    if (FLAGS_ctrl_collect_tracker_internals)
    {
        mono_slam.SetStatsLogger(std::make_unique<DavisonMonoSlamInternalsLogger>(&mono_slam));
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
    std::vector<SE3Transform> cam_orient_cfw_history; // the actual trajectory of the tracker

    UIThreadParams ui_params {};
    ui_params.wait_for_user_input_after_each_frame = FLAGS_ctrl_wait_after_each_frame;
    ui_params.mono_slam = &mono_slam;
    ui_params.tracker_origin_from_world = tracker_origin_from_world;
    ui_params.ellipsoid_cut_thr = static_cast<Scalar>(FLAGS_monoslam_ellipsoid_cut_thr);
    ui_params.cam_orient_cfw_history = &cam_orient_cfw_history;
    ui_params.get_observable_frame_ind_fun = [&observable_frame_ind]() { return observable_frame_ind; };
    ui_params.worker_chat = worker_chat;
    ui_params.ui_swallow_exc = FLAGS_ui_swallow_exc;
    ui_params.ui_tight_loop_relaxing_delay = std::chrono::milliseconds(FLAGS_ui_tight_loop_relaxing_delay_ms);
    if (demo_data_source == DemoDataSource::kVirtualScene)
    {
        ui_params.entire_map = &entire_map;
        ui_params.gt_cam_orient_cfw = &gt_cam_orient_cfw;
    }

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
            pangolin_gui->SetCameraBehindTracker();
        }
    }
#endif

    CheckDavisonMonoSlamConfigurationAndDump(mono_slam, gt_cam_orient_cfw);

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
#if defined(SRK_HAS_PANGOLIN)
            cam_orient_cfw_history.push_back(actual_cam_cfw);
#endif

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

                        // the world axes are drawn on the image to provide richer context about the structure of the scene
                        // (drawing just salient points would be vague)
                        const SE3Transform& rt_cfw = gt_cam_orient_cfw[frame_ind];
                        auto project_fun = [&rt_cfw, &mono_slam](const suriko::Point3& sal_pnt_world) -> Eigen::Matrix<suriko::Scalar, 3, 1>
                        {
                            suriko::Point3 pnt_cam = SE3Apply(rt_cfw, sal_pnt_world);
                            suriko::Point2f pnt_pix = mono_slam.ProjectCameraPoint(pnt_cam);
                            return Eigen::Matrix<suriko::Scalar, 3, 1>(pnt_pix[0], pnt_pix[1], 1);
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

                drawer.DrawScene(mono_slam, image_bgr, &camera_image_bgr);

                std::stringstream strbuf;
                strbuf << "f=" << frame_ind;
                cv::putText(camera_image_bgr, cv::String(strbuf.str()), cv::Point(10, (int)cam_intrinsics.image_size.height - 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));

                auto t2 = std::chrono::high_resolution_clock::now();
                frame_OpenCV_gui_time = t2 - t1;

                cv::imshow("front-camera", camera_image_bgr);
                cv::waitKey(1); // allow to refresh an opencv view
            }
#endif
#if defined(SRK_HAS_PANGOLIN)
            if (FLAGS_ctrl_multi_threaded_mode)
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
                auto stop_wait_on_key = [](int key)
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
