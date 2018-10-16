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
#define DEMO_DATA_SOURCE_TYPE kImageSeqDir

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

void GenerateCameraShotsAlongRectangularPath(const WorldBounds& wb, size_t steps_per_side_x, size_t steps_per_side_y,
    suriko::Point3 eye_offset, 
    suriko::Point3 center_offset,
    const Eigen::Matrix<Scalar, 3, 1>& up,
    std::vector<SE3Transform>* inverse_orient_cams)
{
    std::array<suriko::Point3,5> look_at_base_points = {
        suriko::Point3(wb.XMax, wb.YMin, wb.ZMin),
        suriko::Point3(wb.XMin, wb.YMin, wb.ZMin),
        suriko::Point3(wb.XMin, wb.YMax, wb.ZMin),
        suriko::Point3(wb.XMax, wb.YMax, wb.ZMin),
        suriko::Point3(wb.XMax, wb.YMin, wb.ZMin),
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


    Scalar xmid = (wb.XMin + wb.XMax) / 2;
    Scalar xlen = wb.XMax - wb.XMin;
    Scalar zlen = wb.ZMax - wb.ZMin;
    for (Scalar grid_x = wb.XMin; grid_x < wb.XMax + inclusive_gap; grid_x += cell_size[0])
    {
        for (Scalar grid_y = wb.YMin; grid_y < wb.YMax + inclusive_gap; grid_y += cell_size[1])
        {
            Scalar x = grid_x;
            Scalar y = grid_y;

            Scalar z_perc = std::cos((x - xmid) / xlen * M_PI);
            Scalar z = wb.ZMin + z_perc * zlen;

            // jit x and y so the points can be distinguished during movement
            if (corrupt_salient_points_with_noise)
            {
                x += (*x3D_noise_dis)(*gen);
                y += (*x3D_noise_dis)(*gen);
                z += (*x3D_noise_dis)(*gen);
            }

            SalientPointFragment& frag = entire_map->AddSalientPoint(Point3(x, y, z));
            frag.SyntheticVirtualPointId = next_virtual_point_id++;
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
    std::optional<Scalar> GTDepth; // ground truth depth to salient point in virtual environments
    size_t FragmentId; // the id of a fragment in entire map
};

class DemoCornersMatcher : public CornersMatcherBase
{
    const std::vector<SE3Transform>& gt_cam_orient_cfw_;
    FragmentMap& entire_map_;
    std::array<size_t, 2> img_size_;
    const DavisonMonoSlam* kalman_tracker_;
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
    DemoCornersMatcher(const DavisonMonoSlam* kalman_tracker, const std::vector<SE3Transform>& gt_cam_orient_cfw, FragmentMap& entire_map,
        const std::array<size_t, 2>& img_size)
        : kalman_tracker_(kalman_tracker),
        gt_cam_orient_cfw_(gt_cam_orient_cfw),
        entire_map_(entire_map),
        img_size_(img_size)
    {
        tracker_origin_from_world_.T.setZero();
        tracker_origin_from_world_.R.setIdentity();
    }

    void AnalyzeFrame(size_t frame_ind, const ImageFrame& image) override
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

            const Point3& salient_point_world = fragment.Coord.value();
            suriko::Point3 pnt_tracker = PosTrackerOriginFromWorld(gt_cam_orient_cfw_, salient_point_world, tracker_origin_from_world_);

            suriko::Point3 pnt_camera = SE3Apply(cami_from_tracker, pnt_tracker);
            suriko::Point2 pnt_pix = kalman_tracker_->ProjectCameraPoint(pnt_camera);
            Scalar pix_x = pnt_pix[0];
            Scalar pix_y = pnt_pix[1];
            bool hit_wnd =
                pix_x >= 0 && pix_x < (Scalar)img_size_[0] &&
                pix_y >= 0 && pix_y < (Scalar)img_size_[1];
            if (!hit_wnd)
                continue;

            DavisonMonoSlam::SalPntId sal_pnt_id {};
            if (fragment.UserObj != nullptr)
            {
                // got salient point which hits the current frame and had been seen before
                // auto sal_pnt_id = reinterpret_cast<DavisonMonoSlam::SalPntId>(fragment.UserObj);
                std::memcpy(&sal_pnt_id, &fragment.UserObj, sizeof(decltype(fragment.UserObj)));
                static_assert(sizeof fragment.UserObj >= sizeof sal_pnt_id);
            }

            BlobInfo blob_info;
            blob_info.Coord = pnt_pix;
            blob_info.SalPntIdInTracker = sal_pnt_id;
            blob_info.GTDepth = pnt_camera.Mat().norm();
            blob_info.FragmentId = frag_id;
            detected_blobs_.push_back(blob_info);
        }
    }

    void MatchSalientPoints(size_t frame_ind,
        const ImageFrame& image,
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
        const ImageFrame& image,
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
            if (blob_info.SalPntIdInTracker)
                continue;

            // the tracker is interested in this blob
            size_t blob_id = i;
            new_blob_ids->push_back(CornersMatcherBlobId {blob_id});
        }
    }

    void OnSalientPointIsAssignedToBlobId(SalPntId sal_pnt_id, CornersMatcherBlobId blob_id, const ImageFrame& image) override
    {
        size_t frag_id = detected_blobs_[blob_id.Ind].FragmentId;
        SalientPointFragment& frag = entire_map_.GetSalientPointNew(frag_id);
        
        static_assert(sizeof sal_pnt_id <= sizeof frag.UserObj, "SalPntId must fit into UserObject");
        std::memcpy(&frag.UserObj, &sal_pnt_id, sizeof(sal_pnt_id));
    }

    suriko::Point2 GetBlobCoord(CornersMatcherBlobId blob_id) override
    {
        return detected_blobs_[blob_id.Ind].Coord.Mat();
    }
    
    std::optional<Scalar> GetSalientPointGroundTruthDepth(CornersMatcherBlobId blob_id) override
    {
        return detected_blobs_[blob_id.Ind].GTDepth;
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
public:
    ImagePatchCornersMatcher(DavisonMonoSlam* kalman_tracker)
        :kalman_tracker_(kalman_tracker)
    {
        int nfeatures = 50;
        detector_ = cv::ORB::create(nfeatures);
    }

    void MatchSalientPoints(size_t frame_ind,
        const ImageFrame& image,
        const std::set<SalPntId>& tracking_sal_pnts,
        std::vector<std::pair<DavisonMonoSlam::SalPntId, CornersMatcherBlobId>>* matched_sal_pnts) override
    {
        int result_cols = image.gray.cols - kalman_tracker_->sal_pnt_patch_size_[0] + 1;
        int result_rows = image.gray.rows - kalman_tracker_->sal_pnt_patch_size_[1] + 1;
        cv::Mat result;
        result.create(result_rows, result_cols, CV_32FC1);

        static int method = -1;
        if (method == -1)
        {
            method = CV_TM_SQDIFF;
            method = CV_TM_SQDIFF_NORMED;
            method = CV_TM_CCORR;
            method = CV_TM_CCORR_NORMED; // works
            method = CV_TM_CCOEFF;
            method = CV_TM_CCOEFF_NORMED; // works
        }

        static float max_shift_per_frame = 30;
        static cv::Mat image_gray_with_match;

        for (auto sal_pnt_id : tracking_sal_pnts)
        {
            const SalPntInternal& sal_pnt = kalman_tracker_->GetSalPnt(sal_pnt_id);
            cv::matchTemplate(image.gray, sal_pnt.PatchTemplateInFirstFrame, result, method);
            cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1);

            double minVal;
            double maxVal;
            cv::Point minLoc;
            cv::Point maxLoc;
            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

            auto match_loc = maxLoc;

            static bool debug_ui = false;
            if (debug_ui)
            {
                image.gray.copyTo(image_gray_with_match);
                cv::Rect rect{ match_loc.x, match_loc.y, kalman_tracker_->sal_pnt_patch_size_[0], kalman_tracker_->sal_pnt_patch_size_[1] };
                cv::rectangle(image_gray_with_match, rect, cv::Scalar::all(0));
            }

            int rad_x_int = kalman_tracker_->sal_pnt_patch_size_[0] / 2;
            int rad_y_int = kalman_tracker_->sal_pnt_patch_size_[1] / 2;

            auto center = cv::Point{ maxLoc.x + rad_x_int, maxLoc.y + rad_y_int };
            float diffC = (sal_pnt.PixelCoordInLatestFrame - Eigen::Matrix<Scalar, 2, 1>{center.x, center.y}).norm();
            float diffTL = (sal_pnt.PatchTemplateTopLeftInLatestFrame - Eigen::Matrix<int, 2, 1>{maxLoc.x, maxLoc.y}).norm();
            if (diffTL > max_shift_per_frame)
            {
                if (stop_on_sal_pnt_moved_too_far_)
                    SRK_ASSERT(false) << "sal pnt moved to far away";
                else
                    continue;
            }

            size_t blob_ind = new_keypoints_.size();
            cv::KeyPoint kp{};
            kp.pt = center;
            new_keypoints_.push_back(kp);

            matched_sal_pnts->push_back(std::make_pair(sal_pnt_id, CornersMatcherBlobId{ blob_ind }));
        }
    }

    void RecruitNewSalientPoints(size_t frame_ind,
        const ImageFrame& image,
        const std::set<SalPntId>& tracking_sal_pnts,
        const std::vector<std::pair<DavisonMonoSlam::SalPntId, CornersMatcherBlobId>>& matched_sal_pnts,
        std::vector<CornersMatcherBlobId>* new_blob_ids) override
    {
        std::vector<cv::KeyPoint> keypoints;
        detector_->detect(image.gray, keypoints);

        cv::Mat descr_per_row;
        detector_->compute(image.gray, keypoints, descr_per_row);

        std::vector<cv::KeyPoint> sparse_keypoints;
        const float rad_diag = std::sqrt(suriko::Sqr(kalman_tracker_->sal_pnt_patch_size_[0]) + suriko::Sqr(kalman_tracker_->sal_pnt_patch_size_[1])) / 2;
        FilterOutClosest(keypoints, rad_diag, &sparse_keypoints);

        cv::Mat sparse_img;
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
                    Eigen::Matrix<Scalar, kPixPosComps, 1> exist_pix = kalman_tracker_->GetSalPntPixelCoord(sal_pnt_id);
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

        cv::Mat new_img;
        cv::drawKeypoints(image.gray, new_keypoints_, new_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

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

    ImageFrame GetBlobPatchTemplate(CornersMatcherBlobId blob_id, const ImageFrame& image) override
    {
        const cv::KeyPoint& kp = new_keypoints_[blob_id.Ind];

        int rad_x_int = kalman_tracker_->sal_pnt_patch_size_[0] / 2;
        int rad_y_int = kalman_tracker_->sal_pnt_patch_size_[1] / 2;

        int center_x = (int)kp.pt.x;
        int center_y = (int)kp.pt.y;
        cv::Rect patch_bounds{ center_x - rad_x_int, center_y - rad_y_int, kalman_tracker_->sal_pnt_patch_size_[0], kalman_tracker_->sal_pnt_patch_size_[1] };

        cv::Mat patch_gray;
        image.gray(patch_bounds).copyTo(patch_gray);
        SRK_ASSERT(patch_gray.rows == kalman_tracker_->sal_pnt_patch_size_[1]);
        SRK_ASSERT(patch_gray.cols == kalman_tracker_->sal_pnt_patch_size_[0]);

        ImageFrame patch_template{};
        patch_template.gray = patch_gray;

        if (kSurikoDebug)
        {
            cv::Mat patch;
            image.rgb_debug(patch_bounds).copyTo(patch);
            patch_template.rgb_debug = patch;
        }
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

    fs << "FramesCount" << static_cast<int>(hist.StateSamples.size());
    fs << "AvgFrameProcessingDur" << static_cast<float>(hist.AvgFrameProcessingDur.count()); // seconds
    fs << "Frames" <<"[";

    for (const auto& item : hist.StateSamples)
    {
        fs << "{";

        cv::write(fs, "CurReprojErr", item.CurReprojErr);
        cv::write(fs, "EstimatedSalPnts", static_cast<int>(item.EstimatedSalPnts));
        cv::write(fs, "NewSalPnts", static_cast<int>(item.NewSalPnts));
        cv::write(fs, "CommonSalPnts", static_cast<int>(item.CommonSalPnts));
        cv::write(fs, "DeletedSalPnts", static_cast<int>(item.DeletedSalPnts));
        cv::write(fs, "FrameProcessingDur", item.FrameProcessingDur.count()); // seconds

        Eigen::Map<const Eigen::Matrix<Scalar, 9, 1>> cam_pos_uncert(item.CamPosUncert.data());
        fs << "CamPosUnc_s" <<"[:";
        WriteMatElements(fs, cam_pos_uncert);
        fs << "]";

        if (item.SalPntsUncertMedian.has_value())
        {
            fs << "SalPntUncMedian_s" << "[:";
            WriteMatElements(fs, item.SalPntsUncertMedian.value());
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
DEFINE_int32(viewer_steps_per_side_x, 20, "number of viewer's steps at each side of the rectangle");
DEFINE_int32(viewer_steps_per_side_y, 10, "number of viewer's steps at each side of the rectangle");
DEFINE_double(kalman_estim_var_init_std, 0.001, "");
DEFINE_double(kalman_input_noise_std, 0.08, "");
DEFINE_double(kalman_sal_pnt_init_inv_dist, 1, "");
DEFINE_double(kalman_sal_pnt_init_inv_dist_std, 1, "");
DEFINE_double(kalman_measurm_noise_std, 1, "");
DEFINE_int32(kalman_update_impl, 1, "");
DEFINE_bool(kalman_support_inf_sal_pnts, true, "true to handle salient points in infinity, by multiplying by rho when projecting salient points");
DEFINE_double(kalman_max_new_blobs_in_first_frame, 7, "");
DEFINE_double(kalman_max_new_blobs_per_frame, 1, "");
DEFINE_double(kalman_match_blob_prob, 1, "[0,1] portion of blobs which are matched with ones in the previous frame; 1=all matched, 0=none matched");
DEFINE_int32(kalman_templ_width, 15, "width of patch template");
DEFINE_bool(kalman_stop_on_sal_pnt_moved_too_far, false, "width of patch template");
DEFINE_bool(kalman_fix_estim_vars_covar_symmetry, true, "");
DEFINE_bool(kalman_debug_estim_vars_cov, false, "");
DEFINE_bool(kalman_debug_predicted_vars_cov, false, "");
DEFINE_bool(kalman_fake_localization, false, "");
DEFINE_bool(kalman_fake_sal_pnt_init_inv_dist, false, "");
DEFINE_double(ui_ellipsoid_cut_thr, 0.04, "probability cut threshold for uncertainty ellipsoid");
DEFINE_bool(ui_swallow_exc, true, "true to ignore (swallow) exceptions in UI");
DEFINE_int32(ui_loop_prolong_period_ms, 3000, "");
DEFINE_int32(ui_tight_loop_relaxing_delay_ms, 100, "");
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

int DavisonMonoSlamDemo(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);

    //DemoDataSource demo_data_source;
    //if (FLAGS_scene_source == std::string_view(kImageSeqDirCStr))
    //    demo_data_source = DemoDataSource::kImageSeqDir;
    //else
    //{
    //    if (FLAGS_scene_source != std::string_view(kVirtualCStr))
    //        LOG(INFO) << "WARN: Unknown demo data source '" << FLAGS_scene_source << "', resetting to '" << kVirtualCStr <<"'";
    //    demo_data_source = DemoDataSource::kVirtualScene;
    //}
    //LOG(INFO) << "demo_data_source=" << (demo_data_source == DemoDataSource::kImageSeqDir ? kImageSeqDirCStr: kVirtualCStr);

#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    LOG(INFO) << "world_noise_x3D_std=" << FLAGS_world_noise_x3D_std;
    LOG(INFO) << "world_noise_R_std=" << FLAGS_world_noise_R_std;

    //
    bool corrupt_salient_points_with_noise = FLAGS_world_noise_x3D_std > 0;
    bool corrupt_cam_orient_with_noise = FLAGS_world_noise_R_std > 0;
    std::vector<SE3Transform> gt_cam_orient_cfw; // ground truth camera orientation transforming into camera from world

    WorldBounds wb{};
    wb.XMin = FLAGS_world_xmin;
    wb.XMax = FLAGS_world_xmax;
    wb.YMin = FLAGS_world_ymin;
    wb.YMax = FLAGS_world_ymax;
    wb.ZMin = FLAGS_world_zmin;
    wb.ZMax = FLAGS_world_zmax;
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
    GenerateCameraShotsAlongRectangularPath(wb, FLAGS_viewer_steps_per_side_x, FLAGS_viewer_steps_per_side_y,
        viewer_eye_offset, viewer_center_offset, up, &gt_cam_orient_cfw);

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

    std::array<size_t, 2> img_size = { FLAGS_camera_image_width, FLAGS_camera_image_height };
    LOG(INFO) << "img_size=[" << img_size[0] << "," << img_size[1] << "] pix";

    // TODO: change necessary params so that the primary input is a principal point in pixels (not a focal length in mm or pixel size in mm); remove focal_len_mm / pixel_size_mm from DavisonMonoSlam algo
    // focal_len_pix = focal_len_mm / pixel_size_mm
    // assume dy=PixelSizeMm[1]=0.001mm

    std::array<Scalar, 2> foc_len_pix = { FLAGS_camera_focal_length_pix_x, FLAGS_camera_focal_length_pix_y };
    float pix_size_y = 0.001f;
    float focal_length_mm = foc_len_pix[1] * pix_size_y;
    float pix_size_x = focal_length_mm / foc_len_pix[0];

    CameraIntrinsicParams cam_intrinsics;
    cam_intrinsics.PrincipalPointPixels[0] = FLAGS_camera_princip_point_x;
    cam_intrinsics.PrincipalPointPixels[1] = FLAGS_camera_princip_point_y;
    cam_intrinsics.FocalLengthMm = focal_length_mm;
    cam_intrinsics.PixelSizeMm = std::array<Scalar, 2> {pix_size_x, pix_size_y};
    std::array<Scalar, 2> focal_length_pixels = cam_intrinsics.GetFocalLengthPixels();
    LOG(INFO) << "foc_len=" << cam_intrinsics.FocalLengthMm << " mm"
        << " PixelSize[dx,dy]=[" << cam_intrinsics.PixelSizeMm[0] << "," << cam_intrinsics.PixelSizeMm[1] << "] mm"
        << " PrincipPoint[Cx,Cy]=[" << cam_intrinsics.PrincipalPointPixels[0] << "," << cam_intrinsics.PrincipalPointPixels[1] << "] pix";
    LOG(INFO) << "foc_len[alphax,alphay]=[" << focal_length_pixels[0] << "," << focal_length_pixels[1] << "] pix";

    RadialDistortionParams cam_distort_params;
    cam_distort_params.K1 = 0;
    cam_distort_params.K2 = 0;

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

    DavisonMonoSlam tracker{ };
    tracker.in_multi_threaded_mode_ = FLAGS_ctrl_multi_threaded_mode;
    tracker.between_frames_period_ = 1;
    tracker.cam_intrinsics_ = cam_intrinsics;
    tracker.cam_distort_params_ = cam_distort_params;
    tracker.sal_pnt_init_inv_dist_ = FLAGS_kalman_sal_pnt_init_inv_dist;
    tracker.sal_pnt_init_inv_dist_std_ = FLAGS_kalman_sal_pnt_init_inv_dist_std;
    tracker.SetInputNoiseStd(FLAGS_kalman_input_noise_std);
    tracker.measurm_noise_std_ = FLAGS_kalman_measurm_noise_std;
    tracker.sal_pnt_patch_size_ = { FLAGS_kalman_templ_width, FLAGS_kalman_templ_width };
    tracker.SetCamera(SE3Transform::NoTransform(), FLAGS_kalman_estim_var_init_std);

    tracker.kalman_update_impl_ = FLAGS_kalman_update_impl;
    tracker.support_inf_sal_pnts_ = FLAGS_kalman_support_inf_sal_pnts;
    tracker.fix_estim_vars_covar_symmetry_ = FLAGS_kalman_fix_estim_vars_covar_symmetry;
    tracker.debug_ellipsoid_cut_thr_ = FLAGS_ui_ellipsoid_cut_thr;
#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    tracker.fake_localization_ = FLAGS_kalman_fake_localization;
    tracker.fake_sal_pnt_initial_inv_dist_ = FLAGS_kalman_fake_sal_pnt_init_inv_dist;
    tracker.gt_cami_from_tracker_fun_ = [&gt_cam_orient_cfw, tracker_origin_from_world](size_t frame_ind) -> SE3Transform
    {
        SE3Transform c = CurCamFromTrackerOrigin(gt_cam_orient_cfw, frame_ind, tracker_origin_from_world);
        return c;
    };    
#endif

    tracker.PredictEstimVarsHelper();
    LOG(INFO) << "kalman_update_impl=" << FLAGS_kalman_update_impl;

#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    {
        auto corners_matcher = std::make_unique<DemoCornersMatcher>(&tracker, gt_cam_orient_cfw, entire_map, img_size);
        corners_matcher->tracker_origin_from_world_ = tracker_origin_from_world;

        if (FLAGS_kalman_max_new_blobs_in_first_frame > 0)
            corners_matcher->max_new_blobs_in_first_frame_ = FLAGS_kalman_max_new_blobs_in_first_frame;
        if (FLAGS_kalman_max_new_blobs_per_frame > 0)
            corners_matcher->max_new_blobs_per_frame_ = FLAGS_kalman_max_new_blobs_per_frame;
        if (FLAGS_kalman_match_blob_prob > 0)
            corners_matcher->match_blob_prob_ = FLAGS_kalman_match_blob_prob;

        tracker.SetCornersMatcher(std::move(corners_matcher));
    }
#elif DEMO_DATA_SOURCE_TYPE == kImageSeqDir
    {
        auto corners_matcher = std::make_unique<ImagePatchCornersMatcher>(&tracker);
        corners_matcher->stop_on_sal_pnt_moved_too_far_ = FLAGS_kalman_stop_on_sal_pnt_moved_too_far;
        tracker.SetCornersMatcher(std::move(corners_matcher));
    }
#endif

    if (FLAGS_ctrl_collect_tracker_internals)
    {
        tracker.SetStatsLogger(std::make_unique<DavisonMonoSlamInternalsLogger>(&tracker));
    }

#if defined(SRK_HAS_OPENCV)
    cv::Mat camera_image_bgr = cv::Mat::zeros((int)img_size[1], (int)img_size[0], CV_8UC3);
#endif
#if defined(SRK_HAS_PANGOLIN)
    // across threads shared data
    auto worker_chat = std::make_shared<WorkerChatSharedState>();
    ptrdiff_t observable_frame_ind = -1; // this is visualized by UI, it is one frame less than current frame
    std::vector<SE3Transform> cam_orient_cfw_history; // the actual trajectory of the tracker

    UIThreadParams ui_params {};
    ui_params.wait_for_user_input_after_each_frame = FLAGS_ctrl_wait_after_each_frame;
    ui_params.kalman_slam = &tracker;
    ui_params.tracker_origin_from_world = tracker_origin_from_world;
    ui_params.ellipsoid_cut_thr = FLAGS_ui_ellipsoid_cut_thr;
    ui_params.cam_orient_cfw_history = &cam_orient_cfw_history;
    ui_params.get_observable_frame_ind_fun = [&observable_frame_ind]() { return observable_frame_ind; };
    ui_params.worker_chat = worker_chat;
    ui_params.ui_swallow_exc = FLAGS_ui_swallow_exc;
    ui_params.ui_tight_loop_relaxing_delay = std::chrono::milliseconds(FLAGS_ui_tight_loop_relaxing_delay_ms);
#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
    ui_params.entire_map = &entire_map;
    ui_params.gt_cam_orient_cfw = &gt_cam_orient_cfw;
#endif

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
            pangolin_gui->InitUI();
            int back_dist = 5;
            pangolin_gui->SetCameraBehindTrackerOnce(tracker_origin_from_world, back_dist);
        }
    }
#endif

    std::optional<std::chrono::duration<double>> frame_process_time; // time it took to process current frame by tracker

    ImageFrame image;
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
        cv::Mat image_gray;
        cv::cvtColor(image_bgr, image_gray, CV_BGR2GRAY);
        image.gray = image_gray;
        if (kSurikoDebug)
        {
            cv::Mat image_rgb;
            cv::cvtColor(image_bgr, image_rgb, CV_BGR2RGB);
            image.rgb_debug = image_rgb;
        }
#endif
        auto& corners_matcher = tracker.CornersMatcher();
        corners_matcher.AnalyzeFrame(frame_ind, image);

        // process the frame
        if (!FLAGS_ctrl_debug_skim_over)
        {
            auto t1 = std::chrono::high_resolution_clock::now();

            tracker.ProcessFrame(frame_ind, image);

            auto t2 = std::chrono::high_resolution_clock::now();
            frame_process_time = t2 - t1;

            CameraPosState cam_state;
            tracker.GetCameraPredictedPosState(&cam_state);
            SE3Transform actual_cam_wfc(RotMat(cam_state.OrientationWfc), cam_state.PosW);
            SE3Transform actual_cam_cfw = SE3Inv(actual_cam_wfc);
#if defined(SRK_HAS_PANGOLIN)
            cam_orient_cfw_history.push_back(actual_cam_cfw);
#endif

#if defined(SRK_HAS_PANGOLIN)
            observable_frame_ind = frame_ind;
#endif
        }

        VLOG(4) << "f=" << frame_ind
            << " fps=" << (frame_process_time.has_value() ? 1 / frame_process_time.value().count() : 0.0f)
            << " tracks_count=" << tracker.SalientPointsCount();

#if defined(SRK_HAS_OPENCV)
        if (FLAGS_ctrl_visualize_during_processing)
        {
#if DEMO_DATA_SOURCE_TYPE == kVirtualScene
            camera_image_rgb.setTo(0);
            const SE3Transform& rt_cfw = gt_cam_orient_cfw[frame_ind];
            auto project_fun = [&rt_cfw, &tracker](const suriko::Point3& sal_pnt_world) -> Eigen::Matrix<suriko::Scalar, 3, 1>
            {
                suriko::Point3 pnt_cam = SE3Apply(rt_cfw, sal_pnt_world);
                suriko::Point2 pnt_pix = tracker.ProjectCameraPoint(pnt_cam);
                return Eigen::Matrix<suriko::Scalar, 3, 1>(pnt_pix[0], pnt_pix[1], 1);
            };
            constexpr Scalar f0 = 1;
            suriko_demos::Draw2DProjectedAxes(f0, project_fun, &camera_image_rgb);

            //
            auto a_corners_matcher = dynamic_cast<DemoCornersMatcher*>(&corners_matcher);
            if (a_corners_matcher != nullptr)
            {
                for (const BlobInfo& blob_info : a_corners_matcher->DetectedBlobs())
                {
                    Scalar pix_x = blob_info.Coord[0];
                    Scalar pix_y = blob_info.Coord[1];
                    camera_image_rgb.at<cv::Vec3b>((int)pix_y, (int)pix_x) = cv::Vec3b(0xFF, 0xFF, 0xFF);
                }
            }

            std::stringstream strbuf;
            strbuf << "f=" << frame_ind;
            cv::putText(camera_image_rgb, cv::String(strbuf.str()), cv::Point(10, (int)img_size[1] - 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
#elif DEMO_DATA_SOURCE_TYPE == kImageSeqDir
            image_bgr.copyTo(camera_image_bgr);
            auto [w,h] = tracker.sal_pnt_patch_size_;
            for (SalPntId sal_pnt_id : tracker.GetSalientPoints())
            {
                const SalPntInternal& sal_pnt = tracker.GetSalPnt(sal_pnt_id);
                cv::rectangle(camera_image_bgr, 
                    cv::Rect{ sal_pnt.PatchTemplateTopLeftInLatestFrame[0], sal_pnt.PatchTemplateTopLeftInLatestFrame[1], w, h },
                    cv::Scalar::all(0));
            }
#endif
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

        // update UI
        if (FLAGS_ctrl_visualize_during_processing)
        {
            auto key_of_interest = [](int key)
            {
                return key == 's' || key == 'f' || key == pangolin::PANGO_KEY_ESCAPE;
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
                    key = pangolin_gui->WaitKey(key_of_interest);
                }

                if (key == pangolin::PANGO_KEY_ESCAPE)
                    break;
                
                const bool suppress_observations = (key == 's');

                auto a_corners_matcher = dynamic_cast<DemoCornersMatcher*>(&tracker.CornersMatcher());
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
                    std::optional<int> key = pangolin_gui->RenderFrameAndProlongUILoopOnUserInput(break_on);
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
    const DavisonMonoSlamTrackerInternalsHist& internal_stats = tracker.StatsLogger()->BuildStats();

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