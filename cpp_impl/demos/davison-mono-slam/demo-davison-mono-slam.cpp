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
//#include <filesystem>
//#include <experimental/filesystem>
#include <boost/filesystem.hpp>
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
#include "../stat-helpers.h"
#include "../visualize-helpers.h"
#include "suriko/quat.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/imgproc.hpp> // cv::circle
#include <opencv2/highgui.hpp> // cv::imshow
#endif

#include "demo-davison-mono-slam-ui.h"

namespace suriko_demos_davison_mono_slam
{
using namespace std;
using namespace boost::filesystem;
using namespace suriko;
using namespace suriko::internals;
using namespace suriko::virt_world;

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
    }

    void AnalyzeFrame(size_t frame_ind) override
    {
        detected_blobs_.clear();

        if (suppress_observations_)
            return;

        // determine current camerra's orientation using the ground truth
        const SE3Transform& rt_cfw = gt_cam_orient_cfw_[frame_ind];

        std::vector<size_t> salient_points_ids;
        entire_map_.GetSalientPointsIds(&salient_points_ids);

        for (size_t frag_id : salient_points_ids)
        {
            const SalientPointFragment& fragment = entire_map_.GetSalientPointNew(frag_id);

            const Point3& salient_point = fragment.Coord.value();
            suriko::Point3 pnt_camera = SE3Apply(rt_cfw, salient_point);
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
        const std::set<SalPntId>& matched_sal_pnts,
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
            if (auto it = matched_sal_pnts.find(sal_pnt_id); it != matched_sal_pnts.end())
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

    void OnSalientPointIsAssignedToBlobId(SalPntId sal_pnt_id, CornersMatcherBlobId blob_id) override
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
DEFINE_double(kalman_max_new_blobs_in_first_frame, 7, "");
DEFINE_double(kalman_max_new_blobs_per_frame, 1, "");
DEFINE_double(kalman_match_blob_prob, 1, "[0,1] portion of blobs which are matched with ones in the previous frame; 1=all matched, 0=none matched");
DEFINE_bool(kalman_fix_estim_vars_covar_symmetry, true, "");
DEFINE_bool(kalman_debug_estim_vars_cov, false, "");
DEFINE_bool(kalman_debug_predicted_vars_cov, false, "");
DEFINE_bool(kalman_fake_localization, false, "");
DEFINE_bool(kalman_fake_sal_pnt_init_inv_dist, false, "");
DEFINE_double(ui_ellipsoid_cut_thr, 0.04, "probability cut threshold for uncertainty ellipsoid");
DEFINE_bool(ui_show_data_logger, true, "Whether to show timeline with uncertainty statistics.");
DEFINE_bool(ui_swallow_exc, true, "true to ignore (swallow) exceptions in UI");
DEFINE_bool(ctrl_wait_after_each_frame, false, "true to wait for keypress after each iteration");
DEFINE_bool(ctrl_debug_skim_over, false, "overview the synthetic world without reconstruction");
DEFINE_bool(ctrl_visualize_during_processing, true, "");
DEFINE_bool(ctrl_visualize_after_processing, true, "");
DEFINE_bool(ctrl_collect_tracker_internals, false, "");

int DavisonMonoSlamDemo(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);

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

    constexpr Scalar inclusive_gap = 1e-8; // small value to make iteration inclusive

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    gen.seed(1234);

    std::unique_ptr<std::normal_distribution<Scalar>> x3D_noise_dis;
    if (corrupt_salient_points_with_noise)
        x3D_noise_dis = std::make_unique<std::normal_distribution<Scalar>>(0, FLAGS_world_noise_x3D_std);

    size_t next_virtual_point_id = 6000'000 + 1;
    FragmentMap entire_map;
    entire_map.SetFragmentIdOffsetInternal(1000'000);
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
                x += (*x3D_noise_dis)(gen);
                y += (*x3D_noise_dis)(gen);
                z += (*x3D_noise_dis)(gen);
            }

            SalientPointFragment& frag = entire_map.AddSalientPoint(Point3(x, y, z));
            frag.SyntheticVirtualPointId = next_virtual_point_id++;
        }
    }

    LOG(INFO) << "points_count=" << entire_map.SalientPointsCount();

    // Numerical stability scaler, chosen so that x_pix / f0 and y_pix / f0 is close to 1
    std::array<size_t, 2> img_size = { 800, 600 };
    LOG(INFO) << "img_size=[" << img_size[0] << "," << img_size[1] << "] pix";

    CameraIntrinsicParams cam_intrinsics;
    cam_intrinsics.FocalLengthMm = 2.5;
    cam_intrinsics.PixelSizeMm = std::array<Scalar,2> {0.00284, 0.00378};
    cam_intrinsics.PrincipalPointPixels[0] = img_size[0] / 2.0;
    cam_intrinsics.PrincipalPointPixels[1] = img_size[1] / 2.0;
    std::array<Scalar, 2> focal_length_pixels = cam_intrinsics.GetFocalLengthPixels();
    LOG(INFO) << "foc_len=" << cam_intrinsics.FocalLengthMm << " mm"
        << " PixelSize[dx,dy]=[" << cam_intrinsics.PixelSizeMm[0] << "," << cam_intrinsics.PixelSizeMm[1] << "] mm"
        << " PrincipPoint[Cx,Cy]=[" << cam_intrinsics.PrincipalPointPixels[0] << "," << cam_intrinsics.PrincipalPointPixels[1] << "] pix";
    LOG(INFO) << "foc_len[alphax,alphay]=[" << focal_length_pixels[0] << "," << focal_length_pixels[1] << "] pix";

    RadialDistortionParams cam_distort_params;
    cam_distort_params.K1 = 0;
    cam_distort_params.K2 = 0;

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

    DavisonMonoSlam::DebugPathEnum debug_path = DavisonMonoSlam::DebugPathEnum::DebugNone;
    if (FLAGS_kalman_debug_estim_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugEstimVarsCov;
    if (FLAGS_kalman_debug_predicted_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugPredictedVarsCov;
    DavisonMonoSlam::SetDebugPath(debug_path);

    DavisonMonoSlam tracker_tmp1;

    DavisonMonoSlam tracker{};
    tracker.between_frames_period_ = 1;
    tracker.cam_intrinsics_ = cam_intrinsics;
    tracker.cam_distort_params_ = cam_distort_params;
    tracker.sal_pnt_init_inv_dist_ = FLAGS_kalman_sal_pnt_init_inv_dist;
    tracker.sal_pnt_init_inv_dist_std_ = FLAGS_kalman_sal_pnt_init_inv_dist_std;
    tracker.SetCamera(gt_cam_orient_cfw[0], FLAGS_kalman_estim_var_init_std);
    tracker.SetInputNoiseStd(FLAGS_kalman_input_noise_std);
    tracker.measurm_noise_std_ = FLAGS_kalman_measurm_noise_std;

    tracker.kalman_update_impl_ = FLAGS_kalman_update_impl;
    tracker.fix_estim_vars_covar_symmetry_ = FLAGS_kalman_fix_estim_vars_covar_symmetry;
    tracker.debug_ellipsoid_cut_thr_ = FLAGS_ui_ellipsoid_cut_thr;
    tracker.fake_localization_ = FLAGS_kalman_fake_localization;
    tracker.fake_sal_pnt_initial_inv_dist_ = FLAGS_kalman_fake_sal_pnt_init_inv_dist;
    tracker.gt_cam_orient_world_to_f_ = [&gt_cam_orient_cfw](size_t f) -> SE3Transform
    {
        SE3Transform c = gt_cam_orient_cfw[f];
        return c;
    };    
    tracker.gt_cam_orient_f1f2_ = [&gt_cam_orient_cfw](size_t f0, size_t f1) -> SE3Transform
    {
        SE3Transform c0 = gt_cam_orient_cfw[f0];
        SE3Transform c1 = gt_cam_orient_cfw[f1];
        SE3Transform c1_from_c0 = SE3AFromB(c1, c0);
        return c1_from_c0;
    };
    tracker.gt_salient_point_by_virtual_point_id_fun_ = [&entire_map](size_t synthetic_virtual_point_id) -> suriko::Point3
    {
        const SalientPointFragment* sal_pnt = nullptr;
        if (entire_map.GetSalientPointByVirtualPointIdInternal(synthetic_virtual_point_id, &sal_pnt) && sal_pnt->Coord.has_value())
        {
            const suriko::Point3& pnt_world = sal_pnt->Coord.value();
            return pnt_world;
        }
        AssertFalse();
    };

    tracker.PredictEstimVarsHelper();
    LOG(INFO) << "kalman_update_impl=" << FLAGS_kalman_update_impl;
    
    {
        auto corners_matcher = std::make_unique<DemoCornersMatcher>(&tracker, gt_cam_orient_cfw, entire_map, img_size);
        corners_matcher->max_new_blobs_in_first_frame_ = FLAGS_kalman_max_new_blobs_in_first_frame;
        corners_matcher->max_new_blobs_per_frame_ = FLAGS_kalman_max_new_blobs_per_frame;
        corners_matcher->match_blob_prob_ = FLAGS_kalman_match_blob_prob;
        tracker.SetCornersMatcher(std::move(corners_matcher));
    }

    if (FLAGS_ctrl_collect_tracker_internals)
    {
        tracker.SetStatsLogger(std::make_unique<DavisonMonoSlamInternalsLogger>(&tracker));
    }

#if defined(SRK_HAS_OPENCV)
    cv::Mat camera_image_rgb = cv::Mat::zeros((int)img_size[1], (int)img_size[0], CV_8UC3);
#endif
#if defined(SRK_HAS_PANGOLIN)
    // across threads shared data
    auto worker_chat = std::make_shared<WorkerThreadChat>();
    ptrdiff_t observable_frame_ind = -1; // this is visualized by UI, it is one frame less than current frame
    std::vector<SE3Transform> cam_orient_cfw_history; // the actual trajectory of the tracker
    std::deque<PlotterDataLogItem> plotter_data_log_item_history;

    UIThreadParams ui_params {};
    ui_params.WaitForUserInputAfterEachFrame = FLAGS_ctrl_wait_after_each_frame;
    ui_params.kalman_slam = &tracker;
    ui_params.ellipsoid_cut_thr = FLAGS_ui_ellipsoid_cut_thr;
    ui_params.gt_cam_orient_cfw = &gt_cam_orient_cfw;
    ui_params.cam_orient_cfw_history = &cam_orient_cfw_history;
    ui_params.plotter_data_log_exchange_buf = &plotter_data_log_item_history;
    ui_params.get_observable_frame_ind_fun = [&observable_frame_ind]() { return observable_frame_ind; };
    ui_params.entire_map = &entire_map;
    ui_params.worker_chat = worker_chat;
    ui_params.show_data_logger = FLAGS_ui_show_data_logger;
    ui_params.ui_swallow_exc = FLAGS_ui_swallow_exc;
    std::thread ui_thread;
    if (FLAGS_ctrl_visualize_during_processing)
        ui_thread = std::thread(SceneVisualizationThread, ui_params);
#endif

    std::optional<std::chrono::duration<double>> frame_process_time;
    CornerTrackRepository track_rep;

    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        // orient camera
        const SE3Transform& cam_cfw = gt_cam_orient_cfw[frame_ind];
        SE3Transform cam_wfc = SE3Inv(cam_cfw);

        auto& corners_matcher = dynamic_cast<DemoCornersMatcher&>(tracker.CornersMatcher());
        corners_matcher.AnalyzeFrame(frame_ind);

        // process the frame
        if (!FLAGS_ctrl_debug_skim_over)
        {
            auto t1 = std::chrono::high_resolution_clock::now();

            tracker.ProcessFrame(frame_ind);

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
            constexpr auto kCam = DavisonMonoSlam::kCamStateComps;
            Eigen::Matrix<Scalar, kCam, kCam> cam_state_covar;
            tracker.GetCameraPredictedUncertainty(&cam_state_covar);

            // put new data log entries into plotter queue
            PlotterDataLogItem data_log_item;
            data_log_item.MaxCamPosUncert = cam_state_covar.diagonal().maxCoeff();
            {
                std::lock_guard<std::mutex> lk(ui_params.worker_chat->tracker_and_ui_mutex_);
                plotter_data_log_item_history.push_back(data_log_item);
            }

            observable_frame_ind = frame_ind;
#endif
        }

        VLOG(4) << "f=" << frame_ind
            << " fps=" << (frame_process_time.has_value() ? 1 / frame_process_time.value().count() : 0.0f)
            << " tracks_count=" << tracker.SalientPointsCount();

#if defined(SRK_HAS_OPENCV)
        if (FLAGS_ctrl_visualize_during_processing)
        {
            camera_image_rgb.setTo(0);
            auto project_fun = [&cam_cfw, &tracker](const suriko::Point3& sal_pnt) -> Eigen::Matrix<suriko::Scalar, 3, 1>
            {
                suriko::Point3 pnt_cam = SE3Apply(cam_cfw, sal_pnt);
                suriko::Point2 pnt_pix = tracker.ProjectCameraPoint(pnt_cam);
                return Eigen::Matrix<suriko::Scalar, 3, 1>(pnt_pix[0], pnt_pix[1], 1);
            };
            constexpr Scalar f0 = 1;
            suriko_demos::Draw2DProjectedAxes(f0, project_fun, &camera_image_rgb);

            //
            for (const BlobInfo& blob_info : corners_matcher.DetectedBlobs())
            {
                Scalar pix_x = blob_info.Coord[0];
                Scalar pix_y = blob_info.Coord[1];
                camera_image_rgb.at<cv::Vec3b>((int)pix_y, (int)pix_x) = cv::Vec3b(0xFF, 0xFF, 0xFF);
            }

            std::stringstream strbuf;
            strbuf << "f=" << frame_ind;
            cv::putText(camera_image_rgb, cv::String(strbuf.str()), cv::Point(10, (int)img_size[1] - 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
            cv::imshow("front-camera", camera_image_rgb);
            cv::waitKey(1); // allow to refresh an opencv view
        }
#endif

#if defined(SRK_HAS_PANGOLIN)
        {
            // check if UI requests the exit
            std::lock_guard<std::mutex> lk(worker_chat->exit_worker_mutex);
            if (worker_chat->exit_worker_flag)
                break;
        }
        
        bool suppress_observations = false;
        if (FLAGS_ctrl_wait_after_each_frame)
        {
            std::unique_lock<std::mutex> ulk(worker_chat->resume_worker_mutex);
            worker_chat->resume_worker_flag = false; // reset the waiting flag
            // wait till UI requests to resume processing
            // TODO: if worker blocks, then UI can't request worker to exit; how to coalesce these?
            worker_chat->resume_worker_cv.wait(ulk, [&worker_chat] {return worker_chat->resume_worker_flag; });
            suppress_observations = worker_chat->resume_worker_suppress_observations;
        }
        dynamic_cast<DemoCornersMatcher&>(tracker.CornersMatcher()).SetSuppressObservations(suppress_observations);
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
    if (FLAGS_ctrl_visualize_after_processing && !ui_thread.joinable()) // don't create thread second time
        ui_thread = std::thread(SceneVisualizationThread, ui_params);

    if (ui_thread.joinable())
    {
        VLOG(4) << "Waiting for UI to request the exit";
        {
            // wait for Pangolin UI to request the exit
            std::unique_lock<std::mutex> ulk(worker_chat->exit_worker_mutex);
            worker_chat->exit_worker_cv.wait(ulk, [&worker_chat] {return worker_chat->exit_worker_flag; });
        }
        VLOG(4) << "Got UI notification to exit working thread";
        {
            // notify Pangolin UI to finish visualization thread
            std::lock_guard<std::mutex> lk(worker_chat->exit_ui_mutex);
            worker_chat->exit_ui_flag = true;
        }
        VLOG(4) << "Waiting for UI to perform the exit";
        ui_thread.join();
        VLOG(4) << "UI thread has been shut down";
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