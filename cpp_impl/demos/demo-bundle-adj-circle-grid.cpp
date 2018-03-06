#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <cassert>
#include <cmath>
#include <corecrt_math_defines.h>
#include <tuple>
#include <random>
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
#include "suriko/virt-world/scene-generator.h"
#include "stat-helpers.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/imgproc.hpp> // cv::circle
#include <opencv2/highgui.hpp> // cv::imshow
#endif

namespace suriko_demos
{
using namespace std;
using namespace boost::filesystem;
using namespace suriko;
using namespace suriko::internals;
using namespace suriko::virt_world;

auto ProjectPnt(const Eigen::Matrix<Scalar, 3, 3>& K, const SE3Transform& cam_inverse_orient, const Point3& coord)
{
    Point3 pnt_camera = SE3Apply(cam_inverse_orient, coord);

    // perform general projection 3D->2D
    Eigen::Matrix<Scalar, 3, 1> pnt_img = pnt_camera.Mat() / pnt_camera[2];

    Eigen::Matrix<Scalar, 3, 1> pnt_pix = K * pnt_img;
    return pnt_pix;
}

#if defined(SRK_HAS_OPENCV)
void DrawAxes(const Eigen::Matrix<Scalar, 3, 3>& K, const SE3Transform& cam_inverse_orient, const cv::Mat& camera_image_rgb)
{
    // show center of coordinates as red dot
    std::vector<suriko::Point3> axes_pnts = {
        suriko::Point3(0, 0, 0),
        suriko::Point3(1, 0, 0),
        suriko::Point3(0, 1, 0),
        suriko::Point3(0, 0, 1)
    };
    std::vector<cv::Scalar> axes_colors = {
        CV_RGB(255, 255, 255),
        CV_RGB(255,   0,   0),
        CV_RGB(0, 255,   0),
        CV_RGB(0,   0, 255)
    };
    std::vector<cv::Point2i> axes_pnts2D(axes_pnts.size());
    for (size_t i = 0; i < axes_pnts.size(); ++i)
    {
        Eigen::Matrix<Scalar, 3, 1> p = ProjectPnt(K, cam_inverse_orient, axes_pnts[i]);
        axes_pnts2D[i] = cv::Point2i(
            static_cast<int>(p[0] / p[2]),
            static_cast<int>(p[1] / p[2]));
    }
    for (size_t i = 1; i < axes_pnts.size(); ++i)
    {
        cv::line(camera_image_rgb, axes_pnts2D[0], axes_pnts2D[i], axes_colors[i]); // OX, OZ, OZ segments
        //cv::circle(camera_image_rgb, axes_pnts2D[i], 3, axes_colors[i]);
    }
}
#endif

DEFINE_double(world_xmin, -1, "world xmin");
DEFINE_double(world_xmax, 1, "world xmax");
DEFINE_double(world_ymin, -1, "world ymin");
DEFINE_double(world_ymax, 1, "world ymax");
DEFINE_double(world_zmin, 0, "world zmin");
DEFINE_double(world_zmax, 1, "world zmax");
DEFINE_double(world_cell_size_x, 0.5, "cell size x");
DEFINE_double(world_cell_size_y, 0.5, "cell size y");
DEFINE_double(world_cell_size_z, 0.5, "cell size z");
DEFINE_double(ang_start, -M_PI / 2 + M_PI / 6, "points on the circle for camera shot");
DEFINE_double(ang_end, 2 * M_PI / 3, "");
DEFINE_double(ang_step, M_PI / 180 * 5, "");
DEFINE_double(noise_R_hi, 0.005, "Upper bound of noise uniform distribution for R, 0=no noise (eg: 0.01)");
DEFINE_double(noise_x3D_hi, 0.005, "Upper bound of noise uniform distribution for salient points, 0=no noise (eg: 0.1)");
DEFINE_int32(wait_key_delay, 1, "parameter to cv::waitKey; 0 means 'wait forever'");
DEFINE_double(allowed_repr_err, 1e-5, "Reprojection error change threshold (in pix)");

int CircleGridDemo(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "noise_x3D_hi=" << FLAGS_noise_x3D_hi;
    LOG(INFO) << "noise_R_hi=" << FLAGS_noise_R_hi;

    //
    bool corrupt_salient_points_with_noise = FLAGS_noise_x3D_hi > 0;
    bool corrupt_cam_orient_with_noise = FLAGS_noise_R_hi > 0;
    std::vector<SE3Transform> ground_truth_RT_per_frame;
    std::vector<Eigen::Matrix<Scalar, 3, 3>> intrinsic_cam_mat_per_frame;

    WorldBounds wb{};
    wb.XMin = FLAGS_world_xmin;
    wb.XMax = FLAGS_world_xmax;
    wb.YMin = FLAGS_world_ymin;
    wb.YMax = FLAGS_world_ymax;
    wb.ZMin = FLAGS_world_zmin;
    wb.ZMax = FLAGS_world_zmax;
    std::array<Scalar, 3> cell_size = { FLAGS_world_cell_size_x, FLAGS_world_cell_size_y, FLAGS_world_cell_size_z };
    Scalar rot_radius = 15 * cell_size[0];
    Scalar ascentZ = 10 * cell_size[0];
    suriko::Point3 circle_center(1, 0.5, 0);

    constexpr Scalar inclusive_gap = 1e-8; // small value to make iteration inclusive

    size_t next_synthetic_virtual_point_id = 1000001;
    FragmentMap map;
    Scalar xmid = (wb.XMin + wb.XMax) / 2;
    Scalar xlen = wb.XMax - wb.XMin;
    Scalar zlen = wb.ZMax - wb.ZMin;
    for (Scalar x = wb.XMin; x < wb.XMax + inclusive_gap; x += cell_size[0])
    {
        for (Scalar y = wb.YMin; y < wb.YMax + inclusive_gap; y += cell_size[1])
        {
            Scalar val_z = std::cos((x - xmid) / xlen * M_PI);
            Scalar z = wb.ZMin + val_z * zlen;
            map.AddSalientPointNew(Point3(x, y, z), next_synthetic_virtual_point_id);
            next_synthetic_virtual_point_id += 1;
        }
    }

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    gen.seed(1234);

    FragmentMap map_noise = map; // copy
    if (corrupt_salient_points_with_noise)
    {
        std::uniform_real_distribution<Scalar> dis(FLAGS_noise_x3D_hi / 2, FLAGS_noise_x3D_hi);
        for (SalientPointFragment& fragment : map_noise.SalientPoints())
        {
            if (!fragment.Coord.has_value()) continue;
            suriko::Point3& pnt = fragment.Coord.value();
            auto d1 = dis(gen);
            auto d2 = dis(gen);
            auto d3 = dis(gen);
            pnt[0] += d1;
            pnt[1] += d2;
            pnt[2] += d3;
        }
    }


    LOG(INFO) << "points_count=" << map_noise.SalientPointsCount();

    // lets track each salient point
    size_t next_point_track_id = 0;
    CornerTrackRepository track_rep;
    for (const SalientPointFragment& fragment : map_noise.SalientPoints())
    {
        CornerTrack point_track;
        point_track.TrackId = next_point_track_id++;
        point_track.SyntheticSalientPointId = fragment.SyntheticVirtualPointId.value();
        track_rep.CornerTracks.push_back(point_track);
    }

    Scalar f0 = 1;
    std::array<size_t, 2> img_size = { 800, 600 };
    Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> K;
    K <<
        880, 0, img_size[0] / 2.0,
        0, 660, img_size[1] / 2.0,
        0, 0, 1;

    vector<Scalar> rot_angles;
    for (Scalar ang = FLAGS_ang_start; ; ang += FLAGS_ang_step)
    {
        if ((FLAGS_ang_start < FLAGS_ang_end && ang >= FLAGS_ang_end) ||
            (FLAGS_ang_start > FLAGS_ang_end && ang <= FLAGS_ang_end))
        {
            break;
        }
        rot_angles.push_back(ang);
    }

    LOG(INFO) << "frames_count=" << rot_angles.size();

    GenerateCircleCameraShots(circle_center, rot_radius, ascentZ, rot_angles, &ground_truth_RT_per_frame);

#if defined(SRK_HAS_OPENCV)
    cv::Mat camera_image_rgb = cv::Mat::zeros((int)img_size[1], (int)img_size[0], CV_8UC3);
#endif
    for (size_t ang_ind = 0; ang_ind< ground_truth_RT_per_frame.size(); ++ang_ind)
    {
        const SE3Transform& RT = ground_truth_RT_per_frame[ang_ind];
        intrinsic_cam_mat_per_frame.push_back(K);

#if defined(SRK_HAS_OPENCV)
        camera_image_rgb.setTo(0);
        DrawAxes(K, RT, camera_image_rgb);
#endif
        for (size_t frag_ind = 0; frag_ind < map.SalientPoints().size(); ++frag_ind)
        {
            const SalientPointFragment& frag = map.SalientPoints()[frag_ind];
            if (!frag.Coord.has_value()) continue;

            CornerTrack& track = track_rep.GetPointTrackById(frag_ind);
            CHECK_EQ(track.SyntheticSalientPointId, frag.SyntheticVirtualPointId.value());

            auto pnt_pix = ProjectPnt(K, RT, frag.Coord.value());
            auto pnt2_pix = suriko::Point2(pnt_pix[0] / pnt_pix[2], pnt_pix[1] / pnt_pix[2]);
            track.AddCorner(ang_ind, pnt2_pix);

#if defined(SRK_HAS_OPENCV)
            int pix_x = (int)pnt2_pix[0];
            int pix_y = (int)pnt2_pix[1];
            if (pix_x >= 0 && pix_x < img_size[0] &&
                pix_y >= 0 && pix_y < img_size[1])
                camera_image_rgb.at<cv::Vec3b>(pix_y, pix_x) = cv::Vec3b(0xFF, 0xFF, 0xFF);
#endif
        }

#if defined(SRK_HAS_OPENCV)
        cv::imshow("front-camera", camera_image_rgb);
        cv::waitKey(FLAGS_wait_key_delay); // 0=wait forever
#endif
    }

    std::vector<SE3Transform> inverse_orient_cam_per_frame_noise = ground_truth_RT_per_frame; // copy
    if (corrupt_cam_orient_with_noise)
    {
        std::uniform_real_distribution<Scalar> dis(0, 1);
        for (size_t i = 0; i < inverse_orient_cam_per_frame_noise.size(); ++i)
        {
            SE3Transform& rt = inverse_orient_cam_per_frame_noise[i];

            Eigen::Matrix<Scalar, 3, 1> unity_dir;
            Scalar ang;
            if (!LogSO3(rt.R, &unity_dir, &ang))
            {
                VLOG(4) << "failed R->unit_dir,ang";
                continue;
            }

            Scalar da = dis(gen) * FLAGS_noise_R_hi;
            ang += da;

            Scalar dw1 = dis(gen) * FLAGS_noise_R_hi;
            Scalar dw2 = dis(gen) * FLAGS_noise_R_hi;
            Scalar dw3 = dis(gen) * FLAGS_noise_R_hi;
            unity_dir[0] += dw1;
            unity_dir[1] += dw2;
            unity_dir[2] += dw3;
            unity_dir.normalize();

            bool op = RotMatFromUnityDirAndAngle(unity_dir, ang, &rt.R);
            if (!op)
            {
                VLOG(4) << "failed unit_dir,ang->R";
            }
        }
    }

    auto calc_and_print_stats = [&]()
    {
        MeanStdAlgo stat_pnt;
        for (size_t i = 0; i < map.SalientPointsCount(); ++i)
        {
            Scalar pnt_diff = (map.GetSalientPoint(i).Mat() - map_noise.GetSalientPoint(i).Mat()).norm();
            stat_pnt.Next(pnt_diff);
        }
        LOG(INFO) << "avg_pnt_diff=" << stat_pnt.Mean() << " std_pnt_diff=" << stat_pnt.Std();

        MeanStdAlgo stat_R;
        for (size_t i=0; i<inverse_orient_cam_per_frame_noise.size(); ++i)
        {
            const SE3Transform& rt_gt = ground_truth_RT_per_frame[i];
            const SE3Transform& rt_noise = inverse_orient_cam_per_frame_noise[i];
            Scalar r_diff = (rt_gt.R - rt_noise.R).norm();
            stat_R.Next(r_diff);
        }
        LOG(INFO) << "avg_rot_diff=" << stat_R.Mean() << " std_rot_diff=" << stat_R.Std();
    };
    calc_and_print_stats();

    BundleAdjustmentKanatani ba;
    BundleAdjustmentKanataniTermCriteria term_crit;
    if (FLAGS_allowed_repr_err > 0)
        term_crit.AllowedReprojErrRelativeChange(FLAGS_allowed_repr_err);

    LOG(INFO) << "start bundle adjustment...";
    bool op = ba.ComputeInplace(f0, map_noise, inverse_orient_cam_per_frame_noise, track_rep, nullptr, &intrinsic_cam_mat_per_frame, term_crit);

    LOG(INFO) << "bundle adjustment finished with result: " << op << " (" << ba.OptimizationStatusString() <<")";

    calc_and_print_stats();

    return 0;
}
}

int main(int argc, char* argv[])
{
    int result = 0;
    result = suriko_demos::CircleGridDemo(argc, argv);
    return result;
}
