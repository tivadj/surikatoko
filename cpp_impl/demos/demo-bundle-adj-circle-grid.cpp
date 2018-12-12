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

#include "visualize-helpers.h"

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

DEFINE_double(f0, 600, "Numerical stability scaler, chosen so that x_pix/f0 and y_pix/f0 is close to 1. Kanatani uses f0=600");
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
    std::vector<SE3Transform> gt_cam_orient_cfw; // ground truth camera orientation transforming into camera from world
    std::vector<Eigen::Matrix<Scalar, 3, 3>> intrinsic_cam_mat_per_frame;

    WorldBounds wb{};
    wb.x_min = FLAGS_world_xmin;
    wb.x_max = FLAGS_world_xmax;
    wb.y_min = FLAGS_world_ymin;
    wb.y_max = FLAGS_world_ymax;
    wb.z_min = FLAGS_world_zmin;
    wb.z_max = FLAGS_world_zmax;
    std::array<Scalar, 3> cell_size = { FLAGS_world_cell_size_x, FLAGS_world_cell_size_y, FLAGS_world_cell_size_z };
    Scalar rot_radius = 15 * cell_size[0];
    Scalar ascentZ = 10 * cell_size[0];
    suriko::Point3 circle_center(1, 0.5, 0);

    constexpr Scalar inclusive_gap = 1e-8; // small value to make iteration inclusive

    size_t next_synthetic_virtual_point_id = 1000001;
    FragmentMap map;
    Scalar xmid = (wb.x_min + wb.x_max) / 2;
    Scalar xlen = wb.x_max - wb.x_min;
    Scalar zlen = wb.z_max - wb.z_min;
    for (Scalar x = wb.x_min; x < wb.x_max + inclusive_gap; x += cell_size[0])
    {
        for (Scalar y = wb.y_min; y < wb.y_max + inclusive_gap; y += cell_size[1])
        {
            Scalar val_z = std::cos((x - xmid) / xlen * M_PI);
            Scalar z = wb.z_min + val_z * zlen;
            SalientPointFragment& salient_point = map.AddSalientPointPatch(Point3(x, y, z));
            salient_point.synthetic_virtual_point_id = next_synthetic_virtual_point_id;
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
            if (!fragment.coord.has_value()) continue;
            suriko::Point3& pnt = fragment.coord.value();
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
    CornerTrackRepository track_rep;
    
    std::vector<size_t> salient_points_ids;
    map_noise.GetSalientPointsIds(&salient_points_ids);

    for (size_t sal_pnt_id : salient_points_ids)
    {
        const SalientPointFragment& sal_pnt = map_noise.GetSalientPointNew(sal_pnt_id);

        CornerTrack& point_track = track_rep.AddCornerTrackObj();
        point_track.SalientPointId = sal_pnt_id;
        point_track.SyntheticVirtualPointId = sal_pnt.synthetic_virtual_point_id;
    }

    // Numerical stability scaler, chosen so that x_pix / f0 and y_pix / f0 is close to 1
    Scalar f0 = FLAGS_f0;
    LOG(INFO) << "f0=" << f0;
    Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> num_stab_mat;
    num_stab_mat <<
        1 / f0, 0, 0,
        0, 1 / f0, 0,
        0, 0, 1;

    std::array<size_t, 2> img_size = { 800, 600 };
    Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> K;
    K <<
        880, 0, img_size[0] / 2.0,
        0, 660, img_size[1] / 2.0,
        0, 0, 1;
    K = num_stab_mat * K;

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

    GenerateCircleCameraShots(circle_center, rot_radius, ascentZ, rot_angles, &gt_cam_orient_cfw);

#if defined(SRK_HAS_OPENCV)
    cv::Mat camera_image_rgb = cv::Mat::zeros((int)img_size[1], (int)img_size[0], CV_8UC3);
#endif
    for (size_t ang_ind = 0; ang_ind< gt_cam_orient_cfw.size(); ++ang_ind)
    {
        const SE3Transform& RT_cfw = gt_cam_orient_cfw[ang_ind];
        intrinsic_cam_mat_per_frame.push_back(K);

#if defined(SRK_HAS_OPENCV)
        camera_image_rgb.setTo(0);
        auto project_fun = [&K, &RT_cfw](const suriko::Point3& sal_pnt) -> Eigen::Matrix<suriko::Scalar, 3, 1>
        {
            return ProjectPnt(K, RT_cfw, sal_pnt);
        };
        Draw2DProjectedAxes(f0, project_fun, &camera_image_rgb);
#endif
        for (size_t frag_ind = 0; frag_ind < map.SalientPoints().size(); ++frag_ind)
        {
            const SalientPointFragment& frag = map.SalientPoints()[frag_ind];
            if (!frag.coord.has_value()) continue;

            CornerTrack& track = track_rep.GetPointTrackById(frag_ind);
            CHECK_EQ(track.SyntheticVirtualPointId.value(), frag.synthetic_virtual_point_id.value());

            Eigen::Matrix<Scalar, 3, 1> pnt_homog = ProjectPnt(K, RT_cfw, frag.coord.value());
            auto pnt_div_f0 = Eigen::Matrix<Scalar, 2, 1>(pnt_homog[0] / pnt_homog[2], pnt_homog[1] / pnt_homog[2]);
            auto pnt_pix = suriko::Point2(pnt_div_f0 * f0);
            track.AddCorner(ang_ind, pnt_pix);

#if defined(SRK_HAS_OPENCV)
            int pix_x = (int)pnt_pix[0];
            int pix_y = (int)pnt_pix[1];
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

    std::vector<SE3Transform> inverse_orient_cam_per_frame_noise = gt_cam_orient_cfw; // copy
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
        for (const SalientPointFragment& sal_pnt :  map.SalientPoints())
        {
            const SalientPointFragment* sal_pnt_noise = nullptr;
            bool op = map_noise.GetSalientPointByVirtualPointIdInternal(sal_pnt.synthetic_virtual_point_id.value(), &sal_pnt_noise);
            SRK_ASSERT(op);

            Scalar pnt_diff = (sal_pnt.coord.value().Mat() - sal_pnt_noise->coord.value().Mat()).norm();
            stat_pnt.Next(pnt_diff);
        }
        LOG(INFO) << "avg_pnt_diff=" << stat_pnt.Mean() << " std_pnt_diff=" << stat_pnt.Std();

        MeanStdAlgo stat_R;
        for (size_t i=0; i<inverse_orient_cam_per_frame_noise.size(); ++i)
        {
            const SE3Transform& rt_gt = gt_cam_orient_cfw[i];
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
