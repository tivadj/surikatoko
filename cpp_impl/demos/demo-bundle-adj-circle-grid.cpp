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
#include "stat-helpers.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/highgui.hpp>
#endif

namespace suriko_demos
{
using namespace std;
using namespace boost::filesystem;
using namespace suriko;
using namespace suriko::internals;

static bool ValidateDirectoryExists(const char *flagname, const std::string &value)
{
    boost::filesystem::path test_data_path = boost::filesystem::absolute(value).normalize();
    if (boost::filesystem::is_directory(test_data_path))
        return true;
    std::cout <<"directory " <<test_data_path.string() << " doesn't exist" <<std::endl;
    return false;
}

DEFINE_string(testdata, "NOTFOUND", "Abs/rel path to testdata directory");
DEFINE_validator(testdata, &ValidateDirectoryExists);
DEFINE_double(allowed_repr_err, 1e-5, "Reprojection error change threshold (in pix)");
DEFINE_double(noise_R_hi, 0, "Upper bound of noise uniform distribution for R, 0=no noise (eg: 0.01)");
DEFINE_double(noise_x3D_hi, 0, "Upper bound of noise uniform distribution for salient points, 0=no noise (eg: 0.1)");

struct WorldBounds
{
    Scalar XMin;
    Scalar XMax;
    Scalar YMin;
    Scalar YMax;
    Scalar ZMin;
    Scalar ZMax;
};

int CircleGridDemo(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);

    const string& test_data = FLAGS_testdata;

    boost::filesystem::path test_data_path = boost::filesystem::absolute(test_data).normalize();
    LOG(INFO) <<"testdata=" << test_data_path;

    LOG(INFO) << FLAGS_noise_R_hi;
    //
    bool provide_ground_truth = true;
    bool corrupt_salient_points_with_noise = FLAGS_noise_x3D_hi > 0;
    bool corrupt_cam_orient_with_noise = FLAGS_noise_R_hi > 0;
    std::vector<SE3Transform> ground_truth_RT_per_frame;
    std::vector<Eigen::Matrix<Scalar, 3, 3>> intrinsic_cam_mat_per_frame;

    WorldBounds wb{};
    wb.XMin = -2;
    wb.XMax = 2;
    wb.YMin = -0.5;
    wb.YMax = 0.5;
    wb.ZMin = 0;
    wb.ZMax = 0.5;
    std::array<Scalar, 3> cell_size = { 0.5, 0.5, 0.5 };
    Scalar rot_radius = 15 * cell_size[0];
        
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
    for (Scalar ang = 0; ang < 2*M_PI/3; ang += M_PI/180*5)
    {
        Scalar ang_offset = 3 * M_PI / 2 + M_PI / 6;
        rot_angles.push_back(ang_offset - ang);
    }

    LOG(INFO) << "frames_count=" << rot_angles.size();

#if defined(SRK_HAS_OPENCV)
    cv::Mat_<uchar> camera_image_gray = cv::Mat_<uchar>::zeros((int)img_size[1], (int)img_size[0]);
#endif
    for (size_t ang_ind = 0; ang_ind < rot_angles.size(); ++ang_ind)
    {
        Scalar ang = rot_angles[ang_ind];

        // X is directed to the right, Y - to up
        Eigen::Matrix<Scalar, 4, 4> cam_from_world = Eigen::Matrix<Scalar, 4, 4>::Identity();

        // move to position at the circle from center
        Scalar shiftX = cell_size[0] * std::cos(ang);
        Scalar shiftY = cell_size[0] * std::sin(ang);
        Scalar shiftZ = cell_size[0];
        Eigen::Matrix<Scalar, 3, 1> shift(shiftX, shiftY, shiftZ);
        shift.normalize();
        shift *= rot_radius;

        cam_from_world = SE3Mat(Eigen::Matrix<Scalar, 3, 1>(-shift)) * cam_from_world;

        // rotate OY around OZ so that OY points 'towards center'
        Eigen::Matrix<Scalar, 3, 1> toCenterXOY(-shiftX, -shiftY, 0); // the direction towards center O
        toCenterXOY.normalize();
        Eigen::Matrix<Scalar, 3, 1> oy(0, 1, 0);
        Scalar ang_yawOY = std::acos(oy.dot(toCenterXOY)); // rotate OY 'towards' (parallel to XOY plane) center
        
        // correct sign so that OY is rotated towards center by shortest angle
        Eigen::Matrix<Scalar, 3, 1> oz(0, 0, 1);
        int ang_yawOY_sign = Sign(oy.cross(toCenterXOY).dot(oz));
        ang_yawOY *= ang_yawOY_sign;

        cam_from_world = SE3Mat(RotMat(oz, -ang_yawOY)) * cam_from_world;

        // look down towards the center
        Scalar look_down_ang = std::atan2(shiftZ, Eigen::Matrix<Scalar, 3, 1>(shiftX, shiftY, 0).norm());

        cam_from_world = SE3Mat(RotMat(1, 0, 0, look_down_ang + M_PI/2)) * cam_from_world;
        SE3Transform RT(cam_from_world.topLeftCorner(3, 3), cam_from_world.topRightCorner(3, 1));

        if (provide_ground_truth)
        {
            ground_truth_RT_per_frame.push_back(RT);
        }
        intrinsic_cam_mat_per_frame.push_back(K);

#if defined(SRK_HAS_OPENCV)
        camera_image_gray.setTo(0);
#endif
        for(size_t frag_ind = 0; frag_ind < map.SalientPoints().size(); ++frag_ind)
        {
            const SalientPointFragment& frag = map.SalientPoints()[frag_ind];
            if (!frag.Coord.has_value()) continue;

            Point3 pnt_camera = SE3Apply(RT, frag.Coord.value());

            // perform general projection 3D->2D
            Eigen::Matrix<Scalar, 3, 1> pnt_img = pnt_camera.Mat() / pnt_camera[2];
            
            Eigen::Matrix<Scalar, 3, 1> pnt_pix = K * pnt_img;

            CornerTrack& track = track_rep.GetPointTrackById(frag_ind);
            CHECK_EQ(track.SyntheticSalientPointId, frag.SyntheticVirtualPointId.value());

            auto pnt2_pix = suriko::Point2(pnt_pix[0], pnt_pix[1]);
            track.AddCorner(ang_ind, pnt2_pix);

#if defined(SRK_HAS_OPENCV)
            int pix_x = (int)pnt_pix[0];
            int pix_y = (int)pnt_pix[1];
            camera_image_gray(pix_y, pix_x) = 0xFF;
#endif
        }

#if defined(SRK_HAS_OPENCV)
        cv::imshow("front-camera", camera_image_gray);
        cv::waitKey(100); // 0=wait forever
#endif
    }

    auto inverse_orient_cam_per_frame_noise = ground_truth_RT_per_frame; // copy
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
            unity_dir /= unity_dir.norm();
            

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

    LOG(INFO) << "start bundle adjustment..." << endl;
    bool op = ba.ComputeInplace(f0, map_noise, inverse_orient_cam_per_frame_noise, track_rep, nullptr, &intrinsic_cam_mat_per_frame, term_crit);

    LOG(INFO) << "bundle adjustment finished with result: " << op << " (" << ba.OptimizationStatusString() <<")" << endl;

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
