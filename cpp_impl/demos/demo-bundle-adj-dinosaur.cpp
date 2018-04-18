#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <cassert>
#include <cmath>
#include <tuple>
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

namespace suriko_demos
{
using namespace std;
//using namespace std::experimental::filesystem;
using namespace boost::filesystem;
using namespace suriko;

void PopulateCornersPerFrame(const vector<Scalar>& viff_data_by_row, size_t viff_num_rows, size_t viff_num_cols,
                             CornerTrackRepository *track_rep, ptrdiff_t min_frames_per_point = -1, ptrdiff_t max_points_count = -1)
{
    size_t orig_points_count = viff_num_rows;
    size_t orig_frames_count = viff_num_cols/2;
    size_t next_track_id = 0;
    for (size_t pnt_ind = 0; pnt_ind < orig_points_count; ++pnt_ind)
    {
        if (max_points_count != -1 && track_rep->CornerTracks.size() >= (size_t)max_points_count)
            break;

        CornerTrack track;

        for (size_t frame_ind = 0; frame_ind < orig_frames_count; ++frame_ind)
        {
            size_t i = pnt_ind*viff_num_cols+frame_ind*2;
            auto x = viff_data_by_row[i];
            auto y = viff_data_by_row[i+1];
            if (x == -1 || y == -1) continue;

            track.AddCorner(frame_ind, suriko::Point2(x,y));
        }
        if (!track.HasCorners()) continue; // track without registered corners
        if (min_frames_per_point != -1 && track.CornersCount() < (size_t)min_frames_per_point)
            continue;

        track.SyntheticVirtualPointId = 10000 + pnt_ind;
        track.TrackId = next_track_id++;
        track_rep->CornerTracks.push_back(track);
    }
}

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
DEFINE_double(f0, 600, "Numerical stability scaler, chosen so that x_pix/f0 and y_pix/f0 is close to 1. Kanatani uses f0=600");

int DinoDemo(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);

    const string& test_data = FLAGS_testdata;

    boost::filesystem::path test_data_path = boost::filesystem::absolute(test_data).normalize();
    LOG(INFO) <<"testdata=" << test_data_path;

    ptrdiff_t min_frames_per_point = -1; // set -1 to ignore
    ptrdiff_t max_points_count = -1; // set -1 to ignore
    bool zero_cam_intrinsic_mat_01 = true; // sets K[0,1]=1
    bool equalize_fxfy = false; // makes fx=fy

    auto proj_mats_file_path = (test_data_path / "oxfvisgeom/dinosaur/dinoPs_as_mat108x4.txt").normalize();

    vector<Scalar> P_data_by_row;
    size_t P_num_rows, P_num_cols;
    string err_msg;
    bool op= ReadMatrixFromFile(proj_mats_file_path, '\t', &P_data_by_row, &P_num_rows, &P_num_cols, &err_msg);
    if (!op)
    {
        LOG(ERROR) << err_msg;
        return 1;
    }

    size_t orig_frames_count = P_num_rows / 3; // 36
    LOG(INFO) <<"frames_count=" <<orig_frames_count <<endl;

    auto viff_mats_file_path = (test_data_path / "oxfvisgeom/dinosaur/viff.xy").normalize();
    vector<Scalar> viff_data_by_row;
    size_t viff_num_rows, viff_num_cols;
    op= ReadMatrixFromFile(viff_mats_file_path, ' ', &viff_data_by_row, &viff_num_rows, &viff_num_cols, &err_msg);
    if (!op)
    {
        LOG(ERROR) << err_msg;
        return 1;
    }

    if (orig_frames_count != viff_num_cols / 2)
    {
        LOG(ERROR) << "Inconsistent frames_count";
        return 1;
    }

    size_t orig_points_count = viff_num_rows; // =4983
    LOG(INFO) << "points_count=" <<orig_points_count <<endl;

    CornerTrackRepository track_rep;
    PopulateCornersPerFrame(viff_data_by_row, viff_num_rows, viff_num_cols, &track_rep, min_frames_per_point, max_points_count);
    LOG(INFO) << "subset points_count=" << track_rep.CornerTracks.size() << endl;

    // determine subset of frames
    vector<size_t> subset_point_track_ids;
    track_rep.PopulatePointTrackIds(&subset_point_track_ids);

    //
    vector<Eigen::Matrix<Scalar, 3, 4>> proj_mat_per_frame_original;
    vector<Eigen::Matrix<Scalar, 3, 4>> proj_mat_per_frame_f0scaled;
    vector<SE3Transform> inverse_orient_cam_per_frame;
    vector<Eigen::Matrix<Scalar, 3, 3>> intrinsic_cam_mat_per_frame; // K

    Scalar f0 = FLAGS_f0;
    LOG(INFO) << "f0=" << f0;

    Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> num_stab_mat;
    num_stab_mat <<
        1 / f0, 0, 0,
        0, 1 / f0, 0,
        0, 0, 1;

    for (size_t frame_ind = 0; frame_ind < orig_frames_count; ++frame_ind)
    {
        Eigen::Map<Eigen::Matrix<Scalar, 3, 4, Eigen::RowMajor>> proj_mat_row_major(P_data_by_row.data() + frame_ind * 12, 3, 4);
        Eigen::Matrix<Scalar, 3, 4> proj_mat = proj_mat_row_major;
        proj_mat_per_frame_original.push_back(proj_mat);

        bool op_decomp;
        Scalar scale_factor;
        Eigen::Matrix<Scalar, 3, 3> K;
        SE3Transform direct_orient_cam;
        std::tie(op_decomp, scale_factor, K, direct_orient_cam) = DecomposeProjMat(proj_mat);
        CHECK(op_decomp) << "Can't decompose projection matrix for frame_ind=" << frame_ind << ", P=" << proj_mat;

        Eigen::Matrix<Scalar, 3, 3> Knew = num_stab_mat * K;

        if (zero_cam_intrinsic_mat_01)
            Knew(0, 1) = 0;
        if (equalize_fxfy)
        {
            Scalar favg = (Knew(0, 0) + Knew(1, 1)) / 2; // (fx+fy)/2
            Knew(0, 0) = favg;
            Knew(1, 1) = favg;
        }
        intrinsic_cam_mat_per_frame.push_back(Knew);

        SE3Transform inverse_orient_cam = SE3Inv(direct_orient_cam);
        inverse_orient_cam_per_frame.push_back(inverse_orient_cam);
        
        Eigen::Matrix<Scalar, 3, 4> P_f0scaled;
        P_f0scaled << Knew * inverse_orient_cam.R, Knew * inverse_orient_cam.T;
        proj_mat_per_frame_f0scaled.push_back(P_f0scaled);
    }

    // triangulate 3D points
    vector<Point2> one_pnt_corner_per_frame;
    vector<Eigen::Matrix<Scalar,3,4>> one_pnt_proj_mat_per_frame;
    FragmentMap map;
    for (size_t pnt_track_id : subset_point_track_ids)
    {
        const CornerTrack& corner_track = track_rep.GetPointTrackById(pnt_track_id);

        one_pnt_corner_per_frame.clear();
        one_pnt_proj_mat_per_frame.clear();
        for (size_t frame_ind = 0; frame_ind < orig_frames_count; ++frame_ind)
        {
            optional<Point2> corner = corner_track.GetCorner(frame_ind);
            if (!corner.has_value())
                continue;
            one_pnt_corner_per_frame.push_back(corner.value());
            one_pnt_proj_mat_per_frame.push_back(proj_mat_per_frame_f0scaled[frame_ind]);
        }
        Point3 x3D = Triangulate3DPointByLeastSquares(one_pnt_corner_per_frame, one_pnt_proj_mat_per_frame, f0);
        map.AddSalientPoint(pnt_track_id, x3D);
    }

    static bool debug_reproj_err = false;
    if (debug_reproj_err)
    {
        for (size_t point_track_id : subset_point_track_ids) {
            const auto &point_track = track_rep.GetPointTrackById(point_track_id);

            Point3 x3D = map.GetSalientPoint(point_track_id);
            Eigen::Matrix<Scalar,4,1> x3D_homog(x3D[0], x3D[1], x3D[2], 1);

            for (size_t frame_ind = 0; frame_ind < orig_frames_count; ++frame_ind) {
                const auto &corner = point_track.GetCorner(frame_ind);
                if (!corner.has_value())
                    continue;

                // homogeneous component for corner is constant = 1
                const suriko::Point2& cor = corner.value();

                const auto &P = proj_mat_per_frame_f0scaled[frame_ind];
                Eigen::Matrix<Scalar, 3, 1> x2D_homog = P * x3D_homog;
                VLOG_IF(4, IsClose(0, x2D_homog[2])) << "point at infinity point_track_id=" << point_track_id << " x2D_homog=" << x2D_homog;

                auto pix_x = x2D_homog[0] / x2D_homog[2] * f0;
                auto pix_y = x2D_homog[1] / x2D_homog[2] * f0;
                auto err_pix = (Eigen::Matrix<Scalar, 2, 1>(pix_x, pix_y) - cor.Mat()).norm();
                
                LOG_IF(INFO, err_pix > 5) << "repr err=" <<err_pix <<" for point_track_id=" << point_track_id;
            }
        }
    }

    BundleAdjustmentKanatani ba;
    BundleAdjustmentKanataniTermCriteria term_crit;
    if (FLAGS_allowed_repr_err > 0)
        term_crit.AllowedReprojErrRelativeChange(FLAGS_allowed_repr_err);

    LOG(INFO) << "start bundle adjustment..." <<endl;
    op = ba.ComputeInplace(f0, map, inverse_orient_cam_per_frame, track_rep, nullptr, &intrinsic_cam_mat_per_frame, term_crit);

    LOG(INFO) << "bundle adjustment finished with result: " << op << " (" << ba.OptimizationStatusString() << ")" << endl;
    return 0;
}
}
