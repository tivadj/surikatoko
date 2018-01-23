#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <cassert>
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
namespace suriko_demos
{
using namespace std;
//using namespace std::experimental::filesystem;
using namespace boost::filesystem;
using namespace suriko;

void PopulateCornersPerFrame(const vector<Scalar>& viff_data_by_row, size_t viff_num_rows, size_t viff_num_cols,
                             CornerTrackRepository *track_rep)
{
    size_t points_count = viff_num_rows;
    size_t frames_count = viff_num_cols/2;
    size_t next_track_id = 0;
    for (size_t pnt_ind = 0; pnt_ind < points_count; ++pnt_ind)
    {
        CornerTrack track;

        for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
        {
            size_t i = pnt_ind*viff_num_cols+frame_ind*2;
            auto x = viff_data_by_row[i];
            auto y = viff_data_by_row[i+1];
            if (x == -1 || y == -1) continue;

            track.AddCorner(frame_ind, suriko::Point2(x,y));
        }
        if (!track.HasCorners()) continue; // track without registered corners

        track.SyntheticSalientPointId = 10000 + pnt_ind;
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

int DinoDemo(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);

    const string& test_data = FLAGS_testdata;

	boost::filesystem::path test_data_path = boost::filesystem::absolute(test_data).normalize();
    LOG(INFO) <<"testdata=" << test_data_path;

    int debug = 3;

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

    size_t frames_count = P_num_rows / 3;
    LOG(INFO) <<"frames_count=" <<frames_count <<endl;

    vector<Eigen::Matrix<Scalar,3,4>> proj_mat_per_frame;
    vector<SE3Transform> inverse_orient_cam_per_frame;
    vector<Eigen::Matrix<Scalar, 3, 3>> intrinsic_cam_mat_per_frame; // K

    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        Eigen::Map<Eigen::Matrix<Scalar,3,4,Eigen::RowMajor>> proj_mat_row_major(P_data_by_row.data()+frame_ind*12, 3, 4);
        Eigen::Matrix<Scalar,3,4> proj_mat = proj_mat_row_major;
        proj_mat_per_frame.push_back(proj_mat);

        Scalar scale_factor;
        Eigen::Matrix<Scalar,3,3> K;
        SE3Transform direct_orient_cam;
        std::tie(scale_factor, K, direct_orient_cam) = DecomposeProjMat(proj_mat);

        intrinsic_cam_mat_per_frame.push_back(K);

        auto inverse_orient_cam = SE3Inv(direct_orient_cam);
        inverse_orient_cam_per_frame.push_back(inverse_orient_cam);
    }

    auto viff_mats_file_path = (test_data_path / "oxfvisgeom/dinosaur/viff.xy").normalize();
    vector<Scalar> viff_data_by_row;
    size_t viff_num_rows, viff_num_cols;
    op= ReadMatrixFromFile(viff_mats_file_path, ' ', &viff_data_by_row, &viff_num_rows, &viff_num_cols, &err_msg);
    if (!op)
    {
        LOG(ERROR) << err_msg;
        return 1;
    }

    if (frames_count != viff_num_cols / 2)
    {
        LOG(ERROR) << "Inconsistent frames_count";
        return 1;
    }

    size_t points_count = viff_num_rows; // =4983
    LOG(INFO) << "points_count=" <<points_count <<endl;

    CornerTrackRepository track_rep;
    PopulateCornersPerFrame(viff_data_by_row, viff_num_rows, viff_num_cols, &track_rep);

    vector<size_t> point_track_ids;
    track_rep.PopulatePointTrackIds(&point_track_ids);

    // triangulate 3D points
    vector<Point2> one_pnt_corner_per_frame;
    vector<Eigen::Matrix<Scalar,3,4>> one_pnt_proj_mat_per_frame;
    FragmentMap map;
    for (size_t pnt_track_id : point_track_ids)
    {
        const auto& corner_track = track_rep.GetPointTrackById(pnt_track_id);

        one_pnt_corner_per_frame.clear();
        one_pnt_proj_mat_per_frame.clear();
        for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
        {
            optional<Point2> corner = corner_track.GetCorner(frame_ind);
            if (!corner.has_value())
                continue;
            one_pnt_corner_per_frame.push_back(corner.value());
            one_pnt_proj_mat_per_frame.push_back(proj_mat_per_frame[frame_ind]);
        }
        Scalar f0 = 1;
        Point3 x3D = Triangulate3DPointByLeastSquares(one_pnt_corner_per_frame, one_pnt_proj_mat_per_frame, f0, debug);
        map.AddSalientPoint(pnt_track_id, x3D);
    }

    auto err_initial = BundleAdjustmentKanatani::ReprojError(map, inverse_orient_cam_per_frame, track_rep, nullptr, &intrinsic_cam_mat_per_frame);
    LOG(INFO) <<"err_initial=" <<err_initial <<endl;

	bool debug_reproj_err = false;
    if (debug_reproj_err)
    {
        for (size_t point_track_id : point_track_ids) {
            const auto &point_track = track_rep.GetPointTrackById(point_track_id);

            Point3 x3D = map.GetSalientPoint(point_track_id);
            Eigen::Matrix<Scalar,4,1> x3D_homog(x3D[0], x3D[1], x3D[2], 1);

            for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind) {
                const auto &corner = point_track.GetCorner(frame_ind);
                if (!corner.has_value())
                    continue;

                const auto& cor = corner.value();

                const auto &P = proj_mat_per_frame[frame_ind];
                auto x2D_homog = P * x3D_homog;
                auto pix_x = x2D_homog[0] / x2D_homog[2];
                auto pix_y = x2D_homog[1] / x2D_homog[2];
                auto err = (cor.Mat() - Eigen::Matrix<Scalar,2,1>(pix_x, pix_y)).norm();
                if (err > 1)
                    LOG(INFO) << "repr err=" <<err <<" for point_track_id=" << point_track_id <<endl;
            }
        }
    }

    bool check_derivatives = true;

    BundleAdjustmentKanatani ba;

    LOG(INFO) << "start bundle adjustment..." <<endl;
    op = ba.ComputeInplace(map, inverse_orient_cam_per_frame, track_rep, nullptr, &intrinsic_cam_mat_per_frame, check_derivatives);

    return 0;
}
}
