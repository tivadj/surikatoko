#include <iostream>
#include <vector>
#include <map>
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
#include "suriko/multi-view-factorization.h"
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

void GenerateCameraShotsAlongRectangularPath(const WorldBounds& wb, size_t steps_per_side, 
    Scalar viewer_down_offset, Scalar ascentZ, std::vector<SE3Transform>* inverse_orient_cams)
{
    std::vector<suriko::Point3> look_at_base_points = {
        suriko::Point3(wb.XMax, wb.YMin, wb.ZMin),
        suriko::Point3(wb.XMin, wb.YMin, wb.ZMin),
        suriko::Point3(wb.XMin, wb.YMax, wb.ZMin),
        suriko::Point3(wb.XMax, wb.YMax, wb.ZMin),
        suriko::Point3(wb.XMax, wb.YMin, wb.ZMin),
    };

    Scalar skew_ang = (Scalar)std::atan2(std::abs(wb.XMax - wb.XMin), std::abs(wb.YMax - wb.YMin));
    Scalar viewer_offsetX = viewer_down_offset * std::sin(skew_ang);
    Scalar viewer_offsetY = -viewer_down_offset * std::cos(skew_ang);

    for (size_t base_point_ind = 0; base_point_ind < look_at_base_points.size()-1; ++base_point_ind)
    {
        suriko::Point3 base1 = look_at_base_points[base_point_ind];
        suriko::Point3 base2 = look_at_base_points[base_point_ind+1];

        Eigen::Matrix<Scalar, 3, 1> step = (base2.Mat() - base1.Mat()) / steps_per_side;
        
        for (size_t step_ind=0; step_ind<steps_per_side; ++step_ind)
        {
            suriko::Point3 cur_point = suriko::Point3(base1.Mat() + step * step_ind);

            // X is directed to the right, Y - to up
            Eigen::Matrix<Scalar, 4, 4> cam_from_world = Eigen::Matrix<Scalar, 4, 4>::Identity();

            // translate to 'look to' point
            Eigen::Matrix<Scalar, 3, 1> shift = cur_point.Mat();

            // shift viewer aside
            shift[0] += viewer_offsetX;
            shift[1] += viewer_offsetY;
            shift[2] += ascentZ;

            // minus due to inverse camera orientation (conversion from world to camera)
            cam_from_world = SE3Mat(Eigen::Matrix<Scalar, 3, 1>(-shift)) * cam_from_world;

            // rotate OY around OZ so that OY points towards center in horizontal plane OZ=ascentZ
            Eigen::Matrix<Scalar, 3, 1> oz(0, 0, 1);
            cam_from_world = SE3Mat(RotMat(oz, -skew_ang)) * cam_from_world;

            // look down towards the center 
            Scalar look_down_ang = std::atan2(ascentZ, viewer_down_offset);

            // +pi/2 to direct not y-forward and z-up but z-forward and y-bottom
            cam_from_world = SE3Mat(RotMat(1, 0, 0, look_down_ang + M_PI / 2)) * cam_from_world;
            SE3Transform RT(cam_from_world.topLeftCorner(3, 3), cam_from_world.topRightCorner(3, 1));

            // now camera is directed x-right, y-bottom, z-forward
            inverse_orient_cams->push_back(RT);
        }
    }
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

class DemoCornersMatcher : public CornersMatcherBase
{
    const std::vector<SE3Transform>& gt_cam_orient_cfw_;
    const FragmentMap& entire_map_;
    Eigen::Matrix<Scalar, 3, 3> K_;
    Eigen::Matrix<Scalar, 3, 3> K_inv_;
    std::array<size_t, 2> img_size_;
public:
    DemoCornersMatcher(const Eigen::Matrix<Scalar, 3, 3>& K, const std::vector<SE3Transform>& gt_cam_orient_cfw, const FragmentMap& entire_map,
        const std::array<size_t, 2>& img_size)
        : K_(K),
        gt_cam_orient_cfw_(gt_cam_orient_cfw),
        entire_map_(entire_map),
        img_size_(img_size)
    {
        K_inv_ = K_.inverse();
    }

    void DetectAndMatchCorners(size_t frame_ind, CornerTrackRepository* track_rep) override
    {
        const SE3Transform& rt_cfw = gt_cam_orient_cfw_[frame_ind];

        // determine which salient points are visible
        for (const SalientPointFragment& fragment : entire_map_.SalientPoints())
        {
            const Point3& salient_point = fragment.Coord.value();
            Eigen::Matrix<Scalar, 3, 1> pnt_homog = ProjectPnt(K_, rt_cfw, salient_point);

            auto pnt_pix = Eigen::Matrix<Scalar, 2, 1>(pnt_homog[0] / pnt_homog[2], pnt_homog[1] / pnt_homog[2]);

            Scalar pix_x = pnt_pix[0];
            Scalar pix_y = pnt_pix[1];
            bool hit_wnd =
                pix_x >= 0 && pix_x < (Scalar)img_size_[0] &&
                pix_y >= 0 && pix_y < (Scalar)img_size_[1];
            if (!hit_wnd)
                continue;

            // now, the point is visible in current frame

            CornerTrack* corner_track = nullptr;
            if (fragment.SyntheticVirtualPointId.has_value())
            {
                track_rep->GetFirstPointTrackByFragmentSyntheticId(fragment.SyntheticVirtualPointId.value(), &corner_track);
            }
            if (corner_track == nullptr)
            {
                suriko::CornerTrack& new_corner_track = track_rep->AddCornerTrackObj();
                SRK_ASSERT(!new_corner_track.SalientPointId.has_value()) << "new track is not associated with any reconstructed salient point";

                new_corner_track.SyntheticVirtualPointId = fragment.SyntheticVirtualPointId;

                corner_track = &new_corner_track;
            }

            suriko::Point2 pix(pix_x, pix_y);

            CornerData& corner_data = corner_track->AddCorner(frame_ind);
            corner_data.PixelCoord = pix;
            corner_data.ImageCoord = K_inv_ * pix.AsHomog();
        }
    }
};

DEFINE_double(world_xmin, -1, "world xmin");
DEFINE_double(world_xmax, 1, "world xmax");
DEFINE_double(world_ymin, -1, "world ymin");
DEFINE_double(world_ymax, 1, "world ymax");
DEFINE_double(world_zmin, 0, "world zmin");
DEFINE_double(world_zmax, 1, "world zmax");
DEFINE_double(world_cell_size_x, 0.5, "cell size x");
DEFINE_double(world_cell_size_y, 0.5, "cell size y");
DEFINE_double(world_cell_size_z, 0.5, "cell size z");
DEFINE_double(viewer_offset_down, 10, "viewer's offset from viewed point in the down direction");
DEFINE_double(viewer_ascendZ, 10, "viewer's offset in the up direction");
DEFINE_int32(viewer_steps_per_side, 10, "number of viewer's steps at each side of the rectangle");
DEFINE_double(noise_R_hi, 0.005, "Upper bound of noise uniform distribution for R, 0=no noise (eg: 0.01)");
DEFINE_double(noise_x3D_hi, 0.005, "Upper bound of noise uniform distribution for salient points, 0=no noise (eg: 0.1)");
DEFINE_int32(wait_key_delay, 1, "parameter to cv::waitKey; 0 means 'wait forever'");
DEFINE_bool(debug_skim_over, true, "overview the synthetic world without reconstruction");
DEFINE_bool(fake_mapping, false, "");
DEFINE_bool(fake_localization, false, "");

int MultiViewFactorizationDemo(int argc, char* argv[])
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

    std::normal_distribution<Scalar> noise_dis(0, 0.15/3);

    size_t next_virtual_point_id = 6000'000 + 1;
    FragmentMap entire_map;
    entire_map.SetFragmentIdOffsetInternal(1000'000);
    Scalar xmid = (wb.XMin + wb.XMax) / 2;
    Scalar xlen = wb.XMax - wb.XMin;
    Scalar zlen = wb.ZMax - wb.ZMin;
    for (Scalar x = wb.XMin; x < wb.XMax + inclusive_gap; x += cell_size[0])
    {
        for (Scalar y = wb.YMin; y < wb.YMax + inclusive_gap; y += cell_size[1])
        {
            Scalar x_act = x;
            Scalar y_act = y;

            // jit x and y so the points can be distinguished during movement
            if (corrupt_salient_points_with_noise)
            {
                x_act += noise_dis(gen);
                y_act += noise_dis(gen);
            }

            Scalar val_z = std::cos((x_act - xmid) / xlen * M_PI);
            Scalar z = wb.ZMin + val_z * zlen;
            SalientPointFragment& frag = entire_map.AddSalientPointNew3(Point3(x_act, y_act, z));
            frag.SyntheticVirtualPointId = next_virtual_point_id++;
        }
    }

    LOG(INFO) << "points_count=" << entire_map.SalientPointsCount();

    // Numerical stability scaler, chosen so that x_pix / f0 and y_pix / f0 is close to 1
    std::array<size_t, 2> img_size = { 800, 600 };
    Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> K;
    K <<
        880, 0, img_size[0] / 2.0,
        0, 660, img_size[1] / 2.0,
        0, 0, 1;
    Eigen::Matrix<Scalar, 3, 3> K_inv = K.inverse();

    GenerateCameraShotsAlongRectangularPath(wb, FLAGS_viewer_steps_per_side, FLAGS_viewer_offset_down, FLAGS_viewer_ascendZ, &gt_cam_orient_cfw);

    size_t frames_count = gt_cam_orient_cfw.size();
    LOG(INFO) << "frames_count=" << frames_count;

    std::vector<std::set<size_t>> entire_map_fragment_id_per_frame;

    MultiViewIterativeFactorizer mvf;
    mvf.map_.SetFragmentIdOffsetInternal(2000'000); // not necessary
    mvf.K_ = K;
    mvf.K_inv_ = K_inv;
    mvf.gt_cam_orient_world_to_f_ = [&gt_cam_orient_cfw](size_t f) -> SE3Transform
    {
        SE3Transform c = gt_cam_orient_cfw[f];
        return c;
    };    
    mvf.gt_cam_orient_f1f2_ = [&gt_cam_orient_cfw](size_t f0, size_t f1) -> SE3Transform
    {
        SE3Transform c0 = gt_cam_orient_cfw[f0];
        SE3Transform c1 = gt_cam_orient_cfw[f1];
        SE3Transform c1_from_c0 = SE3AFromB(c1, c0);
        return c1_from_c0;
    };
    mvf.gt_salient_point_by_virtual_point_id_fun_ = [&entire_map](size_t synthetic_virtual_point_id) -> suriko::Point3
    {
        const SalientPointFragment* sal_pnt = nullptr;
        if (entire_map.GetSalientPointByVirtualPointIdInternal(synthetic_virtual_point_id, &sal_pnt) && sal_pnt->Coord.has_value())
        {
            const suriko::Point3& pnt_world = sal_pnt->Coord.value();
            return pnt_world;
        }
        CHECK(false);
    };
    mvf.SetCornersMatcher(std::make_unique<DemoCornersMatcher>(K, gt_cam_orient_cfw, entire_map, img_size));
    mvf.fake_localization_ = FLAGS_fake_localization;
    mvf.fake_mapping_ = FLAGS_fake_mapping;

    struct ComparePointsInfo
    {
        size_t NewPoints = 0;
        size_t CommonPoints = 0;
        size_t DeletedPoints = 0;

        ComparePointsInfo(size_t new_points, size_t common_points, size_t deleted_points)
            : NewPoints(new_points),
              CommonPoints(common_points),
              DeletedPoints(deleted_points)
        {
        }
    };
    std::vector<ComparePointsInfo> compare_cur_frame_to_prev_frames;
    
#if defined(SRK_HAS_OPENCV)
    cv::Mat camera_image_rgb = cv::Mat::zeros((int)img_size[1], (int)img_size[0], CV_8UC3);
#endif
    constexpr size_t well_known_frames_count = 2;
    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        // orient camera
        const SE3Transform& rt_cfw = gt_cam_orient_cfw[frame_ind];
        if (FLAGS_debug_skim_over || frame_ind < well_known_frames_count)
            mvf.cam_orient_cfw_.push_back(rt_cfw);

#if defined(SRK_HAS_OPENCV)
        camera_image_rgb.setTo(0);
        DrawAxes(K, rt_cfw, camera_image_rgb);
#endif
        size_t new_points_per_frame_count = 0;
        size_t new_track_per_frame_count = 0;
        std::set<size_t> entire_fragment_id_per_frame;

        // determine which salient points are visible
        for (const SalientPointFragment& fragment : entire_map.SalientPoints())
        {
            const Point3& salient_point = fragment.Coord.value();
            Eigen::Matrix<Scalar, 3, 1> pnt_homog = ProjectPnt(K, rt_cfw, salient_point);

            auto pnt_pix = Eigen::Matrix<Scalar, 2, 1>(pnt_homog[0] / pnt_homog[2], pnt_homog[1] / pnt_homog[2]);

            Scalar pix_x = pnt_pix[0];
            Scalar pix_y = pnt_pix[1];
            bool hit_wnd =
                pix_x >= 0 && pix_x < (Scalar)img_size[0] &&
                pix_y >= 0 && pix_y < (Scalar)img_size[1];
            if (!hit_wnd)
                continue;

            entire_fragment_id_per_frame.insert(fragment.SyntheticVirtualPointId.value());

#if defined(SRK_HAS_OPENCV)
            camera_image_rgb.at<cv::Vec3b>((int)pix_y, (int)pix_x) = cv::Vec3b(0xFF, 0xFF, 0xFF);
#endif

            if (FLAGS_debug_skim_over || frame_ind < well_known_frames_count)
            {
                CornerTrack* corner_track = nullptr;
                mvf.track_rep_.GetFirstPointTrackByFragmentSyntheticId(fragment.SyntheticVirtualPointId.value(), &corner_track);

                if (corner_track == nullptr)
                {
                    // add new salient points
                    size_t salient_point_id = 0;
                    SalientPointFragment& new_frag = mvf.map_.AddSalientPointNew3(fragment.Coord, &salient_point_id);
                    new_frag.SyntheticVirtualPointId = fragment.SyntheticVirtualPointId; // force id of subset fragment to be identical to fragment id from entire map

                    new_points_per_frame_count += 1;

                    //
                    suriko::CornerTrack& new_corner_track = mvf.track_rep_.AddCornerTrackObj();
                    new_corner_track.SalientPointId = salient_point_id;
                    new_corner_track.SyntheticVirtualPointId = fragment.SyntheticVirtualPointId;

                    corner_track = &new_corner_track;

                    new_track_per_frame_count += 1;
                }

                suriko::Point2 pix(pix_x, pix_y);

                CornerData& corner_data = corner_track->AddCorner(frame_ind);
                corner_data.PixelCoord = pix;
                corner_data.ImageCoord = K_inv * pix.AsHomog();
            }
        }

        //
        entire_map_fragment_id_per_frame.push_back(std::move(entire_fragment_id_per_frame));

        std::vector<size_t> new_points;
        std::vector<size_t> common_points;
        std::vector<size_t> del_points;
        if (frame_ind > 0)
        {
            const std::set<size_t>& s1 = entire_map_fragment_id_per_frame[frame_ind - 1];
            const std::set<size_t>& s2 = entire_map_fragment_id_per_frame[frame_ind];
            std::set_difference(s2.begin(), s2.end(), s1.begin(), s1.end(), std::back_inserter(new_points));
            std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(common_points));
            std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(del_points));
        }
        else
        {
            std::copy(entire_map_fragment_id_per_frame[0].begin(), entire_map_fragment_id_per_frame[0].end(), std::back_inserter(new_points));
        }

        compare_cur_frame_to_prev_frames.clear();
        for (size_t i=0; i<frame_ind; ++i)
        {
            new_points.clear();
            common_points.clear();
            del_points.clear();
            const std::set<size_t>& s1 = entire_map_fragment_id_per_frame[i];
            const std::set<size_t>& s2 = entire_map_fragment_id_per_frame[frame_ind];
            std::set_difference(s2.begin(), s2.end(), s1.begin(), s1.end(), std::back_inserter(new_points));
            std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(common_points));
            std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(del_points));
            compare_cur_frame_to_prev_frames.push_back(ComparePointsInfo(new_points.size(), common_points.size(), del_points.size()));
        }

        VLOG(4) << "f=" << frame_ind
            << " new_points=" << new_points_per_frame_count << " of total=" << mvf.map_.SalientPointsCount()
            << " new_tracks=" << new_track_per_frame_count << " of total=" << mvf.track_rep_.CornerTracks.size()
            << " ncd=" << new_points.size() << "-" << common_points.size() << "-" << del_points.size();

        //std::ostringstream ss;
        //ss << "new points (" << new_points.size() << "): ";
        //for (size_t pnt_id : new_points)
        //    ss << pnt_id << " ";
        //VLOG(4) << ss.str();

#if defined(SRK_HAS_OPENCV)
        cv::imshow("front-camera", camera_image_rgb);
        cv::waitKey(FLAGS_wait_key_delay); // 0=wait forever
#endif
        if (frame_ind < well_known_frames_count)
            mvf.LogReprojError();

        // process the remaining frames
        if (!FLAGS_debug_skim_over && frame_ind >= well_known_frames_count)
        {
            mvf.IntegrateNewFrameCorners(rt_cfw);
        }
    }

#if defined(SRK_HAS_OPENCV)
    cv::waitKey(0); // 0=wait forever
#endif
    return 0;
}
}

int main(int argc, char* argv[])
{
    int result = 0;
    result = suriko_demos::MultiViewFactorizationDemo(argc, argv);
    return result;
}
