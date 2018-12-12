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
#include "visualize-helpers.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/imgproc.hpp> // cv::circle
#include <opencv2/highgui.hpp> // cv::imshow
#endif

#if defined(SRK_HAS_PANGOLIN)
#include <pangolin/pangolin.h>
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

void GenerateCameraShotsAlongRectangularPath(const WorldBounds& wb, size_t steps_per_side_x, size_t steps_per_side_y,
    Scalar viewer_down_offset, Scalar ascentZ, std::vector<SE3Transform>* inverse_orient_cams)
{
    std::array<suriko::Point3,5> look_at_base_points = {
        suriko::Point3(wb.x_max, wb.y_min, wb.z_min),
        suriko::Point3(wb.x_min, wb.y_min, wb.z_min),
        suriko::Point3(wb.x_min, wb.y_max, wb.z_min),
        suriko::Point3(wb.x_max, wb.y_max, wb.z_min),
        suriko::Point3(wb.x_max, wb.y_min, wb.z_min),
    };

    // number of viewer steps per each side is variable
    std::array<size_t, 4> viewer_steps_per_side = {
        steps_per_side_x,
        steps_per_side_y,
        steps_per_side_x,
        steps_per_side_y
    };

    Scalar skew_ang = (Scalar)std::atan2(std::abs(wb.x_max - wb.x_min), std::abs(wb.y_max - wb.y_min));
    Scalar viewer_offsetX = viewer_down_offset * std::sin(skew_ang);
    Scalar viewer_offsetY = -viewer_down_offset * std::cos(skew_ang);

    for (size_t base_point_ind = 0; base_point_ind < look_at_base_points.size()-1; ++base_point_ind)
    {
        suriko::Point3 base1 = look_at_base_points[base_point_ind];
        suriko::Point3 base2 = look_at_base_points[base_point_ind+1];
        size_t steps_per_side = viewer_steps_per_side[base_point_ind];

        Eigen::Matrix<Scalar, 3, 1> step = (base2.Mat() - base1.Mat()) / steps_per_side;
        
        for (size_t step_ind=0; step_ind<steps_per_side; ++step_ind)
        {
            suriko::Point3 cur_point = suriko::Point3(base1.Mat() + step * step_ind);

            auto wfc = LookAtLufWfc(
                cur_point.Mat() + Eigen::Matrix<Scalar, 3, 1>(viewer_offsetX, viewer_offsetY, ascentZ),
                cur_point.Mat(),
                Eigen::Matrix<Scalar, 3, 1>(0, 0, 1));

            // convert XYZ=LUF (left-up-forward) to XYZ=RDF (right-down-forward)
            wfc.R *= RotMat(0, 0, 1, M_PI);

            SE3Transform RT = SE3Inv(wfc);

            // now camera is directed x-right, y-bottom, z-forward
            inverse_orient_cams->push_back(RT);
        }
    }
}

#if defined(SRK_HAS_PANGOLIN)
/// Draw axes in the local coordinates.
void DrawAxes(Scalar axis_seg_len)
{
    auto ax = axis_seg_len;
    glLineWidth(2);
    glBegin(GL_LINES);
    glColor3d(1, 0, 0);
    glVertex3d(0, 0, 0);
    glVertex3d(ax, 0, 0); // OX
    glColor3d(0, 1, 0);
    glVertex3d(0, 0, 0);
    glVertex3d(0, ax, 0); // OY
    glColor3d(0, 0, 1);
    glVertex3d(0, 0, 0);
    glVertex3d(0, 0, ax); // OZ
    glEnd();
}

void DrawPhysicalCamera(const SE3Transform& cam_wfc)
{
    // transform to the camera frame
    std::array<double, 4 * 4> opengl_mat_by_col{};
    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::ColMajor>> opengl_mat(opengl_mat_by_col.data());
    opengl_mat.topLeftCorner<3, 3>() = cam_wfc.R.cast<double>();
    opengl_mat.topRightCorner<3, 1>() = cam_wfc.T.cast<double>();
    opengl_mat(3, 3) = 1;

    glPushMatrix();
    glMultMatrixd(opengl_mat_by_col.data());

    constexpr Scalar ax = 0.4;
    DrawAxes(ax);

    // draw camera in the local coordinates
    constexpr Scalar hw = ax / 3; // halfwidth
    //constexpr std::array<std::array<double, 3>, 5> cam_skel = {
    //    {0, 0, 0},
    //    {hw, hw, ax}, // left top
    //    {-hw, hw, ax }, // right top
    //    {-hw, -hw, ax}, // right bot
    //    {hw, -hw, ax}, // left bot
    //};
    
    constexpr double cam_skel[5][3] = {
        {0, 0, 0},
        {hw, hw, ax}, // left top
        {-hw, hw, ax }, // right top
        {-hw, -hw, ax}, // right bot
        {hw, -hw, ax}, // left bot
    };

    glColor3f(1, 1, 1);

    glLineWidth(1);
    glBegin(GL_LINE_LOOP); // left top of the front plane of the camera
    glVertex3dv(cam_skel[1]);
    glVertex3dv(cam_skel[2]);
    glVertex3dv(cam_skel[3]);
    glVertex3dv(cam_skel[4]);
    glEnd();

    glBegin(GL_LINES); // edges from center to the front plane
    glVertex3dv(cam_skel[0]);
    glVertex3dv(cam_skel[1]);
    glVertex3dv(cam_skel[0]);
    glVertex3dv(cam_skel[2]);
    glVertex3dv(cam_skel[0]);
    glVertex3dv(cam_skel[3]);
    glVertex3dv(cam_skel[0]);
    glVertex3dv(cam_skel[4]);
    glEnd();
    glPopMatrix();
}

void DrawCameras(const std::vector<SE3Transform>& cam_orient_cfw, bool draw_camera_each_frame = false)
{
    Eigen::Matrix<Scalar, 3, 1> cam_pos_world_prev;
    bool cam_pos_world_prev_inited = false;

    for (const auto& cam_cfw : cam_orient_cfw)
    {
        const SE3Transform& cam_wfc = SE3Inv(cam_cfw);

        // get position of the camera in the world: cam_to_world*(0,0,0,1)=cam_pos
        const Eigen::Matrix<Scalar, 3, 1>& cam_pos_world = cam_wfc.T;

        if (cam_pos_world_prev_inited)
        {
            glBegin(GL_LINES);
            glColor3f(1, 1, 1);
            glVertex3d(cam_pos_world_prev[0], cam_pos_world_prev[1], cam_pos_world_prev[2]);
            glVertex3d(cam_pos_world[0], cam_pos_world[1], cam_pos_world[2]);
            glEnd();
        }

        glBegin(GL_POINTS);
        glVertex3d(cam_pos_world[0], cam_pos_world[1], cam_pos_world[2]);
        glEnd();

        if (draw_camera_each_frame)
            DrawPhysicalCamera(cam_wfc);

        cam_pos_world_prev = cam_pos_world;
        cam_pos_world_prev_inited = true;
    }
}

void DrawMap(const FragmentMap& fragment_map)
{
    glColor3d(0.7, 0.7, 0.7);
    glBegin(GL_POINTS);
    for (const SalientPointFragment& point : fragment_map.SalientPoints())
    {
        if (!point.coord.has_value()) continue;
        const suriko::Point3& p = point.coord.value();
        glVertex3d(p.Mat()[0], p.Mat()[1], p.Mat()[2]);
    }
    glEnd();
}

void DrawScene(const std::vector<SE3Transform>& cam_orient_cfw, const FragmentMap& fragment_map)
{
    DrawAxes(0.5);

    DrawMap(fragment_map);

    DrawCameras(cam_orient_cfw, true);
}

struct UIThreadParams
{
    MultiViewIterativeFactorizer* mvf;
};

void SceneVisualizationThread(UIThreadParams ui_params)
{
    VLOG(4) << "SceneVisualizationThread is running";

    constexpr float fw = 640;
    constexpr float fh = 480;
    constexpr int w = (int)fw;
    constexpr int h = (int)fh;

    pangolin::CreateWindowAndBind("3DReconstr", w, h);
    glEnable(GL_DEPTH_TEST);

    float center_x = fw / 2;
    float center_y = fh / 2;

    pangolin::OpenGlRenderState view_state_3d(
        pangolin::ProjectionMatrix(w, h, 420, 420, center_x, center_y, 0.2, 100),
        pangolin::ModelViewLookAt(30, -30, 30, 0, 0, 0, pangolin::AxisZ)
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(view_state_3d);
    pangolin::View& display_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -fw / fh) // TODO: why negative aspect?
        .SetHandler(&handler);

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        display_cam.Activate(view_state_3d);

        std::shared_lock<std::shared_mutex> lock(ui_params.mvf->location_and_map_mutex_);
        DrawScene(ui_params.mvf->cam_orient_cfw_, ui_params.mvf->map_);

        pangolin::FinishFrame();

        std::this_thread::sleep_for(100ms); // make ui thread more 'lightweight'
    }

    VLOG(4) << "SceneVisualizationThread is exiting";
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
        // determine current camerra's orientation using the ground truth
        const SE3Transform& rt_cfw = gt_cam_orient_cfw_[frame_ind];

        // determine which salient points are visible
        for (const SalientPointFragment& fragment : entire_map_.SalientPoints())
        {
            const Point3& salient_point = fragment.coord.value();
            Eigen::Matrix<Scalar, 3, 1> pnt_homog = ProjectPnt(K_, rt_cfw, salient_point);
            SRK_ASSERT(!IsClose(0, pnt_homog[2], 10e-5)) << "points in infinity are unsupported";

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
            if (fragment.synthetic_virtual_point_id.has_value())
            {
                // determine points correspondance using synthatic ids
                track_rep->GetFirstPointTrackByFragmentSyntheticId(fragment.synthetic_virtual_point_id.value(), &corner_track);
            }
            if (corner_track == nullptr)
            {
                suriko::CornerTrack& new_corner_track = track_rep->AddCornerTrackObj();
                SRK_ASSERT(!new_corner_track.SalientPointId.has_value()) << "new track is not associated with any reconstructed salient point";

                new_corner_track.SyntheticVirtualPointId = fragment.synthetic_virtual_point_id;

                corner_track = &new_corner_track;
            }

            suriko::Point2 pix(pix_x, pix_y);

            CornerData& corner_data = corner_track->AddCorner(frame_ind);
            corner_data.pixel_coord = pix;
            corner_data.image_coord = K_inv_ * pix.AsHomog();
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
DEFINE_int32(viewer_steps_per_side_x, 10, "number of viewer's steps at each side of the rectangle");
DEFINE_int32(viewer_steps_per_side_y, 10, "number of viewer's steps at each side of the rectangle");
DEFINE_double(noise_R_std, 0.005, "Standard deviation of noise distribution for R, 0=no noise (eg: 0.01)");
DEFINE_double(noise_x3D_std, 0.005, "Standard deviation of noise distribution for salient points, 0=no noise (eg: 0.1)");
DEFINE_int32(wait_key_delay, 1, "parameter to cv::waitKey; 0 means 'wait forever'");
DEFINE_bool(wait_after_each_frame, true, "true to wait for keypress after each iteration");
DEFINE_bool(debug_skim_over, true, "overview the synthetic world without reconstruction");
DEFINE_bool(fake_mapping, false, "");
DEFINE_bool(fake_localization, false, "");

int MultiViewFactorizationDemo(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);

    LOG(INFO) << "noise_x3D_std=" << FLAGS_noise_x3D_std;
    LOG(INFO) << "noise_R_std=" << FLAGS_noise_R_std;

    //
    bool corrupt_salient_points_with_noise = FLAGS_noise_x3D_std > 0;
    bool corrupt_cam_orient_with_noise = FLAGS_noise_R_std > 0;
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

    constexpr Scalar inclusive_gap = 1e-8; // small value to make iteration inclusive

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    gen.seed(1234);

    std::unique_ptr<std::normal_distribution<Scalar>> x3D_noise_dis;
    if (corrupt_salient_points_with_noise)
        x3D_noise_dis = std::make_unique<std::normal_distribution<Scalar>>(0, FLAGS_noise_x3D_std);

    size_t next_virtual_point_id = 6000'000 + 1;
    FragmentMap entire_map;
    entire_map.SetFragmentIdOffsetInternal(1000'000);
    Scalar xmid = (wb.x_min + wb.x_max) / 2;
    Scalar xlen = wb.x_max - wb.x_min;
    Scalar zlen = wb.z_max - wb.z_min;
    for (Scalar grid_x = wb.x_min; grid_x < wb.x_max + inclusive_gap; grid_x += cell_size[0])
    {
        for (Scalar grid_y = wb.y_min; grid_y < wb.y_max + inclusive_gap; grid_y += cell_size[1])
        {
            Scalar x = grid_x;
            Scalar y = grid_y;
            
            Scalar z_perc = std::cos((x - xmid) / xlen * M_PI);
            Scalar z = wb.z_min + z_perc * zlen;

            // jit x and y so the points can be distinguished during movement
            if (corrupt_salient_points_with_noise)
            {
                x += (*x3D_noise_dis)(gen);
                y += (*x3D_noise_dis)(gen);
                z += (*x3D_noise_dis)(gen);
            }

            SalientPointFragment& frag = entire_map.AddSalientPointPatch(Point3(x, y, z));
            frag.synthetic_virtual_point_id = next_virtual_point_id++;
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

    GenerateCameraShotsAlongRectangularPath(wb, FLAGS_viewer_steps_per_side_x, FLAGS_viewer_steps_per_side_y, FLAGS_viewer_offset_down, FLAGS_viewer_ascendZ, &gt_cam_orient_cfw);

    if (corrupt_cam_orient_with_noise)
    {
        std::normal_distribution<Scalar> cam_orient_noise_dis(0, FLAGS_noise_R_std);
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
        if (entire_map.GetSalientPointByVirtualPointIdInternal(synthetic_virtual_point_id, &sal_pnt) && sal_pnt->coord.has_value())
        {
            const suriko::Point3& pnt_world = sal_pnt->coord.value();
            return pnt_world;
        }
        AssertFalse();
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
#if defined(SRK_HAS_PANGOLIN)
    UIThreadParams ui_params {};
    ui_params.mvf = &mvf;
    std::thread ui_thread(SceneVisualizationThread, ui_params);
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
        auto project_fun = [&K, &rt_cfw](const suriko::Point3& sal_pnt) -> Eigen::Matrix<suriko::Scalar, 3, 1>
        {
            return ProjectPnt(K, rt_cfw, sal_pnt);
        };
        constexpr Scalar f0 = 1;
        Draw2DProjectedAxes(f0, project_fun, &camera_image_rgb);
#endif
        size_t new_points_per_frame_count = 0;
        size_t new_track_per_frame_count = 0;
        std::set<size_t> entire_fragment_id_per_frame;

        // determine which salient points are visible
        for (const SalientPointFragment& fragment : entire_map.SalientPoints())
        {
            const Point3& salient_point = fragment.coord.value();
            Eigen::Matrix<Scalar, 3, 1> pnt_homog = ProjectPnt(K, rt_cfw, salient_point);

            auto pnt_pix = Eigen::Matrix<Scalar, 2, 1>(pnt_homog[0] / pnt_homog[2], pnt_homog[1] / pnt_homog[2]);

            Scalar pix_x = pnt_pix[0];
            Scalar pix_y = pnt_pix[1];
            bool hit_wnd =
                pix_x >= 0 && pix_x < (Scalar)img_size[0] &&
                pix_y >= 0 && pix_y < (Scalar)img_size[1];
            if (!hit_wnd)
                continue;

            entire_fragment_id_per_frame.insert(fragment.synthetic_virtual_point_id.value());

#if defined(SRK_HAS_OPENCV)
            camera_image_rgb.at<cv::Vec3b>((int)pix_y, (int)pix_x) = cv::Vec3b(0xFF, 0xFF, 0xFF);
#endif

            if (FLAGS_debug_skim_over || frame_ind < well_known_frames_count)
            {
                CornerTrack* corner_track = nullptr;
                mvf.track_rep_.GetFirstPointTrackByFragmentSyntheticId(fragment.synthetic_virtual_point_id.value(), &corner_track);

                if (corner_track == nullptr)
                {
                    // add new salient points
                    size_t salient_point_id = 0;
                    SalientPointFragment& new_frag = mvf.map_.AddSalientPointPatch(fragment.coord, &salient_point_id);
                    new_frag.synthetic_virtual_point_id = fragment.synthetic_virtual_point_id; // force id of subset fragment to be identical to fragment id from entire map

                    new_points_per_frame_count += 1;

                    //
                    suriko::CornerTrack& new_corner_track = mvf.track_rep_.AddCornerTrackObj();
                    new_corner_track.SalientPointId = salient_point_id;
                    new_corner_track.SyntheticVirtualPointId = fragment.synthetic_virtual_point_id;

                    corner_track = &new_corner_track;

                    new_track_per_frame_count += 1;
                }

                suriko::Point2 pix(pix_x, pix_y);

                CornerData& corner_data = corner_track->AddCorner(frame_ind);
                corner_data.pixel_coord = pix;
                corner_data.image_coord = K_inv * pix.AsHomog();
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
            << " points_count=" << mvf.map_.SalientPointsCount()
            << " tracks_count=" << mvf.track_rep_.CornerTracks.size()
            << " ncd=" << new_points.size() << "-" << common_points.size() << "-" << del_points.size();

#if defined(SRK_HAS_OPENCV)
        cv::imshow("front-camera", camera_image_rgb);
        if (FLAGS_wait_after_each_frame)
            cv::waitKey(FLAGS_wait_key_delay); // 0=wait forever
        else
            cv::waitKey(1); // wait for a moment to allow OpenCV to redraw the image
#endif
        if (frame_ind < well_known_frames_count)
            mvf.LogReprojError();

        // process the remaining frames
        if (!FLAGS_debug_skim_over && frame_ind >= well_known_frames_count)
        {
            bool op = mvf.IntegrateNewFrameCorners(rt_cfw);
        }
    }
#if defined(SRK_HAS_PANGOLIN)
    ui_thread.join();
#endif

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
