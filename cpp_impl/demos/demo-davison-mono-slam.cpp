#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <functional>
#include <utility>
#include <cassert>
#include <cmath>
#include <corecrt_math_defines.h>
#include <random>
#include <tuple>
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
#include "stat-helpers.h"
#include "visualize-helpers.h"
#include "suriko/quat.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/imgproc.hpp> // cv::circle
#include <opencv2/highgui.hpp> // cv::imshow
#endif

#if defined(SRK_HAS_PANGOLIN)
#include <pangolin/pangolin.h>
#endif

namespace suriko_demos_davison_mono_slam
{
using namespace std;
using namespace boost::filesystem;
using namespace suriko;
using namespace suriko::internals;
using namespace suriko::virt_world;

void GenerateCameraShotsAlongRectangularPath(const WorldBounds& wb, size_t steps_per_side_x, size_t steps_per_side_y,
    Scalar viewer_down_offset, Scalar ascentZ, std::vector<SE3Transform>* inverse_orient_cams)
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

    Scalar skew_ang = (Scalar)std::atan2(std::abs(wb.XMax - wb.XMin), std::abs(wb.YMax - wb.YMin));
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

            // camera orientation is XYZ=right-down-forward
            // convert it to XYZ=left-up-forward which is used by Hartley&Zisserman
            cam_from_world = SE3Mat(RotMat(0, 0, 1, -M_PI)) * cam_from_world;
            
            SE3Transform RT(cam_from_world.topLeftCorner(3, 3), cam_from_world.topRightCorner(3, 1));
            inverse_orient_cams->push_back(RT);
        }
    }
}

class DemoCornersMatcher : public CornersMatcherBase
{
    const std::vector<SE3Transform>& gt_cam_orient_cfw_;
    const FragmentMap& entire_map_;
    std::array<size_t, 2> img_size_;
    const DavisonMonoSlam* kalman_tracker_;
    bool suppress_observations_ = false; // true to make camera magically don't detect any salient points
public:
    DemoCornersMatcher(const DavisonMonoSlam* kalman_tracker, const std::vector<SE3Transform>& gt_cam_orient_cfw, const FragmentMap& entire_map,
        const std::array<size_t, 2>& img_size)
        : kalman_tracker_(kalman_tracker),
        gt_cam_orient_cfw_(gt_cam_orient_cfw),
        entire_map_(entire_map),
        img_size_(img_size)
    {
    }

    void DetectAndMatchCorners(size_t frame_ind, CornerTrackRepository* track_rep) override
    {
        if (suppress_observations_)
            return;

        // determine current camerra's orientation using the ground truth
        const SE3Transform& rt_cfw = gt_cam_orient_cfw_[frame_ind];

        // determine which salient points are visible
        for (const SalientPointFragment& fragment : entire_map_.SalientPoints())
        {
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

            // now, the point is visible in current frame

            CornerTrack* corner_track = nullptr;
            if (fragment.SyntheticVirtualPointId.has_value())
            {
                // determine points correspondance using synthatic ids
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
            corner_data.ImageCoord.setConstant(std::numeric_limits<Scalar>::quiet_NaN());
        }
    }

    void SetSuppressObservations(bool value) { suppress_observations_ = value; }
};

#if defined(SRK_HAS_PANGOLIN)
struct WorkerThreadChat
{
    std::mutex exit_ui_mutex;
    bool exit_ui_flag = false; // true to request UI thread to finish

    std::mutex exit_worker_mutex;
    std::condition_variable exit_worker_cv;
    bool exit_worker_flag = false; // true to request worker thread to stop processing

    std::mutex resume_worker_mutex;
    std::condition_variable resume_worker_cv;
    bool resume_worker_flag = true; // true for worker to do processing, false to pause and wait for resume request from UI
    bool resume_worker_suppress_observations = false; // true to 'cover' camera - no detections are made

    std::shared_mutex location_and_map_mutex_;
};

struct UIThreadParams
{
    DavisonMonoSlam* kalman_slam;
    Scalar ellipsoid_cut_thr;
    bool WaitForUserInputAfterEachFrame;
    std::function<size_t()> get_observable_frame_ind_fun;
    const std::vector<SE3Transform>* gt_cam_orient_cfw;
    const std::vector<SE3Transform>* cam_orient_cfw_history;
    const FragmentMap* entire_map;
    std::shared_ptr<WorkerThreadChat> worker_chat;
};

enum class CamDisplayType
{
    None,
    Dot, // camera is visualized as a dot
    Schematic // camera is visualized schematically
};

/// Draw axes in the local coordinates.
void RenderAxes(Scalar axis_seg_len)
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

void RenderSchematicCamera(const SE3Transform& cam_wfc, const std::array<float, 3>& track_color, CamDisplayType cam_disp_type)
{
    if (cam_disp_type == CamDisplayType::None)
        return;

    // transform to the camera frame
    std::array<double, 4 * 4> opengl_mat_by_col{};
    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::ColMajor>> opengl_mat(opengl_mat_by_col.data());
    opengl_mat.topLeftCorner<3, 3>() = cam_wfc.R.cast<double>();
    opengl_mat.topRightCorner<3, 1>() = cam_wfc.T.cast<double>();
    opengl_mat(3, 3) = 1;

    glPushMatrix();
    glMultMatrixd(opengl_mat_by_col.data());

    // draw camera in the local coordinates
    constexpr Scalar ax = 0.4;
    constexpr Scalar hw = ax / 3; // halfwidth
    constexpr double cam_skel[5][3] = {
        {0, 0, 0},
        {hw, hw, ax}, // left top
        {-hw, hw, ax }, // right top
        {-hw, -hw, ax}, // right bot
        {hw, -hw, ax}, // left bot
    };

    if (cam_disp_type == CamDisplayType::Dot)
    {
        glColor3fv(track_color.data());
        glPointSize(3);
        glBegin(GL_POINTS);
        glVertex3dv(cam_skel[0]);
        glEnd();
    }
    else if (cam_disp_type == CamDisplayType::Schematic)
    {
        RenderAxes(ax);

        glColor3fv(track_color.data());
        glLineWidth(1);

        // render camera schematically
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
    }
    glPopMatrix();
}

void PickRandomPointOnEllipsoid(
    const Eigen::Matrix<Scalar, 3, 1>& cam_pos,
    const Eigen::Matrix<Scalar, 3, 3>& cam_pos_uncert, Scalar ellipsoid_cut_thr,
    std::mt19937& gen, Eigen::Matrix<Scalar, 3, 1>* pos_ellipsoid)
{
    // choose random direction from the center of ellipsoid
    std::uniform_real_distribution<Scalar> distr(-1, 1);
    Eigen::Matrix<Scalar, 3, 1> ray;
    ray[0] = distr(gen);
    ray[1] = distr(gen);
    ray[2] = distr(gen);
    PickPointOnEllipsoid(cam_pos, cam_pos_uncert, ellipsoid_cut_thr, ray, pos_ellipsoid);
}

void RenderEllipsoid(
    const Eigen::Matrix<Scalar, 3, 1>& center,
    const Eigen::Matrix<Scalar, 3, 1>& semi_axes,
    const Eigen::Matrix<Scalar, 3, 3>& rot_mat_world_from_ellipse)
{
    std::array<std::pair<size_t, Scalar>, 3> sorted_semi_axes;
    sorted_semi_axes[0] = { 0, semi_axes[0] };
    sorted_semi_axes[1] = { 1, semi_axes[1] };
    sorted_semi_axes[2] = { 2, semi_axes[2] };
    std::sort(sorted_semi_axes.begin(), sorted_semi_axes.end(), [](auto& p1, auto& p2)
    {
        // sort descending by length of semi-axis
        return p1.second > p2.second;
    });

    // draw the ellipse which is the crossing of ellipsoid and the plane using the two largest semi-axes
    // polar ellipse
    glLineWidth(1);
    glBegin(GL_LINE_LOOP);
    size_t dots_per_ellipse = 12;
    std::array<Scalar, 2> big_semi_axes = { sorted_semi_axes[0].second,sorted_semi_axes[1].second };
    for (size_t i=0; i<dots_per_ellipse; ++i)
    {
        Scalar theta = i * (2 * M_PI) / dots_per_ellipse;
        Scalar cos_theta = std::cos(theta);
        Scalar sin_theta = std::sin(theta);

        // Polar ellipse https://en.wikipedia.org/wiki/Ellipse
        // r=a*b/sqrt((b*cos(theta))^2 + (a*sin(theta))^2)
        Scalar r = big_semi_axes[0] * big_semi_axes[1] / std::sqrt(suriko::Sqr(big_semi_axes[1]*cos_theta) + suriko::Sqr(big_semi_axes[0]*sin_theta));

        Eigen::Matrix<Scalar, 3, 1> wstmp;
        wstmp[0] = r * cos_theta;
        wstmp[1] = r * sin_theta;
        wstmp[2] = 0;

        Eigen::Matrix<Scalar, 3, 1> ws;
        // the largets semi-axis
        ws[sorted_semi_axes[0].first] = r * cos_theta;
        // second largets semi-axis
        ws[sorted_semi_axes[1].first] = r * sin_theta;
        // the smallest semi-axis
        ws[sorted_semi_axes[2].first] = 0;

        // map to the original ellipse with axes not parallel to world axes
        ws += center;
        
        Eigen::Matrix<Scalar, 3, 1> pos_world = rot_mat_world_from_ellipse * ws;
        glVertex3d(pos_world[0], pos_world[1], pos_world[2]);
    }
    glEnd();
}

void RenderUncertaintyEllipsoid(
    const Eigen::Matrix<Scalar, 3, 1>& pos,
    const Eigen::Matrix<Scalar, 3, 3>& pos_uncert,
    Scalar ellipsoid_cut_thr)
{
    Eigen::LLT<Eigen::Matrix<Scalar, 3, 3>> lltOfA(pos_uncert);
    bool op2 = lltOfA.info() != Eigen::NumericalIssue;
    if(!op2)
    {
        LOG(ERROR) << "failed lltOfA.info() != Eigen::NumericalIssue";
        return;
    }

    // uncertainty ellipsoid
    Eigen::Matrix<Scalar, 3, 3> A;
    Eigen::Matrix<Scalar, 3, 1> b;
    Scalar c;
    ExtractEllipsoidFromUncertaintyMat(pos, pos_uncert, ellipsoid_cut_thr, &A, &b, &c);

    Eigen::Matrix<Scalar, 3, 1> center;
    Eigen::Matrix<Scalar, 3, 1> semi_axes;
    Eigen::Matrix<Scalar, 3, 3> rot_mat_world_from_ellipse;
    bool op = GetRotatedEllipsoid(A, b, c, &center, &semi_axes, &rot_mat_world_from_ellipse);
    if (!op)
        return;

    // draw projection of ellipsoid
    RenderEllipsoid(center, semi_axes, rot_mat_world_from_ellipse);
}

bool RenderUncertaintyEllipsoidBySampling(const Eigen::Matrix<Scalar, 3, 1>& cam_pos,
    const Eigen::Matrix<Scalar, 3, 3>& cam_pos_uncert,
    Scalar ellipsoid_cut_thr, size_t ellipsoid_points_count = 256)
{
    Eigen::LLT<Eigen::Matrix<Scalar, 3, 3>> lltOfA(cam_pos_uncert);
    bool op2 = lltOfA.info() != Eigen::NumericalIssue;
    SRK_ASSERT(op2);

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    gen.seed(1234);

    Scalar var1 = std::max(cam_pos_uncert(0, 0), std::max(cam_pos_uncert(1, 1), cam_pos_uncert(2, 2)));
    Scalar sig1 = std::sqrt(var1);
    std::uniform_real_distribution<Scalar> distr(-3*sig1, 3*sig1);

    Eigen::Matrix<Scalar, 3, 3> uncert_inv = cam_pos_uncert.inverse();
    Scalar uncert_det = cam_pos_uncert.determinant();
    std::vector<Eigen::Matrix<Scalar, 3, 1>> ellips_points;
    for (size_t i = 0; i < ellipsoid_points_count; ++i)
    {
        // choose random direction from the center of ellipsoid
        Eigen::Matrix<Scalar, 3, 1> ray;
        ray[0] = distr(gen);
        ray[1] = distr(gen);
        ray[2] = distr(gen);

        // cross ellipsoid with ray
        Scalar b1 = -std::log(suriko::Sqr(ellipsoid_cut_thr)*suriko::Pow3(2*M_PI)*uncert_det);
        Eigen::Matrix<Scalar, 1, 1> b2 = ray.transpose() * uncert_inv * ray;
        Scalar t2 = b1 / b2[0];
        //SRK_ASSERT(t2 >= 0) << "invalid covariance matrix";
        if (t2 < 0)
            return false;

        Scalar t = std::sqrt(t2);

        // crossing of ellipsoid and ray
        Eigen::Matrix<Scalar, 3, 1> pos_world = cam_pos + t * ray;
        ellips_points.push_back(pos_world);
    }

    // draw lines from each point on the ellipsoid to the closest point
    for (size_t i=0; i<ellips_points.size(); ++i)
    {
        auto& p1 = ellips_points[i];
        Scalar closest_dist = 999;
        size_t closest_point_ind = -1;
        for (size_t j = 0; j<ellips_points.size(); ++j)
        {
            if (i == j) continue;
            auto& p2 = ellips_points[j];

            Scalar dist = (p2 - p1).norm();
            if (dist < closest_dist)
            {
                closest_point_ind = j;
                closest_dist = dist;
            }
        }

        glLineWidth(1);
        glBegin(GL_LINES);
        glVertex3d(p1[0], p1[1], p1[2]);
        
        auto& p2 = ellips_points[closest_point_ind];
        glVertex3d(p2[0], p2[1], p2[2]);
        glEnd();
    }
    return true;
}

void RenderCameraTrajectory(const std::vector<SE3Transform>& gt_cam_orient_cfw, 
    const std::array<float,3>& track_color,
    bool display_trajectory,
    CamDisplayType mid_cam_disp_type,
    CamDisplayType last_cam_disp_type)
{
    Eigen::Matrix<Scalar, 3, 1> cam_pos_world_prev;
    bool cam_pos_world_prev_inited = false;

    for (size_t i = 0; i< gt_cam_orient_cfw.size(); ++i)
    {
        const auto& cam_cfw = gt_cam_orient_cfw[i];
        const SE3Transform& cam_wfc = SE3Inv(cam_cfw);

        // get position of the camera in the world: cam_to_world*(0,0,0,1)=cam_pos
        const Eigen::Matrix<Scalar, 3, 1>& cam_pos_world = cam_wfc.T;

        if (display_trajectory && cam_pos_world_prev_inited)
        {
            glBegin(GL_LINES);
            glColor3fv(track_color.data());
            glVertex3d(cam_pos_world_prev[0], cam_pos_world_prev[1], cam_pos_world_prev[2]);
            glVertex3d(cam_pos_world[0], cam_pos_world[1], cam_pos_world[2]);
            glEnd();
        }

        bool last = i == gt_cam_orient_cfw.size() - 1;
        if (last)
            RenderSchematicCamera(cam_wfc, track_color, last_cam_disp_type);
        else 
            RenderSchematicCamera(cam_wfc, track_color, mid_cam_disp_type);

        cam_pos_world_prev = cam_pos_world;
        cam_pos_world_prev_inited = true;
    }
}

void RenderLastCameraUncertEllipsoid(
    const Eigen::Matrix<Scalar, 3, 1>& cam_pos,
    const Eigen::Matrix<Scalar, 3, 3>& cam_pos_uncert,
    const Eigen::Matrix<Scalar, 4, 1>& cam_orient_quat_wfc, Scalar ellipsoid_cut_thr)
{
    Eigen::Matrix<Scalar, 3, 3> cam_orient_wfc;
    RotMatFromQuat(gsl::make_span<const Scalar>(cam_orient_quat_wfc.data(), 4), &cam_orient_wfc);

    RenderUncertaintyEllipsoid(cam_pos, cam_pos_uncert, ellipsoid_cut_thr);
    //bool op = RenderUncertaintyEllipsoidBySampling(cam_pos, cam_pos_uncert, ellipsoid_cut_thr);
    int z = 0;
}

void RenderMap(DavisonMonoSlam* kalman_slam, Scalar ellipsoid_cut_thr, bool display_3D_uncertainties)
{
    size_t sal_pnt_count = kalman_slam->SalientPointsCount();
    for (size_t sal_pnt_ind=0; sal_pnt_ind<sal_pnt_count; ++sal_pnt_ind)
    {
        Eigen::Matrix<Scalar, 3, 1> sal_pnt_pos;
        Eigen::Matrix<Scalar, 3, 3> sal_pnt_pos_uncert;
        kalman_slam->GetSalientPointPredictedPosWithUncertainty(sal_pnt_ind, &sal_pnt_pos, &sal_pnt_pos_uncert);

        glColor3d(0.7, 0.7, 0.7);

        if (display_3D_uncertainties)
        {
            RenderUncertaintyEllipsoid(sal_pnt_pos, sal_pnt_pos_uncert, ellipsoid_cut_thr);
            //bool op = RenderUncertaintyEllipsoidBySampling(sal_pnt_pos, sal_pnt_pos_uncert, ellipsoid_cut_thr);
        }

        glBegin(GL_POINTS);
        glVertex3d(sal_pnt_pos[0], sal_pnt_pos[1], sal_pnt_pos[2]);
        glEnd();
    }
}

void RenderScene(const UIThreadParams& ui_params, DavisonMonoSlam* kalman_slam, Scalar ellipsoid_cut_thr,
    bool display_trajectory,
    CamDisplayType mid_cam_disp_type,
    CamDisplayType last_cam_disp_type,
    bool display_3D_uncertainties)
{
    // world axes
    RenderAxes(0.5);

    RenderMap(kalman_slam, ellipsoid_cut_thr, display_3D_uncertainties);

    if (ui_params.gt_cam_orient_cfw != nullptr)
    {
        std::array<float, 3> track_color{ 232 / 255.0f, 188 / 255.0f, 87 / 255.0f }; // browny
        CamDisplayType gt_last_camera = CamDisplayType::None;
        RenderCameraTrajectory(*ui_params.gt_cam_orient_cfw, track_color, display_trajectory,
            mid_cam_disp_type,
            gt_last_camera);
    }

    std::array<float, 3> actual_track_color{ 128 / 255.0f, 255 / 255.0f, 255 / 255.0f }; // cyan
    if (ui_params.cam_orient_cfw_history != nullptr)
    {
        RenderCameraTrajectory(*ui_params.cam_orient_cfw_history, actual_track_color, display_trajectory,
            mid_cam_disp_type,
            last_cam_disp_type);
    }

    if (display_3D_uncertainties)
    {
        Eigen::Matrix<Scalar, 3, 1> cam_pos;
        Eigen::Matrix<Scalar, 3, 3> cam_pos_uncert;
        Eigen::Matrix<Scalar, 4, 1> cam_orient_quat;
        kalman_slam->GetCameraPredictedPosAndOrientationWithUncertainty(&cam_pos, &cam_pos_uncert, &cam_orient_quat);
        RenderLastCameraUncertEllipsoid(cam_pos, cam_pos_uncert, cam_orient_quat, ellipsoid_cut_thr);
    }
}

class SceneVisualizationPangolinGui
{
public:
    static UIThreadParams s_ui_params;
public:
SceneVisualizationPangolinGui() { }

static void OnForward()
{
    // check if worker request finishing UI thread
    const auto& ui_params = s_ui_params;
    if (!ui_params.WaitForUserInputAfterEachFrame)
        return;

    // request worker to resume processing
    std::lock_guard<std::mutex> lk(ui_params.worker_chat->resume_worker_mutex);
    ui_params.worker_chat->resume_worker_flag = true;
    ui_params.worker_chat->resume_worker_suppress_observations = false;
    ui_params.worker_chat->resume_worker_cv.notify_one();
}

static void OnSkip()
{
    // check if worker request finishing UI thread
    const auto& ui_params = s_ui_params;
    if (!ui_params.WaitForUserInputAfterEachFrame)
        return;

    // request worker to resume processing
    std::lock_guard<std::mutex> lk(ui_params.worker_chat->resume_worker_mutex);
    ui_params.worker_chat->resume_worker_flag = true;
    ui_params.worker_chat->resume_worker_suppress_observations = true;
    ui_params.worker_chat->resume_worker_cv.notify_one();
}

static void OnKeyEsc()
{
    // check if worker request finishing UI thread
    const auto& ui_params = s_ui_params;
    {
        std::lock_guard<std::mutex> lk(ui_params.worker_chat->exit_worker_mutex);
        ui_params.worker_chat->exit_worker_flag = true;
    }
    ui_params.worker_chat->exit_worker_cv.notify_one();
}

void Run()
{
    const auto& ui_params = s_ui_params;

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
        pangolin::ModelViewLookAt(30, -30, 30, 0, 0, 0, pangolin::AxisY)
    );

    constexpr int UI_WIDTH = 180;
    bool has_ground_truth = ui_params.gt_cam_orient_cfw != nullptr;

    // ui panel to the left 
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH))
        .ResizeChildren();

    // 3d content to the right
    pangolin::View& display_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -fw / fh) // TODO: why negative aspect?
        .SetHandler(new pangolin::Handler3D(view_state_3d));

    pangolin::Var<ptrdiff_t> a_frame_ind("ui.frame_ind", -1);
    pangolin::Var<bool> cb_displ_traj("ui.displ_trajectory", true, true);
    pangolin::Var<int> slider_mid_cam_type("ui.mid_cam_type", 1, 0, 2);
    pangolin::Var<bool> cb_displ_mid_cam_type("ui.displ_3D_uncert", true, true);

    pangolin::Var<double> a_cam_x("ui.cam_x", -1);
    pangolin::Var<double> a_cam_x_gt("ui.cam_x_gt", -1);
    pangolin::Var<double> a_cam_y("ui.cam_y", -1);
    pangolin::Var<double> a_cam_y_gt("ui.cam_y_gt", -1);
    pangolin::Var<double> a_cam_z("ui.cam_z", -1);
    pangolin::Var<double> a_cam_z_gt("ui.cam_z_gt", -1);

    pangolin::Var<double> sal_pnt_0_x("ui.sal_pnt_0_x", -1);
    pangolin::Var<double> sal_pnt_0_x_gt("ui.sal_pnt_0_x_gt", -1);
    pangolin::Var<double> sal_pnt_0_y("ui.sal_pnt_0_y", -1);
    pangolin::Var<double> sal_pnt_0_y_gt("ui.sal_pnt_0_y_gt", -1);
    pangolin::Var<double> sal_pnt_0_z("ui.sal_pnt_0_z", -1);
    pangolin::Var<double> sal_pnt_0_z_gt("ui.sal_pnt_0_z_gt", -1);
    pangolin::Var<double> sal_pnt_1_x("ui.sal_pnt_1_x", -1);
    pangolin::Var<double> sal_pnt_1_x_gt("ui.sal_pnt_1_x_gt", -1);
    pangolin::Var<double> sal_pnt_1_y("ui.sal_pnt_1_y", -1);
    pangolin::Var<double> sal_pnt_1_y_gt("ui.sal_pnt_1_y_gt", -1);
    pangolin::Var<double> sal_pnt_1_z("ui.sal_pnt_1_z", -1);
    pangolin::Var<double> sal_pnt_1_z_gt("ui.sal_pnt_1_z_gt", -1);

    pangolin::RegisterKeyPressCallback('s', OnSkip);
    pangolin::RegisterKeyPressCallback('f', OnForward);
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_ESCAPE, OnKeyEsc);

    while (!pangolin::ShouldQuit())
    {
        auto loop_body = [&]()
        {
            // update ui
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            ptrdiff_t frame_ind = ui_params.get_observable_frame_ind_fun();
            if (frame_ind < 0)
                return;

            a_frame_ind = frame_ind;

            if (has_ground_truth)
            {
                SE3Transform cam_orient_wfc = SE3Inv((*ui_params.gt_cam_orient_cfw)[frame_ind]);
                a_cam_x_gt = cam_orient_wfc.T[0];
                a_cam_y_gt = cam_orient_wfc.T[1];
                a_cam_z_gt = cam_orient_wfc.T[2];
            }

            std::shared_lock<std::shared_mutex> lock(ui_params.worker_chat->location_and_map_mutex_);
            CameraPosState cam_state;
            ui_params.kalman_slam->GetCameraPredictedPosState(&cam_state);

            a_cam_x = cam_state.PosW[0];
            a_cam_y = cam_state.PosW[1];
            a_cam_z = cam_state.PosW[2];

            Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_0;
            ui_params.kalman_slam->GetSalientPointPredictedPosWithUncertainty(0, &sal_pnt_0, nullptr);
            sal_pnt_0_x = sal_pnt_0[0];
            sal_pnt_0_y = sal_pnt_0[1];
            sal_pnt_0_z = sal_pnt_0[2];
            Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_1;
            ui_params.kalman_slam->GetSalientPointPredictedPosWithUncertainty(1, &sal_pnt_1, nullptr);
            sal_pnt_1_x = sal_pnt_1[0];
            sal_pnt_1_y = sal_pnt_1[1];
            sal_pnt_1_z = sal_pnt_1[2];

            if (has_ground_truth)
            {
                FragmentMap::DependsOnSalientPointIdInfrustructure();
                if (ui_params.entire_map->SalientPointsCount() > 0)
                {
                    const SalientPointFragment& sal_pnt_0_gt = ui_params.entire_map->GetSalientPointByInternalOrder(0);
                    sal_pnt_0_x_gt = sal_pnt_0_gt.Coord.value()[0];
                    sal_pnt_0_y_gt = sal_pnt_0_gt.Coord.value()[1];
                    sal_pnt_0_z_gt = sal_pnt_0_gt.Coord.value()[2];
                }

                if (ui_params.entire_map->SalientPointsCount() > 1)
                {
                    const SalientPointFragment& sal_pnt_1_gt = ui_params.entire_map->GetSalientPointByInternalOrder(1);
                    sal_pnt_1_x_gt = sal_pnt_1_gt.Coord.value()[0];
                    sal_pnt_1_y_gt = sal_pnt_1_gt.Coord.value()[1];
                    sal_pnt_1_z_gt = sal_pnt_1_gt.Coord.value()[2];
                }
            }

            display_cam.Activate(view_state_3d);

            bool display_trajectory = cb_displ_traj.Get();
            bool display_3D_uncertainties = cb_displ_mid_cam_type.Get();

            CamDisplayType mid_cam_disp_type = CamDisplayType::None;
            int displ_mid_cam_type = slider_mid_cam_type.Get();
            if (displ_mid_cam_type == 0)
                mid_cam_disp_type = CamDisplayType::None;
            else if (displ_mid_cam_type == 1)
            {
                mid_cam_disp_type = CamDisplayType::Dot;
            }
            else if (displ_mid_cam_type == 2)
            {
                mid_cam_disp_type = CamDisplayType::Schematic;
            }

            CamDisplayType last_cam_disp_type = CamDisplayType::Schematic;

            RenderScene(ui_params, ui_params.kalman_slam, ui_params.ellipsoid_cut_thr,
                display_trajectory,
                mid_cam_disp_type,
                last_cam_disp_type,
                display_3D_uncertainties);
        };

        {
            // check if worker request finishing UI thread
            std::lock_guard<std::mutex> lk(ui_params.worker_chat->exit_ui_mutex);
            if (ui_params.worker_chat->exit_ui_flag)
            {
                VLOG(4) << "UI got exit signal";
                break;
            }
        }

        loop_body();

        pangolin::FinishFrame();

        std::this_thread::sleep_for(100ms); // make ui thread more 'lightweight'
    }
}
};

UIThreadParams SceneVisualizationPangolinGui::s_ui_params;

void SceneVisualizationThread(UIThreadParams ui_params) // parameters by value across threads
{
    VLOG(4) << "UI thread is running";

    SceneVisualizationPangolinGui::s_ui_params = ui_params;
    SceneVisualizationPangolinGui pangolin_gui;
    pangolin_gui.Run();

    VLOG(4) << "UI thread is exiting";
}
#endif

DEFINE_double(world_xmin, -1.5, "world xmin");
DEFINE_double(world_xmax, 1.5, "world xmax");
DEFINE_double(world_ymin, -1.5, "world ymin");
DEFINE_double(world_ymax, 1.5, "world ymax");
DEFINE_double(world_zmin, 0, "world zmin");
DEFINE_double(world_zmax, 1, "world zmax");
DEFINE_double(world_cell_size_x, 0.5, "cell size x");
DEFINE_double(world_cell_size_y, 0.5, "cell size y");
DEFINE_double(world_cell_size_z, 0.5, "cell size z");
DEFINE_double(viewer_offset_down, 7, "viewer's offset from viewed point in the down direction");
DEFINE_double(viewer_ascendZ, 7, "viewer's offset in the up direction");
DEFINE_int32(viewer_steps_per_side_x, 20, "number of viewer's steps at each side of the rectangle");
DEFINE_int32(viewer_steps_per_side_y, 10, "number of viewer's steps at each side of the rectangle");
DEFINE_double(noise_R_std, 0.005, "Standard deviation of noise distribution for R, 0=no noise (eg: 0.01)");
DEFINE_double(noise_x3D_std, 0.005, "Standard deviation of noise distribution for salient points, 0=no noise (eg: 0.1)");
DEFINE_double(kalman_estim_var_init_std, 0.001, "");
DEFINE_double(kalman_input_noise_std, 0.08, "");
DEFINE_double(kalman_measurm_noise_std, 1, "");
DEFINE_int32(kalman_update_impl, 1, "");
DEFINE_double(ellipsoid_cut_thr, 0.04, "probability cut threshold for uncertainty ellipsoid");
DEFINE_bool(wait_after_each_frame, false, "true to wait for keypress after each iteration");
DEFINE_bool(debug_skim_over, false, "overview the synthetic world without reconstruction");
DEFINE_bool(kalman_debug_estim_vars_cov, false, "");
DEFINE_bool(kalman_debug_predicted_vars_cov, false, "");
DEFINE_bool(fake_localization, false, "");

int DavisonMonoSlamDemo(int argc, char* argv[])
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
        x3D_noise_dis = std::make_unique<std::normal_distribution<Scalar>>(0, FLAGS_noise_x3D_std);

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

    GenerateCameraShotsAlongRectangularPath(wb, FLAGS_viewer_steps_per_side_x, FLAGS_viewer_steps_per_side_y, FLAGS_viewer_offset_down, FLAGS_viewer_ascendZ, &gt_cam_orient_cfw);

    std::vector<SE3Transform> gt_cam_orient_wfc;
    std::transform(gt_cam_orient_cfw.begin(), gt_cam_orient_cfw.end(), std::back_inserter(gt_cam_orient_wfc), [](auto& t) { return SE3Inv(t); });

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

    DavisonMonoSlam::DebugPathEnum debug_path = DavisonMonoSlam::DebugPathEnum::DebugNone;
    if (FLAGS_kalman_debug_estim_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugEstimVarsCov;
    if (FLAGS_kalman_debug_predicted_vars_cov)
        debug_path = debug_path | DavisonMonoSlam::DebugPathEnum::DebugPredictedVarsCov;
    DavisonMonoSlam::SetDebugPath(debug_path);

    DavisonMonoSlam tracker;
    tracker.between_frames_period_ = 1;
    tracker.cam_intrinsics_ = cam_intrinsics;
    tracker.cam_distort_params_ = cam_distort_params;
    tracker.input_noise_std_ = FLAGS_kalman_input_noise_std;
    tracker.measurm_noise_std_ = FLAGS_kalman_measurm_noise_std;
    tracker.kalman_update_impl_ = FLAGS_kalman_update_impl;
    tracker.debug_ellipsoid_cut_thr_ = FLAGS_ellipsoid_cut_thr;
    tracker.fake_localization_ = FLAGS_fake_localization;
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
    tracker.SetCornersMatcher(std::make_unique<DemoCornersMatcher>(&tracker, gt_cam_orient_cfw, entire_map, img_size));

    // hack: put full prior knowledge into the tracker
    tracker.ResetState(gt_cam_orient_cfw[0], entire_map.SalientPoints(), FLAGS_kalman_estim_var_init_std);

    tracker.PredictEstimVarsHelper();
    LOG(INFO) << "kalman_update_impl=" << FLAGS_kalman_update_impl;

    //
    ptrdiff_t observable_frame_ind = -1; // this is visualized by UI, it is one frame less than current frame
    std::vector<SE3Transform> cam_orient_cfw_history;
    
#if defined(SRK_HAS_OPENCV)
    cv::Mat camera_image_rgb = cv::Mat::zeros((int)img_size[1], (int)img_size[0], CV_8UC3);
#endif
#if defined(SRK_HAS_PANGOLIN)
    // across threads shared data
    auto worker_chat = std::make_shared<WorkerThreadChat>();

    UIThreadParams ui_params {};
    ui_params.WaitForUserInputAfterEachFrame = FLAGS_wait_after_each_frame;
    ui_params.kalman_slam = &tracker;
    ui_params.ellipsoid_cut_thr = FLAGS_ellipsoid_cut_thr;
    ui_params.gt_cam_orient_cfw = &gt_cam_orient_cfw;
    ui_params.cam_orient_cfw_history = &cam_orient_cfw_history;
    ui_params.get_observable_frame_ind_fun = [&observable_frame_ind]() { return observable_frame_ind; };
    ui_params.entire_map = &entire_map;
    ui_params.worker_chat = worker_chat;
    std::thread ui_thread(SceneVisualizationThread, ui_params);
#endif

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

    tracker.LogReprojError();

    int key = 0; // result of opencv::waitKey
    constexpr size_t well_known_frames_count = 0;
    FragmentMap world_map;
    world_map.SetFragmentIdOffsetInternal(2000'000); // not necessary

    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        // orient camera
        const SE3Transform& cam_cfw = gt_cam_orient_cfw[frame_ind];
        SE3Transform cam_wfc = SE3Inv(cam_cfw);

#if defined(SRK_HAS_OPENCV)
        camera_image_rgb.setTo(0);
        auto project_fun = [&cam_cfw, &tracker](const suriko::Point3& sal_pnt) -> Eigen::Matrix<suriko::Scalar, 3, 1>
        {
            suriko::Point3 pnt_cam = SE3Apply(cam_cfw, sal_pnt);
            suriko::Point2 pnt_pix = tracker.ProjectCameraPoint(pnt_cam);
            return Eigen::Matrix<suriko::Scalar, 3, 1>(pnt_pix[0], pnt_pix[1], 1);
        };
        constexpr Scalar f0 = 1;
        suriko_demos::Draw2DProjectedAxes(f0, project_fun, &camera_image_rgb);
#endif
        size_t new_points_per_frame_count = 0;
        size_t new_track_per_frame_count = 0;
        std::set<size_t> entire_fragment_id_per_frame;

        // determine which salient points are visible
        for (const SalientPointFragment& fragment : entire_map.SalientPoints())
        {
            const Point3& salient_point = fragment.Coord.value();
            suriko::Point3 pnt_camera = SE3Apply(cam_cfw, salient_point);
            suriko::Point2 pnt_pix = tracker.ProjectCameraPoint(pnt_camera);
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
                tracker.track_rep_.GetFirstPointTrackByFragmentSyntheticId(fragment.SyntheticVirtualPointId.value(), &corner_track);

                if (corner_track == nullptr)
                {
                    // add new salient points
                    size_t salient_point_id = 0;
                    SalientPointFragment& new_frag = world_map.AddSalientPoint(fragment.Coord, &salient_point_id);
                    new_frag.SyntheticVirtualPointId = fragment.SyntheticVirtualPointId; // force id of subset fragment to be identical to fragment id from entire map

                    new_points_per_frame_count += 1;

                    //
                    suriko::CornerTrack& new_corner_track = tracker.track_rep_.AddCornerTrackObj();
                    new_corner_track.SalientPointId = salient_point_id;
                    new_corner_track.SyntheticVirtualPointId = fragment.SyntheticVirtualPointId;

                    corner_track = &new_corner_track;

                    new_track_per_frame_count += 1;
                }

                if (!corner_track->DebugSalientPointCoord.has_value())
                    corner_track->DebugSalientPointCoord = salient_point;

                suriko::Point2 pix(pix_x, pix_y);

                CornerData& corner_data = corner_track->AddCorner(frame_ind);
                corner_data.PixelCoord = pix;
                //corner_data.ImageCoord = K_inv * pix.AsHomog();
                Scalar s1 = std::numeric_limits<Scalar>::signaling_NaN();
                Scalar s2 = std::numeric_limits<Scalar>::quiet_NaN();
                corner_data.ImageCoord.setConstant(s1);
            }
        }

#if defined(SRK_HAS_OPENCV)
        std::stringstream strbuf;
        strbuf << "f=" << frame_ind;
        cv::putText(camera_image_rgb, cv::String(strbuf.str()), cv::Point(10, (int)img_size[1] - 15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
        cv::imshow("front-camera", camera_image_rgb);
        cv::waitKey(1); // allow to refresh an opencv view
#endif

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
            << " points_count=" << world_map.SalientPointsCount()
            << " tracks_count=" << tracker.track_rep_.CornerTracks.size()
            << " ncd=" << new_points.size() << "-" << common_points.size() << "-" << del_points.size();

        // process the remaining frames
        if (!FLAGS_debug_skim_over && frame_ind >= well_known_frames_count)
        {
            tracker.ProcessFrame(frame_ind);

            CameraPosState cam_state;
            tracker.GetCameraPredictedPosState(&cam_state);
            SE3Transform actual_cam_wfc(RotMat(cam_state.OrientationWfc), cam_state.PosW);
            SE3Transform actual_cam_cfw = SE3Inv(actual_cam_wfc);
            cam_orient_cfw_history.push_back(actual_cam_cfw);

            observable_frame_ind = frame_ind;
        }
        tracker.LogReprojError();

#if defined(SRK_HAS_PANGOLIN)
        {
            // check if UI requests the exit
            std::lock_guard<std::mutex> lk(worker_chat->exit_worker_mutex);
            if (worker_chat->exit_worker_flag)
                break;
        }
        
        bool suppress_observations = false;
        if (FLAGS_wait_after_each_frame)
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
        cv::waitKey(1); // wait for a moment to allow OpenCV to redraw the image
#endif
    }
    VLOG(4) << "Finished processing all the frames";

#if defined(SRK_HAS_PANGOLIN)
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
#elif defined(SRK_HAS_OPENCV)
    cv::waitKey(0); // 0=wait forever
#endif
    return 0;
}
}

int main(int argc, char* argv[])
{
    int result = 0;
    result = suriko_demos_davison_mono_slam::DavisonMonoSlamDemo(argc, argv);
    return result;
}