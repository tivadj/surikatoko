#include "demo-davison-mono-slam-ui.h"
#include <random>
#include <chrono>
#include <thread>
#include <glog/logging.h>
#include "suriko/approx-alg.h"
#include "suriko/quat.h"

#if defined(SRK_HAS_PANGOLIN)
namespace suriko_demos_davison_mono_slam
{
using namespace std::literals::chrono_literals;

std::optional<UIChatMessage> PopMsgUnderLock(std::optional<UIChatMessage>* msg)
{
    std::optional<UIChatMessage> result = *msg;
    *msg = std::nullopt;
    return result;
}

std::optional<WorkerChatMessage> PopMsgUnderLock(std::optional<WorkerChatMessage>* msg)
{
    std::optional<WorkerChatMessage> result = *msg;
    *msg = std::nullopt;
    return result;
}

void LoadSE3TransformIntoOpengGLMat(const SE3Transform& cam_wfc, gsl::span<double> opengl_mat_by_col)
{
    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::ColMajor>> opengl_mat(opengl_mat_by_col.data());
    opengl_mat.topLeftCorner<3, 3>() = cam_wfc.R.cast<double>();
    opengl_mat.topRightCorner<3, 1>() = cam_wfc.T.cast<double>();
    opengl_mat.bottomLeftCorner<1, 3>().setZero();
    opengl_mat(3, 3) = 1;
}

enum class CamDisplayType
{
    None,
    Dot, // camera is visualized as a dot
    Schematic // camera is visualized schematically
};

/// Draw axes in the local coordinates.
void RenderAxes(Scalar axis_seg_len, float line_width)
{
    float old_line_width;
    glGetFloatv(GL_LINE_WIDTH, &old_line_width);

    auto ax = axis_seg_len;
    glLineWidth(line_width);
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

    glLineWidth(old_line_width);
}

void RenderSchematicCamera(const SE3Transform& cam_wfc, const std::array<float, 3>& track_color, CamDisplayType cam_disp_type)
{
    if (cam_disp_type == CamDisplayType::None)
        return;

    // transform to the camera frame
    std::array<double, 4 * 4> opengl_mat_by_col{};
    LoadSE3TransformIntoOpengGLMat(cam_wfc, opengl_mat_by_col);

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
        RenderAxes(ax, 2);

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
    for (size_t i = 0; i < dots_per_ellipse; ++i)
    {
        Scalar theta = i * (2 * M_PI) / dots_per_ellipse;
        Scalar cos_theta = std::cos(theta);
        Scalar sin_theta = std::sin(theta);

        // Polar ellipse https://en.wikipedia.org/wiki/Ellipse
        // r=a*b/sqrt((b*cos(theta))^2 + (a*sin(theta))^2)
        Scalar r = big_semi_axes[0] * big_semi_axes[1] / std::sqrt(suriko::Sqr(big_semi_axes[1] * cos_theta) + suriko::Sqr(big_semi_axes[0] * sin_theta));

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
    Scalar ellipsoid_cut_thr,
    bool ui_swallow_exc)
{
    Eigen::LLT<Eigen::Matrix<Scalar, 3, 3>> lltOfA(pos_uncert);
    bool op2 = lltOfA.info() != Eigen::NumericalIssue;
    if (!op2)
    {
        LOG(ERROR) << "failed lltOfA.info() != Eigen::NumericalIssue";
        return;
    }

    QuadricEllipsoidWithCenter ellipsoid;
    ExtractEllipsoidFromUncertaintyMat(pos, pos_uncert, ellipsoid_cut_thr, &ellipsoid);

    Eigen::Matrix<Scalar, 3, 1> center1;
    Eigen::Matrix<Scalar, 3, 1> semi_axes1;
    Eigen::Matrix<Scalar, 3, 3> rot_mat_world_from_ellipse1;
    bool op1 = GetRotatedEllipsoid(ellipsoid, !ui_swallow_exc, &center1, &semi_axes1, &rot_mat_world_from_ellipse1);
    if (!op1)
    {
        if (ui_swallow_exc)
            return;
        else SRK_ASSERT(op1);
    }

    // draw projection of ellipsoid
    RenderEllipsoid(center1, semi_axes1, rot_mat_world_from_ellipse1);
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
    std::uniform_real_distribution<Scalar> distr(-3 * sig1, 3 * sig1);

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
        Scalar b1 = -std::log(suriko::Sqr(ellipsoid_cut_thr)*suriko::Pow3(2 * M_PI)*uncert_det);
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
    for (size_t i = 0; i < ellips_points.size(); ++i)
    {
        auto& p1 = ellips_points[i];
        Scalar closest_dist = 999;
        size_t closest_point_ind = -1;
        for (size_t j = 0; j < ellips_points.size(); ++j)
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

SE3Transform CurCamFromTrackerOrigin(const std::vector<SE3Transform>& gt_cam_orient_cfw, size_t frame_ind);

void RenderCameraTrajectory(const std::vector<SE3Transform>& gt_cam_orient_cfw,
    const std::array<GLfloat, 3>& track_color,
    bool display_trajectory,
    CamDisplayType mid_cam_disp_type,
    CamDisplayType last_cam_disp_type)
{
    Eigen::Matrix<Scalar, 3, 1> cam_pos_world_prev;
    bool cam_pos_world_prev_inited = false;

    for (size_t i = 0; i < gt_cam_orient_cfw.size(); ++i)
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
    const Eigen::Matrix<Scalar, 4, 1>& cam_orient_quat_wfc, Scalar ellipsoid_cut_thr, bool ui_swallow_exc)
{
    Eigen::Matrix<Scalar, 3, 3> cam_orient_wfc;
    RotMatFromQuat(gsl::make_span<const Scalar>(cam_orient_quat_wfc.data(), 4), &cam_orient_wfc);

    RenderUncertaintyEllipsoid(cam_pos, cam_pos_uncert, ellipsoid_cut_thr, ui_swallow_exc);
    //bool op = RenderUncertaintyEllipsoidBySampling(cam_pos, cam_pos_uncert, ellipsoid_cut_thr);
    int z = 0;
}

void RenderMap(DavisonMonoSlam* kalman_slam, Scalar ellipsoid_cut_thr,
    bool display_3D_uncertainties,
    bool ui_swallow_exc)
{
    size_t sal_pnt_count = kalman_slam->SalientPointsCount();
    for (size_t sal_pnt_ind = 0; sal_pnt_ind < sal_pnt_count; ++sal_pnt_ind)
    {
        Eigen::Matrix<Scalar, 3, 1> sal_pnt_pos;
        Eigen::Matrix<Scalar, 3, 3> sal_pnt_pos_uncert;
        kalman_slam->GetSalientPointPredictedPosWithUncertainty(sal_pnt_ind, &sal_pnt_pos, &sal_pnt_pos_uncert);

        glColor3d(0.7, 0.7, 0.7);

        if (display_3D_uncertainties)
        {
            RenderUncertaintyEllipsoid(sal_pnt_pos, sal_pnt_pos_uncert, ellipsoid_cut_thr, ui_swallow_exc);
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
    bool display_3D_uncertainties,
    bool ui_swallow_exc)
{
    // world axes
    RenderAxes(1, 4);

    bool has_gt_cameras = ui_params.gt_cam_orient_cfw != nullptr;  // gt=ground truth
    if (has_gt_cameras)
    {
        std::array<GLfloat, 3> track_color{ 232 / 255.0f, 188 / 255.0f, 87 / 255.0f }; // browny
        CamDisplayType gt_last_camera = CamDisplayType::None;
        RenderCameraTrajectory(*ui_params.gt_cam_orient_cfw, track_color, display_trajectory,
            mid_cam_disp_type,
            gt_last_camera);
    }

    bool has_gt_sal_pnts = ui_params.entire_map != nullptr;
    if (has_gt_sal_pnts)
    {
        std::array<GLfloat, 3> track_color{ 232 / 255.0f, 188 / 255.0f, 87 / 255.0f }; // browny
        glColor3fv(track_color.data());

        for (const SalientPointFragment& sal_pnt_fragm : ui_params.entire_map->SalientPoints())
        {
            const auto& p = sal_pnt_fragm.Coord.value();
            glBegin(GL_POINTS);
            glVertex3d(p[0], p[1], p[2]);
            glEnd();
        }
    }

    {
        // the scene is drawn in the coordinate system of a tracker (=cam0)

        std::array<double, 4 * 4> tracker_from_world_4x4_by_col{};
        const SE3Transform world_from_tracker = SE3Inv(ui_params.tracker_origin_from_world);
        LoadSE3TransformIntoOpengGLMat(world_from_tracker, tracker_from_world_4x4_by_col);
        glPushMatrix();
        glMultMatrixd(tracker_from_world_4x4_by_col.data());

        RenderAxes(0.5, 2); // axes of the tracker's origin (=cam0)

        RenderMap(kalman_slam, ellipsoid_cut_thr, display_3D_uncertainties, ui_swallow_exc);

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
            RenderLastCameraUncertEllipsoid(cam_pos, cam_pos_uncert, cam_orient_quat, ellipsoid_cut_thr, ui_swallow_exc);
        }

        glPopMatrix();
    }
}

UIThreadParams SceneVisualizationPangolinGui::s_ui_params_;
std::weak_ptr<SceneVisualizationPangolinGui> SceneVisualizationPangolinGui::s_this_ui_;

void SceneVisualizationPangolinGui::InitUI()
{
    constexpr float fw = 640;
    constexpr float fh = 480;
    constexpr int w = (int)fw;
    constexpr int h = (int)fh;

    pangolin::CreateWindowAndBind("3DReconstr", w, h);
    glEnable(GL_DEPTH_TEST);

    float center_x = fw / 2;
    float center_y = fh / 2;

    view_state_3d_ = std::make_unique<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(w, h, 420, 420, center_x, center_y, 0.2, 1500),
        pangolin::ModelViewLookAt(30, -30, 30, 0, 0, 0, pangolin::AxisY)
    );

    constexpr int kUiWidth = 280;

    // ui panel to the left 
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(kUiWidth))
        .ResizeChildren();

    // 3d content to the right
    int display_cam_bot = 0;
    display_cam = &pangolin::CreateDisplay()
        .SetBounds(pangolin::Attach::Pix(display_cam_bot), 1.0, pangolin::Attach::Pix(kUiWidth), 1.0, -fw / fh); // TODO: why negative aspect?

    handler3d_ = std::make_unique<Handler3DImpl>(*view_state_3d_);
    handler3d_->owner_ = this;
    display_cam->SetHandler(handler3d_.get());

    a_frame_ind_ = std::make_unique<pangolin::Var<ptrdiff_t>>("ui.frame_ind", -1);
    cb_displ_traj_ = std::make_unique<pangolin::Var<bool>>("ui.displ_trajectory", true, true);
    slider_mid_cam_type_ = std::make_unique<pangolin::Var<int>>("ui.mid_cam_type", 1, 0, 2);
    cb_displ_mid_cam_type_ = std::make_unique<pangolin::Var<bool>>("ui.displ_3D_uncert", true, true);

    pangolin::RegisterKeyPressCallback('s', []()
    { 
        if (auto form = s_this_ui_.lock())
            form->OnKeyPressed('s'); 
    });
    pangolin::RegisterKeyPressCallback('f', []()
    {
        if (auto form = s_this_ui_.lock())
            form->OnKeyPressed('f');
    });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_ESCAPE, []()
    {
        if (auto form = s_this_ui_.lock())
            form->OnKeyPressed(pangolin::PANGO_KEY_ESCAPE);
    });
}

void SceneVisualizationPangolinGui::RenderFrame()
{
    const auto& ui_params = s_ui_params_;

    // update ui
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ptrdiff_t frame_ind = ui_params.get_observable_frame_ind_fun();
    if (frame_ind < 0)
        return;

    *a_frame_ind_ = frame_ind;

    display_cam->Activate(*view_state_3d_);

    bool display_trajectory = cb_displ_traj_->Get();
    bool display_3D_uncertainties = cb_displ_mid_cam_type_->Get();

    CamDisplayType mid_cam_disp_type = CamDisplayType::None;
    int displ_mid_cam_type = slider_mid_cam_type_->Get();
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
                display_3D_uncertainties,
                ui_params.ui_swallow_exc);
}

int SceneVisualizationPangolinGui::WaitKey()
{
    const auto any_key = [](int key) { return true; };
    return WaitKey(any_key);
}

int SceneVisualizationPangolinGui::WaitKey(std::function<bool(int key)> key_predicate)
{
    key_ = std::nullopt;
    while (true)
    {
        TreatAppCloseAsEscape();
        
        if (key_.has_value() && key_predicate(key_.value()))
            break;
        RenderFrame();
        pangolin::FinishFrame(); // also processes user input
    }

    return key_.value();
}

std::optional<int> SceneVisualizationPangolinGui::RenderFrameAndProlongUILoopOnUserInput(std::function<bool(int key)> break_on)
{
    // If a user provides an input (presses a key or clicks a mouse button)
    // we will prolong the execution of ui loop for couple of seconds.
    // It is designed for single thread scenario.
    // Obviously the calling function is not running at these moments.

    std::optional<std::chrono::steady_clock::time_point> prolonged_ui_loop_end_time;
    bool the_first_iter = true;
    while (true)
    {
        auto is_prolonged = [prolonged_ui_loop_end_time]() -> bool
        {
            if (!prolonged_ui_loop_end_time.has_value())
                return false;
            const auto now = std::chrono::steady_clock::now();
            return now < prolonged_ui_loop_end_time;
        };

        const bool do_ui_loop = the_first_iter || is_prolonged();
        if (!do_ui_loop)
        {
            // the time of prolonged ui loop is elapsed
            //LOG(INFO) << "UI quit prolonged loop (time elapsed)";
            break;
        }

        // do prolonged ui loop
        //LOG(INFO) << "UI loops";

        got_user_input_ = false;

        RenderFrame();
        pangolin::FinishFrame(); // also processes user input

        TreatAppCloseAsEscape();

        if (got_user_input_)
        {
            if (key_.has_value() && break_on(key_.value()))
                break;

            //LOG(INFO) << "UI is prolonged";

            // continue execution of UI loop for couple of seconds
            auto now = std::chrono::steady_clock::now();
            prolonged_ui_loop_end_time = now + ui_loop_prolong_period_;
        }

        std::this_thread::sleep_for(ui_tight_loop_relaxing_delay_);
        the_first_iter = false;
    }
    return key_;
}

void SceneVisualizationPangolinGui::RunInSeparateThread()
{
    const auto& ui_params = s_ui_params_;

    while (true)
    {
        TreatAppCloseAsEscape();

        auto do_continue = [this, &ui_params]() -> bool
        {
            // check if worker request finishing UI thread
            WorkerChatSharedState& worker_chat = *ui_params.worker_chat.get();
            std::lock_guard<std::mutex> lk(worker_chat.the_mutex);
            if (std::optional<UIChatMessage> msg = PopMsgUnderLock(&worker_chat.ui_message); msg.has_value())
            {
                switch (msg.value())
                {
                case UIChatMessage::UIExit:
                    VLOG(4) << "UI got exit signal";
                    return false;
                case UIChatMessage::UIWaitKey:
                    multi_threaded_.form_state = FormState::WaitKey;
                    break;
                default:
                    break;
                }
            }
            return true;
        };

        if (!do_continue())
            break;

        RenderFrame();

        pangolin::FinishFrame(); // also processes user input

        std::this_thread::sleep_for(ui_tight_loop_relaxing_delay_);
    }
}

void SceneVisualizationPangolinGui::OnKeyPressed(int key)
{
    LOG(INFO) << "pressed key " <<key;
    got_user_input_ = true;
    key_ = key;

    bool key_handled = false;

    // check if worker request finishing UI thread
    const auto& ui_params = s_ui_params_;
    if (ui_params.kalman_slam->in_multi_threaded_mode_)
    {
        if (multi_threaded_.form_state == FormState::WaitKey &&
            ui_params.worker_chat->ui_wait_key_predicate_(key))
        {
            multi_threaded_.form_state = FormState::IterateUILoop;

            // request worker to resume processing
            std::lock_guard<std::mutex> lk(ui_params.worker_chat->the_mutex);
            ui_params.worker_chat->worker_message = WorkerChatMessage::WorkerKeyPressed;
            ui_params.worker_chat->ui_pressed_key = key_;
            ui_params.worker_chat->worker_got_new_message_cv.notify_one();
            key_handled = true;
        }
    }

    if (key_handled)
        return;

    switch (key)
    {
    case pangolin::PANGO_KEY_ESCAPE:
        if (ui_params.kalman_slam->in_multi_threaded_mode_)
        {
            {
                std::lock_guard<std::mutex> lk(ui_params.worker_chat->the_mutex);
                ui_params.worker_chat->worker_message = WorkerChatMessage::WorkerExit;
            }
            ui_params.worker_chat->worker_got_new_message_cv.notify_one();
        }
        break;
    default:
        break;
    }
}

void SceneVisualizationPangolinGui::TreatAppCloseAsEscape()
{
    if (pangolin::ShouldQuit())
        OnKeyPressed(pangolin::PANGO_KEY_ESCAPE);
}

SceneVisualizationPangolinGui::Handler3DImpl::Handler3DImpl(pangolin::OpenGlRenderState& cam_state)
    : Handler3D(cam_state), owner_(nullptr)
{
}

void SceneVisualizationPangolinGui::Handler3DImpl::Mouse(pangolin::View& view, pangolin::MouseButton button, int x, int y, bool pressed, int button_state)
{
    pangolin::Handler3D::Mouse(view, button, x, y, pressed, button_state);
}

void SceneVisualizationPangolinGui::Handler3DImpl::MouseMotion(pangolin::View& view, int x, int y, int button_state)
{
    pangolin::Handler3D::MouseMotion(view, x, y, button_state);
}

void SceneVisualizationPangolinGui::Handler3DImpl::Special(pangolin::View& view, pangolin::InputSpecial inType, float x, float y, float p1, float p2, float p3, float p4, int button_state)
{
    pangolin::Handler3D::Special(view, inType, x, y, p1, p2, p3, p4, button_state);
}

std::shared_ptr<SceneVisualizationPangolinGui> SceneVisualizationPangolinGui::New(bool defer_ui_construction)
{
    auto gui = std::make_shared<SceneVisualizationPangolinGui>();
    SceneVisualizationPangolinGui::s_this_ui_ = gui;

    if (!defer_ui_construction)
        gui->InitUI();
    return gui;
}

void SceneVisualizationPangolinGui::SetCameraBehindTrackerOnce(const SE3Transform& tracker_origin_from_world, float back_dist)
{
    auto wfc = SE3Inv(tracker_origin_from_world);
    pangolin::OpenGlMatrix model_view_col_major;

    Eigen::Map< Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>> eigen_mat(static_cast<Scalar*>(model_view_col_major.m));

    eigen_mat =
        internals::SE3Mat(internals::RotMat(0, 1, 0, M_PI)) *  // to axis format of OpenGL
        internals::SE3Mat(Eigen::Matrix<Scalar, 3, 1>{0, 0, back_dist}) *
        internals::SE3Mat(tracker_origin_from_world.R, tracker_origin_from_world.T);

    view_state_3d_->SetModelViewMatrix(model_view_col_major);
}

void SceneVisualizationThread(UIThreadParams ui_params) // parameters by value across threads
{
    VLOG(4) << "UI thread is running";

    SceneVisualizationPangolinGui::s_ui_params_ = ui_params;

    auto pangolin_gui = SceneVisualizationPangolinGui::New();
    pangolin_gui->ui_tight_loop_relaxing_delay_ = ui_params.ui_tight_loop_relaxing_delay;
    pangolin_gui->RunInSeparateThread();

    VLOG(4) << "UI thread is exiting";
}

}
#endif