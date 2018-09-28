#include "demo-davison-mono-slam-ui.h"
#include <random>
#include <chrono>
#include <thread>
#include "suriko/approx-alg.h"
#include "suriko/quat.h"

#if defined(SRK_HAS_PANGOLIN)
namespace suriko_demos_davison_mono_slam
{
using std::literals::chrono_literals::operator""ms;

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

void RenderCameraTrajectory(const std::vector<SE3Transform>& gt_cam_orient_cfw,
    const std::array<float, 3>& track_color,
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
    RenderAxes(0.5);

    RenderMap(kalman_slam, ellipsoid_cut_thr, display_3D_uncertainties, ui_swallow_exc);

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
        RenderLastCameraUncertEllipsoid(cam_pos, cam_pos_uncert, cam_orient_quat, ellipsoid_cut_thr, ui_swallow_exc);
    }
}

UIThreadParams SceneVisualizationPangolinGui::s_ui_params;

SceneVisualizationPangolinGui::SceneVisualizationPangolinGui()
{
}

void SceneVisualizationPangolinGui::OnForward()
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

void SceneVisualizationPangolinGui::OnSkip()
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

void SceneVisualizationPangolinGui::OnKeyEsc()
{
    // check if worker request finishing UI thread
    const auto& ui_params = s_ui_params;
    {
        std::lock_guard<std::mutex> lk(ui_params.worker_chat->exit_worker_mutex);
        ui_params.worker_chat->exit_worker_flag = true;
    }
    ui_params.worker_chat->exit_worker_cv.notify_one();
}

void SceneVisualizationPangolinGui::Run()
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

    constexpr int kUiWidth = 280;
    constexpr int kDataLogHeight = 100;
    bool has_ground_truth = ui_params.gt_cam_orient_cfw != nullptr;
    bool ui_swallow_exc = ui_params.ui_swallow_exc;

    // ui panel to the left 
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(kUiWidth))
        .ResizeChildren();

    int display_cam_bot = ui_params.show_data_logger ? kDataLogHeight : 0;

    // 3d content to the right
    pangolin::View& display_cam = pangolin::CreateDisplay()
        .SetBounds(pangolin::Attach::Pix(display_cam_bot), 1.0, pangolin::Attach::Pix(kUiWidth), 1.0, -fw / fh) // TODO: why negative aspect?
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

    if (ui_params.show_data_logger)
    {
        std::vector<std::string> labels;
        labels.push_back(std::string("cam_cov"));
        data_log_.SetLabels(labels);

        float time_points_count = 30; // time points spread over plotter's width
        float maxY = 1; // data value spread over plotter's height
        plotter_ = std::make_unique<pangolin::Plotter>(&data_log_, 0.0f, time_points_count, 0.0f, maxY, time_points_count, maxY);
        pangolin::Plotter& plotter = *plotter_.get();
        plotter.SetBounds(0.0, pangolin::Attach::Pix(kDataLogHeight), pangolin::Attach::Pix(kUiWidth), 1.0);
        plotter.Track("$i"); // scrolls view as new value appear

        pangolin::DisplayBase().AddDisplay(plotter);
    }

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

            CameraPosState cam_state;
            ui_params.kalman_slam->GetCameraPredictedPosState(&cam_state);

            a_cam_x = cam_state.PosW[0];
            a_cam_y = cam_state.PosW[1];
            a_cam_z = cam_state.PosW[2];

            size_t sal_pnts_count = ui_params.kalman_slam->SalientPointsCount();
            if (sal_pnts_count > 0)
            {
                Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_0;
                ui_params.kalman_slam->GetSalientPointPredictedPosWithUncertainty(0, &sal_pnt_0, nullptr);
                sal_pnt_0_x = sal_pnt_0[0];
                sal_pnt_0_y = sal_pnt_0[1];
                sal_pnt_0_z = sal_pnt_0[2];
            }
            if (sal_pnts_count > 1)
            {
                Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_1;
                ui_params.kalman_slam->GetSalientPointPredictedPosWithUncertainty(1, &sal_pnt_1, nullptr);
                sal_pnt_1_x = sal_pnt_1[0];
                sal_pnt_1_y = sal_pnt_1[1];
                sal_pnt_1_z = sal_pnt_1[2];
            }

            if (has_ground_truth)
            {
                FragmentMap::DependsOnSalientPointIdInfrustructure();
                if (sal_pnts_count > 0)
                {
                    const SalientPointFragment& sal_pnt_0_gt = ui_params.entire_map->GetSalientPointByInternalOrder(0);
                    sal_pnt_0_x_gt = sal_pnt_0_gt.Coord.value()[0];
                    sal_pnt_0_y_gt = sal_pnt_0_gt.Coord.value()[1];
                    sal_pnt_0_z_gt = sal_pnt_0_gt.Coord.value()[2];
                }

                if (sal_pnts_count > 1)
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
                display_3D_uncertainties,
                ui_swallow_exc);

            // pull data log entries from tracker
            {
                std::lock_guard<std::mutex> lk(ui_params.worker_chat->tracker_and_ui_mutex_);
                while (!ui_params.plotter_data_log_exchange_buf->empty())
                {
                    const PlotterDataLogItem& item = ui_params.plotter_data_log_exchange_buf->front();
                    ui_params.plotter_data_log_exchange_buf->pop_front();
                    data_log_.Log(static_cast<float>(item.MaxCamPosUncert));
                }
            }
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

void SceneVisualizationThread(UIThreadParams ui_params) // parameters by value across threads
{
    VLOG(4) << "UI thread is running";

    SceneVisualizationPangolinGui::s_ui_params = ui_params;
    SceneVisualizationPangolinGui pangolin_gui;
    pangolin_gui.Run();

    VLOG(4) << "UI thread is exiting";
}

}
#endif