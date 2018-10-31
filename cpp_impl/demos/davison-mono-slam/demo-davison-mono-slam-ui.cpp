#include "demo-davison-mono-slam-ui.h"
#include <random>
#include <chrono>
#include <thread>
#include <glog/logging.h>
#include "suriko/approx-alg.h"
#include "suriko/quat.h"
#include "suriko/obs-geom.h"
#include "suriko/opengl-helpers.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core/core.hpp> // cv::Mat
#include <opencv2/imgproc.hpp> // cv::circle
#endif

#if defined(SRK_HAS_PANGOLIN)
namespace suriko_demos_davison_mono_slam
{
using namespace std::literals::chrono_literals;
using namespace suriko;
using namespace suriko::internals;

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

void RenderSchematicCamera(const SE3Transform& cam_wfc, const CameraIntrinsicParams& cam_instrinsics, const std::array<float, 3>& track_color, CamDisplayType cam_disp_type)
{
    if (cam_disp_type == CamDisplayType::None)
        return;

    // transform to the camera frame
    std::array<double, 4 * 4> opengl_mat_by_col{};
    LoadSE3TransformIntoOpengGLMat(cam_wfc, opengl_mat_by_col);

    glPushMatrix();
    glMultMatrixd(opengl_mat_by_col.data());

    // draw camera in the local coordinates
    constexpr Scalar cam_z = kCamPlaneZ;

    std::array<Scalar, 2> alpha = cam_instrinsics.FocalLengthPix();
    Scalar half_width = static_cast<Scalar>(cam_instrinsics.image_size[0] / 2.0) / alpha[0];
    Scalar half_height = static_cast<Scalar>(cam_instrinsics.image_size[0] / 2.0) / alpha[1];

    double cam_skel[5][3] = {
        {0, 0, 0},
        {half_width, half_height, cam_z}, // left top
        {-half_width, half_height, cam_z }, // right top
        {-half_width, -half_height, cam_z}, // right bot
        {half_width, -half_height, cam_z}, // left bot
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
        RenderAxes(cam_z, 2);

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

auto EllipsePntPolarToEuclid(Scalar a, Scalar b, Scalar theta) -> suriko::Point2
{
    Scalar cos_theta = std::cos(theta);
    Scalar sin_theta = std::sin(theta);

    // Polar ellipse https://en.wikipedia.org/wiki/Ellipse
    // r=a*b/sqrt((b*cos(theta))^2 + (a*sin(theta))^2)
    Scalar r = a * b / std::sqrt(suriko::Sqr(b * cos_theta) + suriko::Sqr(a * sin_theta));

    suriko::Point2 p;
    p[0] = r * cos_theta;
    p[1] = r * sin_theta;
    return p;
}

void RenderEllipsoid(const RotatedEllipsoid3D& rot_ellipsoid, size_t dots_per_ellipse)
{
    std::array<std::pair<size_t, Scalar>, 3> sorted_semi_axes;
    sorted_semi_axes[0] = { 0, rot_ellipsoid.semi_axes[0] };
    sorted_semi_axes[1] = { 1, rot_ellipsoid.semi_axes[1] };
    sorted_semi_axes[2] = { 2, rot_ellipsoid.semi_axes[2] };
    std::sort(sorted_semi_axes.begin(), sorted_semi_axes.end(), [](auto& p1, auto& p2)
    {
        // sort descending by length of semi-axis
        return p1.second > p2.second;
    });

    enum Axes { Largest, Middle, Smallest };

    // The ellipsoid is drawn as an ellipse, built on two eigenvectors with largest eigenvalues.
    // ellipse axes OX,OY=Largest,Middle

    glLineWidth(1);
    glBegin(GL_LINE_LOOP);
    for (size_t i = 0; i < dots_per_ellipse; ++i)
    {
        // draw in the plane of largest-2nd-largest eigenvectors
        Scalar theta = i * (2 * M_PI) / dots_per_ellipse;
        suriko::Point2 eucl = EllipsePntPolarToEuclid(sorted_semi_axes[Largest].second, sorted_semi_axes[Middle].second, theta);

        // ellipse OX is on largest, ellipse OY on the 2nd-largest

        Eigen::Matrix<Scalar, 3, 1> ws;
        ws[sorted_semi_axes[Largest].first] = eucl[0];  // ellipse OX
        ws[sorted_semi_axes[Middle].first] = eucl[1];  // ellipse OY
        ws[sorted_semi_axes[Smallest].first] = 0;

        // map to the original ellipse with axes not parallel to world axes
        ws += rot_ellipsoid.center_e;

        Eigen::Matrix<Scalar, 3, 1> pos_world = rot_ellipsoid.rot_mat_world_from_ellipse * ws;
        glVertex3d(pos_world[0], pos_world[1], pos_world[2]);
    }
    glEnd();

    const bool almost_circle = sorted_semi_axes[Smallest].second / sorted_semi_axes[Largest].second > 0.9f;
    if (almost_circle)
    {
        // render two strokes, orthogonal to ellipse plane, to represent the ellipsoid is almost a sphere
        // ellipse axes OX,OY=Middle,Smallest
        const Scalar ang_delta = Deg2Rad(5);
        std::array<Scalar, 4> thick_stroke_angs = {
            -ang_delta,
            ang_delta,
            M_PI - ang_delta,
            M_PI + ang_delta
        };
        glBegin(GL_LINES);
        for (Scalar ang : thick_stroke_angs)
        {
            // draw in the plane of middle-smallest eigenvectors
            suriko::Point2 eucl = EllipsePntPolarToEuclid(sorted_semi_axes[Middle].second, sorted_semi_axes[Smallest].second, ang);

            Eigen::Matrix<Scalar, 3, 1> ws;
            ws[sorted_semi_axes[Middle].first] = eucl[0];  // ellipse OX
            ws[sorted_semi_axes[Smallest].first] = eucl[1];  // ellipse OY
            ws[sorted_semi_axes[Largest].first] = 0;

            // map to the original ellipse with axes not parallel to world axes
            ws += rot_ellipsoid.center_e;

            Eigen::Matrix<Scalar, 3, 1> pos_world = rot_ellipsoid.rot_mat_world_from_ellipse * ws;
            glVertex3d(pos_world[0], pos_world[1], pos_world[2]);
        }
        glEnd();
    }
}

void RenderUncertaintyEllipsoid(
    const Eigen::Matrix<Scalar, 3, 1>& pos,
    const Eigen::Matrix<Scalar, 3, 3>& pos_uncert,
    Scalar ellipsoid_cut_thr,
    size_t dots_per_ellipse,
    bool ui_swallow_exc)
{
    Ellipsoid3DWithCenter ellipsoid;
    ExtractEllipsoidFromUncertaintyMat(pos, pos_uncert, ellipsoid_cut_thr, &ellipsoid);

    RotatedEllipsoid3D rot_ellipsoid;
    bool op1 = GetRotatedEllipsoid(ellipsoid, !ui_swallow_exc, &rot_ellipsoid);
    if (!op1)
    {
        if (ui_swallow_exc)
            return;
        else SRK_ASSERT(op1);
    }

    // draw projection of ellipsoid
    RenderEllipsoid(rot_ellipsoid, dots_per_ellipse);
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
    const CameraIntrinsicParams& cam_instrinsics,
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
            RenderSchematicCamera(cam_wfc, cam_instrinsics, track_color, last_cam_disp_type);
        else
            RenderSchematicCamera(cam_wfc, cam_instrinsics, track_color, mid_cam_disp_type);

        cam_pos_world_prev = cam_pos_world;
        cam_pos_world_prev_inited = true;
    }
}

std::array<GLfloat, 3> GetSalientPointColor(const SalPntInternal& sal_pnt)
{
    std::array<GLfloat, 3> new_sal_pnt_color{ 0, 255, 0 }; // green
    std::array<GLfloat, 3> matched_sal_pnt_color{ 255, 0, 0 }; // red
    std::array<GLfloat, 3> unobserved_sal_pnt_color{ 255, 255, 0 }; // yellow
    std::array<GLfloat, 3> default_sal_pnt_color{ 255, 255, 255 };
    std::array<GLfloat, 3>* sal_pnt_color = &default_sal_pnt_color;
    switch (sal_pnt.track_status)
    {
    case SalPntTrackStatus::New:
        sal_pnt_color = &new_sal_pnt_color;
        break;
    case SalPntTrackStatus::Matched:
        sal_pnt_color = &matched_sal_pnt_color;
        break;
    case SalPntTrackStatus::Unobserved:
        sal_pnt_color = &unobserved_sal_pnt_color;
        break;
    default:
        sal_pnt_color = &default_sal_pnt_color;
        break;
    }
    return *sal_pnt_color;
}

void RenderSalientPointPatchTemplate(DavisonMonoSlam* kalman_slam, DavisonMonoSlam::SalPntId sal_pnt_id)
{
    std::optional<SalPntRectFacet> rect = kalman_slam->GetPredictedSalPntFaceRect(sal_pnt_id);
    if (!rect.has_value())
        return;

    const SalPntInternal& sal_pnt = kalman_slam->GetSalientPoint(sal_pnt_id);

    bool in_virtual_mode = sal_pnt.template_in_first_frame.empty();
    if (in_virtual_mode)
    {
        // in virtual mode render just an outline of the patch

        // iterate vertices in such an order, that the front of patch will face the camera
        static constexpr std::array<size_t, 4> rect_inds = {
            SalPntRectFacet::kTopRightInd,
            SalPntRectFacet::kTopLeftInd,
            SalPntRectFacet::kBotLeftInd,
            SalPntRectFacet::kBotRightInd
        };

        glBegin(GL_LINE_LOOP);
        for (auto i : rect_inds)
        {
            const auto& x = rect.value().points[i];
            glVertex3d(x[0], x[1], x[2]);
        }
        glEnd();
    }
    else
    {
        // render an image of rectangular patch, associated with salient point
        const GLsizei texWidth = kalman_slam->sal_pnt_patch_size_[0];
        const GLsizei texHeight = kalman_slam->sal_pnt_patch_size_[1];

        // bind the texture
        glEnable(GL_TEXTURE_2D);
        GLuint texId = -1;
        glGenTextures(1 /*num of textures*/, &texId);

        // always convert to RGB because OpenGL can't get gray images as an input (glTexImage2D.format parameter)
        const GLenum src_texture_format = GL_RGB;
        cv::Mat patch_rgb;
#if defined(SRK_DEBUG)
        patch_rgb = sal_pnt.template_rgb_in_first_frame_debug.clone();  // need cloning because it farther flipped
#else
        cv::cvtColor(sal_pnt.template_in_first_frame, patch_rgb, CV_GRAY2RGB);
#endif
        // cv::Mat must be prepared to be used as texture in OpenGL, see https://stackoverflow.com/questions/16809833/opencv-image-loading-for-opengl-texture
        // OpenCV stores images from top to bottom, while the GL uses bottom to top
        cv::flip(patch_rgb, patch_rgb, 0 /*0=around x-axis*/);

        //set length of one complete row in data (doesn't need to equal image.cols)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, patch_rgb.step / patch_rgb.elemSize());

        //use fast 4-byte alignment (default anyway) if possible
        glPixelStorei(GL_UNPACK_ALIGNMENT, (patch_rgb.step & 3) ? 1 : 4);

        glBindTexture(GL_TEXTURE_2D, texId);
        const GLenum gl_texture_format = GL_RGB;  // GL_LUMINANCE (for gray) or GL_RGB, both work
        glTexImage2D(GL_TEXTURE_2D, 0, gl_texture_format, texWidth, texHeight, 0, src_texture_format, GL_UNSIGNED_BYTE, patch_rgb.data);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);  // do not mix texture color with background

        using PointIndAndTexCoord = std::tuple<size_t, std::array <GLfloat, 2>>;
        std::array< PointIndAndTexCoord, 4> vertex_and_tex_coords = {
            PointIndAndTexCoord { SalPntRectFacet::kTopRightInd, {1.0f, 1.0f}},
            PointIndAndTexCoord { SalPntRectFacet::kTopLeftInd, {0.0f, 1.0f}},
            PointIndAndTexCoord { SalPntRectFacet::kBotLeftInd, {0.0f, 0.0f}},
            PointIndAndTexCoord { SalPntRectFacet::kBotRightInd, {1.0f, 0.0f}},
        };

        glBegin(GL_QUADS);
        for (auto i : { 0, 1, 2, 3 })
        {
            const PointIndAndTexCoord& descr = vertex_and_tex_coords[i];
            const std::array <GLfloat, 2>& tex_coord = std::get<1>(descr);
            glTexCoord2f(tex_coord[0], tex_coord[1]);

            const auto& w1 = rect.value().points[std::get<0>(descr)];
            glVertex3d(w1[0], w1[1], w1[2]);
        }
        glEnd();

        // clear the texture
        glDeleteTextures(1, &texId);
        glDisable(GL_TEXTURE_2D); // may be unnecessary
    }
}

void RenderMap(DavisonMonoSlam* kalman_slam, Scalar ellipsoid_cut_thr,
    bool display_3D_uncertainties,
    size_t dots_per_ellipse,
    bool ui_swallow_exc)
{
    for (DavisonMonoSlam::SalPntId sal_pnt_id : kalman_slam->GetSalientPoints())
    {
        const SalPntInternal& sal_pnt = kalman_slam->GetSalientPoint(sal_pnt_id);

        const size_t sal_pnt_ind = sal_pnt.sal_pnt_ind;

        Eigen::Matrix<Scalar, 3, 1> sal_pnt_pos;
        Eigen::Matrix<Scalar, 3, 3> sal_pnt_pos_uncert;
        kalman_slam->GetSalientPointEstimatedPosWithUncertainty(sal_pnt_ind, &sal_pnt_pos, &sal_pnt_pos_uncert);

        std::array<GLfloat, 3> sal_pnt_color = GetSalientPointColor(sal_pnt);
        glColor3fv(sal_pnt_color.data());

        if (display_3D_uncertainties)
        {
            RenderUncertaintyEllipsoid(sal_pnt_pos, sal_pnt_pos_uncert, ellipsoid_cut_thr, dots_per_ellipse, ui_swallow_exc);
        }

        glBegin(GL_POINTS);
        glVertex3d(sal_pnt_pos[0], sal_pnt_pos[1], sal_pnt_pos[2]);
        glEnd();

        RenderSalientPointPatchTemplate(kalman_slam, sal_pnt_id);
    }
}

void RenderScene(const UIThreadParams& ui_params, DavisonMonoSlam* kalman_slam, const CameraIntrinsicParams& cam_instrinsics, Scalar ellipsoid_cut_thr,
    bool display_trajectory,
    CamDisplayType mid_cam_disp_type,
    CamDisplayType last_cam_disp_type,
    bool display_3D_uncertainties,
    size_t dots_per_ellipse,
    bool ui_swallow_exc)
{
    //glPushMatrix();
    //std::array<double, 4*4>  hartley_zisserman_to_opengl_mat;
    //GetOpenGLFromHartleyZissermanMat(gsl::make_span(hartley_zisserman_to_opengl_mat));
    //glMultMatrixd(hartley_zisserman_to_opengl_mat.data());

    // world axes
    RenderAxes(1, 4);

    bool has_gt_cameras = ui_params.gt_cam_orient_cfw != nullptr;  // gt=ground truth
    if (has_gt_cameras)
    {
        std::array<GLfloat, 3> track_color{ 232 / 255.0f, 188 / 255.0f, 87 / 255.0f }; // browny
        CamDisplayType gt_last_camera = CamDisplayType::None;
        RenderCameraTrajectory(*ui_params.gt_cam_orient_cfw, cam_instrinsics, track_color, display_trajectory,
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
            const auto& p = sal_pnt_fragm.coord.value();
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

        RenderMap(kalman_slam, ellipsoid_cut_thr, display_3D_uncertainties, dots_per_ellipse, ui_swallow_exc);

        // orientation of schematic camera is taken from history
        if (ui_params.cam_orient_cfw_history != nullptr)
        {
            std::array<float, 3> actual_track_color{ 128 / 255.0f, 255 / 255.0f, 255 / 255.0f }; // cyan
            RenderCameraTrajectory(*ui_params.cam_orient_cfw_history, cam_instrinsics, actual_track_color, display_trajectory,
                mid_cam_disp_type,
                last_cam_disp_type);
        }

        if (display_3D_uncertainties)
        {
            Eigen::Matrix<Scalar, 3, 1> cam_pos;
            Eigen::Matrix<Scalar, 3, 3> cam_pos_uncert;
            Eigen::Matrix<Scalar, 4, 1> cam_orient_quat_wfc;
            kalman_slam->GetCameraEstimatedPosAndOrientationWithUncertainty(&cam_pos, &cam_pos_uncert, &cam_orient_quat_wfc);

            Eigen::Matrix<Scalar, 3, 3> cam_orient_wfc;
            RotMatFromQuat(gsl::make_span<const Scalar>(cam_orient_quat_wfc.data(), 4), &cam_orient_wfc);

            RenderUncertaintyEllipsoid(cam_pos, cam_pos_uncert, ellipsoid_cut_thr, dots_per_ellipse, ui_swallow_exc);
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

    RenderScene(ui_params, ui_params.kalman_slam, cam_instrinsics_, ui_params.ellipsoid_cut_thr,
                display_trajectory,
                mid_cam_disp_type,
                last_cam_disp_type,
                display_3D_uncertainties,
                dots_per_uncert_ellipse_,
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
    owner_->got_user_input_ = true;
    pangolin::Handler3D::Mouse(view, button, x, y, pressed, button_state);
}

void SceneVisualizationPangolinGui::Handler3DImpl::MouseMotion(pangolin::View& view, int x, int y, int button_state)
{
    owner_->got_user_input_ = true;
    pangolin::Handler3D::MouseMotion(view, x, y, button_state);
}

void SceneVisualizationPangolinGui::Handler3DImpl::Special(pangolin::View& view, pangolin::InputSpecial inType, float x, float y, float p1, float p2, float p3, float p4, int button_state)
{
    owner_->got_user_input_ = true;
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
    pangolin::OpenGlMatrix model_view_col_major;

    Eigen::Map< Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor>> eigen_mat(static_cast<Scalar*>(model_view_col_major.m));
    eigen_mat =
        internals::SE3Mat(Eigen::Matrix<Scalar, 3, 1>{0, 0, back_dist}) *
        internals::SE3Mat(tracker_origin_from_world.R, tracker_origin_from_world.T);

    internals::ConvertAxesHartleyZissermanToOpenGL(gsl::make_span(model_view_col_major.m));

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

#endif

#if defined(SRK_HAS_OPENCV)

void DrawDistortedEllipse(const DavisonMonoSlam& tracker, const RotatedEllipse2D& ellipse, size_t dots_per_ellipse, cv::Scalar color, cv::Mat* camera_image_bgr)
{
    std::optional<cv::Point> pnt_int_prev;
    for (size_t i = 0; i <= dots_per_ellipse; ++i)
    {
        Scalar theta = i * (2 * M_PI) / dots_per_ellipse;
        suriko::Point2 eucl = EllipsePntPolarToEuclid(ellipse.semi_axes[0], ellipse.semi_axes[1], theta);

        Eigen::Matrix<Scalar, 2, 1> ws = eucl.Mat();

        // map to the original ellipse with axes not parallel to world axes
        ws += ellipse.center_e;

        // the 'world' for ellipse is the OX and OY of camera
        Eigen::Matrix<Scalar, 2, 1> pos_camera = ellipse.rot_mat_world_from_ellipse * ws;

        // due to distortion, the pixel coordinates of contour will not form an ellipse
        // hence usage of 2D ellipse drawing functions is inappropriate
        // instead we just project 3D points onto the image
        suriko::Point2 pnt_pix = tracker.ProjectCameraPoint(suriko::Point3(pos_camera[0], pos_camera[1], kCamPlaneZ));

        cv::Point pnt_int{ static_cast<int>(pnt_pix[0]), static_cast<int>(pnt_pix[1]) };

        if (pnt_int_prev.has_value())
        {
            cv::line(*camera_image_bgr, pnt_int_prev.value(), pnt_int, color);
        }
        pnt_int_prev = pnt_int;
    }
}

void DrawEllipsoidContour(DavisonMonoSlam& tracker, const CameraStateVars& cam_state, const Ellipsoid3DWithCenter& ellipsoid,
    size_t dots_per_ellipse, cv::Scalar sal_pnt_color_bgr, cv::Mat* camera_image_bgr)
{
    // The set of ellipsoid 3D points, visible from given camera position, is in implicit form and to
    // enumerate them is a problem, source: "Perspective Projection of an Ellipsoid", David Eberly, GeometricTools, https://www.geometrictools.com/
    // If it is solved, we can just enumerate them and project-distort into pixels.
    // Instead, we project those contour 3D points onto the camera (z=1) and get the ellipse,
    // 3D points of which can be easily enumerated.

    RotatedEllipse2D rotated_ellipse = tracker.ProjectEllipsoidOnCameraOrApprox(ellipsoid, cam_state);
    DrawDistortedEllipse(tracker, rotated_ellipse, dots_per_ellipse, sal_pnt_color_bgr, camera_image_bgr);
}
#endif

}