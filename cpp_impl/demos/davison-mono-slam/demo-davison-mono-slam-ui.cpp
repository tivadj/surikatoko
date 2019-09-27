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
#include <opencv2/imgproc.hpp> // cv::circle, cv::resize
#endif

namespace suriko_demos_davison_mono_slam
{
using namespace std::literals::chrono_literals;
using namespace suriko;
using namespace suriko::internals;

constexpr void MarkUsedTrackerStateToVisualize() {}

/// theta goes counter-clockwise from the major axis
auto EllipsePntPolarToEuclid(Scalar semi_major, Scalar semi_minor, Scalar theta) -> suriko::Point2f
{
    SRK_ASSERT(semi_major > 0);
    SRK_ASSERT(semi_minor > 0);

    const auto a = semi_major;
    const auto b = semi_minor;

    Scalar cos_theta = std::cos(theta);
    Scalar sin_theta = std::sin(theta);

    // Polar ellipse https://en.wikipedia.org/wiki/Ellipse
    // r=a*b/sqrt((b*cos(theta))^2 + (a*sin(theta))^2)
    Scalar r = a * b / std::sqrt(suriko::Sqr(b * cos_theta) + suriko::Sqr(a * sin_theta));

    suriko::Point2f p;
    p[0] = r * cos_theta;
    p[1] = r * sin_theta;
    return p;
}

SrkColor GetSalientPointColor(const TrackedSalientPoint& sal_pnt)
{
    SrkColor new_sal_pnt_color{ 0, 255, 0 }; // green
    SrkColor matched_sal_pnt_color{ 255, 0, 0 }; // red
    SrkColor unobserved_sal_pnt_color{ 255, 255, 0 }; // yellow
    SrkColor default_sal_pnt_color{ 255, 255, 255 };
    SrkColor* sal_pnt_color = &default_sal_pnt_color;
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

#if defined(SRK_HAS_PANGOLIN)

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

std::array<GLfloat, 3> GLColorRgb(SrkColor c)
{
    return std::array<GLfloat, 3> {
        static_cast<GLfloat>(c.rgb_[0]),
        static_cast<GLfloat>(c.rgb_[1]),
        static_cast<GLfloat>(c.rgb_[2])
    };
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
    Scalar half_width = static_cast<Scalar>(cam_instrinsics.image_size.width / 2.0) / alpha[0];
    Scalar half_height = static_cast<Scalar>(cam_instrinsics.image_size.height / 2.0) / alpha[1];

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

    glLineWidth(1);

    // draw narrow ellipse as a line
    if (IsClose(0, sorted_semi_axes[Middle].second))
    {
        Scalar semi_major = sorted_semi_axes[Largest].second;
        suriko::Point3 p1 = SE3Apply(rot_ellipsoid.world_from_ellipse, suriko::Point3{  semi_major, 0, 0 });
        suriko::Point3 p2 = SE3Apply(rot_ellipsoid.world_from_ellipse, suriko::Point3{ -semi_major, 0, 0 });

        glBegin(GL_LINES);
        glVertex3d(p1[0], p1[1], p1[2]);
        glVertex3d(p2[0], p2[1], p2[2]);
        glEnd();
        return;
    }

    // The ellipsoid is drawn as an ellipse, built on two eigenvectors with largest eigenvalues.
    // ellipse axes OX,OY=Largest,Middle

    glBegin(GL_LINE_LOOP);
    for (size_t i = 0; i < dots_per_ellipse; ++i)
    {
        // draw in the plane of largest-2nd-largest eigenvectors
        Scalar theta = i * (2 * Pi<Scalar>()) / dots_per_ellipse;
        suriko::Point2f eucl = EllipsePntPolarToEuclid(sorted_semi_axes[Largest].second, sorted_semi_axes[Middle].second, theta);

        // ellipse OX is on largest, ellipse OY on the 2nd-largest

        Point3 ws;
        ws[sorted_semi_axes[Largest].first] = eucl[0];  // ellipse OX
        ws[sorted_semi_axes[Middle].first] = eucl[1];  // ellipse OY
        ws[sorted_semi_axes[Smallest].first] = 0;

        auto pos_world = SE3Apply(rot_ellipsoid.world_from_ellipse, ws);
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
            Pi<Scalar>() - ang_delta,
            Pi<Scalar>() + ang_delta
        };
        glBegin(GL_LINES);
        for (Scalar ang : thick_stroke_angs)
        {
            // draw in the plane of middle-smallest eigenvectors
            suriko::Point2f eucl = EllipsePntPolarToEuclid(sorted_semi_axes[Middle].second, sorted_semi_axes[Smallest].second, ang);

            Eigen::Matrix<Scalar, 3, 1> ws;
            ws[sorted_semi_axes[Middle].first] = eucl[0];  // ellipse OX
            ws[sorted_semi_axes[Smallest].first] = eucl[1];  // ellipse OY
            ws[sorted_semi_axes[Largest].first] = 0;

            auto pos_world = SE3Apply(rot_ellipsoid.world_from_ellipse, ToPoint3(ws));
            glVertex3d(pos_world[0], pos_world[1], pos_world[2]);
        }
        glEnd();
    }
}

void RenderPosUncertaintyMatAsEllipsoid(
    const Point3& pos,
    const Eigen::Matrix<Scalar, 3, 3>& pos_uncert,
    Scalar covar3D_to_ellipsoid_chi_square,
    size_t dots_per_ellipse,
    bool ui_swallow_exc)
{
    auto [op, rot_ellipsoid] = GetRotatedUncertaintyEllipsoidFromCovMat(pos_uncert, pos, covar3D_to_ellipsoid_chi_square);
    static_assert(std::is_same_v<decltype(rot_ellipsoid), RotatedEllipsoid3D>);
    if (!op)
    {
        if (ui_swallow_exc) return;
        SRK_ASSERT(op);
    }

    // draw projection of ellipsoid
    RenderEllipsoid(rot_ellipsoid, dots_per_ellipse);
}

void RenderCameraTrajectory(const std::vector<SE3Transform>& gt_cam_orient_cfw,
    const CameraIntrinsicParams& cam_instrinsics,
    const std::array<GLfloat, 3>& track_color,
    bool display_trajectory,
    CamDisplayType mid_cam_disp_type)
{
    Point3 cam_pos_world_prev;
    bool cam_pos_world_prev_inited = false;

    for (size_t i = 0; i < gt_cam_orient_cfw.size(); ++i)
    {
        const auto& cam_cfw = gt_cam_orient_cfw[i];
        const SE3Transform& cam_wfc = SE3Inv(cam_cfw);

        // get position of the camera in the world: cam_to_world*(0,0,0,1)=cam_pos
        const Point3& cam_pos_world = cam_wfc.T;

        if (display_trajectory && cam_pos_world_prev_inited)
        {
            glBegin(GL_LINES);
            glColor3fv(track_color.data());
            glVertex3d(cam_pos_world_prev[0], cam_pos_world_prev[1], cam_pos_world_prev[2]);
            glVertex3d(cam_pos_world[0], cam_pos_world[1], cam_pos_world[2]);
            glEnd();
        }

        RenderSchematicCamera(cam_wfc, cam_instrinsics, track_color, mid_cam_disp_type);

        cam_pos_world_prev = cam_pos_world;
        cam_pos_world_prev_inited = true;
    }
}

void RenderSalientTemplate(const DavisonMonoSlam* mono_slam, DavisonMonoSlam::SalPntId sal_pnt_id)
{
    MarkUsedTrackerStateToVisualize();
    std::optional<SalPntRectFacet> rect = mono_slam->ProtrudeSalientTemplateIntoWorld(sal_pnt_id);
    if (!rect.has_value())
        return;

    const TrackedSalientPoint& sal_pnt = mono_slam->GetSalientPoint(sal_pnt_id);

    bool in_virtual_mode = sal_pnt.initial_templ_gray_.empty();
    if (in_virtual_mode)
    {
        // in virtual mode render just an outline of the template

        // iterate vertices in such an order, that the front of template will face the camera
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
        // render an image of rectangular template, associated with salient point
        // NOTE: glTexImage2D requires a texture size to be 2^N
        const GLsizei tex_width = static_cast<GLsizei>(CeilPow2N(mono_slam->sal_pnt_templ_size_.width));
        const GLsizei tex_height = static_cast<GLsizei>(CeilPow2N(mono_slam->sal_pnt_templ_size_.height));

        glEnable(GL_TEXTURE_2D);

        std::array<GLuint, 1> texId{ 0 };
        glGenTextures((GLsizei)texId.size() /*num of textures*/, texId.data());

        // always convert to RGB/BGR because OpenGL can't get gray images as an input (glTexImage2D.format parameter)
        GLenum src_texture_format = GL_BGR;
        static cv::Mat tex_bgr(tex_height, tex_width, CV_8UC3);

        // put template into the bottom-left corner of the texture
        cv::Rect templ_bounds{ 0, tex_height - mono_slam->sal_pnt_templ_size_.height, mono_slam->sal_pnt_templ_size_.width, mono_slam->sal_pnt_templ_size_.height };
        cv::Mat templ_submat{ tex_bgr, templ_bounds };

        bool templ_constructed = false;
#if defined(SRK_DEBUG)
        // visualize colored templates if available as it simplifies recognition of a scene
        // NOTE: need a copy of the template because later it is flipped
        if (!sal_pnt.initial_templ_bgr_debug.empty())
        {
            sal_pnt.initial_templ_bgr_debug.copyTo(templ_submat);
            templ_constructed = true;
        }
#endif
        if (!templ_constructed)
        {
            cv::cvtColor(sal_pnt.initial_templ_gray_, templ_submat, cv::COLOR_GRAY2BGR);
        }

        // cv::Mat must be prepared to be used as texture in OpenGL, see https://stackoverflow.com/questions/16809833/opencv-image-loading-for-opengl-texture
        // OpenCV stores images from top to bottom, while the GL uses bottom to top
        cv::flip(tex_bgr, tex_bgr, 0 /*0=around x-axis*/);

        //set length of one complete row in data (doesn't need to equal image.cols)
        const int row_length = static_cast<int>(tex_bgr.step / tex_bgr.elemSize());
        glPixelStorei(GL_UNPACK_ROW_LENGTH, row_length);

        //use fast 4-byte alignment (default anyway) if possible
        const int row_alignment = static_cast<int>((tex_bgr.step & 3) ? 1 : 4);
        glPixelStorei(GL_UNPACK_ALIGNMENT, row_alignment);

        glBindTexture(GL_TEXTURE_2D, texId[0]);
        const GLint gl_texture_format = GL_RGB;  // GL_LUMINANCE (for color->gray conversion) or GL_RGB, both work
        glTexImage2D(GL_TEXTURE_2D, 0, gl_texture_format, tex_width, tex_height, 0, src_texture_format, GL_UNSIGNED_BYTE, tex_bgr.data);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);  // do not mix texture color with background

        // template is in the bottom-left corner of the texture
        GLfloat tex_max_x = static_cast<GLfloat>(mono_slam->sal_pnt_templ_size_.width) / tex_width;
        GLfloat tex_max_y = static_cast<GLfloat>(mono_slam->sal_pnt_templ_size_.height) / tex_height;

        using PointIndAndTexCoord = std::tuple<size_t, std::array <GLfloat, 2>>;
        std::array< PointIndAndTexCoord, 4> vertex_and_tex_coords = {
            PointIndAndTexCoord { SalPntRectFacet::kTopRightInd, {tex_max_x, tex_max_y}},
            PointIndAndTexCoord { SalPntRectFacet::kTopLeftInd, {0.0f, tex_max_y}},
            PointIndAndTexCoord { SalPntRectFacet::kBotLeftInd, {0.0f, 0.0f}},
            PointIndAndTexCoord { SalPntRectFacet::kBotRightInd, {tex_max_x, 0.0f}},
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
        glDeleteTextures((GLsizei)texId.size(), texId.data());
        glDisable(GL_TEXTURE_2D); // may be unnecessary
    }
}

void RenderMap(const DavisonMonoSlam* mono_slam, Scalar covar3D_to_ellipsoid_chi_square,
    bool display_3D_uncertainties,
    size_t dots_per_ellipse,
    bool ui_swallow_exc)
{
    for (DavisonMonoSlam::SalPntId sal_pnt_id : mono_slam->GetSalientPoints())
    {
        const TrackedSalientPoint& sal_pnt = mono_slam->GetSalientPoint(sal_pnt_id);

        SrkColor sal_pnt_color = GetSalientPointColor(sal_pnt);
        glColor3fv(GLColorRgb(sal_pnt_color).data());

        MarkUsedTrackerStateToVisualize();
        Point3 sal_pnt_pos;
        Eigen::Matrix<Scalar, 3, 3> sal_pnt_pos_uncert;
        bool got_3d_pos = mono_slam->GetSalientPointEstimated3DPosWithUncertaintyNew(sal_pnt_id, &sal_pnt_pos, &sal_pnt_pos_uncert);
        if (got_3d_pos)
        {
            if (display_3D_uncertainties)
            {
                RenderPosUncertaintyMatAsEllipsoid(sal_pnt_pos, sal_pnt_pos_uncert, covar3D_to_ellipsoid_chi_square, dots_per_ellipse, ui_swallow_exc);
            }

            glBegin(GL_POINTS);
            glVertex3d(sal_pnt_pos[0], sal_pnt_pos[1], sal_pnt_pos[2]);
            glEnd();

            // render template 'cards' only if the salient point was found in current frame
            // NOTE: the estimated pos of a salient point and a corresponding template rectangle (which is a 3D unprojection
            // of salient point pixels template) may be visually off, which indicates some errors in estimation
            if (sal_pnt.IsDetected())
            {
                RenderSalientTemplate(mono_slam, sal_pnt_id);
            }
            else
            {
                SRK_ASSERT(true);
            }
        }
        else
        {
            // TODO: sal pnt in inf; render
            // render unity direction vector from the center of the first camera the salient point is seen
        }
    }
}

void RenderScene(const UIThreadParams& ui_params, const DavisonMonoSlam* mono_slam, const CameraIntrinsicParams& cam_instrinsics, Scalar covar3D_to_ellipsoid_chi_square,
    bool display_trajectory,
    CamDisplayType mid_cam_disp_type,
    bool display_3D_uncertainties,
    bool display_ground_truth,
    size_t dots_per_ellipse,
    bool ui_swallow_exc)
{
    // World (ground truth) is rendered inside 'world' coordinates.
    // A tracker is positioned somewhere inside world and is rendered inside its own frame, usually the first camera position.
    // A world is chosen as the primary (identity) origin because one may think of multiple trackers inside the world.

    if (bool has_gt_data = ui_params.entire_map != nullptr; has_gt_data && display_ground_truth)
    {
        // world axes
        RenderAxes(1, 4);

        if (bool has_gt_cameras = ui_params.gt_cam_orient_cfw != nullptr)
        {
            std::array<GLfloat, 3> track_color{ 232 / 255.0f, 188 / 255.0f, 87 / 255.0f }; // browny
            RenderCameraTrajectory(*ui_params.gt_cam_orient_cfw, cam_instrinsics, track_color, display_trajectory,
                mid_cam_disp_type);
        }
        if (bool has_gt_sal_pnts = ui_params.entire_map != nullptr)
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
    }

    {
        // the scene is drawn in the coordinate system of a tracker (=cam0)

        std::array<double, 4 * 4> world_from_tracker_4x4_by_col{};
        const SE3Transform world_from_tracker = SE3Inv(ui_params.tracker_origin_from_world);
        LoadSE3TransformIntoOpengGLMat(world_from_tracker, world_from_tracker_4x4_by_col);

        glPushMatrix();
        glMultMatrixd(world_from_tracker_4x4_by_col.data());

        RenderAxes(0.5, 2); // axes of the tracker's origin (=cam0)

        RenderMap(mono_slam, covar3D_to_ellipsoid_chi_square, display_3D_uncertainties, dots_per_ellipse, ui_swallow_exc);

        // render history of camera's positions (schematic)
        std::array<float, 3> actual_track_color{ 128 / 255.0f, 255 / 255.0f, 255 / 255.0f }; // cyan
        if (ui_params.cam_orient_cfw_history != nullptr)
        {
            RenderCameraTrajectory(*ui_params.cam_orient_cfw_history, cam_instrinsics, actual_track_color, display_trajectory,
                mid_cam_disp_type);
        }

        // render current (the latest) camera position
        {
            MarkUsedTrackerStateToVisualize();
            CameraStateVars cam_vars = mono_slam->GetCameraEstimatedVars();
            SE3Transform cam_wfc = CamWfc(cam_vars);
            RenderSchematicCamera(cam_wfc, cam_instrinsics, actual_track_color, CamDisplayType::Schematic);
        }

        if (display_3D_uncertainties)
        {
            Point3 cam_pos;
            Eigen::Matrix<Scalar, 3, 3> cam_pos_uncert;
            Eigen::Matrix<Scalar, 4, 1> cam_orient_quat_wfc;
            mono_slam->GetCameraEstimatedPosAndOrientationWithUncertainty(&cam_pos, &cam_pos_uncert, &cam_orient_quat_wfc);

            Eigen::Matrix<Scalar, 3, 3> cam_orient_wfc;
            RotMatFromQuat(gsl::make_span<const Scalar>(cam_orient_quat_wfc.data(), 4), &cam_orient_wfc);

            RenderPosUncertaintyMatAsEllipsoid(cam_pos, cam_pos_uncert, covar3D_to_ellipsoid_chi_square, dots_per_ellipse, ui_swallow_exc);
        }

        glPopMatrix();
    }
}

UIThreadParams SceneVisualizationPangolinGui::s_ui_params_;
std::weak_ptr<SceneVisualizationPangolinGui> SceneVisualizationPangolinGui::s_this_ui_;

void SceneVisualizationPangolinGui::InitUI()
{
    float fw = 1280;
    float fh = 960;
    int w = (int)fw;
    int h = (int)fh;

    pangolin::CreateWindowAndBind("3DReconstr", w, h);
    glEnable(GL_DEPTH_TEST);

    float center_x = fw / 2;
    float center_y = fh / 2;

    view_state_3d_ = std::make_unique<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(w, h, 420, 420, center_x, center_y, 0.2, 1500),
        pangolin::ModelViewLookAt(30, -30, 30, 0, 0, 0, pangolin::AxisY)
    );

    int kUiWidth = 160;

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
    cb_displ_ground_truth_ = std::make_unique<pangolin::Var<bool>>("ui.displ_gt", false, true);
    slider_mid_cam_type_ = std::make_unique<pangolin::Var<int>>("ui.mid_cam_type", 1, 0, 2);
    cb_displ_mid_cam_type_ = std::make_unique<pangolin::Var<bool>>("ui.displ_3D_uncert", true, true);
    button_set_viewer_behind_camera_ = std::make_unique<pangolin::Var<bool>>("ui.view_behind_cam", false, false);

    // Pangolin doesn't allow generic key handler, so have to set a specific handler for all possible input keys
    for (int key : allowed_key_pressed_codes_)
    {
        pangolin::RegisterKeyPressCallback(key, [key]()
        {
            if (auto form = s_this_ui_.lock())
                form->OnKeyPressed(key);
        });
    }

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

    // support frame_ind=-1, when no frames available, the scene may be rendered too
    ptrdiff_t frame_ind = ui_params.get_observable_frame_ind_fun();

    *a_frame_ind_ = frame_ind;

    display_cam->Activate(*view_state_3d_);

    if (pangolin::Pushed(*button_set_viewer_behind_camera_))
        SetCameraBehindTracker();

    bool display_trajectory = cb_displ_traj_->Get();
    bool display_3D_uncertainties = cb_displ_mid_cam_type_->Get();
    bool display_ground_truth = cb_displ_ground_truth_->Get();

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

    RenderScene(ui_params, ui_params.mono_slam, cam_instrinsics_, ui_params.covar3D_to_ellipsoid_chi_square,
                display_trajectory,
                mid_cam_disp_type,
                display_3D_uncertainties,
                display_ground_truth,
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
        
        if (key_.has_value())
        {
            if (key_predicate(key_.value()))
                break;
            if (key_pressed_handler_ != nullptr)
            {
                KeyHandlerResult handler_result = key_pressed_handler_(key_.value());

                // if stop is requested then break out without resetting the pressed key
                if (handler_result.stop_wait_loop) break;
                if (handler_result.handled)
                    key_ = std::nullopt;
            }
        }
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

        //got_user_input_ = false;

        RenderFrame();
        pangolin::FinishFrame(); // also processes user input

        TreatAppCloseAsEscape();

        if (key_.has_value())
        {
            if (break_on(key_.value()))
                break;
            if (key_pressed_handler_ != nullptr)
            {
                KeyHandlerResult handler_result = key_pressed_handler_(key_.value());
                if (handler_result.handled)
                    key_ = std::nullopt;
                if (handler_result.stop_wait_loop) break;
            }
        }
        
        if (got_user_input_)
        {
            //LOG(INFO) << "UI is prolonged";

            // continue execution of UI loop for couple of seconds
            auto now = std::chrono::steady_clock::now();
            prolonged_ui_loop_end_time = now + ui_loop_prolong_period_;
            got_user_input_ = false;
        }

        std::this_thread::sleep_for(ui_tight_loop_relaxing_delay_);
        the_first_iter = false;
    }
    return key_;
}

void SceneVisualizationPangolinGui::SetOnRenderOutputFilePath(const std::string& filename_prefix) const
{
    display_cam->SaveOnRender(filename_prefix);
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
    VLOG(6) << "pressed key '" <<(char)key << "'(" << key << ")";
    got_user_input_ = true;
    key_ = key;

    bool key_handled = false;

    // check if worker request finishing UI thread
    const auto& ui_params = s_ui_params_;
    if (ui_params.mono_slam->in_multi_threaded_mode_)
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
        if (ui_params.mono_slam->in_multi_threaded_mode_)
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

void SceneVisualizationPangolinGui::SetCameraBehindTracker()
{
    const SE3Transform& tfw = s_ui_params_.tracker_origin_from_world;  // transform tracker from world

    const DavisonMonoSlam& mono_slam = *s_ui_params_.mono_slam;

    // take camera's coordinates from filter
    CameraStateVars cam_vars = mono_slam.GetCameraEstimatedVars();
    SE3Transform cam_tfc = CamWfc(cam_vars);
    SE3Transform cam_cft = SE3Inv(cam_tfc);

    static Scalar behind_cam_dist = 5;

    Eigen::Matrix<Scalar, 4, 4, Eigen::ColMajor> model_view =
        internals::SE3Mat(std::nullopt, Point3{0, 0, behind_cam_dist})*
        internals::SE3Mat(cam_cft.R, cam_cft.T)*
        internals::SE3Mat(tfw.R, tfw.T);
    auto model_view_span = gsl::make_span(model_view.data(), model_view.size());
    internals::ConvertAxesHartleyZissermanToOpenGL(model_view_span);

    pangolin::OpenGlMatrix model_view_col_major;
    std::copy_n(model_view.data(), model_view.size(), model_view_col_major.m);

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

cv::Scalar OcvColorBgr(SrkColor c)
{
    return cv::Scalar {
        static_cast<double>(c.rgb_[2]),  // reverse order of color components
        static_cast<double>(c.rgb_[1]),
        static_cast<double>(c.rgb_[0])
    };
}

enum class LineType
{
    Solid,
    Dashed,
    Dotted
};

/// Draws ellipse in camera plane by dividing it into points and projecting/distorting them into pixels.
void DrawDistortedEllipseOnPicture(const RotatedEllipse2D& ellipse_pix,
    size_t dots_per_ellipse,
    cv::Scalar color,
    LineType line_type,
    std::function<suriko::Point2f(suriko::Point2f)> transform_ellipse_pnt_fun,
    cv::Mat* camera_image_bgr)
{
    // draw narrow ellipse as a line
    if (IsClose(0, ellipse_pix.semi_axes[1]))
    {
        Scalar semi_major = ellipse_pix.semi_axes[0];
        suriko::Point2f p1 = SE2Apply(ellipse_pix.world_from_ellipse, suriko::Point2f{  semi_major, 0 });
        suriko::Point2f p2 = SE2Apply(ellipse_pix.world_from_ellipse, suriko::Point2f{ -semi_major, 0 });
        cv::line(*camera_image_bgr, 
            cv::Point{ static_cast<int>(p1.X()), static_cast<int>(p1.Y()) },
            cv::Point{ static_cast<int>(p2.X()), static_cast<int>(p2.Y()) },
            color);
        return;
    }

    std::optional<cv::Point> pnt_int_prev;
    for (size_t i = 0; i <= dots_per_ellipse; ++i)
    {
        Scalar theta = i * (2 * Pi<Scalar>()) / dots_per_ellipse;
        suriko::Point2f eucl = EllipsePntPolarToEuclid(ellipse_pix.semi_axes[0], ellipse_pix.semi_axes[1], theta);

        Eigen::Matrix<Scalar, 2, 1> ws = eucl.Mat();
        suriko::Point2f pnt_pix = SE2Apply(ellipse_pix.world_from_ellipse, suriko::Point2f{ ws });

        if (transform_ellipse_pnt_fun != nullptr)
            pnt_pix = transform_ellipse_pnt_fun(pnt_pix);

        cv::Point pnt_int{ static_cast<int>(pnt_pix[0]), static_cast<int>(pnt_pix[1]) };

        if (pnt_int_prev.has_value())
        {
            if (line_type == LineType::Solid || line_type == LineType::Dashed && i % 2 == 0)
                cv::line(*camera_image_bgr, pnt_int_prev.value(), pnt_int, color);
        }
        if (line_type == LineType::Dotted)
            cv::drawMarker(*camera_image_bgr, pnt_int, color, cv::MarkerTypes::MARKER_SQUARE, 1); // square of size 1 = point
        pnt_int_prev = pnt_int;
    }
}

void DavisonMonoSlam2DDrawer::DrawEstimatedSalientPoint(const DavisonMonoSlam& mono_slam, SalPntId sal_pnt_id,
    cv::Mat* out_image_bgr) const
{
    const TrackedSalientPoint& sal_pnt = mono_slam.GetSalientPoint(sal_pnt_id);
    SrkColor sal_pnt_color = GetSalientPointColor(sal_pnt);
    cv::Scalar sal_pnt_color_bgr = OcvColorBgr(sal_pnt_color);

    // we draw ellipse as a current representation of an area where a salient point is positioned
    MarkUsedTrackerStateToVisualize();
    auto [op_corner, corner_ellipse] = mono_slam.GetSalientPointProjectedUncertEllipse(FilterStageType::Estimated, sal_pnt_id);
    if (op_corner)
    {
        static_assert(std::is_same_v<decltype(corner_ellipse), RotatedEllipse2D>);
        DrawDistortedEllipseOnPicture(corner_ellipse, dots_per_uncert_ellipse_, sal_pnt_color_bgr, LineType::Solid, nullptr, out_image_bgr);
    }

    // we draw a rectangle of a search area, where a salient point is expected to be in the next frame
    if (auto [op, predict_pos_ellipse] = mono_slam.GetPredictedSalientPointProjectedUncertEllipse(sal_pnt_id); op)
    {
        static_assert(std::is_same_v<decltype(predict_pos_ellipse), RotatedEllipse2D>);
        Rect corner_bounds = GetEllipseBounds2(predict_pos_ellipse);
        Recti predict_pos_rect = EncompassRect(corner_bounds);

        // highlight salient points outside of predicted search rect
        SrkColor search_rect_color = { 192, 192, 192 };
        if (op_corner)
        {
            Point2i corner{ corner_ellipse.world_from_ellipse.T[0], corner_ellipse.world_from_ellipse.T[1] };
            bool is_in =
                corner.x > predict_pos_rect.x && corner.x < predict_pos_rect.Right() &&
                corner.y > predict_pos_rect.y && corner.y < predict_pos_rect.Bottom();
            if (!is_in)
                search_rect_color = SrkColor{ 255,255,0 };
        }

        cv::Scalar search_rect_color_bgr = OcvColorBgr(search_rect_color);

        cv::rectangle(*out_image_bgr,
            cv::Point{ predict_pos_rect.x, predict_pos_rect.y }, cv::Point{ predict_pos_rect.Right(), predict_pos_rect.Bottom() }, search_rect_color_bgr);
    }
}

void DavisonMonoSlam2DDrawer::DrawScene(const DavisonMonoSlam& mono_slam, cv::Mat* out_image_bgr) const
{
    for (SalPntId sal_pnt_id : mono_slam.GetSalientPoints())
    {
        DrawEstimatedSalientPoint(mono_slam, sal_pnt_id, out_image_bgr);
    }
}
#endif

}