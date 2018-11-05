#pragma once
#include <memory>
#include <mutex>
#include <chrono>
#include "suriko/rt-config.h"
#include "suriko/davison-mono-slam.h"

#if defined(SRK_HAS_PANGOLIN)
#include <pangolin/pangolin.h>
#endif

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core/core.hpp> // cv::Mat
#endif

namespace suriko_demos_davison_mono_slam
{
using namespace suriko;

std::array<GLfloat, 3> GetSalientPointColor(const SalPntPatch& sal_pnt);

#if defined(SRK_HAS_PANGOLIN)
enum class UIChatMessage
{
    UIWaitKey,            // Worker asks UI to wait for any key pressed
    UIExit                // Worker asks UI to exit
};

enum class WorkerChatMessage
{
    WorkerKeyPressed,     // UI sends Worker a key, pressed by user
    WorkerExit,           // UI asks Worker to exit
};

struct WorkerChatSharedState
{
    std::mutex the_mutex;

    std::optional<UIChatMessage> ui_message;
    std::optional<int> ui_pressed_key; // the key which was pressed in UI
    std::function<bool(int)> ui_wait_key_predicate_ = nullptr;

    std::optional<WorkerChatMessage> worker_message;
    std::condition_variable worker_got_new_message_cv; // Worker waits for CV until UI sends a message
};

std::optional<UIChatMessage> PopMsgUnderLock(std::optional<UIChatMessage>* msg);
std::optional<WorkerChatMessage> PopMsgUnderLock(std::optional<WorkerChatMessage>* msg);

struct UIThreadParams
{
    DavisonMonoSlam* mono_slam;
    SE3Transform tracker_origin_from_world;
    Scalar ellipsoid_cut_thr;
    bool wait_for_user_input_after_each_frame;
    std::function<size_t()> get_observable_frame_ind_fun;
    const std::vector<SE3Transform>* gt_cam_orient_cfw;
    const std::vector<SE3Transform>* cam_orient_cfw_history;
    const FragmentMap* entire_map;
    std::shared_ptr<WorkerChatSharedState> worker_chat;
    bool ui_swallow_exc;
    std::chrono::milliseconds ui_tight_loop_relaxing_delay;
};

class SceneVisualizationPangolinGui
{
public:
    static std::shared_ptr<SceneVisualizationPangolinGui> New(bool defer_ui_construction = false);

    SceneVisualizationPangolinGui() = default;  // use New() instead;
    
    /// Initializes the UI if it was not done during construction.
    void InitUI();

    void RunInSeparateThread();

    int WaitKey();
    int WaitKey(std::function<bool(int key)> key_predicate);

    std::optional<int> RenderFrameAndProlongUILoopOnUserInput(std::function<bool(int key)> break_on);

    /// Puts observer behind given camera position, so that the camera will be in front of observer at distance 'back_dist'.
    void SetCameraBehindTrackerOnce(const SE3Transform& tracker_origin_from_world, float back_dist);
private:
    void RenderFrame();
    void OnKeyPressed(int key);
    void TreatAppCloseAsEscape();

    enum class FormState
    {
        IterateUILoop, // default, form just iterates in UI loop
        WaitKey        // UI waits for input key
    };
    class Handler3DImpl : public pangolin::Handler3D
    {
    public:
        Handler3DImpl(pangolin::OpenGlRenderState& cam_state);

        void Mouse(pangolin::View& view, pangolin::MouseButton button, int x, int y, bool pressed, int button_state) override;
        void MouseMotion(pangolin::View&, int x, int y, int button_state) override;
        void Special(pangolin::View&, pangolin::InputSpecial inType, float x, float y, float p1, float p2, float p3, float p4, int button_state) override;

        SceneVisualizationPangolinGui* owner_;
    };
public:
    static UIThreadParams s_ui_params_;
    
    // Pangolin's callbacks are static functions. This pointer is used in those functions to get the pointer to GUI Form object.
    static std::weak_ptr<SceneVisualizationPangolinGui> s_this_ui_;

    std::chrono::milliseconds ui_tight_loop_relaxing_delay_ = std::chrono::milliseconds(1000);  // makes ui thread more 'lightweight'
    std::chrono::milliseconds ui_loop_prolong_period_ = std::chrono::milliseconds(3000);  // time from user input till ui loop finishes
    size_t dots_per_uncert_ellipse_ = 4;
    CameraIntrinsicParams cam_instrinsics_;
private:
    bool got_user_input_ = false;  // indicates that a user made some input (pressed a key or clicked a mouse button)
    std::optional<int> key_;
    std::function<bool(int)> key_predicate_;

    struct
    {
        FormState form_state = FormState::IterateUILoop;
    } multi_threaded_;

    //
    std::unique_ptr<pangolin::OpenGlRenderState> view_state_3d_;
    pangolin::View* display_cam = nullptr;
    std::unique_ptr<Handler3DImpl> handler3d_;

    // ui elements are outside the UI loop execution function
    // (and declared here), because in single threaded scenario we want
    // the part of rendering of a UI frame to be called in isolation.
    std::unique_ptr<pangolin::Var<ptrdiff_t>> a_frame_ind_;
    std::unique_ptr<pangolin::Var<bool>> cb_displ_traj_;
    std::unique_ptr<pangolin::Var<int>> slider_mid_cam_type_;
    std::unique_ptr<pangolin::Var<bool>> cb_displ_mid_cam_type_;
};

// parameters by value across threads
void SceneVisualizationThread(UIThreadParams ui_params);
#endif

#if defined(SRK_HAS_OPENCV)
/// Draws ellipse in camera plane by dividing it into points and projecting/distorting them into pixels.
void DrawDistortedEllipse(const DavisonMonoSlam& mono_slam, const RotatedEllipse2D& ellipse, size_t dots_per_ellipse, cv::Scalar color, cv::Mat* camera_image_bgr);

/// Draw visible from given camera contour of ellipsoid.
void DrawEllipsoidContour(DavisonMonoSlam& mono_slam, const CameraStateVars& cam_state, const Ellipsoid3DWithCenter& ellipsoid,
    size_t dots_per_ellipse, cv::Scalar sal_pnt_color_bgr, cv::Mat* camera_image_bgr);

class DavisonMonoSlam2DDrawer
{
public:
    void DrawScene(DavisonMonoSlam& mono_slam, const cv::Mat& background_image_bgr, cv::Mat* out_image_bgr) const;
    
    void DavisonMonoSlam2DDrawer::DrawEstimatedSalientPoint(DavisonMonoSlam& mono_slam, const SalPntPatch& sal_pnt,
        cv::Mat* out_image_bgr) const;
public:
    Scalar ellipse_cut_thr_;
    int dots_per_uncert_ellipse_;
};
#endif

}
