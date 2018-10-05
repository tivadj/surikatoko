#pragma once
#include <memory>
#include <mutex>
#include <chrono>
#include "suriko/rt-config.h"
#include "suriko/davison-mono-slam.h"

#if defined(SRK_HAS_PANGOLIN)
#include <pangolin/pangolin.h>

namespace suriko_demos_davison_mono_slam
{
using namespace suriko;

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

    std::mutex tracker_and_ui_mutex_;
};

struct UIThreadParams
{
    DavisonMonoSlam* kalman_slam;
    Scalar ellipsoid_cut_thr;
    bool wait_for_user_input_after_each_frame;
    std::function<size_t()> get_observable_frame_ind_fun;
    const std::vector<SE3Transform>* gt_cam_orient_cfw;
    const std::vector<SE3Transform>* cam_orient_cfw_history;
    const FragmentMap* entire_map;
    std::shared_ptr<WorkerThreadChat> worker_chat;
    bool ui_swallow_exc;
    std::chrono::milliseconds ui_tight_loop_relaxing_delay;
};

class SceneVisualizationPangolinGui
{
public:
    explicit SceneVisualizationPangolinGui(bool defer_ui_construction = false);

    /// Initializes the UI if it was not done during construction.
    void InitUI();


    void Run();
    void RenderFrame();
    void RenderFrameAndProlongUILoopOnUserInput();
private:
    static void OnForward();
    static void OnSkip();

    static void OnKeyEsc();

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
    std::chrono::milliseconds ui_tight_loop_relaxing_delay_ = std::chrono::milliseconds(1000);  // makes ui thread more 'lightweight'
    std::chrono::milliseconds ui_loop_prolong_period_ = std::chrono::milliseconds(3000);  // time from user input till ui loop finishes
private:
    static bool got_user_input_;  // indicates that a user made some input (pressed a key or clicked a mouse button)
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
}
#endif
