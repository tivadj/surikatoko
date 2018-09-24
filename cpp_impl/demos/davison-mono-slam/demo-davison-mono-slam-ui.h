#pragma once
#include <memory>
#include <mutex>
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

struct PlotterDataLogItem
{
    Scalar MaxCamPosUncert;
};

struct UIThreadParams
{
    DavisonMonoSlam* kalman_slam;
    Scalar ellipsoid_cut_thr;
    bool WaitForUserInputAfterEachFrame;
    std::function<size_t()> get_observable_frame_ind_fun;
    const std::vector<SE3Transform>* gt_cam_orient_cfw;
    const std::vector<SE3Transform>* cam_orient_cfw_history;
    std::deque<PlotterDataLogItem>* plotter_data_log_exchange_buf;
    const FragmentMap* entire_map;
    std::shared_ptr<WorkerThreadChat> worker_chat;
    bool show_data_logger;
    bool ui_swallow_exc;
};

class SceneVisualizationPangolinGui
{
public:
    static UIThreadParams s_ui_params;
private:
    std::unique_ptr<pangolin::Plotter> plotter_;
    pangolin::DataLog data_log_;
public:
    SceneVisualizationPangolinGui();

    static void OnForward();
    static void OnSkip();

    static void OnKeyEsc();

    void Run();
};

// parameters by value across threads
void SceneVisualizationThread(UIThreadParams ui_params);
}
#endif
