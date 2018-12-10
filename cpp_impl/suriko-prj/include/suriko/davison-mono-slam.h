#pragma once
#include <set>
#include <vector>
#include <memory>
#include <set>
#include <shared_mutex>
#include <functional>
#include <gsl/span>
#include "suriko/obs-geom.h"

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core/core.hpp> // cv::Mat
#endif

namespace suriko {
namespace
{
    constexpr Scalar kNan = std::numeric_limits<Scalar>::quiet_NaN();
    constexpr size_t kEucl3 = 3; // x: 3 for position
    constexpr size_t kQuat4 = 4; // q: 4 for quaternion orientation
    constexpr size_t kVelocComps = kEucl3; // v: 3 for velocity
    constexpr size_t kAngVelocComps = kEucl3; // w: 3 for angular velocity
    constexpr size_t kAccelComps = kEucl3; // a: 3 for acceleration
    constexpr size_t kAngAccelComps = kEucl3; // alpha: 3 for angular acceleration
    constexpr size_t kPixPosComps = 2; // rows and columns
    constexpr Scalar kCamPlaneZ = 1; // z=1 in [x,y,1]

    // [x q v w], x: 3 for position, q: 4 for quaternion orientation, v: 3 for velocity, w: 3 for angular velocity
    constexpr size_t kCamStateComps = kEucl3 + kQuat4 + kVelocComps + kAngVelocComps; // 13
    constexpr size_t kInputNoiseComps = kVelocComps + kAngVelocComps; // Qk.rows: velocity and angular velocity are updated an each iteration by noise
    constexpr size_t kSalientPointPolarCompsCount = 3; // [theta elevation rho], theta: 1 for azimuth angle, 1 for elevation angle, rho: 1 for distance
    constexpr size_t kRhoComps = 1; // inverse distance
    constexpr size_t kSalientPointComps = kEucl3 + kSalientPointPolarCompsCount;

    void DependsOnOverallPackOrder() {}
    void DependsOnCameraPosPackOrder() {}
    void DependsOnSalientPointPackOrder() {}
    void DependsOnInputNoisePackOrder() {}
}

/// Opaque data to represent detected in image corner.
struct CornersMatcherBlobId
{
    size_t Ind; // possibly, index of blob in the blobs array
};

/// Helper class to transfer around a gray and BGR image from camera. The BGR image is used for debugging purposes.
struct ImageFrame
{
    cv::Mat gray;
#if defined(SRK_DEBUG)
    cv::Mat bgr_debug;
#endif
};

void CopyBgr(const ImageFrame& image, cv::Mat* out_image_bgr);

/// Status of a salient point during tracking.
enum class SalPntTrackStatus
{
    New,         // created in current frame
    Matched,     // observed and matched to one in some previous frame
    Unobserved   // undetected in current frame
};

/// Represents the portion of the image, which is the projection of salient image into a camera.
struct SalPntPatch
{
    size_t estim_vars_ind; // index into X[13+6N,1] and P[13+6N,13+6N] matrices
    size_t sal_pnt_ind; // order of the salient point in the sequence of salient points

    // The distorted coordinates in the current camera, corresponds to the center of the image template.
    suriko::Point2 pixel_coord_real;

    suriko::Pointi template_top_left_int;

    SalPntTrackStatus track_status;

    // Rectangular portion of the gray image corresponding to salient point, projected in current frame.
    cv::Mat template_gray_in_first_frame;
#if defined(SRK_DEBUG)
    cv::Mat template_bgr_in_first_frame_debug;
#endif

    Eigen::Matrix<Scalar, kPixPosComps, 1> OffsetFromTopLeft() const
    {
        return pixel_coord_real.Mat() - Eigen::Matrix<Scalar, kPixPosComps, 1>{template_top_left_int.x, template_top_left_int.y};
    }
};

/// Represents publicly transferable key to refer to a salient point.
/// It is valid even if other salient points are removed or new salient points added.
struct SalPntId
{
    union
    {
        SalPntPatch* sal_pnt_internal;
        ptrdiff_t sal_pnt_as_bits_internal;
    };

    SalPntId() = default;
    SalPntId(SalPntPatch* sal_pnt) : sal_pnt_internal(sal_pnt) {}
    operator bool() const { return sal_pnt_internal != nullptr; }
};
inline bool operator<(SalPntId x, SalPntId y) { return x.sal_pnt_as_bits_internal < y.sal_pnt_as_bits_internal; }

/// We separate the tracking of existing salient points, for which the position in the latest
/// frame is known, from occasional search for extra salient points.
/// The first is the hot path in workflow when camera doesn't move much. The number of salient
/// points is constant. It should be fast.
/// The second case incur the growth of estimated variables' covariance matrix, and this is 
/// an extra overhead we pursue to avoid.
class CornersMatcherBase
{
public:
    virtual ~CornersMatcherBase() = default;

    virtual void AnalyzeFrame(size_t frame_ind, const ImageFrame& image) {}
    virtual void OnSalientPointIsAssignedToBlobId(SalPntId sal_pnt_id, CornersMatcherBlobId blob_id, const ImageFrame& image) {}

    virtual void MatchSalientPoints(size_t frame_ind,
        const ImageFrame& image,
        const std::set<SalPntId>& tracking_sal_pnts,
        std::vector<std::pair<SalPntId, CornersMatcherBlobId>>* matched_sal_pnts) {}

    virtual void RecruitNewSalientPoints(size_t frame_ind,
        const ImageFrame& image,
        const std::set<SalPntId>& tracking_sal_pnts,
        const std::vector<std::pair<SalPntId, CornersMatcherBlobId>>& matched_sal_pnts,
        std::vector<CornersMatcherBlobId>* new_blob_ids) {}

    virtual suriko::Point2 GetBlobCoord(CornersMatcherBlobId blob_id) = 0;
    
    virtual ImageFrame GetBlobPatchTemplate(CornersMatcherBlobId blob_id, const ImageFrame& image) {
        return ImageFrame{};
    }

    virtual std::optional<Scalar> GetSalientPointGroundTruthInvDepth(CornersMatcherBlobId blob_id) { return std::nullopt; };
};

/// ax=f/dx and ay=f/dy 
/// (alpha_x = focal_length_x_meters / pixel_width_meters)
struct CameraIntrinsicParams
{
    suriko::Sizei image_size;  // [width, height] image resolution

    std::array<Scalar, 2> principal_point_pix; // [Cx,Cy] in pixels

    Scalar focal_length_mm;  // =f, focal length in millimiters
    std::array<Scalar,2> pixel_size_mm; // [dx,dy] in millimeters

    /// Focal length in pixels (alphax=f/dx, alphay=f/dy)
    std::array<Scalar, 2> FocalLengthPix() const { return { focal_length_mm / pixel_size_mm[0], focal_length_mm / pixel_size_mm[1] }; }
};

/// scale_factor=1+k1*r^2+k2*r^4
struct RadialDistortionParams
{
    Scalar k1;
    Scalar k2;
};

struct CameraStateVars
{
    Eigen::Matrix<Scalar, kEucl3, 1> pos_w; // in world frame
    Eigen::Matrix<Scalar, kQuat4, 1> orientation_wfc;
    Eigen::Matrix<Scalar, kVelocComps, 1> velocity_w; // in world frame
    Eigen::Matrix<Scalar, kAngVelocComps, 1> angular_velocity_c; // in camera frame
};

SE3Transform CamWfc(const CameraStateVars& cam_state);

enum class FilterStageType
{
    Estimated,  // current state, k
    Predicted   // state at step k+1
};

struct MeanAndCov2D
{
    Eigen::Matrix<Scalar, kPixPosComps, 1> mean;
    Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> cov;
};

struct SalPntRectFacet
{
    std::array<suriko::Point3,4> points;

    static constexpr size_t kTopLeftInd = 0;
    static constexpr size_t kTopRightInd = 1;
    static constexpr size_t kBotLeftInd = 2;
    static constexpr size_t kBotRightInd = 3;

    suriko::Point3& TopLeft() { return points[kTopLeftInd]; }
    suriko::Point3& TopRight() { return points[kTopRightInd]; }
    suriko::Point3& BotLeft() { return points[kBotLeftInd]; }
    suriko::Point3& BotRight() { return points[kBotRightInd]; }
};

/// Represents a state of the tracker.
struct DavisonMonoSlamTrackerInternalsSlice
{
    std::chrono::duration<double> frame_processing_dur; // frame processing duration
    Eigen::Matrix<Scalar, 3, 1> cam_pos_w;
    Eigen::Matrix<Scalar, 3, 3> cam_pos_uncert;
    Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps> cam_state_uncert;
    std::optional<Eigen::Matrix<Scalar, 3, 3>> sal_pnts_uncert_median; // median of uncertainty of all salient points; null if there are 0 salient points
    Scalar cur_reproj_err;
    size_t estimated_sal_pnts; // number of tracking salient points (which are stored in estimated variables array)
    size_t new_sal_pnts; // number of new salient points allocated in current frame
    size_t common_sal_pnts; // number of same salient points in the previous and current frame
    size_t deleted_sal_pnts; // number of deleted salient points in current frame
};

/// Represents the history of the tracker processing a sequence of frames.
struct DavisonMonoSlamTrackerInternalsHist
{
    std::chrono::duration<double> avg_frame_processing_dur;
    std::vector<DavisonMonoSlamTrackerInternalsSlice> state_samples;
};

class DavisonMonoSlam;

/// Base class for logging statistics of tracker.
class DavisonMonoSlamInternalsLogger
{
    using Clock = std::chrono::high_resolution_clock;
    DavisonMonoSlam* mono_slam_ = nullptr;
    DavisonMonoSlamTrackerInternalsSlice cur_stats_;
    DavisonMonoSlamTrackerInternalsHist hist_;
    Clock::time_point frame_start_time_point_;
    Clock::time_point frame_finish_time_point_;
public:
    DavisonMonoSlamInternalsLogger(DavisonMonoSlam* mono_slam);
    virtual ~DavisonMonoSlamInternalsLogger() = default;

    virtual void StartNewFrameStats();
    virtual void FinishFrameStats();
    virtual void NotifyNewComDelSalPnts(size_t new_count, size_t common_count, size_t deleted_count);
    virtual void NotifyEstimatedSalPnts(size_t estimated_sal_pnts_count);

    DavisonMonoSlamTrackerInternalsHist& BuildStats();
};

/// Implementation of MonoSlam by Andrew Davison https://www.doc.ic.ac.uk/~ajd/Scene/index.html
/// The algorithm uses Kalman filter to estimate camera's location and map features (salient points).
/// source: book "Structure from Motion using the Extended Kalman Filter" Civera 2011 (further SfM_EKF_Civera)
class DavisonMonoSlam
{
public:
    static constexpr size_t kCamStateComps = kCamStateComps;
    static constexpr Scalar kFiniteDiffEpsDebug = (Scalar)1e-5; // used for debugging derivatives

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> EigenDynMat;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> EigenDynVec;
public:
    enum class DebugPathEnum
    {
        DebugNone             = 0,
        DebugCamCov           = 1 << 1,
        DebugSalPntCov        = 1 << 2,
        DebugEstimVarsCov     = 1 << 3,
        DebugPredictedVarsCov = 1 << 4
    };
    //inline DebugPathEnum operator|(DebugPathEnum a, DebugPathEnum b)
    //{
    //    return static_cast<DebugPathEnum>(static_cast<int>(a) | static_cast<int>(b));
    //}

    using SalPntId = SalPntId;
private:
    static DebugPathEnum s_debug_path_;

    EigenDynVec estim_vars_; // x[13+N*6], camera position plus all salient points
    EigenDynMat estim_vars_covar_; // P[13+N*6, 13+N*6], state's covariance matrix
    std::vector<std::unique_ptr<SalPntPatch>> sal_pnts_; // the set of tracked salient points
    std::set<SalPntId> sal_pnts_as_ids_; // the set ids of tracked salient points
    std::set<SalPntId> latest_frame_sal_pnts_; // contains subset of salient points which were tracked in the latest frame

    Eigen::Matrix<Scalar, kSalientPointComps, kSalientPointComps> input_noise_covar_; // Qk[6,6] input noise covariance matrix

    std::shared_mutex predicted_estim_vars_mutex_;
    EigenDynVec predicted_estim_vars_; // x[13+N*6]
    EigenDynMat predicted_estim_vars_covar_; // P[13+N*6, 13+N*6]
public:
    bool in_multi_threaded_mode_ = false;  // true to expect the clients to read predicted vars from different thread; locks are used to protect from conflicting access
    Scalar between_frames_period_ = 1; // elapsed time between two consecutive frames
    Scalar input_noise_std_ = 1;
    Scalar measurm_noise_std_ = 1;
    Scalar sal_pnt_init_inv_dist_ = 1; // rho0, the inverse depth of a salient point in the first camera in which the point is seen
    Scalar sal_pnt_init_inv_dist_std_ = 1; // std(rho0)
    
    // width and height of a patch template of a salient point
    // Davison used patches of 15x15 (see "Simultaneous localization and map-building using active vision" para 3.1, Davison, Murray, 2002)
    suriko::Sizei sal_pnt_patch_size_ = { 15, 15 };

    std::optional<int> debug_max_sal_pnt_coun_;

    // camera
    CameraIntrinsicParams cam_intrinsics_{};
    RadialDistortionParams cam_distort_params_{};
public:
    std::function<SE3Transform(size_t frame_ind)> gt_cami_from_tracker_fun_;  // gets ground truth camera position in coordinates of tracker
    Scalar debug_ellipsoid_cut_thr_ = 0.04; // value 0.05 corresponds to 2sig
    bool fake_localization_ = false; // true to get camera orientation from ground truth
    bool fake_sal_pnt_initial_inv_dist_ = false; // true to correctly initialize points depth in virtual environments

    /// There are 3 implementations of incorporating m observed corners (corner=pixel, 2x1 mat).
    /// 1. Stack all corners in one [2m,1] vector. Require inverting one [2m,2m] innovation matrix.
    /// 2. Process each corner individually. Require inverting m innovation matrices of size [2x2].
    /// 3. Process [x,y] component of each corner individually. Require inverting 2m scalars.
    int kalman_update_impl_ = 0;

    bool fix_estim_vars_covar_symmetry_ = false;
private:
    std::unique_ptr<CornersMatcherBase> corners_matcher_;
    std::unique_ptr<DavisonMonoSlamInternalsLogger> stats_logger_;
private:
    struct
    {
        EigenDynMat R; // R[2m,2m]
        EigenDynMat H; // H[2m,13+N*6]

        EigenDynVec zk; // [2m,1]
        EigenDynVec projected_sal_pnts; // [2m,1]

        EigenDynMat filter_gain; // P[13+N*6, 2m] m=number of observed points
        EigenDynMat innov_var; // [13+N*6, 13+N*6]
        EigenDynMat innov_var_inv; // P[13+N*6, 13+N*6]
        EigenDynMat H_P; // H*P, [13+N*6, 13+N*6]
        EigenDynMat Knew; // H*P, [13+N*6, 13+N*6]
        EigenDynMat estim_vars_covar_new; // P[13+N*6, 13+N*6]
        EigenDynMat K_S; // K*S, [13+N*6, 2m]
    } stacked_update_cache_;
    struct
    {
        Eigen::Matrix<Scalar, Eigen::Dynamic, kPixPosComps> P_Hxy; // P*Hx or P*Hy, [13+N*6, 2]
        Eigen::Matrix<Scalar, Eigen::Dynamic, kPixPosComps> Knew; // K[13+N*6, 2]
        Eigen::Matrix<Scalar, Eigen::Dynamic, kPixPosComps> K_S; // K*S, [13+N*6, 2]
    } one_obs_per_update_cache_;
    struct
    {
        EigenDynVec Knew; // [13+N*6, 1]
    } one_comp_of_obs_per_update_cache_;
    struct
    {
        Eigen::Matrix<Scalar, kSalientPointComps, Eigen::Dynamic> J_P; // [7,13+N*6]
    } add_sal_pnt_cache_;
public:
    DavisonMonoSlam();

    void SetCamera(const SE3Transform& cam_pos_cfw, Scalar estim_var_init_std);
    
    void SetInputNoiseStd(Scalar input_noise_std);

    void ProcessFrame(size_t frame_ind, const ImageFrame& image);

    void PredictEstimVarsHelper();

    suriko::Point2 ProjectCameraPoint(const suriko::Point3& pnt_camera) const;

    size_t EstimatedVarsCount() const;

    CameraStateVars GetCameraEstimatedVars();
    CameraStateVars GetCameraPredictedVars();

    void GetCameraEstimatedPosAndOrientationWithUncertainty(Eigen::Matrix<Scalar, kEucl3,1>* pos_mean, 
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert,
        Eigen::Matrix<Scalar, kQuat4, 1>* orient_quat);

    void GetCameraEstimatedVarsUncertainty(Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* cam_covar) const;

    suriko::Point2 GetSalPntPixelCoord(SalPntId sal_pnt_id) const;

    bool GetSalientPointEstimated3DPosWithUncertaintyNew(SalPntId sal_pnt_id,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert);

    bool GetSalientPointPredicted3DPosWithUncertaintyNew(SalPntId sal_pnt_id,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert);

    auto GetSalientPointProjected2DPosWithUncertainty(FilterStageType filter_stage, SalPntId sal_pnt_id)
        ->MeanAndCov2D;

    Scalar CurrentFrameReprojError() const;

    size_t SalientPointsCount() const;

    const std::set<SalPntId>& GetSalientPoints() const;

    SalPntPatch& GetSalientPoint(SalPntId id);
    const SalPntPatch& GetSalientPoint(SalPntId id) const;

    suriko::Pointi TemplateTopLeftInt(const suriko::Point2& center) const;

    /// This returns null if the salient point is in the infinity and finite coordinates of patch template can't be calculated.
    std::optional<SalPntRectFacet> GetEstimatedSalPntFaceRect(SalPntId sal_pnt_id) const;

    void SetCornersMatcher(std::unique_ptr<CornersMatcherBase> corners_matcher);
    CornersMatcherBase& CornersMatcher();

    void SetStatsLogger(std::unique_ptr<DavisonMonoSlamInternalsLogger> stats_logger);
    DavisonMonoSlamInternalsLogger* StatsLogger() const;

    static void SetDebugPath(DebugPathEnum debug_path);
private:
    struct SalPntProjectionIntermidVars
    {
        Eigen::Matrix<Scalar, kEucl3, 1> hc; // euclidean position of salient point in camera coordinates
        Eigen::Matrix<Scalar, kEucl3, 1> first_cam_sal_pnt_unity_dir; // unity direction from first camera to the salient point in world coordinates
    };

    struct SalientPointStateVars
    {
        Eigen::Matrix<Scalar, kEucl3, 1> first_cam_pos_w; // the position of the camera (in world frame) the salient point was first seen
        
        // polar coordinates of the salient point in the camera where the feature was seen for the first time
        Scalar azimuth_theta_w; // theta=azimuth, rotates clockwise around worldOY, zero corresponds to worldOZ direction
        Scalar elevation_phi_w; // elevation=latin_phi, rotates clockwise around worldOX, zero corresponds to worldOZ direction
        Scalar inverse_dist_rho; // inverse distance (=rho) from the first camera, where the salient point was seen the first time, to this salient point
    };

    void ResetCamera(Scalar estim_var_init_std, bool init_estim_vars = true);

    void CheckCameraAndSalientPointsCovs(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar) const;

    auto GetFilterState(FilterStageType filter_stage) -> std::tuple<EigenDynVec*, EigenDynMat*>;

    CameraStateVars GetCameraStateVars(FilterStageType filter_stage);

    void GetCameraPosAndOrientationWithUncertainty(FilterStageType filter_stage,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert,
        Eigen::Matrix<Scalar, kQuat4, 1>* orient_quat);

    bool GetSalientPoint3DPosWithUncertaintyHelper(FilterStageType filter_stage, SalPntId sal_pnt_id,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert);

    void FillRk(size_t obs_sal_pnt_count, EigenDynMat* Rk) const;
    void FillRk2x2(Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* Rk) const;

    void PredictCameraMotionByKinematicModel(gsl::span<const Scalar> cam_state, gsl::span<Scalar> new_cam_state,
        const Eigen::Matrix<Scalar, kInputNoiseComps, 1>* noise_state = nullptr,
        bool normalize_quat = true) const;
    void PredictEstimVars(EigenDynVec* predicted_estim_vars, EigenDynMat* predicted_estim_vars_covar) const;

    void ProcessFrame_StackedObservationsPerUpdate(size_t frame_ind);
    void ProcessFrame_OneObservationPerUpdate(size_t frame_ind);
    void ProcessFrame_OneComponentOfOneObservationPerUpdate(size_t frame_ind);
    void OnEstimVarsChanged(size_t frame_ind);

    SalPntId AddSalientPoint(const CameraStateVars& cam_state, suriko::Point2 corner, 
        ImageFrame patch_template,
        std::optional<Scalar> pnt_inv_dist_gt);

    gsl::span<Scalar> EstimVarsCamPosW();
    Eigen::Matrix<Scalar, kQuat4, 1> EstimVarsCamQuat() const;
    Eigen::Matrix<Scalar, kAngVelocComps, 1> EstimVarsCamAngularVelocity() const;
    size_t SalientPointOffset(size_t sal_pnt_ind) const;
    //inline SalPntInternal& GetSalPnt(SalPntId id);
    //inline const SalPntInternal& GetSalPnt(SalPntId id) const;

    void LoadCameraStateVarsFromArray(gsl::span<const Scalar> src, CameraStateVars* result) const;

    void LoadSalientPointDataFromArray(gsl::span<const Scalar> src, SalientPointStateVars* result) const;
    SalientPointStateVars LoadSalientPointDataFromSrcEstimVars(const EigenDynVec& src_estim_vars, const SalPntPatch& sal_pnt) const;

    void PropagateSalPntPosUncertainty(const SalientPointStateVars& sal_pnt,
        const Eigen::Matrix<Scalar, kSalientPointComps, kSalientPointComps>& sal_pnt_covar,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* sal_pnt_pos_uncert) const;

    /// NOTE: The resultant 2D uncertainty does depend on the uncertainty of the camera frame in which the salient point is projected.
    auto GetSalientPointProjected2DPosWithUncertainty(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar,
        const SalPntPatch& sal_pnt) ->MeanAndCov2D;

    /// NOTE: The resultant uncertainty doesn't respect uncertainty of the current camera frame.
    bool GetSalientPoint3DPosWithUncertainty(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar,
        const SalPntPatch& sal_pnt,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const;

    /// Ensures that given salient point can be correctly handled (rendering, position prediction etc).
    void CheckSalientPoint(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar, 
        const SalPntPatch& sal_pnt) const;

    std::optional<SalPntRectFacet> GetSalientPointFaceRect(const EigenDynVec& src_estim_vars, const SalPntPatch& sal_pnt) const;

    void PixelCoordinateToCamera(const Eigen::Matrix<Scalar, kPixPosComps, 1>& hu, Eigen::Matrix<Scalar, kEucl3, 1>* pos_camera) const;

    //

    void Deriv_cam_state_by_cam_state(Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* result) const;

    void FiniteDiff_cam_state_by_cam_state(gsl::span<const Scalar> cam_state, Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* result) const;

    void Deriv_cam_state_by_input_noise(Eigen::Matrix<Scalar, kCamStateComps, kInputNoiseComps>* result) const;

    void FiniteDiff_cam_state_by_input_noise(Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kCamStateComps, kInputNoiseComps>* result) const;

    void Deriv_Hrowblock_by_estim_vars(const SalPntPatch& sal_pnt,
        const CameraStateVars& cam_state, const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const EigenDynVec& derive_at_pnt,
        Eigen::Matrix<Scalar, kPixPosComps, Eigen::Dynamic>* Hrowblock_by_estim_vars) const;

    void Deriv_hd_by_cam_state_and_sal_pnt(const SalPntPatch& sal_pnt,
        const CameraStateVars& cam_state, const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const EigenDynVec& derive_at_pnt,
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_cam_state,
        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt,
        Eigen::Matrix<Scalar, kPixPosComps, 1>* hd = nullptr) const;

    void Deriv_H_by_estim_vars(const CameraStateVars& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const EigenDynVec& derive_at_pnt,
        EigenDynMat* H_by_estim_vars) const;

    // Derivative of distorted observed corner (in pixels) by undistorted observed corner (in pixels).
    void Deriv_hu_by_hd(suriko::Point2 corner_pix, Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* hu_by_hd) const;
    void Deriv_hd_by_hu(suriko::Point2 corner_pix, Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* hd_by_hu) const;

    // Derivative of distorted observed corner (in pixels) by camera's state variables
    void Deriv_hu_by_hc(const SalPntProjectionIntermidVars& proj_hist, Eigen::Matrix<Scalar, kPixPosComps, kEucl3 >* dhu_by_dhc) const;

    void Deriv_R_by_q(const Eigen::Matrix<Scalar, kQuat4, 1>& q,
        Eigen::Matrix<Scalar, 3, 3>* dR_by_dq0,
        Eigen::Matrix<Scalar, 3, 3>* dR_by_dq1,
        Eigen::Matrix<Scalar, 3, 3>* dR_by_dq2,
        Eigen::Matrix<Scalar, 3, 3>* dR_by_dq3) const;

    // Derivative of distorted observed corner (in pixels) by camera's state variables (13 vars).
    void Deriv_hd_by_camera_state(const SalientPointStateVars& sal_pnt,
        const CameraStateVars& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const SalPntProjectionIntermidVars& proj_hist,
        const Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>& hd_by_dhu,
        const Eigen::Matrix<Scalar, kPixPosComps, kEucl3>& dhu_by_dhc,
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_xc) const;

    // Derivative of distorted observed corner (in pixels) by salient point's variables (6 vars).
    void Deriv_hd_by_sal_pnt(const SalientPointStateVars& sal_pnt,
        const CameraStateVars& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>& hd_by_dhu,
        const Eigen::Matrix<Scalar, kPixPosComps, kEucl3>& hu_by_dhc,
        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt) const;

    void Deriv_azim_theta_elev_phi_by_hw(
        const Eigen::Matrix<Scalar, kEucl3, 1>& hw,
        Eigen::Matrix<Scalar, 1, kEucl3>* azim_theta_by_hw,
        Eigen::Matrix<Scalar, 1, kEucl3>* elev_phi_by_hw) const;

    void Deriv_sal_pnt_by_cam_q(const CameraStateVars& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, 1>& hc, 
        const Eigen::Matrix<Scalar, 1, kEucl3>& azim_theta_by_hw,
        const Eigen::Matrix<Scalar, 1, kEucl3>& elev_phi_by_hw,
        Eigen::Matrix<Scalar, kSalientPointComps, kQuat4>* sal_pnt_by_cam_q) const;

    void FiniteDiff_hd_by_camera_state(const EigenDynVec& derive_at_pnt,
        const SalientPointStateVars& sal_pnt,
        Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_xc) const;
    
    void FiniteDiff_hd_by_sal_pnt_state(const CameraStateVars& cam_state, 
        const SalPntPatch& sal_pnt,
        const EigenDynVec& derive_at_pnt,
        Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_y) const;


    // derivative of qk+1 (next step camera orientation) by wk (camera orientation)
    void Deriv_q3_by_w(Scalar deltaT, Eigen::Matrix<Scalar, kQuat4, kEucl3>* result) const;
    
    static void CameraCoordinatesEuclidUnityDirFromPolarAngles(Scalar azimuth_theta, Scalar elevation_phi, Scalar* hx, Scalar* hy, Scalar* hz);
    
    // projection

    Eigen::Matrix<Scalar, kPixPosComps, 1> ProjectCameraSalientPoint(
        const Eigen::Matrix<Scalar, kEucl3, 1>& pnt_camera,
        SalPntProjectionIntermidVars *proj_hist) const;

    std::optional<Eigen::Matrix<Scalar, kEucl3, 1>> InternalSalientPointToCamera(
        const SalientPointStateVars& sal_pnt_vars,
        const CameraStateVars& cam_state,
        bool scaled_by_inv_dist,
        SalPntProjectionIntermidVars *proj_hist) const;

    Eigen::Matrix<Scalar, kPixPosComps, 1> ProjectInternalSalientPoint(const CameraStateVars& cam_state, const SalientPointStateVars& sal_pnt_vars, SalPntProjectionIntermidVars *proj_hist) const;

    RotatedEllipse2D ApproxProjectEllipsoidOnCameraByBeaconPoints(const Ellipsoid3DWithCenter& ellipsoid, const CameraStateVars& cam_state);
    RotatedEllipse2D ApproxProjectEllipsoidOnCameraByBeaconPoints(const RotatedEllipsoid3D& rot_ellipsoid, const CameraStateVars& cam_state);
    
    void FixSymmetricMat(EigenDynMat* sym_mat) const;

    static bool DebugPath(DebugPathEnum debug_path);
};

inline DavisonMonoSlam::DebugPathEnum operator|(DavisonMonoSlam::DebugPathEnum a, DavisonMonoSlam::DebugPathEnum b)
{
    return static_cast<DavisonMonoSlam::DebugPathEnum>(static_cast<int>(a) | static_cast<int>(b));
}
inline DavisonMonoSlam::DebugPathEnum operator&(DavisonMonoSlam::DebugPathEnum a, DavisonMonoSlam::DebugPathEnum b)
{
    return static_cast<DavisonMonoSlam::DebugPathEnum>(static_cast<int>(a) & static_cast<int>(b));
}

}
