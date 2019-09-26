#pragma once
#include <set>
#include <vector>
#include <memory>
#include <set>
#include <shared_mutex>
#include <functional>
#include <gsl/span>

#if defined(SRK_HAS_OPENCV)
#include <opencv2/core/core.hpp> // cv::Mat
#endif

#include "suriko/obs-geom.h"
#include "suriko/image-proc.h"

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
    constexpr size_t kProcessNoiseComps = kVelocComps + kAngVelocComps; // Qk.rows: velocity and angular velocity are updated an each iteration by noise
    constexpr size_t kSalientPointPolarCompsCount = 3; // [theta elevation rho], theta: 1 for azimuth angle, 1 for elevation angle, rho: 1 for distance
    constexpr size_t kRho = 1; // inverse distance
    constexpr size_t kXyzSalientPointComps = kEucl3;  // [XYZ]=[3x1]
    constexpr size_t kSphericalSalientPointComps = kEucl3 + kSalientPointPolarCompsCount;  // [FirstCamXYZ theta elevation rho]=[6x1]

    /// Specifies the data to store for each salient point.
    enum class SalPntComps
    {
        kXyz,                      // [3x1] [X Y Z] the Euclidean point case
        kSphericalFirstCamInvDist  // [6x1] [x y z azim elev rho] inverse depth case
    };

// The flags to specify the representation of a salient point is for debugging 
// (to demarcate the code which depends on a particular representation).
// Salient point = Euclid 3D
#define XYZ_SAL_PNT_REPRES
// Salient point = azimuth - elevation - inverse distance
#define SPHER_SAL_PNT_REPRES

// or set compiler flag eg: SAL_PNT_REPRES=1
#ifndef SAL_PNT_REPRES
#  define SAL_PNT_REPRES 2
#endif

    namespace intern
    {
#if SAL_PNT_REPRES == 1
        // XYZ salient point
        constexpr size_t kSalientPointComps = kXyzSalientPointComps;  // [3x1]
        constexpr SalPntComps kSalPntRepres = SalPntComps::kXyz;
#elif SAL_PNT_REPRES == 2
        // spherical salient point with first camera position and inverse distance
        constexpr size_t kSalientPointComps = kSphericalSalientPointComps;  // [6x1]
        constexpr SalPntComps kSalPntRepres = SalPntComps::kSphericalFirstCamInvDist;
#endif
    }

    // the index of a camera frame to choose for the origin of tracker
    constexpr size_t kTrackerOriginCamInd = 0;

    struct DavisonMonoSlamConstants
    {
        static constexpr size_t kCamStateComps = kEucl3 + kQuat4 + kVelocComps + kAngVelocComps; // 13
    };
    namespace intern
    {
        constexpr size_t kCamStateComps = kEucl3 + kQuat4 + kVelocComps + kAngVelocComps; // 13
    };

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

/// Status of a salient point during tracking.
enum class SalPntTrackStatus
{
    New,         // created in current frame
    Matched,     // observed and matched to one in some previous frame
    Unobserved,  // undetected in current frame
    Deleted      // exists as descriptor but isn't represented in state array or error covariance matrix
};

struct TemplMatchStats
{
    Scalar templ_mean_;               // the mean of a template
    Scalar templ_sqrt_sum_sqr_diff_;  // the part of denominator in formula of a correlation coefficient (=sqrt(sum))
};

/// Hints the source from which template's center is derived.
enum class SalPntTemplCenterDeriveFrom
{
    None,
    ImageBlob,      ///< salient point is detected in the frame and corresponding image blob provides the template center
    EstimatedState  ///< estimated state is projected onto the camera plane
};

// Internal
suriko::Point2i TemplateTopLeftInt(const suriko::Point2f& center, suriko::Sizei templ_size);

/// Represents the portion of the image, which is the projection of salient image into a camera.
struct TrackedSalientPoint
{
    size_t sal_pnt_ind; // order of the salient point in the sequence of salient points
    size_t estim_vars_ind; // index into X[13+6N,1] and P[13+6N,13+6N] matrices

    SalPntTrackStatus track_status;
    size_t undetected_frames_count = 0;  // number of frames for which this salient point isn't detected; 0 if it is observed.

    // The distorted coordinates in the current camera, corresponds to the center of the image template.
    std::optional <suriko::Point2f> templ_center_pix_;
    suriko::Point2f offset_from_top_left_;  // =center-top_left; initialized once for the first frame

    size_t initial_frame_ind_synthetic_only_;  // the frame ind where a salient point was first seen; used only in virtual scenarios
#if defined(SRK_DEBUG)
    SalPntTemplCenterDeriveFrom templ_center_derived_from_debug_ = SalPntTemplCenterDeriveFrom::None;
    std::optional<suriko::Point2i> templ_top_left_pix_debug_; // in pixels
    suriko::Point2f initial_templ_center_pix_debug_;          // in pixels
    suriko::Point2i initial_templ_top_left_pix_debug_;        // in pixels

    // tracking of template center; used to debug large jumps
    size_t prev_detection_frame_ind_debug_ = static_cast<size_t>(-1);  // the latest frame, where the sal pnt was detected
    suriko::Point2f prev_detection_templ_center_pix_debug_ = suriko::Point2f{-1, -1};  // in pixels
#endif

    // As the template of the image, related to this salient point, doesn't change during tracking,
    // we may cache some related statistics.
    TemplMatchStats templ_stats;

    // Rectangular portion of the gray image corresponding to salient point, projected in current frame.
    cv::Mat initial_templ_gray_;
#if defined(SRK_DEBUG)
    cv::Mat initial_templ_bgr_debug;
#endif

    /// True, for this salient point to be recognized in the current frame.
    bool IsDetected() const
    {
        return track_status == SalPntTrackStatus::New || track_status == SalPntTrackStatus::Matched;
    }

    bool IsDeleted() const { return track_status == SalPntTrackStatus::Deleted; }

    void SetTemplCenterPix(suriko::Point2f center, suriko::Sizei templ_size)
    {
        templ_center_pix_ = center;
#if defined(SRK_DEBUG)
        templ_top_left_pix_debug_ = TemplateTopLeftInt(center, templ_size);
#endif
    }

    void ResetTemplCenterPix()
    {
        templ_center_pix_ = std::nullopt;
#if defined(SRK_DEBUG)
        templ_center_derived_from_debug_ = SalPntTemplCenterDeriveFrom::None;
        templ_top_left_pix_debug_ = std::nullopt;
#endif
    }

    suriko::Point2f OffsetFromTopLeft() const { return offset_from_top_left_; }
};

/// Represents publicly transferable key to refer to a salient point.
/// It is valid even if other salient points are removed or new salient points added.
struct SalPntId
{
    union
    {
        TrackedSalientPoint* sal_pnt_internal;
        ptrdiff_t sal_pnt_as_bits_internal;
    };

    SalPntId() = default;
    constexpr SalPntId(TrackedSalientPoint* sal_pnt) : sal_pnt_internal(sal_pnt) {}
    constexpr bool HasId() const { return sal_pnt_internal != nullptr; }
    auto static constexpr Null() { return SalPntId{ nullptr }; }
};
inline bool operator<(SalPntId x, SalPntId y) { return x.sal_pnt_as_bits_internal < y.sal_pnt_as_bits_internal; }
inline bool operator==(SalPntId x, SalPntId y) { return x.sal_pnt_as_bits_internal == y.sal_pnt_as_bits_internal; }
inline bool operator!=(SalPntId x, SalPntId y) { return !operator==(x,y); }

class DavisonMonoSlam;

/// We separate the tracking of existing salient points, for which the position in the latest
/// frame is known, from occasional search for extra salient points.
/// The first is the hot path in workflow when camera doesn't move much. The number of salient
/// points is constant. It should be fast.
/// The second case incur the growth of estimated variables' covariance matrix, and this is 
/// an extra overhead we pursue to avoid.
class CornersMatcherBase
{
protected:
    bool suppress_observations_ = false; // true to make camera magically don't detect any salient points
public:
    virtual ~CornersMatcherBase() = default;

    virtual void AnalyzeFrame(size_t frame_ind, const Picture& image) {}
    virtual void OnSalientPointIsAssignedToBlobId(SalPntId sal_pnt_id, CornersMatcherBlobId blob_id, const Picture& image) {}

    virtual void MatchSalientPoints(
        const DavisonMonoSlam& mono_slam,
        const std::set<SalPntId>& tracking_sal_pnts,
        size_t frame_ind,
        const Picture& image,
        std::vector<std::pair<SalPntId, CornersMatcherBlobId>>* matched_sal_pnts) {}

    virtual void RecruitNewSalientPoints(
        const DavisonMonoSlam& mono_slam,
        const std::set<SalPntId>& tracking_sal_pnts,
        const std::vector<std::pair<SalPntId, CornersMatcherBlobId>>& matched_sal_pnts,
        size_t frame_ind,
        const Picture& image,
        std::vector<CornersMatcherBlobId>* new_blob_ids) {}

    virtual suriko::Point2f GetBlobCoord(CornersMatcherBlobId blob_id) = 0;

    /// Gets rectangle around the given blob of requested size.
    /// @param templ_size the size of requested image portion
    virtual Picture GetBlobTemplate(CornersMatcherBlobId blob_id, const Picture& image, suriko::Sizei templ_size) {
        return Picture{};
    }

    virtual std::optional<Scalar> GetSalientPointGroundTruthInvDepth(CornersMatcherBlobId blob_id) { return std::nullopt; };

    void SetSuppressObservations(bool value) { suppress_observations_ = value; }
};

/// ax=f/dx and ay=f/dy 
/// (alpha_x = focal_length_x_meters / pixel_width_meters)
struct CameraIntrinsicParams
{
    suriko::Sizei image_size;  // [width, height] image resolution

    std::array<Scalar, 2> principal_point_pix; // [Cx,Cy] in pixels

    Scalar focal_length_mm;  // =f, focal length in millimeters
    
    // Used in distortion model.
    std::array<Scalar,2> pixel_size_mm; // [dx,dy] in millimeters

    /// Focal length in pixels (alphax=f/dx, alphay=f/dy)
    std::array<Scalar, 2> FocalLengthPix() const { return { focal_length_mm / pixel_size_mm[0], focal_length_mm / pixel_size_mm[1] }; }
};

Scalar Calc_rd(const CameraIntrinsicParams& cam_intrinsics, const Point2f& h_distorted);

/// Radial distortion as described in "Introduction to modern photogrammetry", Mikhail, 2001.
/// The distortion model uses the scale factor=1+k1*r^2+k2*r^4
/// Reconstructed history:
/// In 2007 Davison used Swaminathan and Nayar distortion model as per paper "MonoSLAM Real-Time Single Camera SLAM", Davison, 2007.
/// In 2008 Davison switched to Mikhail distortion model as per paper "Inverse depth parametrization for monocular SLAM", Civera, Davison, Montiel, 2008.
struct MikhailRadialDistortionParams
{
    // Radial distortion model as in A.26: ru=rd(1+k1*rd^2+k2*rd^4)
    // (k1,k2)=(0,0) means no distortion.
    Scalar k1 = 0;  // the coefficient before rd^2, measured in mm^-2
    Scalar k2 = 0;  // the coefficient before rd^4, measured in mm^-4
};

/// Position and orientation of an object in 3D space.
struct FramePosOrient
{
    Eigen::Matrix<Scalar, kEucl3, 1> pos_w; // in world frame
    Eigen::Matrix<Scalar, kQuat4, 1> orientation_wfc;
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

/// Represents 3D point with optional known distance to it (when point is in infinity).
struct Dir3DAndDistance
{
    Eigen::Matrix<Scalar, kEucl3, 1> unity_dir;  // unity direction to the point
    std::optional<Scalar> dist;             // distance to the point or null for a point in infinity
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
    static constexpr auto kCamStateComps = intern::kCamStateComps;

    std::chrono::duration<double> frame_processing_dur; // frame processing duration
    Scalar cur_reproj_err_meas = -1;  // measured
    Scalar cur_reproj_err_pred = -1;  // predicted
    Scalar optimal_estim_mul_err = -1; // product of optimal estimate and its error, should be zero (optimal estimate and its error are uncorrelated E[x_hat*x_err']=0)
    size_t estimated_sal_pnts; // number of tracking salient points (which are stored in estimated variables array)
    size_t new_sal_pnts; // number of new salient points allocated in current frame
    size_t common_sal_pnts; // number of same salient points in the previous and current frame
    size_t deleted_sal_pnts; // number of deleted salient points in current frame

    std::optional<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> estim_err; // =x_estimated-x_ground_truth, available only in virtual mode
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> estim_err_std;

    Eigen::Matrix<Scalar, kCamStateComps, 1> cam_state;
    std::optional<Eigen::Matrix<Scalar, kCamStateComps, 1>> cam_state_gt; // available only in virtual mode

    std::optional<Eigen::Matrix<Scalar, 3, 3>> sal_pnts_uncert_median; // median of uncertainty of all salient points; null if there are 0 salient points

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> meas_residual;  // =obs-project(estimate)=z-h(x_hat)
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> meas_residual_std;
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
    DavisonMonoSlamTrackerInternalsSlice cur_stats_;
    DavisonMonoSlamTrackerInternalsHist hist_;
    Clock::time_point frame_start_time_point_;
    Clock::time_point frame_finish_time_point_;
public:
    DavisonMonoSlamInternalsLogger();
    virtual ~DavisonMonoSlamInternalsLogger() = default;

    virtual void StartNewFrameStats();
    virtual void RecordFrameFinishTime();
    virtual void PushCurFrameStats();
    virtual void NotifyNewComDelSalPnts(size_t new_count, size_t common_count, size_t deleted_count);
    virtual void NotifyEstimatedSalPnts(size_t estimated_sal_pnts_count);
    DavisonMonoSlamTrackerInternalsSlice& CurStats() { return cur_stats_; }

    DavisonMonoSlamTrackerInternalsHist& BuildStats();
};

/// Implementation of MonoSlam by Andrew Davison https://www.doc.ic.ac.uk/~ajd/Scene/index.html
/// The algorithm uses Kalman filter to estimate camera's location and map features (salient points).
/// source: book "Structure from Motion using the Extended Kalman Filter" Civera 2011 (further SfM_EKF_Civera)
class DavisonMonoSlam
{
public:
    static constexpr auto kCamStateComps = intern::kCamStateComps;
    static constexpr size_t kSalientPointComps = intern::kSalientPointComps;
    static constexpr SalPntComps kSalPntRepres = intern::kSalPntRepres;
    static constexpr Scalar kFiniteDiffEpsDebug = (Scalar)1e-5; // used for debugging derivatives

    using EigenDynMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using EigenDynVec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
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

    using SalPntId = suriko::SalPntId;
private:
    static DebugPathEnum s_debug_path_;
    EigenDynVec estim_vars_; // x[13+N*6], camera position plus all salient points
    EigenDynMat estim_vars_covar_; // P[13+N*6, 13+N*6], state's covariance matrix

    std::shared_mutex predicted_estim_vars_mutex_;  // NOTE: this field prevents this tracker from copying
    EigenDynVec predicted_estim_vars_; // x[13+N*6]
    EigenDynMat predicted_estim_vars_covar_; // P[13+N*6, 13+N*6]

    std::vector<std::unique_ptr<TrackedSalientPoint>> sal_pnts_; // the set of descriptors of salient points (including deleted salient points)
    size_t estim_sal_pnts_count_ = 0;  // number of salient points in error covariance matrix; this doesn't include deleted salient points

public:
    bool in_multi_threaded_mode_ = false;  // true to expect the clients to read predicted vars from different thread; locks are used to protect from conflicting access
    Scalar seconds_per_frame_ = 1; // in seconds, elapsed time between two consecutive frames

    // drastically affects performance: it increases uncertainty regions of salient points, hence the search regions, used for salient points correspondence, are increased
    // used to init bot-right 3x3 of Qk[6,6], uncertainty in camera dynamic model motion
    Scalar process_noise_linear_velocity_std_ = 1;  // in mm, this much camera can move per one time step

    // this much each component of 3x1 camera orientation angle-vector can change per one time step
    // measured in units of angle-vector rotation representation
    // used to init top-left 3x3 of Qk[6,6], uncertainty in camera dynamic model motion
    Scalar process_noise_angular_velocity_std_ = 0.01f;  // in radians, influence how much camera can rotate

    Eigen::Matrix<Scalar, kProcessNoiseComps, kProcessNoiseComps> process_noise_covar_; // Qk[6,6] process noise covariance matrix

    Scalar measurm_noise_std_pix_ = 1;

    // default camera's uncertainty
    Scalar cam_pos_x_std_m_ = 0; // in meters
    Scalar cam_pos_y_std_m_ = 0; // in meters
    Scalar cam_pos_z_std_m_ = 0; // in meters
    Scalar cam_orient_q_comp_std_ = 0;
    Scalar cam_vel_std_ = 0; // in meters
    Scalar cam_ang_vel_std_ = 0; // in meters

    // default salient point's uncertainty
    Scalar sal_pnt_init_inv_dist_ = 1; // rho0, the inverse depth of a salient point in the first camera in which the point is seen
    Scalar sal_pnt_init_inv_dist_std_ = 1; // std(rho0)
    Scalar sal_pnt_pos_x_std_if_gt_ = 0;
    Scalar sal_pnt_pos_y_std_if_gt_ = 0;
    Scalar sal_pnt_pos_z_std_if_gt_ = 0;
    Scalar sal_pnt_first_cam_pos_std_if_gt_ = 0;
    Scalar sal_pnt_azimuth_std_if_gt_ = 0;
    Scalar sal_pnt_elevation_std_if_gt_ = 0;
    Scalar sal_pnt_inv_dist_std_if_gt_ = 0;  // used for ground truth setup

    bool force_xyz_sal_pnt_pos_diagonal_uncert_ = false;
    std::optional<size_t> sal_pnt_max_undetected_frames_count_;  // salient points greater than this value are removed from tracker
    std::optional<Scalar> sal_pnt_negative_inv_rho_substitute_;  // >=0 value, this replaces negative inv rho of a salient point (SP), preventing SP from jumping behind the camera

    // width and height of an image template of a salient point
    // Davison used templates of 15x15 (see "Simultaneous localization and map-building using active vision" para 3.1, Davison, Murray, 2002)
    suriko::Sizei sal_pnt_templ_size_ = { 15, 15 };

    // this allow to register a new salient point only if the distance (between the centers of templates)
    // to the closest salient point is greater than this value;
    // this prevents overlapping of templates of tracked salient points
    std::optional<Scalar> closest_sal_pnt_templ_min_dist_pix_;

    // The confidence interval determining the size of ellipse, extracted from 2D covariance matrix.
    // The value 95% corresponds to chi^2=5.99146 in ellipse equation (x/a)^2+(y/b)^2=chi^2 where a,b are semi major,minor axes of ellipse.
    // [SfM_EKF_Civera para 3.7.2 on page 50]
    Scalar covar2D_to_ellipse_confidence_ = 0.95f;  // {confidence interval, chi-square}={95%,5.99146}, {99%,9.21034}

    std::optional<int> debug_max_sal_pnt_coun_;

    // camera
    CameraIntrinsicParams cam_intrinsics_{};
    bool cam_enable_distortion_ = true;
    MikhailRadialDistortionParams cam_distort_params_{};
public:
    std::function<SE3Transform(size_t frame_ind)> gt_cami_from_world_fun_;  // used to get the first camera cam0 in the world coordinates
    std::function<SE3Transform(size_t frame_ind)> gt_cami_from_tracker_fun_;  // gets ground truth camera position in coordinates of tracker
    std::function<std::optional<SE3Transform>(SE3Transform tracker_from_world, size_t frame_ind)> gt_cami_from_tracker_new_;  // gets ground truth camera frame in coordinates of a given tracker
    std::function<Dir3DAndDistance(SE3Transform tracker_from_world, SE3Transform camera_from_tracker, SalPntId sal_pnt_id)> gt_sal_pnt_in_camera_fun_;  // gets ground truth 3D position of salient point in coordinates of tracker

    bool sal_pnt_perfect_init_inv_dist_ = false; // true to correctly initialize points depth in virtual environments
    int set_estim_state_covar_to_gt_impl_ = 2;  // 1=sets diagonal covariance in estimation space, 2=set correlations as if 'AddNewSalientPoint' is called on each salient point

    /// There are 3 implementations of incorporating m observed corners (corner=pixel, 2x1 mat).
    /// 1. Stack all corners in one [2m,1] vector. Require inverting one [2m,2m] innovation matrix.
    /// 2. Process each corner individually. Require inverting m innovation matrices of size [2x2].
    /// 3. Process [x,y] component of each corner individually. Require inverting 2m scalars.
    /// 4. 1-Point RANSAC
    int mono_slam_update_impl_ = 1;

    /// Threshold to collect low-innovation salient points in 1-point RANSAC algorithm.
    std::optional<Scalar> one_point_ransac_corner_max_divergence_pix_;
    std::optional<Scalar> one_point_ransac_high_innov_chi_square_thresh_pix2_;

    bool fix_estim_vars_covar_symmetry_ = false;
private:
    std::shared_ptr<CornersMatcherBase> corners_matcher_;
    std::shared_ptr<DavisonMonoSlamInternalsLogger> stats_logger_;
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
        EigenDynMat K_H_minus_I; // K*H, [13+N*6, 13+N*6]
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
        Eigen::Matrix<Scalar, Eigen::Dynamic, kQuat4> P_down; // [dyn,4]
    } quat_normalization_cache_;
public:
    DavisonMonoSlam();
    DavisonMonoSlam(const DavisonMonoSlam& src);
    void CopyFrom(const DavisonMonoSlam& src);
    DavisonMonoSlam& operator=(const DavisonMonoSlam& src);

    void ResetCamera();
    void SetCameraVelocity(std::optional<suriko::Point3> cam_vel_tracker, std::optional<suriko::Point3> cam_ang_vel_c);
    void SetCameraStateCovarHelper();
private:
    void SetCameraState(EigenDynVec* src_estim_vars);
    void SetCameraStateCovar(EigenDynMat* src_estim_vars_covar);
public:
    void SetProcessNoiseStd(std::optional<Scalar> process_noise_linear_velocity_std, std::optional <Scalar> process_noise_angular_velocity_std);

    void ProcessFrame(size_t frame_ind, const Picture& image);

    suriko::Point2f ProjectCameraPoint(const suriko::Point3& pnt_camera) const;

    size_t EstimatedVarsCount() const;

    CameraStateVars GetCameraEstimatedVars();
    CameraStateVars GetCameraEstimatedVars() const;
    CameraStateVars GetCameraPredictedVars();
    CameraStateVars GetCameraPredictedVars() const;

    void GetCameraEstimatedPosAndOrientationWithUncertainty(Eigen::Matrix<Scalar, kEucl3,1>* pos_mean, 
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert,
        Eigen::Matrix<Scalar, kQuat4, 1>* orient_quat) const;

    void GetCameraEstimatedVarsUncertainty(Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* cam_covar) const;

    std::optional<suriko::Point2f> GetDetectedSalientTemplCenter(SalPntId sal_pnt_id) const;

    bool GetSalientPointEstimated3DPosWithUncertaintyNew(SalPntId sal_pnt_id,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const;

    bool GetSalientPointPredicted3DPosWithUncertaintyNew(SalPntId sal_pnt_id,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const;

    auto GetSalientPointProjected2DPosWithUncertainty(FilterStageType filter_stage, SalPntId sal_pnt_id) const
        ->std::tuple<bool, MeanAndCov2D>;

    std::tuple<bool, RotatedEllipse2D> GetSalientPointProjectedUncertEllipse(FilterStageType filter_stage, SalPntId sal_pnt_id) const;
    std::tuple<bool, RotatedEllipse2D> GetPredictedSalientPointProjectedUncertEllipse(SalPntId sal_pnt_id) const;

    std::optional<Scalar> CurrentFrameReprojError(FilterStageType filter_stage = FilterStageType::Predicted) const;

    size_t SalientPointsCount() const;

    const std::set<SalPntId>& GetSalientPoints() const;

    TrackedSalientPoint& GetSalientPoint(SalPntId id);
    const TrackedSalientPoint& GetSalientPoint(SalPntId id) const;
    
    SalPntId GetSalientPointIdByOrderInEstimCovMat(size_t sal_pnt_ind) const;

    void CheckSalientPointsConsistency() const;

    suriko::Point2i TemplateTopLeftInt(const suriko::Point2f& center) const;

    // New salient points should be farther away from other salient points in the picture by this distance.
    Scalar ClosestSalientPointTemplateMinDistance() const;

    /// Calculates the 3D rectangle, corresponding to a salient point's template.
    /// The information, used in the calculation is:
    /// 1. The XY boundary of a template in the picture; 
    /// 2. The estimated depth of a salient point.
    /// This returns null if the salient point is in the infinity and finite coordinates of a template can't be calculated.
    std::optional<SalPntRectFacet> ProtrudeSalientTemplateIntoWorld(SalPntId sal_pnt_id) const;

    void SetCornersMatcher(std::shared_ptr<CornersMatcherBase> corners_matcher);
    CornersMatcherBase& CornersMatcher();

    void SetStatsLogger(std::shared_ptr<DavisonMonoSlamInternalsLogger> stats_logger);
    DavisonMonoSlamInternalsLogger* StatsLogger() const;

    static void SetDebugPath(DebugPathEnum debug_path);

    void SetEstimStateAndCovarToGroundTruth(size_t frame_ind);

    void DumpTrackerState(std::ostringstream& os) const;
private:
    struct SalPntProjectionIntermidVars
    {
        Eigen::Matrix<Scalar, kEucl3, 1> hc; // euclidean position of salient point in camera coordinates
#if defined(SPHER_SAL_PNT_REPRES)
        Eigen::Matrix<Scalar, kEucl3, 1> first_cam_sal_pnt_unity_dir; // unity direction from first camera to the salient point in world coordinates
#endif
    };

    /// Salient point, defined in spherical coordinates with origin in the specified 'first' camera.
    /// Axes are aligned with the tracker (world) orientation frame (shown using the '_w' suffix).
    /// The distance is in the inverse format.
    struct SphericalSalientPoint
    {
        Eigen::Matrix<Scalar, kEucl3, 1> first_cam_pos_w; // the position of the camera (in world frame) the salient point was first seen

        // polar coordinates of the salient point in the camera where the feature was seen for the first time
        Scalar azimuth_theta_w = kNan; // theta=azimuth, rotates clockwise around worldOY, zero corresponds to worldOZ direction
        Scalar elevation_phi_w = kNan; // elevation=latin_phi, rotates clockwise around worldOX, zero corresponds to worldOZ direction
        Scalar inverse_dist_rho = kNan; // inverse distance (=rho) from the first camera, where the salient point was seen the first time, to this salient point
    };

    // Some variables which are created when a salient point is projected into the camera.
    // These values are used to calculate derivatives
    struct SphericalSalientPointIntermProjVars
    {
        //FramePosOrient cam;  // the camera to which a salient point is projected

        //suriko::Point2f corner_pix;  // in the first camera

        Eigen::Matrix<Scalar, kEucl3, 1> hc;  // A.58
        Eigen::Matrix<Scalar, kEucl3, 1> hw;  // A.59
        Eigen::Matrix<Scalar, kEucl3, kEucl3> Rwfc;  // 
    };

    struct SphericalSalientPointWithBuildInfo
    {
        SphericalSalientPoint spher_sal_pnt;
        
        size_t first_cam_frame_ind;

        SphericalSalientPointIntermProjVars proj_interm_vars;
    };

    /// The union of all  possible representations of a salient point
    struct MorphableSalientPoint
    {
#if defined(XYZ_SAL_PNT_REPRES)
        Eigen::Matrix<Scalar, kEucl3, 1> pos_w; // the salient point's position
#endif
#if defined(SPHER_SAL_PNT_REPRES)
        Eigen::Matrix<Scalar, kEucl3, 1> first_cam_pos_w; // the position of the camera (in world frame) the salient point was first seen
        
        // polar coordinates of the salient point in the camera where the feature was seen for the first time
        Scalar azimuth_theta_w = kNan; // theta=azimuth, rotates clockwise around worldOY, zero corresponds to worldOZ direction
        Scalar elevation_phi_w = kNan; // elevation=latin_phi, rotates clockwise around worldOX, zero corresponds to worldOZ direction
        Scalar inverse_dist_rho = kNan; // inverse distance (=rho) from the first camera, where the salient point was seen the first time, to this salient point
#endif
    };

    static void CopyFrom(SphericalSalientPoint* dst, const MorphableSalientPoint& s)
    {
        dst->first_cam_pos_w = s.first_cam_pos_w;
        dst->azimuth_theta_w = s.azimuth_theta_w;
        dst->elevation_phi_w = s.elevation_phi_w;
        dst->inverse_dist_rho = s.inverse_dist_rho;
    }

    /// Spherical salient point can't be transformed into XYZ representation when the point is in infinity.
    static bool ConvertXyzFromSphericalSalientPoint(const SphericalSalientPoint& sal_pnt_vars, Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean);

    static bool ConvertSphericalFromXyzSalientPoint(const Eigen::Matrix <Scalar, kXyzSalientPointComps, 1>& sal_pnt_pos_w,
        const SE3Transform& first_cam_tfc,
        SphericalSalientPoint* spher_sal_pnt);

    static void ConvertMorphableFromSphericalSalientPoints(const std::vector<SphericalSalientPointWithBuildInfo>& sal_pnt_build_infos,
        SalPntComps sal_pnt_repres,
        std::vector<MorphableSalientPoint>* sal_pnts);

    void CheckCameraAndSalientPointsCovs(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar) const;

    auto GetFilterStage(FilterStageType filter_stage) -> std::tuple<EigenDynVec*, EigenDynMat*>;
    auto GetFilterStage(FilterStageType filter_stage) const -> std::tuple<const EigenDynVec*, const EigenDynMat*>;

    CameraStateVars GetCameraStateVars(FilterStageType filter_stage);
    CameraStateVars GetCameraStateVars(FilterStageType filter_stage) const;

    void GetCameraPosAndOrientationWithUncertainty(FilterStageType filter_stage,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert,
        Eigen::Matrix<Scalar, kQuat4, 1>* orient_quat) const;

    bool GetSalientPoint3DPosWithUncertaintyHelper(FilterStageType filter_stage, SalPntId sal_pnt_id,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const;

    void FillRk(size_t obs_sal_pnt_count, EigenDynMat* Rk) const;
    void FillRk2x2(Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* Rk) const;

    void PredictCameraMotionByKinematicModel(gsl::span<const Scalar> cam_state, gsl::span<Scalar> new_cam_state,
        const Eigen::Matrix<Scalar, kProcessNoiseComps, 1>* noise_state = nullptr) const;
    void PredictEstimVars(
        const EigenDynVec& src_estim_vars, const EigenDynMat& src_estim_vars_covar,
        EigenDynVec* predicted_estim_vars, EigenDynMat* predicted_estim_vars_covar) const;

    /// Removes salient points' state in estimation matrices. Salient point's descriptors are marked deleted.
    void RemoveSalientPointsState(gsl::span<size_t> sal_pnt_inds_to_delete_desc);
    void RemoveLongTermUnobservedSalientPoints(std::vector<SalPntId>* deleted_sal_pnt_ids);
    void RemoveSalientPointsWithNonextractableUncertEllipsoid(EigenDynVec* src_estim_vars,
        EigenDynMat* src_estim_vars_covar);
    void RemoveMarkedDeletedSalientPointsDescriptors();

    void ProcessFrame_StackedObservationsPerUpdate(size_t frame_ind, const std::vector<SalPntId>& latest_frame_sal_pnt_ids);
    void ProcessFrame_StackedObservationsPerUpdateCore(size_t frame_ind, const std::vector<SalPntId>& latest_frame_sal_pnt_ids, EigenDynVec* src_estim_vars, EigenDynMat* src_estim_vars_covar);
    void ProcessFrame_OneObservationPerUpdate(size_t frame_ind, const std::vector<SalPntId>& latest_frame_sal_pnt_ids);

    void OnePointRansac_GetConsensusMatches(const std::vector<std::pair<SalPntId, suriko::Point2f>>& matched_sal_pnt_to_corner,
        const EigenDynVec& src_estim_vars, const EigenDynMat& Pprev, Scalar corner_max_divergence_pix,
        std::vector<std::pair<SalPntId, suriko::Point2f>>* low_innov_inliers);
    std::tuple<size_t, size_t> ProcessFrame_OnePointRansacUpdateCore(size_t frame_ind, const std::vector<std::pair<SalPntId, suriko::Point2f>>& matched_sal_pnt_to_corner);
    void ProcessFrame_OnePointRansacUpdate(size_t frame_ind, const std::vector<std::pair<SalPntId, suriko::Point2f>>& matched_sal_pnt_to_corner);

    void ProcessFrame_OneComponentOfOneObservationPerUpdate(size_t frame_ind, const std::vector<SalPntId>& latest_frame_sal_pnt_ids);
    void ComputeEstimSalientPointSearchRects(const std::vector<SalPntId>& latest_frame_sal_pnt_ids, const EigenDynMat& innov_var);
    void NormalizeCameraOrientationQuaternionAndCovariances(EigenDynVec* src_estim_vars, EigenDynMat* src_estim_vars_covar);
    void EnsureSalientPointPositiveInvDepth(EigenDynVec* src_estim_vars);
    void EnsureNonnegativeStateVariance(EigenDynMat* src_estim_vars_covar);
    void OnEstimVarsChanged(size_t frame_ind);
    void FinishFrameStats(size_t frame_ind);
    size_t RecruitNewSalientPoints(size_t frame_ind, const Picture& image, const std::vector<std::pair<SalPntId, CornersMatcherBlobId>>& matched_sal_pnts);
    void PredictStateAndCovariance();

    // Updates the centers of detected template.
    void ProcessFrameOnExit_UpdateSalientPoint(size_t frame_ind);

    void SetNonObservedSalientPointCorner(const EigenDynVec& src_estim_vars);

    void AllocateAndInitStateForNewSalientPoint(size_t new_sal_pnt_var_ind,
        const CameraStateVars& cam_state, suriko::Point2f corner_pix, std::optional<Scalar> pnt_inv_dist_gt);
    
    SphericalSalientPoint GetNewSphericalSalientPointState(
        const CameraStateVars& first_cam_state,
        suriko::Point2f first_cam_corner_pix,
        std::optional<Scalar> first_cam_sal_pnt_inv_dist_gt,
        SphericalSalientPointIntermProjVars* interm_proj_vars) const;

    void GetNewSphericalSalientPointCovar(
        const CameraStateVars& first_cam_state,
        suriko::Point2f first_cam_corner_pix,
        const SphericalSalientPointIntermProjVars& proj_side_effect_vars,
        size_t take_estim_vars_count,
        Eigen::Matrix<Scalar, kSphericalSalientPointComps, kSphericalSalientPointComps>* spher_sal_pnt_autocovar,
        Eigen::Matrix<Scalar, kSphericalSalientPointComps, Eigen::Dynamic>* spher_sal_pnt_to_other_covar) const;

    void GetDefaultXyzSalientPointCovarOrConvertFromSpherical(
        const SphericalSalientPoint& spher_sal_pnt_vars,
        size_t take_estim_vars_count,
        const Eigen::Matrix<Scalar, kSphericalSalientPointComps, kSphericalSalientPointComps>& spher_sal_pnt_autocovar,
        const Eigen::Matrix<Scalar, kSphericalSalientPointComps, Eigen::Dynamic>& spher_sal_pnt_to_other_covar,
        Eigen::Matrix<Scalar, kXyzSalientPointComps, kXyzSalientPointComps>* xyz_sal_pnt_autocovar,
        Eigen::Matrix<Scalar, kXyzSalientPointComps, Eigen::Dynamic>* xyz_sal_pnt_to_other_covar) const;

    SalPntId AddSalientPoint(size_t frame_ind, const CameraStateVars& cam_state, suriko::Point2f corner, 
        Picture templ_img, TemplMatchStats templ_stats,
        std::optional<Scalar> pnt_inv_dist_gt);

    gsl::span<Scalar> EstimVarsCamPosW();
    Eigen::Matrix<Scalar, kQuat4, 1> EstimVarsCamQuat() const;
    Eigen::Matrix<Scalar, kAngVelocComps, 1> EstimVarsCamAngularVelocity() const;
    size_t SalientPointOffset(size_t sal_pnt_ind) const;
    //inline SalPntInternal& GetSalPnt(SalPntId id);
    //inline const SalPntInternal& GetSalPnt(SalPntId id) const;

    void LoadCameraStateVarsFromArray(gsl::span<const Scalar> src, CameraStateVars* result) const;

    void LoadSalientPointDataFromArray(gsl::span<const Scalar> src, MorphableSalientPoint* result) const;
    MorphableSalientPoint LoadSalientPointDataFromSrcEstimVars(const EigenDynVec& src_estim_vars, size_t estim_vars_ind) const;
    MorphableSalientPoint LoadSalientPointDataFromSrcEstimVars(const EigenDynVec& src_estim_vars, const TrackedSalientPoint& sal_pnt) const;

    void SaveSalientPointDataToArray(const MorphableSalientPoint& sal_pnt_vars, gsl::span<Scalar> dst) const;
    void SaveSalientPointDataToArray(const SphericalSalientPoint& sal_pnt_vars, gsl::span<Scalar> dst) const;
    void SaveCameraStateToArray(const CameraStateVars& cam_state, gsl::span<Scalar> dst) const;
    void SaveEstimVars(const CameraStateVars& cam_state,
        const std::vector<MorphableSalientPoint>& sal_pnts,
        EigenDynVec* dst_estim_vars);

    void DerivSalPnt_xyz_by_spher(const SphericalSalientPoint& sal_pnt_vars,
        Eigen::Matrix<Scalar, kXyzSalientPointComps, kSphericalSalientPointComps>* sal_pnt_3_by_6) const;

#if defined(SPHER_SAL_PNT_REPRES)
    void PropagateSalPntPosUncertainty(const SphericalSalientPoint& sal_pnt,
        const Eigen::Matrix<Scalar, kSphericalSalientPointComps, kSphericalSalientPointComps>& sal_pnt_covar,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* sal_pnt_pos_uncert) const;
#endif

    bool GetSalientPointPositionUncertainty(
        const EigenDynMat& src_estim_vars_covar,
        const TrackedSalientPoint& sal_pnt,
        const MorphableSalientPoint& sal_pnt_vars,
        bool can_throw,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* sal_pnt_pos_covar) const;

    /// NOTE: The resultant 2D uncertainty does depend on the uncertainty of the camera frame in which the salient point is projected.
    auto GetSalientPointProjected2DPosWithUncertainty(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar,
        const TrackedSalientPoint& sal_pnt) const->std::tuple<bool,MeanAndCov2D>;

    /// NOTE: The resultant uncertainty doesn't respect uncertainty of the current camera frame.
    bool GetSalientPoint3DPosWithUncertainty(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar,
        const TrackedSalientPoint& sal_pnt,
        bool can_throw,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const;

    /// Ensures that given salient point can be correctly handled (rendering, position prediction etc).
    bool CheckSalientPoint(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar, 
        const TrackedSalientPoint& sal_pnt,
        bool can_throw) const;

    std::optional<SalPntRectFacet> ProtrudeSalientPointTemplIntoWorld(const EigenDynVec& src_estim_vars, const TrackedSalientPoint& sal_pnt) const;

    void BackprojectPixelIntoCameraPlane(const Eigen::Matrix<Scalar, kPixPosComps, 1>& hu, Eigen::Matrix<Scalar, kEucl3, 1>* pos_camera) const;

    //

    void Deriv_cam_state_by_cam_state(Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* result) const;

    void FiniteDiff_cam_state_by_cam_state(gsl::span<const Scalar> cam_state, Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* result) const;

    void Deriv_cam_state_by_process_noise(Eigen::Matrix<Scalar, kCamStateComps, kProcessNoiseComps>* result) const;

    void FiniteDiff_cam_state_by_process_noise(Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kCamStateComps, kProcessNoiseComps>* result) const;

    void Deriv_hd_by_cam_state_and_sal_pnt(
        const EigenDynVec& derive_at_pnt,
        const CameraStateVars& cam_state, const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const TrackedSalientPoint& sal_pnt,
        const MorphableSalientPoint& sal_pnt_vars,
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_cam_state,
        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt,
        Eigen::Matrix<Scalar, kPixPosComps, 1>* hd = nullptr) const;

    void Deriv_H_by_estim_vars(const CameraStateVars& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const EigenDynVec& derive_at_pnt,
        const std::vector<SalPntId>& latest_frame_sal_pnt_ids,
        EigenDynMat* H_by_estim_vars) const;

    // Derivative of distorted observed corner (in pixels) by undistorted observed corner (in pixels).
    void Deriv_hu_by_hd(suriko::Point2f corner_pix, Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* hu_by_hd) const;
    void Deriv_hd_by_hu(suriko::Point2f corner_pix, Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* hd_by_hu) const;

    // Derivative of distorted observed corner (in pixels) by camera's state variables
    void Deriv_hu_by_hc(const SalPntProjectionIntermidVars& proj_hist, Eigen::Matrix<Scalar, kPixPosComps, kEucl3 >* dhu_by_dhc) const;

    void Deriv_hc_by_hu(Eigen::Matrix<Scalar, kEucl3, kPixPosComps>* hc_by_hu) const;

    void Deriv_R_by_q(const Eigen::Matrix<Scalar, kQuat4, 1>& q,
        Eigen::Matrix<Scalar, 3, 3>* dR_by_dq0,
        Eigen::Matrix<Scalar, 3, 3>* dR_by_dq1,
        Eigen::Matrix<Scalar, 3, 3>* dR_by_dq2,
        Eigen::Matrix<Scalar, 3, 3>* dR_by_dq3) const;

    // Derivative of distorted observed corner (in pixels) by camera's state variables (13 vars).
    void Deriv_hd_by_camera_state(const MorphableSalientPoint& sal_pnt,
        const CameraStateVars& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const SalPntProjectionIntermidVars& proj_hist,
        const Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>& hd_by_dhu,
        const Eigen::Matrix<Scalar, kPixPosComps, kEucl3>& dhu_by_dhc,
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_cam) const;

    // Derivative of distorted observed corner (in pixels) by salient point's variables (6 vars).
    void Deriv_hd_by_sal_pnt(const MorphableSalientPoint& sal_pnt,
        const CameraStateVars& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>& hd_by_dhu,
        const Eigen::Matrix<Scalar, kPixPosComps, kEucl3>& hu_by_dhc,
        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt) const;

    void Deriv_azim_theta_elev_phi_by_hw(
        const Eigen::Matrix<Scalar, kEucl3, 1>& hw,
        Eigen::Matrix<Scalar, 1, kEucl3>* azim_theta_by_hw,
        Eigen::Matrix<Scalar, 1, kEucl3>* elev_phi_by_hw) const;

    void FiniteDiff_hd_by_camera_state(const EigenDynVec& derive_at_pnt,
        const MorphableSalientPoint& sal_pnt,
        Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_xc) const;
    
    void FiniteDiff_hd_by_sal_pnt_state(const CameraStateVars& cam_state, 
        const TrackedSalientPoint& sal_pnt,
        const EigenDynVec& derive_at_pnt,
        Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_y) const;


    // derivative of qk+1 (next step camera orientation) by wk (camera orientation)
    void Deriv_q3_by_w(Scalar deltaT, Eigen::Matrix<Scalar, kQuat4, kEucl3>* result) const;
    void Deriv_q1_by_w(Scalar deltaT, Eigen::Matrix<Scalar, kQuat4, kEucl3>* result) const;
    
    static void CameraCoordinatesEuclidUnityDirFromPolarAngles(Scalar azimuth_theta, Scalar elevation_phi, Scalar* hx, Scalar* hy, Scalar* hz);
    
    // projection

    Eigen::Matrix<Scalar, kPixPosComps, 1> ProjectCameraSalientPoint(
        const Eigen::Matrix<Scalar, kEucl3, 1>& pnt_camera,
        SalPntProjectionIntermidVars *proj_hist) const;

    std::optional<Eigen::Matrix<Scalar, kEucl3, 1>> InternalSalientPointToCamera(
        const MorphableSalientPoint& sal_pnt_vars,
        const CameraStateVars& cam_state,
        bool scaled_by_inv_dist,
        SalPntProjectionIntermidVars *proj_hist) const;

    Eigen::Matrix<Scalar, kPixPosComps, 1> ProjectInternalSalientPoint(const CameraStateVars& cam_state, const MorphableSalientPoint& sal_pnt_vars, SalPntProjectionIntermidVars *proj_hist) const;

    void GetGroundTruthEstimVars(size_t frame_ind,
        CameraStateVars* cam_state,
        std::vector<SphericalSalientPointWithBuildInfo>* sal_pnt_build_infos) const;

    void SetCamStateCovarToGroundTruth(EigenDynMat* src_estim_vars_covar) const;

    Eigen::Matrix<Scalar, kEucl3, kEucl3> GetDefaultXyzSalientPointCovar() const;

    // Sets variances for estimated state. Works directly in estimated space. In virtual mode only.
    void SetEstimStateCovarInEstimSpace(size_t frame_ind);

    // The uncertainty of a salient point is formed in standard way for 'inverse depth' salient point representation.
    void SetEstimStateCovarLikeInAddNewSalPnt(size_t frame_ind,
        const std::vector<SphericalSalientPointWithBuildInfo>& sal_pnt_build_infos);
    
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
