#pragma once
#include <set>
#include <vector>
#include <memory>
#include <shared_mutex>
#include <functional>
#include <gsl/span>
#include "suriko/obs-geom.h"

namespace suriko {
namespace
{
    constexpr size_t kEucl3 = 3; // x: 3 for position
    constexpr size_t kQuat4 = 4; // q: 4 for quaternion orientation
    constexpr size_t kVelocComps = kEucl3; // v: 3 for velocity
    constexpr size_t kAngVelocComps = kEucl3; // w: 3 for angular velocity
    constexpr size_t kAccelComps = kEucl3; // a: 3 for acceleration
    constexpr size_t kAngAccelComps = kEucl3; // alpha: 3 for angular acceleration

    void DependsOnOverallPackOrder() {}
    void DependsOnCameraPosPackOrder() {}
    void DependsOnSalientPointPackOrder() {}
    void DependsOnInputNoisePackOrder() {}
}
class CornersMatcherBase
{
public:
    virtual void DetectAndMatchCorners(size_t frame_ind, CornerTrackRepository* track_rep) = 0;

    virtual ~CornersMatcherBase() = default;
};

/// ax=f/dx and ay=f/dy 
/// (alpha_x = focal_length_x_meters / pixel_width_meters)
struct CameraIntrinsicParams
{
    Scalar FocalLengthMm; // =f in millimiters
    std::array<Scalar,2> PixelSizeMm; // [dx,dy] in millimeters
    std::array<Scalar,2> PrincipalPointPixels; // [Cx,Cy] in pixels

    std::array<Scalar, 2> GetFocalLengthPixels() const { return std::array<Scalar, 2> { FocalLengthMm / PixelSizeMm[0], FocalLengthMm / PixelSizeMm[1]}; }
};

/// scale_factor=1+k1*r^2+k2*r^4
struct RadialDistortionParams
{
    Scalar K1;
    Scalar K2;
};

struct CameraPosState
{
    Eigen::Matrix<Scalar, kEucl3, 1> PosW; // in world frame
    Eigen::Matrix<Scalar, kQuat4, 1> OrientationWfc;
    Eigen::Matrix<Scalar, kVelocComps, 1> VelocityW; // in world frame
    Eigen::Matrix<Scalar, kAngVelocComps, 1> AngularVelocityC; // in camera frame
};

/// Implementation of MonoSlam by Andrew Davison https://www.doc.ic.ac.uk/~ajd/Scene/index.html
/// The algorithm uses Kalman filter to estimate camera's location and map features (salient points).
/// source: book "Structure from Motion using the Extended Kalman Filter" Civera 2011 (further SfM_EKF_Civera)
class DavisonMonoSlam
{
    // [x q v w], x: 3 for position, q: 4 for quaternion orientation, v: 3 for velocity, w: 3 for angular velocity
    static constexpr size_t kCamStateComps = kEucl3 + kQuat4 + kVelocComps + kAngVelocComps; // 13

    static constexpr size_t kInputNoiseComps = kVelocComps + kAngVelocComps; // Qk.rows: velocity and angular velocity are updated an each iteration by noise
    static constexpr size_t kSalientPointPolarCompsCount = 3; // [theta elevation rho], theta: 1 for azimuth angle, 1 for elevation angle, rho: 1 for distance
    static constexpr size_t kSalientPointComps = kEucl3 + kSalientPointPolarCompsCount;

    static constexpr size_t kPixPosComps = 2; // rows and columns

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
private:
    static DebugPathEnum s_debug_path_;

    EigenDynVec estim_vars_; // x[13+N*6], camera position plus all salient points
    EigenDynMat estim_vars_covar_; // P[13+N*6, 13+N*6], state's covariance matrix
    
    Eigen::Matrix<Scalar, kSalientPointComps, kSalientPointComps> input_noise_covar_; // Qk[6,6] input noise covariance matrix

    EigenDynVec predicted_estim_vars_; // x[13+N*6]
    EigenDynMat predicted_estim_vars_covar_; // P[13+N*6, 13+N*6]

    EigenDynMat filter_gain_; // P[13+N*6, 2m] m=number of observed points
    EigenDynMat measurm_noise_covar_; // R[2m,2m]
public:
    Scalar between_frames_period_ = 1; // elapsed time between two consecutive frames
    Scalar input_noise_std_ = 1;
    Scalar measurm_noise_std_ = 1;

    /// There are 3 implementations of incorporating m observed corners (corner=pixel, 2x1 mat).
    /// 1. Stack all corners in one [2m,1] vector. Require inverting one [2m,2m] innovation matrix.
    /// 2. Process each corner individually. Require inverting m innovation matrices of size [2x2].
    /// 3. Process [x,y] component of each corner individually. Require inverting 2m scalars.
    int kalman_update_impl_ = 0;

    std::unique_ptr<CornersMatcherBase> corners_matcher_;
    CornerTrackRepository track_rep_; // TODO: tracker shouldn't contain a history of corners per image (use only corners from latest image)
    // camera
    CameraIntrinsicParams cam_intrinsics_{};
    RadialDistortionParams cam_distort_params_{};
public:
    std::function<SE3Transform(size_t, size_t)> gt_cam_orient_f1f2_;
    std::function<SE3Transform(size_t)> gt_cam_orient_world_to_f_;
    std::function <Point3(size_t)> gt_salient_point_by_virtual_point_id_fun_;
    Scalar debug_ellipsoid_cut_thr_ = 0.04; // value 0.05 corresponds to 2sig
    bool fake_localization_ = false; // true to get camera orientation from ground truth
public:
    DavisonMonoSlam() = default;

    void ResetState(const SE3Transform& cam_pos_cfw, const std::vector<SalientPointFragment>& salient_feats,
        Scalar estim_var_init_std);

    void ProcessFrame(size_t frame_ind);

    void PredictEstimVarsHelper();

    suriko::Point2 ProjectCameraPoint(const suriko::Point3& pnt_camera) const;

    void LogReprojError() const;

    void SetCornersMatcher(std::unique_ptr<CornersMatcherBase> corners_matcher);
    CornersMatcherBase& CornersMatcher();

    size_t SalientPointsCount() const;
    size_t EstimatedVarsCount() const;
    size_t FramesCount() const;

    void GetCameraPredictedPosState(CameraPosState* result) const;

    void GetCameraPredictedPosAndOrientationWithUncertainty(Eigen::Matrix<Scalar, kEucl3,1>* pos_mean, 
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert,
        Eigen::Matrix<Scalar, kQuat4, 1>* orient_quat) const;
    void GetSalientPointPredictedPosWithUncertainty(size_t salient_pnt_ind, 
        Eigen::Matrix<Scalar, kEucl3,1>* pos_mean, 
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const;

    static void SetDebugPath(DebugPathEnum debug_path);
private:
    struct SalPntProjectionIntermidVars
    {
        Eigen::Matrix<Scalar, kEucl3, 1> hc; // euclidean position of salient point in camera coordinates
        Eigen::Matrix<Scalar, kEucl3, 1> FirstCamSalPntUnityDir; // unity direction from first camera to the salient point in world coordinates
    };

    struct SalientPointInternal
    {
        Eigen::Matrix<Scalar, kEucl3, 1> FirstCamPosW; // the position of the camera (in world frame) the salient point was first seen
        Scalar AzimuthThetaW; // theta=azimuth, rotates clockwise around worldOY, zero corresponds to worldOZ direction
        Scalar ElevationPhiW; // elevation=latin_phi, rotates clockwise around worldOX, zero corresponds to worldOZ direction
        Scalar InverseDistRho; // inverse distance to point=rho from the first camera the salient point was first seen

        Scalar GetDist() const; // distance to point = 1/rho
    };

    void CheckCameraAndSalientPointsCovs(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar) const;

    void FillRk(size_t matched_corners, EigenDynMat* Rk) const;
    void FillRk2x2(Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* Rk) const;

    void PredictCameraMotionByKinematicModel(gsl::span<const Scalar> cam_state, gsl::span<Scalar> new_cam_state,
        const Eigen::Matrix<Scalar, kInputNoiseComps, 1>* noise_state = nullptr,
        bool normalize_quat = true) const;
    void PredictEstimVars(size_t frame_ind, EigenDynVec* predicted_estim_vars, EigenDynMat* predicted_estim_vars_covar) const;

    void ProcessFrame_StackedObservationsPerUpdate(size_t frame_ind, const std::vector<size_t>& matched_track_ids);
    void ProcessFrame_OneObservationPerUpdate(size_t frame_ind, const std::vector<size_t>& matched_track_ids);
    void ProcessFrame_OneComponentOfOneObservationPerUpdate(size_t frame_ind, const std::vector<size_t>& matched_track_ids);
    void OnEstimVarsChanged(size_t frame_ind);

    Eigen::Matrix<Scalar, kQuat4, 1> EstimVarsCamQuat() const;
    Eigen::Matrix<Scalar, kAngVelocComps, 1> EstimVarsCamAngularVelocity() const;
    size_t SalientPointOffset(size_t sal_pnt_ind) const;

    void LoadCameraPosDataFromArray(gsl::span<const Scalar> src, CameraPosState* result) const;

    void LoadSalientPointDataFromArray(gsl::span<const Scalar> src, SalientPointInternal* result) const;
    void LoadSalientPointPredictedPosWithUncertainty(
        const EigenDynVec& src_estim_vars,
        const EigenDynMat& src_estim_vars_covar,
        size_t salient_pnt_ind,
        Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
        Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const;

    //

    void Deriv_cam_state_by_cam_state(Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* result) const;

    void FiniteDiff_cam_state_by_cam_state(gsl::span<const Scalar> cam_state, Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* result) const;

    void Deriv_cam_state_by_input_noise(Eigen::Matrix<Scalar, kCamStateComps, kInputNoiseComps>* result) const;

    void FiniteDiff_cam_state_by_input_noise(Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kCamStateComps, kInputNoiseComps>* result) const;

    void Deriv_Hrowblock_by_estim_vars(size_t obs_sal_pnt_ind, size_t frame_ind,
        const CameraPosState& cam_state, const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const std::vector<size_t>& matched_track_ids,
        const EigenDynVec& derive_at_pnt,
        Eigen::Matrix<Scalar, kPixPosComps, Eigen::Dynamic>* Hrowblock_by_estim_vars) const;

    void Deriv_hd_by_cam_state_and_sal_pnt(size_t obs_sal_pnt_ind, size_t frame_ind,
        const CameraPosState& cam_state, const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const std::vector<size_t>& matched_track_ids,
        const EigenDynVec& derive_at_pnt,
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_cam_state,
        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt) const;

    void Deriv_H_by_estim_vars(size_t frame_ind,
        const CameraPosState& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const std::vector<size_t>& matched_track_ids, 
        const EigenDynVec& derive_at_pnt,
        EigenDynMat* H_by_estim_vars) const;

    // Derivative of distorted observed corner (in pixels) by undistorted observed corner (in pixels).
    void Deriv_hd_by_hu(suriko::Point2 corner_pix, Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* hd_by_hu) const;

    // Derivative of distorted observed corner (in pixels) by camera's state variables
    void Deriv_hu_by_hc(const SalPntProjectionIntermidVars& proj_hist, Eigen::Matrix<Scalar, kPixPosComps, kEucl3 >* dhu_by_dhc) const;

    // Derivative of distorted observed corner (in pixels) by camera's state variables (13 vars).
    void Deriv_hd_by_camera_state(const SalientPointInternal& sal_pnt,
        const CameraPosState& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const SalPntProjectionIntermidVars& proj_hist,
        const Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>& hd_by_dhu,
        const Eigen::Matrix<Scalar, kPixPosComps, kEucl3>& dhu_by_dhc,
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_xc) const;

    // Derivative of distorted observed corner (in pixels) by salient point's variables (6 vars).
    void Deriv_hd_by_sal_pnt(const SalientPointInternal& sal_pnt,
        const CameraPosState& cam_state,
        const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
        const Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>& hd_by_dhu,
        const Eigen::Matrix<Scalar, kPixPosComps, kEucl3>& hu_by_dhc,
        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt) const;

    void FiniteDiff_hd_by_camera_state(const EigenDynVec& derive_at_pnt,
        const SalientPointInternal& sal_pnt,
        Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_xc) const;
    
    void FiniteDiff_hd_by_sal_pnt_state(const CameraPosState& cam_state, size_t obs_sal_pnt_ind,
        const EigenDynVec& derive_at_pnt,
        Scalar finite_diff_eps,
        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_y) const;


    // derivative of qk+1 (next step camera orientation) by wk (camera orientation)
    void Deriv_q3_by_w(Scalar deltaT, Eigen::Matrix<Scalar, kQuat4, kEucl3>* result) const;
    
    static void CameraCoordinatesPolarFromEuclid(Scalar hx, Scalar hy, Scalar hz, Scalar* azimuth_theta, Scalar* elevation_phi, Scalar* dist);
    static void CameraCoordinatesEuclidFromPolar(Scalar azimuth, Scalar elevation_phi, Scalar dist, Scalar* hx, Scalar* hy, Scalar* hz);
    static void CameraCoordinatesEuclidUnityDirFromPolarAngles(Scalar azimuth_theta, Scalar elevation_phi, Scalar* hx, Scalar* hy, Scalar* hz);
    
    void SalientPointWorldFromInternal(const SalientPointInternal& sal_pnt, 
        const SE3Transform& first_cam_wfc,
        Eigen::Matrix<Scalar, 3, 1>* sal_pnt_in_world) const;
    void SalientPointInternalFromWorld(const Eigen::Matrix<Scalar, 3, 1>& sal_pnt_in_world,
        const SE3Transform& first_cam_wfc,
        SalientPointInternal* sal_pnt) const;

    // projection

    Eigen::Matrix<Scalar, kPixPosComps, 1> ProjectCameraSalientPoint(
        const Eigen::Matrix<Scalar, kEucl3, 1>& pnt_camera,
        SalPntProjectionIntermidVars *proj_hist) const;

    Eigen::Matrix<Scalar, kPixPosComps, 1> ProjectInternalSalientPoint(const CameraPosState& cam_state, const SalientPointInternal& sal_pnt, SalPntProjectionIntermidVars *proj_hist) const;

    Scalar CurrentFrameReprojError() const;
    
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
