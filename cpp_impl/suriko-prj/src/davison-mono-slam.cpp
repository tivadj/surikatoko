#include <random>
#include "suriko/davison-mono-slam.h"
#include <glog/logging.h>
#include "suriko/approx-alg.h"
#include "suriko/quat.h"
#include "suriko/eigen-helpers.hpp"
#include "suriko/rand-stuff.h"

namespace suriko
{
template <typename EigenMat>
auto Span(EigenMat& m) -> gsl::span<Scalar>
{
    return gsl::make_span<Scalar>(m.data(), static_cast<gsl::span<Scalar>::index_type>(m.size()));
}

template <typename EigenMat>
auto Span(const EigenMat& m) -> gsl::span<const Scalar>
{
    return gsl::make_span<const Scalar>(m.data(), static_cast<gsl::span<const Scalar>::index_type>(m.size()));
}

template <typename EigenMat>
auto Span(EigenMat& m, size_t count) -> gsl::span<Scalar>
{
    return gsl::make_span<Scalar>(m.data(), static_cast<gsl::span<Scalar>::index_type>(count));
}

template <typename EigenMat>
auto Span(const EigenMat& m, size_t count) -> gsl::span<const Scalar>
{
    return gsl::make_span<const Scalar>(m.data(), static_cast<gsl::span<Scalar>::index_type>(count));
}

template <typename EigenMat>
auto SpanAu(EigenMat& m, size_t count) -> gsl::span<typename EigenMat::Scalar>
{
    typedef typename EigenMat::Scalar S;
    // fails: m.data()=const double* and it doesn't match gsl::span<double>
    return gsl::make_span<S>(m.data(), static_cast<typename gsl::span<S>::index_type>(count));
}

// static
DavisonMonoSlam::DebugPathEnum DavisonMonoSlam::s_debug_path_ = DebugPathEnum::DebugNone;

DavisonMonoSlam::DavisonMonoSlam()
{
    SetInputNoiseStd(input_noise_std_);

    Scalar estim_var_init_std = 1;
    ResetCamera(estim_var_init_std);
}

void DavisonMonoSlam::ResetCamera(Scalar estim_var_init_std)
{
    // state vector
    size_t n = kCamStateComps;
    estim_vars_.setZero(n, 1);
    gsl::span<Scalar> state_span = gsl::make_span(estim_vars_.data(), n);

    // camera position
    DependsOnCameraPosPackOrder();
    state_span[0] = 0;
    state_span[1] = 0;
    state_span[2] = 0;

    // camera orientation
    state_span[3] = 1;
    state_span[4] = 0;
    state_span[5] = 0;
    state_span[6] = 0;

    // camera velocity; at each iteration is increased by acceleration in the form of the gaussian noise
    state_span[7] = 0;
    state_span[8] = 0;
    state_span[9] = 0;

    // camera angular velocity; at each iteration is increased by acceleration in the form of the gaussian noise
    state_span[10] = 0;
    state_span[11] = 0;
    state_span[12] = 0;

    Scalar estim_var_init_variance = suriko::Sqr(estim_var_init_std);

    // state uncertainty matrix
    estim_vars_covar_.setZero(n, n);
    // camera position
    estim_vars_covar_(0, 0) = estim_var_init_variance;
    estim_vars_covar_(1, 1) = estim_var_init_variance;
    estim_vars_covar_(2, 2) = estim_var_init_variance;
    // camera orientation (quaternion)
    estim_vars_covar_(3, 3) = estim_var_init_variance;
    estim_vars_covar_(4, 4) = estim_var_init_variance;
    estim_vars_covar_(5, 5) = estim_var_init_variance;
    estim_vars_covar_(6, 6) = estim_var_init_variance;
    // camera speed
    estim_vars_covar_(7, 7) = estim_var_init_variance;
    estim_vars_covar_(8, 8) = estim_var_init_variance;
    estim_vars_covar_(9, 9) = estim_var_init_variance;
    // camera angular speed
    estim_vars_covar_(10, 10) = estim_var_init_variance;
    estim_vars_covar_(11, 11) = estim_var_init_variance;
    estim_vars_covar_(12, 12) = estim_var_init_variance;
}

void DavisonMonoSlam::SetCamera(const SE3Transform& cam_pos_cfw, Scalar estim_var_init_std)
{
    auto cam_pos_wfc = SE3Inv(cam_pos_cfw);

    // state vector
    size_t n = kCamStateComps;
    estim_vars_.setZero(n, 1);
    gsl::span<Scalar> state_span = gsl::make_span(estim_vars_.data(), n);

    // camera position
    DependsOnCameraPosPackOrder();
    state_span[0] = cam_pos_wfc.T[0];
    state_span[1] = cam_pos_wfc.T[1];
    state_span[2] = cam_pos_wfc.T[2];

    // camera orientation
    gsl::span<Scalar> cam_pos_wfc_quat = gsl::make_span(estim_vars_.data() + kEucl3, kQuat4);
    bool op = QuatFromRotationMat(cam_pos_wfc.R, cam_pos_wfc_quat);
    SRK_ASSERT(op);

    Scalar qlen = std::sqrt(
        suriko::Sqr(cam_pos_wfc_quat[0]) +
        suriko::Sqr(cam_pos_wfc_quat[1]) +
        suriko::Sqr(cam_pos_wfc_quat[2]) +
        suriko::Sqr(cam_pos_wfc_quat[3]));

    state_span[3] = cam_pos_wfc_quat[0];
    state_span[4] = cam_pos_wfc_quat[1];
    state_span[5] = cam_pos_wfc_quat[2];
    state_span[6] = cam_pos_wfc_quat[3];

    // camera velocity; at each iteration is increased by acceleration in the form of the gaussian noise
    state_span[7] = 0;
    state_span[8] = 0;
    state_span[9] = 0;

    // camera angular velocity; at each iteration is increased by acceleration in the form of the gaussian noise
    state_span[10] = 0;
    state_span[11] = 0;
    state_span[12] = 0;

    Scalar estim_var_init_variance = suriko::Sqr(estim_var_init_std);

    // state uncertainty matrix
    estim_vars_covar_.setZero(n, n);
    // camera position
    estim_vars_covar_(0, 0) = estim_var_init_variance;
    estim_vars_covar_(1, 1) = estim_var_init_variance;
    estim_vars_covar_(2, 2) = estim_var_init_variance;
    // camera orientation (quaternion)
    estim_vars_covar_(3, 3) = estim_var_init_variance;
    estim_vars_covar_(4, 4) = estim_var_init_variance;
    estim_vars_covar_(5, 5) = estim_var_init_variance;
    estim_vars_covar_(6, 6) = estim_var_init_variance;
    // camera speed
    estim_vars_covar_(7, 7) = estim_var_init_variance;
    estim_vars_covar_(8, 8) = estim_var_init_variance;
    estim_vars_covar_(9, 9) = estim_var_init_variance;
    // camera angular speed
    estim_vars_covar_(10, 10) = estim_var_init_variance;
    estim_vars_covar_(11, 11) = estim_var_init_variance;
    estim_vars_covar_(12, 12) = estim_var_init_variance;
}

void DavisonMonoSlam::SetInputNoiseStd(Scalar input_noise_std)
{
    input_noise_std_ = input_noise_std;
    
    Scalar input_noise_std_variance = suriko::Sqr(input_noise_std);

    input_noise_covar_.setZero();
    input_noise_covar_(0, 0) = input_noise_std_variance;
    input_noise_covar_(1, 1) = input_noise_std_variance;
    input_noise_covar_(2, 2) = input_noise_std_variance;
    input_noise_covar_(3, 3) = input_noise_std_variance;
    input_noise_covar_(4, 4) = input_noise_std_variance;
    input_noise_covar_(5, 5) = input_noise_std_variance;
}

void DavisonMonoSlam::CameraCoordinatesPolarFromEuclid(Scalar hx, Scalar hy, Scalar hz, Scalar* azimuth_theta, Scalar* elevation_phi, Scalar* dist)
{
    // azimuth=theta, formula A.60
    *azimuth_theta = std::atan2(hx, hz);
    
    // elevation=phi, formula A.61
    *elevation_phi = std::atan2(-hy, std::sqrt(hx*hx + hz * hz));

    *dist = std::sqrt(hx*hx + hy * hy + hz * hz);
}

void DavisonMonoSlam::CameraCoordinatesEuclidFromPolar(Scalar azimuth_theta, Scalar elevation_phi, Scalar dist, Scalar* hx, Scalar* hy, Scalar* hz)
{
    // polar -> euclidean position of salient point in world frame with center in camera position
    // theta=azimuth
    Scalar cos_th = std::cos(azimuth_theta);
    Scalar sin_th = std::sin(azimuth_theta);

    // phi=elevation
    Scalar cos_ph = std::cos(elevation_phi);
    Scalar sin_ph = std::sin(elevation_phi);

    *hx =  dist * cos_ph * sin_th;
    *hy = -dist * sin_ph;
    *hz =  dist * cos_ph * cos_th;
}

void DavisonMonoSlam::CameraCoordinatesEuclidUnityDirFromPolarAngles(Scalar azimuth_theta, Scalar elevation_phi, Scalar* hx, Scalar* hy, Scalar* hz)
{
    // polar -> euclidean position of salient point in world frame with center in camera position
    // theta=azimuth
    Scalar cos_th = std::cos(azimuth_theta);
    Scalar sin_th = std::sin(azimuth_theta);

    // phi=elevation
    Scalar cos_ph = std::cos(elevation_phi);
    Scalar sin_ph = std::sin(elevation_phi);

    *hx =  cos_ph * sin_th;
    *hy = -sin_ph;
    *hz =  cos_ph * cos_th;
}

void DavisonMonoSlam::SalientPointInternalFromWorld(const Eigen::Matrix<Scalar, 3, 1>& sal_pnt_in_world,
    const SE3Transform& first_cam_wfc,
    EstimVarsSalientPoint* sal_pnt) const
{
    sal_pnt->FirstCamPosW = first_cam_wfc.T;

    // position of the salient point in world coordinates as looking from camera position
    Eigen::Matrix<Scalar, 3, 1> sal_pnt_cam_origin = sal_pnt_in_world - sal_pnt->FirstCamPosW;
    Scalar dist;
    CameraCoordinatesPolarFromEuclid(sal_pnt_cam_origin[0], sal_pnt_cam_origin[1], sal_pnt_cam_origin[2], &sal_pnt->AzimuthThetaW, &sal_pnt->ElevationPhiW, &dist);
    sal_pnt->InverseDistRho = 1 / dist;
}

void DavisonMonoSlam::SalientPointWorldFromInternal(const EstimVarsSalientPoint& sal_pnt,
    const SE3Transform& first_cam_wfc,
    Eigen::Matrix<Scalar, 3, 1>* sal_pnt_in_world) const
{
    Eigen::Matrix<Scalar, kEucl3, 1> m;
    CameraCoordinatesEuclidUnityDirFromPolarAngles(sal_pnt.AzimuthThetaW, sal_pnt.ElevationPhiW, &m[0], &m[1], &m[2]);

    // the camera must be the one where the salient point was seen the first time
    Scalar first_cam_pos_diff = (first_cam_wfc.T - sal_pnt.FirstCamPosW).norm();
    SRK_ASSERT(first_cam_pos_diff < 1);

    Scalar dist = 1 / sal_pnt.InverseDistRho;
    *sal_pnt_in_world = sal_pnt.FirstCamPosW + first_cam_wfc.R * (m*dist);
}

template <typename EigenMat>
void CheckUncertCovMat(const EigenMat& m)
{
    // determinant must be posi tive for expression 1/sqrt(det(Sig)*(2pi)^3) to exist
    auto det = m.determinant();
    SRK_ASSERT(det > 0);
}

void DavisonMonoSlam::CheckCameraAndSalientPointsCovs(
    const EigenDynVec& src_estim_vars,
    const EigenDynMat& src_estim_vars_covar) const
{
    Eigen::Matrix<Scalar, kEucl3, kEucl3> cam_pos_cov = estim_vars_covar_.block<kEucl3, kEucl3>(0, 0);
    auto cam_pos_cov_inv = cam_pos_cov.inverse().eval();
    CheckUncertCovMat(cam_pos_cov);

    size_t sal_pnts_count = SalientPointsCount();
    for (size_t salient_pnt_ind = 0; salient_pnt_ind < sal_pnts_count; ++salient_pnt_ind)
    {
        Eigen::Matrix<Scalar, 3, 1> sal_pnt_pos;
        Eigen::Matrix<Scalar, 3, 3> sal_pnt_pos_uncert;
        LoadSalientPointPredictedPosWithUncertainty(src_estim_vars, src_estim_vars_covar, salient_pnt_ind, &sal_pnt_pos, &sal_pnt_pos_uncert);
        CheckUncertCovMat(sal_pnt_pos_uncert);
    }
}

void DavisonMonoSlam::FillRk2x2(Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* Rk) const
{
    Scalar measurm_noise_variance = suriko::Sqr(static_cast<Scalar>(measurm_noise_std_));
    *Rk << measurm_noise_variance, 0, 0, measurm_noise_variance;
}

void DavisonMonoSlam::FillRk(size_t obs_sal_pnt_count, EigenDynMat* Rk) const
{
    Rk->setZero(obs_sal_pnt_count * kPixPosComps, obs_sal_pnt_count * kPixPosComps);

    Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> noise_one;
    FillRk2x2(&noise_one);
    
    for (size_t i = 0; i < obs_sal_pnt_count; ++i)
    {
        size_t off = i * kPixPosComps;
        Rk->block<kPixPosComps, kPixPosComps>(off, off) = noise_one;
    }
}

void DavisonMonoSlam::PredictCameraMotionByKinematicModel(gsl::span<const Scalar> cam_state, gsl::span<Scalar> new_cam_state,
    const Eigen::Matrix<Scalar, kInputNoiseComps, 1>* noise_state,
    bool normalize_quat) const
{
    Eigen::Map<const Eigen::Matrix<Scalar, kCamStateComps, 1>> cam_state_mat(cam_state.data());

    //Eigen::Map<const Eigen::Matrix<Scalar, kEucl3, 1>> cam_pos(&cam_state[0]);
    //Eigen::Map<const Eigen::Matrix<Scalar, kQuat4, 1>> cam_orient_quat(&cam_state[kEucl3]);
    //Eigen::Map<const Eigen::Matrix<Scalar, kVelocComps, 1>> cam_vel(&cam_state[kEucl3 + kQuat4]);
    //Eigen::Map<const Eigen::Matrix<Scalar, kAngVelocComps, 1>> cam_ang_vel(&cam_state[kEucl3 + kQuat4 + kVelocComps]);
    Eigen::Matrix<Scalar, kEucl3, 1> cam_pos = cam_state_mat.middleRows<kEucl3>(0);
    Eigen::Matrix<Scalar, kQuat4, 1> cam_orient_quat = cam_state_mat.middleRows<kQuat4>(kEucl3);
    Eigen::Matrix<Scalar, kVelocComps, 1> cam_vel = cam_state_mat.middleRows< kVelocComps>(kEucl3 + kQuat4);
    Eigen::Matrix<Scalar, kAngVelocComps, 1> cam_ang_vel = cam_state_mat.middleRows<kAngVelocComps>(kEucl3 + kQuat4 + kVelocComps);

    Eigen::Map<Eigen::Matrix<Scalar, kEucl3, 1>> new_cam_pos(&new_cam_state[0]);
    Eigen::Map<Eigen::Matrix<Scalar, kQuat4, 1>> new_cam_orient_quat(&new_cam_state[kEucl3]);
    Eigen::Map<Eigen::Matrix<Scalar, kVelocComps, 1>> new_cam_vel(&new_cam_state[kEucl3 + kQuat4]);
    Eigen::Map<Eigen::Matrix<Scalar, kAngVelocComps, 1>> new_cam_ang_vel(&new_cam_state[kEucl3 + kQuat4 + kVelocComps]);

    // camera position
    Scalar dT = between_frames_period_;
    new_cam_pos = cam_pos + cam_vel * dT;

    DependsOnInputNoisePackOrder();
    if (noise_state != nullptr)
        new_cam_pos += noise_state->topRows<kAccelComps>() * dT;

    // camera orientation
    Eigen::Matrix<Scalar, kVelocComps, 1> cam_orient_delta = cam_ang_vel * dT;
    if (noise_state != nullptr)
        cam_orient_delta += noise_state->middleRows<kAngAccelComps>(kAccelComps) * dT;

    Eigen::Matrix<Scalar, kQuat4, 1> cam_orient_delta_quat{};
    QuatFromAxisAngle(cam_orient_delta, &cam_orient_delta_quat);

    Eigen::Matrix<Scalar, kQuat4, 1> new_cam_orient_quat_tmp;
    QuatMult(cam_orient_quat, cam_orient_delta_quat, &new_cam_orient_quat_tmp);

    Scalar qorig_len = cam_orient_quat.norm();

    // normalize quaternion (formula A.141)
    Scalar q_len = new_cam_orient_quat_tmp.norm();
    if (normalize_quat)
        new_cam_orient_quat_tmp *= 1/q_len;
    Scalar q_len2 = new_cam_orient_quat_tmp.norm();

    new_cam_orient_quat = new_cam_orient_quat_tmp;

    // camera velocity is unchanged
    new_cam_vel = cam_vel;
    if (noise_state != nullptr)
        new_cam_vel += noise_state->middleRows<kAngAccelComps>(kAccelComps);

    // camera angular velocity is unchanged
    new_cam_ang_vel = cam_ang_vel;
    if (noise_state != nullptr)
        new_cam_vel += noise_state->middleRows<kAngAccelComps>(kAccelComps);
}

void DavisonMonoSlam::PredictEstimVars(EigenDynVec* predicted_estim_vars, EigenDynMat* predicted_estim_vars_covar) const
{
    // estimated vars
    std::array<Scalar, kCamStateComps> new_cam{};
    PredictCameraMotionByKinematicModel(gsl::make_span<const Scalar>(estim_vars_.data(), kCamStateComps), new_cam);

    //
    Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps> F;
    Deriv_cam_state_by_cam_state(&F);

    Eigen::Matrix<Scalar, kCamStateComps, kInputNoiseComps> G;
    Deriv_cam_state_by_input_noise(&G);

    static bool debug_F_G_derivatives = false;
    if (debug_F_G_derivatives)
    {
        Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps> finite_diff_F;
        FiniteDiff_cam_state_by_cam_state(gsl::make_span<const Scalar>(estim_vars_.data(), kCamStateComps), kFiniteDiffEpsDebug, &finite_diff_F);

        Scalar diff1 = (finite_diff_F - F).norm();

        Eigen::Matrix<Scalar, kCamStateComps, kInputNoiseComps> finite_diff_G;
        FiniteDiff_cam_state_by_input_noise(kFiniteDiffEpsDebug, &finite_diff_G);

        Scalar diff2 = (finite_diff_G - G).norm();
        int z = 0;
    }

    // Pvv = F*Pvv*Ft+G*Q*Gt
    Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps> Pvv_new =
        F * estim_vars_covar_.topLeftCorner<kCamStateComps, kCamStateComps>() * F.transpose() +
        G * input_noise_covar_ * G.transpose();
    
    // Pvm = F*Pvm
    size_t sal_pnts_vars_count = SalientPointsCount() * kSalientPointComps;
    Eigen::Matrix<Scalar, kCamStateComps, Eigen::Dynamic> Pvm_new = 
        F * estim_vars_covar_.topRightCorner(kCamStateComps, sal_pnts_vars_count);

    // update x
    *predicted_estim_vars = estim_vars_;
    predicted_estim_vars->topRows<kCamStateComps>() = Eigen::Map<const Eigen::Matrix<Scalar, kCamStateComps, 1>>(new_cam.data(), kCamStateComps);

    // update P
    *predicted_estim_vars_covar = estim_vars_covar_;
    predicted_estim_vars_covar->topLeftCorner<kCamStateComps, kCamStateComps>() = Pvv_new;
    predicted_estim_vars_covar->topRightCorner(kCamStateComps, sal_pnts_vars_count) = Pvm_new;
    predicted_estim_vars_covar->bottomLeftCorner(sal_pnts_vars_count, kCamStateComps) = Pvm_new.transpose();

    if (fix_estim_vars_covar_symmetry_)
        FixSymmetricMat(predicted_estim_vars_covar);
}

void DavisonMonoSlam::PredictEstimVarsHelper()
{
    // make predictions
    PredictEstimVars(&predicted_estim_vars_, &predicted_estim_vars_covar_);
}

void DavisonMonoSlam::ProcessFrame(size_t frame_ind)
{
    corners_matcher_->AnalyzeFrame(frame_ind);

    std::vector<std::pair<SalPntId, CornersMatcherBlobId>> matched_sal_pnts;
    corners_matcher_->MatchSalientPoints(frame_ind, latest_frame_sal_pnts_, &matched_sal_pnts);

    latest_frame_sal_pnts_.clear();
    for (auto[sal_pnt_id, blob_id] : matched_sal_pnts)
    {
        Point2 coord = corners_matcher_->GetBlobCoord(blob_id);
        GetSalPnt(sal_pnt_id).PixelCoordInLatestFrame = coord.Mat();
        latest_frame_sal_pnts_.insert(sal_pnt_id);
    }

    switch (kalman_update_impl_)
    {
    default:
    case 1:
        ProcessFrame_StackedObservationsPerUpdate(frame_ind);
        break;
    case 2:
        ProcessFrame_OneObservationPerUpdate(frame_ind);
        break;
    case 3:
        ProcessFrame_OneComponentOfOneObservationPerUpdate(frame_ind);
        break;
    }

    // eagerly try allocate new salient points
    std::vector<CornersMatcherBlobId> new_blobs;
    this->corners_matcher_->RecruitNewSalientPoints(frame_ind, latest_frame_sal_pnts_, &new_blobs);
    if (!new_blobs.empty())
    {
        CameraPosState cam_state;
        LoadCameraPosDataFromArray(Span(estim_vars_, kCamStateComps), &cam_state);

        for (auto blob_id : new_blobs)
        {
            Point2 coord = corners_matcher_->GetBlobCoord(blob_id);

            std::optional<Scalar> pnt_dist_gt;
            if (fake_sal_pnt_initial_inv_dist_)
            {
                pnt_dist_gt = corners_matcher_->GetSalientPointGroundTruthDepth(blob_id);
            }
            SalPntId sal_pnt_id = AddSalientPoint(cam_state, coord.Mat(), pnt_dist_gt);
            latest_frame_sal_pnts_.insert(sal_pnt_id);
            corners_matcher_->OnSalientPointIsAssignedToBlobId(sal_pnt_id, blob_id);
        }

        // now the estimated variables are changed, the dependent predicted variables must be updated too
        predicted_estim_vars_.resizeLike(estim_vars_);
        predicted_estim_vars_covar_.resizeLike(estim_vars_covar_);
    }

    // make predictions
    PredictEstimVars(&predicted_estim_vars_, &predicted_estim_vars_covar_);

    static bool debug_predicted_vars = false;
    if (debug_predicted_vars || DebugPath(DebugPathEnum::DebugPredictedVarsCov))
    {
        CheckCameraAndSalientPointsCovs(predicted_estim_vars_, predicted_estim_vars_covar_);
    }
}

void DavisonMonoSlam::ProcessFrame_StackedObservationsPerUpdate(size_t frame_ind)
{
    if (!latest_frame_sal_pnts_.empty())
    {
        // improve predicted estimation with the info from observations
        std::swap(estim_vars_, predicted_estim_vars_);
        std::swap(estim_vars_covar_, predicted_estim_vars_covar_);

        if (kSurikoDebug)
        {
            //predicted_estim_vars_.setConstant(kNan);
            //predicted_estim_vars_covar_.setConstant(kNan);
            // TODO: fix me; initialize predicted, because UI reads it without sync!
            //predicted_estim_vars_ = estim_vars_;
            //predicted_estim_vars_covar_ = estim_vars_covar_;
        }

        const auto& derive_at_pnt = estim_vars_;
        const auto& Pprev = estim_vars_covar_;

        CameraPosState cam_state;
        LoadCameraPosDataFromArray(Span(derive_at_pnt, kCamStateComps), &cam_state);

        Eigen::Matrix<Scalar, kEucl3, kEucl3> cam_orient_wfc;
        RotMatFromQuat(gsl::make_span<const Scalar>(cam_state.OrientationWfc.data(), kQuat4), &cam_orient_wfc);

        auto& cache = stacked_update_cache_;
        
        //
        //EigenDynMat Hk; // [2m,13+6n]
        auto& Hk = cache.H_;
        Deriv_H_by_estim_vars(cam_state, cam_orient_wfc, derive_at_pnt, &Hk);

        // evaluate filter gain
        //EigenDynMat Rk;
        auto& Rk = cache.R_;
        size_t obs_sal_pnt_count = latest_frame_sal_pnts_.size();
        FillRk(obs_sal_pnt_count, &Rk);

        // innovation variance S=H*P*Ht
        //auto innov_var = Hk * Pprev * Hk.transpose() + Rk; // [2m,2m]
        cache.H_P_.noalias() = Hk * Pprev;
        auto& innov_var = cache.innov_var_;
        innov_var.noalias() = cache.H_P_ * Hk.transpose(); // [2m,2m]
        innov_var.noalias() += Rk;
        
        //EigenDynMat innov_var_inv = innov_var.inverse();
        auto& innov_var_inv = cache.innov_var_inv_;
        innov_var_inv.noalias() = innov_var.inverse();

        // K=P*Ht*inv(S)
        //EigenDynMat Knew = Pprev * Hk.transpose() * innov_var_inv; // [13+6n,2m]
        auto& Knew = cache.Knew_;
        Knew.noalias() = cache.H_P_.transpose() * innov_var_inv; // [13+6n,2m]

        //
        //Eigen::Matrix<Scalar, Eigen::Dynamic, 1> zk;
        auto& zk = cache.zk_;
        zk.resize(obs_sal_pnt_count * kPixPosComps, 1);

        //Eigen::Matrix<Scalar, Eigen::Dynamic, 1> projected_sal_pnts;
        auto& projected_sal_pnts = cache.projected_sal_pnts_;
        projected_sal_pnts.resizeLike(zk);

        size_t obs_sal_pnt_ind = -1;
        for (SalPntId obs_sal_pnt_id : latest_frame_sal_pnts_)
        {
            ++obs_sal_pnt_ind;

            const SalPntInternal& sal_pnt = GetSalPnt(obs_sal_pnt_id);
            Point2 corner_pix = sal_pnt.PixelCoordInLatestFrame;
            zk.middleRows<kPixPosComps>(obs_sal_pnt_ind * kPixPosComps) = corner_pix.Mat();

            // project salient point into current camera

            EstimVarsSalientPoint sal_pnt_vars;
            LoadSalientPointDataFromArray(Span(derive_at_pnt).subspan(sal_pnt.EstimVarsInd, kSalientPointComps), &sal_pnt_vars);

            Eigen::Matrix<Scalar, kPixPosComps, 1> hd = ProjectInternalSalientPoint(cam_state, sal_pnt_vars, nullptr);
            projected_sal_pnts.middleRows<kPixPosComps>(obs_sal_pnt_ind * kPixPosComps) = hd;

            Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps> hd_by_cam_state;
            Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps> hd_by_sal_pnt;
            Deriv_hd_by_cam_state_and_sal_pnt(sal_pnt, cam_state, cam_orient_wfc, derive_at_pnt, &hd_by_cam_state, &hd_by_sal_pnt);
            SRK_ASSERT(true);
        }

        // Xnew=Xold+K(z-obs)
        // update estimated variables
        //EigenDynVec estim_vars_delta = Knew * (zk - projected_sal_pnts);
        //estim_vars_.noalias() = derive_at_pnt + estim_vars_delta;
        //estim_vars_ = derive_at_pnt;
        estim_vars_.noalias() += Knew * (zk - projected_sal_pnts);

        // update covariance matrix
        //size_t n = EstimatedVarsCount();
        //auto ident = EigenDynMat::Identity(n, n);
        //estim_vars_covar_.noalias() = (ident - Knew * Hk) * Pprev; // way1
        //estim_vars_covar_.noalias() = Pprev - Knew * innov_var * Knew.transpose(); // way2, 10% faster than way1
        
        cache.K_S_.noalias() = Knew * innov_var;

        //estim_vars_covar_ = Pprev;
        estim_vars_covar_.noalias() -= cache.K_S_ * Knew.transpose();

        // way1, impl of (I-K*H)P
        //stacked_update_cache_.tmp1_.noalias() = Knew * Hk;
        //stacked_update_cache_.tmp1_ -= ident;
        //estim_vars_covar_.noalias() = -stacked_update_cache_.tmp1_ * Pprev;

        if (fix_estim_vars_covar_symmetry_)
            FixSymmetricMat(&estim_vars_covar_);

        if (kSurikoDebug && gt_cam_orient_world_to_f_ != nullptr) // ground truth
        {
            SE3Transform cam_orient_cfw_gt = gt_cam_orient_world_to_f_(frame_ind);
            SE3Transform cam_orient_wfc_gt = SE3Inv(cam_orient_cfw_gt);

            Eigen::Matrix<Scalar, kQuat4, 1> cam_orient_wfc_quat;
            QuatFromRotationMatNoRChecks(cam_orient_wfc_gt.R, gsl::make_span<Scalar>(cam_orient_wfc_quat.data(), kQuat4));

            // print norm of delta with gt (pos,orient)
            CameraPosState cam_state_new;
            LoadCameraPosDataFromArray(gsl::make_span<const Scalar>(estim_vars_.data(), kCamStateComps), &cam_state_new);
            Scalar d1 = (cam_orient_wfc_gt.T - cam_state_new.PosW).norm();
            Scalar d2 = (cam_orient_wfc_quat - cam_state_new.OrientationWfc).norm();
            Scalar diff_gt = d1 + d2;
            Scalar estim_change = (zk - projected_sal_pnts).norm();
            VLOG(4) << "diff_gt=" << diff_gt << " zk-obs=" << estim_change;
        }

        static bool debug_estim_vars = false;
        if (debug_estim_vars || DebugPath(DebugPathEnum::DebugEstimVarsCov))
        {
            CheckCameraAndSalientPointsCovs(estim_vars_, estim_vars_covar_);
        }
    }
    else
    {
        // we have no observations => current state <- prediction
        std::swap(estim_vars_, predicted_estim_vars_);
        std::swap(estim_vars_covar_, predicted_estim_vars_covar_);
    }

    OnEstimVarsChanged(frame_ind);
}

void DavisonMonoSlam::ProcessFrame_OneObservationPerUpdate(size_t frame_ind)
{
    if (!latest_frame_sal_pnts_.empty())
    {
        // improve predicted estimation with the info from observations
        std::swap(estim_vars_, predicted_estim_vars_);
        std::swap(estim_vars_covar_, predicted_estim_vars_covar_);
        
        if (kSurikoDebug)
        {
            //predicted_estim_vars_.setConstant(kNan);
            //predicted_estim_vars_covar_.setConstant(kNan);
            // TODO: fix me; initialize predicted, because UI reads it without sync!
            predicted_estim_vars_ = estim_vars_;
            predicted_estim_vars_covar_ = estim_vars_covar_;
        }

        Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> Rk;
        FillRk2x2(&Rk);

        Scalar diff_vars_total = 0;
        Scalar diff_cov_total = 0;
        for (SalPntId obs_sal_pnt_id : latest_frame_sal_pnts_)
        {
            const SalPntInternal& sal_pnt = GetSalPnt(obs_sal_pnt_id);

            // the point where derivatives are calculated at
            const EigenDynVec& derive_at_pnt = estim_vars_;
            const EigenDynMat& Pprev = estim_vars_covar_;

            CameraPosState cam_state;
            LoadCameraPosDataFromArray(Span(derive_at_pnt, kCamStateComps), &cam_state);

            Eigen::Matrix<Scalar, kEucl3, kEucl3> cam_orient_wfc;
            RotMatFromQuat(gsl::make_span<const Scalar>(cam_state.OrientationWfc.data(), kQuat4), &cam_orient_wfc);

            DependsOnOverallPackOrder();
            const Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>& Pxx =
                Pprev.topLeftCorner<kCamStateComps, kCamStateComps>(); // camera-camera covariance

            Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps> hd_by_cam_state;
            Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps> hd_by_sal_pnt;
            Deriv_hd_by_cam_state_and_sal_pnt(sal_pnt, cam_state, cam_orient_wfc, derive_at_pnt, &hd_by_cam_state, &hd_by_sal_pnt);

            // 1. innovation variance S[2,2]

            size_t off = sal_pnt.EstimVarsInd;
            const Eigen::Matrix<Scalar, kCamStateComps, kSalientPointComps>& Pxy = 
                Pprev.block<kCamStateComps, kSalientPointComps>(0, off); // camera-sal_pnt covariance
            const Eigen::Matrix<Scalar, kSalientPointComps, kSalientPointComps>& Pyy =
                Pprev.block<kSalientPointComps, kSalientPointComps>(off, off); // sal_pnt-sal_pnt covariance

            Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> mid = 
                hd_by_cam_state * Pxy * hd_by_sal_pnt.transpose();

            Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> innov_var_2x2 =
                hd_by_cam_state * Pxx * hd_by_cam_state.transpose() +
                mid + mid.transpose() +
                hd_by_sal_pnt * Pyy * hd_by_sal_pnt.transpose() +
                Rk;
            Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> innov_var_inv_2x2 = innov_var_2x2.inverse();

            // 2. filter gain [13+6n, 2]: K=(Px*Hx+Py*Hy)*inv(S)

            one_obs_per_update_cache_.P_Hxy_.noalias() = Pprev.leftCols<kCamStateComps>() * hd_by_cam_state.transpose(); // P*Hx
            one_obs_per_update_cache_.P_Hxy_.noalias() += Pprev.middleCols<kSalientPointComps>(off) * hd_by_sal_pnt.transpose(); // P*Hy

            auto& Knew = one_obs_per_update_cache_.Knew_;
            Knew.noalias() = one_obs_per_update_cache_.P_Hxy_ * innov_var_inv_2x2;

            // 3. update X and P using info derived from salient point observation
            suriko::Point2 corner_pix = sal_pnt.PixelCoordInLatestFrame;
            
            // project salient point into current camera

            EstimVarsSalientPoint sal_pnt_vars;
            LoadSalientPointDataFromArray(Span(derive_at_pnt).subspan(sal_pnt.EstimVarsInd, kSalientPointComps), &sal_pnt_vars);

            Eigen::Matrix<Scalar, kPixPosComps, 1> hd = ProjectInternalSalientPoint(cam_state, sal_pnt_vars, nullptr);

            //
            auto estim_vars_delta = Knew * (corner_pix.Mat() - hd);

            one_obs_per_update_cache_.K_S_.noalias() = Knew * innov_var_2x2; // cache
            auto estim_vars_covar_delta = one_obs_per_update_cache_.K_S_ * Knew.transpose();

            if (kSurikoDebug)
            {
                Scalar estim_vars_delta_norm = estim_vars_delta.norm();
                diff_vars_total += estim_vars_delta_norm;
                
                Scalar estim_vars_covar_delta_norm = estim_vars_covar_delta.norm();
                diff_cov_total += estim_vars_covar_delta_norm;
            }

            //
            estim_vars_.noalias() += estim_vars_delta;
            estim_vars_covar_.noalias() -= estim_vars_covar_delta;
        }
        if (fix_estim_vars_covar_symmetry_)
            FixSymmetricMat(&estim_vars_covar_);

        if (kSurikoDebug)
        {
            VLOG(4) << "diff_vars=" << diff_vars_total << " diff_cov=" << diff_cov_total;
        }
    }
    else
    {
        // we have no observations => current state <- prediction
        std::swap(estim_vars_, predicted_estim_vars_);
        std::swap(estim_vars_covar_, predicted_estim_vars_covar_);
    }

    static bool debug_estim_vars = false;
    if (debug_estim_vars || DebugPath(DebugPathEnum::DebugEstimVarsCov))
    {
        CheckCameraAndSalientPointsCovs(estim_vars_, estim_vars_covar_);
    }

    OnEstimVarsChanged(frame_ind);
}

void DavisonMonoSlam::ProcessFrame_OneComponentOfOneObservationPerUpdate(size_t frame_ind)
{
    if (!latest_frame_sal_pnts_.empty())
    {
        // improve predicted estimation with the info from observations
        std::swap(estim_vars_, predicted_estim_vars_);
        std::swap(estim_vars_covar_, predicted_estim_vars_covar_);
        
        if (kSurikoDebug)
        {
            //predicted_estim_vars_.setConstant(kNan);
            //predicted_estim_vars_covar_.setConstant(kNan);
            // TODO: fix me; initialize predicted, because UI reads it without sync!
            predicted_estim_vars_ = estim_vars_;
            predicted_estim_vars_covar_ = estim_vars_covar_;
        }

        Scalar diff_vars_total = 0;
        Scalar diff_cov_total = 0;
        Scalar measurm_noise_variance = suriko::Sqr(static_cast<Scalar>(measurm_noise_std_)); // R[1,1]

        for (SalPntId obs_sal_pnt_id : latest_frame_sal_pnts_)
        {
            const SalPntInternal& sal_pnt = GetSalPnt(obs_sal_pnt_id);

            // get observation corner
            Point2 corner_pix = sal_pnt.PixelCoordInLatestFrame;

            for (size_t obs_comp_ind = 0; obs_comp_ind < kPixPosComps; ++obs_comp_ind)
            {
                // the point where derivatives are calculated at
                // attach to the latest state and P
                const EigenDynVec& derive_at_pnt = estim_vars_;
                const EigenDynMat& Pprev = estim_vars_covar_;

                CameraPosState cam_state;
                LoadCameraPosDataFromArray(Span(derive_at_pnt, kCamStateComps), &cam_state);

                Eigen::Matrix<Scalar, kEucl3, kEucl3> cam_orient_wfc;
                RotMatFromQuat(gsl::make_span<const Scalar>(cam_state.OrientationWfc.data(), kQuat4), &cam_orient_wfc);

                DependsOnOverallPackOrder();
                const Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>& Pxx =
                    Pprev.topLeftCorner<kCamStateComps, kCamStateComps>(); // camera-camera covariance

                size_t off = sal_pnt.EstimVarsInd;
                const Eigen::Matrix<Scalar, kCamStateComps, kSalientPointComps>& Pxy =
                    Pprev.block<kCamStateComps, kSalientPointComps>(0, off); // camera-sal_pnt covariance
                const Eigen::Matrix<Scalar, kSalientPointComps, kSalientPointComps>& Pyy =
                    Pprev.block<kSalientPointComps, kSalientPointComps>(off, off); // sal_pnt-sal_pnt covariance

                Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps> hd_by_cam_state;
                Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps> hd_by_sal_pnt;
                Deriv_hd_by_cam_state_and_sal_pnt(sal_pnt, cam_state, cam_orient_wfc, derive_at_pnt, &hd_by_cam_state, &hd_by_sal_pnt);

                // 1. innovation variance is a scalar (one element matrix S[1,1])
                auto obs_comp_by_cam_state = hd_by_cam_state.middleRows<1>(obs_comp_ind); // [1,13]
                auto obs_comp_by_sal_pnt = hd_by_sal_pnt.middleRows<1>(obs_comp_ind); // [1,6]

                typedef Eigen::Matrix<Scalar, 1, 1> EigenMat11;

                EigenMat11 mid_1x1 = obs_comp_by_cam_state * Pxy * obs_comp_by_sal_pnt.transpose();

                EigenMat11 innov_var_1x1 =
                    obs_comp_by_cam_state * Pxx * obs_comp_by_cam_state.transpose() +
                    obs_comp_by_sal_pnt * Pyy * obs_comp_by_sal_pnt.transpose();

                Scalar innov_var = innov_var_1x1[0] + 2 * mid_1x1[0] + measurm_noise_variance;

                Scalar innov_var_inv = 1 / innov_var;

                // 2. filter gain [13+6n, 1]: K=(Px*Hx+Py*Hy)*inv(S)
                auto& Knew = one_comp_of_obs_per_update_cache_.Knew_;
                Knew.noalias() = innov_var_inv * Pprev.leftCols<kCamStateComps>() * obs_comp_by_cam_state.transpose();
                Knew.noalias() += innov_var_inv * Pprev.middleCols<kSalientPointComps>(off) * obs_comp_by_sal_pnt.transpose();

                //
                // project salient point into current camera
                EstimVarsSalientPoint sal_pnt_vars;
                LoadSalientPointDataFromArray(Span(derive_at_pnt).subspan(sal_pnt.EstimVarsInd, kSalientPointComps), &sal_pnt_vars);

                Eigen::Matrix<Scalar, kPixPosComps, 1> hd = ProjectInternalSalientPoint(cam_state, sal_pnt_vars, nullptr);

                // 3. update X and P using info derived from salient point observation

                auto estim_vars_delta = Knew * (corner_pix[obs_comp_ind] - hd[obs_comp_ind]);

                // keep outer product K*Kt lazy ([13+6n,1]*[1,13+6n]=[13+6n,13+6n])
                
                auto estim_vars_covar_delta = innov_var * (Knew * Knew.transpose()); // [13+6n,13+6n]

                // NOTE: (K*Kt)S is 3 times slower than S*(K*Kt) or (S*K)Kt, S=scalar. Why?
                //auto estim_vars_covar_delta = (Knew * Knew.transpose()) * innov_var; // [13+6n,13+6n] slow!!!
                
                if (kSurikoDebug)
                {
                    Scalar estim_vars_delta_norm = estim_vars_delta.norm();
                    diff_vars_total += estim_vars_delta_norm;

                    Scalar estim_vars_covar_delta_norm = estim_vars_covar_delta.norm();
                    diff_cov_total += estim_vars_covar_delta_norm;
                }

                //
                estim_vars_.noalias() += estim_vars_delta;
                estim_vars_covar_.noalias() -= estim_vars_covar_delta;
            }
        }

        if (fix_estim_vars_covar_symmetry_)
            FixSymmetricMat(&estim_vars_covar_);

        if (kSurikoDebug)
        {
            VLOG(4) << "diff_vars=" << diff_vars_total << " diff_cov=" << diff_cov_total;
        }
    }
    else
    {
        // we have no observations => current state <- prediction
        std::swap(estim_vars_, predicted_estim_vars_);
        std::swap(estim_vars_covar_, predicted_estim_vars_covar_);
    }

    static bool debug_estim_vars = false;
    if (debug_estim_vars || DebugPath(DebugPathEnum::DebugEstimVarsCov))
    {
        CheckCameraAndSalientPointsCovs(estim_vars_, estim_vars_covar_);
    }

    OnEstimVarsChanged(frame_ind);
}

void DavisonMonoSlam::OnEstimVarsChanged(size_t frame_ind)
{
    if (fake_localization_ && gt_cam_orient_world_to_f_ != nullptr)
    {
        SE3Transform cam_orient_cfw = gt_cam_orient_world_to_f_(frame_ind);
        SE3Transform cam_orient_wfc = SE3Inv(cam_orient_cfw);

        std::array<Scalar, kQuat4> cam_orient_wfc_quat;
        QuatFromRotationMatNoRChecks(cam_orient_wfc.R, cam_orient_wfc_quat);

        // camera position
        DependsOnCameraPosPackOrder();
        estim_vars_[0] = cam_orient_wfc.T[0];
        estim_vars_[1] = cam_orient_wfc.T[1];
        estim_vars_[2] = cam_orient_wfc.T[2];
        
        // camera orientation
        estim_vars_[3] = cam_orient_wfc_quat[0];
        estim_vars_[4] = cam_orient_wfc_quat[1];
        estim_vars_[5] = cam_orient_wfc_quat[2];
        estim_vars_[6] = cam_orient_wfc_quat[3];
    }
}

DavisonMonoSlam::SalPntId DavisonMonoSlam::AddSalientPoint(const CameraPosState& cam_state, suriko::Point2 corner_pix, std::optional<Scalar> pnt_dist_gt)
{
    // undistort 2D image coordinate
    Eigen::Matrix<Scalar, kPixPosComps, 1> hd = corner_pix.Mat(); // distorted
    Eigen::Matrix<Scalar, kPixPosComps, 1> hu = hd; // undistorted
    
    Scalar Cx = cam_intrinsics_.PrincipalPointPixels[0];
    Scalar Cy = cam_intrinsics_.PrincipalPointPixels[1];
    Scalar dx = cam_intrinsics_.PixelSizeMm[0];
    Scalar dy = cam_intrinsics_.PixelSizeMm[1];
    Scalar f = cam_intrinsics_.FocalLengthMm;

    // A.58
    Eigen::Matrix<Scalar, kEucl3, 1> hc;
    hc[0] = -(hu[0] - Cx) / (f / dx);
    hc[1] = -(hu[1] - Cy) / (f / dy);
    hc[2] = 1;

    Eigen::Matrix<Scalar, kEucl3, kEucl3> Rwfc;
    RotMatFromQuat(Span(cam_state.OrientationWfc), &Rwfc);

    // A.59
    Eigen::Matrix<Scalar, kEucl3, 1> hw = Rwfc * hc;
    Scalar azim_theta = std::atan2(hw[0], hw[2]);
    Scalar elev_phi = std::atan2(-hw[1], std::sqrt(suriko::Sqr(hw[0]) + suriko::Sqr(hw[2])));

    // allocate space for estimated variables
    size_t old_vars_count = EstimatedVarsCount();
    estim_vars_.conservativeResize(old_vars_count + kSalientPointComps);

    size_t old_sal_pnts_count = SalientPointsCount();
    size_t sal_pnt_var_ind = SalientPointOffset(old_sal_pnts_count);
    auto dst_vars = gsl::make_span<Scalar>(&estim_vars_[sal_pnt_var_ind], kSalientPointComps);
    DependsOnSalientPointPackOrder();
    dst_vars[0] = cam_state.PosW[0];
    dst_vars[1] = cam_state.PosW[1];
    dst_vars[2] = cam_state.PosW[2];
    dst_vars[3] = azim_theta;
    dst_vars[4] = elev_phi;
    
    // NOTE: initial inverse depth is constant because the first image can't provide the depth information
    if (pnt_dist_gt.has_value())
        dst_vars[5] = 1/pnt_dist_gt.value();
    else
        dst_vars[5] = sal_pnt_init_inv_dist_;

    // P

    // Pold is augmented with 6 rows and columns corresponding to how new salient point interact with all other
    // variables and itself. So Pnew=Pold+6rowscols. The values of Pold itself are unchanged.
    // the Eigen's conservative resize uses temporary to resize and copy matrix, slow
    estim_vars_covar_.conservativeResize(old_vars_count + kSalientPointComps, old_vars_count + kSalientPointComps);

    Eigen::Matrix<Scalar, 1, kEucl3> azim_theta_by_hw;
    Eigen::Matrix<Scalar, 1, kEucl3> elev_phi_by_hw;
    Deriv_azim_theta_elev_phi_by_hw(hw, &azim_theta_by_hw, &elev_phi_by_hw);

    Eigen::Matrix<Scalar, kSalientPointComps, kQuat4> sal_pnt_by_cam_q;
    Deriv_sal_pnt_by_cam_q(cam_state, hc, azim_theta_by_hw, elev_phi_by_hw, &sal_pnt_by_cam_q);

    constexpr size_t kCamPQ = kEucl3 + kQuat4;

    Eigen::Matrix<Scalar, kSalientPointComps, kCamPQ> sal_pnt_by_cam;
    sal_pnt_by_cam.block<kEucl3, kEucl3>(0, 0).setIdentity();
    sal_pnt_by_cam.block<kEucl3, kEucl3>(kEucl3, 0).setZero();
    sal_pnt_by_cam.block<kSalientPointComps, kQuat4>(0, kEucl3) = sal_pnt_by_cam_q;

    // // A76-A.79
    Eigen::Matrix<Scalar, kSalientPointComps-kRhoComps, kEucl3> sal_pnt_by_hw;
    sal_pnt_by_hw.topRows<kEucl3>().setZero();
    sal_pnt_by_hw.middleRows<1>(kEucl3) = azim_theta_by_hw;
    sal_pnt_by_hw.middleRows<1>(kEucl3+1) = elev_phi_by_hw; // +1 for azimuth component

    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& hw_by_hc = Rwfc;
    
    Eigen::Matrix<Scalar, kEucl3, kPixPosComps> hc_by_hu;
    hc_by_hu.setZero();
    hc_by_hu(0, 0) = -dx / f;
    hc_by_hu(1, 1) = -dy / f;

    Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> hu_by_hd;
    Deriv_hu_by_hd(corner_pix, &hu_by_hd);

    // A.75
    Eigen::Matrix<Scalar, kSalientPointComps, kPixPosComps + kRhoComps> sal_pnt_by_h_rho;
    sal_pnt_by_h_rho.topLeftCorner<kSalientPointComps - kRhoComps, kPixPosComps>() = sal_pnt_by_hw * hw_by_hc * hc_by_hu * hu_by_hd;
    sal_pnt_by_h_rho.bottomLeftCorner<kRhoComps, kPixPosComps>().setZero();
    sal_pnt_by_h_rho.rightCols<kRhoComps>().setZero();
    sal_pnt_by_h_rho.bottomRightCorner<kRhoComps, kRhoComps>().setOnes(); // single element

    // store J*P in the bottom left corner of Pnew

    auto Pnew_bottom_left = estim_vars_covar_.bottomLeftCorner<kSalientPointComps, Eigen::Dynamic>(kSalientPointComps, old_vars_count);
    Pnew_bottom_left.noalias() = sal_pnt_by_cam * estim_vars_covar_.topLeftCorner<kCamPQ, Eigen::Dynamic>(kEucl3 + kQuat4, old_vars_count);

    estim_vars_covar_.topRightCorner<Eigen::Dynamic, kSalientPointComps>(old_vars_count, kSalientPointComps) = Pnew_bottom_left.transpose();

    // P bottom right corner
    Eigen::Matrix <Scalar, kPixPosComps, kPixPosComps> R;
    FillRk2x2(&R);

    auto Pyy = estim_vars_covar_.bottomRightCorner<kSalientPointComps, kSalientPointComps>();
    Pyy = Pnew_bottom_left.leftCols<kCamPQ>() * sal_pnt_by_cam.transpose();
    Pyy +=
        sal_pnt_by_h_rho.leftCols<kPixPosComps>() * R * sal_pnt_by_h_rho.leftCols<kPixPosComps>().transpose() +
        sal_pnt_by_h_rho.rightCols<kRhoComps>() * sal_pnt_init_inv_dist_std_ * sal_pnt_by_h_rho.rightCols<kRhoComps>().transpose();

    //
    if (kSurikoDebug)
    {
        EstimVarsSalientPoint sal_pnt_vars;
        LoadSalientPointDataFromArray(Span(estim_vars_), &sal_pnt_vars);

        Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_pos;
        Eigen::Matrix<Scalar, kEucl3, kEucl3> sal_pnt_pos_uncert;
        LoadSalientPointPredictedPosWithUncertainty(estim_vars_, estim_vars_covar_, old_sal_pnts_count, &sal_pnt_pos, &sal_pnt_pos_uncert);

        bool ok = CanExtractEllipsoid(sal_pnt_pos_uncert);
        SRK_ASSERT(ok);
    }

    //
    static bool debug_estimated_vars = false;
    if (debug_estimated_vars || DebugPath(DebugPathEnum::DebugEstimVarsCov))
    {
        size_t salient_pnt_ind = old_sal_pnts_count;
        Eigen::Matrix<Scalar, 3, 1> pos_mean;
        Eigen::Matrix<Scalar, 3, 3> pos_uncert;
        LoadSalientPointPredictedPosWithUncertainty(estim_vars_, estim_vars_covar_, salient_pnt_ind, &pos_mean, &pos_uncert);

        QuadricEllipsoidWithCenter ellipsoid;
        ExtractEllipsoidFromUncertaintyMat(pos_mean, pos_uncert, 0.05, &ellipsoid);
        SRK_ASSERT(true);
    }

    // ID of new salient point
    sal_pnts_.push_back(std::make_unique<SalPntInternal>());
    SalPntInternal& sal_pnt = *sal_pnts_.back().get();
    sal_pnt.EstimVarsInd = sal_pnt_var_ind;
    sal_pnt.SalPntIndDebug = old_sal_pnts_count;
    sal_pnt.PixelCoordInLatestFrame = hd;
    return SalPntId(sal_pnts_.back().get());
}

void DavisonMonoSlam::Deriv_hu_by_hd(suriko::Point2 corner_pix, Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* hu_by_hd) const
{
    Scalar ud = corner_pix[0];
    Scalar vd = corner_pix[1];
    Scalar Cx = cam_intrinsics_.PrincipalPointPixels[0];
    Scalar Cy = cam_intrinsics_.PrincipalPointPixels[1];
    Scalar dx = cam_intrinsics_.PixelSizeMm[0];
    Scalar dy = cam_intrinsics_.PixelSizeMm[1];
    Scalar k1 = cam_distort_params_.K1;
    Scalar k2 = cam_distort_params_.K2;

    Scalar rd = std::sqrt(
        suriko::Sqr(dx * (ud - Cx)) +
        suriko::Sqr(dy * (vd - Cy)));

    Scalar tort = 1 + k1 * suriko::Sqr(rd) + k2 * suriko::Pow4(rd);

    Scalar p2 = (k1 + 2 * k2 * suriko::Sqr(rd));

    auto& r = *hu_by_hd;
    r(0, 0) = tort + 2 * suriko::Sqr(dx * (ud - Cx))*p2;
    r(1, 1) = tort + 2 * suriko::Sqr(dy * (vd - Cy))*p2;
    r(1, 0) = 2 * suriko::Sqr(dx) * (vd - Cy)*(ud - Cx)*p2;
    r(0, 1) = 2 * suriko::Sqr(dy) * (vd - Cy)*(ud - Cx)*p2;
}

void DavisonMonoSlam::Deriv_hd_by_hu(suriko::Point2 corner_pix, Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* hd_by_hu) const
{
    Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> hu_by_hd;
    Deriv_hu_by_hd(corner_pix, &hu_by_hd);

    *hd_by_hu = hu_by_hd.inverse(); // A.33
    hd_by_hu->setIdentity(); // TODO: for now, suppport only 'no distortion' model
}

void DavisonMonoSlam::Deriv_hu_by_hc(const SalPntProjectionIntermidVars& proj_hist, Eigen::Matrix<Scalar, kPixPosComps, kEucl3>* hu_by_hc) const
{
    Scalar f = cam_intrinsics_.FocalLengthMm;
    Scalar hcx = proj_hist.hc[0];
    Scalar hcy = proj_hist.hc[1];
    Scalar hcz = proj_hist.hc[2];
    Scalar dx = cam_intrinsics_.PixelSizeMm[0];
    Scalar dy = cam_intrinsics_.PixelSizeMm[1];
    auto& m = *hu_by_hc;
    m(0, 0) = -f / (dx * hcz);
    m(1, 0) = 0;
    m(0, 1) = 0;
    m(1, 1) = -f / (dy * hcz);
    m(0, 2) = f * hcx / (dx * hcz * hcz);
    m(1, 2) = f * hcy / (dy * hcz * hcz);
}

void DavisonMonoSlam::Deriv_R_by_q(const Eigen::Matrix<Scalar, kQuat4, 1>& q,
    Eigen::Matrix<Scalar, 3, 3>* dR_by_dq0,
    Eigen::Matrix<Scalar, 3, 3>* dR_by_dq1,
    Eigen::Matrix<Scalar, 3, 3>* dR_by_dq2,
    Eigen::Matrix<Scalar, 3, 3>* dR_by_dq3) const
{
    // A.46-A.49
    *dR_by_dq0 <<
        2 * q[0], -2 * q[3], 2 * q[2],
        2 * q[3], 2 * q[0], -2 * q[1],
        -2 * q[2], 2 * q[1], 2 * q[0];

    *dR_by_dq1 <<
        2 * q[1], 2 * q[2], 2 * q[3],
        2 * q[2], -2 * q[1], -2 * q[0],
        2 * q[3], 2 * q[0], -2 * q[1];

    *dR_by_dq2 <<
        -2 * q[2], 2 * q[1], 2 * q[0],
        2 * q[1], 2 * q[2], 2 * q[3],
        -2 * q[0], 2 * q[3], -2 * q[2];

    *dR_by_dq3 <<
        -2 * q[3], -2 * q[0], 2 * q[1],
        2 * q[0], -2 * q[3], 2 * q[2],
        2 * q[1], 2 * q[2], 2 * q[3];
}

void DavisonMonoSlam::Deriv_hd_by_camera_state(const EstimVarsSalientPoint& sal_pnt,
    const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
    const SalPntProjectionIntermidVars& proj_hist,
    const Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>& hd_by_hu,
    const Eigen::Matrix<Scalar, kPixPosComps, kEucl3>& hu_by_hc,
    Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_xc) const
{
    hd_by_xc->setZero();

    //
    Eigen::Matrix<Scalar, kEucl3, kEucl3> Rcw = cam_orient_wfc.transpose();

    Eigen::Matrix<Scalar, kEucl3, kEucl3> dhc_by_dr = -sal_pnt.InverseDistRho * Rcw;

    Eigen::Matrix<Scalar, kPixPosComps, kEucl3> dh_by_dr = hd_by_hu * hu_by_hc * dhc_by_dr; // A.31
    hd_by_xc->middleCols<kEucl3>(0) = dh_by_dr;


    //
    Eigen::Matrix<Scalar, kQuat4, 1> cam_orient_cfw = QuatInverse(cam_state.OrientationWfc);
    const auto& q = cam_orient_cfw;

    Eigen::Matrix<Scalar, kEucl3, kEucl3> dR_by_dq0;
    Eigen::Matrix<Scalar, kEucl3, kEucl3> dR_by_dq1;
    Eigen::Matrix<Scalar, kEucl3, kEucl3> dR_by_dq2;
    Eigen::Matrix<Scalar, kEucl3, kEucl3> dR_by_dq3;
    Deriv_R_by_q(q, &dR_by_dq0, &dR_by_dq1, &dR_by_dq2, &dR_by_dq3);

    Eigen::Matrix<Scalar, 3, 1> cam_pos = cam_state.PosW;
    Eigen::Matrix<Scalar, 3, 1> first_cam_pos = sal_pnt.FirstCamPosW;

    Eigen::Matrix<Scalar, 3, 1> part2 = sal_pnt.InverseDistRho * (first_cam_pos - cam_pos) + proj_hist.FirstCamSalPntUnityDir;

    Eigen::Matrix<Scalar, 3, 4> dhc_by_dqcw; // A.40
    dhc_by_dqcw.middleCols<1>(0) = dR_by_dq0 * part2;
    dhc_by_dqcw.middleCols<1>(1) = dR_by_dq1 * part2;
    dhc_by_dqcw.middleCols<1>(2) = dR_by_dq2 * part2;
    dhc_by_dqcw.middleCols<1>(3) = dR_by_dq3 * part2;

    Eigen::Matrix<Scalar, 4, 4> dqcw_by_dqwc; // A.39
    dqcw_by_dqwc.setIdentity();
    dqcw_by_dqwc(1, 1) = -1;
    dqcw_by_dqwc(2, 2) = -1;
    dqcw_by_dqwc(3, 3) = -1;

    Eigen::Matrix<Scalar, kEucl3, kQuat4> dhc_by_dqwc = dhc_by_dqcw * dqcw_by_dqwc;

    //
    Eigen::Matrix<Scalar, kPixPosComps, kQuat4> dh_by_dqwc = hd_by_hu * hu_by_hc * dhc_by_dqwc;
    hd_by_xc->middleCols<kQuat4>(kEucl3) = dh_by_dqwc;
}

void DavisonMonoSlam::Deriv_hd_by_sal_pnt(const EstimVarsSalientPoint& sal_pnt,
    const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
    const Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>& hd_by_hu,
    const Eigen::Matrix<Scalar, kPixPosComps, kEucl3>& hu_by_hc,
    Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt) const
{
    Eigen::Matrix<Scalar, kEucl3, kEucl3> Rcw = cam_orient_wfc.transpose();

    Scalar cos_phi = std::cos(sal_pnt.ElevationPhiW);
    Scalar sin_phi = std::sin(sal_pnt.ElevationPhiW);
    Scalar cos_theta = std::cos(sal_pnt.AzimuthThetaW);
    Scalar sin_theta = std::sin(sal_pnt.AzimuthThetaW);

    // A.53
    Eigen::Matrix<Scalar, 3, 1> dm_by_dtheta; // azimuth
    dm_by_dtheta[0] = cos_phi * cos_theta;
    dm_by_dtheta[1] = 0;
    dm_by_dtheta[2] = -cos_phi * sin_theta;

    // A.54
    Eigen::Matrix<Scalar, 3, 1> dm_by_dphi; // elevation
    dm_by_dphi[0] = -sin_phi * sin_theta;
    dm_by_dphi[1] = -cos_phi;
    dm_by_dphi[2] = -sin_phi * cos_theta;

    // A.52
    Eigen::Matrix<Scalar, 3, kSalientPointComps> dhc_by_dy;
    dhc_by_dy.middleCols<3>(0) = sal_pnt.InverseDistRho * Rcw;
    dhc_by_dy.middleCols<1>(3) = Rcw * dm_by_dtheta;
    dhc_by_dy.middleCols<1>(4) = Rcw * dm_by_dphi;
    dhc_by_dy.middleCols<1>(5) = Rcw * (sal_pnt.FirstCamPosW - cam_state.PosW);

    // A.51
    *hd_by_sal_pnt = hd_by_hu * hu_by_hc * dhc_by_dy;
}

void DavisonMonoSlam::Deriv_azim_theta_elev_phi_by_hw(
    const Eigen::Matrix<Scalar, kEucl3, 1>& hw,
    Eigen::Matrix<Scalar, 1, kEucl3>* azim_theta_by_hw,
    Eigen::Matrix<Scalar, 1, kEucl3>* elev_phi_by_hw) const
{
    Scalar dist_xz_sqr = suriko::Sqr(hw[0]) + suriko::Sqr(hw[2]);

    (*azim_theta_by_hw)[0] = hw[2] / dist_xz_sqr;
    (*azim_theta_by_hw)[1] = 0;
    (*azim_theta_by_hw)[2] = -hw[0] / dist_xz_sqr;

    Scalar dist_sqr = dist_xz_sqr + suriko::Sqr(hw[1]);
    Scalar dist_xz = std::sqrt(dist_xz_sqr);
    Scalar s = hw[1] / (dist_sqr * dist_xz);

    (*elev_phi_by_hw)[0] = hw[0] * s;
    (*elev_phi_by_hw)[1] = -dist_xz / dist_sqr;
    (*elev_phi_by_hw)[2] = hw[2] * s;
}

void DavisonMonoSlam::Deriv_sal_pnt_by_cam_q(const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, 1>& hc,
    const Eigen::Matrix<Scalar, 1, kEucl3>& azim_theta_by_hw,
    const Eigen::Matrix<Scalar, 1, kEucl3>& elev_phi_by_hw,
    Eigen::Matrix<Scalar, kSalientPointComps, kQuat4>* sal_pnt_by_cam_q) const
{
    Eigen::Matrix<Scalar, kEucl3, kEucl3> dR_by_dq0;
    Eigen::Matrix<Scalar, kEucl3, kEucl3> dR_by_dq1;
    Eigen::Matrix<Scalar, kEucl3, kEucl3> dR_by_dq2;
    Eigen::Matrix<Scalar, kEucl3, kEucl3> dR_by_dq3;
    Deriv_R_by_q(cam_state.OrientationWfc, &dR_by_dq0, &dR_by_dq1, &dR_by_dq2, &dR_by_dq3);

    // A.73
    Eigen::Matrix<Scalar, kEucl3, kQuat4> hw_by_qwfc;
    hw_by_qwfc.middleCols<1>(0) = dR_by_dq0 * hc;
    hw_by_qwfc.middleCols<1>(1) = dR_by_dq1 * hc;
    hw_by_qwfc.middleCols<1>(2) = dR_by_dq2 * hc;
    hw_by_qwfc.middleCols<1>(3) = dR_by_dq3 * hc;

    sal_pnt_by_cam_q->topRows<kEucl3>().setZero();
    sal_pnt_by_cam_q->bottomRows<kRhoComps>().setZero();
    sal_pnt_by_cam_q->middleRows<1>(kEucl3) = azim_theta_by_hw * hw_by_qwfc;
    sal_pnt_by_cam_q->middleRows<1>(kEucl3 + 1) = elev_phi_by_hw * hw_by_qwfc; // +1 for azimuth component
}

Eigen::Matrix<Scalar, kPixPosComps,1> DavisonMonoSlam::ProjectInternalSalientPoint(const CameraPosState& cam_state, const EstimVarsSalientPoint& sal_pnt,
    SalPntProjectionIntermidVars *proj_hist) const
{
    Eigen::Matrix<Scalar, kEucl3, kEucl3> camk_orient_wfc;
    RotMatFromQuat(gsl::make_span<const Scalar>(cam_state.OrientationWfc.data(), kQuat4), &camk_orient_wfc);

    Eigen::Matrix<Scalar, kEucl3, kEucl3> camk_orient_cfw33 = camk_orient_wfc.transpose();

    SE3Transform camk_wfc(camk_orient_wfc, cam_state.PosW);
    SE3Transform camk_orient_cfw = SE3Inv(camk_wfc);

    //
    Eigen::Matrix<Scalar, kEucl3, 1> m;
    CameraCoordinatesEuclidUnityDirFromPolarAngles(sal_pnt.AzimuthThetaW, sal_pnt.ElevationPhiW, &m[0], &m[1], &m[2]);

    if (proj_hist != nullptr)
    {
        proj_hist->FirstCamSalPntUnityDir = m;
    }

    Scalar dist = sal_pnt.GetDist();
    auto m_cam_dir = (camk_orient_cfw33 * m).eval();
    auto m_cam = (camk_orient_cfw33 * m * sal_pnt.GetDist()).eval();

    // salient point in the world
    Eigen::Matrix<Scalar, kEucl3, 1> p_world = sal_pnt.FirstCamPosW + (1 / sal_pnt.InverseDistRho) * m;

    // salient point in the cam-k (naive way)
    Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_world_cam_orig = sal_pnt.FirstCamPosW - cam_state.PosW + (1 / sal_pnt.InverseDistRho)*m;
    Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_camk_simple1 = camk_orient_cfw33 * (sal_pnt.FirstCamPosW - cam_state.PosW + (1 / sal_pnt.InverseDistRho)*m);
    suriko::Point3 p_world_back_simple1 = SE3Apply(camk_wfc, suriko::Point3(sal_pnt_camk_simple1));

    // direction to the salient point in the cam-k divided by distance to the feature from the first camera
    Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_camk_scaled = camk_orient_cfw33 * (sal_pnt.InverseDistRho * (sal_pnt.FirstCamPosW - cam_state.PosW) + m);
    Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_camk = sal_pnt_camk_scaled / sal_pnt.InverseDistRho;

    const auto& sal_pnt_cam = sal_pnt_camk_scaled;
    Eigen::Matrix<Scalar, kPixPosComps, 1> hd = ProjectCameraSalientPoint(sal_pnt_cam, proj_hist);
    return hd;
}

Eigen::Matrix<Scalar, kPixPosComps,1> DavisonMonoSlam::ProjectCameraSalientPoint(
    const Eigen::Matrix<Scalar, kEucl3, 1>& pnt_camera,
    SalPntProjectionIntermidVars *proj_hist) const
{
    std::array<Scalar, 2> focal_length_pixels = cam_intrinsics_.GetFocalLengthPixels();

    // hc(X,Y,Z)->hu(uu,vu): project 3D salient point in camera-k onto image (pixels)
    //const auto& sal_pnt_cam = sal_pnt_camk_scaled;
    const auto& sal_pnt_cam = pnt_camera;
    Eigen::Matrix<Scalar, kPixPosComps, 1> hu;
    hu[0] = cam_intrinsics_.PrincipalPointPixels[0] - focal_length_pixels[0] * sal_pnt_cam[0] / sal_pnt_cam[2];
    hu[1] = cam_intrinsics_.PrincipalPointPixels[1] - focal_length_pixels[1] * sal_pnt_cam[1] / sal_pnt_cam[2];

    // hu->hd: distort image coordinates
    Scalar ru = std::sqrt(
        suriko::Sqr(cam_intrinsics_.PixelSizeMm[0] * (hu[0] - cam_intrinsics_.PrincipalPointPixels[0])) +
        suriko::Sqr(cam_intrinsics_.PixelSizeMm[1] * (hu[1] - cam_intrinsics_.PrincipalPointPixels[1])));
    
    // solve polynomial fun(rd)=k2*rd^5+k1*rd^3+rd-ru=0
    Scalar rd = ru; // TODO:

    Eigen::Matrix<Scalar, kPixPosComps, 1> hd;
    hd[0] = hu[0]; // TODO: not impl
    hd[1] = hu[1];
    
    if (proj_hist != nullptr)
    {
        proj_hist->hc = sal_pnt_cam;
    }
    return hd;
}

suriko::Point2 DavisonMonoSlam::ProjectCameraPoint(const suriko::Point3& pnt_camera) const
{
    suriko::Point2 result{};
    result.Mat() = ProjectCameraSalientPoint(pnt_camera.Mat(), nullptr);
    return result;
}

void DavisonMonoSlam::Deriv_hd_by_cam_state_and_sal_pnt(const SalPntInternal& sal_pnt,
    const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
    const EigenDynVec& derive_at_pnt,
    Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_cam_state,
    Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt) const
{
    // project salient point into current camera

    EstimVarsSalientPoint sal_pnt_vars;
    size_t off = sal_pnt.EstimVarsInd;
    LoadSalientPointDataFromArray(Span(derive_at_pnt).subspan(off, kSalientPointComps), &sal_pnt_vars);

    SalPntProjectionIntermidVars proj_hist{};
    Eigen::Matrix<Scalar, kPixPosComps, 1> hd = ProjectInternalSalientPoint(cam_state, sal_pnt_vars, &proj_hist);

    // calculate derivatives

    // how distorted pixels coordinates depend on undistorted pixels coordinates
    Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> hd_by_hu;
    Point2 corner_pix = sal_pnt.PixelCoordInLatestFrame;
    Deriv_hd_by_hu(corner_pix, &hd_by_hu);

    // A.34 how undistorted pixels coordinates hu=[uu,vu] depend on salient point (in camera) 3D meter coordinates [hcx,hcy,hcz] (A.23)
    Eigen::Matrix<Scalar, kPixPosComps, kEucl3> hu_by_hc;
    Deriv_hu_by_hc(proj_hist, &hu_by_hc);

    Deriv_hd_by_camera_state(sal_pnt_vars, cam_state, cam_orient_wfc, proj_hist, hd_by_hu, hu_by_hc, hd_by_cam_state);

    Deriv_hd_by_sal_pnt(sal_pnt_vars, cam_state, cam_orient_wfc, hd_by_hu, hu_by_hc, hd_by_sal_pnt);

    static bool debug_corner_coord_derivatives = false;
    if (debug_corner_coord_derivatives)
    {
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps> finite_diff_hd_by_xc;
        FiniteDiff_hd_by_camera_state(derive_at_pnt, sal_pnt_vars, kFiniteDiffEpsDebug, &finite_diff_hd_by_xc);

        Scalar diff1 = (finite_diff_hd_by_xc - *hd_by_cam_state).norm();

        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps> finite_diff_hd_by_y;
        FiniteDiff_hd_by_sal_pnt_state(cam_state, sal_pnt, derive_at_pnt, kFiniteDiffEpsDebug, &finite_diff_hd_by_y);

        Scalar diff2 = (finite_diff_hd_by_y - *hd_by_sal_pnt).norm();
        SRK_ASSERT(true);
    }
}

void DavisonMonoSlam::Deriv_Hrowblock_by_estim_vars(const SalPntInternal& sal_pnt,
    const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
    const EigenDynVec& derive_at_pnt,
    Eigen::Matrix<Scalar, kPixPosComps, Eigen::Dynamic>* Hrowblock_by_estim_vars) const
{
    Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps> hd_by_cam_state;
    Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps> hd_by_sal_pnt;
    Deriv_hd_by_cam_state_and_sal_pnt(sal_pnt, cam_state, cam_orient_wfc, derive_at_pnt, &hd_by_cam_state, &hd_by_sal_pnt);
    
    // by camera variables
    Hrowblock_by_estim_vars->middleCols<kCamStateComps>(0) = hd_by_cam_state;

    // by salient point variables
    // observed corner position (hd) depends only on the position of corresponding salient point (and not on any other salient point)
    size_t off = sal_pnt.EstimVarsInd;
    Hrowblock_by_estim_vars->middleCols<kSalientPointComps>(off) = hd_by_sal_pnt;
}

void DavisonMonoSlam::Deriv_H_by_estim_vars(const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
    const EigenDynVec& derive_at_pnt,
    EigenDynMat* H_by_estim_vars) const
{
    EigenDynMat& H = *H_by_estim_vars;

    size_t n = EstimatedVarsCount();
    size_t matched_corners = latest_frame_sal_pnts_.size();
    H.resize(kPixPosComps * matched_corners, n);
    H.setZero();

    //
    size_t obs_sal_pnt_ind = -1;
    for (SalPntId obs_sal_pnt_id : latest_frame_sal_pnts_)
    {
        ++obs_sal_pnt_ind;

        Eigen::Matrix<Scalar, kPixPosComps, Eigen::Dynamic> Hrowblock;
        Hrowblock.resize(Eigen::NoChange, n);
        Hrowblock.setZero();

        const SalPntInternal& sal_pnt = GetSalPnt(obs_sal_pnt_id);
        Deriv_Hrowblock_by_estim_vars(sal_pnt, cam_state, cam_orient_wfc, derive_at_pnt, &Hrowblock);

        H.middleRows<kPixPosComps>(obs_sal_pnt_ind*kPixPosComps) = Hrowblock;
    }
}

void DavisonMonoSlam::FiniteDiff_hd_by_camera_state(const EigenDynVec& derive_at_pnt, 
    const EstimVarsSalientPoint& sal_pnt,
    Scalar finite_diff_eps,
    Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_xc) const
{
    for (size_t var_ind = 0; var_ind < kCamStateComps; ++var_ind)
    {
        // copy cam_state
        Eigen::Matrix<Scalar, kCamStateComps, 1> cam_state = derive_at_pnt.topRows<kCamStateComps>();
        cam_state[var_ind] += finite_diff_eps;

        CameraPosState cam_state_right;
        LoadCameraPosDataFromArray(gsl::make_span<const Scalar>(cam_state.data(), kCamStateComps), &cam_state_right);

        Eigen::Matrix<Scalar, kPixPosComps, 1> hd_right = ProjectInternalSalientPoint(cam_state_right, sal_pnt, nullptr);
        
        //
        cam_state[var_ind] -= 2 * finite_diff_eps;

        CameraPosState cam_state_left;
        LoadCameraPosDataFromArray(gsl::make_span<const Scalar>(cam_state.data(), kCamStateComps), &cam_state_left);

        Eigen::Matrix<Scalar, kPixPosComps, 1> hd_left = ProjectInternalSalientPoint(cam_state_left, sal_pnt, nullptr);
        hd_by_xc->middleCols<1>(var_ind) = (hd_right - hd_left) / (2 * finite_diff_eps);
    }
}

void DavisonMonoSlam::FiniteDiff_hd_by_sal_pnt_state(const CameraPosState& cam_state, 
    const SalPntInternal& sal_pnt,
    const EigenDynVec& derive_at_pnt,
    Scalar finite_diff_eps,
    Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_y) const
{
    size_t off = sal_pnt.EstimVarsInd;
    for (size_t var_ind = 0; var_ind < kSalientPointComps; ++var_ind)
    {
        // copy cam_state
        Eigen::Matrix<Scalar, kSalientPointComps, 1> sal_pnt_state = derive_at_pnt.middleRows<kSalientPointComps>(off);
        sal_pnt_state[var_ind] += finite_diff_eps;

        EstimVarsSalientPoint sal_pnt_right;
        LoadSalientPointDataFromArray(gsl::make_span<const Scalar>(sal_pnt_state.data(), kSalientPointComps), &sal_pnt_right);

        Eigen::Matrix<Scalar, kPixPosComps, 1> hd_right = ProjectInternalSalientPoint(cam_state, sal_pnt_right, nullptr);
        
        //
        sal_pnt_state[var_ind] -= 2 * finite_diff_eps;

        EstimVarsSalientPoint sal_pnt_left;
        LoadSalientPointDataFromArray(gsl::make_span<const Scalar>(sal_pnt_state.data(), kSalientPointComps), &sal_pnt_left);

        Eigen::Matrix<Scalar, kPixPosComps, 1> hd_left = ProjectInternalSalientPoint(cam_state, sal_pnt_left, nullptr);
        hd_by_y->middleCols<1>(var_ind) = (hd_right - hd_left) / (2 * finite_diff_eps);
    }
}

size_t DavisonMonoSlam::SalientPointsCount() const
{
    return sal_pnts_.size();
}

size_t DavisonMonoSlam::EstimatedVarsCount() const
{
    return estim_vars_.size();
}

void DavisonMonoSlam::Deriv_cam_state_by_cam_state(Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* result) const
{
    Scalar dT = between_frames_period_;

    auto& m = *result;
    m.setIdentity();
    m.block<kEucl3, kEucl3>(0, kEucl3 + kQuat4) = Eigen::Matrix<Scalar, kEucl3, kEucl3>::Identity() * dT;

    // derivative of qk+1 with respect to qk

    Eigen::Matrix<Scalar, kAngVelocComps, 1> w = EstimVarsCamAngularVelocity();
    Eigen::Matrix<Scalar, kAngVelocComps, 1> delta_orient = w * dT;

    Eigen::Matrix<Scalar, kQuat4, 1> q1;
    QuatFromAxisAngle(delta_orient, &q1);

    // formula A.12
    Eigen::Matrix<Scalar, kQuat4, kQuat4> q3_by_q2;
    q3_by_q2 <<
        q1[0], -q1[1], -q1[2], -q1[3],
        q1[1],  q1[0],  q1[3], -q1[2],
        q1[2], -q1[3],  q1[0],  q1[1],
        q1[3],  q1[2], -q1[1],  q1[0];
    m.block<kQuat4, kQuat4>(kEucl3, kEucl3) = q3_by_q2;

    //
    Eigen::Matrix<Scalar, kQuat4, kAngVelocComps> q3_by_w;
    Deriv_q3_by_w(dT, &q3_by_w);
    m.block<kQuat4, kAngVelocComps>(kEucl3, kEucl3 + kQuat4 +  kVelocComps) = q3_by_w;
    SRK_ASSERT(m.allFinite());
}

void DavisonMonoSlam::FiniteDiff_cam_state_by_cam_state(gsl::span<const Scalar> cam_state, Scalar finite_diff_eps,
    Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* result) const
{
    for (size_t var_ind = 0; var_ind < kCamStateComps; ++var_ind)
    {
        Eigen::Matrix<Scalar, kCamStateComps, 1> mut_state = Eigen::Map<const Eigen::Matrix<Scalar, kCamStateComps, 1>>(cam_state.data()); // copy state
        mut_state[var_ind] += finite_diff_eps;

        CameraPosState cam_right;
        auto cam_right_array = Span(mut_state);
        LoadCameraPosDataFromArray(cam_right_array, &cam_right);

        // note, we do  not normalize quaternion after applying motion model because the finite difference increment which
        // has been applied to the state vector, breaks unity of quaternions. Then normalization of a quaternion propagates
        // modifications to other it's components. This diverges the finite difference result from close form of a derivative.
        bool norm_quat = false;

        Eigen::Matrix<Scalar, kCamStateComps, 1> value_right;
        PredictCameraMotionByKinematicModel(cam_right_array, Span(value_right), nullptr, norm_quat);

        //
        mut_state[var_ind] -= 2 * finite_diff_eps;

        CameraPosState cam_left;
        auto cam_left_array = Span(mut_state);
        LoadCameraPosDataFromArray(cam_left_array, &cam_left);

        Eigen::Matrix<Scalar, kCamStateComps, 1> value_left;
        PredictCameraMotionByKinematicModel(cam_left_array, Span(value_left), nullptr, norm_quat);

        Eigen::Matrix<Scalar, kCamStateComps, 1> col_diff = value_right -value_left;
        Eigen::Matrix<Scalar, kCamStateComps, 1> col = col_diff / (2 * finite_diff_eps);
        result->middleCols<1>(var_ind) = col;
    }
}

void DavisonMonoSlam::Deriv_cam_state_by_input_noise(Eigen::Matrix<Scalar, kCamStateComps, kInputNoiseComps>* result) const
{
    Scalar dT = between_frames_period_;

    auto& m = *result;
    m.setZero();
    const auto id3x3 = Eigen::Matrix<Scalar, kEucl3, kEucl3>::Identity();
    m.block<kEucl3, kEucl3>(0, 0) = dT * id3x3;
    m.block<kEucl3, kEucl3>(kEucl3 + kQuat4, 0) = id3x3;
    m.block<kEucl3, kEucl3>(kEucl3 + kQuat4 + kVelocComps, kEucl3) = id3x3;

    // derivative of q3 with respect to capital omega is the same as the little omega
    // because in A.9 small omega and capital omega are interchangable
    Eigen::Matrix<Scalar, kQuat4, kAngVelocComps> q3_by_cap_omega;
    Deriv_q3_by_w(dT, &q3_by_cap_omega);

    m.block<kQuat4, kAngVelocComps>(kEucl3, kEucl3) = q3_by_cap_omega;
    SRK_ASSERT(m.allFinite());
}

void DavisonMonoSlam::FiniteDiff_cam_state_by_input_noise(Scalar finite_diff_eps,
    Eigen::Matrix<Scalar, kCamStateComps, kInputNoiseComps>* result) const
{
    typedef Eigen::Matrix<Scalar, kInputNoiseComps, 1> NoiseVec;
    const NoiseVec noise = NoiseVec::Zero(); // calculate finite differences at zero noise

    gsl::span<const Scalar> cam_state_array = gsl::make_span<const Scalar>(estim_vars_.data(), kCamStateComps);

    for (size_t var_ind = 0; var_ind < kInputNoiseComps; ++var_ind)
    {
        // copy state
        NoiseVec mut_state = Eigen::Map<const NoiseVec>(noise.data());
        mut_state[var_ind] += finite_diff_eps;

        bool norm_quat = false;
        Eigen::Matrix<Scalar, kCamStateComps, 1> value_right;
        PredictCameraMotionByKinematicModel(cam_state_array, Span(value_right), &mut_state, norm_quat);

        //
        mut_state[var_ind] -= 2 * finite_diff_eps;

        Eigen::Matrix<Scalar, kCamStateComps, 1> value_left;
        PredictCameraMotionByKinematicModel(cam_state_array, Span(value_left), &mut_state, norm_quat);

        Eigen::Matrix<Scalar, kCamStateComps, 1> col_diff = value_right - value_left;
        Eigen::Matrix<Scalar, kCamStateComps, 1> col = col_diff / (2 * finite_diff_eps);
        result->middleCols<1>(var_ind) = col;
    }
}

void DavisonMonoSlam::Deriv_q3_by_w(Scalar deltaT, Eigen::Matrix<Scalar, kQuat4, kEucl3>* result) const
{
    auto& m = *result;

    Eigen::Matrix<Scalar, kAngVelocComps, 1> w = EstimVarsCamAngularVelocity();
    Scalar w_norm = w.norm();
    if (IsClose(0, w_norm))
    {
        m.setZero();
        return;
    }

    Eigen::Matrix<Scalar, kQuat4, 1> q2 = EstimVarsCamQuat();

    // formula A.14
    Eigen::Matrix<Scalar, kQuat4, kQuat4> q3_by_q1;
    q3_by_q1 <<
        q2[0], -q2[1], -q2[2], -q2[3],
        q2[1], q2[0], -q2[3], q2[2],
        q2[2], q2[3], q2[0], -q2[1],
        q2[3], -q2[2], q2[1], q2[0];

    Eigen::Matrix<Scalar, kQuat4, kAngVelocComps> q1_by_wk;

    // top row
    for (size_t i = 0; i<kAngVelocComps; ++i)
    {
        q1_by_wk(0, i) = -0.5*deltaT*w[i] / w_norm * std::sin(0.5*w_norm*deltaT);
    }

    Scalar c = std::cos(0.5*w_norm*deltaT);
    Scalar s = std::sin(0.5*w_norm*deltaT);

    // next 3 rows
    for (size_t i = 0; i<kAngVelocComps; ++i)
        for (size_t j = 0; j < kAngVelocComps; ++j)
        {
            if (i == j) // on 'diagonal'
            {
                Scalar rat = w[i] / w_norm;
                q1_by_wk(1 + i, i) = 0.5*deltaT*rat*rat*c + (1 / w_norm)*s*(1 - rat * rat);
            }
            else // off 'diagonal'
            {
                q1_by_wk(1 + i, j) = w[i] * w[j] / (w_norm*w_norm)*(0.5*deltaT*c - (1 / w_norm)*s);
            }
        }
    // A.13
    m = q3_by_q1 * q1_by_wk;
}

Scalar DavisonMonoSlam::EstimVarsSalientPoint::GetDist() const
{
    SRK_ASSERT(!IsClose(0, InverseDistRho));
    return 1 / InverseDistRho;
}

void DavisonMonoSlam::LoadSalientPointDataFromArray(gsl::span<const Scalar> src, EstimVarsSalientPoint* result) const
{
    DependsOnSalientPointPackOrder();
    result->FirstCamPosW[0] = src[0];
    result->FirstCamPosW[1] = src[1];
    result->FirstCamPosW[2] = src[2];
    result->AzimuthThetaW = src[3];
    result->ElevationPhiW = src[4];
    result->InverseDistRho = src[5];
}

size_t DavisonMonoSlam::SalientPointOffset(size_t sal_pnt_ind) const
{
    DependsOnOverallPackOrder();
    return kCamStateComps + sal_pnt_ind * kSalientPointComps;
}

SalPntInternal& DavisonMonoSlam::GetSalPnt(SalPntId id)
{
    return *id.sal_pnt_internal_;
}

const SalPntInternal& DavisonMonoSlam::GetSalPnt(SalPntId id) const
{
    return *id.sal_pnt_internal_;
}

void DavisonMonoSlam::GetCameraPredictedPosState(CameraPosState* result) const
{
    const auto& src_estim_vars = predicted_estim_vars_;
    LoadCameraPosDataFromArray(gsl::make_span<const Scalar>(src_estim_vars.data(), kCamStateComps), result);
}

void DavisonMonoSlam::LoadCameraPosDataFromArray(gsl::span<const Scalar> src, CameraPosState* result) const
{
    DependsOnCameraPosPackOrder();
    CameraPosState& c = *result;
    c.PosW[0] = src[0];
    c.PosW[1] = src[1];
    c.PosW[2] = src[2];
    c.OrientationWfc[0] = src[3];
    c.OrientationWfc[1] = src[4];
    c.OrientationWfc[2] = src[5];
    c.OrientationWfc[3] = src[6];
    c.VelocityW[0] = src[7];
    c.VelocityW[1] = src[8];
    c.VelocityW[2] = src[9];
    c.AngularVelocityC[0] = src[10];
    c.AngularVelocityC[1] = src[11];
    c.AngularVelocityC[2] = src[12];
}

void DavisonMonoSlam::GetCameraPredictedPosAndOrientationWithUncertainty(
    Eigen::Matrix<Scalar, kEucl3, 1>* cam_pos,
    Eigen::Matrix<Scalar, kEucl3, kEucl3>* cam_pos_uncert,
    Eigen::Matrix<Scalar, kQuat4, 1>* cam_orient_quat) const
{
    const auto& src_estim_vars = predicted_estim_vars_;
    const auto& src_estim_vars_covar = predicted_estim_vars_covar_;

    DependsOnCameraPosPackOrder();

    // mean of camera position
    auto& m = *cam_pos;
    m[0] = src_estim_vars[0];
    m[1] = src_estim_vars[1];
    m[2] = src_estim_vars[2];
    SRK_ASSERT(m.allFinite());

    // uncertainty of camera position
    const auto& orig_uncert = src_estim_vars_covar.block<kEucl3, kEucl3>(0, 0);

    auto& unc = *cam_pos_uncert;
    unc = orig_uncert;
    SRK_ASSERT(unc.allFinite());

    auto& q = *cam_orient_quat;
    q[0] = src_estim_vars[3];
    q[1] = src_estim_vars[4];
    q[2] = src_estim_vars[5];
    q[3] = src_estim_vars[6];
    SRK_ASSERT(q.allFinite());
}

void DavisonMonoSlam::GetCameraPredictedUncertainty(Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps>* cam_covar) const
{
    *cam_covar = predicted_estim_vars_covar_.topLeftCorner<kCamStateComps, kCamStateComps>();
}

void DavisonMonoSlam::PropagateSalPntPosUncertainty(const EstimVarsSalientPoint& sal_pnt,
    const Eigen::Matrix<Scalar, kSalientPointComps, kSalientPointComps>& sal_pnt_covar,
    Eigen::Matrix<Scalar, kEucl3, kEucl3>* sal_pnt_pos_uncert) const
{
    // propagate camera pos (Xc,Yc,Zc) and (theta-azimuth, phi-elevation,rho) into salient point pos (X,Y,Z)
    Eigen::Matrix<Scalar, kEucl3, kSalientPointComps> jacob;
    jacob.setIdentity();

    Scalar cos_theta = std::cos(sal_pnt.AzimuthThetaW);
    Scalar sin_theta = std::sin(sal_pnt.AzimuthThetaW);
    Scalar cos_phi = std::cos(sal_pnt.ElevationPhiW);
    Scalar sin_phi = std::sin(sal_pnt.ElevationPhiW);

    Scalar dist = 1 / sal_pnt.InverseDistRho;

    // deriv of (Xc,Yc,Zc) by theta-azimuth
    jacob(0, kEucl3) = dist * cos_phi * cos_theta;
    jacob(1, kEucl3) = 0;
    jacob(2, kEucl3) = -dist * cos_phi * sin_theta;

    // deriv of (Xc,Yc,Zc) by phi-elevation
    jacob(0, kEucl3 + 1) = -dist * sin_phi * sin_theta;
    jacob(1, kEucl3 + 1) = -dist * cos_phi;
    jacob(2, kEucl3 + 1) = -dist * sin_phi * cos_theta;

    Scalar dist2 = 1 / suriko::Sqr(sal_pnt.InverseDistRho);

    // deriv of (Xc,Yc,Zc) by rho
    jacob(0, kEucl3 + 2) = -dist2 * cos_phi * sin_theta;
    jacob(1, kEucl3 + 2) = dist2 * sin_phi;
    jacob(2, kEucl3 + 2) = -dist2 * cos_phi * cos_theta;

    // original uncertainty
    const auto& orig_uncert = sal_pnt_covar;

    // propagate uncertainty
    auto& unc = *sal_pnt_pos_uncert;
    unc = jacob * orig_uncert * jacob.transpose();
    SRK_ASSERT(unc.allFinite());
}

void DavisonMonoSlam::LoadSalientPointPredictedPosWithUncertainty(
    const EigenDynVec& src_estim_vars,
    const EigenDynMat& src_estim_vars_covar,
    size_t salient_pnt_ind,
    Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
    Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const
{
    EstimVarsSalientPoint sal_pnt;
    size_t sal_pnt_off = SalientPointOffset(salient_pnt_ind);
    LoadSalientPointDataFromArray(Span(src_estim_vars).subspan(sal_pnt_off, kSalientPointComps), &sal_pnt);

    Scalar hx, hy, hz;
    CameraCoordinatesEuclidFromPolar(sal_pnt.AzimuthThetaW, sal_pnt.ElevationPhiW, sal_pnt.GetDist(), &hx, &hy, &hz);

    // salient point in world coordinates = camera position + position of the salient point in the camera
    auto& m = *pos_mean;
    m[0] = sal_pnt.FirstCamPosW[0] + hx;
    m[1] = sal_pnt.FirstCamPosW[1] + hy;
    m[2] = sal_pnt.FirstCamPosW[2] + hz;

    if (pos_uncert == nullptr)
        return;

    Eigen::Matrix<Scalar, kSalientPointComps, kSalientPointComps> orig_uncert = 
        src_estim_vars_covar.block<kSalientPointComps, kSalientPointComps>(sal_pnt_off, sal_pnt_off);

    // use first order approximation 
    PropagateSalPntPosUncertainty(sal_pnt, orig_uncert, pos_uncert);

    static bool simulate_propagation = false;
    if (simulate_propagation)
    {
        auto propag_fun = [](const auto& in_mat, auto* out_mat) -> void
        {
            Scalar theta2 = in_mat[3];
            Scalar phi2 = in_mat[4];
            Scalar rho2 = in_mat[5];
            Scalar dist2 = 1 / rho2;
            Scalar hx2, hy2, hz2;
            CameraCoordinatesEuclidFromPolar(theta2, phi2, dist2, &hx2, &hy2, &hz2);

            auto& y = *out_mat;
            y[0] = in_mat[0] + hx2;
            y[1] = in_mat[1] + hy2;
            y[2] = in_mat[2] + hz2;
        };

        Eigen::Map<const Eigen::Matrix<Scalar, kSalientPointComps, 1>> y_mean_mat(&src_estim_vars[sal_pnt_off]);
        Eigen::Matrix<Scalar, kSalientPointComps, 1> y_mean = y_mean_mat;
        Eigen::Matrix<Scalar, kSalientPointComps, kSalientPointComps> y_uncert = orig_uncert.eval();

        static size_t gen_samples_count = 100000;
        static std::mt19937 gen{ 811 };
        Eigen::Matrix<Scalar, kEucl3, kEucl3> simul_uncert;
        PropagateUncertaintyUsingSimulation(y_mean, y_uncert, propag_fun, gen_samples_count, &gen, &simul_uncert);
    }
}

void DavisonMonoSlam::GetSalientPointPredictedPosWithUncertainty(size_t salient_pnt_ind,
    Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean,
    Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const
{
    const auto& src_estim_vars = predicted_estim_vars_;
    const auto& src_estim_vars_covar = predicted_estim_vars_covar_;
    LoadSalientPointPredictedPosWithUncertainty(src_estim_vars, src_estim_vars_covar, salient_pnt_ind, pos_mean, pos_uncert);
}

gsl::span<Scalar> DavisonMonoSlam::EstimVarsCamPosW()
{
    return gsl::make_span<Scalar>(estim_vars_.data(), kEucl3);
}

Eigen::Matrix<Scalar, kQuat4, 1> DavisonMonoSlam::EstimVarsCamQuat() const
{
    DependsOnCameraPosPackOrder();
    return estim_vars_.middleRows<kQuat4>(kEucl3);
}

Eigen::Matrix<Scalar, kAngVelocComps, 1> DavisonMonoSlam::EstimVarsCamAngularVelocity() const
{
    DependsOnCameraPosPackOrder();
    return estim_vars_.middleRows< kAngVelocComps>(kEucl3 + kQuat4 + kVelocComps);
}

void DavisonMonoSlam::SetCornersMatcher(std::unique_ptr<CornersMatcherBase> corners_matcher)
{
    corners_matcher_.swap(corners_matcher);
}

CornersMatcherBase& DavisonMonoSlam::CornersMatcher()
{
    return *corners_matcher_.get();
}

void DavisonMonoSlam::LogReprojError() const
{
    Scalar err = CurrentFrameReprojError();
    VLOG(4) << "ReprojError=" << err;
}

Scalar DavisonMonoSlam::CurrentFrameReprojError() const
{
    // specify state the reprojection error is based on
    const auto& src_estim_vars = predicted_estim_vars_;

    CameraPosState cam_state;
    LoadCameraPosDataFromArray(Span(src_estim_vars, kCamStateComps), &cam_state);

    Scalar err_sum = 0;
    for (SalPntId obs_sal_pnt_id : latest_frame_sal_pnts_)
    {
        const SalPntInternal& sal_pnt = GetSalPnt(obs_sal_pnt_id);

        EstimVarsSalientPoint sal_pnt_vars;
        LoadSalientPointDataFromArray(Span(src_estim_vars).subspan(sal_pnt.EstimVarsInd, kSalientPointComps), &sal_pnt_vars);

        Eigen::Matrix<Scalar, kPixPosComps, 1> pix = ProjectInternalSalientPoint(cam_state, sal_pnt_vars, nullptr);

        suriko::Point2 corner_pix = sal_pnt.PixelCoordInLatestFrame;
        Scalar err_one = (corner_pix.Mat() - pix).norm();
        err_sum += err_one;
    }

    return err_sum;
}
void DavisonMonoSlam::FixSymmetricMat(EigenDynMat* sym_mat) const
{
    auto& m = *sym_mat;
    m = (m + m.transpose()).eval() / 2;
}

void DavisonMonoSlam::SetDebugPath(DebugPathEnum debug_path)
{
    s_debug_path_ = debug_path;
}
bool DavisonMonoSlam::DebugPath(DebugPathEnum debug_path)
{
    return (s_debug_path_ & debug_path) != DebugPathEnum::DebugNone;
}
}
