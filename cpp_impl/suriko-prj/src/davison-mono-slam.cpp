#include <random>
#include "suriko/davison-mono-slam.h"
#include <glog/logging.h>
#include "suriko/approx-alg.h"
#include "suriko/quat.h"
#include "suriko/eigen-helpers.hpp"

namespace suriko
{

void DavisonMonoSlam::CameraCoordinatesPolarFromEuclid(Scalar hx, Scalar hy, Scalar hz, Scalar* theta, Scalar* phi, Scalar* dist)
{
    // azimuth=theta, formula A.60
    *theta = std::atan2(hx, hz);
    
    // elevation=phi, formula A.61
    *phi = std::atan2(-hy, std::sqrt(hx*hx + hz * hz));

    *dist = std::sqrt(hx*hx + hy * hy + hz * hz);
}

void DavisonMonoSlam::CameraCoordinatesEuclidFromPolar(Scalar theta, Scalar phi, Scalar dist, Scalar* hx, Scalar* hy, Scalar* hz)
{
    // polar -> euclidean position of salient point in world frame with center in camera position
    // theta=azimuth
    Scalar cos_theta = std::cos(theta);
    Scalar sin_theta = std::sin(theta);

    // phi=elevation
    Scalar cos_ef = std::cos(phi);
    Scalar sin_ef = std::sin(phi);

    *hx = dist * cos_ef * sin_theta;
    *hy = -dist * sin_ef;
    *hz = dist * cos_ef * cos_theta;
}

void DavisonMonoSlam::CameraCoordinatesEuclidUnityDirFromPolarAngles(Scalar theta, Scalar phi, Scalar* hx, Scalar* hy, Scalar* hz)
{
    // polar -> euclidean position of salient point in world frame with center in camera position
    // theta=azimuth
    Scalar cos_theta = std::cos(theta);
    Scalar sin_theta = std::sin(theta);

    // phi=elevation
    Scalar cos_ef = std::cos(phi);
    Scalar sin_ef = std::sin(phi);

    *hx = cos_ef * sin_theta;
    *hy = -sin_ef;
    *hz = cos_ef * cos_theta;
}

void DavisonMonoSlam::ResetState(const SE3Transform& cam_pos_cfw, const std::vector<SalientPointFragment>& salient_feats,
    Scalar estim_var_init_std)
{
    auto cam_pos_wfc = SE3Inv(cam_pos_cfw);

    // state vector
    size_t n = kCamStateComps + salient_feats.size() * kSalientPointComps;

    estim_vars_covar_.setZero(n, n);
    filter_gain_.setZero(n, kPixPosComps * salient_feats.size()); // TODO: should be [13+6N,2m]

    //
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

    for (size_t i=0; i<salient_feats.size(); ++i)
    {
        const suriko::Point3& p = salient_feats[i].Coord.value();
        gsl::span<Scalar> sal_pnt = state_span.subspan(kCamStateComps + i * kSalientPointComps, kSalientPointComps);
        
        // camera position, when the salient point is first seen
        DependsOnSalientPointPackOrder();
        sal_pnt[0] = cam_pos_wfc.T[0];
        sal_pnt[1] = cam_pos_wfc.T[1];
        sal_pnt[2] = cam_pos_wfc.T[2];

        // position of the salient point in camera coordinates
        Eigen::Matrix<Scalar, 3, 1> sal_pnt_in_camera = p.Mat() - cam_pos_wfc.T;
        Scalar azimuth;
        Scalar elevation;
        Scalar dist;
        CameraCoordinatesPolarFromEuclid(sal_pnt_in_camera[0], sal_pnt_in_camera[1], sal_pnt_in_camera[2], &azimuth, &elevation, &dist);
        Scalar inv_dist_rho = 1 / dist;
        sal_pnt[3] = azimuth;
        sal_pnt[4] = elevation;
        sal_pnt[5] = inv_dist_rho;
    }

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
    
    for (size_t i = 0; i < salient_feats.size(); ++i)
    {
        size_t off = kCamStateComps + i * kSalientPointComps;
        estim_vars_covar_(off+0, off+0) = estim_var_init_variance;
        estim_vars_covar_(off+1, off+1) = estim_var_init_variance;
        estim_vars_covar_(off+2, off+2) = estim_var_init_variance;
        estim_vars_covar_(off+3, off+3) = estim_var_init_variance;
        estim_vars_covar_(off+4, off+4) = estim_var_init_variance;
        estim_vars_covar_(off+5, off+5) = estim_var_init_variance;
    }
    SRK_ASSERT(estim_vars_covar_.allFinite());

    //
    Scalar input_noise_std_variance = suriko::Sqr(input_noise_std_);
    input_noise_covar_.setZero();
    input_noise_covar_(0, 0) = input_noise_std_variance;
    input_noise_covar_(1, 1) = input_noise_std_variance;
    input_noise_covar_(2, 2) = input_noise_std_variance;
    input_noise_covar_(3, 3) = input_noise_std_variance;
    input_noise_covar_(4, 4) = input_noise_std_variance;
    input_noise_covar_(5, 5) = input_noise_std_variance;
}

void DavisonMonoSlam::FillRk2x2(Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* Rk) const
{
    Scalar measurm_noise_variance = suriko::Sqr(static_cast<Scalar>(measurm_noise_std_));
    *Rk << measurm_noise_variance, 0, 0, measurm_noise_variance;
}

void DavisonMonoSlam::FillRk(size_t matched_corners, EigenDynMat* Rk) const
{
    Rk->setZero(matched_corners * kPixPosComps, matched_corners * kPixPosComps);

    Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> noise_one;
    FillRk2x2(&noise_one);
    
    for (size_t i = 0; i < matched_corners; ++i)
    {
        size_t off = i * kPixPosComps;
        Rk->block<kPixPosComps, kPixPosComps>(off, off) = noise_one;
    }
}

void DavisonMonoSlam::PredictCameraState(gsl::span<const Scalar> cam_state, gsl::span<Scalar> new_cam_state) const
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

    // camera orientation
    Eigen::Matrix<Scalar, kVelocComps, 1> cam_orient_delta = cam_ang_vel * dT;

    Eigen::Matrix<Scalar, kQuat4, 1> cam_orient_delta_quat{};
    QuatFromAxisAngle(cam_orient_delta, &cam_orient_delta_quat);

    Eigen::Matrix<Scalar, kQuat4, 1> new_cam_orient_quat_tmp;
    QuatMult(cam_orient_quat, cam_orient_delta_quat, &new_cam_orient_quat_tmp);

    Scalar qorig_len = cam_orient_quat.norm();

    // normalize quaternion (formula A.141)
    Scalar q_len = new_cam_orient_quat_tmp.norm();
    new_cam_orient_quat_tmp /= q_len;
    Scalar q_len2 = new_cam_orient_quat_tmp.norm();

    new_cam_orient_quat = new_cam_orient_quat_tmp;

    // camera velocity is unchanged
    new_cam_vel = cam_vel;
    // camera angular velocity is unchanged
    new_cam_ang_vel = cam_ang_vel;
}

void DavisonMonoSlam::PredictEstimVars(size_t frame_ind, EigenDynVec* predicted_estim_vars, EigenDynMat* predicted_estim_vars_covar) const
{
    // estimated vars
    std::array<Scalar, kCamStateComps> new_cam{};
    PredictCameraState(gsl::make_span<const Scalar>(estim_vars_.data(), kCamStateComps), new_cam);

    //
    Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps> Fk;
    Deriv_cam_state_by_cam_state(&Fk);

    Eigen::Matrix<Scalar, kCamStateComps, kInputNoiseComps> Gk;
    Deriv_cam_state_by_input_noise(&Gk);

    // Pvv = F*Pvv*Ft+G*Q*Gt
    Eigen::Matrix<Scalar, kCamStateComps, kCamStateComps> Pvv_new =
        Fk * estim_vars_covar_.topLeftCorner<kCamStateComps, kCamStateComps>() * Fk.transpose() +
        Gk * input_noise_covar_ * Gk.transpose();
    
    // Pvm = F*Pvm
    size_t sal_pnts_vars_count = SalientPointsCount() * kSalientPointComps;
    Eigen::Matrix<Scalar, kCamStateComps, Eigen::Dynamic> Pvm_new = 
        Fk * estim_vars_covar_.topRightCorner(kCamStateComps, sal_pnts_vars_count);

    // update x
    *predicted_estim_vars = estim_vars_;
    predicted_estim_vars->topRows<kCamStateComps>() = Eigen::Map<const Eigen::Matrix<Scalar, kCamStateComps, 1>>(new_cam.data(), kCamStateComps);

    // update P
    *predicted_estim_vars_covar = estim_vars_covar_;
    predicted_estim_vars_covar->topLeftCorner<kCamStateComps, kCamStateComps>() = Pvv_new;
    predicted_estim_vars_covar->topRightCorner(kCamStateComps, sal_pnts_vars_count) = Pvm_new;
    predicted_estim_vars_covar->bottomLeftCorner(sal_pnts_vars_count, kCamStateComps) = Pvm_new.transpose();
}

void DavisonMonoSlam::PredictEstimVarsHelper()
{
    // make predictions
    PredictEstimVars(0, &predicted_estim_vars_, &predicted_estim_vars_covar_);
}

void DavisonMonoSlam::ProcessFrame(size_t frame_ind, bool suppress_observations, int kalman_update_impl)
{
    this->corners_matcher_->DetectAndMatchCorners(frame_ind, &track_rep_);

    // find the number of matched observations
    std::vector<size_t> matched_track_ids;
    track_rep_.IteratePointsMarker();
    for (const CornerTrack& track : track_rep_.CornerTracks)
    {
        std::optional<CornerData> corner = track.GetCornerData(frame_ind);
        if (corner.has_value())
        {
            matched_track_ids.push_back(track.TrackId);
        }
    }
    if (suppress_observations)
        matched_track_ids.clear();

    static int impl = kalman_update_impl;
    if (impl == 1)
        ProcessFrame_StackedObservationsPerUpdate(frame_ind, matched_track_ids);
    else if (impl == 2)
        ProcessFrame_OneObservationPerUpdate(frame_ind, matched_track_ids);
    else if (impl == 3)
        ProcessFrame_OneComponentOfOneObservationPerUpdate(frame_ind, suppress_observations);

    // make predictions
    PredictEstimVars(frame_ind, &predicted_estim_vars_, &predicted_estim_vars_covar_);

    if (VLOG_IS_ON(4))
    {
        Eigen::Matrix<Scalar, kEucl3, kEucl3> cam_pos_cov = estim_vars_covar_.block<kEucl3, kEucl3>(0, 0);
        EigenDynVec x_cov_diag = cam_pos_cov.diagonal();
        Scalar max_variance = x_cov_diag.maxCoeff();

        // uncertainty ellipsoid
        Eigen::Matrix<Scalar, 3, 3> A;
        Eigen::Matrix<Scalar, 3, 1> b;
        Scalar c;
        Eigen::Matrix<Scalar, kEucl3, 1> cam_pos = estim_vars_.topRows<kEucl3>();
        ExtractEllipsoidFromUncertaintyMat(cam_pos, cam_pos_cov, debug_ellipsoid_cut_thr_, &A, &b, &c);

        Eigen::Matrix<Scalar, 3, 1> center;
        Eigen::Matrix<Scalar, 3, 1> semi_axes;
        Eigen::Matrix<Scalar, 3, 3> rot_mat_world_from_ellipse;
        bool op = GetRotatedEllipsoid(A, b, c, &center, &semi_axes, &rot_mat_world_from_ellipse);
        SRK_ASSERT(op);

        VLOG(4) << "cam_pos=[" << cam_pos[0] << "," << cam_pos[1] << "," << cam_pos[2] << "]"
            << " cam_max_variance=" << max_variance
            << " Adiag=" << A(0, 0) << " " << A(1, 1) << " " << A(2, 2)
            << " b=" << b[0] << " " << b[1] << " " << b[2]
            << " c=" << c
            << " semi_axe=" << semi_axes[0] << " " << semi_axes[1] << " " << semi_axes[2];
    }
}

void DavisonMonoSlam::ProcessFrame_StackedObservationsPerUpdate(size_t frame_ind, const std::vector<size_t>& matched_track_ids)
{
    static bool tmp1 = false;
    if (tmp1)
    {
        OnEstimVarsChanged(frame_ind);
    }

    CameraPosState cam_state;
    GetCameraPosState(&cam_state);

    Eigen::Matrix<Scalar, kEucl3, kEucl3> cam_orient_wfc;
    RotMatFromQuat(gsl::make_span<const Scalar>(cam_state.OrientationWfc.data(), kQuat4), &cam_orient_wfc);

    size_t matched_corners = matched_track_ids.size();
    if (matched_corners > 0)
    {
        //
        EigenDynMat Hk; // [2m,13+6n]
        Deriv_H_by_estim_vars(frame_ind, cam_state, cam_orient_wfc, matched_track_ids, &Hk);

        // evaluate filter gain
        auto& Rk = measurm_noise_covar_;
        FillRk(matched_corners, &Rk);

        const auto& Pprev = predicted_estim_vars_covar_;
        auto innov_var = Hk * Pprev * Hk.transpose() + Rk;
        EigenDynMat innov_variance_inv = innov_var.inverse();
        EigenDynMat Knew = Pprev * Hk.transpose() * innov_variance_inv;
        filter_gain_ = Knew;

        //
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> zk;
        zk.resize(matched_corners * kPixPosComps, 1);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> projected_sal_pnts;
        projected_sal_pnts.resize(matched_corners * kPixPosComps, 1);

        for (size_t obs_sal_pnt_ind = 0; obs_sal_pnt_ind < matched_track_ids.size(); ++obs_sal_pnt_ind)
        {
            size_t track_id = matched_track_ids[obs_sal_pnt_ind];
            const CornerTrack& track = track_rep_.GetPointTrackById(track_id);
            std::optional<CornerData> corner = track.GetCornerData(frame_ind);
            Point2 corner_pix = corner.value().PixelCoord;
            zk.middleRows<kPixPosComps>(obs_sal_pnt_ind * kPixPosComps) = corner_pix.Mat();

            // project salient point into current camera

            SalientPointInternal sal_pnt;
            GetSalientPointData(obs_sal_pnt_ind, &sal_pnt);

            Eigen::Matrix<Scalar, kPixPosComps, 1> hd = ProjectInternalSalientPoint(cam_state, sal_pnt, nullptr);
            projected_sal_pnts.middleRows<kPixPosComps>(obs_sal_pnt_ind * kPixPosComps) = hd;

            Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps> hd_by_cam_state;
            Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps> hd_by_sal_pnt;
            Deriv_hd_by_cam_state_and_sal_pnt(obs_sal_pnt_ind, frame_ind, cam_state, cam_orient_wfc, matched_track_ids, &hd_by_cam_state, &hd_by_sal_pnt);

            int z = 9;
        }
        Scalar estim_change = (zk - projected_sal_pnts).norm();
        
        // update estimated variables
        EigenDynVec estim_vars_delta = filter_gain_ * (zk - projected_sal_pnts);
        Scalar estim_vars_delta_norm = estim_vars_delta.norm();
        EigenDynVec estim_vars_new = predicted_estim_vars_ + estim_vars_delta;
        Scalar diff_vars = (estim_vars_new - estim_vars_).norm();

        // updae covariance matrix
        size_t n = EstimatedVarsCount();
        auto ident = EigenDynMat::Identity(n, n);
        EigenDynMat estim_vars_covar_new = (ident - filter_gain_ * Hk) * Pprev;
        Scalar diff_cov = (estim_vars_covar_new - estim_vars_covar_).norm();
        VLOG(4) << "diff_vars=" << diff_vars << " diff_cov=" << diff_cov;

        // gt
        SE3Transform cam_orient_cfw_gt = gt_cam_orient_world_to_f_(frame_ind);
        SE3Transform cam_orient_wfc_gt = SE3Inv(cam_orient_cfw_gt);

        Eigen::Matrix<Scalar, kQuat4, 1> cam_orient_wfc_quat;
        QuatFromRotationMatNoRChecks(cam_orient_wfc_gt.R, gsl::make_span<Scalar>(cam_orient_wfc_quat.data(), kQuat4));

        // print norm of delta with gt (pos,orient)
        CameraPosState cam_state_new;
        LoadCameraPosDataFromArray(gsl::make_span<const Scalar>(estim_vars_new.data(), kCamStateComps), &cam_state_new);
        Scalar d1 = (cam_orient_wfc_gt.T - cam_state_new.PosW).norm();
        Scalar d2 = (cam_orient_wfc_quat - cam_state_new.OrientationWfc).norm();
        Scalar diff_gt = d1 + d2;
        VLOG(4) << "diff_gt=" << diff_gt << " zk-obs=" << estim_change;

        //
        estim_vars_ = estim_vars_new;
        estim_vars_covar_ = estim_vars_covar_new;
    }
    else
    {
        // we have no observations => current state <- prediction
        std::swap(estim_vars_, predicted_estim_vars_);
        std::swap(estim_vars_covar_, predicted_estim_vars_covar_);
    }

    OnEstimVarsChanged(frame_ind);
}

void DavisonMonoSlam::ProcessFrame_OneObservationPerUpdate(size_t frame_ind, const std::vector<size_t>& matched_track_ids)
{
    CameraPosState cam_state;
    GetCameraPosState(&cam_state);

    Eigen::Matrix<Scalar, kEucl3, kEucl3> cam_orient_wfc;
    RotMatFromQuat(gsl::make_span<const Scalar>(cam_state.OrientationWfc.data(), kQuat4), &cam_orient_wfc);

    Scalar diff_vars_total = 0;
    Scalar diff_cov_total = 0;

    size_t matched_corners = matched_track_ids.size();
    if (matched_corners > 0)
    {
        Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> Rk;
        FillRk2x2(&Rk);

        for (size_t obs_sal_pnt_ind = 0; obs_sal_pnt_ind < matched_track_ids.size(); ++obs_sal_pnt_ind)
        {
            Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps> hd_by_cam_state;
            Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps> hd_by_sal_pnt;
            Deriv_hd_by_cam_state_and_sal_pnt(obs_sal_pnt_ind, frame_ind, cam_state, cam_orient_wfc, matched_track_ids, &hd_by_cam_state, &hd_by_sal_pnt);

            // innovation variance
            const auto& Pprev = predicted_estim_vars_covar_;

            DependsOnOverallPackOrder();
            const auto& Pxx = Pprev.topLeftCorner<kCamStateComps, kCamStateComps>(); // camera-camera covariance

            size_t off = kCamStateComps + obs_sal_pnt_ind * kSalientPointComps;
            const auto& Pxy = Pprev.block<kCamStateComps, kSalientPointComps>(0, off); // camera-sal_pnt covariance
            const auto& Pyy = Pprev.block<kSalientPointComps, kSalientPointComps>(off, off); // sal_pnt-sal_pnt covariance
            Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> mid = 
                hd_by_cam_state * Pxy * hd_by_sal_pnt.transpose();

            Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> innov_var =
                hd_by_cam_state * Pxx * hd_by_cam_state.transpose() +
                mid + mid.transpose() +
                hd_by_sal_pnt * Pyy * hd_by_sal_pnt.transpose() +
                Rk;
            Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> innov_variance_inv = innov_var.inverse();

            // evaluate filter gain [13+6n, 2]
            EigenDynMat Knew = (
                Pprev.leftCols<kCamStateComps>() * hd_by_cam_state.transpose() +
                Pprev.middleCols<kSalientPointComps>(off) * hd_by_sal_pnt.transpose()) * innov_variance_inv;
            filter_gain_ = Knew;

            //
            size_t track_id = matched_track_ids[obs_sal_pnt_ind];
            const CornerTrack& track = track_rep_.GetPointTrackById(track_id);
            std::optional<CornerData> corner = track.GetCornerData(frame_ind);
            
            Point2 corner_pix = corner.value().PixelCoord;

            // project salient point into current camera

            SalientPointInternal sal_pnt;
            GetSalientPointData(obs_sal_pnt_ind, &sal_pnt);

            Eigen::Matrix<Scalar, kPixPosComps, 1> hd = ProjectInternalSalientPoint(cam_state, sal_pnt, nullptr);

            //
            EigenDynVec estim_vars_delta = Knew * (corner_pix.Mat() - hd);
            Scalar estim_vars_delta_norm = estim_vars_delta.norm();
            diff_vars_total += estim_vars_delta_norm;

            //
            EigenDynMat estim_vars_covar_delta = Knew * innov_var * Knew.transpose();
            Scalar estim_vars_covar_delta_norm = estim_vars_covar_delta.norm();
            diff_cov_total += estim_vars_covar_delta_norm;

            //
            estim_vars_ += estim_vars_delta;
            estim_vars_covar_ -= estim_vars_covar_delta;
        }
        VLOG(4) << "diff_vars=" << diff_vars_total << " diff_cov=" << diff_cov_total;
    }
    else
    {
        // we have no observations => current state <- prediction
        std::swap(estim_vars_, predicted_estim_vars_);
        std::swap(estim_vars_covar_, predicted_estim_vars_covar_);
    }

    OnEstimVarsChanged(frame_ind);
}

void DavisonMonoSlam::ProcessFrame_OneComponentOfOneObservationPerUpdate(size_t frame_ind, bool suppress_observations)
{
    
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

void DavisonMonoSlam::Deriv_hd_by_hu(suriko::Point2 corner_pix, Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>* hd_by_hu) const
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

    Eigen::Matrix<Scalar, 2, 2> dhu_by_dhd;
    dhu_by_dhd(0, 0) = tort + 2 * suriko::Sqr(dx * (ud - Cx))*p2;
    dhu_by_dhd(1, 1) = tort + 2 * suriko::Sqr(dy * (vd - Cy))*p2;
    dhu_by_dhd(1, 0) = 2 * suriko::Sqr(dx) * (vd - Cy)*(ud - Cx)*p2;
    dhu_by_dhd(0, 1) = 2 * suriko::Sqr(dy) * (vd - Cy)*(ud - Cx)*p2;

    *hd_by_hu = dhu_by_dhd.inverse(); // A.33
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

void DavisonMonoSlam::Deriv_hd_by_camera_state(const SalientPointInternal& sal_pnt,
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

    // A.46-A.49
    Eigen::Matrix<Scalar, 3, 3> dR_by_dq0;
    dR_by_dq0 <<
        2 * q[0], -2 * q[3], 2 * q[2],
        2 * q[3], 2 * q[0], -2 * q[1],
        -2 * q[2], 2 * q[1], 2 * q[0];

    Eigen::Matrix<Scalar, 3, 3> dR_by_dq1;
    dR_by_dq1 <<
        2 * q[1], 2 * q[2], 2 * q[3],
        2 * q[2], -2 * q[1], -2 * q[0],
        2 * q[3], 2 * q[0], -2 * q[1];

    Eigen::Matrix<Scalar, 3, 3> dR_by_dq2;
    dR_by_dq2 <<
        -2 * q[2], 2 * q[1], 2 * q[0],
        2 * q[1], 2 * q[2], 2 * q[3],
        -2 * q[0], 2 * q[3], -2 * q[2];

    Eigen::Matrix<Scalar, 3, 3> dR_by_dq3;
    dR_by_dq3 <<
        -2 * q[3], -2 * q[0], 2 * q[1],
        2 * q[0], -2 * q[3], 2 * q[2],
        2 * q[1], 2 * q[2], 2 * q[3];

    Eigen::Matrix<Scalar, 3, 1> cam_pos = cam_state.PosW;
    Eigen::Matrix<Scalar, 3, 1> first_cam_pos = sal_pnt.FirstCamPos;

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

void DavisonMonoSlam::Deriv_hd_by_sal_pnt(const SalientPointInternal& sal_pnt,
    const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
    const Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps>& hd_by_hu,
    const Eigen::Matrix<Scalar, kPixPosComps, kEucl3>& hu_by_hc,
    Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt) const
{
    Eigen::Matrix<Scalar, kEucl3, kEucl3> Rcw = cam_orient_wfc.transpose();

    Scalar cos_phi = std::cos(sal_pnt.ElevationPhi);
    Scalar sin_phi = std::sin(sal_pnt.ElevationPhi);
    Scalar cos_theta = std::cos(sal_pnt.AzimuthTheta);
    Scalar sin_theta = std::sin(sal_pnt.AzimuthTheta);

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
    dhc_by_dy.middleCols<1>(5) = Rcw * (sal_pnt.FirstCamPos - cam_state.PosW);

    // A.51
    *hd_by_sal_pnt = hd_by_hu * hu_by_hc * dhc_by_dy;
}

Eigen::Matrix<Scalar, DavisonMonoSlam::kPixPosComps,1> DavisonMonoSlam::ProjectInternalSalientPoint(const CameraPosState& cam_state, const SalientPointInternal& sal_pnt,
    SalPntProjectionIntermidVars *proj_hist) const
{
    Eigen::Matrix<Scalar, kEucl3, kEucl3> camk_orient_wfc;
    RotMatFromQuat(gsl::make_span<const Scalar>(cam_state.OrientationWfc.data(), kQuat4), &camk_orient_wfc);

    Eigen::Matrix<Scalar, kEucl3, kEucl3> camk_orient_cfw33 = camk_orient_wfc.transpose();

    SE3Transform camk_wfc(camk_orient_wfc, cam_state.PosW);
    SE3Transform camk_orient_cfw = SE3Inv(camk_wfc);

    //
    Eigen::Matrix<Scalar, kEucl3, 1> m;
    CameraCoordinatesEuclidUnityDirFromPolarAngles(sal_pnt.AzimuthTheta, sal_pnt.ElevationPhi, &m[0], &m[1], &m[2]);

    if (proj_hist != nullptr)
    {
        proj_hist->FirstCamSalPntUnityDir = m;
    }

    Scalar dist = sal_pnt.GetDist();

    // salient point in the world
    Eigen::Matrix<Scalar, kEucl3, 1> p_world = sal_pnt.FirstCamPos + (1 / sal_pnt.InverseDistRho)*m;

    // salient point in the cam-k (naive way)
    Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_camk_simple1 = camk_orient_cfw33 * (sal_pnt.FirstCamPos - cam_state.PosW + (1 / sal_pnt.InverseDistRho)*m);
    suriko::Point3 p_world_back_simple1 = SE3Apply(camk_wfc, suriko::Point3(sal_pnt_camk_simple1));

    // direction to the salient point in the cam-k divided by distance to the feature from the first camera
    Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_camk_scaled = camk_orient_cfw33 * (sal_pnt.InverseDistRho * (sal_pnt.FirstCamPos - cam_state.PosW) + m);
    Eigen::Matrix<Scalar, kEucl3, 1> sal_pnt_camk = sal_pnt_camk_scaled / sal_pnt.InverseDistRho;

    const auto& sal_pnt_cam = sal_pnt_camk_scaled;
    Eigen::Matrix<Scalar, kPixPosComps, 1> hd = ProjectCameraSalientPoint(sal_pnt_cam, proj_hist);
    return hd;
}

Eigen::Matrix<Scalar, DavisonMonoSlam::kPixPosComps,1> DavisonMonoSlam::ProjectCameraSalientPoint(
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

void DavisonMonoSlam::Deriv_hd_by_cam_state_and_sal_pnt(size_t obs_sal_pnt_ind, size_t frame_ind,
    const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
    const std::vector<size_t>& matched_track_ids,
    Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_cam_state,
    Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_sal_pnt) const
{
    size_t track_id = matched_track_ids[obs_sal_pnt_ind];
    const CornerTrack& track = track_rep_.GetPointTrackById(track_id);
    std::optional<CornerData> corner = track.GetCornerData(frame_ind);
    Point2 corner_pix = corner.value().PixelCoord;

    // project salient point into current camera

    SalientPointInternal sal_pnt;
    GetSalientPointData(obs_sal_pnt_ind, &sal_pnt);

    SalPntProjectionIntermidVars proj_hist{};
    Eigen::Matrix<Scalar, kPixPosComps, 1> hd = ProjectInternalSalientPoint(cam_state, sal_pnt, &proj_hist);

    // calculate derivatives

    // how distorted pixels coordinates depend on undistorted pixels coordinates
    Eigen::Matrix<Scalar, kPixPosComps, kPixPosComps> hd_by_hu;
    Deriv_hd_by_hu(corner_pix, &hd_by_hu);

    // A.34 how undistorted pixels coordinates hu=[uu,vu] depend on salient point (in camera) 3D meter coordinates [hcx,hcy,hcz] (A.23)
    Eigen::Matrix<Scalar, kPixPosComps, kEucl3> hu_by_hc;
    Deriv_hu_by_hc(proj_hist, &hu_by_hc);

    Deriv_hd_by_camera_state(sal_pnt, cam_state, cam_orient_wfc, proj_hist, hd_by_hu, hu_by_hc, hd_by_cam_state);

    Deriv_hd_by_sal_pnt(sal_pnt, cam_state, cam_orient_wfc, hd_by_hu, hu_by_hc, hd_by_sal_pnt);

    static bool debug_corner_coord_derivatives = false;
    if (debug_corner_coord_derivatives)
    {
        Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps> finite_diff_hd_by_xc;
        FiniteDiff_hd_by_camera_state(sal_pnt, kFiniteDiffEpsDebug, &finite_diff_hd_by_xc);

        Scalar diff1 = (finite_diff_hd_by_xc - *hd_by_cam_state).norm();

        Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps> finite_diff_hd_by_y;
        FiniteDiff_hd_by_sal_pnt_state(obs_sal_pnt_ind, cam_state, kFiniteDiffEpsDebug, &finite_diff_hd_by_y);

        Scalar diff2 = (finite_diff_hd_by_y - *hd_by_sal_pnt).norm();

        int z = 0;
    }
}

void DavisonMonoSlam::Deriv_Hrowblock_by_estim_vars(size_t obs_sal_pnt_ind, size_t frame_ind,
    const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
    const std::vector<size_t>& matched_track_ids,
    Eigen::Matrix<Scalar, kPixPosComps, Eigen::Dynamic>* Hrowblock_by_estim_vars) const
{
    Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps> hd_by_cam_state;
    Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps> hd_by_sal_pnt;
    Deriv_hd_by_cam_state_and_sal_pnt(obs_sal_pnt_ind, frame_ind, cam_state, cam_orient_wfc, matched_track_ids, &hd_by_cam_state, &hd_by_sal_pnt);
    
    // by camera variables
    Hrowblock_by_estim_vars->middleCols<kCamStateComps>(0) = hd_by_cam_state;

    // by salient point variables
    // observed corner position (hd) depends only on the position of corresponding salient point (and not on any other salient point)
    Hrowblock_by_estim_vars->middleCols<kSalientPointComps>(kCamStateComps + obs_sal_pnt_ind * kSalientPointComps) = hd_by_sal_pnt;
}

void DavisonMonoSlam::Deriv_H_by_estim_vars(size_t frame_ind,
    const CameraPosState& cam_state,
    const Eigen::Matrix<Scalar, kEucl3, kEucl3>& cam_orient_wfc,
    const std::vector<size_t>& matched_track_ids, EigenDynMat* H_by_estim_vars) const
{
    EigenDynMat& H = *H_by_estim_vars;

    size_t n = EstimatedVarsCount();
    size_t matched_corners = matched_track_ids.size();
    H.resize(kPixPosComps * matched_corners, n);
    H.setZero();

    //

    for (size_t obs_sal_pnt_ind = 0; obs_sal_pnt_ind < matched_track_ids.size(); ++obs_sal_pnt_ind)
    {
        Eigen::Matrix<Scalar, kPixPosComps, Eigen::Dynamic> Hrowblock;
        Hrowblock.resize(Eigen::NoChange, n);
        Hrowblock.setZero();
        Deriv_Hrowblock_by_estim_vars(obs_sal_pnt_ind, frame_ind, cam_state, cam_orient_wfc, matched_track_ids, &Hrowblock);

        H.middleRows<kPixPosComps>(obs_sal_pnt_ind*kPixPosComps) = Hrowblock;

        //// by camera variables
        //H.block<kPixPosComps, kCamStateComps>(obs_sal_pnt_ind*kPixPosComps, 0) = hd_by_cam_state;

        //// by salient point variables
        //// observed corner position (hd) depends only on the position of corresponding salient point (and not on any other salient point)
        //H.block<kPixPosComps, kSalientPointComps>(obs_sal_pnt_ind*kPixPosComps, kCamStateComps + obs_sal_pnt_ind * kSalientPointComps) = hd_by_sal_pnt;
    }
}

void DavisonMonoSlam::FiniteDiff_hd_by_camera_state(const SalientPointInternal& sal_pnt,
    Scalar finite_diff_eps,
    Eigen::Matrix<Scalar, kPixPosComps, kCamStateComps>* hd_by_xc) const
{
    Eigen::Matrix<Scalar, kCamStateComps, 1> cam_state = estim_vars_.topRows<kCamStateComps>();

    // copy cam_state
    for (size_t var_ind = 0; var_ind < kCamStateComps; ++var_ind)
    {
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

void DavisonMonoSlam::FiniteDiff_hd_by_sal_pnt_state(size_t obs_sal_pnt_ind, const CameraPosState& cam_state,
    Scalar finite_diff_eps,
    Eigen::Matrix<Scalar, kPixPosComps, kSalientPointComps>* hd_by_y) const
{
    Eigen::Matrix<Scalar, kSalientPointComps, 1> sal_pnt_state = estim_vars_.middleRows<kSalientPointComps>(kCamStateComps + obs_sal_pnt_ind * kSalientPointComps);

    // copy cam_state
    for (size_t var_ind = 0; var_ind < kSalientPointComps; ++var_ind)
    {
        sal_pnt_state[var_ind] += finite_diff_eps;

        SalientPointInternal sal_pnt_right;
        LoadSalientPointDataFromArray(gsl::make_span<const Scalar>(sal_pnt_state.data(), kSalientPointComps), &sal_pnt_right);

        Eigen::Matrix<Scalar, kPixPosComps, 1> hd_right = ProjectInternalSalientPoint(cam_state, sal_pnt_right, nullptr);
        
        //
        sal_pnt_state[var_ind] -= 2 * finite_diff_eps;

        SalientPointInternal sal_pnt_left;
        LoadSalientPointDataFromArray(gsl::make_span<const Scalar>(sal_pnt_state.data(), kSalientPointComps), &sal_pnt_left);

        Eigen::Matrix<Scalar, kPixPosComps, 1> hd_left = ProjectInternalSalientPoint(cam_state, sal_pnt_left, nullptr);
        hd_by_y->middleCols<1>(var_ind) = (hd_right - hd_left) / (2 * finite_diff_eps);
    }
}

size_t DavisonMonoSlam::SalientPointsCount() const
{
    return (estim_vars_covar_.cols() - kCamStateComps) / kSalientPointComps;
}

size_t DavisonMonoSlam::EstimatedVarsCount() const
{
    return estim_vars_covar_.cols();
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
    Eigen::Matrix<Scalar, kQuat4, kAngVelocComps> q3_by_wk;
    Deriv_q3_by_w(dT, &q3_by_wk);
    m.block<kQuat4, kAngVelocComps>(kEucl3, kEucl3 + kQuat4 +  kVelocComps) = q3_by_wk;
    SRK_ASSERT(m.allFinite());
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
    Eigen::Matrix<Scalar, kQuat4, kAngVelocComps> q3_by_cap_omega;
    Deriv_q3_by_w(dT, &q3_by_cap_omega);

    m.block<kQuat4, kAngVelocComps>(kEucl3, kEucl3) = q3_by_cap_omega;
    SRK_ASSERT(m.allFinite());
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
    // next 3 rows
    for (size_t i = 0; i<kAngVelocComps; ++i)
        for (size_t j = 0; j < kAngVelocComps; ++j)
        {
            Scalar c = std::cos(0.5*w_norm*deltaT);
            Scalar s = std::sin(0.5*w_norm*deltaT);

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
    m = q3_by_q1 * q1_by_wk;
}

void DavisonMonoSlam::GetUncertaintCameraOrientation(
    Eigen::Matrix<Scalar, kEucl3, 1>* cam_pos,
    Eigen::Matrix<Scalar, kEucl3, kEucl3>* cam_pos_uncert,
    Eigen::Matrix<Scalar, kQuat4, 1>* cam_orient_quat) const
{
    DependsOnCameraPosPackOrder();

    // mean of camera position
    auto& m = *cam_pos;
    m[0] = estim_vars_[0];
    m[1] = estim_vars_[1];
    m[2] = estim_vars_[2];
    SRK_ASSERT(m.allFinite());

    // uncertainty of camera position
    const auto& orig_uncert = estim_vars_covar_.block<kEucl3, kEucl3>(0, 0);

    auto& unc = *cam_pos_uncert;
    unc = orig_uncert;
    SRK_ASSERT(unc.allFinite());

    auto& q = *cam_orient_quat;
    q[0] = estim_vars_[3];
    q[1] = estim_vars_[4];
    q[2] = estim_vars_[5];
    q[3] = estim_vars_[6];
    SRK_ASSERT(q.allFinite());
}

struct normal_random_variable_dyn
{
    normal_random_variable_dyn(Eigen::MatrixXd const& covar)
        : normal_random_variable_dyn(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable_dyn(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ 811 };
        static bool gen_seeded = false;
        if (!gen_seeded)
        {
            gen.seed(811);
            gen_seeded = true;
        }        
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};
template <typename _Scalar, int Dim>
struct normal_random_variable
{
    normal_random_variable(
        Eigen::Matrix<_Scalar,Dim,1> const& mean, 
        Eigen::Matrix<_Scalar,Dim,Dim> const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<_Scalar, Dim, Dim>> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::Matrix<_Scalar, Dim, 1> mean;
    Eigen::Matrix<_Scalar, Dim, Dim> transform;

    Eigen::Matrix<_Scalar, Dim, 1> operator()() const
    {
        static std::mt19937 gen{ 811 };
        static bool gen_seeded = false;
        if (!gen_seeded)
        {
            gen.seed(811);
            gen_seeded = true;
        }        
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::Matrix<_Scalar, Dim, 1>{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};
template <typename _Scalar, int Dim>
void CalcCovarMat(const std::vector <Eigen::Matrix<_Scalar, Dim,1>>& samples, Eigen::Matrix<_Scalar, Dim, Dim>* covar_mat)
{
    if (samples.empty())
        return;
    int size = samples[0].rows();
    
    Eigen::Matrix<_Scalar, Dim, 1> samples_mean;
    samples_mean.setZero(size);

    Eigen::Matrix<_Scalar, Dim, Dim> xy_mean;
    xy_mean.setZero(size, size);
    for (size_t i = 0; i<samples.size(); ++i)
    {
        const auto& s = samples[i];
        samples_mean += s;

        for (int row = 0; row < covar_mat->rows(); ++row)
            for (int col = row; col < covar_mat->cols(); ++col)
            {
                xy_mean(row,col) += s[row] * s[col];
            }
    }
    samples_mean *= (1.0 / samples.size());
    xy_mean *= (1.0 / samples.size());
    
    //
    covar_mat->setZero();
    for (int row = 0; row < covar_mat->rows(); ++row)
        for (int col = row; col < covar_mat->cols(); ++col)
        {
            Scalar var = xy_mean(row, col) - samples_mean[row] * samples_mean[col];
            (*covar_mat)(row, col) = var;
            (*covar_mat)(col, row) = var;
        }
}

Scalar DavisonMonoSlam::SalientPointInternal::GetDist() const
{
    SRK_ASSERT(!IsClose(0, InverseDistRho));
    return 1 / InverseDistRho;
}

void DavisonMonoSlam::LoadSalientPointDataFromArray(gsl::span<const Scalar> src, SalientPointInternal* result) const
{
    DependsOnSalientPointPackOrder();
    result->FirstCamPos[0] = src[0];
    result->FirstCamPos[1] = src[1];
    result->FirstCamPos[2] = src[2];
    result->AzimuthTheta = src[3];
    result->ElevationPhi = src[4];
    result->InverseDistRho = src[5];
}

size_t DavisonMonoSlam::SalientPointOffset(size_t sal_pnt_ind) const
{
    DependsOnOverallPackOrder();
    return kCamStateComps + sal_pnt_ind * kSalientPointComps;
}

void DavisonMonoSlam::GetSalientPointData(size_t sal_pnt_ind, SalientPointInternal* result) const
{
    size_t off = SalientPointOffset(sal_pnt_ind);
    auto sal_pnt_vars = gsl::make_span<const Scalar>(&estim_vars_[off], kSalientPointComps);
    LoadSalientPointDataFromArray(sal_pnt_vars, result);
}

void DavisonMonoSlam::GetCameraPosState(CameraPosState* result) const
{
    LoadCameraPosDataFromArray(gsl::make_span<const Scalar>(estim_vars_.data(), kCamStateComps), result);
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

void DavisonMonoSlam::GetSalientPointWorldPosAndUncertainty(size_t salient_pnt_ind, 
    Eigen::Matrix<Scalar, kEucl3, 1>* pos_mean, 
    Eigen::Matrix<Scalar, kEucl3, kEucl3>* pos_uncert) const
{
    SalientPointInternal sal_point;
    GetSalientPointData(salient_pnt_ind, &sal_point);

    Scalar hx, hy, hz;
    CameraCoordinatesEuclidFromPolar(sal_point.AzimuthTheta, sal_point.ElevationPhi, sal_point.GetDist(), &hx, &hy, &hz);

    // salient point in world coordinates = camera position + position of the salient point in the camera
    auto& m = *pos_mean;
    m[0] = sal_point.FirstCamPos[0] + hx;
    m[1] = sal_point.FirstCamPos[1] + hy;
    m[2] = sal_point.FirstCamPos[2] + hz;

    if (pos_uncert == nullptr)
        return;

    // propogate camera pos (Xc,Yc,Zc) and (theta-azimuth, phi-elevation,rho) into salient point pos (X,Y,Z)
    Eigen::Matrix<Scalar, kEucl3, kSalientPointComps> jacob;
    jacob.setIdentity();

    Scalar cos_theta = std::cos(sal_point.AzimuthTheta);
    Scalar sin_theta = std::sin(sal_point.AzimuthTheta);
    Scalar cos_phi = std::cos(sal_point.ElevationPhi);
    Scalar sin_phi = std::sin(sal_point.ElevationPhi);

    // deriv of (Xc,Yc,Zc) by theta-azimuth
    jacob(0, kEucl3) = sal_point.InverseDistRho * cos_phi * cos_theta;
    jacob(1, kEucl3) = 0;
    jacob(2, kEucl3) = -sal_point.InverseDistRho * cos_phi * sin_theta;

    // deriv of (Xc,Yc,Zc) by phi-elevation
    jacob(0, kEucl3 + 1) = -sal_point.InverseDistRho * sin_phi * sin_theta;
    jacob(1, kEucl3 + 1) = -sal_point.InverseDistRho * cos_phi;
    jacob(2, kEucl3 + 1) = -sal_point.InverseDistRho * sin_phi * cos_theta;

    // deriv of (Xc,Yc,Zc) by rho
    jacob(0, kEucl3 + 2) = cos_phi * sin_theta;
    jacob(1, kEucl3 + 2) = -sin_phi;
    jacob(2, kEucl3 + 2) = cos_phi * cos_theta;

    // mean of position
    size_t off = kCamStateComps + salient_pnt_ind * kSalientPointComps;

    // original uncertainty
    const auto& orig_uncert = estim_vars_covar_.block<kSalientPointComps, kSalientPointComps>(off, off);
    Eigen::Matrix<Scalar, kSalientPointComps, kSalientPointComps> orig_uncert_tmp = orig_uncert;

    // propagate uncertainty
    auto& unc = *pos_uncert;
    unc = jacob * orig_uncert * jacob.transpose();
    SRK_ASSERT(unc.allFinite());
    // TODO: check the mean and uncertainty covariance matrix can be interpreted as ellipsoid
    // TODO: check propogation matrix by hand

    static bool simulate_propagation = false;
    if (!simulate_propagation)
        return;

    // test propagation
    Eigen::Map<const Eigen::Matrix<Scalar, kSalientPointComps, 1>> orig_mean_mat(&estim_vars_[off]);
    Eigen::VectorXd orig_mean = orig_mean_mat;
    normal_random_variable<Scalar,kSalientPointComps> orig_rand1{ orig_mean, orig_uncert };
    std::vector<Eigen::Matrix<Scalar, kSalientPointComps, 1>> items1_origin;
    std::vector<Eigen::Matrix<Scalar, 3, 1>> items1;
    for (size_t i = 0; i<10000; ++i)
    {
        Eigen::Matrix<Scalar, kSalientPointComps,1> s = orig_rand1();
        items1_origin.push_back(s);
        
        Eigen::Matrix<Scalar, 3, 1> cp(s[0], s[1], s[2]);

        Scalar theta2 = s[3];
        Scalar phi2 = s[4];
        Scalar rho2 = s[5];
        Scalar dist2 = 1 / rho2;
        Scalar hx2, hy2, hz2;
        CameraCoordinatesEuclidFromPolar(theta2, phi2, dist2, &hx2, &hy2, &hz2);

        Eigen::Matrix<Scalar,3,1> s_new(3);
        s_new[0] = cp[0] + hx;
        s_new[1] = cp[1] + hy;
        s_new[2] = cp[2] + hz;
        items1.push_back(s_new);
    }

    Eigen::Matrix<Scalar, 6, 6> calc_covar3;
    CalcCovarMat<Scalar, 6>(items1_origin, &calc_covar3);

    Eigen::Matrix<Scalar,3,3> calc_covar2;
    CalcCovarMat<Scalar,3>(items1, &calc_covar2);

    unc = calc_covar2;


    //int size = 3;
    //Eigen::MatrixXd covar(size, size);
    //covar <<
    //    0.5, 0.3, 0.7,
    //    0.4, 2, 1,
    //    0.7, 1, 1.5;

    //Eigen::LLT<Eigen::Matrix<Scalar, 3, 3>> lltOfA(covar);
    //bool op2 = lltOfA.info() != Eigen::NumericalIssue;
    //SRK_ASSERT(op2);

    //normal_random_variable sample{ covar };
    //
    //std::vector <Eigen::VectorXd> data1;

    //for (size_t i=0; i<10000; ++i)
    //{
    //    Eigen::VectorXd s = sample();
    //    data1.push_back(s);
    //}
    //Eigen::MatrixXd calc_covar(size, size);
    //CalcCovarMat(data1, &calc_covar);
    //int z = 0;
}

void ProppogateUncertainty(
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* mean,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* uncert,
    void fun(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& xin, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* xout))
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    gen.seed(1234);

    std::normal_distribution<Scalar> x3D_noise_dis();

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

size_t DavisonMonoSlam::FramesCount() const
{
    size_t frames_count = track_rep_.FramesCount();
    return frames_count;
}

void DavisonMonoSlam::SetCornersMatcher(std::unique_ptr<CornersMatcherBase> corners_matcher)
{
    corners_matcher_.swap(corners_matcher);
}

void DavisonMonoSlam::LogReprojError() const
{
    Scalar err = CurrentFrameReprojError();
    VLOG(4) << "ReprojError=" << err;
}

Scalar DavisonMonoSlam::CurrentFrameReprojError() const
{
    // find the number of matched observations
    size_t frames_count = track_rep_.FramesCount();
    if (frames_count == 0)
        return 0;
    size_t frame_ind = frames_count - 1;

    std::vector<size_t> matched_track_ids;
    track_rep_.IteratePointsMarker();
    for (const CornerTrack& track : track_rep_.CornerTracks)
    {
        std::optional<CornerData> corner = track.GetCornerData(frame_ind);
        if (corner.has_value())
        {
            matched_track_ids.push_back(track.TrackId);
        }
    }
    size_t matched_corners = matched_track_ids.size();

    CameraPosState cam_state;
    GetCameraPosState(&cam_state);

    Scalar err_sum = 0;

    size_t pnts_count = SalientPointsCount();
    for (size_t obs_sal_pnt_ind =0; obs_sal_pnt_ind  < pnts_count; ++obs_sal_pnt_ind )
    {
        size_t track_id = matched_track_ids[obs_sal_pnt_ind];

        SalientPointInternal sal_pnt;
        GetSalientPointData(obs_sal_pnt_ind, &sal_pnt);

        Eigen::Matrix<Scalar, kPixPosComps, 1> pix = ProjectInternalSalientPoint(cam_state, sal_pnt, nullptr);

        //
        const CornerTrack& track = track_rep_.GetPointTrackById(track_id);
        std::optional<CornerData> corner = track.GetCornerData(frame_ind);
        Point2 corner_pix = corner.value().PixelCoord;
        Scalar err_one = (corner_pix.Mat() - pix).norm();
        err_sum += err_one;
    }

    return err_sum;
}
}
