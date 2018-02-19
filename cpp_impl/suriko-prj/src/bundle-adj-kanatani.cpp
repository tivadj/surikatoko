#include <string>
#include <array>
#include <algorithm> // std::fill
#include <vector>
#include <optional>
#include <cmath> // std::isnan
#include <iostream>
#include <gsl/span>
#include <gsl/gsl_assert>
#include <glog/logging.h>
#include <Eigen/Dense>
#include "suriko/approx-alg.h"
#include "suriko/bundle-adj-kanatani.h"
#include "suriko/obs-geom.h"
#include "suriko/rt-config.h"
#include "suriko/eigen-helpers.hpp"

namespace suriko
{

namespace {
    constexpr auto kWVarsCount = BundleAdjustmentKanatani::kWVarsCount;

    constexpr bool kDebugCorrectSalientPoints = true; // true to optimize salient points
    constexpr bool kDebugCorrectCamIntrinsics = true;
    constexpr bool kDebugCorrectTranslations = true;
    constexpr bool kDebugCorrectRotations = true;

    /// Declares code dependency on the order of variables [[fx fy u0 v0] Tx Ty Tz Wx Wy Wz]
    void MarkOptVarsOrderDependency() {}

    /// Axis angle can't be recovered for identity R matrix. This situation may occur when at some point the
    /// camera orientation coincide with the orientation at the first time point of the first frame. Such camera orientation
    /// may be inferred from the neighbour camera orientations.
    void MarkUndeterminedCameraOrientationAxisAngle() {}
}


/// Apply infinitesimal correction w to rotation matrix.
void IncrementRotMat(const Eigen::Matrix<Scalar, kWVarsCount, kWVarsCount>& R, const Eigen::Matrix<Scalar, kWVarsCount, 1>& w_delta,
    Eigen::Matrix<Scalar, kWVarsCount, kWVarsCount>* Rnew)
{
    constexpr int impl = 2;
    if (impl == 1)
    {
        // Rnew = (I + hat(w))*R where hat(w) is [3x3] and w is 3-element axis angle
        Eigen::Matrix<Scalar, kWVarsCount, kWVarsCount> direct_w_delta_hat;
        SkewSymmetricMat(w_delta, &direct_w_delta_hat);

        *Rnew = (Eigen::Matrix<Scalar, kWVarsCount, kWVarsCount>::Identity() + direct_w_delta_hat) * R;
    }
    else
    {
        // Rnew = Rodrigues(w)*R
        Eigen::Matrix<Scalar, kWVarsCount, kWVarsCount> rot_w;
        bool op = RotMatFromAxisAngle(w_delta, &rot_w);
        if (op)
        {
            *Rnew = rot_w * R; // formula 24 page 4
        }
        else
        {
            // Rodrigues formula failed. Keep R unchanged.
            *Rnew = R;
        }
    }
    std::string msg;
    bool is_so3 = IsSpecialOrthogonal(*Rnew, &msg);
    SRK_ASSERT(is_so3) << msg;
}

void AddDeltaToFrameInplace(gsl::span<const Scalar> frame_vars_delta, Eigen::Matrix<Scalar, 3, 3>* cam_intrinsics_mat, SE3Transform* direct_orient_cam)
{
    MarkOptVarsOrderDependency();
    Eigen::Matrix<Scalar, 3, 3>& K = *cam_intrinsics_mat;
    Scalar delta_fx = frame_vars_delta[0];
    K(0, 0) += delta_fx;
    Scalar delta_fy = frame_vars_delta[1];
    K(1, 1) += delta_fy;
    Scalar delta_u0 = frame_vars_delta[2];
    K(0, 2) += delta_u0;
    Scalar delta_v0 = frame_vars_delta[3];
    K(1, 2) += delta_v0;

    SE3Transform& rt = *direct_orient_cam;
    Scalar direct_tx = frame_vars_delta[4];
    rt.T[0] += direct_tx;
    Scalar direct_ty = frame_vars_delta[5];
    rt.T[1] += direct_ty;
    Scalar direct_tz = frame_vars_delta[6];
    rt.T[2] += direct_tz;

    Eigen::Map<const Eigen::Matrix<Scalar, kWVarsCount, 1>> direct_w_delta(&frame_vars_delta[7]);

    Eigen::Matrix<Scalar, kWVarsCount, kWVarsCount> newR;
    IncrementRotMat(rt.R, direct_w_delta, &newR);
    rt.R = newR;
}

//
SceneNormalizer::SceneNormalizer(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams, Scalar t1y, size_t unity_comp_ind)
        :map_(map),
    inverse_orient_cams_(inverse_orient_cams),
    normalized_t1y_dist_(t1y),
    unity_comp_ind_(unity_comp_ind)
{
    CHECK(t1y == 0 || t1y == 1 && "Only T1x and T1y is implemented");
}

auto SceneNormalizer::Opposite(SceneNormalizer::NormalizeAction action) {
    switch(action)
    {
    case NormalizeAction::Normalize:
        return NormalizeAction::Revert;
    case NormalizeAction::Revert:
        return NormalizeAction::Normalize;
    }
    AssertFalse();
}

SE3Transform SceneNormalizer::NormalizeOrRevertRT(const SE3Transform& inverse_orient_camk, const SE3Transform& inverse_orient_cam0, Scalar world_scale, NormalizeAction action, bool check_back_conv)
{
    const auto& Rk = inverse_orient_camk.R;
    const auto& Tk = inverse_orient_camk.T;

    SE3Transform result;
    if (action == NormalizeAction::Normalize)
    {
        result.R = Rk * inverse_orient_cam0.R.transpose();
        result.T = world_scale * (Tk - Rk * inverse_orient_cam0.R.transpose() * inverse_orient_cam0.T);
    } else if (action == NormalizeAction::Revert)
    {
        result.R = Rk * inverse_orient_cam0.R;

        auto Tktmp = Tk / world_scale;
        result.T = Tktmp + Rk * inverse_orient_cam0.T;
    }
    if (check_back_conv)
    {
        auto back_rt = NormalizeOrRevertRT(result, inverse_orient_cam0, world_scale, Opposite(action), false);
        Scalar diffR = (Rk - back_rt.R).norm();
        Scalar diffT = (Tk - back_rt.T).norm();
        assert(IsClose(0, diffR, 1e-3) && "Error in normalization or reverting of R");
        assert(IsClose(0, diffT, 1e-3) && "Error in normalization or reverting of T");
    }
    return result;
}

// TODO: back conversion check can be moved to unit testing
suriko::Point3 SceneNormalizer::NormalizeOrRevertPoint(const suriko::Point3& x3D, const SE3Transform& inverse_orient_cam0, Scalar world_scale, NormalizeAction action, bool check_back_conv)
{
    suriko::Point3 result(0, 0, 0);
    if (action == NormalizeAction::Normalize)
    {
        // RT for frame0 transform 3D point from world into coordinates of first camera
        result = SE3Apply(inverse_orient_cam0, x3D); // = R0*X + T0
        result.Mat() *= world_scale;
    } else if (action == NormalizeAction::Revert)
    {
        Eigen::Matrix<Scalar, 3, 1> X3Dtmp = x3D.Mat() * (1 / world_scale);
        result.Mat() = inverse_orient_cam0.R.transpose() * (X3Dtmp - inverse_orient_cam0.T);
    }
    if (check_back_conv)
    {
        auto back_pnt = NormalizeOrRevertPoint(result, inverse_orient_cam0, world_scale, Opposite(action), false);
        Scalar diff = (back_pnt.Mat() - x3D.Mat()).norm();
        assert(IsClose(0, diff, 1e-3) && "Error in normalization or reverting");
    }
    return result;
}

/// Modify structure so that it becomes 'normalized'.
/// The structure is updated in-place, because a copy of salient points and orientations of a camera can be too expensive to make.
bool SceneNormalizer::NormalizeWorldInplaceInternal()
{
    const SE3Transform& rt0 = (*inverse_orient_cams_)[0];
    const SE3Transform& rt1 = (*inverse_orient_cams_)[1];

    const auto& get0_from1 = SE3AFromB(rt0, rt1);
    const Eigen::Matrix<Scalar, 3, 1>& initial_camera_shift = get0_from1.T;

    // make y-component of the first camera shift a unity (formula 27) T1y==1
    Scalar initial_camera_shift_y = initial_camera_shift[unity_comp_ind_];
    Scalar atol = 1e-5;
    if (IsClose(0, initial_camera_shift_y, atol))
        return false; // can't normalize because of zero translation

    world_scale_ = normalized_t1y_dist_ / initial_camera_shift[unity_comp_ind_];
    // world_scale_ = 1;
    prenorm_rt0_ = rt0; // copy

    // update salient points
    auto point_track_count = map_->PointTrackCount();
    for (size_t pnt_track_id = 0; pnt_track_id < point_track_count; ++pnt_track_id)
    {
        suriko::Point3 salient_point = map_->GetSalientPoint(pnt_track_id);
        auto newX = NormalizeOrRevertPoint(salient_point, prenorm_rt0_, world_scale_, NormalizeAction::Normalize);
        map_->SetSalientPoint(pnt_track_id, newX);
    }

    // update orientations of the camera, so that the first camera becomes the center of the world
    auto frames_count = inverse_orient_cams_->size();
    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        auto rt = (*inverse_orient_cams_)[frame_ind];
        auto new_rt = NormalizeOrRevertRT(rt, prenorm_rt0_, world_scale_, NormalizeAction::Normalize);
        (*inverse_orient_cams_)[frame_ind] = new_rt;
    }
    bool check_post_cond = true;
    if (check_post_cond)
    {
        std::string err_msg;
        bool is_norm = CheckWorldIsNormalized(*inverse_orient_cams_, normalized_t1y_dist_, unity_comp_ind_, &err_msg);
        assert(is_norm);
    }
    return true;
}

void SceneNormalizer::RevertNormalization()
{
    // NOTE: if modifications were made after normalization, the reversion process won't modify the scene into
    // the initial (pre-normalization) state

    // update salient points
    auto point_track_count = map_->PointTrackCount();
    for (size_t pnt_track_id = 0; pnt_track_id < point_track_count; ++pnt_track_id)
    {
        suriko::Point3 salient_point = map_->GetSalientPoint(pnt_track_id);
        auto revertX = NormalizeOrRevertPoint(salient_point, prenorm_rt0_, world_scale_, NormalizeAction::Revert);
        map_->SetSalientPoint(pnt_track_id, revertX);
    }

    // update orientations of the camera
    auto frames_count = inverse_orient_cams_->size();
    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        auto rt = (*inverse_orient_cams_)[frame_ind];
        auto revert_rt = NormalizeOrRevertRT(rt, prenorm_rt0_, world_scale_, NormalizeAction::Revert);
        (*inverse_orient_cams_)[frame_ind] = revert_rt;
    }
}

auto NormalizeSceneInplace(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams, Scalar t1y_dist, size_t unity_comp_ind, bool* success)
{
    *success = false;

    auto scene_normalizer = SceneNormalizer(map, inverse_orient_cams, t1y_dist, unity_comp_ind);
    if (scene_normalizer.NormalizeWorldInplaceInternal()) {
        *success = true;
    }
    return scene_normalizer;
}

bool CheckWorldIsNormalized(const std::vector<SE3Transform>& inverse_orient_cams, Scalar t1y, int unity_comp_ind, std::string* err_msg)
{
    assert(inverse_orient_cams.size() >= 2);

    // the first frame is the identity
    const auto& rt0 = inverse_orient_cams[0];

    if (!rt0.R.isIdentity())
    {
        if (err_msg != nullptr) {
            std::stringstream buf;
            buf << "R0=Identity but was\n" <<rt0.R;
            *err_msg = buf.str();
        }
        return false;
    }
    Scalar atol = 1e-3;
    const auto diffT = rt0.T - Eigen::Matrix<Scalar,3,1>::Zero();
    if (diffT.norm() >= atol)
    {
        if (err_msg != nullptr) {
            std::stringstream buf;
            buf << "T0=zeros(3) but was {}" <<rt0.T;
            *err_msg = buf.str();
        }
        return false;
    }
    // the second frame has translation of unity length
    const auto& rt1 = SE3Inv(inverse_orient_cams[1]);
    const auto& t1 = rt1.T;

    if (!IsClose(t1y, t1[unity_comp_ind], atol))
    {
        if (err_msg != nullptr) {
            std::stringstream buf;
            buf << "expected T1y=1 but T1 was {}" <<t1;
            *err_msg = buf.str();
        }
        return false;
    }
    return true;
}

/// Represents overlapping parameters for some world 3D point.
class SalientPointPatch
{
    size_t point_track_id_;
    suriko::Point3 pnt3D_world_;
public:
    SalientPointPatch(size_t point_track_id, const Point3 &pnt3D_world)
        : point_track_id_(point_track_id),
              pnt3D_world_(pnt3D_world) {}
    size_t PointTrackId() const{
        return point_track_id_;
    }
    const suriko::Point3& PatchedSalientPoint() const{
        return pnt3D_world_;
    }
};

/// Represents overlapping parameters for camera orientation for some frame.
class FramePatch
{
    size_t frame_ind_ = (size_t)-1;
    std::optional<Eigen::Matrix<Scalar, 3, 3>> K_;
    std::optional<SE3Transform> inverse_orient_cam_;
public:
    FramePatch() = default;

    FramePatch(size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& K)
        : frame_ind_(frame_ind),
            K_(K)
    {
    }

    FramePatch(size_t frame_ind, const SE3Transform& inverse_orient_cam)
        : frame_ind_(frame_ind),
            inverse_orient_cam_(inverse_orient_cam)
    {
    }

    FramePatch(size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& K, const SE3Transform& inverse_orient_cam)
        : frame_ind_(frame_ind),
        K_(K),
        inverse_orient_cam_(inverse_orient_cam)
    {
    }

    size_t FrameInd() const
    {
        return frame_ind_;
    }

    const std::optional<Eigen::Matrix<Scalar, 3, 3>>& K() const
    {
        return K_;
    }

    const std::optional<SE3Transform>& InverseOrientCam() const
    {
        return inverse_orient_cam_;
    }
};

/// Internal routine to compute the bandle adjustment's reprojection error
/// with ability to overlap variables for specific salient point and/or camera orientation.
Scalar ReprojErrorWithOverlap(const FragmentMap& map,
                                const std::vector<SE3Transform>& inverse_orient_cams,
                                const CornerTrackRepository& track_rep,
                                const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat,
                                const std::vector<Eigen::Matrix<Scalar, 3, 3>>* intrinsic_cam_mats,
                                const SalientPointPatch* point_patch,
                                const FramePatch* frame_patch)
{
    CHECK(intrinsic_cam_mats != nullptr);

    size_t points_count = map.PointTrackCount();
    CHECK(points_count == track_rep.CornerTracks.size() && "Each 3D point must be tracked");

    Scalar err_sum = 0;

    size_t frames_count = inverse_orient_cams.size();
    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        const SE3Transform* pInverse_orient_cam = &inverse_orient_cams[frame_ind];
        const Eigen::Matrix<Scalar, 3, 3>* pK = &(*intrinsic_cam_mats)[frame_ind];

        // try to patch
        if (frame_patch != nullptr && frame_patch->FrameInd() == frame_ind)
        {
            if (frame_patch->InverseOrientCam().has_value())
                pInverse_orient_cam = &frame_patch->InverseOrientCam().value();

            if (frame_patch->K().has_value())
                pK = &frame_patch->K().value();
        }

        for (const CornerTrack& point_track : track_rep.CornerTracks)
        {
            std::optional<suriko::Point2> corner = point_track.GetCorner(frame_ind);
            if (!corner.has_value())
            {
                // the salient point is not detected in current frame and 
                // hence doesn't influence the reprojection error
                continue;
            }

            suriko::Point2 corner_pix = corner.value();
            suriko::Point3 x3D = map.GetSalientPoint(point_track.TrackId);

            // try patch
            if (point_patch != nullptr && point_track.TrackId == point_patch->PointTrackId())
                x3D = point_patch->PatchedSalientPoint();

            suriko::Point3 x3D_cam = SE3Apply(*pInverse_orient_cam, x3D);
            suriko::Point3 x3D_pix = suriko::Point3((*pK) * x3D_cam.Mat());
            // TODO: replace Point3 ctr with ToPoint factory method, error: call to 'ToPoint' is ambiguous

            bool zero_z = IsClose(0, x3D_pix[2], 1e-5);
            SRK_ASSERT(!zero_z) << "homog 2D point can't have Z=0";

            Scalar x = x3D_pix[0] / x3D_pix[2];
            Scalar y = x3D_pix[1] / x3D_pix[2];

            Scalar one_err = Sqr(x - corner_pix[0]) + Sqr(y - corner_pix[1]);
            SRK_ASSERT(std::isfinite(one_err));

            err_sum += one_err;
        }
    }
    SRK_ASSERT(std::isfinite(err_sum));
    return err_sum;
}

void BundleAdjustmentKanatani::UpdateNormalizePattern()
{
    MarkOptVarsOrderDependency();

    size_t off = 0;
    size_t out_ind = 0;
    
    // R0 = Identity, T0 = [0, 0, 0], T1y = 1
    // Frame 0
    off += kIntrinsicVarsCount;
    normalize_pattern_[out_ind++] = off + 0; // T0x
    normalize_pattern_[out_ind++] = off + 1; // T0y
    normalize_pattern_[out_ind++] = off + 2; // T0z
    normalize_pattern_[out_ind++] = off + 3; // W0x
    normalize_pattern_[out_ind++] = off + 4; // W0y
    normalize_pattern_[out_ind++] = off + 5; // W0z
    off += kTVarsCount + kWVarsCount;

    // Frame 1
    off += kIntrinsicVarsCount;
    normalize_pattern_[out_ind++] = off + unity_comp_ind_; // (T1x or) T1y

    normalize_pattern_count_ = out_ind;
    SRK_ASSERT(normalize_pattern_count_ <= normalize_pattern_.size());
}

Scalar BundleAdjustmentKanatani::ReprojError(const FragmentMap& map,
                                                const std::vector<SE3Transform>& inverse_orient_cams,
                                                const CornerTrackRepository& track_rep,
                                                const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat,
                                                const std::vector<Eigen::Matrix<Scalar, 3, 3>>* intrinsic_cam_mats)
{
    SalientPointPatch* point_patch = nullptr;
    FramePatch* frame_patch = nullptr;
    return ReprojErrorWithOverlap(map, inverse_orient_cams, track_rep, shared_intrinsic_cam_mat, intrinsic_cam_mats,
                                    point_patch, frame_patch);
}

bool BundleAdjustmentKanatani::ComputeInplace(FragmentMap& map,
                                                std::vector<SE3Transform>& inverse_orient_cams,
                                                const CornerTrackRepository& track_rep,
                                                const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat,
                                                std::vector<Eigen::Matrix<Scalar, 3, 3>>* intrinsic_cam_mats)
{
    this->map_ = &map;
    this->inverse_orient_cams_ = &inverse_orient_cams;
    this->track_rep_ = &track_rep;
    this->shared_intrinsic_cam_mat_ = shared_intrinsic_cam_mat;
    this->intrinsic_cam_mats_ = intrinsic_cam_mats;

    // An optimization problem (bundle adjustment) is indeterminant as is, so the boundary condition is introduced:
    // R0=Identity, T0=zeros(3), t2y=1
    bool normalize_op = false;
    scene_normalizer_ = NormalizeSceneInplace(&map, &inverse_orient_cams, t1y_, unity_comp_ind_, &normalize_op);
    if (!normalize_op)
        return false;

    //
    frame_vars_count_ = kIntrinsicVarsCount + kTVarsCount + kWVarsCount;
    SRK_ASSERT(frame_vars_count_ <= kMaxFrameVarsCount);
    
    UpdateNormalizePattern();

    // allocate space
    size_t points_count = map_->PointTrackCount();
    size_t frames_count = inverse_orient_cams_->size();

    gradE_finite_diff.resize(points_count * kPointVarsCount + frames_count * frame_vars_count_);
    // [3*points_count+10*frames_count]
    deriv_second_point_finite_diff.resize(points_count * kPointVarsCount, kPointVarsCount); // [3*points_count,3]
    deriv_second_frame_finite_diff.resize(frames_count * frame_vars_count_, frame_vars_count_); // [10*frames_count,10]
    deriv_second_pointframe_finite_diff.resize(points_count * kPointVarsCount, frames_count * frame_vars_count_); // [3*points_count,10*frames_count]
    corrections_.resize(points_count * kPointVarsCount + frame_vars_count_ * frames_count, 1); // corrections of vars
    
    size_t normalized_frame_vars_count = frame_vars_count_ * frames_count - normalize_pattern_count_;
    decomp_lin_sys_left_side1_.resize(normalized_frame_vars_count, normalized_frame_vars_count);
    decomp_lin_sys_right_side_.resize(normalized_frame_vars_count, 1);
    matG_.resize(normalized_frame_vars_count, normalized_frame_vars_count);
    normalized_vars_count_ = kPointVarsCount * points_count + frame_vars_count_ * frames_count - normalize_pattern_count_;

    bool result = ComputeOnNormalizedWorld();

    if (kSurikoDebug)
    {
        // check world is still normalized after optimization
        std::string err_msg;
        if (!CheckWorldIsNormalized(inverse_orient_cams, t1y_, unity_comp_ind_, &err_msg))
        {
            CHECK(false) <<err_msg;
        }
    }
    scene_normalizer_.RevertNormalization();

    return result;
}

bool BundleAdjustmentKanatani::ComputeOnNormalizedWorld()
{
    Scalar err_value_initial = ReprojError(*map_, *inverse_orient_cams_, *track_rep_, nullptr, intrinsic_cam_mats_);
    Scalar err_value = err_value_initial;

    // NOTE: we don't check absolute error here, because corrupted with noise data may have arbitrary large reproj err

    size_t it = 1;
    Scalar hessian_factor = 0.0001; // hessian's diagonal multiplier
    VLOG(4) << "initial reproj_err=" << err_value_initial << " hessian_factor=" << hessian_factor;

    while(true)
    {
        constexpr Scalar finite_diff_eps_debug = 1e-5;
        ComputeCloseFormReprErrorDerivatives(&gradE_finite_diff, &deriv_second_point_finite_diff, &deriv_second_frame_finite_diff, &deriv_second_pointframe_finite_diff, finite_diff_eps_debug);


        enum class TargFunDecreaseResult { Decreased, HessianOverflow };

        // tries to decrease target optimization function by varying hessian's diagonal factor
        auto try_decrease_targ_fun = [this](Scalar entry_err_value, Scalar* hessian_factor, Scalar* err_value_new) -> TargFunDecreaseResult
        {
            // backup current state(world points and camera orientations)
            FragmentMap map_copy = *map_;
            std::vector<Eigen::Matrix<Scalar, 3, 3>> intrinsic_cam_mats_copy = *intrinsic_cam_mats_;
            std::vector<SE3Transform> inverse_orient_cam_copy = *inverse_orient_cams_;

            // loop to find a hessian factor which decreases the target optimization function
            while (true)
            {
                // 1. the normalization(known R0, T0, T1y) is applied to corrections introducing gaps(plane->gaps)
                // 2. the linear system of equations is solved for vector of gapped corrections
                // 3. the gapped corrections are converted back to the plane vector(gaps->plane)
                // 4. the plane vector of corrections is used to adjust the optimization variables

                bool impl = true;
                if (impl)
                    EstimateCorrectionsDecomposedInTwoPhases(gradE_finite_diff, deriv_second_point_finite_diff, deriv_second_frame_finite_diff, deriv_second_pointframe_finite_diff, *hessian_factor, &corrections_);
                else
                    EstimateCorrectionsNaive(gradE_finite_diff, deriv_second_point_finite_diff, deriv_second_frame_finite_diff, deriv_second_pointframe_finite_diff, *hessian_factor, &corrections_);

                ApplyCorrections(corrections_);

                *err_value_new = ReprojError(*map_, *inverse_orient_cams_, *track_rep_, nullptr, intrinsic_cam_mats_);
                VLOG(4) << "try_hessian: got reproj_err=" << *err_value_new << " for hessian_factor=" << *hessian_factor;
                
                Scalar err_value_change = *err_value_new - entry_err_value;
                bool target_fun_decreased = err_value_change < 0;
                if (target_fun_decreased)
                    return TargFunDecreaseResult::Decreased;

                // at this point, the value of target minimization function increases, try again with different hessian factor
                // restore saved state
                *map_ = map_copy;
                *intrinsic_cam_mats_ = intrinsic_cam_mats_copy;
                *inverse_orient_cams_ = inverse_orient_cam_copy;

                *hessian_factor *= 10; // prefer more the Steepest descent

                if (*hessian_factor >  max_hessian_factor_)
                {
                    // prevent overflow for too big factors
                    return TargFunDecreaseResult::HessianOverflow;
                }
            }
            AssertFalse();
        };

        Scalar err_value_new = std::numeric_limits<Scalar>::quiet_NaN();
        TargFunDecreaseResult decrease_result = try_decrease_targ_fun(err_value, &hessian_factor, &err_value_new);

        VLOG(4) << "it=" << it << " reproj_err=" << err_value_new << " hessian_factor=" << hessian_factor;

        if (decrease_result == TargFunDecreaseResult::HessianOverflow)
            return false; // failed to optimize

        Scalar err_value_change = err_value_new - err_value;
        SRK_ASSERT(err_value_change < 0) << "Target error function's value must decrease";

        if (std::abs(err_value_change) < min_err_change_abs_)
            return true; // success, reached target level of minimization of reprojection error

        err_value = err_value_new;

        // reprojection error is decreased, but not enough => continue
        hessian_factor /= 10; // prefer more the Gauss - Newton
        it += 1;
    }
    AssertFalse();
}

auto BundleAdjustmentKanatani::GetFiniteDiffFirstPartialDerivPoint(size_t point_track_id, const suriko::Point3& pnt3D_world, size_t var1, Scalar finite_diff_eps) const -> Scalar
{
    suriko::Point3 pnt3D_left = pnt3D_world; // copy
    pnt3D_left[var1] -= finite_diff_eps;
    SalientPointPatch point_patch_left(point_track_id, pnt3D_left);

    Scalar x1_err_sum = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_,
                                                intrinsic_cam_mats_, &point_patch_left, nullptr);

    suriko::Point3 pnt3D_right = pnt3D_world; // copy
    pnt3D_right[var1] += finite_diff_eps;
    SalientPointPatch point_patch_right(point_track_id, pnt3D_right);

    Scalar x2_err_sum = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_,
                                                intrinsic_cam_mats_, &point_patch_right, nullptr);
    return (x2_err_sum - x1_err_sum) / (2 * finite_diff_eps);
}

auto BundleAdjustmentKanatani::GetFiniteDiffSecondPartialDerivPointPoint(size_t point_track_id, const suriko::Point3& pnt3D_world, size_t var1, size_t var2, Scalar finite_diff_eps) const -> Scalar
{
    // second order central finite difference formula
    // https://en.wikipedia.org/wiki/Finite_difference

    suriko::Point3 pnt3D_tmp = pnt3D_world; // copy
    pnt3D_tmp[var1] += finite_diff_eps;
    pnt3D_tmp[var2] += finite_diff_eps;
    SalientPointPatch point_patch1(point_track_id, pnt3D_tmp);
    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch1, nullptr);

    pnt3D_tmp = pnt3D_world; // copy
    pnt3D_tmp[var1] += finite_diff_eps;
    pnt3D_tmp[var2] += -finite_diff_eps;
    SalientPointPatch point_patch2(point_track_id, pnt3D_tmp);
    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch2, nullptr);

    pnt3D_tmp = pnt3D_world; // copy
    pnt3D_tmp[var1] += -finite_diff_eps;
    pnt3D_tmp[var2] += finite_diff_eps;
    SalientPointPatch point_patch3(point_track_id, pnt3D_tmp);
    Scalar e3 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch3, nullptr);

    pnt3D_tmp = pnt3D_world; // copy
    pnt3D_tmp[var1] += -finite_diff_eps;
    pnt3D_tmp[var2] += -finite_diff_eps;
    SalientPointPatch point_patch4(point_track_id, pnt3D_tmp);
    Scalar e4 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch4, nullptr);

    Scalar value = (e1 - e2 - e3 + e4) / (4 * finite_diff_eps * finite_diff_eps);
    return value;
}

auto BundleAdjustmentKanatani::GetFiniteDiffFirstPartialDerivFocalLengthFxFy(size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& K, size_t fxfy_ind, Scalar finite_diff_eps) const -> Scalar
{
    auto K_left = K; // copy
    K_left(fxfy_ind, fxfy_ind) -= finite_diff_eps;
    FramePatch patch_left(frame_ind, K_left);

    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch_left);

    auto K_right = K; // copy
    K_right(fxfy_ind, fxfy_ind) += finite_diff_eps;
    FramePatch patch_right(frame_ind, K_right);

    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch_right);
    Scalar value = (e2 - e1) / (2 * finite_diff_eps);
    return value;
}

auto BundleAdjustmentKanatani::GetFiniteDiffFirstPartialDerivPrincipalPoint(size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& K, size_t u0v0_ind, Scalar finite_diff_eps) const -> Scalar
{
    auto K_left = K; // copy
    K_left(u0v0_ind, 2) -= finite_diff_eps;
    FramePatch patch_left(frame_ind, K_left);

    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch_left);

    auto K_right = K; // copy
    K_right(u0v0_ind, 2) += finite_diff_eps;
    FramePatch patch_right(frame_ind, K_right);

    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch_right);
    Scalar value = (e2 - e1) / (2 * finite_diff_eps);
    return value;
}

auto BundleAdjustmentKanatani::GetFiniteDiffFirstPartialDerivTranslationDirect(
    size_t frame_ind, const SE3Transform& direct_orient_cam, size_t tind, Scalar finite_diff_eps) const -> Scalar
{
    // The RT is in direct mode, because all close form derivatives calculated for direct RT-mode, not inverse RT mode.

    SE3Transform direct_rt_left = direct_orient_cam; // copy
    direct_rt_left.T[tind] -= finite_diff_eps;
    SE3Transform invertse_rt_left = SE3Inv(direct_rt_left);
    FramePatch patch_left(frame_ind, invertse_rt_left);

    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch_left);

    SE3Transform direct_rt_right = direct_orient_cam; // copy
    direct_rt_right.T[tind] += finite_diff_eps;
    SE3Transform invertse_rt_right = SE3Inv(direct_rt_right);
    FramePatch patch_right(frame_ind, invertse_rt_right);

    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch_right);
    Scalar value = (e2 - e1) / (2 * finite_diff_eps);
    return value;
}

auto BundleAdjustmentKanatani::GetFiniteDiffFirstPartialDerivRotation(size_t frame_ind, const SE3Transform& direct_orient_cam, size_t wind, Scalar finite_diff_eps) const -> Scalar
{
    Eigen::Matrix<Scalar, kWVarsCount, 1> direct_w_delta;
    direct_w_delta.fill(0);
    direct_w_delta[wind] -= finite_diff_eps;

    Eigen::Matrix<Scalar, kWVarsCount, kWVarsCount> direct_Rnew;
    IncrementRotMat(direct_orient_cam.R, direct_w_delta, &direct_Rnew);

    SE3Transform direct_rt1(direct_Rnew, direct_orient_cam.T);
    SE3Transform invertse_rt1 = SE3Inv(direct_rt1);
    FramePatch patch1(frame_ind, invertse_rt1);

    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch1);

    //
    direct_w_delta.fill(0);
    direct_w_delta[wind] += finite_diff_eps;
    IncrementRotMat(direct_orient_cam.R, direct_w_delta, &direct_Rnew);

    SE3Transform direct_rt2(direct_Rnew, direct_orient_cam.T);
    SE3Transform invertse_rt2 = SE3Inv(direct_rt2);
    FramePatch patch2(frame_ind, invertse_rt2);

    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch2);
    Scalar value = (e2 - e1) / (2 * finite_diff_eps);
    return value;
}

auto BundleAdjustmentKanatani::GetFiniteDiffSecondPartialDerivFrameFrame(size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& cam_intrinsics_mat, 
    const SE3Transform& direct_orient_cam, size_t frame_var_ind1, size_t frame_var_ind2, Scalar finite_diff_eps) const -> Scalar
{
    std::array<Scalar, kMaxFrameVarsCount> frame_vars_delta;
    
    // var1 + eps, var2 + eps
    frame_vars_delta.fill(0);
    frame_vars_delta[frame_var_ind1] += finite_diff_eps;
    frame_vars_delta[frame_var_ind2] += finite_diff_eps;
    Eigen::Matrix<Scalar, 3, 3> K = cam_intrinsics_mat;
    SE3Transform direct_rt = direct_orient_cam;
    AddDeltaToFrameInplace(frame_vars_delta, &K, &direct_rt);
    SE3Transform inverse_orient_cam = SE3Inv(direct_rt);
    FramePatch frame_patch_right_right(frame_ind, K, inverse_orient_cam);

    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &frame_patch_right_right);

    // var1 + eps, var2 - eps
    frame_vars_delta.fill(0);
    frame_vars_delta[frame_var_ind1] += finite_diff_eps;
    frame_vars_delta[frame_var_ind2] -= finite_diff_eps;
    K = cam_intrinsics_mat;
    direct_rt = direct_orient_cam;
    AddDeltaToFrameInplace(frame_vars_delta, &K, &direct_rt);
    inverse_orient_cam = SE3Inv(direct_rt);
    FramePatch frame_patch_right_left(frame_ind, K, inverse_orient_cam);

    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &frame_patch_right_left);

    // var1 - eps, var2 + eps
    frame_vars_delta.fill(0);
    frame_vars_delta[frame_var_ind1] -= finite_diff_eps;
    frame_vars_delta[frame_var_ind2] += finite_diff_eps;
    K = cam_intrinsics_mat;
    direct_rt = direct_orient_cam;
    AddDeltaToFrameInplace(frame_vars_delta, &K, &direct_rt);
    inverse_orient_cam = SE3Inv(direct_rt);
    FramePatch frame_patch_left_right(frame_ind, K, inverse_orient_cam);
    Scalar e3 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &frame_patch_left_right);

    // var1 - eps, var2 - eps
    frame_vars_delta.fill(0);
    frame_vars_delta[frame_var_ind1] -= finite_diff_eps;
    frame_vars_delta[frame_var_ind2] -= finite_diff_eps;
    K = cam_intrinsics_mat;
    direct_rt = direct_orient_cam;
    AddDeltaToFrameInplace(frame_vars_delta, &K, &direct_rt);
    inverse_orient_cam = SE3Inv(direct_rt);
    FramePatch frame_patch_left_left(frame_ind, K, inverse_orient_cam);
    Scalar e4 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &frame_patch_left_left);

    // second order central finite difference formula
    // https://en.wikipedia.org/wiki/Finite_difference
    Scalar value = (e1 - e2 - e3 + e4) / (4 * finite_diff_eps * finite_diff_eps);
    return value;
}

auto BundleAdjustmentKanatani::GetFiniteDiffSecondPartialDerivPointFrame(
    size_t point_track_id, const suriko::Point3& pnt3D_world, size_t point_var_ind,
    size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& cam_intrinsics_mat,
    const SE3Transform& direct_orient_cam, size_t frame_var_ind,
    Scalar finite_diff_eps) const -> Scalar
{
    suriko::Point3 point1 = pnt3D_world; // copy
    point1[point_var_ind] -= finite_diff_eps;
    SalientPointPatch point_patch1(point_track_id, point1);

    suriko::Point3 point2 = pnt3D_world; // copy
    point2[point_var_ind] += finite_diff_eps;
    SalientPointPatch point_patch2(point_track_id, point2);

    //
    std::array<Scalar, kMaxFrameVarsCount> frame_vars_delta{};
    frame_vars_delta[frame_var_ind] = -finite_diff_eps;
    Eigen::Matrix<Scalar, 3, 3> K1 = cam_intrinsics_mat;
    SE3Transform direct_rt1 = direct_orient_cam;
    AddDeltaToFrameInplace(frame_vars_delta, &K1, &direct_rt1);
    SE3Transform inverse_orient_cam1 = SE3Inv(direct_rt1);
    FramePatch frame_patch1(frame_ind, K1, inverse_orient_cam1);

    frame_vars_delta[frame_var_ind] = finite_diff_eps;
    Eigen::Matrix<Scalar, 3, 3> K2 = cam_intrinsics_mat;
    SE3Transform direct_rt2 = direct_orient_cam;
    AddDeltaToFrameInplace(frame_vars_delta, &K2, &direct_rt2);
    SE3Transform inverse_orient_cam2 = SE3Inv(direct_rt2);
    FramePatch frame_patch2(frame_ind, K2, inverse_orient_cam2);
    FramePatch frame_patch2tmp(frame_ind, K2, inverse_orient_cam1);

    // var1 + eps, var2 + eps
    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch2, &frame_patch2);

    // var1 + eps, var2 - eps
    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch2, &frame_patch1);

    // var1 - eps, var2 + eps
    Scalar e3 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch1, &frame_patch2);

    // var1 - eps, var2 - eps
    Scalar e4 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch1, &frame_patch1);
    
    Scalar etmp1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch1, &frame_patch2);
    Scalar etmp2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch1, &frame_patch2tmp);

    // second order central finite difference formula
    // https://en.wikipedia.org/wiki/Finite_difference
    Scalar value = (e1 - e2 - e3 + e4) / (4 * finite_diff_eps * finite_diff_eps);
    return value;
}

void BundleAdjustmentKanatani::ComputeCloseFormReprErrorDerivatives(std::vector<Scalar>* grad_error,
    EigenDynMat* deriv_second_pointpoint,
    EigenDynMat* deriv_second_frameframe,
    EigenDynMat* deriv_second_pointframe, Scalar finite_diff_eps)
{
    Scalar rough_rtol = 0.2; // used to compare close and finite difference derivatives
    size_t points_count = map_->PointTrackCount();
    size_t frames_count = inverse_orient_cams_->size();

    // each reproj error derivative with respect to all variables is a sum of parts => initialize with zeros
    std::fill(grad_error->begin(), grad_error->end(), 0);
    deriv_second_pointpoint->fill(0);
    deriv_second_frameframe->fill(0);
    deriv_second_pointframe->fill(0);

    // The derivatives of error function with respect to variables are computed in two steps:
    // 1. The Pqr derivatives are computed. Pqr=P*[X Y Z 1] where P[3x4] is a projection matrix; Pqr[3x1].
    // 2. The error derivatives are computed using the Pqr derivatives.

    // compute point derivatives
    track_rep_->IteratePointsMarker();
    for (size_t point_track_id = 0; point_track_id < points_count; ++point_track_id)
    {
        const suriko::CornerTrack& point_track = track_rep_->GetPointTrackById(point_track_id);

        const suriko::Point3& salient_point = map_->GetSalientPoint(point_track_id);

        size_t pnt_ind = point_track_id;
        gsl::span<Scalar> grad_point = gsl::make_span(&(*grad_error)[pnt_ind * kPointVarsCount], kPointVarsCount);

        for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
        {
            std::optional<suriko::Point2> corner_pix_opt = point_track.GetCorner(frame_ind);
            if (!corner_pix_opt.has_value())
            {
                // Te salient point is not detected in current frame and hence doesn't influence the reprojection error.
                continue;
            }
            
            const suriko::Point2& corner_pix = corner_pix_opt.value();

            const SE3Transform& inverse_orient_cam = (*inverse_orient_cams_)[frame_ind];
            const Eigen::Matrix<Scalar, 3, 3>& K = (*intrinsic_cam_mats_)[frame_ind];

            Scalar f0 = K(2, 2);

            suriko::Point3 x3D_cam = SE3Apply(inverse_orient_cam, salient_point);
            Eigen::Matrix<Scalar, 3, 1> x3D_pix = K * x3D_cam.Mat();
            const Eigen::Matrix<Scalar, 3, 1>& pqr = x3D_pix;

            Eigen::Matrix<Scalar, 3, 4> P;
            P << K * inverse_orient_cam.R, K * inverse_orient_cam.T;

            Eigen::Matrix<Scalar, kPointVarsCount, PqrCount> point_pqr_deriv;
            ComputePointPqrDerivatives(P, &point_pqr_deriv);

            // 1st derivative Point
            for (size_t xyz_ind = 0; xyz_ind < kPointVarsCount; ++xyz_ind)
            {
                Scalar ax = FirstDerivFromPqrDerivative(f0, pqr, corner_pix, point_pqr_deriv(xyz_ind, 0), point_pqr_deriv(xyz_ind, 1), point_pqr_deriv(xyz_ind, 2));
                grad_point[xyz_ind] += ax;
            }

            // 2nd derivative Point - Point
            for (size_t var1 = 0; var1 < kPointVarsCount; ++var1)
            {
                Scalar gradp_byvar1 = point_pqr_deriv(var1, kPCompInd);
                Scalar gradq_byvar1 = point_pqr_deriv(var1, kQCompInd);
                Scalar gradr_byvar1 = point_pqr_deriv(var1, kRCompInd);
                for (size_t var2 = 0; var2 < kPointVarsCount; ++var2)
                {
                    Scalar gradp_byvar2 = point_pqr_deriv(var2, kPCompInd);
                    Scalar gradq_byvar2 = point_pqr_deriv(var2, kQCompInd);
                    Scalar gradr_byvar2 = point_pqr_deriv(var2, kRCompInd);
                    Scalar ax = SecondDerivFromPqrDerivative(pqr, gradp_byvar1, gradq_byvar1, gradr_byvar1, gradp_byvar2, gradq_byvar2, gradr_byvar2);
                    (*deriv_second_pointpoint)(pnt_ind * kPointVarsCount + var1, var2) += ax;
                }
            }
        }

        // all frames where the current point visible are processed, and now derivative with respect to point are ready and can be checked
        if (kSurikoDebug)
        {
            // check point hessian is invertible
            Eigen::Matrix<Scalar, kPointVarsCount, kPointVarsCount> point_hessian = (*deriv_second_pointpoint).middleRows<kPointVarsCount>(pnt_ind * kPointVarsCount);

            Eigen::Matrix<Scalar, kPointVarsCount, kPointVarsCount> point_hessian_inv;
            bool is_inverted = false;
            Scalar det = 0;
            point_hessian.computeInverseAndDetWithCheck(point_hessian_inv, det, is_inverted);
            CHECK(is_inverted) << "3x3 matrix of second derivatives of points is not invertible, point_track_id=" << point_track_id << " det=" << det << " mat=\n" << point_hessian;

            if (debug_reproj_error_first_derivatives_) // slow, check if requested
            {
                // 1st derivative Point
                for (size_t var1 = 0; var1 < kPointVarsCount; ++var1)
                {
                    Scalar close_deriv = grad_point[var1];
                    Scalar finite_diff_deriv = GetFiniteDiffFirstPartialDerivPoint(point_track_id, salient_point, var1, finite_diff_eps);
                    if (!IsClose(finite_diff_deriv, close_deriv, rough_rtol, 0))
                        VLOG(4) << "d1point[" << var1 << "] mismatch finitediff:" << finite_diff_deriv << " close:" << close_deriv;
                }
            }
            if (debug_reproj_error_derivatives_pointpoint_) // slow, check if requested
            {
                // 2nd derivative Point - Point
                for (size_t var1 = 0; var1 < kPointVarsCount; ++var1)
                    for (size_t var2 = 0; var2 < kPointVarsCount; ++var2)
                    {
                        Scalar close_deriv = (*deriv_second_pointpoint)(pnt_ind * kPointVarsCount + var1, var2);

                        Scalar finite_diff_deriv = GetFiniteDiffSecondPartialDerivPointPoint(point_track_id, salient_point, var1, var2, finite_diff_eps);
                        if (!IsClose(finite_diff_deriv, close_deriv, rough_rtol, 0))
                            VLOG(4) << "d2pointpoint[" << var1 << "," << var2 << "] mismatch finitediff:" << finite_diff_deriv << " close:" << close_deriv;
                    }
            }
        }
    }

    // dT
    size_t grad_frames_section_offset = points_count * kPointVarsCount; // frames goes after points
    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        const SE3Transform& inverse_orient_cam = (*inverse_orient_cams_)[frame_ind];
        const Eigen::Matrix<Scalar, 3, 3>& K = (*intrinsic_cam_mats_)[frame_ind];
        Scalar f0 = K(2, 2);

        size_t grad_cur_frame_offset = grad_frames_section_offset + frame_ind * frame_vars_count_;
        gsl::span<Scalar> grad_frame = gsl::make_span(&(*grad_error)[grad_cur_frame_offset], frame_vars_count_);

        track_rep_->IteratePointsMarker();
        for (size_t point_track_id = 0; point_track_id < points_count; ++point_track_id)
        {
            const suriko::CornerTrack& point_track = track_rep_->GetPointTrackById(point_track_id);
            std::optional<suriko::Point2> corner_pix_opt = point_track.GetCorner(frame_ind);
            if (!corner_pix_opt.has_value())
            {
                // Te salient point is not detected in current frame and hence doesn't influence the reprojection error.
                continue;
            }
            
            const suriko::Point2& corner_pix = corner_pix_opt.value();

            const suriko::Point3& salient_point = map_->GetSalientPoint(point_track_id);

            suriko::Point3 x3D_cam = SE3Apply(inverse_orient_cam, salient_point);
            Eigen::Matrix<Scalar, 3, 1> x3D_pix = K * x3D_cam.Mat();
            const Eigen::Matrix<Scalar, 3, 1>& pqr = x3D_pix;

            Eigen::Matrix<Scalar, kMaxFrameVarsCount, PqrCount> frame_pqr_deriv;
            size_t actual_frame_vars_count = 0;
            ComputeFramePqrDerivatives(K, inverse_orient_cam, salient_point, corner_pix, &frame_pqr_deriv, &actual_frame_vars_count);
            SRK_ASSERT(frame_vars_count_ == actual_frame_vars_count) << "all " << frame_vars_count_ << " pqr derivatives must be set, but only " << actual_frame_vars_count << " were set";

            // 1st derivative of error with respect to all frame variables [[fx fy u0 v0] Tx Ty Tz Wx Wy Wz]
            for (size_t frame_var_ind = 0; frame_var_ind < frame_vars_count_; ++frame_var_ind)
            {
                Scalar dp_by_var = frame_pqr_deriv(frame_var_ind, kPCompInd);
                Scalar dq_by_var = frame_pqr_deriv(frame_var_ind, kQCompInd);
                Scalar dr_by_var = frame_pqr_deriv(frame_var_ind, kRCompInd);
                Scalar s = FirstDerivFromPqrDerivative(f0, pqr, corner_pix, dp_by_var, dq_by_var, dr_by_var);
                grad_frame[frame_var_ind] += s;
            }

            // 2nd derivative of reprojection error with respect to Frame - Frame variables
            for (size_t var1 = 0; var1 < frame_vars_count_; ++var1)
            {
                Scalar dp_by_var1 = frame_pqr_deriv(var1, kPCompInd);
                Scalar dq_by_var1 = frame_pqr_deriv(var1, kQCompInd);
                Scalar dr_by_var1 = frame_pqr_deriv(var1, kRCompInd);
                
                for (size_t var2 = 0; var2 < frame_vars_count_; ++var2)
                {
                    Scalar dp_by_var2 = frame_pqr_deriv(var2, kPCompInd);
                    Scalar dq_by_var2 = frame_pqr_deriv(var2, kQCompInd);
                    Scalar dr_by_var2 = frame_pqr_deriv(var2, kRCompInd);

                    Scalar s = SecondDerivFromPqrDerivative(pqr, dp_by_var1, dq_by_var1, dr_by_var1, dp_by_var2, dq_by_var2, dr_by_var2);
                    (*deriv_second_frameframe)(frame_ind * frame_vars_count_ + var1, var2) += s;
                }
            }

            Eigen::Matrix<Scalar, 3, 4> P;
            P << K * inverse_orient_cam.R, K * inverse_orient_cam.T;

            Eigen::Matrix<Scalar, kPointVarsCount, PqrCount> point_pqr_deriv;
            ComputePointPqrDerivatives(P, &point_pqr_deriv);

            // 2nd derivative of reprojection error with respect to Point - Frame variables
            for (size_t point_var_ind = 0; point_var_ind < kPointVarsCount; ++point_var_ind)
            {
                Scalar gradp_by_var1 = point_pqr_deriv(point_var_ind, kPCompInd);
                Scalar gradq_by_var1 = point_pqr_deriv(point_var_ind, kQCompInd);
                Scalar gradr_by_var1 = point_pqr_deriv(point_var_ind, kRCompInd);

                for (size_t frame_var_ind = 0; frame_var_ind < frame_vars_count_; ++frame_var_ind)
                {
                    Scalar gradp_by_var2 = frame_pqr_deriv(frame_var_ind, kPCompInd);
                    Scalar gradq_by_var2 = frame_pqr_deriv(frame_var_ind, kQCompInd);
                    Scalar gradr_by_var2 = frame_pqr_deriv(frame_var_ind, kRCompInd);

                    Scalar s = SecondDerivFromPqrDerivative(pqr, gradp_by_var1, gradq_by_var1, gradr_by_var1, gradp_by_var2, gradq_by_var2, gradr_by_var2);
                    size_t pnt_ind = point_track_id;
                    (*deriv_second_pointframe)(pnt_ind * kPointVarsCount + point_var_ind, frame_ind * frame_vars_count_ + frame_var_ind) += s;
                }
            }
        }

        // all points in the current frame are processed, and now derivative with respect to frame are ready and can be checked
        if (kSurikoDebug)
        {
            SE3Transform direct_orient_cam = SE3Inv(inverse_orient_cam);
            if (debug_reproj_error_first_derivatives_) // slow, check if requested
            {
                // 1st derivative with respect to Point variables
                for (size_t var1 = 0; var1 < frame_vars_count_; ++var1)
                {
                    Scalar close_deriv = grad_frame[var1];

                    auto frame_finite_diff_deriv_fun = [this, frame_ind, &K, &direct_orient_cam, finite_diff_eps](size_t fram_var_ind) -> Scalar
                    {
                        SRK_ASSERT(fram_var_ind < frame_vars_count_);

                        size_t inside_frame_ind = fram_var_ind;
                        if (inside_frame_ind < kFxFyCount)
                            return GetFiniteDiffFirstPartialDerivFocalLengthFxFy(frame_ind, K, inside_frame_ind, finite_diff_eps);
                        inside_frame_ind -= kFxFyCount;

                        if (inside_frame_ind < kU0V0Count)
                            return GetFiniteDiffFirstPartialDerivPrincipalPoint(frame_ind, K, inside_frame_ind, finite_diff_eps);
                        inside_frame_ind -= kU0V0Count;

                        if (inside_frame_ind < kTVarsCount)
                            return GetFiniteDiffFirstPartialDerivTranslationDirect(frame_ind, direct_orient_cam, inside_frame_ind, finite_diff_eps);
                        inside_frame_ind -= kTVarsCount;

                        return GetFiniteDiffFirstPartialDerivRotation(frame_ind, direct_orient_cam, inside_frame_ind, finite_diff_eps);
                    };
                    Scalar finite_diff_deriv = frame_finite_diff_deriv_fun(var1);

                    if (!IsClose(finite_diff_deriv, close_deriv, rough_rtol, 0))
                        VLOG(4) << "d1frame[" << var1 << "] mismatch finitediff:" << finite_diff_deriv << " close:" << close_deriv;
                }
            }
            if (debug_reproj_error_derivatives_frameframe_) // slow, check if requested
            {
                // 2nd derivative with respect to Frame - Frame variables
                for (size_t var1 = 0; var1 < frame_vars_count_; ++var1)
                    for (size_t var2 = 0; var2 < frame_vars_count_; ++var2)
                    {
                        Scalar close_deriv = (*deriv_second_frameframe)(frame_ind * frame_vars_count_ + var1, var2);

                        Scalar finite_diff_deriv = GetFiniteDiffSecondPartialDerivFrameFrame(frame_ind, K, direct_orient_cam, var1, var2, finite_diff_eps);
                        if (!IsClose(finite_diff_deriv, close_deriv, rough_rtol, 0))
                            VLOG(4) << "d2frameframe[" << var1 << "," << var2 << "] mismatch finitediff:" << finite_diff_deriv << " close:" << close_deriv;
                    }
            }
        }
    }

    if (kSurikoDebug && debug_reproj_error_derivatives_pointframe_) // slow, check if requested
    {
        for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
        {
            const SE3Transform& inverse_orient_cam = (*inverse_orient_cams_)[frame_ind];
            SE3Transform direct_orient_cam = SE3Inv(inverse_orient_cam);
            const Eigen::Matrix<Scalar, 3, 3>& K = (*intrinsic_cam_mats_)[frame_ind];

            track_rep_->IteratePointsMarker();
            for (size_t point_track_id = 0; point_track_id < points_count; ++point_track_id)
            {
                const suriko::Point3& salient_point = map_->GetSalientPoint(point_track_id);

                // 2nd derivative with respect to Point - Frame variables
                for (size_t point_var = 0; debug_reproj_error_derivatives_pointframe_ && point_var < kPointVarsCount; ++point_var)
                    for (size_t frame_var = 0; frame_var < frame_vars_count_; ++frame_var)
                    {
                        size_t pnt_ind = point_track_id;
                        Scalar close_deriv = (*deriv_second_pointframe)(pnt_ind * kPointVarsCount + point_var, frame_ind * frame_vars_count_ + frame_var);

                        Scalar finite_diff_deriv = GetFiniteDiffSecondPartialDerivPointFrame(point_track_id, salient_point, point_var, frame_ind, K, direct_orient_cam, frame_var, finite_diff_eps);
                        if (!IsClose(finite_diff_deriv, close_deriv, rough_rtol, 0))
                            VLOG(4) << "d2frameframe[" << point_var << "," << frame_var << "] mismatch finitediff:" << finite_diff_deriv << " close:" << close_deriv;
                    }
            }
        }
    }
}

void BundleAdjustmentKanatani::ComputePointPqrDerivatives(const Eigen::Matrix<Scalar,3,4>& P, Eigen::Matrix<Scalar, kPointVarsCount, PqrCount>* point_pqr_deriv) const
{
    point_pqr_deriv->row(0) = P.col(0); // d(p, q, r) / dX
    point_pqr_deriv->row(1) = P.col(1); // d(p, q, r) / dY
    point_pqr_deriv->row(2) = P.col(2); // d(p, q, r) / dZ
}

void BundleAdjustmentKanatani::ComputeFramePqrDerivatives(const Eigen::Matrix<Scalar, 3, 3>& K, const SE3Transform& inverse_orient_cam, 
    const suriko::Point3& salient_point, const suriko::Point2& corner_pix,
    Eigen::Matrix<Scalar, kMaxFrameVarsCount, PqrCount>* frame_pqr_deriv, gsl::not_null<size_t*> out_frame_vars_count) const
{
    Scalar fx = K(0, 0);
    Scalar fy = K(1, 1);
    Scalar u0 = K(0, 2);
    Scalar v0 = K(1, 2);
    Scalar f0 = K(2, 2);

    suriko::Point3 x3D_cam = SE3Apply(inverse_orient_cam, salient_point);
    Eigen::Matrix<Scalar, 3, 1> x3D_pix = K * x3D_cam.Mat();
    const Eigen::Matrix<Scalar, 3, 1>& pqr = x3D_pix;

    MarkOptVarsOrderDependency();
    size_t inside_frame = 0;

    // Pqr derivatives with respect to camera intrinsic variables [fx fy u0 v0]
    // fx
    Eigen::Matrix<Scalar, kMaxFrameVarsCount, PqrCount>& pqr_deriv = *frame_pqr_deriv;
    pqr_deriv(inside_frame, kPCompInd) = (1 / fx) * pqr[0] - u0 / (f0 * fx) * pqr[2];
    pqr_deriv(inside_frame, kQCompInd) = 0;
    pqr_deriv(inside_frame, kRCompInd) = 0;
    inside_frame += 1;

    // fy
    pqr_deriv(inside_frame, kPCompInd) = 0;
    pqr_deriv(inside_frame, kQCompInd) = (1 / fy) * pqr[1] - v0 / (f0 * fy) * pqr[2];
    pqr_deriv(inside_frame, kRCompInd) = 0;
    inside_frame += 1;

    // u0
    pqr_deriv(inside_frame, kPCompInd) = (1 / f0) * pqr[2];
    pqr_deriv(inside_frame, kQCompInd) = 0;
    pqr_deriv(inside_frame, kRCompInd) = 0;
    inside_frame += 1;

    // v0
    pqr_deriv(inside_frame, kPCompInd) = 0;
    pqr_deriv(inside_frame, kQCompInd) = (1 / f0) * pqr[2];
    pqr_deriv(inside_frame, kRCompInd) = 0;
    inside_frame += 1;

    const SE3Transform& direct_orient_cam = SE3Inv(inverse_orient_cam);

    // 1st derivaive of error with respect to T translation
    Eigen::Matrix<Scalar, kTVarsCount, PqrCount> tvars_pqr_deriv;
    tvars_pqr_deriv.col(0) = -(fx * direct_orient_cam.R.col(0) + u0 * direct_orient_cam.R.col(2)); // dp / d(Tx, Ty, Tz)
    tvars_pqr_deriv.col(1) = -(fy * direct_orient_cam.R.col(1) + v0 * direct_orient_cam.R.col(2)); // dq / d(Tx, Ty, Tz)
    tvars_pqr_deriv.col(2) = -(f0 * direct_orient_cam.R.col(2)); // dr / d(Tx, Ty, Tz)

    pqr_deriv.middleRows<kTVarsCount>(inside_frame) = tvars_pqr_deriv;
    inside_frame += kTVarsCount;

    // 1st derivative of error with respect to direct W (axis angle representation of rotation)
    Eigen::Matrix<Scalar, kWVarsCount, 1> rot1 = fx * direct_orient_cam.R.col(0) + u0 * direct_orient_cam.R.col(2);
    Eigen::Matrix<Scalar, kWVarsCount, 1> rot2 = fy * direct_orient_cam.R.col(1) + v0 * direct_orient_cam.R.col(2);
    Eigen::Matrix<Scalar, kWVarsCount, 1> rot3 = f0 * direct_orient_cam.R.col(2);

    Eigen::Matrix<Scalar, kPointVarsCount, 1> t_to_salient_point = salient_point.Mat() - direct_orient_cam.T;

    Eigen::Matrix<Scalar, kWVarsCount, PqrCount> wvars_pqr_deriv;
    wvars_pqr_deriv.col(kPCompInd) = rot1.cross(t_to_salient_point);
    wvars_pqr_deriv.col(kQCompInd) = rot2.cross(t_to_salient_point);
    wvars_pqr_deriv.col(kRCompInd) = rot3.cross(t_to_salient_point);

    pqr_deriv.middleRows<kWVarsCount>(inside_frame) = wvars_pqr_deriv;
    inside_frame += kWVarsCount;
    *out_frame_vars_count = inside_frame;
}

// formula 8, returns scalar or vector depending on gradp_byvar type
Scalar BundleAdjustmentKanatani::FirstDerivFromPqrDerivative(Scalar f0, const Eigen::Matrix<Scalar, 3, 1>& pqr, const suriko::Point2& corner_pix,
    Scalar gradp_byvar, Scalar gradq_byvar, Scalar gradr_byvar) const
{
    SRK_ASSERT(!IsClose(0, pqr[2])) << "z != 0 because z is in denominator";
    Scalar result =
        (pqr[0] / pqr[2] - corner_pix[0] / f0) * (pqr[2] * gradp_byvar - pqr[0] * gradr_byvar) +
        (pqr[1] / pqr[2] - corner_pix[1] / f0) * (pqr[2] * gradq_byvar - pqr[1] * gradr_byvar);
    result *= 2 / (pqr[2] * pqr[2]);
    return result;
}

// formula 9
Scalar BundleAdjustmentKanatani::SecondDerivFromPqrDerivative(const Eigen::Matrix<Scalar, 3, 1>& pqr,
    Scalar gradp_byvar1, Scalar gradq_byvar1, Scalar gradr_byvar1,
    Scalar gradp_byvar2, Scalar gradq_byvar2, Scalar gradr_byvar2) const
{
    Scalar s =
        (pqr[2] * gradp_byvar1 - pqr[0] * gradr_byvar1) * (pqr[2] * gradp_byvar2 - pqr[0] * gradr_byvar2) +
        (pqr[2] * gradq_byvar1 - pqr[1] * gradr_byvar1) * (pqr[2] * gradq_byvar2 - pqr[1] * gradr_byvar2);
    s *= 2 / (pqr[2] * pqr[2] * pqr[2] * pqr[2]);
    return s;
}

void BundleAdjustmentKanatani::FillHessian(const EigenDynMat& deriv_second_pointpoint,
    const EigenDynMat& deriv_second_frameframe,
    const EigenDynMat& deriv_second_pointframe, Scalar hessian_factor, EigenDynMat* hessian)
{
    size_t points_count = map_->PointTrackCount();
    size_t frames_count = inverse_orient_cams_->size();
    
    SRK_ASSERT(hessian->rows() == hessian->rows()) << "Provide square matrix";
    SRK_ASSERT((size_t)hessian->rows() == points_count * kPointVarsCount + frames_count * frame_vars_count_);

    hessian->fill(0);

    track_rep_->IteratePointsMarker();
    for (size_t point_track_id = 0; point_track_id < points_count; ++point_track_id)
    {
        size_t pnt_ind = point_track_id;
        size_t vert_offset = pnt_ind * kPointVarsCount;
        
        // [3x3] point-point matrices on the diagonal
        hessian->block<kPointVarsCount, kPointVarsCount>(vert_offset, vert_offset) = deriv_second_pointpoint.middleRows<kPointVarsCount>(vert_offset);

        const auto& point_frame = deriv_second_pointframe.block(vert_offset, 0, kPointVarsCount, frames_count * frame_vars_count_); // [3x9*frames_count]
        hessian->block(vert_offset, points_count * kPointVarsCount, kPointVarsCount, frames_count * frame_vars_count_) = point_frame;
        hessian->block(points_count * kPointVarsCount, vert_offset, frames_count * frame_vars_count_, kPointVarsCount) = point_frame.transpose();
    }

    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        size_t vert_offset = frame_ind * frame_vars_count_;
        size_t cur_frame_offset = points_count * kPointVarsCount + vert_offset;

        // [9x9] frame-frame matrices
        hessian->block(cur_frame_offset, cur_frame_offset, frame_vars_count_, frame_vars_count_) = deriv_second_frameframe.middleRows(vert_offset, frame_vars_count_);
    }
    
    // scale diagonal elements
    for (Eigen::Index i = 0; i < hessian->rows(); ++i)
    {
        (*hessian)(i, i) *= (1 + hessian_factor);
    }
}

void BundleAdjustmentKanatani::FillCorrectionsGapsFromNormalized(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& normalized_corrections, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* corrections_with_gaps)
{
    constexpr Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();

    auto& corrs_with_gaps = *corrections_with_gaps;
    if (kSurikoDebug)
    {
        corrs_with_gaps.fill(nan);
    }

    size_t in_ind = 0;
    size_t out_ind = 0;
    size_t points_count = map_->PointTrackCount();

    // copy points corrections intact
    size_t num = points_count * kPointVarsCount;
    corrs_with_gaps.middleRows(out_ind, num) = normalized_corrections.middleRows(in_ind, num);
    in_ind += num;
    out_ind += num;

    constexpr Scalar no_adj = 0; // set zero corrections for frame0(T0 = [0, 0, 0] R0 = Identity), which means 'no adjustments'

    // Frame0
    // [fx fy u0 v0]
    num = kIntrinsicVarsCount;
    corrs_with_gaps.middleRows(out_ind, num) = normalized_corrections.middleRows(in_ind, num);
    in_ind += num;
    out_ind += num;

    // zero [T0x T0y T0z] and [W0x W0y W0z] where W(axis in angle - axis representation) means identity rotation
    num = kWVarsCount + kTVarsCount;
    corrs_with_gaps.middleRows(out_ind, num).fill(no_adj);
    out_ind += num;
    // in_ind is unchanged because list of corrections with gaps has no such variables

    // Frame1
    // [fx fy u0 v0]
    num = kIntrinsicVarsCount;
    corrs_with_gaps.middleRows(out_ind, num) = normalized_corrections.middleRows(in_ind, num);
    in_ind += num;
    out_ind += num;

    // set zero correction for frame1 T1y = fixed_const
    corrs_with_gaps[out_ind + unity_comp_ind_] = no_adj; // T1x or T1y

    // copy other corrections of T1 intact
    SRK_ASSERT(unity_comp_ind_ == 0 || unity_comp_ind_ == 1);
    corrs_with_gaps[out_ind + 1 - unity_comp_ind_] = normalized_corrections[in_ind + 0]; // T1x or T1y
    corrs_with_gaps[out_ind + 2]                   = normalized_corrections[in_ind + 1]; // T1z
    in_ind += kTVarsCount - 1; // count(T1x T1y T1z without T1x or T1y)=2
    out_ind += kTVarsCount;

    // copy corrections for other frames intact
    num = corrs_with_gaps.rows() - out_ind;
    corrs_with_gaps.middleRows(out_ind, num) = normalized_corrections.middleRows(in_ind, num);
    in_ind += num;
    out_ind += num;
    SRK_ASSERT((size_t)corrs_with_gaps.size() == out_ind);
    SRK_ASSERT((size_t)normalized_corrections.size() == in_ind);
    SRK_ASSERT(corrs_with_gaps.allFinite());
    bool check_back = false;
    if (kSurikoDebug && check_back)
    {
        EigenDynMat conv_back = corrs_with_gaps; // copy

        auto norm_pattern = normalize_pattern_; // copy
        for (size_t i = 0; i < normalize_pattern_count_; ++i)
            norm_pattern[i] += points_count * kPointVarsCount; // offset point variables

        auto remove_rows = gsl::make_span(norm_pattern.data(), normalize_pattern_count_);
        auto remove_cols = gsl::span<size_t>(nullptr);
        RemoveRowsAndColsInplace(remove_rows, remove_cols, &conv_back);
        Scalar diff_value = (normalized_corrections - conv_back).norm();
        CHECK(diff_value < 0.1);
    }
}

void BundleAdjustmentKanatani::EstimateCorrectionsNaive(const std::vector<Scalar>& grad_error, const EigenDynMat& deriv_second_pointpoint,
    const EigenDynMat& deriv_second_frameframe,
    const EigenDynMat& deriv_second_pointframe, Scalar hessian_factor, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* corrections_with_gaps)
{
    size_t n = corrections_with_gaps->size();
    EigenDynMat hessian(n, n);
    FillHessian(deriv_second_pointpoint, deriv_second_frameframe, deriv_second_pointframe, hessian_factor, &hessian);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> right_side(n, 1);
    right_side = -Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(grad_error.data(), n, 1);

    // remove rows/columns corresponding to normalized variables
    //std::vector<size_t> remove_pat;
    //EigenDynMat tmp_hessian = hessian;
    //EigenDynMat tmp_right_side = right_side;
    //for (size_t i = 0; i < normalize_pattern_count_; ++i)
    //{
    //    size_t rem_row_or_col_ind = normalize_pattern_[i];
    //    tmp_hessian = removeRow(tmp_hessian, rem_row_or_col_ind);
    //    tmp_hessian = removeCol(tmp_hessian, rem_row_or_col_ind);

    //    tmp_right_side = removeCol(tmp_right_side, rem_row_or_col_ind);
    //}

    //hessian = std::move(tmp_hessian);
    //right_side = std::move(tmp_right_side);

    //
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> normalized_corrections = hessian.householderQr().solve(right_side);

    bool check = true;
    if (check)
    {
        EigenDynMat right_direct = hessian * normalized_corrections;
        Scalar diff_value = (right_direct - right_side).norm();
        CHECK(diff_value < 1) << "must";
    }

    FillCorrectionsGapsFromNormalized(normalized_corrections, corrections_with_gaps);
}

void BundleAdjustmentKanatani::EstimateCorrectionsDecomposedInTwoPhases(const std::vector<Scalar>& grad_error, const EigenDynMat& deriv_second_pointpoint,
    const EigenDynMat& deriv_second_frameframe,
    const EigenDynMat& deriv_second_pointframe, Scalar hessian_factor, 
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* corrections_with_gaps)
{
    size_t points_count = map_->PointTrackCount();
    size_t frames_count = inverse_orient_cams_->size();

    // convert 2nd derivatives Frame - Frame matrix into the square shape
    auto fill_matG = [this, &deriv_second_frameframe, frames_count, hessian_factor](EigenDynMat* frame_frame_mat) {
        EigenDynMat& matG = *frame_frame_mat;
        matG.fill(0);
        size_t out_ind = 0;
        for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
        {
            const auto& ax = deriv_second_frameframe.middleRows(frame_ind * frame_vars_count_, frame_vars_count_);

            MarkOptVarsOrderDependency();
            size_t block_height;
            if (frame_ind == 0)
            {
                // take [fx fy u0 v0], skip R0=Identity and T0=[0 0 0]
                EigenDynMat block = ax.topLeftCorner<kIntrinsicVarsCount, kIntrinsicVarsCount>();
                //RemoveRowsAndColsInplace()
                block_height = (size_t)block.rows();

                matG.block(out_ind, out_ind, block_height, block_height) = block;
            }
            else if (frame_ind == 1)
            {
                // skip (T1x or) T1y row and column
                EigenDynMat block = ax;
                //LOG(INFO) << "\n" << block;

                //block = removeRow(block, unity_comp_ind_);
                //block = removeCol(block, unity_comp_ind_);
                std::array<size_t, 1> remove_rows = { unity_comp_ind_ };
                RemoveRowsAndColsInplace(remove_rows, remove_rows, &block);
                //LOG(INFO) << "\n" << block;

                block_height = (size_t)block.rows();

                matG.block(out_ind, out_ind, block_height, block_height) = block;
            }
            else
            {
                // [9x9] frame-frame matrices
                matG.block(out_ind, out_ind, frame_vars_count_, frame_vars_count_) = ax;
                block_height = (size_t)ax.rows();
            }

            // scale diagonal elements
            for (size_t i = 0; i < block_height; ++i)
                matG(out_ind + i, out_ind + i) *= 1 + hessian_factor;

            out_ind += block_height;
        }
    };

    auto get_scaled_point_hessian = [&deriv_second_pointpoint](size_t pnt_ind, Scalar hessian_factor,  Eigen::Matrix<Scalar, kPointVarsCount, kPointVarsCount>* point_hessian) -> void
    {
        *point_hessian = deriv_second_pointpoint.middleRows<kPointVarsCount>(pnt_ind * kPointVarsCount); // copy
        
        // scale diagonal elements
        for (size_t i = 0; i < kPointVarsCount; ++i)
            (*point_hessian)(i, i) *= 1 + hessian_factor;

        SRK_ASSERT(point_hessian->allFinite()) << "Possibly to big hessian factor c = " << hessian_factor;
    };

    auto get_normalized_point_frame = [this,&deriv_second_pointframe](size_t pnt_ind, EigenDynMat* mat) -> void
    {
        // make a copy because normalized columns have to be removed
        *mat = deriv_second_pointframe.middleRows<kPointVarsCount>(pnt_ind * kPointVarsCount); // 3x9*frames_count

        // delete normalized columns
        auto remove_cols = gsl::make_span(normalize_pattern_.data(), normalize_pattern_count_);
        auto remove_rows = gsl::span<size_t>(nullptr);
        RemoveRowsAndColsInplace(remove_rows, remove_cols, mat);
    };

    EigenDynMat& matG = matG_; // [9*frames_count,9*frames_count]
    fill_matG(&matG);

    // calculate deltas for frame unknowns

    EigenDynMat& decomp_lin_sys_left_side = decomp_lin_sys_left_side1_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& decomp_lin_sys_right_side = decomp_lin_sys_right_side_;

    decomp_lin_sys_left_side.fill(0);
    decomp_lin_sys_right_side.fill(0);

    track_rep_->IteratePointsMarker();
    for (size_t point_track_id = 0; point_track_id < points_count; ++point_track_id)
    {
        size_t pnt_ind = point_track_id;

        EigenDynMat point_frame;
        get_normalized_point_frame(pnt_ind, &point_frame);

        Eigen::Matrix<Scalar, kPointVarsCount, kPointVarsCount> point_hessian;
        get_scaled_point_hessian(pnt_ind, hessian_factor, &point_hessian);
        
        Eigen::Matrix<Scalar, kPointVarsCount, kPointVarsCount> point_hessian_inv = point_hessian.inverse();

        // left side
        EigenDynMat ax1 = point_frame.transpose() * point_hessian_inv * point_frame;
        decomp_lin_sys_left_side += ax1;

        // right side
        Eigen::Map<const Eigen::Matrix<Scalar, kPointVarsCount, 1>> gradeE_point(&grad_error[pnt_ind * kPointVarsCount]);
        EigenDynMat ax2 = point_frame.transpose() * point_hessian_inv * gradeE_point;
        decomp_lin_sys_right_side += ax2;
    }

    decomp_lin_sys_left_side = matG - decomp_lin_sys_left_side; // G-sum(F.E.F)

    size_t normalized_frame_vars_count = (size_t)decomp_lin_sys_right_side.rows();
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> frame_derivs_packed(&grad_error[points_count * kPointVarsCount], normalized_frame_vars_count);

    decomp_lin_sys_right_side -= frame_derivs_packed; // sum(F.E.gradE) - Df

    //
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> corrections_frame = decomp_lin_sys_left_side.householderQr().solve(decomp_lin_sys_right_side);

    // calculate deltas for point unknowns
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> normalized_corrections;
    normalized_corrections.resize(normalized_vars_count_, 1);

    track_rep_->IteratePointsMarker();
    for (size_t point_track_id = 0; point_track_id < points_count; ++point_track_id)
    {
        size_t pnt_ind = point_track_id;
        
        EigenDynMat point_frame;
        get_normalized_point_frame(pnt_ind, &point_frame);

        Eigen::Map<const Eigen::Matrix<Scalar, kPointVarsCount, 1>> gradeE_point(&grad_error[pnt_ind * kPointVarsCount]);

        Eigen::Matrix<Scalar, kPointVarsCount, kPointVarsCount> point_hessian;
        get_scaled_point_hessian(pnt_ind, hessian_factor, &point_hessian);

        Eigen::Matrix<Scalar, kPointVarsCount, kPointVarsCount> point_hessian_inv = point_hessian.inverse();

        Eigen::Matrix<Scalar, kPointVarsCount, 1> corrects_one_point = - point_hessian_inv * (point_frame * corrections_frame + gradeE_point);
        normalized_corrections.middleRows<kPointVarsCount>(pnt_ind*kPointVarsCount) = corrects_one_point;
    }

    normalized_corrections.middleRows(points_count * kPointVarsCount, corrections_frame.rows()) = corrections_frame;

    FillCorrectionsGapsFromNormalized(normalized_corrections, corrections_with_gaps);
}

void BundleAdjustmentKanatani::ApplyCorrections(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& corrections_with_gaps)
{
    size_t points_count = map_->PointTrackCount();
    size_t frames_count = inverse_orient_cams_->size();

    track_rep_->IteratePointsMarker();
    for (size_t point_track_id = 0; point_track_id < points_count; ++point_track_id)
    {
        size_t pnt_ind = point_track_id;
        Eigen::Map<const Eigen::Matrix<Scalar, kPointVarsCount, 1>> delta_point(&corrections_with_gaps[pnt_ind * kPointVarsCount]);

        suriko::Point3& salient_point = map_->GetSalientPoint(point_track_id);
        if (kDebugCorrectSalientPoints)
            salient_point.Mat() += delta_point;
    }

    MarkOptVarsOrderDependency();
    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        size_t cur_offset = points_count * kPointVarsCount + frame_ind * frame_vars_count_;

        // camera intrinsics
        gsl::span<const Scalar> delta_cam_intrinsics = gsl::make_span(&corrections_with_gaps[cur_offset], kIntrinsicVarsCount);
        Eigen::Matrix<Scalar, 3, 3> K = (*intrinsic_cam_mats_)[frame_ind];
        if (kDebugCorrectCamIntrinsics)
        {
            K(0, 0) += delta_cam_intrinsics[0]; // fx
            K(1, 1) += delta_cam_intrinsics[1]; // fy
            K(0, 2) += delta_cam_intrinsics[2]; // u0
            K(1, 2) += delta_cam_intrinsics[3]; // v0
        }
        cur_offset += kIntrinsicVarsCount;

        // Rotation-Translation
        SE3Transform& inverse_orient_cam = (*inverse_orient_cams_)[frame_ind];
        SE3Transform direct_orient_cam = SE3Inv(inverse_orient_cam);
        
        gsl::span<const Scalar> direct_deltaT = gsl::make_span(&corrections_with_gaps[cur_offset], kTVarsCount);
        if (kDebugCorrectTranslations)
        {
            direct_orient_cam.T[0] += direct_deltaT[0];
            direct_orient_cam.T[1] += direct_deltaT[1];
            direct_orient_cam.T[2] += direct_deltaT[2];
        }
        cur_offset += kTVarsCount;

        //gsl::span<const Scalar> deltaW = gsl::make_span(&corrections[cur_offset], kWVarsCount);
        Eigen::Map<const Eigen::Matrix<Scalar, kWVarsCount, 1>> direct_deltaW(&corrections_with_gaps[cur_offset]);
        Eigen::Matrix<Scalar, kWVarsCount, kWVarsCount> new_directR;
        IncrementRotMat(direct_orient_cam.R, direct_deltaW, &new_directR);

        if (kDebugCorrectRotations)
        {
            direct_orient_cam.R = new_directR;
        }

        if (kDebugCorrectTranslations || kDebugCorrectRotations)
            inverse_orient_cam = SE3Inv(direct_orient_cam);
        cur_offset += kWVarsCount;
    }
}
}
