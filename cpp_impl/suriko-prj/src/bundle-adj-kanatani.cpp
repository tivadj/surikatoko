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

namespace suriko
{

SceneNormalizer::SceneNormalizer(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams, Scalar t1y, int unity_comp_ind)
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

auto NormalizeSceneInplace(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams, Scalar t1y_dist, int unity_comp_ind, bool* success)
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

    void PopulateFrom(BundleAdjustmentKanatani::FrameFromOptVarsUpdater& frame_vars_updater)
    {
        frame_ind_ = frame_vars_updater.FrameInd();
        K_ = frame_vars_updater.CamIntrinsicsMat();
        inverse_orient_cam_ = frame_vars_updater.InverseOrientCam();
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
                continue;

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

/// :return: True if optimization converges successfully.
/// Stop conditions:
/// 1) If a change of error function slows down and becomes less than self.min_err_change
/// NOTE: There is no sense to provide absolute threshold on error function because when noise exist, the error will
/// not get close to zero.
bool BundleAdjustmentKanatani::ComputeInplace(FragmentMap& map,
                                                std::vector<SE3Transform>& inverse_orient_cams,
                                                const CornerTrackRepository& track_rep,
                                                const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat,
                                                const std::vector<Eigen::Matrix<Scalar, 3, 3>>* intrinsic_cam_mats,
                                                bool check_derivatives)
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

    // allocate space
    size_t points_count = map_->PointTrackCount();
    size_t frames_count = inverse_orient_cams_->size();

    gradE_finite_diff.resize(points_count * kPointVarsCount + frames_count * frame_vars_count_);
    // [3*points_count+10*frames_count]
    deriv_second_point_finite_diff.resize(points_count * kPointVarsCount, kPointVarsCount); // [3*points_count,3]
    deriv_second_frame_finite_diff.resize(frames_count * frame_vars_count_, frame_vars_count_); // [10*frames_count,10]
    deriv_second_pointframe_finite_diff.resize(points_count * kPointVarsCount, frames_count * frame_vars_count_);
    // [3*points_count,10*frames_count]

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
    Scalar finite_diff_eps = 1e-5;
    ComputeDerivativesFiniteDifference(finite_diff_eps, &gradE_finite_diff, &deriv_second_point_finite_diff, &deriv_second_frame_finite_diff, &deriv_second_pointframe_finite_diff);
    return true;
}

void BundleAdjustmentKanatani::ComputeDerivativesFiniteDifference(
    Scalar finite_diff_eps,
    std::vector<Scalar>* gradE,
    EigenDynMat* deriv_second_pointpoint,
    EigenDynMat* deriv_second_frameframe,
    EigenDynMat* deriv_second_pointframe)
{
    size_t points_count = map_->PointTrackCount();
    size_t frames_count = inverse_orient_cams_->size();

    static const Scalar kNan = std::numeric_limits<Scalar>::quiet_NaN();
    if (kSurikoDebug)
    {
        std::fill(gradE->begin(), gradE->end(), kNan);
        deriv_second_pointpoint->fill(kNan);
        deriv_second_frameframe->fill(kNan);
        deriv_second_pointframe->fill(kNan);
    }

    // compute point derivatives
    track_rep_->IteratePointsMarker();
    for (size_t point_track_id = 0; point_track_id < points_count; ++point_track_id)
    {
        LOG(INFO) << "point_track_id=" << point_track_id;
        const suriko::CornerTrack& point_track = track_rep_->GetPointTrackById(point_track_id);

        const suriko::Point3& salient_point = map_->GetSalientPoint(point_track_id);

        size_t pnt_ind = point_track_id;

        // 1st derivative Point
        gsl::span<Scalar> gradPoint = gsl::make_span(&(*gradE)[pnt_ind * kPointVarsCount], kPointVarsCount);
        for (size_t var1 = 0; var1 < kPointVarsCount; ++var1)
            gradPoint[var1] = GetFiniteDiffFirstPartialDerivPoint(point_track_id, salient_point, var1, finite_diff_eps);

        // 2nd derivative Point - Point
        for (size_t var1 = 0; var1 < kPointVarsCount; ++var1)
            for (size_t var2 = 0; var2 < kPointVarsCount; ++var2)
                (*deriv_second_pointpoint)(pnt_ind * kPointVarsCount + var1, var2) = GetFiniteDiffSecondPartialDerivPoint(point_track_id, salient_point, var1, var2, finite_diff_eps);

        if (kSurikoDebug)
        {
            // check point hessian is invertible
            Eigen::Matrix<Scalar, kPointVarsCount, kPointVarsCount> point_hessian = (*deriv_second_pointpoint).middleRows<kPointVarsCount>(pnt_ind * kPointVarsCount);

            Eigen::Matrix<Scalar, kPointVarsCount, kPointVarsCount> point_hessian_inv;
            bool is_inverted = false;
            Scalar det = 0;
            point_hessian.computeInverseAndDetWithCheck(point_hessian_inv, det, is_inverted);
            CHECK(is_inverted) << "3x3 matrix of second derivatives of points is not invertible, point_track_id=" <<point_track_id << " det=" << det << " mat=\n" << point_hessian;
        }
    }

    // dT
    size_t grad_frames_section_offset = points_count * kPointVarsCount; // frames goes after points
    for (size_t frame_ind = 1; frame_ind < frames_count; ++frame_ind)
    {
        LOG(INFO) << "frame_ind=" << frame_ind;

        const SE3Transform& inverse_orient_cam = (*inverse_orient_cams_)[frame_ind];
        const Eigen::Matrix<Scalar, 3, 3>& K = (*intrinsic_cam_mats_)[frame_ind];

        size_t grad_cur_frame_offset = grad_frames_section_offset + frame_ind * frame_vars_count_;

        track_rep_->IteratePointsMarker();
        for (size_t point_track_id = 0; point_track_id < points_count; ++point_track_id)
        {
            const suriko::CornerTrack& point_track = track_rep_->GetPointTrackById(point_track_id);

            const suriko::Point3& salient_point = map_->GetSalientPoint(point_track_id);

            size_t pnt_ind = point_track_id;

            MarkOptVarsOrderDependency();
            size_t inside_frame = 0;

            gsl::span<Scalar> grad_cam_intrinsic_vars = gsl::make_span(&(*gradE)[grad_cur_frame_offset], frame_vars_count_);

            // finite difference of intrinsic variables
            for (size_t fxfy_ind = 0; fxfy_ind < kFxFyCount; ++fxfy_ind)
            {
                grad_cam_intrinsic_vars[fxfy_ind] = 
                    GetFiniteDiffFirstPartialDerivFocalLengthFxFy(frame_ind, K, fxfy_ind, finite_diff_eps);
            }
            inside_frame += kFxFyCount;

            for (size_t u0v0_ind = 0; u0v0_ind < kU0V0Count; ++u0v0_ind)
            {
                grad_cam_intrinsic_vars[inside_frame + u0v0_ind] =
                    GetFiniteDiffFirstDerivPrincipalPoint(frame_ind, K, u0v0_ind, finite_diff_eps);
            }
            inside_frame += kU0V0Count;

            const SE3Transform& direct_orient_cam = SE3Inv(inverse_orient_cam);

            // 1st derivaive of T translation
            for (size_t tind = 0; tind < kTVarsCount; ++tind)
            {
                grad_cam_intrinsic_vars[inside_frame + tind] =
                    GetFiniteDiffFirstPartialDerivTranslationDirect(frame_ind, direct_orient_cam, tind, finite_diff_eps);
            }
            inside_frame += kTVarsCount;

            // 1st derivative of W (axis angle representation of rotation)
            // Axis angle can't be recovered for identity R matrix. This situation may occur when at some point the
            // camera orientation coincide with the orientation at the first time point. Then such camera orientation
            // may be inferred from the neighbour camera orientations.
            Eigen::Matrix<Scalar, 3, 1> direct_w;
            bool op = AxisAngleFromRotMat(direct_orient_cam.R, &direct_w);
            if (op)
            {
                for (size_t wind = 0; wind < kWVarsCount; ++wind)
                {
                    grad_cam_intrinsic_vars[inside_frame + wind] =
                        GetFiniteDiffFirstPartialDerivRotation(frame_ind, direct_orient_cam, direct_w, wind, finite_diff_eps);
                }
                inside_frame += kWVarsCount;

                FrameFromOptVarsUpdater frame_vars_updater(frame_ind, K, direct_orient_cam);

                // 2nd derivative Frame - Frame
                for (size_t var1 = 0; var1 < frame_vars_count_; ++var1)
                    for (size_t var2 = 0; var2 < frame_vars_count_; ++var2)
                    {
                        Scalar ax = GetFiniteDiffSecondPartialDerivFrameFrame(frame_vars_updater, var1, var2, finite_diff_eps);
                        (*deriv_second_frameframe)(frame_ind * frame_vars_count_ + var1, var2) = ax;
                    }

                // 2nd derivative Point - Frame
                for (size_t point_var_ind = 0; point_var_ind < kPointVarsCount; ++point_var_ind)
                    for (size_t frame_var_ind = 0; frame_var_ind < frame_vars_count_; ++frame_var_ind)
                    {
                        Scalar ax = GetFiniteDiffSecondPartialDerivPointFrame(point_track_id, salient_point, frame_vars_updater, point_var_ind, frame_var_ind, finite_diff_eps);
                        (*deriv_second_pointframe)(pnt_ind * kPointVarsCount + point_var_ind, frame_ind * frame_vars_count_ + frame_var_ind) = ax;
                    }
            }
        }
    }

    if (kSurikoDebug) // postcondition: all derivatives must be set
    {
        auto isfinite_pred = [](auto x) -> bool { return std::isfinite(x); };

        bool c1 = std::all_of(gradE->begin(), gradE->end(), isfinite_pred);
        CHECK(c1) << "failed to compute gradE";

        bool c2 = deriv_second_pointpoint->unaryExpr(isfinite_pred).all();
        CHECK(c2) << "failed to compute deriv_second_pointpoint";

        bool c3 = deriv_second_frameframe->unaryExpr(isfinite_pred).all();
        CHECK(c3) << "failed to compute deriv_second_frameframe";

        bool c4 = deriv_second_pointframe->unaryExpr(isfinite_pred).all();
        CHECK(c4) << "failed to compute deriv_second_pointframe";
    }
}

auto BundleAdjustmentKanatani::GetFiniteDiffFirstPartialDerivPoint(size_t point_track_id,
                                                                    const suriko::Point3& pnt3D_world, size_t var1,
                                                                    Scalar finite_diff_eps) const -> Scalar
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

auto BundleAdjustmentKanatani::GetFiniteDiffSecondPartialDerivPoint(size_t point_track_id,
                                                                    const suriko::Point3& pnt3D_world, size_t var1,
                                                                    size_t var2,
                                                                    Scalar finite_diff_eps) const -> Scalar
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

auto BundleAdjustmentKanatani::GetFiniteDiffFirstPartialDerivFocalLengthFxFy(
    size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& K, size_t fxfy_ind, Scalar finite_diff_eps) const -> Scalar
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

auto BundleAdjustmentKanatani::GetFiniteDiffFirstDerivPrincipalPoint(size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& K, size_t u0v0_ind, Scalar finite_diff_eps) const -> Scalar
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

auto BundleAdjustmentKanatani::GetFiniteDiffFirstPartialDerivRotation(size_t frame_ind, const SE3Transform& direct_orient_cam, const Eigen::Matrix<Scalar, 3, 1>& w_direct, size_t wind, Scalar finite_diff_eps) const -> Scalar
{
    Eigen::Matrix<Scalar, 3, 3> rot_mat;

    Eigen::Matrix<Scalar, 3, 1> direct_w1 = w_direct; // copy
    direct_w1[wind] -= finite_diff_eps;
    bool op = RotMatFromAxisAngle(direct_w1, &rot_mat);
    CHECK(op); // TODO: or return 0?
    SE3Transform direct_rt1(rot_mat, direct_orient_cam.T);
    SE3Transform invertse_rt1 = SE3Inv(direct_rt1);
    FramePatch patch1(frame_ind, invertse_rt1);

    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch1);

    Eigen::Matrix<Scalar, 3, 1> direct_w2 = w_direct; // copy
    direct_w2[wind] += finite_diff_eps;
    op = RotMatFromAxisAngle(direct_w2, &rot_mat);
    CHECK(op); // TODO: or return 0?
    SE3Transform direct_rt2(rot_mat, direct_orient_cam.T);
    SE3Transform invertse_rt2 = SE3Inv(direct_rt2);
    FramePatch patch2(frame_ind, invertse_rt2);

    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &patch2);
    Scalar value = (e2 - e1) / (2 * finite_diff_eps);
    return value;
}

auto BundleAdjustmentKanatani::GetFiniteDiffSecondPartialDerivFrameFrame(
    const FrameFromOptVarsUpdater& frame_vars_updater, size_t frame_var_ind1, size_t frame_var_ind2,
    Scalar finite_diff_eps) const -> Scalar
{
    FrameFromOptVarsUpdater frame_vars_updater1 = frame_vars_updater; // copy
    frame_vars_updater1.AddDelta(frame_var_ind1, finite_diff_eps);
    frame_vars_updater1.AddDelta(frame_var_ind2, finite_diff_eps);
    FramePatch frame_patch_right_right;
    frame_patch_right_right.PopulateFrom(frame_vars_updater1);

    FrameFromOptVarsUpdater frame_vars_updater2 = frame_vars_updater; // copy
    frame_vars_updater2.AddDelta(frame_var_ind1, finite_diff_eps);
    frame_vars_updater2.AddDelta(frame_var_ind2, -finite_diff_eps);
    FramePatch frame_patch_right_left;
    frame_patch_right_left.PopulateFrom(frame_vars_updater2);

    FrameFromOptVarsUpdater frame_vars_updater3 = frame_vars_updater; // copy
    frame_vars_updater3.AddDelta(frame_var_ind1, -finite_diff_eps);
    frame_vars_updater3.AddDelta(frame_var_ind2, finite_diff_eps);
    FramePatch frame_patch_left_right;
    frame_patch_left_right.PopulateFrom(frame_vars_updater3);

    FrameFromOptVarsUpdater frame_vars_updater4 = frame_vars_updater; // copy
    frame_vars_updater4.AddDelta(frame_var_ind1, -finite_diff_eps);
    frame_vars_updater4.AddDelta(frame_var_ind2, -finite_diff_eps);
    FramePatch frame_patch_left_left;
    frame_patch_left_left.PopulateFrom(frame_vars_updater4);

    // var1 + eps, var2 + eps
    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &frame_patch_right_right);

    // var1 + eps, var2 - eps
    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &frame_patch_right_left);

    // var1 - eps, var2 + eps
    Scalar e3 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &frame_patch_left_right);

    // var1 - eps, var2 - eps
    Scalar e4 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, nullptr, &frame_patch_left_left);

    // second order central finite difference formula
    // https://en.wikipedia.org/wiki/Finite_difference
    //Scalar value = (e1 - e2 - e3 + e4) / (4 * finite_diff_eps * finite_diff_eps);
    Scalar k = 1 / finite_diff_eps;
    Scalar value = (e1 - e2 - e3 + e4) *0.25*k*k;
    return value;
}

auto BundleAdjustmentKanatani::GetFiniteDiffSecondPartialDerivPointFrame(
    size_t point_track_id, const suriko::Point3& pnt3D_world,
    const FrameFromOptVarsUpdater& frame_vars_updater, size_t point_var_ind, size_t frame_var_ind,
    Scalar finite_diff_eps) const -> Scalar
{
    suriko::Point3 point1 = pnt3D_world; // copy
    point1[point_var_ind] -= finite_diff_eps;
    SalientPointPatch point_patch1(point_track_id, point1);

    suriko::Point3 point2 = pnt3D_world; // copy
    point2[point_var_ind] += finite_diff_eps;
    SalientPointPatch point_patch2(point_track_id, point2);

    //
    FrameFromOptVarsUpdater frame_vars_updater1 = frame_vars_updater; // copy
    frame_vars_updater1.AddDelta(frame_var_ind, -finite_diff_eps);
    FramePatch frame_patch1;
    frame_patch1.PopulateFrom(frame_vars_updater1);

    FrameFromOptVarsUpdater frame_vars_updater2 = frame_vars_updater; // copy
    frame_vars_updater2.AddDelta(frame_var_ind, finite_diff_eps);
    FramePatch frame_patch2;
    frame_patch2.PopulateFrom(frame_vars_updater2);

    // var1 + eps, var2 + eps
    Scalar e1 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch2, &frame_patch2);

    // var1 + eps, var2 - eps
    Scalar e2 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch2, &frame_patch1);

    // var1 - eps, var2 + eps
    Scalar e3 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch1, &frame_patch2);

    // var1 - eps, var2 - eps
    Scalar e4 = ReprojErrorWithOverlap(*map_, *inverse_orient_cams_, *track_rep_, shared_intrinsic_cam_mat_, intrinsic_cam_mats_, &point_patch1, &frame_patch1);

    // second order central finite difference formula
    // https://en.wikipedia.org/wiki/Finite_difference
    Scalar value = (e1 - e2 - e3 + e4) / (4 * finite_diff_eps * finite_diff_eps);
    return value;
}

BundleAdjustmentKanatani::FrameFromOptVarsUpdater::FrameFromOptVarsUpdater(
    size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& cam_intrinsics_mat, const SE3Transform& direct_orient_cam)
    : frame_ind_(frame_ind),
        cam_intrinsics_mat_(cam_intrinsics_mat),
        direct_orient_cam_(direct_orient_cam)
{
    inverse_orient_cam_ = SE3Inv(direct_orient_cam_);
    direct_orient_cam_valid_ = true;
    inverse_orient_cam_valid_ = true;
    UpdatePackedVars();
}

void BundleAdjustmentKanatani::FrameFromOptVarsUpdater::UpdatePackedVars()
{
    frame_vars_[0] = cam_intrinsics_mat_(0, 0); // fx
    frame_vars_[1] = cam_intrinsics_mat_(1, 1); // fy
    frame_vars_[2] = cam_intrinsics_mat_(0, 2); // u0
    frame_vars_[3] = cam_intrinsics_mat_(1, 2); // v0
    frame_vars_[4] = direct_orient_cam_.T[0]; // Tx
    frame_vars_[5] = direct_orient_cam_.T[1]; // Ty
    frame_vars_[6] = direct_orient_cam_.T[2]; // Tz

    //
    SRK_ASSERT(direct_orient_cam_valid_);

    Eigen::Matrix<Scalar, 3, 1> direct_w;
    bool op = AxisAngleFromRotMat(direct_orient_cam_.R, &direct_w);
    CHECK(op);

    frame_vars_[7] = direct_w[0]; // Wx
    frame_vars_[8] = direct_w[1]; // Wy
    frame_vars_[9] = direct_w[2]; // Wz
}

void BundleAdjustmentKanatani::FrameFromOptVarsUpdater::AddDelta(size_t var_ind, Scalar value)
{
    frame_vars_[var_ind] += value;

    // update dependent 'expanded' structues

    if (var_ind < kIntrinsicVarsCount)
    {
        switch (var_ind)
        {
        case 0:
        case 1:
            cam_intrinsics_mat_(var_ind, var_ind) += value;
            break;
        case 2:
        case 3:
            {
                size_t row = var_ind - kFxFyCount; // count([fx,fy])=2
                cam_intrinsics_mat_(row, 2) += value;
                break;
            }
        default: AssertFalse();
        }
    }
    else if (var_ind < kIntrinsicVarsCount + kTVarsCount)
    {
        size_t i = var_ind - kIntrinsicVarsCount;

        // updating translation in direct camera leads to corresponding changes in both R and T in inverse camera
        if (direct_orient_cam_valid_)
        {
            // update direct camera orientation if it is up to date
            direct_orient_cam_.T[i] += value;
        }
        inverse_orient_cam_valid_ = false;
    }
    else
    {
        SRK_ASSERT(var_ind < kIntrinsicVarsCount + kTVarsCount + kWVarsCount);
        // updating rotation in the form of axis angle invalidates a direct (and inverse) camera orientation 
        direct_orient_cam_valid_ = false;
        inverse_orient_cam_valid_ = false;
    }
}

void BundleAdjustmentKanatani::FrameFromOptVarsUpdater::EnsureCameraOrientaionValid()
{
    if (!direct_orient_cam_valid_)
    {
        direct_orient_cam_.T[0] = frame_vars_[4]; // Tx
        direct_orient_cam_.T[1] = frame_vars_[5]; // Ty
        direct_orient_cam_.T[2] = frame_vars_[6]; // Tz

        Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> direct_w_mat(&frame_vars_[kIntrinsicVarsCount + kTVarsCount]);
        bool op = RotMatFromAxisAngle(direct_w_mat, &direct_orient_cam_.R);
        CHECK(op);
        direct_orient_cam_valid_ = true;
    }
    if (!inverse_orient_cam_valid_)
    {
        inverse_orient_cam_ = SE3Inv(direct_orient_cam_);
        inverse_orient_cam_valid_ = true;
    }
}
}
