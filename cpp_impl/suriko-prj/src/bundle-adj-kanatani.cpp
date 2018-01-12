#include <string>
#include <array>
#include <vector>
#include <optional>
#include <cmath> // std::isnan
#include <iostream>
#include <Eigen/Dense>
#include "suriko/approx-alg.h"
#include "suriko/bundle-adj-kanatani.h"
#include "suriko/obs-geom.h"

namespace suriko
{

SceneNormalizer::SceneNormalizer(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams, Scalar t1y, Scalar unity_comp_ind)
        :map_(map),
         inverse_orient_cams_(inverse_orient_cams),
         normalized_t1y_dist_(t1y),
         unity_comp_ind_(unity_comp_ind)
{
    assert(t1y == 0 || t1y == 1 && "Only T1x and T1y is implemented");
}

auto SceneNormalizer::Opposite(SceneNormalizer::NormalizeAction action) {
    switch(action)
    {
        case NormalizeAction::Normalize:
            return NormalizeAction::Revert;
        case NormalizeAction::Revert:
            return NormalizeAction::Normalize;
    }
}

SE3Transform SceneNormalizer::NormalizeOrRevertRT(const SE3Transform& inverse_orient_camk,
                                        const SE3Transform& inverse_orient_cam0, Scalar world_scale, NormalizeAction action, bool check_back_conv)
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
suriko::Point3 SceneNormalizer::NormalizeOrRevertPoint(const suriko::Point3& x3D,
                                             const SE3Transform& inverse_orient_cam0, Scalar world_scale, NormalizeAction action, bool check_back_conv)
{
    suriko::Point3 result(0,0,0);
    if (action == NormalizeAction::Normalize)
    {
        // RT for frame0 transform 3D point from world into coordinates of first camera
        result = SE3Apply(inverse_orient_cam0, x3D); // = R0*X + T0
        result.Mat() *= world_scale;
    } else if (action == NormalizeAction::Revert)
    {
        Eigen::Matrix<Scalar,3,1> X3Dtmp = x3D.Mat() * (1/ world_scale);
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

auto NormalizeSceneInplace(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams,
                           Scalar t1y_dist, int unity_comp_ind, bool* success)
{
    *success = false;

    auto scene_normalizer = SceneNormalizer(map, inverse_orient_cams, t1y_dist, unity_comp_ind);
    if (scene_normalizer.NormalizeWorldInplaceInternal()) {
        *success = true;
    }
    return scene_normalizer;
}

bool CheckWorldIsNormalized(const std::vector<SE3Transform>& inverse_orient_cams, Scalar t1y, int unity_comp_ind,
                            std::string* err_msg)
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

/// Performs Bundle adjustment (BA) inplace. Iteratively shifts world points and cameras position and orientation so
/// that the reprojection error is minimized.
/// TODO: think about it: the synthetic scene, corrupted with noise, probably will not be 'repaired' (adjusted) to zero reprojection error.
/// source: "Bundle adjustment for 3-d reconstruction" Kanatani Sugaya 2010
bool BundleAdjustmentKanatani::ReprojError(const FragmentMap& map,
                        const std::vector<SE3Transform>& inverse_orient_cams,
                        const CornerTrackRepository& track_rep,
                        const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat,
                        const std::vector<Eigen::Matrix<Scalar, 3, 3>>* intrinsic_cam_mats)
{
    assert(intrinsic_cam_mats != nullptr);

    size_t points_count = map.PointTrackCount();
    assert(points_count == track_rep.CornerTracks.size() && "Each 3D point must be tracked");

    Scalar err_sum = 0;

    size_t frames_count = inverse_orient_cams.size();
    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        const SE3Transform& inverse_orient_cam = inverse_orient_cams[frame_ind];
        const Eigen::Matrix<Scalar, 3, 3>& K = (*intrinsic_cam_mats)[frame_ind];

        for(const CornerTrack& point_track : track_rep.CornerTracks)
        {
            std::optional<suriko::Point2> corner = point_track.GetCorner(frame_ind);
            if (!corner.has_value())
                continue;

            suriko::Point2 corner_pix = corner.value();

            suriko::Point3 x3D = map.GetSalientPoint(point_track.TrackId);

            suriko::Point3 x3D_cam = SE3Apply(inverse_orient_cam, x3D);
            suriko::Point3 x3D_pix = suriko::Point3(K * x3D_cam.Mat()); // TODO: replace Point3 ctr with ToPoint factory method, error: call to 'ToPoint' is ambiguous

            bool zero_z = IsClose(0, x3D_pix[2], 1e-5);
            SRK_ASSERT(!zero_z && "homog 2D point can't have Z=0");

            Scalar x = x3D_pix[0] / x3D_pix[2];
            Scalar y = x3D_pix[1] / x3D_pix[2];

            Scalar one_err = Sqr(x - corner_pix[0]) + Sqr(y - corner_pix[1]);
            if (one_err > 10)
            {
                std::cout <<" pnt_track_id" <<point_track.TrackId;
            }
            std::cout <<"err_sum=" <<err_sum <<std::endl;
            SRK_ASSERT(std::isfinite(one_err));

            err_sum += one_err;
        }
        std::cout <<err_sum <<std::endl;
    }
    SRK_ASSERT(std::isfinite(err_sum));
    return err_sum;
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

    bool result = ComputeOnNormalizedWorld();

    if (kSurikoDebug)
    {
        // check world is still normalized after optimization
        std::string err_msg;
        if (!CheckWorldIsNormalized(inverse_orient_cams, t1y_, unity_comp_ind_, &err_msg))
        {
            std::cout <<err_msg;
            SRK_ASSERT(false);
        }
    }
    scene_normalizer_.RevertNormalization();

    return true;
}

bool BundleAdjustmentKanatani::ComputeOnNormalizedWorld()
{
    return true;
}
}
