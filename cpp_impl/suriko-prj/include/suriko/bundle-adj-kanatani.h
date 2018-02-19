﻿#pragma once
#include <string>
#include <array>
#include <vector>
#include <optional>
#include <cmath> // std::isnan
#include <iostream>
#include <Eigen/Dense>
#include "suriko/obs-geom.h"

namespace suriko
{

// forward declaration for friend
//template <typename Scalar> class SceneNormalizer;

auto NormalizeSceneInplace(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams,
                                   Scalar t1y_dist, size_t unity_comp_ind, bool* success);

/// Performs normalization of world points , so that (R0,T0) is the identity rotation plus zero translation and T1y=1.
    /// Changes the and camera positions/orientations so that
class SceneNormalizer
{
    FragmentMap* map_ = nullptr;
    std::vector<SE3Transform>* inverse_orient_cams_ = nullptr;
    Scalar normalized_t1y_dist_ = Scalar(); // expected T1y (or T1x) distance after normalization
    int unity_comp_ind_ = 1; // index of 3-element T1(x,y,z) to normalize (0 to use T1x; 1 to use T1y)

    // store pre-normalized state
    SE3Transform prenorm_rt0_;
    Scalar world_scale_ = 0; // world, scaled with this multiplier, is transformed into normalized world

    friend auto NormalizeSceneInplace(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams,
                                              Scalar t1y_norm, size_t unity_comp_ind, bool* success);

    SceneNormalizer(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams, Scalar t1y, size_t unity_comp_ind);

    enum class NormalizeAction
    {
        Normalize,
        Revert
    };

    static auto Opposite(NormalizeAction action);

    static SE3Transform NormalizeOrRevertRT(const SE3Transform& inverse_orient_camk,
        const SE3Transform& inverse_orient_cam0, Scalar world_scale, NormalizeAction action, bool check_back_conv=true);

    // TODO: back conversion check can be moved to unit testing
    static suriko::Point3 NormalizeOrRevertPoint(const suriko::Point3& x3D,
        const SE3Transform& inverse_orient_cam0, Scalar world_scale, NormalizeAction action, bool check_back_conv=true);

    /// Modify structure so that it becomes 'normalized'.
    /// The structure is updated in-place, because a copy of salient points and orientations of a camera can be too expensive to make.
    bool NormalizeWorldInplaceInternal();
public:
    SceneNormalizer() = default; // TODO: fragile

    void RevertNormalization();
};

/// Normalizes the salient points and orientations of the camera so that R0=Identity, T0=zeros(3), T1[unity_comp_ind]=t1y_dist.
/// Usually unity_comp_ind=1 (that is, y-component of T1) and t1y_dist=1 (unity).
auto NormalizeSceneInplace(FragmentMap* map, std::vector<SE3Transform>* inverse_orient_cams,
        Scalar t1y_dist, size_t unity_comp_ind, bool* success);

bool CheckWorldIsNormalized(const std::vector<SE3Transform>& inverse_orient_cams, Scalar t1y, int unity_comp_ind,
    std::string* err_msg = nullptr);

/// Performs Bundle adjustment (BA) inplace. Iteratively shifts world points and cameras position and orientation so
/// that the reprojection error is minimized.
/// TODO: think about it: the synthetic scene, corrupted with noise, probably will not be 'repaired' (adjusted) to zero reprojection error.
/// source: "Bundle adjustment for 3-d reconstruction" Kanatani Sugaya 2010
class BundleAdjustmentKanatani
{
    static const size_t kPointVarsCount = 3; // number of variables in 3D point [X,Y,Z]

    static const size_t kIntrinsicVarsCount = 4; // count({fx,fy,u0,v0})=4
    static const size_t kFxFyCount = 2; // count({fx, fy})=2
    static const size_t kU0V0Count = 2; // count({u0, v0})=2
    static const size_t kTVarsCount = 3; // count({directTx, directTy, directTz})=3
    static const size_t PqrCount = 3; // count({p,q,r})=3, pqr=P*[X,Y,Z,1]
    static const size_t kPCompInd = 0;
    static const size_t kQCompInd = 1;
    static const size_t kRCompInd = 2;
    //static const size_t kWVarsCount = internals::kWVarsCount;
public:
    static const size_t kWVarsCount = 3; // count({directWx, directWy, directWz})=3
private:
    // maximum count of ([fx fy u0 v0 Tx Ty Tz Wx Wy Wz])
    // camera intrinsics: 0 if K is shared for all frames; 3 for [f u0 v0] if fx=fy; 4 for [fx fy u0 v0]
    // hence max count of camera intrinsics variables is 4
    // direct mode camera translation: 3 for [directTx directTy directTz]
    // direct mode camera axis angle: 3 for [directWx directWy directWz]
    static const size_t kMaxFrameVarsCount = 10;

private:
    FragmentMap* map_ = nullptr;
    std::vector<SE3Transform>* inverse_orient_cams_ = nullptr;
    const CornerTrackRepository* track_rep_ = nullptr;
    const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat_ = nullptr;
    std::vector<Eigen::Matrix<Scalar, 3, 3>> * intrinsic_cam_mats_ = nullptr;
    SceneNormalizer scene_normalizer_;
    
    Scalar t1y_ = 1.0; // const, y-component of the first camera shift, usually T1y==1
    size_t unity_comp_ind_ = 1; // 0 for X, 1 for Y; index of T1 to be set to unity
    Scalar min_err_change_abs_ = 1e-8;
    Scalar max_hessian_factor_ = 1e6; // set None to skip the check of the hessian's factor

    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> EigenDynMat;

    // cache gradients
    std::vector<Scalar> gradE_finite_diff;
    EigenDynMat deriv_second_point_finite_diff;
    EigenDynMat deriv_second_frame_finite_diff;
    EigenDynMat deriv_second_pointframe_finite_diff;
    
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> corrections_;

    // data to solve linear equations
    EigenDynMat matG_;
    EigenDynMat decomp_lin_sys_left_side1_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> decomp_lin_sys_right_side_;

    size_t frame_vars_count_ = 0; // number of variables to parameterize a camera orientation [[fx fy u0 v0] Tx Ty Tz Wx Wy Wz]
    
    // some corrections are fixed during normalization, thus the gaps appear :
    // T0 = [0 0 0] R0 = Identity, T1y(or T1x) = 1 determined by @unity_comp_ind
    // corrections plain : [point_corrections 0fx 0fy 0u0 0v0 0Tx 0Ty 0Tz 0Wx 0Wy 0Wz 1fx 1fy 1u0 1v0 1Tx 1Ty 1Tz 1Wx 1Wy 1Wz]
    // corrections wgaps : [point_corrections 0fx 0fy 0u0 0v0                         1fx 1fy 1u0 1v0 1Tx     1Tz 1Wx 1Wy 1Wz]
    std::array<size_t, 7> normalize_pattern_; // count({T0x,T0y,T0z,W0x,W0y,W0z,T1y})=7
    size_t normalize_pattern_count_ = 0; // number of valid indices of variables
    size_t normalized_vars_count_ = 0; // 3*points_count+10*frames_count-normalize_pattern_count_

    // True to compare close derivatives of reprojection error with finite difference approximation.
    // The computation of finite differences is slow and this value should be false in normal debug and release modes.
    // TODO: figure out how to optimize away in runtime
    bool debug_reproj_error_first_derivatives_ = false;
    bool debug_reproj_error_derivatives_pointpoint_ = false;
    bool debug_reproj_error_derivatives_frameframe_ = false;
    bool debug_reproj_error_derivatives_pointframe_ = false;
public:
    static Scalar ReprojError(const FragmentMap& map,
                            const std::vector<SE3Transform>& inverse_orient_cams,
                            const CornerTrackRepository& track_rep,
                            const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat = nullptr,
                            const std::vector<Eigen::Matrix<Scalar, 3, 3>>* intrinsic_cam_mats = nullptr);

    /// :return: True if optimization converges successfully.
    /// Stop conditions:
    /// 1) If a change of error function slows down and becomes less than self.min_err_change
    bool ComputeInplace(FragmentMap& map,
                        std::vector<SE3Transform>& inverse_orient_cams,
                        const CornerTrackRepository& track_rep,
                        const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat = nullptr,
                        std::vector<Eigen::Matrix<Scalar, 3, 3>>* intrinsic_cam_mats = nullptr);

private:
    void UpdateNormalizePattern();

    bool ComputeOnNormalizedWorld();

    auto GetFiniteDiffFirstPartialDerivPoint(size_t point_track_id, const suriko::Point3& pnt3D_world, size_t var1, Scalar finite_diff_eps) const -> Scalar;
    auto GetFiniteDiffFirstPartialDerivFocalLengthFxFy(size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& K, size_t fxfy_ind, Scalar finite_diff_eps) const -> Scalar;
    auto GetFiniteDiffFirstPartialDerivPrincipalPoint(size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& K, size_t u0v0_ind, Scalar finite_diff_eps) const -> Scalar;
    auto GetFiniteDiffFirstPartialDerivTranslationDirect(size_t frame_ind, const SE3Transform& direct_orient_cam, size_t tind, Scalar finite_diff_eps) const -> Scalar;
    auto GetFiniteDiffFirstPartialDerivRotation(size_t frame_ind, const SE3Transform& direct_orient_cam, size_t wind, Scalar finite_diff_eps) const->Scalar;
    
    auto GetFiniteDiffSecondPartialDerivPointPoint(size_t point_track_id, const suriko::Point3& pnt3D_world, size_t var1, size_t var2, Scalar finite_diff_eps) const->Scalar;
    
    auto GetFiniteDiffSecondPartialDerivFrameFrame(size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& cam_intrinsics_mat,
        const SE3Transform& direct_orient_cam, size_t frame_var_ind1, size_t frame_var_ind2, Scalar finite_diff_eps) const->Scalar;
    
    auto GetFiniteDiffSecondPartialDerivPointFrame(
        size_t point_track_id, const suriko::Point3& pnt3D_world, size_t point_var_ind,
        size_t frame_ind, const Eigen::Matrix<Scalar, 3, 3>& cam_intrinsics_mat,
        const SE3Transform& direct_orient_cam, size_t frame_var_ind,
        Scalar finite_diff_eps) const->Scalar;
    //
    void ComputeCloseFormReprErrorDerivatives(std::vector<Scalar>* grad_error,
        EigenDynMat* deriv_second_pointpoint,
        EigenDynMat* deriv_second_frameframe,
        EigenDynMat* deriv_second_pointframe,
        [[maybe_unused]] Scalar finite_diff_eps);

    /// Each row contains(p, q, r) derivatives for X(row = 0), Y, Z variables
    void ComputePointPqrDerivatives(const Eigen::Matrix<Scalar, 3, 4>& P, Eigen::Matrix<Scalar, 3, 3>* point_pqr_deriv) const;

    void ComputeFramePqrDerivatives(const Eigen::Matrix<Scalar, 3, 3>& K, const SE3Transform& inverse_orient_cam, 
        const suriko::Point3& salient_point, const suriko::Point2& corner_pix,
        Eigen::Matrix<Scalar, kMaxFrameVarsCount, PqrCount>* frame_pqr_deriv, gsl::not_null<size_t*> out_frame_vars_count) const;

    Scalar FirstDerivFromPqrDerivative(Scalar f0, const Eigen::Matrix<Scalar, 3, 1>& pqr, const suriko::Point2& corner_pix,
        Scalar gradp_byvar, Scalar gradq_byvar, Scalar gradr_byvar) const;

    Scalar SecondDerivFromPqrDerivative(const Eigen::Matrix<Scalar, 3, 1>& pqr,
        Scalar gradp_byvar1, Scalar gradq_byvar1, Scalar gradr_byvar1,
        Scalar gradp_byvar2, Scalar gradq_byvar2, Scalar gradr_byvar2) const;

    void EstimateCorrectionsNaive(const std::vector<Scalar>& grad_error, const EigenDynMat& deriv_second_pointpoint,
        const EigenDynMat& deriv_second_frameframe,
        const EigenDynMat& deriv_second_pointframe, Scalar hessian_factor, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* corrections_with_gaps);

    void FillHessian(const EigenDynMat& deriv_second_pointpoint,
        const EigenDynMat& deriv_second_frameframe,
        const EigenDynMat& deriv_second_pointframe, Scalar hessian_factor, EigenDynMat* hessian);

    void FillCorrectionsGapsFromNormalized(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& normalized_corrections, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* corrections_with_gaps);

    void EstimateCorrectionsDecomposedInTwoPhases(const std::vector<Scalar>& grad_error, const EigenDynMat& deriv_second_pointpoint,
        const EigenDynMat& deriv_second_frameframe,
        const EigenDynMat& deriv_second_pointframe, Scalar hessian_factor, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* corrections_with_gaps);

    void ApplyCorrections(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& corrections_with_gaps);
};
}
