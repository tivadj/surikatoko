#pragma once
#include <set>
#include <vector>
#include <memory>
#include <shared_mutex>
#include <functional>
#include "suriko/obs-geom.h"

namespace suriko {
class CornersMatcherBase
{
public:
    virtual void DetectAndMatchCorners(size_t frame_ind, CornerTrackRepository* track_rep) = 0;

    virtual ~CornersMatcherBase() = default;
};

class MultiViewIterativeFactorizer
{
    static constexpr Scalar kF0 = 1; // numerical stability factor to equalize image width, height and 1 (homogeneous component)
public:
    // shared
    std::shared_mutex location_and_map_mutex_;
    FragmentMap map_;
    std::vector<SE3Transform> cam_orient_cfw_; // orientations of cameras, transforming into camera from world (cfw)
public:
    std::unique_ptr<CornersMatcherBase> corners_matcher_;
    CornerTrackRepository track_rep_;
public:
    std::function<SE3Transform(size_t, size_t)> gt_cam_orient_f1f2_;
    std::function<SE3Transform(size_t)> gt_cam_orient_world_to_f_;
    std::function <Point3(size_t)> gt_salient_point_by_virtual_point_id_fun_;
    Eigen::Matrix<Scalar, 3, 3> K_;
    Eigen::Matrix<Scalar, 3, 3> K_inv_;
    SE3Transform tmp_cam_new_from_anchor_;
    bool fake_localization_ = false; // true to get camera orientation from ground truth
    bool fake_mapping_ = false;  // true to get salient 3D points from ground truth
public:
    MultiViewIterativeFactorizer();

    bool IntegrateNewFrameCorners(const SE3Transform& gt_cam_orient_cfw);

    void LogReprojError() const;

    void SetCornersMatcher(std::unique_ptr<CornersMatcherBase> corners_matcher);

private:
    // Counts the number of common points between two frames.
    size_t CountCommonPoints(size_t a_frame_ind, const std::set<size_t>& a_frame_track_ids, size_t b_frame_ind,
        std::vector<size_t>* common_point_ids = nullptr) const;
    
    bool FindRelativeMotionMultiPoints(size_t anchor_frame_ind, size_t target_frame_ind, const std::vector<size_t>& common_track_ids,
        const std::vector<Scalar>& pnt_depthes_anchor, SE3Transform* cam_frame_from_anchor) const;

    SE3Transform GetFrameRelativeRTFromAnchor(size_t anchor_frame_ind, size_t target_frame_ind, const std::vector<size_t>& common_track_ids, const std::vector<Scalar>& pnt_depthes_anchor) const;

    struct PointInFrameInfo
    {
        size_t FrameInd;
        Eigen::Matrix<Scalar, 3, 1> CoordMeter;
        SE3Transform FrameFromBase;
    };

    size_t CollectFrameInfoListForPoint(size_t track_id, std::vector<PointInFrameInfo>* pnt_ids);
    Scalar Estimate3DPointDepthFromFrames(const std::vector<PointInFrameInfo>& pnt_per_frame_infos);
    
    Scalar Get3DPointDepth(size_t track_id, size_t base_frame_ind) const;
    size_t FindAnchorFrame(size_t targ_frame_ind, std::vector<size_t>* common_track_ids) const;
    size_t FramesCount() const;

    static bool ReprojError(Scalar f0, 
        const FragmentMap& map,
        const std::vector<SE3Transform>& cam_orient_cfw,
        const CornerTrackRepository& track_rep,
        const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat,
        Scalar* reproj_err);
};
}
