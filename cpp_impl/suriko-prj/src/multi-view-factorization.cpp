#include "suriko/multi-view-factorization.h"
#include "suriko/approx-alg.h"

namespace suriko
{

void PopulateCornerTrackIds(const CornerTrackRepository& track_rep, size_t frame_ind, std::set<size_t>* track_ids)
{
    for (const CornerTrack& corner_track : track_rep.CornerTracks)
    {
        std::optional<Point2> corner = corner_track.GetCorner(frame_ind);
        if (corner.has_value())
            track_ids->insert(corner_track.TrackId);
    }
}

MultiViewIterativeFactorizer::MultiViewIterativeFactorizer()
{
    corners_matcher_ = nullptr;
}

size_t MultiViewIterativeFactorizer::CountCommonPoints(size_t a_frame_ind, const std::set<size_t>& a_frame_track_ids, size_t b_frame_ind,
    std::vector<size_t>* common_point_ids) const
{
    size_t result = 0;
    for (size_t track_id : a_frame_track_ids)
    {
        const CornerTrack& corner_track = track_rep_.GetPointTrackById(track_id);

        std::optional<Point2> corner = corner_track.GetCorner(b_frame_ind);
        if (!corner.has_value())
            continue;

        // require for corner to have candidate 3D point, so that further 3D coordinate can be used in reconstruction
        if (!corner_track.SalientPointId.has_value())
            continue;

        result += 1;
        if (common_point_ids != nullptr) common_point_ids->push_back(track_id);
    }
    return result;
}

size_t MultiViewIterativeFactorizer::FindAnchorFrame(size_t targ_frame_ind, std::vector<size_t>* common_track_ids) const
{
    std::set<size_t> targ_frame_track_ids;
    PopulateCornerTrackIds(track_rep_, targ_frame_ind, &targ_frame_track_ids);

    std::vector<size_t> common_points_per_frame(targ_frame_ind);
    for (size_t prev_frame_ind = 0; prev_frame_ind < targ_frame_ind; ++prev_frame_ind)
    {
        common_points_per_frame[prev_frame_ind] = CountCommonPoints(targ_frame_ind, targ_frame_track_ids, prev_frame_ind);
    }

    auto max_it = std::max_element(common_points_per_frame.begin(), common_points_per_frame.end());
    size_t anchor_frame_ind = std::distance(common_points_per_frame.begin(), max_it);
    size_t common_points_count = common_points_per_frame[anchor_frame_ind];

    // get the actual ids of common points
    CountCommonPoints(targ_frame_ind, targ_frame_track_ids, anchor_frame_ind, common_track_ids);
    SRK_ASSERT(common_points_count == common_track_ids->size());

    return anchor_frame_ind;
}

// gets the depth of a point in the given frame
Scalar MultiViewIterativeFactorizer::Get3DPointDepth(size_t track_id, size_t anchor_frame_ind) const
{
    const SE3Transform& anchor_frame_cfw = cam_orient_cfw_[anchor_frame_ind];

    const CornerTrack& corner_track = track_rep_.GetPointTrackById(track_id);
    SRK_ASSERT(corner_track.SalientPointId.has_value()) << "salient point is not reconstructed yet for corner_track=" << corner_track.TrackId;

    size_t salient_point_id = corner_track.SalientPointId.value();
    suriko::Point3 x3D = map_.GetSalientPoint(salient_point_id);

    suriko::Point3 x3D_anchor = SE3Apply(anchor_frame_cfw, x3D);
    return x3D_anchor[2];
}

/// Projects camera movement with noisy R, so that R becomes valid special orthogonal matrix.
bool ProjectOntoSO3(const SE3Transform& noisy_RT, SE3Transform* valid_RT)
{
    // project noisy[R, T] onto SO(3) (see MASKS, formula 8.41 and 8.42)
    Eigen::JacobiSVD<Eigen::Matrix<Scalar, 3, 3>> R_noisy_svd = noisy_RT.R.jacobiSvd().compute(noisy_RT.R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Scalar det_S = R_noisy_svd.singularValues().prod();

    Eigen::Matrix<Scalar, 3, 3> S33 = R_noisy_svd.singularValues().asDiagonal();
    Scalar det_S2 = S33.determinant();

    if (IsClose(0, det_S))
        return false;

    Eigen::Matrix<Scalar, 3, 3> no_guts = R_noisy_svd.matrixU() * R_noisy_svd.matrixV().transpose();
    
    Scalar no_guts_det = no_guts.determinant();
    int sign = Sign(no_guts_det);
    valid_RT->R = sign * no_guts;
    
    Scalar s = sign / std::cbrt(det_S);
    valid_RT->T = s * noisy_RT.T;
    
    if (kSurikoDebug)
    {
        std::string msg;
        SRK_ASSERT(IsSpecialOrthogonal(valid_RT->R, &msg)) << msg;
    }
    return true;
}

bool MultiViewIterativeFactorizer::FindRelativeMotionMultiPoints(size_t anchor_frame_ind, size_t target_frame_ind, const std::vector<size_t>& common_track_ids,
    const std::vector<Scalar>& pnt_depthes_anchor, SE3Transform* cam_frame_from_anchor) const
{
    size_t points_count = pnt_depthes_anchor.size();
    SRK_ASSERT(common_track_ids.size() == points_count);
    
    Eigen::Matrix<Scalar, 12, 1> r_and_t_ok;
    r_and_t_ok.block(0, 0, 9, 1) = Eigen::Map<const Eigen::Matrix<Scalar, 9, 1>, Eigen::ColMajor>(this->tmp_cam_new_from_anchor_.R.data());
    r_and_t_ok.block(9, 0, 3, 1) = Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>>(this->tmp_cam_new_from_anchor_.T.data());

    // estimage camera position[R, T] given distances to all 3D points pj in frame1
    //Eigen::Matrix<Scalar, Eigen::Dynamic, 12> A;
    //A.resize(points_count * 3, Eigen::NoChange);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> A; // thin svd is available for matrices with dynamic columns
    A.resize(points_count * 3, 12);
    static bool norm = false;
    for (size_t i = 0; i < common_track_ids.size(); ++i)
    {
        size_t track_id = common_track_ids[i];
        
        const CornerTrack& corner_track = track_rep_.GetPointTrackById(track_id);
        Eigen::Matrix<Scalar, 3, 1> c1 = corner_track.GetCornerData(anchor_frame_ind).value().ImageCoord;
        Eigen::Matrix<Scalar, 3, 1> c2 = corner_track.GetCornerData(target_frame_ind).value().ImageCoord;

        Eigen::Matrix<Scalar, 3, 1> c2_homog = c2;
        if (norm)
            c2_homog /= c2_homog.norm();

        Eigen::Matrix<Scalar, 3, 3> c2_skew;
        SkewSymmetricMat(c2_homog, &c2_skew);

        // manual Kronecker product
        Eigen::Matrix<Scalar, 3, 1> c1_homog = c1;
        if (norm)
            c1_homog /= c1_homog.norm();
        for (size_t c1_comp_ind = 0; c1_comp_ind < 3; ++c1_comp_ind)
        {
            A.block<3, 3>(i * 3, c1_comp_ind * 3) = c1_homog[c1_comp_ind] * c2_skew;
        }
        Scalar depth_in_anchor = pnt_depthes_anchor[i];
        Scalar alpha = 1 / depth_in_anchor;
        A.block<3, 3>(i * 3, 9) = alpha * c2_skew;

        //
        Eigen::Matrix<Scalar, 3, 12> Aline;
        for (size_t c1_comp_ind = 0; c1_comp_ind < 3; ++c1_comp_ind)
        {
            Aline.block<3, 3>(0, c1_comp_ind * 3) = c1_homog[c1_comp_ind] * c2_skew;
        }
        Aline.block<3, 3>(0, 9) = alpha * c2_skew;
        Eigen::Matrix<Scalar, 3, 1> theright = Aline * r_and_t_ok;
        CHECK(true);
    }

    Eigen::JacobiSVD<decltype(A)> A_svd = A.jacobiSvd().compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix<Scalar, 12, 1> r_and_t = A_svd.matrixV().rightCols<1>();

    Scalar d = (A * r_and_t).norm();

    Eigen::JacobiSVD<decltype(A)> A_svdF = A.jacobiSvd().compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<Scalar, 12, 1> r_and_tF = A_svdF.matrixV().rightCols<1>();

    Scalar dF = (A * r_and_tF).norm();

    Eigen::Map<Eigen::Matrix<Scalar, 3, 3>, Eigen::ColMajor> R_noisy(r_and_t.data());
    Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> T_noisy(&r_and_t[9]);

    bool op = ProjectOntoSO3(SE3Transform(R_noisy, T_noisy), cam_frame_from_anchor);
    return op;
}

SE3Transform MultiViewIterativeFactorizer::GetFrameRelativeRTFromAnchor(size_t anchor_frame_ind, size_t target_frame_ind, const std::vector<size_t>& common_track_ids, const std::vector<Scalar>& pnt_depthes_anchor) const
{
    // motion estimation step
    SE3Transform cam_frame_from_anchor{};
    bool suc = FindRelativeMotionMultiPoints(anchor_frame_ind, target_frame_ind, common_track_ids, pnt_depthes_anchor, &cam_frame_from_anchor);
    SRK_ASSERT(suc);

    return cam_frame_from_anchor;
}

size_t MultiViewIterativeFactorizer::CollectFrameInfoListForPoint(size_t track_id, std::vector<PointInFrameInfo>* pnt_per_frame_infos)
{
    const CornerTrack& track = track_rep_.GetPointTrackById(track_id);
    std::optional<size_t> base_frame_ind;
    track.EachCorner([this, &base_frame_ind, pnt_per_frame_infos](size_t frame_ind, const std::optional<CornerData>& corner_data)
    {
        if (!corner_data.has_value())
            return;
        const Eigen::Matrix<Scalar, 3, 1>& x_meter = corner_data.value().ImageCoord;

        if (!base_frame_ind.has_value())
            base_frame_ind = frame_ind;

        const SE3Transform& base_from_world = cam_orient_cfw_[base_frame_ind.value()];
        const SE3Transform& framei_from_world = cam_orient_cfw_[frame_ind];
        SE3Transform framei_from_base = SE3AFromB(framei_from_world, base_from_world);

        pnt_per_frame_infos->push_back(PointInFrameInfo{ frame_ind, x_meter, framei_from_base });
    });
    return base_frame_ind.value();
}

Scalar MultiViewIterativeFactorizer::Estimate3DPointDepthFromFrames(const std::vector<PointInFrameInfo>& pnt_per_frame_infos)
{
    // find distances to all 3D points in frame1 given position[Ri, Ti] of each frame
    // (MASKS formula 8.44)
    Scalar alpha_num = 0;
    Scalar alpha_den = 0;

    size_t base_frame_ind = pnt_per_frame_infos[0].FrameInd;
    Eigen::Matrix<Scalar, 3, 1> x1 = pnt_per_frame_infos[0].CoordMeter;

    for (size_t i = 1; i < pnt_per_frame_infos.size(); ++i)
    {
        const PointInFrameInfo& info = pnt_per_frame_infos[i];

        Eigen::Matrix<Scalar, 3, 1> xi = info.CoordMeter;
        Eigen::Matrix<Scalar, 3, 3> xi_skew;
        SkewSymmetricMat(xi, &xi_skew);

        SE3Transform framei_from_base = info.FrameFromBase;

        Eigen::Matrix<Scalar, 3, 1> h1 = xi_skew * framei_from_base.T;
        Eigen::Matrix<Scalar, 3, 1> h2 = xi_skew * framei_from_base.R * x1;

        alpha_num += h1.dot(h2);
        alpha_den += h1.squaredNorm();
    }

    Scalar alpha = -alpha_num / alpha_den;
    Scalar dist = 1 / alpha;
    return dist;
}

void MultiViewIterativeFactorizer::IntegrateNewFrameCorners(const SE3Transform& gt_cam_orient_cfw)
{
    size_t new_frame_ind = FramesCount();

    this->corners_matcher_->DetectAndMatchCorners(new_frame_ind, &track_rep_);

    std::vector<size_t> common_point_ids;
    size_t anchor_frame_ind = FindAnchorFrame(new_frame_ind, &common_point_ids);

    VLOG(4) << "f=" << new_frame_ind << " anchored on " << anchor_frame_ind << " based on common_points=" << common_point_ids.size();

    // find depthes of the salient points in the anchor frame
    std::vector<Scalar> pnt_depthes_anchor(common_point_ids.size());
    for (size_t i = 0; i < common_point_ids.size(); ++i)
        pnt_depthes_anchor[i] = Get3DPointDepth(common_point_ids[i], anchor_frame_ind);

    tmp_cam_new_from_anchor_ = gt_cam_orient_f1f2_(anchor_frame_ind, new_frame_ind);

    SE3Transform cam_new_from_anchor = GetFrameRelativeRTFromAnchor(anchor_frame_ind, new_frame_ind, common_point_ids, pnt_depthes_anchor);

    if (gt_cam_orient_f1f2_ != nullptr)
    {
        SE3Transform gt_cam_new_from_anchor = gt_cam_orient_f1f2_(anchor_frame_ind, new_frame_ind);
        Scalar diff_value =
            (gt_cam_new_from_anchor.R - cam_new_from_anchor.R).norm() +
            (gt_cam_new_from_anchor.T - cam_new_from_anchor.T).norm();
        if (diff_value > 1)
            VLOG(4) << "failed cam localiz frame_ind=" << new_frame_ind << " diff_value=" << diff_value;
    }
    
    const SE3Transform& anchor_from_world = cam_orient_cfw_[anchor_frame_ind];
    SE3Transform cam_new_from_world = SE3Compose(cam_new_from_anchor, anchor_from_world);
    
    if (fake_localization_)
        cam_orient_cfw_.push_back(gt_cam_orient_cfw); // ground truth
    else
        cam_orient_cfw_.push_back(cam_new_from_world); // real
    

    // find tracks for which the salient point can be reconstructed
    // it can't be new track because it has only one projection in this frame
    // it can't be deleted track because no info is added
    // hence the candidate tracks for reconstruction are in common set
    std::set<size_t> frame_track_ids;
    PopulateCornerTrackIds(track_rep_, new_frame_ind, &frame_track_ids);

    size_t reconstructed_salient_points_count = 0;

    std::vector<PointInFrameInfo> pnt_per_frame_infos;
    for (size_t track_id : frame_track_ids)
    {
        CornerTrack& track = track_rep_.GetPointTrackById(track_id);
        if (track.SalientPointId.has_value()) // already reconstructed
            continue;

        pnt_per_frame_infos.clear();
        size_t base_frame_ind = CollectFrameInfoListForPoint(track_id, &pnt_per_frame_infos);

        if (pnt_per_frame_infos.size() <= 1)
            continue;

        //
        Scalar depth_base = Estimate3DPointDepthFromFrames(pnt_per_frame_infos);

        //
        
        Eigen::Matrix<Scalar, 3, 1> x3D_base = track.GetCornerData(base_frame_ind).value().ImageCoord;
        x3D_base *= depth_base;

        const SE3Transform& base_from_world = cam_orient_cfw_[base_frame_ind];
        suriko::Point3 x3D_world = SE3Apply(SE3Inv(base_from_world), x3D_base);

        // find expected 3D point
        if (track.SyntheticVirtualPointId.has_value() && gt_salient_point_by_virtual_point_id_fun_ != nullptr)
        {
            suriko::Point3 expect_x3D_world = gt_salient_point_by_virtual_point_id_fun_(track.SyntheticVirtualPointId.value());
            Scalar diff_value = (expect_x3D_world.Mat() - x3D_world.Mat()).norm();
            if (diff_value > 1)
                VLOG(4) << "failed pnt reconstr synth_id=" << track.SyntheticVirtualPointId.value() << " diff_value=" << diff_value;

            if (fake_mapping_)
                x3D_world = expect_x3D_world; // imitate working mapping
        }

        // create new 3D salient point
        size_t salient_point_id = 0;
        SalientPointFragment& salient_point = map_.AddSalientPoint(x3D_world, &salient_point_id);
        salient_point.SyntheticVirtualPointId = track.SyntheticVirtualPointId;

        track.SalientPointId = salient_point_id;
        
        reconstructed_salient_points_count += 1;
    }
    
    LogReprojError();
}

size_t MultiViewIterativeFactorizer::FramesCount() const
{
    return cam_orient_cfw_.size();
}

void MultiViewIterativeFactorizer::SetCornersMatcher(std::unique_ptr<CornersMatcherBase> corners_matcher)
{
    corners_matcher_.swap(corners_matcher);
}

void MultiViewIterativeFactorizer::LogReprojError() const
{
    constexpr Scalar f0 = 1; // numerical stability factor to equalize image width, height and 1 (homogeneous component)
    Scalar err = ReprojError(f0, map_, cam_orient_cfw_, track_rep_, &K_);
    VLOG(4) << "ReprojError=" << err;
}

Scalar MultiViewIterativeFactorizer::ReprojError(Scalar f0,
    const FragmentMap& map,
    const std::vector<SE3Transform>& cam_orient_cfw,
    const CornerTrackRepository& track_rep,
    const Eigen::Matrix<Scalar, 3, 3>* shared_intrinsic_cam_mat)
{
    CHECK(!IsClose(0, f0)) << "f0 != 0";

    Scalar err_sum = 0;

    size_t frames_count = cam_orient_cfw.size();
    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        const SE3Transform* pInverse_orient_cam = &cam_orient_cfw[frame_ind];
        const Eigen::Matrix<Scalar, 3, 3>* pK = shared_intrinsic_cam_mat;

        for (const CornerTrack& point_track : track_rep.CornerTracks)
        {
            if (!point_track.SalientPointId.has_value()) // require 3D point for projection
                continue;

            std::optional<suriko::Point2> corner = point_track.GetCorner(frame_ind);
            if (!corner.has_value())
            {
                // the salient point is not detected in current frame and 
                // hence doesn't influence the reprojection error
                continue;
            }

            suriko::Point2 corner_pix = corner.value();
            Eigen::Matrix<Scalar, 2, 1> corner_div_f0 = corner_pix.Mat() / f0;
            suriko::Point3 x3D = map.GetSalientPoint(point_track.SalientPointId.value());

            // the evaluation below is due to BA3DRKanSug2010 formula 4
            suriko::Point3 x3D_cam = SE3Apply(*pInverse_orient_cam, x3D);
            suriko::Point3 x_img_h = suriko::Point3((*pK) * x3D_cam.Mat());
            // TODO: replace Point3 ctr with ToPoint factory method, error: call to 'ToPoint' is ambiguous

            bool zero_z = IsClose(0, x_img_h[2], 1e-5);
            SRK_ASSERT(!zero_z) << "homog 2D point can't have Z=0";

            Scalar x = x_img_h[0] / x_img_h[2];
            Scalar y = x_img_h[1] / x_img_h[2];

            // for numerical stability, the error is measured not in pixels but in pixels/f0
            Scalar one_err = Sqr(x - corner_div_f0[0]) + Sqr(y - corner_div_f0[1]);
            SRK_ASSERT(std::isfinite(one_err));

            err_sum += one_err;
        }
    }
    SRK_ASSERT(std::isfinite(err_sum));
    return err_sum;
}
}
