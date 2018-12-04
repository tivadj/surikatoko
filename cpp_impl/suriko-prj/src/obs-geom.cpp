#include <string>
#include <cmath> // std::sqrt
#include <algorithm> // std::clamp
#include <glog/logging.h>
#include <Eigen/Cholesky>
#include "suriko/approx-alg.h"
#include "suriko/obs-geom.h"

namespace suriko
{
//auto ToPoint(const Eigen::Matrix<Scalar,2,1>& m) -> suriko::Point2 { return suriko::Point2(m); }
//auto ToPoint(const Eigen::Matrix<Scalar,3,1>& m) -> suriko::Point3 { return suriko::Point3(m); }

bool operator == (const Recti& lhs, const Recti& rhs)
{
    return
        lhs.x == rhs.x &&
        lhs.y == rhs.y &&
        lhs.width == rhs.width &&
        lhs.height == rhs.height;
}

std::ostream& operator<<(std::ostream& os, const Recti& r)
{
    return os << "("
        << r.x << " "
        << r.y << " "
        << r.width << " "
        << r.height << " )";
}

std::optional<Recti> IntersectRects(const Recti& a, const Recti& b)
{
    // check no crossing in horizontal direction
    const Recti* left = &a;
    const Recti* right = &b;
    if (left->x > right->x)
        std::swap(left, right);

    if (left->Right() <= right->x)
        return std::nullopt;

    // check no crossing in vertical direction
    const Recti* top = &a;
    const Recti* bot = &b;
    if (top->y > bot->y)
        std::swap(top, bot);

    if (top->Bottom() <= bot->y)
        return std::nullopt;
    
    // now, there is certainly some non-empty crossing
    auto x1 = right->x;
    auto x2 = std::min(a.Right(), b.Right());
    auto y1 = bot->y;
    auto y2 = std::min(a.Bottom(), b.Bottom());
    return RectFromSides(x1, y1, x2, y2);
}

Recti DeflateRect(const Recti& a, int left, int top, int right, int bottom)
{
    Recti result{
        a.x + left,
        a.y + top,
        a.width - (left + right),
        a.height - (top + bottom)
    };
    return result;
}

auto SE3Inv(const SE3Transform& rt) -> SE3Transform {
    SE3Transform result;
    result.R = rt.R.transpose();
    result.T = - result.R * rt.T;
    return result;
}

auto SE2Apply(const SE2Transform& rt, const suriko::Point2& x)->suriko::Point2
{
    return suriko::Point2 { rt.R * x.Mat() + rt.T };
}

auto SE3Apply(const SE3Transform& rt, const suriko::Point3& x) -> suriko::Point3
{
    // 0-copy
    suriko::Point3 result(0,0,0);
    result.Mat() = rt.R * x.Mat() + rt.T;
    return result;
    // 1-copy
//    Eigen::Matrix<Scalar,3,1> result= rt.R * x.Mat() + rt.T;
//    return ToPoint(result);
}

auto SE3Compose(const SE3Transform& rt1, const SE3Transform& rt2) -> suriko::SE3Transform
{
    SE3Transform result;
    result.R = rt1.R * rt2.R;
    result.T = rt1.R * rt2.T + rt1.T;
    return result;
}

auto SE3AFromB(const SE3Transform& a_from_world, const SE3Transform& b_from_world) -> suriko::SE3Transform
{
    return SE3Compose(a_from_world, SE3Inv(b_from_world));
}

FragmentMap::FragmentMap(size_t fragment_id_offset)
    : fragment_id_offset_(fragment_id_offset),
    next_salient_point_id_(fragment_id_offset + 1)
{
}

SalientPointFragment& FragmentMap::AddSalientPoint(const std::optional<suriko::Point3> &coord, size_t* salient_point_id)
{
    size_t new_id = next_salient_point_id_++;

    size_t new_ind1 = salient_points_.size();
    size_t new_ind2 = new_id - fragment_id_offset_ - 1;
    SRK_ASSERT(new_ind1 == new_ind2);

    if (salient_point_id != nullptr)
        *salient_point_id = new_id;

    salient_points_.resize(salient_points_.size() + 1);
    
    SalientPointFragment& frag = salient_points_.back();
    frag.coord = coord;
    return frag;
}

void FragmentMap::SetSalientPoint(size_t point_track_id, const suriko::Point3 &coord)
{
    SRK_ASSERT(point_track_id < salient_points_.size());
    SalientPointFragment& frag = salient_points_[point_track_id];
    frag.coord = coord;
}

void FragmentMap::SetSalientPointNew(size_t fragment_id, const std::optional<suriko::Point3> &coord, std::optional<size_t> syntheticVirtualPointId)
{
    SRK_ASSERT(fragment_id < salient_points_.size());
    SalientPointFragment& frag = salient_points_[fragment_id];
    frag.synthetic_virtual_point_id = syntheticVirtualPointId;
    frag.coord = coord;
}

const SalientPointFragment& FragmentMap::GetSalientPointNew(size_t salient_point_id) const
{
    size_t ind = SalientPointIdToInd(salient_point_id);
    CHECK(ind < salient_points_.size());
    return salient_points_[ind];
}

SalientPointFragment& FragmentMap::GetSalientPointNew(size_t salient_point_id)
{
    size_t ind = SalientPointIdToInd(salient_point_id);
    CHECK(ind < salient_points_.size());
    return salient_points_[ind];
}

const SalientPointFragment& FragmentMap::GetSalientPointByInternalOrder(size_t sal_pnt_array_ind) const
{
    return salient_points_[sal_pnt_array_ind];
}

const suriko::Point3& FragmentMap::GetSalientPoint(size_t salient_point_id) const
{
    size_t ind = SalientPointIdToInd(salient_point_id);
    CHECK(ind < salient_points_.size());
    const std::optional<suriko::Point3>& sal_pnt = salient_points_[ind].coord;
    SRK_ASSERT(sal_pnt.has_value());
    return sal_pnt.value();
}

suriko::Point3& FragmentMap::GetSalientPoint(size_t salient_point_id)
{
    size_t ind = SalientPointIdToInd(salient_point_id);
    CHECK(ind < salient_points_.size());
    std::optional<suriko::Point3>& sal_pnt = salient_points_[ind].coord;
    SRK_ASSERT(sal_pnt.has_value());
    return sal_pnt.value();
}
bool FragmentMap::GetSalientPointByVirtualPointIdInternal(size_t salient_point_id, const SalientPointFragment** fragment)
{
    *fragment = nullptr;
    for (const SalientPointFragment& p : salient_points_)
    {
        if (p.synthetic_virtual_point_id == salient_point_id)
        {
            *fragment = &p;
            return true;
        }
    }
    return false;
}

void FragmentMap::SetFragmentIdOffsetInternal(size_t fragment_id_offset)
{
    fragment_id_offset_ = fragment_id_offset;
    next_salient_point_id_ = fragment_id_offset + 1;
}

size_t FragmentMap::SalientPointIdToInd(size_t salient_point_id) const
{
    SRK_ASSERT(salient_point_id >= fragment_id_offset_ + 1);
    return salient_point_id - fragment_id_offset_ - 1;
}

size_t FragmentMap::SalientPointIndToId(size_t salient_point_ind) const
{
    return salient_point_ind + fragment_id_offset_ + 1;
}

void FragmentMap::GetSalientPointsIds(std::vector<size_t>* salient_points_ids) const
{
    for (size_t i=0; i<salient_points_.size(); ++i)
    {
        size_t salient_point_id = SalientPointIndToId(i);
        salient_points_ids->push_back(salient_point_id);
    }
}

bool CornerTrack::HasCorners() const
{
    return StartFrameInd != -1;
}

size_t CornerTrack::CornersCount() const
{
    return CoordPerFramePixels.size();
}

void CornerTrack::AddCorner(size_t frame_ind, const suriko::Point2& value)
{
    if (StartFrameInd == -1)
        StartFrameInd = frame_ind;
    else
    {
        SRK_ASSERT(StartFrameInd >= 0);
        CHECK((size_t)StartFrameInd <= frame_ind) << "Can insert points later than the initial (start) frame"
            << " StartFrameInd=" << StartFrameInd << " frame_ind=" << frame_ind;
    }

    CornerData corner_data;
    corner_data.pixel_coord = value;
    CoordPerFramePixels.push_back(std::optional<CornerData>(corner_data));
    CheckConsistent();
}

CornerData& CornerTrack::AddCorner(size_t frame_ind)
{
    if (StartFrameInd == -1)
        StartFrameInd = frame_ind;
    else
    {
        SRK_ASSERT(StartFrameInd >= 0);
        CHECK((size_t)StartFrameInd <= frame_ind) << "Can insert points later than the initial (start) frame"
            << " StartFrameInd=" << StartFrameInd << " frame_ind=" << frame_ind;
    }
    ptrdiff_t local_ind = frame_ind - (ptrdiff_t)StartFrameInd;

    // the salient point may not be registered in some frames and the gaps appear
    CoordPerFramePixels.resize(local_ind + 1);

    CornerData corner_data {};
    CoordPerFramePixels.back() = std::optional<CornerData>(corner_data);

    CheckConsistent();
    return CoordPerFramePixels.back().value();
}

std::optional<suriko::Point2> CornerTrack::GetCorner(size_t frame_ind) const
{
    CHECK(StartFrameInd != -1);
    ptrdiff_t local_ind = frame_ind - StartFrameInd;

    if (local_ind < 0 || (size_t)local_ind >= CoordPerFramePixels.size())
        return std::optional<suriko::Point2>();

    std::optional<CornerData> corner_data = CoordPerFramePixels[local_ind];
    if (!corner_data.has_value())
        return std::optional<suriko::Point2>();
    return corner_data.value().pixel_coord;
}

std::optional<CornerData> CornerTrack::GetCornerData(size_t frame_ind) const
{
    CHECK(StartFrameInd != -1);
    ptrdiff_t local_ind = frame_ind - StartFrameInd;

    if (local_ind < 0 || (size_t)local_ind >= CoordPerFramePixels.size())
        return std::optional<CornerData>();
    return CoordPerFramePixels[local_ind];
}

void CornerTrack::CheckConsistent()
{
    if (StartFrameInd != -1)
        SRK_ASSERT(!CoordPerFramePixels.empty());
    else
        SRK_ASSERT(CoordPerFramePixels.empty());
}

void CornerTrack::EachCorner(std::function<void(size_t, const std::optional<CornerData>&)> on_item) const
{
    SRK_ASSERT(StartFrameInd != -1);
    for (size_t i=0; i<CoordPerFramePixels.size(); ++i)
    {
        on_item((size_t)StartFrameInd + i, CoordPerFramePixels[i]);
    }
}

suriko::CornerTrack& CornerTrackRepository::GetPointTrackById(size_t point_track_id)
{
    size_t point_track_ind = point_track_id;
    return CornerTracks[point_track_ind];
}

bool CornerTrackRepository::GetFirstPointTrackByFragmentSyntheticId(size_t salient_point_id, suriko::CornerTrack** corner_track)
{
    for (CornerTrack& track : CornerTracks)
        if (track.SyntheticVirtualPointId == salient_point_id)
        {
            *corner_track = &track;
            return true;
        }
    return false;
}

size_t CornerTrackRepository::CornerTracksCount() const
{
    return CornerTracks.size();
}

size_t CornerTrackRepository::ReconstructedCornerTracksCount() const
{
    size_t result = 0;
    for (const CornerTrack& track : CornerTracks)
    {
        if (track.SalientPointId.has_value())
            result += 1;
    }
    return result;
}

size_t CornerTrackRepository::FramesCount() const
{
    if (CornerTracks.empty())
        return 0;
    size_t corners_count = CornerTracks[0].CornersCount();
    return corners_count;
}

suriko::CornerTrack& CornerTrackRepository::AddCornerTrackObj()
{
    size_t new_track_id = CornerTracks.size();
    CornerTrack new_track;
    new_track.TrackId = new_track_id;
    CornerTracks.push_back(new_track);
    return CornerTracks.back();
}

const suriko::CornerTrack& CornerTrackRepository::GetPointTrackById(size_t point_track_id) const
{
    size_t point_track_ind = point_track_id;
    return CornerTracks[point_track_ind];
}

void CornerTrackRepository::PopulatePointTrackIds(std::vector<size_t> *result)
{
    for (size_t pnt_ind=0;pnt_ind<CornerTracks.size(); ++pnt_ind)
        result->push_back(pnt_ind);
}

bool IsIdentity(const Eigen::Matrix<Scalar, 3, 3>& M, Scalar rtol, Scalar atol, std::string* msg)
{
    typedef Eigen::Matrix<Scalar, 3, 3>::Index IndT;
    for (IndT row = 0; row < M.rows(); ++row)
    {
        for (IndT col = 0; col < M.cols(); ++col)
        {
            Scalar expect_value = row == col ? (Scalar)1 : (Scalar)0;
            Scalar cell = M(row, col);
            if (!IsClose(expect_value, cell, rtol, atol))
            {
                if (msg != nullptr)
                {
                    std::stringstream ss;
                    ss << "expected M(" <<row <<"," <<col <<") to be" << expect_value
                        <<" but actual=" <<cell;
                    *msg = ss.str();
                }
                return false;
            }
        }
    }
    return true;
}

template <size_t N>
bool IsOrthogonal(const Eigen::Matrix<Scalar,N,N>& R, std::string* msg) {
    Scalar rtol = 1.0e-3;
    Scalar atol = 1.0e-3;
    auto rt_r = (R.transpose() * R).eval();
    bool is_ident = rt_r.isIdentity(atol);
    if (!is_ident)
    {
        if (msg != nullptr)
        {
            std::stringstream ss;
            ss << "failed Rt.R=I, R=\n" << R;
            *msg = ss.str();
        }
        return false;
    }
    return true;
}

bool IsOrthogonal(const Eigen::Matrix<Scalar, 2, 2>& R, std::string* msg) { return IsOrthogonal<2>(R, msg); }
bool IsOrthogonal(const Eigen::Matrix<Scalar, 3, 3>& R, std::string* msg) { return IsOrthogonal<3>(R, msg); }

bool IsSpecialOrthogonal(const Eigen::Matrix<Scalar,3,3>& R, std::string* msg) {
    bool is_ortho = IsOrthogonal(R, msg);
    if (!is_ortho)
        return false;

    Scalar rdet = R.determinant();

    Scalar rtol = 1.0e-3;
    Scalar atol = 1.0e-3;
    bool det_one = IsClose(1, rdet, rtol, atol);
    if (!det_one)
    {
        if (msg != nullptr)
        {
            std::stringstream ss;
            ss << "failed det(R)=1, actual detR=" << rdet << " R=\n" << R;
            *msg = ss.str();
        }
        return false;
    }
    return true;
}

void SkewSymmetricMat(const Eigen::Matrix<Scalar, 3, 1>& v, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> skew_mat)
{
    *skew_mat << 
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
}

auto RotMatFromUnityDirAndAngle(const Eigen::Matrix<Scalar, 3, 1>& unity_dir, Scalar ang, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> rot_mat, bool check_input) -> bool
{
    // skip precondition checking in Release mode on user request (check_input=false)
    if (check_input || kSurikoDebug)
    {
        // direction must be a unity vector
        Scalar dir_len = unity_dir.norm();
        SRK_ASSERT(IsClose(1, dir_len)) << "provide valid unity_dir";

        // Rotating about some unity direction by zero angle is described by the identity matrix,
        // which we can return here. But the symmetric operation of computing direction and angle from R 
        // is not possible - the direction can't be recovered.
        if (IsClose(0, ang)) return false;
    }
	
    Scalar s = std::sin(ang);
    Scalar c = std::cos(ang);

    Eigen::Matrix<Scalar, 3, 3> skew1;
    SkewSymmetricMat(unity_dir, &skew1);

    *rot_mat = Eigen::Matrix<Scalar, 3, 3>::Identity() + s * skew1 + (1 - c) * skew1 * skew1;

    if (kSurikoDebug) // postcondition
    {
        std::string msg;
        bool ok = IsSpecialOrthogonal(*rot_mat, &msg);
        CHECK(ok) << msg;
    }
    return true;
}

auto RotMatFromAxisAngle(const Eigen::Matrix<Scalar, 3, 1>& axis_angle, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> rot_mat) -> bool
{
    Scalar ang = axis_angle.norm();
    if (IsClose(0, ang)) return false;

    Eigen::Matrix<Scalar, 3, 1> unity_dir = axis_angle / ang;
    const bool check_input = false;
    return RotMatFromUnityDirAndAngle(unity_dir, ang, rot_mat, check_input);
}

auto LogSO3(const Eigen::Matrix<Scalar, 3, 3>& rot_mat, gsl::not_null<Eigen::Matrix<Scalar, 3, 1>*> unity_dir, gsl::not_null<Scalar*> ang, bool check_input) -> bool
{
    // skip precondition checking in Release mode on user request (check_input=false)
    if (check_input || kSurikoDebug)
    {
        std::string msg;
        bool ok = IsSpecialOrthogonal(rot_mat, &msg);
        CHECK(ok) << msg;
    }
    Scalar cos_ang = (Scalar)(0.5*(rot_mat.trace() - 1));
    cos_ang = std::clamp<Scalar>(cos_ang, -1, 1); // the cosine may be slightly off due to rounding errors

    Scalar sin_ang = (Scalar)std::sqrt(1.0 - cos_ang * cos_ang);
    Scalar atol = 1e-3;
    if (IsClose(0, sin_ang, 0, atol))
        return false;

    auto& udir = *unity_dir;
    udir[0] = rot_mat(2, 1) - rot_mat(1, 2);
    udir[1] = rot_mat(0, 2) - rot_mat(2, 0);
    udir[2] = rot_mat(1, 0) - rot_mat(0, 1);
    udir *= (Scalar)0.5 / sin_ang;

    // direction vector is already close to unity, but due to rounding errors it diverges
    // TODO: check where the rounding error appears
    Scalar dirlen = udir.norm();
    udir *= 1 / dirlen;

    *ang = std::acos(cos_ang);
    return true;
}

auto AxisAngleFromRotMat(const Eigen::Matrix<Scalar, 3, 3>& rot_mat, gsl::not_null<Eigen::Matrix<Scalar, 3, 1>*> dir) -> bool
{
    Eigen::Matrix<Scalar, 3, 1> unity_dir;
    Scalar ang;
    bool op = LogSO3(rot_mat, &unity_dir, &ang);
    if (!op) return false;

    *dir = unity_dir * ang;
    return true;
}

auto DecomposeProjMat(const Eigen::Matrix<Scalar, 3, 4> &proj_mat, bool check_post_cond)
-> std::tuple<bool, Scalar, Eigen::Matrix<Scalar, 3, 3>, SE3Transform>
{
    using namespace Eigen;
    typedef Matrix<Scalar,3,3> Mat33;

    // copy the input, because we may flip sign later
    Mat33 Q = proj_mat.leftCols(3);
    Matrix<Scalar,3,1> q = proj_mat.rightCols(1);

    // ensure that R will have positive determinant
    int P_sign = 1;
    Scalar Q_det = Q.determinant();
    if (Q_det < 0)
    {
        P_sign = -1;
        Q *= -1;
        q *= -1;
    }

    // find translation T
    Mat33 Q_inv = Q.inverse();
    Matrix<Scalar,3,1> t = -Q_inv * q;

    // find rotation R
    Mat33 QQt = Q * Q.transpose();

    // QQt is inverted to allow further use Cholesky decomposition to find K
    Mat33 QQt_inv = QQt.inverse();

    // Cholesky decomposition
    LLT<Mat33> llt(QQt_inv);
    if (llt.info() == Eigen::NumericalIssue)
    {
        VLOG(4) << "got negative matrix";
        return std::make_tuple(false, Scalar{}, Eigen::Matrix<Scalar, 3, 3>{}, SE3Transform{});
    }
    Mat33 C = llt.matrixL();

    // we need upper triangular matrix, but Eigen::LLT returns lower triangular
    C.transposeInPlace();

    Mat33 R = (C * Q).transpose();

    if (check_post_cond)
    {
        std::string err_msg;
        bool op = IsSpecialOrthogonal(R, &err_msg);
        CHECK(op) <<err_msg <<std::endl;
    }

    // find intrinsic parameters K
    Mat33 C_inv = C.inverse();
    Scalar c_last = C_inv(2, 2);
    SRK_ASSERT(!IsClose(0, c_last)) << "det(P)<3";

    Mat33 K = C_inv * (1/c_last);

    Scalar scale_factor = P_sign * c_last;

    if (check_post_cond)
    {
        Eigen::Matrix<Scalar, 3, 4> rt34;
        rt34 <<Mat33::Identity(), -t;
        Eigen::Matrix<Scalar, 3, 4> P_back = scale_factor * K * R.transpose() * rt34;
        auto diff = (proj_mat - P_back).norm();
        SRK_ASSERT(diff < 1e-2) << "Failed to decompose P[3x4]->R,T,K" << "diff=" << diff;
    }

    SE3Transform direct_orient_cam(R,t);
    return std::make_tuple(true, scale_factor, K, direct_orient_cam);
}

auto Triangulate3DPointByLeastSquares(const std::vector<suriko::Point2> &xs2D, const std::vector<Eigen::Matrix<Scalar,3,4>> &proj_mat_list, Scalar f0) -> suriko::Point3
{
    size_t frames_count_P = proj_mat_list.size();
    size_t frames_count_xs = xs2D.size();
    CHECK_EQ(frames_count_P, frames_count_xs) << "Provide two lists of 2D coordinates and projection matrices of the same length";

    size_t frames_count = frames_count_P;
    CHECK(frames_count >= 2) << "Provide 2 or more projections of a 3D point";

    // populate matrices A and B to solve for least squares
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> A(frames_count * 2, 3);
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> B(frames_count * 2);

    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind)
    {
        const auto &x2D = xs2D[frame_ind];
        auto x = x2D[0];
        auto y = x2D[1];
        const auto &P = proj_mat_list[frame_ind];
        A(frame_ind * 2, 0) = x * P(2, 0) - f0 * P(0, 0);
        A(frame_ind * 2, 1) = x * P(2, 1) - f0 * P(0, 1);
        A(frame_ind * 2, 2) = x * P(2, 2) - f0 * P(0, 2);
        A(frame_ind * 2 + 1, 0) = y * P(2, 0) - f0 * P(1, 0);
        A(frame_ind * 2 + 1, 1) = y * P(2, 1) - f0 * P(1, 1);
        A(frame_ind * 2 + 1, 2) = y * P(2, 2) - f0 * P(1, 2);

        B(frame_ind * 2) = -(x * P(2, 3) - f0 * P(0, 3));
        B(frame_ind * 2 + 1) = -(y * P(2, 3) - f0 * P(1, 3));
    }

#define LEAST_SQ 2
#if LEAST_SQ == 1
    const auto& jacobi_svd = A.jacobiSvd(Eigen::ComputeThinU|Eigen::ComputeThinV);
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sol = jacobi_svd.solve(B);
#elif LEAST_SQ == 2
    const auto& householder_qr = A.colPivHouseholderQr();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sol = householder_qr.solve(B);
#endif

    static bool debug = false;
    if (debug)
    {
        const Eigen::Matrix<Scalar,Eigen::Dynamic,1> diff_vec = A * sol - B;
        Scalar diff = diff_vec.norm();
        LOG_IF(INFO, diff > 5) << "warn: big diff=" << diff << " frames_count=" << frames_count;
    }
    suriko::Point3 x3D(sol(0), sol(1), sol(2));
    return x3D;
}

SE3Transform LookAtLufWfc(
    const Eigen::Matrix<Scalar, 3, 1>& eye,
    const Eigen::Matrix<Scalar, 3, 1>& center,
    const Eigen::Matrix<Scalar, 3, 1>& up)
{
    // align OZ with view direction
    Eigen::Matrix<Scalar, 3, 1> forward_dir = center - eye;
    forward_dir.normalize();

    // align OY to match up vector
    // cam_up = up - proj(up onto forward_dir)
    Eigen::Matrix<Scalar, 3, 1> cam_up_dir = up - forward_dir * up.dot(forward_dir); // new OY
    cam_up_dir.normalize();

    SE3Transform world_from_cam;
    world_from_cam.R.middleCols<1>(0) = cam_up_dir.cross(forward_dir);
    world_from_cam.R.middleCols<1>(1) = cam_up_dir;
    world_from_cam.R.middleCols<1>(2) = forward_dir;
    world_from_cam.T = eye;
    return world_from_cam;
}

Scalar GetUncertaintyEllipsoidProbabilityCutValue(
    const Eigen::Matrix<Scalar, 3, 3>& gauss_sigma,
    Scalar portion_of_max_prob, bool check_det = true)
{
    Scalar uncert_det = gauss_sigma.determinant();
    static bool test_det = true;
    if (test_det && check_det)
        SRK_ASSERT(uncert_det >= 0);
    Scalar max_prob = 1 / std::sqrt(suriko::Pow3(2 * M_PI) * uncert_det);
    Scalar cut_value = portion_of_max_prob * max_prob;
    return cut_value;
}

void PickPointOnEllipsoid(
    const Eigen::Matrix<Scalar, 3, 1>& cam_pos,
    const Eigen::Matrix<Scalar, 3, 3>& cam_pos_uncert, Scalar ellipsoid_cut_thr,
    const Eigen::Matrix<Scalar, 3, 1>& ray,
    Eigen::Matrix<Scalar, 3, 1>* pos_ellipsoid)
{
    Eigen::Matrix<Scalar, 3, 3> uncert_inv = cam_pos_uncert.inverse();
    Scalar uncert_det = cam_pos_uncert.determinant();
    Scalar cut_value = GetUncertaintyEllipsoidProbabilityCutValue(cam_pos_uncert, ellipsoid_cut_thr);

    // cross ellipsoid with ray
    Scalar b1 = -std::log(suriko::Sqr(cut_value)*suriko::Pow3(2 * M_PI)*uncert_det);
    Eigen::Matrix<Scalar, 1, 1> b2 = ray.transpose() * uncert_inv * ray;
    Scalar t2 = b1 / b2[0];
    SRK_ASSERT(t2 >= 0) << "invalid covariance matrix";

    Scalar t = std::sqrt(t2);

    // crossing of ellipsoid and ray
    *pos_ellipsoid = cam_pos + t * ray;
}

void PickPointOnEllipsoid(
    const Ellipsoid3DWithCenter& ellipsoid,
    const Eigen::Matrix<Scalar, 3, 1>& ray,
    Eigen::Matrix<Scalar, 3, 1>* pos_ellipsoid)
{
    auto den_mat = ray.transpose() * ellipsoid.A * ray;
    Scalar ratio = ellipsoid.right_side / den_mat[0];
    SRK_ASSERT(ratio >= 0);

    Scalar t = std::sqrt(ratio);
    *pos_ellipsoid = ellipsoid.center + t * ray;
}

/// Calculates A,b,c ellipsoid coefficients as in equation xAx+bx+c=0. eg: prop_thr=0.05 for 2sigma.
void ExtractEllipsoidFromUncertaintyMat(
    const Eigen::Matrix<Scalar, 3, 1>& gauss_mean, 
    const Eigen::Matrix<Scalar, 3, 3>& gauss_sigma, Scalar ellipsoid_cut_thr,
    Eigen::Matrix<Scalar, 3, 3>* A,
    Eigen::Matrix<Scalar, 3, 1>* b, Scalar* c)
{
    Eigen::Matrix<Scalar, 3, 3> uncert_inv = gauss_sigma.inverse();
    *A = uncert_inv;
    *b = -2 * uncert_inv * gauss_mean;

    Scalar uncert_det = gauss_sigma.determinant();
    Scalar cut_value = GetUncertaintyEllipsoidProbabilityCutValue(gauss_sigma, ellipsoid_cut_thr);

    Scalar c1 = std::log(suriko::Sqr(cut_value)*suriko::Pow3(2 * M_PI)*uncert_det);
    Eigen::Matrix<Scalar, 1, 1> c2 = gauss_mean.transpose() * uncert_inv * gauss_mean;
    *c = c1 + c2[0];

    if (VLOG_IS_ON(4))
    {
        Eigen::Matrix<Scalar, 3, 1> ray{ 0.5, 0.5, 0.5 };
        Eigen::Matrix<Scalar, 3, 1> pos_ellipsoid;
        PickPointOnEllipsoid(gauss_mean, gauss_sigma, ellipsoid_cut_thr, ray, &pos_ellipsoid);

        Scalar diff = (pos_ellipsoid.transpose() * (*A) * pos_ellipsoid + b->transpose() * pos_ellipsoid)[0] + (*c);
        //SRK_ASSERT(IsClose(0, diff, AbsRelTol<Scalar>(1, 0.1)));
        SRK_ASSERT(true);
    }
}

bool ValidateEllipsoid(const Ellipsoid3DWithCenter& maybe_ellipsoid)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 3, 3>> eigen_solver(maybe_ellipsoid.A);
    bool op = eigen_solver.info() == Eigen::Success;
    if (!op)
        return false;
    const Eigen::Matrix<Scalar, 3, 1>& vs = eigen_solver.eigenvalues().eval();
    const bool all_posotive = vs[0] > 0 && vs[1] > 0 && vs[2] > 0;
    return all_posotive;
}

bool ValidateEllipse(const Ellipse2DWithCenter& maybe_ellipse)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> eigen_solver(maybe_ellipse.A);
    bool op = eigen_solver.info() == Eigen::Success;
    if (!op)
        return false;
    const Eigen::Matrix<Scalar, 2, 1>& vs = eigen_solver.eigenvalues().eval();
    const bool all_posotive = vs[0] > 0 && vs[1] > 0;
    return all_posotive;
}

Rect GetEllipseBounds(const Ellipse2DWithCenter& ellipse)
{
    Scalar a1 = ellipse.A(0, 0);
    Scalar a2 = ellipse.A(0, 1);
    Scalar a4 = ellipse.A(1, 1);
    Scalar m1 = ellipse.center[0];
    Scalar m2 = ellipse.center[1];

    Scalar dx = std::sqrt(ellipse.right_side / (a1 - a2 * a2 / a4));
    Scalar dy = std::sqrt(ellipse.right_side / (a4 - a2 * a2 / a1));

    Rect result;
    result.x = m1 - dx;
    result.width = 2 * dx;
    result.y = m2 - dy;
    result.height = 2 * dy;
    return result;
}

/// Calculates A,b,c ellipsoid coefficients as in equation xAx+bx+c=0. eg: prop_thr=0.05 for 2sigma.
void ExtractEllipsoidFromUncertaintyMat(
    const Eigen::Matrix<Scalar, 3, 1>& gauss_mean, 
    const Eigen::Matrix<Scalar, 3, 3>& gauss_sigma, Scalar ellipsoid_cut_thr,
    Ellipsoid3DWithCenter* ellipsoid)
{
    ellipsoid->center = gauss_mean;

    Eigen::Matrix<Scalar, 3, 3> uncert_inv = gauss_sigma.inverse();
    ellipsoid->A = uncert_inv;

    //
    Scalar uncert_det = gauss_sigma.determinant();
    Scalar cut_value = GetUncertaintyEllipsoidProbabilityCutValue(gauss_sigma, ellipsoid_cut_thr);

    Scalar c1 = -std::log(suriko::Sqr(cut_value)*suriko::Pow3(2 * M_PI)*uncert_det);
    ellipsoid->right_side = c1;
    //ellipsoid->right_side = 7.814;

    if (kSurikoDebug) { SRK_ASSERT(ValidateEllipsoid(*ellipsoid)); }

    if (VLOG_IS_ON(4))
    {
        Eigen::Matrix<Scalar, 3, 1> ray{ 0.5, 0.5, 0.5 };
        Eigen::Matrix<Scalar, 3, 1> pos_ellipsoid;
        PickPointOnEllipsoid(*ellipsoid, ray, &pos_ellipsoid);

        Eigen::Matrix<Scalar, 3, 1> arm = pos_ellipsoid - ellipsoid->center;
        Eigen::Matrix<Scalar, 1, 1> zero_mat = arm.transpose() *ellipsoid->A * arm;
        Scalar zero = zero_mat[0] - ellipsoid->right_side;
        //SRK_ASSERT(IsClose(0, diff, AbsRelTol<Scalar>(1, 0.1)));
        SRK_ASSERT(true);
    }
}

bool EqRotEllip(const RotatedEllipsoid3D& lhs, const RotatedEllipsoid3D& rhs, Scalar diff)
{
    Scalar diff1a = (lhs.semi_axes - rhs.semi_axes).norm();
    Scalar diff1b = (lhs.semi_axes + rhs.semi_axes).norm();

    Scalar diffR1a = (lhs.world_from_ellipse.R.leftCols<1>() - rhs.world_from_ellipse.R.leftCols<1>()).norm();
    Scalar diffR1b = (lhs.world_from_ellipse.R.leftCols<1>() + rhs.world_from_ellipse.R.leftCols<1>()).norm();
    Scalar diffR2a = (lhs.world_from_ellipse.R.middleCols<1>(1) - rhs.world_from_ellipse.R.middleCols<1>(1)).norm();
    Scalar diffR2b = (lhs.world_from_ellipse.R.middleCols<1>(1) + rhs.world_from_ellipse.R.middleCols<1>(1)).norm();
    Scalar diffR3a = (lhs.world_from_ellipse.R.rightCols<1>() - rhs.world_from_ellipse.R.rightCols<1>()).norm();
    Scalar diffR3b = (lhs.world_from_ellipse.R.rightCols<1>() + rhs.world_from_ellipse.R.rightCols<1>()).norm();

    Scalar diff3a = (lhs.world_from_ellipse.T - rhs.world_from_ellipse.T).norm();
    Scalar diff3b = (lhs.world_from_ellipse.T + rhs.world_from_ellipse.T).norm();

    AbsRelTol<Scalar> art{ diff, 0 };
    bool a = IsClose(0, diff1a, art) || IsClose(0, diff1b, art);
    bool b1 = IsClose(0, diffR1a, art) || IsClose(0, diffR1b, art);
    bool b2 = IsClose(0, diffR2a, art) || IsClose(0, diffR2b, art);
    bool b3 = IsClose(0, diffR3a, art) || IsClose(0, diffR3b, art);
    bool c = IsClose(0, diff3a, art) || IsClose(0, diff3b, art);
    return a && b1 && b2 && b3 && c;
}

void GetRotatedUncertaintyEllipsoidFromCovMat(const Eigen::Matrix<Scalar, 3, 3>& cov, const Eigen::Matrix<Scalar, 3, 1>& mean,
    Scalar ellipsoid_cut_thr,
    RotatedEllipsoid3D* result)
{
    // check symmetry
    Scalar sym_diff = (cov - cov.transpose()).norm();
    SRK_ASSERT(IsClose(0, sym_diff, AbsTol(0.001)));

    Scalar uncert_det = cov.determinant();
    bool check_det = false;
    Scalar cut_value = GetUncertaintyEllipsoidProbabilityCutValue(cov, ellipsoid_cut_thr, check_det);

    // right side of the ellipse equation: (x-mu)A(x-mu)=right_side
    Scalar right_side_old = -std::log(suriko::Sqr(cut_value)*suriko::Pow3(2 * M_PI)*uncert_det);
    Scalar right_side = 7.814;

    //
    // A=V*D*inv(V)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 3, 3>> eigen_solver(cov);
    bool op = eigen_solver.info() == Eigen::Success;
    SRK_ASSERT(op);

    Eigen::Matrix<Scalar, 3, 3> eig_vecs = eigen_solver.eigenvectors(); // rot_mat_ellipse_from_world
    Eigen::Matrix<Scalar, 3, 1> dd = eigen_solver.eigenvalues();
    
    // Eigen::SelfAdjointEigenSolver sorts eigenvalues in ascending order
    // but we want the semi-axes of an allipse to be in decreasing order
    int major_col_ind = 2;
    int minor_col_ind = 0;
    if (dd[0] == dd[2])
    {
        // all eigenvalues are equal and may be treated as already ordered in decreasing order
        std::swap(major_col_ind, minor_col_ind);
    }

    // order semi-axes from max to min
    auto& R = result->world_from_ellipse.R;
    R.middleCols<1>(0) = eig_vecs.middleCols<1>(major_col_ind);
    R.middleCols<1>(1) = eig_vecs.middleCols<1>(1);
    R.middleCols<1>(2) = eig_vecs.middleCols<1>(minor_col_ind);

    // eigenvectors is an orthogonal matrix (det=+1 or -1), but it may not be orthonormal (det=+1)
    Scalar R_det = R.determinant();
    if (R_det < 0)
    {
        // fix eigenvectors to have det=1
        // changing the sign of the last column will flip the sign of determinant (to be positive)
        R.rightCols<1>() = -R.rightCols<1>();
    }

    bool is_spec_ortho = IsSpecialOrthogonal(R);
    SRK_ASSERT(is_spec_ortho); // further relying on inv(R)=Rt

    {
        Eigen::Matrix<Scalar, 3, 1> left_col = R.middleCols<1>(0);
        Eigen::Matrix<Scalar, 3, 1> right_col = left_col.cross(R.middleCols<1>(1));
        Scalar should_zero = (right_col - R.middleCols<1>(2)).norm();
        SRK_ASSERT(IsClose(0, should_zero));
    }

    // order semi-axes from max to min
    auto& semi = result->semi_axes;
    semi[0] = std::sqrt(right_side * dd[major_col_ind]);
    semi[1] = std::sqrt(right_side * dd[1]);
    semi[2] = std::sqrt(right_side * dd[minor_col_ind]);

    result->world_from_ellipse.T = mean;
}

bool GetRotatedEllipsoid(const Ellipsoid3DWithCenter& ellipsoid, bool can_throw, RotatedEllipsoid3D* result)
{
    // check symmetry
    // A=V*D*inv(V)
    // Eigen::SelfAdjointEigenSolver sorts eigenvalues in ascending order
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 3, 3>> eigen_solver(ellipsoid.A);
    bool op = eigen_solver.info() == Eigen::Success;
    if (!op)
    {
        if (can_throw)
            SRK_ASSERT(false);
        else
            return false;
    }

    Eigen::Matrix<Scalar, 3, 3> R = eigen_solver.eigenvectors(); // rot_mat_ellipse_from_world
    Scalar R_det = R.determinant();
    if (R_det < 0) // det(R)=-1
    {
        // fix eigen vectors to have det=1
        // Eigen::SelfAdjointEigenSolver sorts eigenvalues in ascending order, so
        // we will fix the first eigenvector based on the last two eigenvectors
        auto mid_col = R.middleCols<1>(1).eval();
        Eigen::Matrix<Scalar, 3, 1> new_left = mid_col.cross(R.rightCols<1>());
        Scalar should_zero = (new_left - (-R.leftCols<1>())).norm();
        SRK_ASSERT(IsClose(0, should_zero, AbsTol(0.1)));
        R.leftCols<1>() = -R.leftCols<1>();
    }
    bool is_spec_ortho = IsSpecialOrthogonal(R);
    SRK_ASSERT(is_spec_ortho); // further relying on inv(R)=Rt
    
    //
    result->world_from_ellipse.R = R;
    result->world_from_ellipse.T = ellipsoid.center;

    Eigen::Matrix<Scalar, 3, 1> dd = eigen_solver.eigenvalues();

    auto& semi = result->semi_axes;
    semi[0] = std::sqrt(ellipsoid.right_side / dd[0]);
    semi[1] = std::sqrt(ellipsoid.right_side / dd[1]);
    semi[2] = std::sqrt(ellipsoid.right_side / dd[2]);
    return true;
}

RotatedEllipsoid3D GetRotatedEllipsoid(const Ellipsoid3DWithCenter& ellipsoid)
{
    RotatedEllipsoid3D result;
    bool op = GetRotatedEllipsoid(ellipsoid, true, &result);
    SRK_ASSERT(op);
    return result;
}

RotatedEllipse2D GetRotatedEllipse2D(const Ellipse2DWithCenter& ellipse)
{
    // check symmetry
    // A=V*D*inv(V)
    // Eigen::SelfAdjointEigenSolver sorts eigenvalues in ascending order
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> eigen_solver(ellipse.A);
    bool op = eigen_solver.info() == Eigen::Success;
    SRK_ASSERT(op);

    Eigen::Matrix<Scalar, 2, 2> R = eigen_solver.eigenvectors(); // rot_mat_ellipse_from_world
    bool is_ortho = IsOrthogonal(R);
    SRK_ASSERT(is_ortho); // further relying on inv(R)=Rt

    //
    RotatedEllipse2D result;
    result.world_from_ellipse.R = R;
    result.world_from_ellipse.T = ellipse.center;

    Eigen::Matrix<Scalar, 2, 1> dd = eigen_solver.eigenvalues();
    result.semi_axes[0] = std::sqrt(ellipse.right_side / dd[0]);
    result.semi_axes[1] = std::sqrt(ellipse.right_side / dd[1]);
    return result;
}

bool CanExtractEllipsoid(const Eigen::Matrix<Scalar, 3, 3>& pos_cov, bool cov_mat_directly_to_rot_ellipsoid)
{
    Eigen::Matrix<Scalar, 3, 1> pos(0, 0, 0);

    if (!cov_mat_directly_to_rot_ellipsoid)
    {
        Ellipsoid3DWithCenter ellipsoid;
        ExtractEllipsoidFromUncertaintyMat(pos, pos_cov, 0.05, &ellipsoid);
        
        RotatedEllipsoid3D rot_ellipsoid;
        bool suc = GetRotatedEllipsoid(ellipsoid, false, &rot_ellipsoid);
        return suc;
    }
    else
    {

        RotatedEllipsoid3D rot_ellipsoid2;
        GetRotatedUncertaintyEllipsoidFromCovMat(pos_cov, pos, 0.05, &rot_ellipsoid2);
        //bool diff_ok = EqRotEllip(rot_ellipsoid, rot_ellipsoid2, 0.1);
        //SRK_ASSERT(diff_ok);
        return true;
    }
}

bool ProjectEllipsoidOnCamera(const Ellipsoid3DWithCenter& ellipsoid,
    const Eigen::Matrix<Scalar, 3, 1>& eye,
    const Eigen::Matrix<Scalar, 3, 1>& cam_plane_u,
    const Eigen::Matrix<Scalar, 3, 1>& cam_plane_v,
    const Eigen::Matrix<Scalar, 3, 1>& n, Scalar lam,
    Ellipse2DWithCenter* result)
{
    const auto& u = cam_plane_u;
    const auto& v = cam_plane_v;
    
    // find M which specifies the set of directions {d} towards the projected ellipse
    // {d} = d such that d*M*d=0
    Eigen::Matrix<Scalar, 3, 1> em = eye - ellipsoid.center;
    Eigen::Matrix<Scalar, 3, 1> v2 = ellipsoid.A * em;
    Eigen::Matrix<Scalar, 3, 3> M = 4 * (v2 * v2.transpose() - ellipsoid.A * (em.transpose() * v2 - ellipsoid.right_side));

    // find k0, ..., k5 for sig*P*sig+q*sig+k5=0
    Scalar k0 = (u.transpose() * M * u)[0];
    Scalar k1 = (u.transpose() * M * v)[0];
    Scalar k2 = (v.transpose() * M * v)[0];

    Scalar lam_ne = lam - n.transpose() * eye;
    Scalar k3 = 2 * lam_ne*(u.transpose() * M * n)[0];
    Scalar k4 = 2 * lam_ne*(v.transpose() * M * n)[0];
    
    Scalar k5 = suriko::Sqr(lam_ne) * (n.transpose() * M * n)[0];

    // transform it into out ellipse
    Eigen::Matrix<Scalar, 2, 2>& P = result->A;
    P(0, 0) = k0;
    P(0, 1) = k1;
    P(1, 0) = k1;
    P(1, 1) = k2;
    Eigen::Matrix<Scalar, 2, 1> q { k3, k4 };

    // transform ellipse form x*A*x+q*x+c==0 into (x-m)A(x-m)=r
    Eigen::Matrix<Scalar, 2, 1> center = P.colPivHouseholderQr().solve(-0.5 * q);
    Scalar right_side = center.transpose() * P * center - k5;

    // for convenience, make right side positive
    if (right_side < 0)
    {
        right_side = -right_side;
        P *= -1;
    }

    result->center = center;
    result->right_side = right_side;
    if (ValidateEllipse(*result))
        return true;

    // occur when ellipsoid 'behind' the camera
    return false;
}

void ProjectEllipsoidOnCamera_FillEllipseOutlineDirectionsMat3x3(
    const RotatedEllipsoid3D& rot_ellip,
    const Eigen::Matrix<Scalar, 3, 1>& eye,
    Eigen::Matrix<Scalar, 3, 3>* M);

bool ProjectEllipsoidOnCamera(const RotatedEllipsoid3D& rot_ellip,
    const Eigen::Matrix<Scalar, 3, 1>& eye,
    const Eigen::Matrix<Scalar, 3, 1>& cam_plane_u,
    const Eigen::Matrix<Scalar, 3, 1>& cam_plane_v,
    const Eigen::Matrix<Scalar, 3, 1>& n, Scalar lam,
    Ellipse2DWithCenter* result)
{
    // eye (pos of camera) in the camera plane in world coordinates
    Eigen::Matrix<Scalar, 3, 1> eye_on_cam_plane = eye + (lam - (n.transpose()*eye)[0])*n; // theta

    // check if eye is inside the ellipsoid
    SE3Transform ellip_from_world = SE3Inv(rot_ellip.world_from_ellipse);
    suriko::Point3 eye_in_ellip = SE3Apply(ellip_from_world, eye);
    Scalar ellip_pnt =
        suriko::Sqr(eye_in_ellip[0]) / suriko::Sqr(rot_ellip.semi_axes[0]) +
        suriko::Sqr(eye_in_ellip[1]) / suriko::Sqr(rot_ellip.semi_axes[1]) +
        suriko::Sqr(eye_in_ellip[2]) / suriko::Sqr(rot_ellip.semi_axes[2]);
    bool eye_inside_ellip = ellip_pnt <= RotatedEllipsoid3D::kRightSide;

    // find M which specifies the set of directions {d} towards the projected ellipse
    // {d} = d such that d*M*d=0
    Eigen::Matrix<Scalar, 3, 3> M;
    ProjectEllipsoidOnCamera_FillEllipseOutlineDirectionsMat3x3(rot_ellip, eye, &M);

    const auto& u = cam_plane_u;
    const auto& v = cam_plane_v;

    // find k0, ..., k5 for sig*P*sig+q*sig+k5=0
    Scalar k0 = (u.transpose() * M * u)[0];
    Scalar k1 = (u.transpose() * M * v)[0];
    Scalar k2 = (v.transpose() * M * v)[0];

    Scalar lam_ne = lam - n.transpose() * eye;
    Scalar k3 = 2 * lam_ne*(u.transpose() * M * n)[0];
    Scalar k4 = 2 * lam_ne*(v.transpose() * M * n)[0];
    
    Scalar k5 = suriko::Sqr(lam_ne) * (n.transpose() * M * n)[0];

    // transform it into out ellipse
    Eigen::Matrix<Scalar, 2, 2>& P = result->A;
    P(0, 0) = k0;
    P(0, 1) = k1;
    P(1, 0) = k1;
    P(1, 1) = k2;
    Eigen::Matrix<Scalar, 2, 1> q { k3, k4 };

    // transform ellipse form x*A*x+q*x+c==0 into (x-m)A(x-m)=r
    Eigen::Matrix<Scalar, 2, 1> center = P.colPivHouseholderQr().solve(-0.5 * q);
    Scalar right_side = center.transpose() * P * center - k5;

    // for convenience, make right side positive
    if (right_side < 0)
    {
        right_side = -right_side;
        P *= -1;
    }

    result->center = center;
    result->right_side = right_side;
    if (ValidateEllipse(*result))
        return true;

    // occur when ellipsoid 'behind' the camera
    bool ok = eye_inside_ellip;
    return false;
}

bool IntersectRotEllipsoidAndPlane(const RotatedEllipsoid3D& rot_ellip,
    const Eigen::Matrix<Scalar, 3, 1>& eye,
    const Eigen::Matrix<Scalar, 3, 1>& cam_plane_u,
    const Eigen::Matrix<Scalar, 3, 1>& cam_plane_v,
    const Eigen::Matrix<Scalar, 3, 1>& n, Scalar lam,
    Ellipse2DWithCenter* result)
{
    // eye (pos of camera) in the camera plane in world coordinates
    Eigen::Matrix<Scalar, 3, 1> eye_on_cam_plane = eye + (lam - (n.transpose()*eye)[0])*n; // theta

    // check if eye is inside the ellipsoid
    SE3Transform ellip_from_world = SE3Inv(rot_ellip.world_from_ellipse);
    suriko::Point3 eye_in_ellip = SE3Apply(ellip_from_world, eye);
    Scalar ellip_pnt =
        suriko::Sqr(eye_in_ellip[0]) / suriko::Sqr(rot_ellip.semi_axes[0]) +
        suriko::Sqr(eye_in_ellip[1]) / suriko::Sqr(rot_ellip.semi_axes[1]) +
        suriko::Sqr(eye_in_ellip[2]) / suriko::Sqr(rot_ellip.semi_axes[2]);
    bool eye_inside_ellip = ellip_pnt <= RotatedEllipsoid3D::kRightSide;

    Eigen::Matrix<Scalar, 3, 1> Rt_u = rot_ellip.world_from_ellipse.R.transpose() * cam_plane_u;
    Eigen::Matrix<Scalar, 3, 1> Rt_v = rot_ellip.world_from_ellipse.R.transpose() * cam_plane_v;

    // the crossing of ellipsoid and plane is an ellipse: (s,t)P(s,t)+q*(s,t)+freeterm==0
    Eigen::Matrix<Scalar, 2, 2>& P = result->A;

    // s^2
    P(0, 0) =
        suriko::Sqr(Rt_u[0] / rot_ellip.semi_axes[0]) +
        suriko::Sqr(Rt_u[1] / rot_ellip.semi_axes[1]) +
        suriko::Sqr(Rt_u[2] / rot_ellip.semi_axes[2]);
    // t^2
    P(1, 1) =
        suriko::Sqr(Rt_v[0] / rot_ellip.semi_axes[0]) +
        suriko::Sqr(Rt_v[1] / rot_ellip.semi_axes[1]) +
        suriko::Sqr(Rt_v[2] / rot_ellip.semi_axes[2]);

    // s*t, divided by 2
    Scalar coef_st_half =
        Rt_u[0] * Rt_v[0] / suriko::Sqr(rot_ellip.semi_axes[0]) +
        Rt_u[1] * Rt_v[1] / suriko::Sqr(rot_ellip.semi_axes[1]) +
        Rt_u[2] * Rt_v[2] / suriko::Sqr(rot_ellip.semi_axes[2]);
    P(0, 1) = P(1, 0) = coef_st_half;

    //
    Scalar lam_ne = lam - n.transpose() * eye;
    Eigen::Matrix<Scalar, 3, 1> Rt_n = rot_ellip.world_from_ellipse.R.transpose() * n;
    Eigen::Matrix<Scalar, 3, 1> Rt_eyemT = rot_ellip.world_from_ellipse.R.transpose() * (eye - rot_ellip.world_from_ellipse.T);

    Scalar coef_s =
        (Rt_n[0] * lam_ne + Rt_eyemT[0]) * Rt_u[0] / suriko::Sqr(rot_ellip.semi_axes[0]) +
        (Rt_n[1] * lam_ne + Rt_eyemT[1]) * Rt_u[1] / suriko::Sqr(rot_ellip.semi_axes[1]) +
        (Rt_n[2] * lam_ne + Rt_eyemT[2]) * Rt_u[2] / suriko::Sqr(rot_ellip.semi_axes[2]);
    coef_s *= 2;

    Scalar coef_t =
        (Rt_n[0] * lam_ne + Rt_eyemT[0]) * Rt_v[0] / suriko::Sqr(rot_ellip.semi_axes[0]) +
        (Rt_n[1] * lam_ne + Rt_eyemT[1]) * Rt_v[1] / suriko::Sqr(rot_ellip.semi_axes[1]) +
        (Rt_n[2] * lam_ne + Rt_eyemT[2]) * Rt_v[2] / suriko::Sqr(rot_ellip.semi_axes[2]);
    coef_t *= 2;

    Eigen::Matrix<Scalar, 3, 1> Rt_eyepT = rot_ellip.world_from_ellipse.R.transpose() * (eye + rot_ellip.world_from_ellipse.T);

    Scalar coef_free =
        suriko::Sqr((-Rt_n[0] * lam_ne + Rt_eyepT[0]) / rot_ellip.semi_axes[0]) +
        suriko::Sqr((-Rt_n[1] * lam_ne + Rt_eyepT[1]) / rot_ellip.semi_axes[1]) +
        suriko::Sqr((-Rt_n[2] * lam_ne + Rt_eyepT[2]) / rot_ellip.semi_axes[2]);
    coef_free -= 1;

    Eigen::Matrix<Scalar, 2, 1> q { coef_s, coef_t };

    // transform ellipse form x*A*x+q*x+c==0 into (x-m)A(x-m)=r
    Eigen::Matrix<Scalar, 2, 1> center = P.colPivHouseholderQr().solve(-0.5 * q);
    Scalar right_side = center.transpose() * P * center - coef_free;

    // for convenience, make right side positive
    if (right_side < 0)
    {
        right_side = -right_side;
        P *= -1;
    }

    result->center = center;
    result->right_side = right_side;
    if (ValidateEllipse(*result))
        return true;

    // occur when ellipsoid 'behind' the camera
    bool ok = eye_inside_ellip;
    return false;
}

namespace internals
{
    Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 3>* rot_mat, const Eigen::Matrix<Scalar, 3, 1>* translation)
    {
        Eigen::Matrix<Scalar, 4, 4> result;
        Eigen::Matrix<Scalar, 3, 3> identity3 = Eigen::Matrix<Scalar, 3, 3>::Identity();
        if (rot_mat == nullptr)
            rot_mat = &identity3;
    
        Eigen::Matrix<Scalar, 3, 1> zero_translation(0, 0, 0);
        if (translation == nullptr)
            translation = &zero_translation;

        result.topLeftCorner(3, 3) = *rot_mat;
        result.topRightCorner(3, 1) = *translation;
        result(3, 0) = 0;
        result(3, 1) = 0;
        result(3, 2) = 0;
        result(3, 3) = 1;
        return result;
    }

    Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 3>& rot_mat, const Eigen::Matrix<Scalar, 3, 1>& translation)
    {
        return SE3Mat(&rot_mat, &translation);
    }

    Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 3>& rot_mat)
    {
        return SE3Mat(&rot_mat, nullptr);
    }

    Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 1>& translation)
    {
        return SE3Mat(nullptr, &translation);
    }

    Eigen::Matrix<Scalar, 3, 3> RotMat(const Eigen::Matrix<Scalar, 3, 1>& unity_dir, Scalar ang)
    {
        Eigen::Matrix<Scalar, 3, 3> result;
        if (!RotMatFromUnityDirAndAngle(unity_dir, ang, &result))
            result = Eigen::Matrix<Scalar, 3, 3>::Identity();
        return result;
    }

    Eigen::Matrix<Scalar, 3, 3> RotMat(Scalar unity_dir_x, Scalar unity_dir_y, Scalar unity_dir_z, Scalar ang)
    {
        Eigen::Matrix<Scalar, 3, 1> unity_dir(unity_dir_x, unity_dir_y, unity_dir_z);
        return RotMat(unity_dir, ang);
    }
}
}