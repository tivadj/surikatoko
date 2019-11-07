#include <string>
#include <cmath> // std::sqrt
#include <algorithm> // std::clamp
#include <glog/logging.h>
#include <gsl/span>
#include <Eigen/Cholesky>
#include "suriko/approx-alg.h"
#include "suriko/obs-geom.h"
#include "suriko/stat-helpers.h"

namespace suriko
{
Point3 Normalized(const Point3& p)
{
    Point3 dir = p;
    CHECK(Normalize(&dir));
    return dir;
}

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

Recti TruncateRect(const Rect& a)
{
    Recti result{
        static_cast<int>(a.x),
        static_cast<int>(a.y),
        static_cast<int>(a.width),
        static_cast<int>(a.height)
    };
    return result;
}

Recti EncompassRect(const Rect& a)
{
    auto x1 = static_cast<int>(std::floor(a.x));
    auto x2 = static_cast<int>(std::floor(a.Right()));
    auto y1 = static_cast<int>(std::floor(a.y));
    auto y2 = static_cast<int>(std::floor(a.Bottom()));
    return Recti{ x1, y1, x2 - x1, y2 - y1 };
}

// Ensures the size of the rectangle is at least of a given value, keeping the center intact.
Recti ClampRectWhenFixedCenter(const Recti& r, suriko::Sizei min_size)
{
    Recti result = r;
    if (result.width < min_size.width)
    {
        int expand_x = min_size.width - result.width;
        int expand_left_x = expand_x / 2;
        result.x -= expand_left_x;
        result.width = min_size.width;
    }

    if (result.height < min_size.height)
    {
        int expand_y = min_size.height - result.height;
        int expand_up_y = expand_y / 2;
        result.y -= expand_up_y;
        result.height = min_size.height;
    }
    return result;
}

auto SE3Inv(const SE3Transform& rt) -> SE3Transform {
    SE3Transform result;
    result.R = rt.R.transpose();
    result.T = - result.R * rt.T;
    return result;
}

auto SE2Apply(const SE2Transform& rt, const suriko::Point2f& x)->suriko::Point2f
{
    return suriko::Point2f { rt.R * x.Mat() + rt.T };
}

auto SE3Apply(const SE3Transform& rt, const suriko::Point3& x) -> suriko::Point3
{
    // 0-copy
    suriko::Point3 result = rt.R * x + rt.T;
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

auto SE3BFromA(const SE3Transform& a_from_world, const SE3Transform& b_from_world) -> suriko::SE3Transform
{
    return SE3Compose(SE3Inv(a_from_world), b_from_world);
}

FragmentMap::FragmentMap(size_t fragment_id_offset)
    : fragment_id_offset_(fragment_id_offset),
    next_salient_point_id_(fragment_id_offset + 1)
{
}

SalientPointFragment& FragmentMap::AddSalientPointTempl(const std::optional<suriko::Point3> &coord, size_t* salient_point_id)
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

void CornerTrack::AddCorner(size_t frame_ind, const suriko::Point2f& value)
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

std::optional<suriko::Point2f> CornerTrack::GetCorner(size_t frame_ind) const
{
    CHECK(StartFrameInd != -1);
    ptrdiff_t local_ind = frame_ind - StartFrameInd;

    if (local_ind < 0 || (size_t)local_ind >= CoordPerFramePixels.size())
        return std::optional<suriko::Point2f>();

    std::optional<CornerData> corner_data = CoordPerFramePixels[local_ind];
    if (!corner_data.has_value())
        return std::optional<suriko::Point2f>();
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
    Scalar rtol = 1.0e-3f;
    Scalar atol = 1.0e-3f;
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

    Scalar rtol = 1.0e-3f;
    Scalar atol = 1.0e-3f;
    bool det_one = IsClose(1.0f, rdet, rtol, atol);
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

bool IsSpecialOrthogonal(const Eigen::Matrix<Scalar,2,2>& R, std::string* msg) {
    bool is_ortho = IsOrthogonal(R, msg);
    if (!is_ortho)
        return false;

    Scalar rdet = R.determinant();

    Scalar rtol = 1.0e-3f;
    Scalar atol = 1.0e-3f;
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

void SkewSymmetricMat(const Point3& v, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> skew_mat)
{
    *skew_mat << 
        0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
}

auto RotMatFromUnityDirAndAngle(const Point3& unity_dir, Scalar ang, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> rot_mat, bool check_input) -> bool
{
    // skip precondition checking in Release mode on user request (check_input=false)
    if (check_input || kSurikoDebug)
    {
        // direction must be a unity vector
        Scalar dir_len = Norm(unity_dir);
        if (!IsClose(1, dir_len))
            return false;  // provide valid unity_dir

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

auto RotMatFromAxisAngle(const Point3& axis_angle, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> rot_mat) -> bool
{
    Scalar ang = Norm(axis_angle);
    if (IsClose(0, ang)) return false;

    Point3 unity_dir = axis_angle / ang;
    const bool check_input = false;
    return RotMatFromUnityDirAndAngle(unity_dir, ang, rot_mat, check_input);
}

auto LogSO3(const Eigen::Matrix<Scalar, 3, 3>& rot_mat, gsl::not_null<Point3*> unity_dir, gsl::not_null<Scalar*> ang, bool check_input) -> bool
{
    // skip precondition checking in Release mode on user request (check_input=false)
    if (check_input || kSurikoDebug)
    {
        std::string msg;
        bool ok = IsSpecialOrthogonal(rot_mat, &msg);
        CHECK(ok) << msg;
    }
    Scalar cos_ang = 0.5f*(rot_mat.trace() - 1);
    cos_ang = std::clamp<Scalar>(cos_ang, -1, 1); // the cosine may be slightly off due to rounding errors

    Scalar sin_ang = std::sqrt(1.0f - cos_ang * cos_ang);
    Scalar atol = 1e-3f;
    if (IsClose(0, sin_ang, 0, atol))
        return false;

    auto& udir = *unity_dir;
    udir[0] = rot_mat(2, 1) - rot_mat(1, 2);
    udir[1] = rot_mat(0, 2) - rot_mat(2, 0);
    udir[2] = rot_mat(1, 0) - rot_mat(0, 1);
    udir *= 0.5f / sin_ang;

    // direction vector is already close to unity, but due to rounding errors it diverges
    // TODO: check where the rounding error appears
    Scalar dirlen = Norm(udir);
    udir *= 1 / dirlen;

    *ang = std::acos(cos_ang);
    return true;
}

auto AxisAngleFromRotMat(const Eigen::Matrix<Scalar, 3, 3>& rot_mat, gsl::not_null<Point3*> dir) -> bool
{
    Point3 unity_dir;
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
    Point3 q = ToPoint3(proj_mat.rightCols(1));

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
    Point3 t = -Q_inv * q;

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

auto Triangulate3DPointByLeastSquares(const std::vector<suriko::Point2f> &xs2D, const std::vector<Eigen::Matrix<Scalar,3,4>> &proj_mat_list, Scalar f0) -> suriko::Point3
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
    const Point3& eye,
    const Point3& center,
    const Point3& up)
{
    // align OZ with view direction
    Point3 forward_dir = center - eye;
    CHECK(Normalize(&forward_dir));

    // align OY to match up vector
    // cam_up = up - proj(up onto forward_dir)
    Point3 cam_up_dir = up - forward_dir * Dot(up, forward_dir); // new OY
    CHECK(Normalize(&cam_up_dir));

    SE3Transform world_from_cam;
    world_from_cam.R.middleCols<1>(0) = Mat(Cross(cam_up_dir, forward_dir));
    world_from_cam.R.middleCols<1>(1) = Mat(cam_up_dir);
    world_from_cam.R.middleCols<1>(2) = Mat(forward_dir);
    world_from_cam.T = eye;
    return world_from_cam;
}

Rect GetEllipseBounds2(const RotatedEllipse2D& rotated_ellipse)
{
    Scalar r1 = rotated_ellipse.world_from_ellipse.R(0, 0);
    Scalar r2 = rotated_ellipse.world_from_ellipse.R(0, 1);
    Scalar r3 = rotated_ellipse.world_from_ellipse.R(1, 0);
    Scalar r4 = rotated_ellipse.world_from_ellipse.R(1, 1);
    Scalar a = rotated_ellipse.semi_axes[0];
    Scalar b = rotated_ellipse.semi_axes[1];

    Scalar den_abs = std::abs(r1 * r4 - r2 * r3);

    Scalar x1, x2;
    {
        Scalar discr_sqrt = std::sqrt(b * b * r3 * r3 + a * a * r4 * r4);
        x1 = rotated_ellipse.world_from_ellipse.T[0] - discr_sqrt / den_abs;
        x2 = rotated_ellipse.world_from_ellipse.T[0] + discr_sqrt / den_abs;
    }
    Scalar y1, y2;
    {
        Scalar discr_sqrt = std::sqrt(b * b * r1 * r1 + a * a * r2 * r2);
        y1 = rotated_ellipse.world_from_ellipse.T[1] - discr_sqrt / den_abs;
        y2 = rotated_ellipse.world_from_ellipse.T[1] + discr_sqrt / den_abs;
    }

    Rect result;
    result.x = x1;
    result.width = x2 - x1;
    result.y = y1;
    result.height = y2 - y1;
    return result;
}

std::tuple<bool, RotatedEllipsoid3D> GetRotatedUncertaintyEllipsoidFromCovMat(const Eigen::Matrix<Scalar, 3, 3>& cov, const Point3& mean,
    Scalar covar3D_to_ellipsoid_chi_square)
{
    // check symmetry
    Scalar sym_diff = (cov - cov.transpose()).norm();
    SRK_ASSERT(IsCloseAbs(0, sym_diff, 0.001f));

    //
    // A=V*D*inv(V)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 3, 3>> eigen_solver(cov);
    bool op = eigen_solver.info() == Eigen::Success;
    SRK_ASSERT(op);

    Eigen::Matrix<Scalar, 3, 3> eig_vecs = eigen_solver.eigenvectors(); // rot_mat_ellipse_from_world
    Eigen::Matrix<Scalar, 3, 1> dd = eigen_solver.eigenvalues();

    // each semi-axis of an ellipse must be positive
    for (int i=0; i<dd.rows(); ++i)
    {
        Scalar val = dd[i];
        // Fix small errors when semi-axis is a small negative number,
        // which may occur when dealing with zero-covariance variables.
        if (val < 0)
        {
            if (!IsClose(0, val)) return std::make_tuple(false, RotatedEllipsoid3D{});
            dd[i] = 0;
        }
    }
    
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
    auto result = RotatedEllipsoid3D{};
    auto& R = result.world_from_ellipse.R;
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
        SRK_ASSERT(IsCloseAbs(0, should_zero, 0.001));
    }

    // right side of the ellipse equation: (x-mu)A(x-mu)=right_side
    Scalar right_side = covar3D_to_ellipsoid_chi_square;

    // order semi-axes from max to min
    auto& semi = result.semi_axes;
    semi[0] = std::sqrt(right_side * dd[major_col_ind]);
    semi[1] = std::sqrt(right_side * dd[1]);
    semi[2] = std::sqrt(right_side * dd[minor_col_ind]);

    result.world_from_ellipse.T = mean;

    SRK_ASSERT(IsFinite(semi[0]));
    SRK_ASSERT(IsFinite(semi[1]));
    SRK_ASSERT(IsFinite(semi[2]));
    return std::make_tuple(true, result);
}

// Note, on why the direct conversion is used: covariance_matrix -> rotated_ellipse=(semi_axes,R).
// We can calculate rotated ellipse from covariance matrix in multiple steps in such a sequence:
// covariance_matrix -> ellipse -> rotated_ellipse
// where ellipse=(x-mu)A(x-mu) and rotated_ellipse=(semi_axes,R)
// R[2x2]=rotation matrix to get world coordinates from rotated ellipse coordinates
// But it requires calculation of A=inv(Cov) which may not be stable.
// Instead we eigen-decompose Cov=V*D*inv(V), and inverse the diagonal matrix, which is more stable.
// Thus skipping the intermediate step improves the stablility.
bool Get2DRotatedEllipseFromCovMat(const Eigen::Matrix<Scalar, 2, 2>& cov,
    Scalar covar2D_to_ellipse_confidence,
    Eigen::Matrix<Scalar, 2, 1>* semi_axes,
    Eigen::Matrix<Scalar, 2, 2>* world_from_ellipse)
{
    SRK_ASSERT(covar2D_to_ellipse_confidence >= 0 && covar2D_to_ellipse_confidence < 1);
    static constexpr size_t kDim = 2;

    // check symmetry
    Scalar sym_diff = (cov - cov.transpose()).norm();
    SRK_ASSERT(IsCloseAbs(0, sym_diff, 0.001));

    //
    // A=V*D*inv(V)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, kDim, kDim>> eigen_solver(cov);
    bool op = eigen_solver.info() == Eigen::Success;
    SRK_ASSERT(op);

    Eigen::Matrix<Scalar, kDim, kDim> eig_vecs = eigen_solver.eigenvectors(); // rot_mat_ellipse_from_world
    Eigen::Matrix<Scalar, kDim, 1> dd = eigen_solver.eigenvalues();

    // each semi-axis of an ellipse must be positive
    for (int i = 0; i < dd.rows(); ++i)
    {
        Scalar val = dd[i];
        // Fix small errors when semi-axis is a small negative number,
        // which may occur when dealing with zero-covariance variables.
        if (val < 0)
        {
            if (!IsClose(0, val)) return false;
            dd[i] = 0;
        }
    }

    // Eigen::SelfAdjointEigenSolver sorts eigenvalues in ascending order
    int major_col_ind = kDim - 1;
    int minor_col_ind = 0;

    if (dd[minor_col_ind] == dd[major_col_ind])
    {
        // all eigenvalues are equal and may be treated as already sorted in decreasing order;
        // no reordering is required
        std::swap(major_col_ind, minor_col_ind);
    }

    // order semi-axes from max to min

    auto& R = *world_from_ellipse;
    R.middleCols<1>(0) = eig_vecs.middleCols<1>(major_col_ind);
    R.middleCols<1>(1) = eig_vecs.middleCols<1>(minor_col_ind);

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

    Scalar right_side = 2 * std::log(1 / (1 - covar2D_to_ellipse_confidence));

    // note multiplication (not division) in a=sqrt(rs*lam) as we skip calculating inverse of covariance matrix
    // and directly calculate inverse of diagonal D matrix in Sig=V*D*inv(V)
    // order semi-axes from max to min
    auto& semi = *semi_axes;
    semi[0] = std::sqrt(right_side * dd[major_col_ind]);
    semi[1] = std::sqrt(right_side * dd[minor_col_ind]);
    SRK_ASSERT(IsFinite(semi[0]));
    SRK_ASSERT(IsFinite(semi[1]));
    return true;
}

std::tuple<bool,RotatedEllipse2D> Get2DRotatedEllipseFromCovMat(
    const Eigen::Matrix<Scalar, 2, 2>& covar,
    const Eigen::Matrix<Scalar, 2, 1>& mean,
    Scalar covar2D_to_ellipse_confidence)
{
    Eigen::Matrix<Scalar, 2, 1> semi_axes;
    Eigen::Matrix<Scalar, 2, 2> world_from_ellipse;
    if (!Get2DRotatedEllipseFromCovMat(covar, covar2D_to_ellipse_confidence, &semi_axes, &world_from_ellipse))
        return std::make_tuple(false, RotatedEllipse2D{});

    SE2Transform wfe{ world_from_ellipse , mean };
    RotatedEllipse2D result{ semi_axes , wfe };
    return std::make_tuple(true, result);
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
        SRK_ASSERT(IsCloseAbs(0, should_zero, 0.1));
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

namespace internals
{
    Eigen::Matrix<Scalar, 4, 4> SE3Mat(std::optional<Eigen::Matrix<Scalar, 3, 3>> rot_mat, std::optional<Point3> translation)
    {
        if (!rot_mat.has_value())
            rot_mat = rot_mat = Eigen::Matrix<Scalar, 3, 3>::Identity();
        if (!translation.has_value())
            translation = Point3{ 0, 0, 0 };

        Eigen::Matrix<Scalar, 4, 4> result;
        result.topLeftCorner(3, 3) = rot_mat.value();
        result.topRightCorner(3, 1) = Mat(translation.value());
        result(3, 0) = 0;
        result(3, 1) = 0;
        result(3, 2) = 0;
        result(3, 3) = 1;
        return result;
    }

    Eigen::Matrix<Scalar, 3, 3> RotMat(const Point3& unity_dir, Scalar ang)
    {
        Eigen::Matrix<Scalar, 3, 3> result;
        if (!RotMatFromUnityDirAndAngle(unity_dir, ang, &result))
            result = Eigen::Matrix<Scalar, 3, 3>::Identity();
        return result;
    }

    Eigen::Matrix<Scalar, 3, 3> RotMat(Scalar unity_dir_x, Scalar unity_dir_y, Scalar unity_dir_z, Scalar ang)
    {
        Point3 unity_dir(unity_dir_x, unity_dir_y, unity_dir_z);
        return RotMat(unity_dir, ang);
    }
}

// Calculates difference between two trajectories, aka Relative Pose Error (RPE).
// source: "A benchmark for the evaluation of RGB-D SLAM systems", Sturm, 2012.
bool CalcRelativePoseError(
    gsl::span<const std::optional<SE3Transform>> ground_cfw,
    gsl::span<const std::optional<SE3Transform>> estim_cfw,
    std::vector<Scalar>* errs)
{
    if (ground_cfw.size() != estim_cfw.size()) return false;

    std::optional<SE3Transform> prev_expect_opt = !ground_cfw.empty() ? ground_cfw[0] : std::nullopt;
    std::optional<SE3Transform> prev_actual_opt = !estim_cfw.empty() ? estim_cfw[0] : std::nullopt;
    size_t comm_size = std::min(ground_cfw.size(), estim_cfw.size());
    for (size_t i = 1; i < comm_size; ++i)
    {
        const std::optional<SE3Transform>& expect_opt = ground_cfw[i];
        const std::optional<SE3Transform>& actual_opt = estim_cfw[i];
        if (expect_opt.has_value() && actual_opt.has_value() &&
            prev_expect_opt.has_value() && prev_actual_opt.has_value())
        {
            // formula (1) and (2)
            SE3Transform expect_cur_from_prev = SE3BFromA(expect_opt.value(), prev_expect_opt.value());
            SE3Transform actual_cur_from_prev = SE3BFromA(actual_opt.value(), prev_actual_opt.value());
            SE3Transform rel_err = SE3BFromA(expect_cur_from_prev, actual_cur_from_prev);
            Scalar err2 = rel_err.T.squaredNorm();
            Scalar err = std::sqrt(err2);
            errs->push_back(err);
        }
        prev_expect_opt = expect_opt;
        prev_actual_opt = actual_opt;
    }
    return true;
}

auto CalcTrajectoryErrStats(const std::vector<Scalar>& errs) -> std::optional<ErrWithMoments>
{
    if (errs.empty()) return std::nullopt;

    Scalar err2_sum = 0;
    MeanStdAlgo stat_calc;
    for (Scalar err : errs)
    {
        Scalar err2 = suriko::Sqr(err);
        err2_sum += err2;
        stat_calc.Next(err);
    }

    Scalar mean_of_squares = err2_sum / errs.size();
    Scalar rmse = std::sqrt(mean_of_squares);

    std::vector<Scalar> nums_workspace;
    std::optional<Scalar> median = LeftMedian(errs, &nums_workspace);

    ErrWithMoments val_moms;
    val_moms.median = median.value();
    val_moms.mean = stat_calc.Mean();
    val_moms.std = stat_calc.Std();
    val_moms.min = stat_calc.Min().value();
    val_moms.max = stat_calc.Max().value();
    val_moms.rmse = rmse;
    return val_moms;
}

Scalar CalcTrajectoryLength(
    const std::vector<std::optional<SE3Transform>>* cam_cfw_opt,
    const std::vector<SE3Transform>* cam_cfw)
{
    CHECK(cam_cfw != nullptr ^ cam_cfw_opt != nullptr);

    std::optional<SE3Transform> prev_opt = cam_cfw_opt != nullptr ? (*cam_cfw_opt)[0] : (*cam_cfw)[0];
    size_t size = cam_cfw_opt != nullptr ? cam_cfw_opt->size() : cam_cfw->size();
    Scalar shifts_sum = 0;
    for (size_t i = 1; i < size; ++i)
    {
        const std::optional<SE3Transform>& cur_opt = cam_cfw_opt != nullptr ? (*cam_cfw_opt)[i] : (*cam_cfw)[i];
        if (cur_opt.has_value() && prev_opt.has_value())
        {
            SE3Transform cur_from_prev = SE3BFromA(prev_opt.value(), cur_opt.value());
            Scalar s = cur_from_prev.T.norm();
            shifts_sum += s;
        }
        prev_opt = cur_opt;
    }
    return shifts_sum;
}
}
