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

auto SE3Inv(const SE3Transform& rt) -> SE3Transform {
    SE3Transform result;
    result.R = rt.R.transpose();
    result.T = - result.R * rt.T;
    return result;
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

void FragmentMap::AddSalientPoint(size_t point_track_id, const std::optional<suriko::Point3> &coord)
{
    if (point_track_id >= salient_points.size())
        salient_points.resize(point_track_id+1);
    if (coord.has_value())
        SetSalientPoint(point_track_id, coord.value());
    salient_points_count += 1;
}

SalientPointFragment& FragmentMap::AddSalientPointNew3(const std::optional<suriko::Point3> &coord, size_t* salient_point_id)
{
    size_t new_id = next_salient_point_id_++;

    size_t new_ind1 = salient_points.size();
    size_t new_ind2 = new_id - fragment_id_offset_ - 1;
    SRK_ASSERT(new_ind1 == new_ind2);

    if (salient_point_id != nullptr)
        *salient_point_id = new_id;

    salient_points.resize(salient_points.size() + 1);
    
    SalientPointFragment& frag = salient_points.back();
    frag.Coord = coord;
    salient_points_count += 1;
    return frag;
}

void FragmentMap::SetSalientPoint(size_t point_track_id, const suriko::Point3 &coord)
{
    SRK_ASSERT(point_track_id < salient_points.size());
    SalientPointFragment& frag = salient_points[point_track_id];
    frag.Coord = coord;
}

void FragmentMap::SetSalientPointNew(size_t fragment_id, const std::optional<suriko::Point3> &coord, std::optional<size_t> syntheticVirtualPointId)
{
    SRK_ASSERT(fragment_id < salient_points.size());
    SalientPointFragment& frag = salient_points[fragment_id];
    frag.SyntheticVirtualPointId = syntheticVirtualPointId;
    frag.Coord = coord;
}

const suriko::Point3& FragmentMap::GetSalientPoint(size_t salient_point_id) const
{
    size_t ind = SalientPointIdToInd(salient_point_id);
    CHECK(ind < salient_points.size());
    const std::optional<suriko::Point3>& sal_pnt = salient_points[ind].Coord;
    SRK_ASSERT(sal_pnt.has_value());
    return sal_pnt.value();
}

suriko::Point3& FragmentMap::GetSalientPoint(size_t salient_point_id)
{
    size_t ind = SalientPointIdToInd(salient_point_id);
    CHECK(ind < salient_points.size());
    std::optional<suriko::Point3>& sal_pnt = salient_points[ind].Coord;
    SRK_ASSERT(sal_pnt.has_value());
    return sal_pnt.value();
}
bool FragmentMap::GetSalientPointByVirtualPointIdInternal(size_t salient_point_id, const SalientPointFragment** fragment)
{
    *fragment = nullptr;
    for (const SalientPointFragment& p : salient_points)
    {
        if (p.SyntheticVirtualPointId == salient_point_id)
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
    corner_data.PixelCoord = value;
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
    if (local_ind >= CoordPerFramePixels.size())
    {
        CornerData corner_data;
        CoordPerFramePixels.push_back(std::optional<CornerData>(corner_data));
        CheckConsistent();
    }
    return CoordPerFramePixels.back().value();
}

std::optional<suriko::Point2> CornerTrack::GetCorner(size_t frame_ind) const
{
    CHECK(StartFrameInd != -1);
    ptrdiff_t local_ind = frame_ind - StartFrameInd;

    if (local_ind < 0 || (size_t)local_ind >= CoordPerFramePixels.size())
        return std::optional<suriko::Point2>();
    return CoordPerFramePixels[local_ind].value().PixelCoord;
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

bool IsSpecialOrthogonal(const Eigen::Matrix<Scalar,3,3>& R, std::string* msg) {
    Scalar rtol = 1.0e-3;
    Scalar atol = 1.0e-3;
    bool is_ident = (R.transpose() * R).isIdentity(atol);
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
    Scalar rdet = R.determinant();
    bool is_one = IsClose(1, rdet, rtol, atol);
    if (!is_one)
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
        if (!IsClose(1, dir_len)) return false;

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
    Scalar cos_ang = 0.5*(rot_mat.trace() - 1);
    cos_ang = std::clamp<Scalar>(cos_ang, -1, 1); // the cosine may be slightly off due to rounding errors

    Scalar sin_ang = std::sqrt(1.0 - cos_ang * cos_ang);
    Scalar atol = 1e-3;
    if (IsClose(0, sin_ang, 0, atol))
        return false;

    auto& udir = *unity_dir;
    udir[0] = rot_mat(2, 1) - rot_mat(1, 2);
    udir[1] = rot_mat(0, 2) - rot_mat(2, 0);
    udir[2] = rot_mat(1, 0) - rot_mat(0, 1);
    udir *= 0.5 / sin_ang;

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