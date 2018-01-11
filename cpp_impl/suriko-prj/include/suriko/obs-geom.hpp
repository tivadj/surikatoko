#pragma once
#include <array>
#include <tuple>
#include <vector>
#include <optional>
#include <cmath> // std::sqrt
#include <Eigen/Cholesky>
#include <Eigen/Dense> // Eigen::Matrix
#include "suriko/approx-alg.hpp"
#include "suriko/rt-config.h"

namespace suriko
{
template <class Scalar>
class Point2 {
    Eigen::Matrix<Scalar,2,1> mat_;
public:
    Point2(const Eigen::Matrix<Scalar, 2, 1> &m) : mat_(m) {}

    template <typename F0, typename F1>
    Point2(const F0 &x, const F1 &y) {
        mat_(0) = x;
        mat_(1) = y;
    }

    const Eigen::Matrix<Scalar,2,1>& Mat() const { return mat_; };
          Eigen::Matrix<Scalar,2,1>& Mat()       { return mat_; };

    Scalar  operator[] (size_t i) const { return mat_(i); };
    Scalar& operator[] (size_t i)       { return mat_(i); };
};

//template <typename Scalar>
//auto ToPoint(const Eigen::Matrix<Scalar,2,1>& m) { return suriko::Point2<Scalar>(m); }

template <class Scalar>
class Point3 {
    Eigen::Matrix<Scalar,3,1> mat_;
public:
    Point3(const Eigen::Matrix<Scalar, 3, 1> &m) : mat_(m) { }

    template <typename F0, typename F1, typename F2>
    Point3(const F0 &x, const F1 &y, const F2 &z) {
        mat_(0) = x;
        mat_(1) = y;
        mat_(2) = z;
    }

    const Eigen::Matrix<Scalar,3,1>& Mat() const { return mat_; };
          Eigen::Matrix<Scalar,3,1>& Mat()       { return mat_; };

    Scalar  operator[] (size_t i) const { return mat_(i); };
    Scalar& operator[] (size_t i)       { return mat_(i); };
};

template <typename Scalar>
auto ToPoint(const Eigen::Matrix<Scalar,3,1>& m) { return suriko::Point3<Scalar>(m); }

    /// SE3=Special Euclidean transformation in 3D.
/// Direct camera movement transforms 3D points from camera frame into world frame.
/// Inverse camera movement transforms 3D points from world frame into camera frame.
template <class Scalar>
struct SE3Transform
{
    Eigen::Matrix<Scalar, 3, 1> T;
    Eigen::Matrix<Scalar, 3, 3> R;

    SE3Transform() = default;
    SE3Transform(const Eigen::Matrix<Scalar, 3, 3>& R, const Eigen::Matrix<Scalar, 3, 1>& T) : R(R), T(T) {}
};

template <typename Scalar>
auto SE3Inv(const SE3Transform<Scalar>& rt) -> SE3Transform<Scalar>
{
    SE3Transform<Scalar> result;
    result.R = rt.R.transpose();
    result.T = - result.R * rt.T;
    return result;
}

template <typename Scalar>
auto SE3Apply(const SE3Transform<Scalar>& rt, const suriko::Point3<Scalar>& x) -> suriko::Point3<Scalar>
{
    // 0-copy
    suriko::Point3<Scalar> result(0,0,0);
    result.Mat() = rt.R * x.Mat() + rt.T;
    return result;
    // 1-copy
//    Eigen::Matrix<Scalar,3,1> result= rt.R * x.Mat() + rt.T;
//    return ToPoint(result);
}

template <typename Scalar>
auto SE3Compose(const SE3Transform<Scalar>& rt1, const SE3Transform<Scalar>& rt2) -> suriko::SE3Transform<Scalar>
{
    SE3Transform<Scalar> result;
    result.R = rt1.R * rt2.R;
    result.T = rt1.R * rt2.T + rt1.T;
    return result;
}

template <typename Scalar>
auto SE3AFromB(const SE3Transform<Scalar>& a_from_world, const SE3Transform<Scalar>& b_from_world) -> suriko::SE3Transform<Scalar>
{
    return SE3Compose(a_from_world, SE3Inv(b_from_world));
}

/// The space with salient 3D points.
template <typename Scalar>
class FragmentMap
{
	size_t point_track_count = 0;
    std::vector<std::optional<suriko::Point3<Scalar>>> salient_points;
public:
    void AddSalientPoint(size_t point_track_id, const std::optional<suriko::Point3<Scalar>> &value)
    {
        if (point_track_id >= salient_points.size())
            salient_points.resize(point_track_id+1);
        if (value.has_value())
            SetSalientPoint(point_track_id, value.value());
        point_track_count += 1;
    }
    void SetSalientPoint(size_t point_track_id, const suriko::Point3<Scalar> &value)
    {
        assert(point_track_id < salient_points.size());
        salient_points[point_track_id] = value;
    }
    suriko::Point3<Scalar> GetSalientPoint(size_t point_track_id) const
    {
        assert(point_track_id < salient_points.size());
        std::optional<suriko::Point3<Scalar>> sal_pnt = salient_points[point_track_id];
        SRK_ASSERT(sal_pnt.has_value());
        return sal_pnt.value();
    }
    size_t PointTrackCount() const { return point_track_count; }
};

template <class Scalar>
class CornerTrack
{
    ptrdiff_t StartFrameInd = -1;
    std::vector<std::optional<suriko::Point2<Scalar>>> CoordPerFramePixels;
public:
    size_t TrackId;
    size_t SyntheticSalientPointId; // only available for artificially generated scenes where world's 3D points are known
public:
    CornerTrack() = default;

    bool HasCorners() const {
        return StartFrameInd != -1;
    }

    void AddCorner(size_t frame_ind, const suriko::Point2<Scalar>& value)
    {
        if (StartFrameInd == -1)
            StartFrameInd = frame_ind;
        else
            SRK_ASSERT(frame_ind >= StartFrameInd && "Can insert points later than the initial (start) frame");

        CoordPerFramePixels.push_back(std::optional<suriko::Point2<Scalar>>(value));
        CheckConsistent();
    }

    std::optional<suriko::Point2<Scalar>> GetCorner(size_t frame_ind) const
    {
        SRK_ASSERT(StartFrameInd != -1);
        ptrdiff_t local_ind = frame_ind - StartFrameInd;
        if (local_ind < 0 || local_ind >= CoordPerFramePixels.size())
            return std::optional<suriko::Point2<Scalar>>();
        return CoordPerFramePixels[local_ind];
    }
private:
    void CheckConsistent()
    {
        if (StartFrameInd != -1)
            SRK_ASSERT(!CoordPerFramePixels.empty());
        else
            SRK_ASSERT(CoordPerFramePixels.empty());
    }
};

template <class Scalar>
class CornerTrackRepository
{
public:
	std::vector<suriko::CornerTrack<Scalar>> CornerTracks;

    suriko::CornerTrack<Scalar>& GetByPointId(size_t point_id)
    {
        size_t pnt_ind = point_id;
        return CornerTracks[pnt_ind];
    }

    void PopulatePointTrackIds(std::vector<size_t> *result) {
        for (size_t pnt_ind=0;pnt_ind<CornerTracks.size(); ++pnt_ind)
            result->push_back(pnt_ind);
    }
};


/// Checks if Rt*R=I and det(R)=1.
template <typename Scalar>
bool IsSpecialOrthogonal(const Eigen::Matrix<Scalar,3,3>& R, std::string* msg = nullptr) {
    Scalar rtol = 1.0e-3;
    Scalar atol = 1.0e-3;
    bool is_ident = (R.transpose() * R).isIdentity(atol);
    if (!is_ident) {
        if (msg != nullptr) {
            std::stringstream ss;
            ss << "failed Rt.R=I, R=\n" << R;
            *msg = ss.str();
        }
        return false;
    }
    Scalar rdet = R.determinant();
    bool is_one = IsClose(1, rdet, rtol, atol);
    if (!is_one) {
        if (msg != nullptr) {
            std::stringstream ss;
            ss << "failed det(R)=1, actual detR=" << rdet << " R=\n" << R;
            *msg = ss.str();
        }
        return false;
    }
    return true;
}

//template <class F>
//class SalientPointTracker
//{
//public:
//	bool ComputeInplace(const std::vector<suriko::Point3<F>>& salient_points, const std::vector<SE3Transform<F>>& cam_inverse_orient,
//		const SalientPointTrackRepository<F>& track_repo)
//	{
//		return true;
//	}
//};

/// Decomposes P[3x4] -> R[3x3],T[3],K[3x3] so that P=scale*K*Rt*[I|-T]
/// where K=matrix of intrinsic parameters
/// where R,T = euclidian motion from camera to world coordinates
/// source: "Bundle adjustment for 3-d reconstruction" Appendix A, Kanatani Sugaya 2010
template <typename Scalar>
auto DecomposeProjMat(const Eigen::Matrix<Scalar, 3, 4> &proj_mat, bool check_post_cond = true)
-> std::tuple<Scalar, Eigen::Matrix<Scalar, 3, 3>, SE3Transform<Scalar>>
{
    using namespace Eigen;
    typedef Matrix<Scalar,3,3> Mat33;

    // copy the input, because we may flip sign later
    Mat33 Q = proj_mat.leftCols(3);
    Matrix<Scalar,3,1> q = proj_mat.rightCols(1);

    // ensure that R will have positive determinant
    int P_sign = 1;
    Scalar Q_det = Q.determinant();
    if (Q_det < 0) {
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
    LLT<Mat33> llt(QQt_inv); // Cholesky decomposition
    Mat33 C = llt.matrixL();

    // we need upper triangular matrix, but Eigen::LLT returns lower triangular
    C.transposeInPlace();

    Mat33 R = (C * Q).transpose();

    if (check_post_cond)
    {
        std::string err_msg;
        if (!IsSpecialOrthogonal(R, &err_msg))
            std::cerr <<err_msg <<std::endl;
    }

    // find intrinsic parameters K
    Mat33 C_inv = C.inverse();
    Scalar c_last = C_inv(2, 2);
    //assert not np.isclose(0, c_last), "division by zero, c_last={}".format(c_last)
    Mat33 K = C_inv * (1/c_last);

    Scalar scale_factor = P_sign * c_last;

    if (check_post_cond)
    {
        Eigen::Matrix<Scalar, 3, 4> right;
        right <<Mat33::Identity(), -t;
        Eigen::Matrix<Scalar, 3, 4> P_back = scale_factor * K * R.transpose() * right;
        auto diff = (proj_mat - P_back).norm();
        assert(diff < 1e-2 && "Failed to decompose P[3x4]->R,T,K"); // (diff={})
    }

    SE3Transform<Scalar> direct_orient_cam(R,t);
    return std::make_tuple(scale_factor, K, direct_orient_cam);
}

/// Finds the 3D coordinate of a world point from a list of corresponding 2D pixels in multiple images.
/// The orientation of the camera for each shot is specified in the list of projection matrices.
template <typename Scalar>
auto Triangulate3DPointByLeastSquares(const std::vector<suriko::Point2<Scalar>> &xs2D,
                                 const std::vector<Eigen::Matrix<Scalar,3,4>> &proj_mat_list, Scalar f0, int debug)
    -> suriko::Point3<Scalar>
{
    size_t frames_count_P = proj_mat_list.size();
    size_t frames_count_xs = xs2D.size();
    SRK_ASSERT(frames_count_P == frames_count_xs && "Provide two lists of 2D coordinates and projection matrices of the same length");

    size_t frames_count = frames_count_P;
    SRK_ASSERT(frames_count >= 2 && "Provide 2 or more projections of a 3D point");

    // populate matrices A and B to solve for least squares
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> A(frames_count * 2, 3);
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> B(frames_count * 2);

    for (size_t frame_ind = 0; frame_ind < frames_count; ++frame_ind) {
        const auto &x2D = xs2D[frame_ind];
        auto x = x2D[0];
        auto y = x2D[1];
        const auto &P = proj_mat_list[frame_ind];
        A(frame_ind * 2 + 0, 0) = x * P(2, 0) - f0 * P(0, 0);
        A(frame_ind * 2 + 0, 1) = x * P(2, 1) - f0 * P(0, 1);
        A(frame_ind * 2 + 0, 2) = x * P(2, 2) - f0 * P(0, 2);
        A(frame_ind * 2 + 1, 0) = y * P(2, 0) - f0 * P(1, 0);
        A(frame_ind * 2 + 1, 1) = y * P(2, 1) - f0 * P(1, 1);
        A(frame_ind * 2 + 1, 2) = y * P(2, 2) - f0 * P(1, 2);
        B(frame_ind * 2 + 0) = -(x * P(2, 3) - f0 * P(0, 3));
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

    if (debug >= 4) {
        const Eigen::Matrix<Scalar,Eigen::Dynamic,1> diff_vec = A * sol - B;
        float diff = diff_vec.norm();
        if (diff > 0.1) {
            std::cout << "warn: big diff=" << diff << " frames_count=" << frames_count << std::endl;
        }
    }
    suriko::Point3<Scalar> x3D(sol(0), sol(1), sol(2));
    return x3D;
}
}