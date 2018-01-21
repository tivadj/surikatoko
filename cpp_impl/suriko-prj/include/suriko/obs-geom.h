#pragma once
#include <array>
#include <tuple>
#include <vector>
#include <optional>
#include <Eigen/Dense> // Eigen::Matrix
#include "suriko/rt-config.h"

namespace suriko
{
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

//auto ToPoint(const Eigen::Matrix<Scalar,2,1>& m) -> suriko::Point2;

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

//auto ToPoint(const Eigen::Matrix<Scalar,3,1>& m) -> suriko::Point3;

    /// SE3=Special Euclidean transformation in 3D.
/// Direct camera movement transforms 3D points from camera frame into world frame.
/// Inverse camera movement transforms 3D points from world frame into camera frame.
struct SE3Transform
{
    Eigen::Matrix<Scalar, 3, 1> T;
    Eigen::Matrix<Scalar, 3, 3> R;

    SE3Transform() = default;
    SE3Transform(const Eigen::Matrix<Scalar, 3, 3>& R, const Eigen::Matrix<Scalar, 3, 1>& T) : R(R), T(T) {}
};

auto SE3Inv(const SE3Transform& rt) -> SE3Transform;
auto SE3Apply(const SE3Transform& rt, const suriko::Point3& x) -> suriko::Point3;
auto SE3Compose(const SE3Transform& rt1, const SE3Transform& rt2) -> suriko::SE3Transform;
auto SE3AFromB(const SE3Transform& a_from_world, const SE3Transform& b_from_world) -> suriko::SE3Transform;

/// The space with salient 3D points.
class FragmentMap
{
	size_t point_track_count = 0;
    std::vector<std::optional<suriko::Point3>> salient_points;
public:
    void AddSalientPoint(size_t point_track_id, const std::optional<suriko::Point3> &value);

    void SetSalientPoint(size_t point_track_id, const suriko::Point3 &value);

    const suriko::Point3& GetSalientPoint(size_t point_track_id) const;

    size_t PointTrackCount() const { return point_track_count; }
};

class CornerTrack
{
    ptrdiff_t StartFrameInd = -1;
    std::vector<std::optional<suriko::Point2>> CoordPerFramePixels;
public:
    size_t TrackId;
    size_t SyntheticSalientPointId; // only available for artificially generated scenes where world's 3D points are known
public:
    CornerTrack() = default;

    bool HasCorners() const;

    void AddCorner(size_t frame_ind, const suriko::Point2& value);

    std::optional<suriko::Point2> GetCorner(size_t frame_ind) const;
private:
    void CheckConsistent();
};

class CornerTrackRepository
{
public:
	std::vector<suriko::CornerTrack> CornerTracks;

    const suriko::CornerTrack& GetPointTrackById(size_t point_track_id) const;
          suriko::CornerTrack& GetPointTrackById(size_t point_track_id);

    void PopulatePointTrackIds(std::vector<size_t> *result);

    void IteratePointsMarker() const {}
};


/// Checks if Rt*R=I and det(R)=1.
bool IsSpecialOrthogonal(const Eigen::Matrix<Scalar,3,3>& R, std::string* msg = nullptr);

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
auto DecomposeProjMat(const Eigen::Matrix<Scalar, 3, 4> &proj_mat, bool check_post_cond = true)
-> std::tuple<Scalar, Eigen::Matrix<Scalar, 3, 3>, SE3Transform>;

/// Finds the 3D coordinate of a world point from a list of corresponding 2D pixels in multiple images.
/// The orientation of the camera for each shot is specified in the list of projection matrices.
auto Triangulate3DPointByLeastSquares(const std::vector<suriko::Point2> &xs2D,
                                 const std::vector<Eigen::Matrix<Scalar,3,4>> &proj_mat_list, Scalar f0, int debug)
    -> suriko::Point3;
}