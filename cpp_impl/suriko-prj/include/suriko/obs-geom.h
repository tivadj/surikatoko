#pragma once
#include <array>
#include <tuple>
#include <vector>
#include <optional>
#include <gsl/pointers> // gsl::not_null
#include <Eigen/Dense> // Eigen::Matrix
#include "suriko/rt-config.h"

namespace suriko
{
class Point2 {
    Eigen::Matrix<Scalar,2,1> mat_;
public:
    Point2() = default;
    Point2(const Eigen::Matrix<Scalar, 2, 1> &m) : mat_(m) {}

    template <typename F0, typename F1>
    Point2(const F0 &x, const F1 &y) {
        mat_(0) = x;
        mat_(1) = y;
    }

    const Eigen::Matrix<Scalar,2,1>& Mat() const { return mat_; };
          Eigen::Matrix<Scalar,2,1>& Mat()       { return mat_; };

    Eigen::Matrix<Scalar, 3, 1> AsHomog() const { return Eigen::Matrix<Scalar, 3, 1> {mat_(0), mat_(1), 1}; }

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

/// The 3D point inside the map.
struct SalientPointFragment
{
    std::optional<size_t> SyntheticVirtualPointId;
    std::optional<suriko::Point3> Coord; // TODO: when it is null?
};

/// The space with salient 3D points.
class FragmentMap
{
    std::vector<SalientPointFragment> salient_points_;
    size_t fragment_id_offset_;
    size_t next_salient_point_id_;
public:
    FragmentMap(size_t fragment_id_offset = 1000'000);

    SalientPointFragment& AddSalientPoint(const std::optional<suriko::Point3> &coord, size_t* salient_point_id = nullptr);

    void SetSalientPoint(size_t point_track_id, const suriko::Point3 &coord);
    void SetSalientPointNew(size_t fragment_id, const std::optional<suriko::Point3> &coord, std::optional<size_t> syntheticVirtualPointId);

    const SalientPointFragment& GetSalientPointNew(size_t salient_point_id) const;

    const suriko::Point3& GetSalientPoint(size_t salient_point_id) const;
          suriko::Point3& GetSalientPoint(size_t salient_point_id);

    bool GetSalientPointByVirtualPointIdInternal(size_t salient_point_id, const SalientPointFragment** fragment);

    size_t SalientPointsCount() const { return salient_points_.size(); }
    const std::vector<SalientPointFragment>& SalientPoints() const { return salient_points_; }
          std::vector<SalientPointFragment>& SalientPoints()       { return salient_points_; }
    void GetSalientPointsIds(std::vector<size_t>* salient_points_ids);

    void SetFragmentIdOffsetInternal(size_t fragment_id_offset);
private:
    size_t SalientPointIdToInd(size_t salient_point_id) const;
    size_t SalientPointIndToId(size_t salient_point_ind) const;
};

struct CornerData
{
    suriko::Point2 PixelCoord;
    Eigen::Matrix<Scalar, 3, 1> ImageCoord;
};

class CornerTrack
{
public:
    size_t TrackId;
private:
    ptrdiff_t StartFrameInd = -1;
    std::vector<std::optional<CornerData>> CoordPerFramePixels;
public:
    // Represents the reference to corresponding 3D salient point.
    // The value is null when the track of corners exist and the corresponding salient point is not reconstructed yet.
    std::optional<size_t> SalientPointId;

    // Represents the user generated id of a salient point in synthetic worlds.
    // The value may be used to match salient points. The non null value indicates that synthetic data is processed.
    std::optional<size_t> SyntheticVirtualPointId; // only available for artificially generated scenes where world's 3D points are known
public:
    CornerTrack() = default;

    bool HasCorners() const;
    size_t CornersCount() const;

    void AddCorner(size_t frame_ind, const suriko::Point2& value);
    CornerData& AddCorner(size_t frame_ind);

    std::optional<suriko::Point2> GetCorner(size_t frame_ind) const;
    std::optional<CornerData> GetCornerData(size_t frame_ind) const;

    void EachCorner(std::function<void(size_t, const std::optional<CornerData>&)> on_item) const;
private:
    void CheckConsistent();
};

class CornerTrackRepository
{
public:
    std::vector<suriko::CornerTrack> CornerTracks;

    suriko::CornerTrack& AddCornerTrackObj();

    const suriko::CornerTrack& GetPointTrackById(size_t point_track_id) const;
          suriko::CornerTrack& GetPointTrackById(size_t point_track_id);
    
    bool GetFirstPointTrackByFragmentSyntheticId(size_t salient_point_id, suriko::CornerTrack** corner_track);

    void PopulatePointTrackIds(std::vector<size_t> *result);

    void IteratePointsMarker() const {}
};

/// Constructs 3x3 skew symmetric matrix from 3-element vector.
void SkewSymmetricMat(const Eigen::Matrix<Scalar, 3, 1>& v, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> skew_mat);

/// Creates the rotation matrix around the vector @n by angle @ang in radians.
/// This uses the Rodrigues formula.
[[nodiscard]]
auto RotMatFromUnityDirAndAngle(const Eigen::Matrix<Scalar, 3, 1>& unity_dir, Scalar ang, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> rot_mat, bool check_input = true) -> bool;

[[nodiscard]]
auto RotMatFromAxisAngle(const Eigen::Matrix<Scalar, 3, 1>& axis_angle, gsl::not_null<Eigen::Matrix<Scalar, 3, 3>*> rot_mat) -> bool;

/// Checks if Rt*R=I and det(R)=1.
bool IsSpecialOrthogonal(const Eigen::Matrix<Scalar,3,3>& R, std::string* msg = nullptr);

/// Logarithm of SO(3) : R[3x3]->(n, ang) where n = rotation vector, ang = angle in radians.
[[nodiscard]]
auto LogSO3(const Eigen::Matrix<Scalar, 3, 3>& rot_mat, gsl::not_null<Eigen::Matrix<Scalar, 3, 1>*> unity_dir, gsl::not_null<Scalar*> ang, bool check_input = true) -> bool;

[[nodiscard]]
auto AxisAngleFromRotMat(const Eigen::Matrix<Scalar, 3, 3>& rot_mat, gsl::not_null<Eigen::Matrix<Scalar, 3, 1>*> dir) -> bool;

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
[[nodiscard]]
auto DecomposeProjMat(const Eigen::Matrix<Scalar, 3, 4> &proj_mat, bool check_post_cond = true)
-> std::tuple<bool, Scalar, Eigen::Matrix<Scalar, 3, 3>, SE3Transform>;

/// Finds the 3D coordinate of a world point from a list of corresponding 2D pixels in multiple images.
/// The orientation of the camera for each shot is specified in the list of projection matrices.
auto Triangulate3DPointByLeastSquares(const std::vector<suriko::Point2> &xs2D,
                                 const std::vector<Eigen::Matrix<Scalar,3,4>> &proj_mat_list, Scalar f0)
    -> suriko::Point3;

namespace internals
{
Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 3>* rot_mat, const Eigen::Matrix<Scalar, 3, 1>* translation);
Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 3>& rot_mat, const Eigen::Matrix<Scalar, 3, 1>& translation);
Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 3>& rot_mat);
Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 1>& translation);

/// Generates rotation matrix from direction and angle to rotate. For zero angle it returns the identity matrix.
Eigen::Matrix<Scalar, 3, 3> RotMat(const Eigen::Matrix<Scalar, 3, 1>& unity_dir, Scalar ang);
Eigen::Matrix<Scalar, 3, 3> RotMat(Scalar unity_dir_x, Scalar unity_dir_y, Scalar unity_dir_z, Scalar ang);
}
}