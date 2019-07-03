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
class Point2f {
    Eigen::Matrix<Scalar,2,1> mat_;
public:
    Point2f() = default;
    explicit Point2f(const Eigen::Matrix<Scalar, 2, 1> &m) : mat_(m) {}

    template <typename F0, typename F1>
    Point2f(const F0 &x, const F1 &y) {
        mat_(0) = static_cast<Scalar>(x);
        mat_(1) = static_cast<Scalar>(y);
    }

    const Eigen::Matrix<Scalar,2,1>& Mat() const { return mat_; };
          Eigen::Matrix<Scalar,2,1>& Mat()       { return mat_; };

    Eigen::Matrix<Scalar, 3, 1> AsHomog() const { return Eigen::Matrix<Scalar, 3, 1> {mat_(0), mat_(1), 1}; }

    Scalar X() const { return mat_[0]; }
    Scalar Y() const { return mat_[1]; }

    Scalar  operator[] (size_t i) const { return mat_(i); };
    Scalar& operator[] (size_t i)       { return mat_(i); };
};

struct Point2i
{
    int x, y;
    auto Mat() const { return Eigen::Matrix<int, 2, 1> { x, y}; }
};

//auto ToPoint(const Eigen::Matrix<Scalar,2,1>& m) -> suriko::Point2;

class Point3 {
    Eigen::Matrix<Scalar,3,1> mat_;
public:
    Point3() = default;
    Point3(const Eigen::Matrix<Scalar, 3, 1> &m) : mat_(m) { }

    template <typename F0, typename F1, typename F2>
    Point3(const F0 &x, const F1 &y, const F2 &z) {
        mat_(0) = static_cast<Scalar>(x);
        mat_(1) = static_cast<Scalar>(y);
        mat_(2) = static_cast<Scalar>(z);
    }

    const Eigen::Matrix<Scalar,3,1>& Mat() const { return mat_; };
          Eigen::Matrix<Scalar,3,1>& Mat()       { return mat_; };

    Scalar  operator[] (size_t i) const { return mat_(i); };
    Scalar& operator[] (size_t i)       { return mat_(i); };
    Scalar  X() const { return mat_[0]; }
    Scalar& X()       { return mat_[0]; }
    Scalar  Y() const { return mat_[1]; }
    Scalar& Y()       { return mat_[1]; }
    Scalar  Z() const { return mat_[2]; }
    Scalar& Z()       { return mat_[2]; }
};

struct Sizei
{
    int width, height;
};

template <typename F>
struct RectProto
{
    F x, y, width, height;
    int Right() const { return x + width; }
    int Bottom() const { return y + height; }
    auto TopLeft() const { return suriko::Point2i{ x, y }; }
    auto BotRight() const { return suriko::Point2i{ x + width, y + height }; }
};

template <typename F>
auto RectFromSides(F left, F top, F right, F bottom) -> RectProto<F>
{
    return RectProto<F> {left, top, right - left, bottom - top};
}

using Rect = RectProto<Scalar>;
using Recti = RectProto<int>;

bool operator == (const Recti& lhs, const Recti& rhs);
/// This is isued in GTest framework.
std::ostream& operator<<(std::ostream& os, const Recti& r);
std::optional<Recti> IntersectRects(const Recti& a, const Recti& b);
Recti DeflateRect(const Recti& a, int left, int top, int right, int bottom);
Recti TruncateRect(const Rect& a);
Recti EncompassRect(const Rect& a);
Recti ClampRectWhenFixedCenter(const Recti& r, suriko::Sizei min_size);

//auto ToPoint(const Eigen::Matrix<Scalar,3,1>& m) -> suriko::Point3;

struct SE2Transform
{
    Eigen::Matrix<Scalar, 2, 1> T;
    Eigen::Matrix<Scalar, 2, 2> R;

    SE2Transform() = default;
    SE2Transform(const Eigen::Matrix<Scalar, 2, 2>& R, const Eigen::Matrix<Scalar, 2, 1>& T) : R(R), T(T) {}

    static SE2Transform NoTransform() {
        return SE2Transform(
            Eigen::Matrix<Scalar, 2, 2>::Identity(),
            Eigen::Matrix<Scalar, 2, 1>::Zero());
    }
};

/// SE3=Special Euclidean transformation in 3D.
/// Direct camera movement transforms 3D points from camera frame into world frame.
/// Inverse camera movement transforms 3D points from world frame into camera frame.
struct SE3Transform
{
    Eigen::Matrix<Scalar, 3, 1> T;
    Eigen::Matrix<Scalar, 3, 3> R;

    SE3Transform() = default;
    SE3Transform(const Eigen::Matrix<Scalar, 3, 3>& R, const Eigen::Matrix<Scalar, 3, 1>& T) : R(R), T(T) {}
    
    static SE3Transform NoTransform() { 
        return SE3Transform(
            Eigen::Matrix<Scalar, 3, 3>::Identity(),
            Eigen::Matrix<Scalar, 3, 1>::Zero());
    }
};

auto SE3Inv(const SE3Transform& rt) -> SE3Transform;
auto SE2Apply(const SE2Transform& rt, const suriko::Point2f& x)->suriko::Point2f;
auto SE3Apply(const SE3Transform& rt, const suriko::Point3& x) -> suriko::Point3;
auto SE3Compose(const SE3Transform& rt1, const SE3Transform& rt2) -> suriko::SE3Transform;
auto SE3AFromB(const SE3Transform& a_from_world, const SE3Transform& b_from_world) -> suriko::SE3Transform;

/// The 3D point inside the map.
struct SalientPointFragment
{
    std::optional<size_t> synthetic_virtual_point_id;
    std::optional<suriko::Point3> coord; // TODO: when it is null?
    void* user_obj = nullptr; // ptr to salient object in tracker object
};

/// The space with salient 3D points.
class FragmentMap
{
public:
    // TODO: how to organize identity of a salient point
    static void DependsOnSalientPointIdInfrustructure() {}
private:
    std::vector<SalientPointFragment> salient_points_;
    size_t fragment_id_offset_;
    size_t next_salient_point_id_;
public:
    FragmentMap(size_t fragment_id_offset = 1000'000);

    SalientPointFragment& AddSalientPointTempl(const std::optional<suriko::Point3> &coord, size_t* salient_point_id = nullptr);

    void SetSalientPoint(size_t point_track_id, const suriko::Point3 &coord);
    void SetSalientPointNew(size_t fragment_id, const std::optional<suriko::Point3> &coord, std::optional<size_t> syntheticVirtualPointId);

    const SalientPointFragment& GetSalientPointNew(size_t salient_point_id) const;
          SalientPointFragment& GetSalientPointNew(size_t salient_point_id);

    const SalientPointFragment& GetSalientPointByInternalOrder(size_t sal_pnt_array_ind) const;

    const suriko::Point3& GetSalientPoint(size_t salient_point_id) const;
          suriko::Point3& GetSalientPoint(size_t salient_point_id);

    bool GetSalientPointByVirtualPointIdInternal(size_t salient_point_id, const SalientPointFragment** fragment);

    size_t SalientPointsCount() const { return salient_points_.size(); }
    const std::vector<SalientPointFragment>& SalientPoints() const { return salient_points_; }
          std::vector<SalientPointFragment>& SalientPoints()       { return salient_points_; }
    void GetSalientPointsIds(std::vector<size_t>* salient_points_ids) const;

    void SetFragmentIdOffsetInternal(size_t fragment_id_offset);
private:
    size_t SalientPointIdToInd(size_t salient_point_id) const;
    size_t SalientPointIndToId(size_t salient_point_ind) const;
};

struct CornerData
{
    suriko::Point2f pixel_coord;
    Eigen::Matrix<Scalar, 3, 1> image_coord;
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

    std::optional<suriko::Point3> DebugSalientPointCoord; // for debugging, saves world position of corresponding salient point
public:
    CornerTrack() = default;

    bool HasCorners() const;
    size_t CornersCount() const;

    void AddCorner(size_t frame_ind, const suriko::Point2f& value);
    CornerData& AddCorner(size_t frame_ind);

    std::optional<suriko::Point2f> GetCorner(size_t frame_ind) const;
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

    size_t CornerTracksCount() const;
    size_t ReconstructedCornerTracksCount() const;
    size_t FramesCount() const;

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

/// Checks if Rt*R=I.
[[nodiscard]]
bool IsOrthogonal(const Eigen::Matrix<Scalar,3,3>& R, std::string* msg = nullptr);
bool IsOrthogonal(const Eigen::Matrix<Scalar,2,2>& R, std::string* msg = nullptr);

/// Checks if Rt*R=I and det(R)=1.
[[nodiscard]]
bool IsSpecialOrthogonal(const Eigen::Matrix<Scalar,2,2>& R, std::string* msg = nullptr);
bool IsSpecialOrthogonal(const Eigen::Matrix<Scalar,3,3>& R, std::string* msg = nullptr);

/// Checks if M=Identity.
[[nodiscard]]
bool IsIdentity(const Eigen::Matrix<Scalar, 3, 3>& M, Scalar rtol, Scalar atol, std::string* msg = nullptr);

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
auto Triangulate3DPointByLeastSquares(const std::vector<suriko::Point2f> &xs2D,
                                 const std::vector<Eigen::Matrix<Scalar,3,4>> &proj_mat_list, Scalar f0)
    -> suriko::Point3;

/// Constructs SE3 transformation to look from <b>eye</b> point onto the target <b>center</b> point,
/// which lies on the surface with normal <b>up</b>.
/// Result camera's orientation is in XYZ=LUF (left-up-right) orientation, like in Hartley&Zisserman book.
/// Result transformation converts camera points into world (thus WFC=world from camera).
[[nodiscard]]
SE3Transform LookAtLufWfc(
    const Eigen::Matrix<Scalar, 3, 1>& eye,
    const Eigen::Matrix<Scalar, 3, 1>& center,
    const Eigen::Matrix<Scalar, 3, 1>& up);

/// (x-c)*A*(x-c)=r where x[3x1], c[3x1], A[3,3], r=scalar
struct Ellipsoid3DWithCenter
{
    Eigen::Matrix<Scalar, 3, 3> A;
    Eigen::Matrix<Scalar, 3, 1> center; // c
    Scalar right_side; // r
};

bool ValidateEllipsoid(const Ellipsoid3DWithCenter& maybe_ellipsoid);

/// Equation of ellipse is: (x-c)*A*(x-c)=r where c[2x1] is the center of ellipse, A[2,2] is a matrix, r is a scalar.
/// x[2x1] belongs to ellipse if it satisfies the equation.
struct Ellipse2DWithCenter
{
    Eigen::Matrix<Scalar, 2, 2> A;
    Eigen::Matrix<Scalar, 2, 1> center; // c
    Scalar right_side; // r
};

bool ValidateEllipse(const Ellipse2DWithCenter& maybe_ellipsoid);
Rect GetEllipseBounds(const Ellipse2DWithCenter& ellipse);

/// Represents a 2D ellipse, for which the eigenvectors are found.
struct RotatedEllipse2D
{
    Eigen::Matrix<Scalar, 2, 1> semi_axes;
    SE2Transform world_from_ellipse;
};

Rect GetEllipseBounds2(const RotatedEllipse2D& rotated_ellipse);

/// Represents a 2D ellipse, for which the eigenvectors are found.
struct RotatedEllipsoid3D
{
    static constexpr Scalar kRightSide = 1; // right side in x^2/a^2+y^2/b^2+z^2/c^2==1
    Eigen::Matrix<Scalar, 3, 1> semi_axes;
    SE3Transform world_from_ellipse;
};

[[nodiscard]]
std::tuple<bool,RotatedEllipsoid3D> GetRotatedUncertaintyEllipsoidFromCovMat(const Eigen::Matrix<Scalar, 3, 3>& cov, const Eigen::Matrix<Scalar, 3, 1>& mean,
    Scalar covar3D_to_ellipsoid_chi_square);

[[nodiscard]]
bool Get2DRotatedEllipseFromCovMat(const Eigen::Matrix<Scalar, 2, 2>& cov,
    Scalar covar2D_to_ellipse_confidence,
    Eigen::Matrix<Scalar, 2, 1>* semi_axes,
    Eigen::Matrix<Scalar, 2, 2>* world_from_ellipse);

[[nodiscard]]
std::tuple<bool,RotatedEllipse2D> Get2DRotatedEllipseFromCovMat(
    const Eigen::Matrix<Scalar, 2, 2>& covar,
    const Eigen::Matrix<Scalar, 2, 1>& mean,
    Scalar covar2D_to_ellipse_confidence);

bool GetRotatedEllipsoid(const Ellipsoid3DWithCenter& ellipsoid, bool can_throw, RotatedEllipsoid3D* result);
RotatedEllipsoid3D GetRotatedEllipsoid(const Ellipsoid3DWithCenter& ellipsoid);

RotatedEllipse2D GetRotatedEllipse2D(const Ellipse2DWithCenter& ellipsoid);

namespace internals
{
Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 3>* rot_mat, const Eigen::Matrix<Scalar, 3, 1>* translation);
Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 3>& rot_mat, const Eigen::Matrix<Scalar, 3, 1>& translation);
Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 3>& rot_mat);
Eigen::Matrix<Scalar, 4, 4> SE3Mat(const Eigen::Matrix<Scalar, 3, 1>& translation);

/// Generates rotation matrix from direction and angle to rotate. For zero angle it returns the identity matrix.
Eigen::Matrix<Scalar, 3, 3> RotMat(const Eigen::Matrix<Scalar, 3, 1>& unity_dir, Scalar ang);
Eigen::Matrix<Scalar, 3, 3> RotMat(Scalar unity_dir_x, Scalar unity_dir_y, Scalar unity_dir_z, Scalar ang);

template <typename F>
constexpr auto Deg2Rad(F x)
{
    using Float = std::common_type_t<float, F>; // if input is int, force output to be float
    return x * static_cast<Float>(M_PI / 180);
}
}
}