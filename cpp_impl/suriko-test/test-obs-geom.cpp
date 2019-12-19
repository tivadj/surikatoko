#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "suriko/obs-geom.h"
#include "suriko/rt-config.h"

namespace suriko_test
{
using namespace suriko;

class ObsGeomTest : public testing::Test
{
public:
	Scalar atol = (Scalar)1e-5;
};

TEST_F(ObsGeomTest, SkewSymmetricMatConstruction)
{
    Point3 v{ 1,2,3 };
	Eigen::Matrix<Scalar, 3, 3> skew_mat;
	SkewSymmetricMat(v, &skew_mat);

    EXPECT_DOUBLE_EQ( 0, skew_mat(0, 0));
    EXPECT_DOUBLE_EQ(-3, skew_mat(0, 1));
    EXPECT_DOUBLE_EQ( 3, skew_mat(1, 0));
}

TEST_F(ObsGeomTest, RotMatFromAxisAngle)
{
    Point3 dir(1,1,1);
	dir *= static_cast<Scalar>(2*M_PI/3) / Norm(dir);
	Eigen::Matrix<Scalar, 3, 3> R120;
	bool op = RotMatFromAxisAngle(dir, &R120);
	ASSERT_TRUE(op);

	// point V(len,0,0) should be rotated into (0,len,0) by R120

	Eigen::Matrix<Scalar, 3, 1> v(10,0,0);
	Eigen::Matrix<Scalar, 3, 1> v_new = R120 * v;
	EXPECT_NEAR(0, v_new[0], atol);
	EXPECT_NEAR(10, v_new[1], atol);
	EXPECT_NEAR(0, v_new[2], atol);
}

TEST_F(ObsGeomTest, AxisAngle_To_RotMat_And_Back)
{
    Point3 dir{ 1,1,1 };
	dir *= static_cast<Scalar>(M_PI / 4) / Norm(dir); // len=pi/4

	Eigen::Matrix<Scalar, 3, 3> rot_mat;
	bool op = RotMatFromAxisAngle(dir, &rot_mat);
	ASSERT_TRUE(op);

    Point3 dir_back;
	op = AxisAngleFromRotMat(rot_mat, &dir_back);
	ASSERT_TRUE(op);

	EXPECT_NEAR(dir[0], dir_back[0], atol);
	EXPECT_NEAR(dir[1], dir_back[1], atol);
	EXPECT_NEAR(dir[2], dir_back[2], atol);
}

TEST_F(ObsGeomTest, AxisAngleCornerCases)
{
    Point3 dir(0,0,0);
	Eigen::Matrix<Scalar, 3, 3> rot_mat;
	bool op = RotMatFromAxisAngle(dir, &rot_mat);
	EXPECT_FALSE(op);
}

TEST_F(ObsGeomTest, UnityDirAndAngleCornerCases)
{
    Point3 unity_dir(0,0,0);
	Eigen::Matrix<Scalar, 3, 3> rot_mat;
	bool op = RotMatFromUnityDirAndAngle(unity_dir, 100, &rot_mat);
	EXPECT_FALSE(op) <<"dir.length != 0 is unchecked";

    unity_dir = Point3{ 1, 1, 1 };
	op = RotMatFromUnityDirAndAngle(unity_dir, 0, &rot_mat);
	EXPECT_FALSE(op) << "ang != 0 is unchecked";
}

TEST_F(ObsGeomTest, AlignTrajectoriesHorn_Simple1)
{
    std::vector<Point3> src_traj;
    for (size_t i=0; i<100; ++i)
    {
        auto x = static_cast<Scalar>(1 + i);
        auto y = std::log(x);
        src_traj.push_back(Point3{ x, y, 0 });
    }

    {
        // rotate counter clockwise by 90deg
        Eigen::Matrix<Scalar, 3, 3> rot_mat;
        bool opaxis = RotMatFromUnityDirAndAngle(Point3{ 0,0,1 }, M_PI / 2, &rot_mat, true);
        CHECK(opaxis);
        SE3Transform gt_rt{ rot_mat, Point3{1,1,0} };  // ground truth

        std::vector<Point3> dst_traj(src_traj.size());
        std::transform(src_traj.begin(), src_traj.end(), dst_traj.begin(),
            [&gt_rt](const auto& p) { return SE3Apply(gt_rt, p); });

        // should recover transformation in question
        auto [op, dst_from_src] = AlignTrajectoryHornAFromB(dst_traj, src_traj);
        ASSERT_TRUE(op);
        auto r_diffvalue = (dst_from_src.R - gt_rt.R).norm();
        EXPECT_NEAR(0, r_diffvalue, atol);
        auto t_diffvalue = (dst_from_src.T - gt_rt.T).norm();
        EXPECT_NEAR(0, t_diffvalue, atol);

        // source points are mapped closely to destination points
        auto p0_src = src_traj[0];
        auto p0_dst = dst_traj[0];
        auto p0_src_to_dst = SE3Apply(dst_from_src, p0_src);
        auto pnt_diffvalue = Norm(Mat(p0_dst) - Mat(p0_src_to_dst));
        EXPECT_NEAR(0, pnt_diffvalue, atol);
    }

    {
        // should recover identity transformation (no transformation)
        auto [op, estim_rt] = AlignTrajectoryHornAFromB(src_traj, src_traj);
        ASSERT_TRUE(op);
        using Mat33 = decltype(estim_rt.R);
        auto r_diffvalue = (estim_rt.R - Mat33::Identity()).norm();
        EXPECT_NEAR(0, r_diffvalue, atol);
        auto t_norm = estim_rt.T.norm();
        EXPECT_NEAR(0, t_norm, atol);
    }
}

}
