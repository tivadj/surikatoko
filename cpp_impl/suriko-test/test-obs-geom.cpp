#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "suriko/obs-geom.h"
#include "suriko/rt-config.h"
#include <corecrt_math_defines.h>

namespace suriko_test
{
using namespace suriko;

class ObsGeomTest : public testing::Test
{
public:
	Scalar atol = 1e-5;
};

TEST_F(ObsGeomTest, SkewSymmetricMatConstruction)
{
	Eigen::Matrix<Scalar, 3, 1> v(1,2,3);
	Eigen::Matrix<Scalar, 3, 3> skew_mat;
	SkewSymmetricMat(v, &skew_mat);

    EXPECT_FLOAT_EQ(Scalar{ 0 }, skew_mat(0, 0));
    EXPECT_FLOAT_EQ(Scalar{ -3 }, skew_mat(0, 1));
    EXPECT_FLOAT_EQ(Scalar{ 3 }, skew_mat(1, 0));
}

TEST_F(ObsGeomTest, RotMatFromAxisAngle)
{
	Eigen::Matrix<Scalar, 3, 1> dir(1,1,1);
	dir *= 2*M_PI/3 / dir.norm();
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
	Eigen::Matrix<Scalar, 3, 1> dir(1,1,1);
	dir *= M_PI / 4 / dir.norm(); // len=pi/4

	Eigen::Matrix<Scalar, 3, 3> rot_mat;
	bool op = RotMatFromAxisAngle(dir, &rot_mat);
	ASSERT_TRUE(op);

	Eigen::Matrix<Scalar, 3, 1> dir_back;
	op = AxisAngleFromRotMat(rot_mat, &dir_back);
	ASSERT_TRUE(op);

	EXPECT_NEAR(dir[0], dir_back[0], atol);
	EXPECT_NEAR(dir[1], dir_back[1], atol);
	EXPECT_NEAR(dir[2], dir_back[2], atol);
}

TEST_F(ObsGeomTest, AxisAngleCornerCases)
{
	Eigen::Matrix<Scalar, 3, 1> dir(0,0,0);
	Eigen::Matrix<Scalar, 3, 3> rot_mat;
	bool op = RotMatFromAxisAngle(dir, &rot_mat);
	EXPECT_FALSE(op);
}

TEST_F(ObsGeomTest, UnityDirAndAngleCornerCases)
{
	Eigen::Matrix<Scalar, 3, 1> unity_dir(0,0,0);
	Eigen::Matrix<Scalar, 3, 3> rot_mat;
	bool op = RotMatFromUnityDirAndAngle(unity_dir, 100, &rot_mat);
	EXPECT_FALSE(op) <<"dir.length != 0 is unchecked";

	unity_dir << 1, 1, 1;
	op = RotMatFromUnityDirAndAngle(unity_dir, 0, &rot_mat);
	EXPECT_FALSE(op) << "ang != 0 is unchecked";
}


}
