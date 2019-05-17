#include <vector>
#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "suriko/obs-geom.h"
#include "suriko/rt-config.h"
#include "suriko/bundle-adj-kanatani.h"
#include "suriko/virt-world/scene-generator.h"

namespace suriko_test
{
using namespace suriko;
using namespace suriko::virt_world;

class BAKanataniTest : public testing::Test
{
public:
    Scalar atol_ = static_cast<Scalar>(1e-2);
};

TEST_F(BAKanataniTest, NormalizationSimple)
{
    // [id, coord] list of salient points
    std::vector<std::pair<size_t, suriko::Point3>> pnts3D = {
        {0, suriko::Point3(-1, 0, 0)}, // A
        {0, suriko::Point3(-0.5, 0.866, 0) }, // B
        {0, suriko::Point3(0, 1, 0) }, // C
        {0, suriko::Point3(1, 0, 0) }, // D
        {0, suriko::Point3(0, -1, 0) }, // K
    };  
    FragmentMap map;
    for (size_t i=0; i<pnts3D.size(); ++i)
    {
        size_t salient_point_id = -1;
        map.AddSalientPointTempl(pnts3D[i].second, &salient_point_id);
        pnts3D[i].first = salient_point_id;
    }

    suriko::Point3 circle_center(0, 0, 0);
    Scalar circle_radius = 1;
    Scalar ascentZ = 0; // work in XOY plane
    std::vector<Scalar> rot_angles = { static_cast<Scalar>(3*M_PI/2 + M_PI/6), static_cast<Scalar>(3*M_PI/2) };
    std::vector<SE3Transform> inverse_orient_cams;
    GenerateCircleCameraShots(circle_center, circle_radius, ascentZ, rot_angles, &inverse_orient_cams);

    std::vector<SE3Transform> inverse_orient_cams_before_norm = inverse_orient_cams; // copy

    Scalar t1_dist = 1;
    size_t unity_comp_ind = 0;
    bool success = false;
    SceneNormalizer sn = NormalizeSceneInplace(&map, &inverse_orient_cams, t1_dist, unity_comp_ind, &success);
    ASSERT_TRUE(success) << "normalization failed";

    // check cam0
    const SE3Transform& cam0_CifromC0 = inverse_orient_cams[0];
    EXPECT_NEAR(0, cam0_CifromC0.T.norm(), atol_) << "cam0 is the new center of the world";
    
    std::string msg;
    bool op = IsIdentity(cam0_CifromC0.R, 0, atol_);
    ASSERT_TRUE(op) << "failed R=Identity" << msg;

    // check cam1
    SE3Transform cam1_from_cam2 = SE3Inv(inverse_orient_cams[1]);
    EXPECT_NEAR(std::abs(cam1_from_cam2.T[unity_comp_ind]), t1_dist, 0.01) << "Scaling of world failed: expect t1y=" << t1_dist << " actual T1=" << cam1_from_cam2.T;

    // map is inflated *2 because world scale factor was 2
    Scalar s = sn.WorldScale();

    // check points in cam0 (new world center)
    std::vector<suriko::Point3> pnts3D_cam0 = {
        suriko::Point3(-0.866*s, 0, 1.5*s), // A
        suriko::Point3(0, 0, 2*s), // B
        suriko::Point3(0.5*s, 0, 1.866*s), // C
        suriko::Point3(0.866*s, 0, 0.5*s), // D
        suriko::Point3(-0.5*s, 0, 0.133975*s), // K
    };

    for (size_t i=0; i<pnts3D.size(); ++i)
    {
        Point3 pnt_expect = pnts3D_cam0[i];
        size_t salient_point_id = pnts3D[i].first;
        Point3 pnt3D_cam0 = map.GetSalientPoint(salient_point_id);
        EXPECT_TRUE((pnt_expect.Mat() - pnt3D_cam0.Mat()).norm() < atol_) << "cam0 3D point mismatch i=" << i << " P1=" << pnt_expect.Mat() << " P2=" << pnt3D_cam0.Mat();
    }

    // check points in cam1
    std::vector<suriko::Point3> pnts3D_cam1 = {
        suriko::Point3(-1*s, 0, 1*s), // A
        suriko::Point3(-0.5*s, 0, 1.866*s), // B
        suriko::Point3(0, 0, 2*s), // C
        suriko::Point3(1*s, 0, 1*s), // D
        suriko::Point3(0, 0, 0), // K
    };

    SE3Transform cam1_from_cam0 = inverse_orient_cams[1]; // cami from cam0
    for (size_t i=0; i<pnts3D.size(); ++i)
    {
        Point3 pnt_expect = pnts3D_cam1[i];
        size_t salient_point_id = pnts3D[i].first;
        Point3 p3D_cam0 = map.GetSalientPoint(salient_point_id);
        
        Point3 pnt_actual = SE3Apply(cam1_from_cam0, p3D_cam0);
        EXPECT_TRUE((pnt_expect.Mat() - pnt_actual.Mat()).norm() < atol_) << "cam1 3D point mismatch i=" << i << " P1=" << pnt_expect.Mat() << " P2=" << pnt_actual.Mat();
    }

    sn.RevertNormalization();

    // check camera orientations are restored
    for (size_t i=0; i<inverse_orient_cams.size(); ++i)
    {
        SE3Transform expect = inverse_orient_cams_before_norm[i];
        SE3Transform actual = inverse_orient_cams[i];
        Scalar d1 = (expect.T - actual.T).norm();
        Scalar d2 = (expect.R - actual.R).norm();
        EXPECT_TRUE(d1 < atol_);
        EXPECT_TRUE(d2 < atol_);
    }

    // check salient points are restored
    for (size_t i = 0; i<pnts3D.size(); ++i)
    {
        Point3 pnt_expect = pnts3D[i].second;
        size_t salient_point_id = pnts3D[i].first;
        Point3 pnt_actual = map.GetSalientPoint(salient_point_id);
        EXPECT_TRUE((pnt_expect.Mat() - pnt_actual.Mat()).norm() < atol_) << "cam0 3D point mismatch i=" << i << " P1=" << pnt_expect.Mat() << " P2=" << pnt_actual.Mat();
    }
}

}
