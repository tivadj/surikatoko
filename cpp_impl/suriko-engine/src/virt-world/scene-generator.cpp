#include <glog/logging.h>
#include "suriko/approx-alg.h"
#include "suriko/virt-world/scene-generator.h"

namespace suriko { namespace virt_world {

using namespace suriko::internals;

void GenerateCircleCameraShots(const suriko::Point3& circle_center, Scalar circle_radius, Scalar ascentZ, gsl::span<const Scalar> rot_angles,
    std::vector<SE3Transform>* inverse_orient_cams)
{
    for (gsl::span<const Scalar>::index_type ang_ind = 0; ang_ind < rot_angles.size(); ++ang_ind)
    {
        Scalar ang = rot_angles[ang_ind];

        // X is directed to the right, Y - to up
        Eigen::Matrix<Scalar, 4, 4> cam_from_world = Eigen::Matrix<Scalar, 4, 4>::Identity();

        // translate to circle center
        Point3 shift = circle_center;

        // translate to camera position at the circumference of the circle
        Point3 center_to_cam_pos(
            circle_radius * std::cos(ang),
            circle_radius * std::sin(ang),
            ascentZ
        );
        shift += center_to_cam_pos;

        // minus due to inverse camera orientation (conversion from world to camera)
        cam_from_world = SE3Mat(std::nullopt ,-shift) * cam_from_world;

        // rotate OY around OZ so that OY points towards center in horizontal plane OZ=ascentZ
        Point3 cam_pos_to_center_circle = Point3{ -shift[0], -shift[1], 0 }; // the direction towards center O
        CHECK(Normalize(&cam_pos_to_center_circle));
        Point3 oy{ 0, 1, 0 };
        Scalar ang_yawOY = std::acos(Dot(oy,cam_pos_to_center_circle)); // rotate OY 'towards' (parallel to XOY plane) center

        // correct sign so that OY is rotated towards center by shortest angle
        Point3 oz{ 0, 0, 1 };
        int ang_yawOY_sign = Sign(Dot(Cross(oy,cam_pos_to_center_circle),oz));
        ang_yawOY *= ang_yawOY_sign;

        cam_from_world = SE3Mat(RotMat(oz, -ang_yawOY), std::nullopt) * cam_from_world;

        // look down towards the center
        Scalar look_down_ang = std::atan2(center_to_cam_pos[2], Eigen::Matrix<Scalar, 3, 1>(center_to_cam_pos[0], center_to_cam_pos[1], 0).norm());

        // +pi/2 to direct not y-forward and z-up but z-forward and y-bottom
        cam_from_world = SE3Mat(RotMat(1, 0, 0, look_down_ang + Pi<Scalar>() / 2), std::nullopt) * cam_from_world;
        SE3Transform RT(cam_from_world.topLeftCorner(3, 3), ToPoint3(cam_from_world.topRightCorner(3, 1)));

        // now camera is directed x-right, y-bottom, z-forward
        inverse_orient_cams->push_back(RT);
    }
}

void GenerateCameraShotsRightAndLeft(const WorldBounds& wb,
    const Point3& eye_offset,
    const Point3& center_offset,
    const Point3& up,
    Scalar max_deviation,
    int num_steps,
    std::vector<std::optional<SE3Transform>>* cam_orient_cfw)
{
    suriko::Point3 look_at_me{ wb.x_min, wb.y_min, 0 };

    for (int i = 0; i < num_steps; ++i)
    {
        suriko::Point3 eye_offset_extra{};
        switch (i % 4)
        {
        case 0:
            break;
        case 1:
            eye_offset_extra[0] += max_deviation;
            break;
        case 2:
            break;
        case 3:
            eye_offset_extra[0] -= max_deviation;
            break;
        default:
            SRK_ASSERT(false);
        }

        Point3 eye = look_at_me + eye_offset + eye_offset_extra;

        Point3 center = look_at_me + center_offset;

        auto wfc = LookAtLufWfc(eye, center, up);

        SE3Transform RT = SE3Inv(wfc);

        cam_orient_cfw->push_back(RT);
    }
}

void GenerateCameraShotsOscilateRightAndLeft(const WorldBounds& wb,
    const Point3& eye,
    const Point3& center,
    const Point3& up,
    Scalar max_deviation,
    int periods_count,
    int shots_per_period,
    bool head_straight,
    std::vector<std::optional<SE3Transform>>* cam_orient_cfw)
{
    Point3 view_dir = center - eye;
    CHECK(Normalize(&view_dir));

    Point3 right_dir = Cross(view_dir, up);
    CHECK(Normalize(&right_dir));

    int max_shots = periods_count * shots_per_period;
    for (int shot_ind = 0; shot_ind < max_shots; ++shot_ind)
    {
        auto w = 2 * Pi<Scalar>() / shots_per_period * shot_ind;

        auto right_deviation = std::sin(w) * max_deviation;

        Point3 shifted_eye = eye + right_dir * right_deviation;

        Point3 cur_center;
        if (head_straight)
            cur_center = shifted_eye + view_dir;  // the direction of view is constant
        else
            cur_center = center;            // the center point we are looking at is constant

        auto wfc = LookAtLufWfc(shifted_eye, cur_center, up);

        SE3Transform RT = SE3Inv(wfc);

        cam_orient_cfw->push_back(RT);
    }
}

void GenerateCameraShotsRotateLeftAndRight(const WorldBounds& wb,
    const Point3& eye,
    const Point3& up,
    Scalar min_ang, Scalar max_ang,
    int periods_count,
    int shots_per_period,
    std::vector<std::optional<SE3Transform>>* cam_orient_cfw)
{
    Scalar init_ang = (min_ang + max_ang) / 2;
    Scalar half_fov = (max_ang - min_ang) / 2;

    int max_shots = periods_count * shots_per_period;
    for (int shot_ind = 0; shot_ind < max_shots; ++shot_ind)
    {
        auto w = 2 * Pi<Scalar>() / shots_per_period * shot_ind;

        auto cur_ang = init_ang + std::sin(w) * half_fov;

        Point3 view_dir;
        view_dir[0] = std::cos(cur_ang);
        view_dir[1] = std::sin(cur_ang);
        view_dir[2] = 0;

        auto wfc = LookAtLufWfc(eye, eye + view_dir, up);

        SE3Transform RT = SE3Inv(wfc);

        cam_orient_cfw->push_back(RT);
    }
}

void GenerateCameraShots3DPath(const WorldBounds& wb,
    const std::vector<LookAtComponents>& cam_poses, int periods_count,
    std::vector<std::optional<SE3Transform>>* cam_orient_cfw)
{
    for (int i = 0; i < periods_count; ++i)
    {
        for (const auto& p : cam_poses)
        {
            SE3Transform wfc = LookAtLufWfc(p.eye, p.center, p.up);
            SE3Transform cfw = SE3Inv(wfc);;
            cam_orient_cfw->push_back(cfw);
        }
    }
}

}}