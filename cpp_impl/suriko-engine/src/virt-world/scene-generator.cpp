#include "suriko/approx-alg.h"
#include "suriko/virt-world/scene-generator.h"

namespace suriko { namespace virt_world {

using namespace suriko::internals;

void GenerateCircleCameraShots(const suriko::Point3& circle_center, Scalar circle_radius, Scalar ascentZ, gsl::span<const Scalar> rot_angles, std::vector<SE3Transform>* inverse_orient_cams)
{
    for (gsl::span<const Scalar>::index_type ang_ind = 0; ang_ind < rot_angles.size(); ++ang_ind)
    {
        Scalar ang = rot_angles[ang_ind];

        // X is directed to the right, Y - to up
        Eigen::Matrix<Scalar, 4, 4> cam_from_world = Eigen::Matrix<Scalar, 4, 4>::Identity();

        // translate to circle center
        Eigen::Matrix<Scalar, 3, 1> shift = circle_center.Mat();

        // translate to camera position at the circumference of the circle
        Eigen::Matrix<Scalar, 3, 1> center_to_cam_pos(
            circle_radius * std::cos(ang),
            circle_radius * std::sin(ang),
            ascentZ
        );
        shift += center_to_cam_pos;

        // minus due to inverse camera orientation (conversion from world to camera)
        cam_from_world = SE3Mat(Eigen::Matrix<Scalar, 3, 1>(-shift)) * cam_from_world;

        // rotate OY around OZ so that OY points towards center in horizontal plane OZ=ascentZ
        Eigen::Matrix<Scalar, 3, 1> cam_pos_to_center_circle = -Eigen::Matrix<Scalar, 3, 1>(shift[0], shift[1], 0); // the direction towards center O
        cam_pos_to_center_circle.normalize();
        Eigen::Matrix<Scalar, 3, 1> oy(0, 1, 0);
        Scalar ang_yawOY = std::acos(oy.dot(cam_pos_to_center_circle)); // rotate OY 'towards' (parallel to XOY plane) center

        // correct sign so that OY is rotated towards center by shortest angle
        Eigen::Matrix<Scalar, 3, 1> oz(0, 0, 1);
        int ang_yawOY_sign = Sign(oy.cross(cam_pos_to_center_circle).dot(oz));
        ang_yawOY *= ang_yawOY_sign;

        cam_from_world = SE3Mat(RotMat(oz, -ang_yawOY)) * cam_from_world;

        // look down towards the center
        Scalar look_down_ang = std::atan2(center_to_cam_pos[2], Eigen::Matrix<Scalar, 3, 1>(center_to_cam_pos[0], center_to_cam_pos[1], 0).norm());

        // +pi/2 to direct not y-forward and z-up but z-forward and y-bottom
        cam_from_world = SE3Mat(RotMat(1, 0, 0, look_down_ang + Pi<Scalar>() / 2)) * cam_from_world;
        SE3Transform RT(cam_from_world.topLeftCorner(3, 3), cam_from_world.topRightCorner(3, 1));

        // now camera is directed x-right, y-bottom, z-forward
        inverse_orient_cams->push_back(RT);
    }
}

void GenerateCameraShotsRightAndLeft(const WorldBounds& wb,
    suriko::Point3 eye_offset,
    suriko::Point3 center_offset,
    const Eigen::Matrix<Scalar, 3, 1>& up,
    Scalar max_deviation,
    int num_steps,
    std::vector<SE3Transform>* inverse_orient_cams)
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

        Eigen::Matrix<Scalar, 3, 1> eye = look_at_me.Mat() + eye_offset.Mat() + eye_offset_extra.Mat();

        Eigen::Matrix<Scalar, 3, 1> center = look_at_me.Mat() + center_offset.Mat();

        auto wfc = LookAtLufWfc(eye, center, up);

        SE3Transform RT = SE3Inv(wfc);

        inverse_orient_cams->push_back(RT);
    }
}

void GenerateCameraShotsOscilateRightAndLeft(const WorldBounds& wb,
    suriko::Point3 eye,
    suriko::Point3 center,
    const Eigen::Matrix<Scalar, 3, 1>& up,
    Scalar max_deviation,
    int periods_count,
    int shots_per_period,
    bool head_straight,
    std::vector<SE3Transform>* inverse_orient_cams)
{
    Eigen::Matrix<Scalar, 3, 1> view_dir = center.Mat() - eye.Mat();
    view_dir.normalize();

    Eigen::Matrix<Scalar, 3, 1> right_dir = view_dir.cross(up);
    right_dir.normalize();

    int max_shots = periods_count * shots_per_period;
    for (int shot_ind = 0; shot_ind < max_shots; ++shot_ind)
    {
        auto w = 2 * Pi<Scalar>() / shots_per_period * shot_ind;

        auto right_deviation = std::sin(w) * max_deviation;

        Eigen::Matrix<Scalar, 3, 1> shifted_eye = eye.Mat() + right_dir * right_deviation;

        Eigen::Matrix<Scalar, 3, 1> cur_center;
        if (head_straight)
            cur_center = shifted_eye + view_dir;  // the direction of view is constant
        else
            cur_center = center.Mat();            // the center point we are looking at is constant

        auto wfc = LookAtLufWfc(shifted_eye, cur_center, up);

        SE3Transform RT = SE3Inv(wfc);

        inverse_orient_cams->push_back(RT);
    }
}

void GenerateCameraShotsRotateLeftAndRight(const WorldBounds& wb,
    suriko::Point3 eye,
    const Eigen::Matrix<Scalar, 3, 1>& up,
    Scalar min_ang, Scalar max_ang,
    int periods_count,
    int shots_per_period,
    std::vector<SE3Transform>* inverse_orient_cams)
{
    Scalar init_ang = (min_ang + max_ang) / 2;
    Scalar half_fov = (max_ang - min_ang) / 2;

    int max_shots = periods_count * shots_per_period;
    for (int shot_ind = 0; shot_ind < max_shots; ++shot_ind)
    {
        auto w = 2 * Pi<Scalar>() / shots_per_period * shot_ind;

        auto cur_ang = init_ang + std::sin(w) * half_fov;

        Eigen::Matrix<Scalar, 3, 1> view_dir;
        view_dir[0] = std::cos(cur_ang);
        view_dir[1] = std::sin(cur_ang);
        view_dir[2] = 0;

        auto wfc = LookAtLufWfc(eye.Mat(), eye.Mat() + view_dir, up);

        SE3Transform RT = SE3Inv(wfc);

        inverse_orient_cams->push_back(RT);
    }
}

void GenerateCameraShots3DPath(const WorldBounds& wb,
    const std::vector<LookAtComponents>& cam_poses, int periods_count,
    std::vector<SE3Transform>* inverse_orient_cams)
{
    for (int i = 0; i < periods_count; ++i)
    {
        for (const auto& p : cam_poses)
        {
            SE3Transform wfc = LookAtLufWfc(p.eye.Mat(), p.center.Mat(), p.up.Mat());
            SE3Transform cfw = SE3Inv(wfc);;
            inverse_orient_cams->push_back(cfw);
        }
    }
}

}}