#pragma once
#include <vector>
#include <gsl/span>
#include "suriko/rt-config.h"
#include "suriko/obs-geom.h"

namespace suriko { namespace virt_world {

struct WorldBounds
{
    Scalar x_min;
    Scalar x_max;
    Scalar y_min;
    Scalar y_max;
    Scalar z_min;
    Scalar z_max;
};

void GenerateCircleCameraShots(const suriko::Point3& circle_center, Scalar circle_radius, Scalar ascentZ,
    gsl::span<const Scalar> rot_angles, std::vector<SE3Transform>* inverse_orient_cams);

}}