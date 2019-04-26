#pragma once
#include <Eigen/Dense>
#include "suriko/rt-config.h"

namespace suriko
{
void OrthonormalizeGramSchmidtInplace(Eigen::Matrix<Scalar, 3, 3>* mat);
}