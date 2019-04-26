#include "suriko/lin-alg.h"
#include "suriko/obs-geom.h"

namespace suriko
{
void OrthonormalizeGramSchmidtInplace(Eigen::Matrix<Scalar, 3, 3>* mat)
{
    // source: "3D Math Primer for Graphics and Game Development 2nd", Fletcher Dunn, 2011, paragraph 6.3.3
    Eigen::Matrix<Scalar, 3, 1> r1 = mat->leftCols<1>();
    r1.normalize();

    Eigen::Matrix<Scalar, 3, 1> r2 = mat->middleCols<1>(1);
    r2 -= r2.dot(r1) * r1;
    r2.normalize();

    Eigen::Matrix<Scalar, 3, 1> r3 = r1.cross(r2);

    *mat << r1, r2, r3;

    if (kSurikoDebug)
    {
        std::string msg;
        bool op = IsSpecialOrthogonal(*mat, &msg);
        SRK_ASSERT(op) << msg;
    }
}
}
