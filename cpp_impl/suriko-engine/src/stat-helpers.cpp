#include "suriko/stat-helpers.h"
#include "suriko/approx-alg.h"

namespace suriko
{
using namespace suriko;

void MeanStdAlgo::Next(Scalar num)
{
    num_sum_ += num;
    num_sqr_sum_ += Sqr(num);
    nums_count += 1;
}

void MeanStdAlgo::Reset()
{
    num_sum_ = 0;
    num_sqr_sum_ = 0;
    nums_count = 0;
}

Scalar MeanStdAlgo::Mean() const
{
    return num_sum_ / nums_count;
}

Scalar MeanStdAlgo::Std() const
{
    return num_sqr_sum_ / nums_count - Sqr(num_sum_ / nums_count);
}
}