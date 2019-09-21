#pragma once
#include "suriko/rt-config.h"

namespace suriko
{
/// Compute mean and standard deviation of a sequence.
class MeanStdAlgo
{
    size_t nums_count = 0;
    suriko::Scalar num_sum_ = 0;
    suriko::Scalar num_sqr_sum_ = 0;
public:
    void Next(suriko::Scalar num);
    void Reset();

    suriko::Scalar Mean() const;
    suriko::Scalar Std() const;
};
}