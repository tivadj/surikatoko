#pragma once
#include <optional>
#include "suriko/rt-config.h"

namespace suriko
{
/// Compute mean and standard deviation of a sequence.
class MeanStdAlgo
{
    size_t nums_count_ = 0;
    suriko::Scalar num_sum_ = 0;
    suriko::Scalar num_sqr_sum_ = 0;
    std::optional<Scalar>  min_;
    std::optional<Scalar>  max_;
public:
    void Next(suriko::Scalar num);
    void Reset();

    suriko::Scalar Mean() const;
    suriko::Scalar Std() const;
    std::optional<suriko::Scalar> Min() const;
    std::optional<suriko::Scalar> Max() const;
};

// Computes the median if the number of elements is odd, otherwise choose left element from the central pair.
std::optional<Scalar> LeftMedianInplace(std::vector<Scalar>* nums);
std::optional<Scalar> LeftMedian(const std::vector<Scalar>& nums, std::vector<Scalar>* nums_workspace);

}