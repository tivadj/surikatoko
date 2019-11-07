#include <algorithm>
#include "suriko/stat-helpers.h"
#include "suriko/approx-alg.h"

namespace suriko
{
using namespace suriko;

void MeanStdAlgo::Reset()
{
    num_sum_ = 0;
    num_sqr_sum_ = 0;
    nums_count_ = 0;
    min_.reset();
    max_.reset();
}

void MeanStdAlgo::Next(Scalar num)
{
    num_sum_ += num;
    num_sqr_sum_ += Sqr(num);
    nums_count_ += 1;
    if (!min_.has_value() || num < min_.value())
        min_ = num;
    if (!max_.has_value() || num > max_.value())
        max_ = num;
}

Scalar MeanStdAlgo::Mean() const
{
    return num_sum_ / nums_count_;
}

Scalar MeanStdAlgo::Std() const
{
    Scalar var = num_sqr_sum_ / nums_count_ - Sqr(num_sum_ / nums_count_);
    return std::sqrt(var);
}

std::optional<suriko::Scalar> MeanStdAlgo::Min() const
{
    return min_;
}
std::optional<suriko::Scalar> MeanStdAlgo::Max() const
{
    return max_;
}

std::optional<Scalar> LeftMedianInplace(std::vector<Scalar>* nums)
{
    auto& ns = *nums;
    if (ns.empty()) return std::nullopt;

    auto mid = ns.begin() + ns.size() / 2;
    std::nth_element(ns.begin(), mid, ns.end());  // O(n)
    Scalar result = *mid;  // as if elements count is always odd
    return result;
}

std::optional<Scalar> LeftMedian(const std::vector<Scalar>& nums, std::vector<Scalar>* nums_workspace)
{
    if (nums.empty()) return std::nullopt;

    auto& nums_copy = *nums_workspace;
    nums_copy.assign(nums.begin(), nums.end());
    return LeftMedianInplace(&nums_copy);
}

}
