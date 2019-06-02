#pragma once
#include <cmath> // M_PI
#include <type_traits> // std::common_type
namespace suriko
{
// https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.isclose.html
template<typename F1, typename F2>
bool IsClose(F1 a, F2 b,
             typename std::common_type<F1,F2>::type rtol = 1.0e-5,
             typename std::common_type<F1,F2>::type atol = 1.0e-8)
{
    typedef typename std::common_type<F1,F2>::type F;
    auto aa = static_cast<F>(a);
    auto bb = static_cast<F>(b);
    return std::abs(a - b) <= (atol + rtol * std::abs(std::max<F>(aa, bb)));
}

template<typename F1, typename F2, typename F3>
bool IsCloseAbs(F1 a, F2 b, F3 atol = 1.0e-8)
{
    using F = typename std::common_type<F1, F2>::type;
    return std::abs(a - b) <= static_cast<F>(atol);
}

template<typename F>
bool IsFinite(F x)
{
    return std::isfinite(x);
}

template <typename F>
constexpr auto Sqr(F x) -> F { return x*x; }

template <typename F>
constexpr auto Pow3(F x) -> F { return x*x*x; }

template <typename F>
constexpr auto Pow4(F x) -> F { return x*x*x*x; }

template <typename F>
constexpr auto Sign(F x) -> int { return x >= 0 ? 1 : -1; }

template <typename F>
constexpr auto CeilPow2N(F x) { return std::pow(2, std::ceil(std::log2(x))); };

template <typename F>
constexpr auto Pi() { return static_cast<F>(M_PI); }

}
