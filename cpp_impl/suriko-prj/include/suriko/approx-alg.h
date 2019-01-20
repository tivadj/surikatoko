#pragma once
#include <cmath>
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
    return std::abs(a - b) <= (atol + rtol * std::abs(std::max<F>(a, b)));
}

/// Absolute/relative tolerance.
template <typename F>
struct AbsRelTol
{
    F ATol; // absolute tolerance
    F RTol; // relative tolerance
    AbsRelTol(F atol, F rtol) : ATol(atol), RTol(rtol) {}
};

template <typename F>
AbsRelTol<F> AbsTol(F atol) { return AbsRelTol<F>(atol, 0); }

template <typename F>
AbsRelTol<F> RelTol(F rtol) { return AbsRelTol<F>(0, rtol); }

template<typename F1, typename F2>
bool IsClose(F1 a, F2 b, AbsRelTol<typename std::common_type<F1, F2>::type> tol)
{
    return IsClose(a, b, tol.RTol, tol.ATol);
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

}
