#pragma once
#include <exception> // std::terminate
#include <gsl/gsl_assert>

#define SRK_DEBUG 1

// SRK_ASSERT is similar to standard assert macros, but can be
#if defined(SRK_DEBUG)
#include <glog/logging.h>
// use Google glog CHECK macro
#define SRK_ASSERT(expr) CHECK(expr)
#else
#define SRK_ASSERT(expr) void(0)
#endif

namespace suriko {

	static const bool kSurikoDebug =
#if defined(SRK_DEBUG)
		true;
#else
		false;
#endif

typedef double Scalar;

/// Indicates that the point of function call is never reached. This allows to satisfy the compiler,
/// which otherwise emits a warning "not all control paths return a value".
//[[noreturn]] inline void AssertFalse() { std::terminate(); }
[[noreturn]] inline void AssertFalse() { Ensures(false); }
}