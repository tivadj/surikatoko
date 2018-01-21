#pragma once

#define SRK_DEBUG 1

#if defined(WIN32)

// This is from C:\Program Files (x86)\Windows Kits\10\Include\10.0.16299.0\ucrt\assert.h
_ACRTIMP void __cdecl _wassert(
        _In_z_ wchar_t const* _Message,
        _In_z_ wchar_t const* _File,
        _In_   unsigned       _Line
);

#define SRK_ASSERT(expression) (void)(                                                       \
            (!!(expression)) ||                                                              \
            (_wassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0) \
        )
#else
// Ubuntu
#define SRK_ASSERT(expr)  \
    (static_cast <bool> (expr) \
      ? void (0) \
      : __assert_fail (#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))

#endif

namespace suriko {

static const bool kSurikoDebug = true;

typedef double Scalar;
}