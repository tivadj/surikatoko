#pragma once

#define SRK_ASSERT(expr)  \
    (static_cast <bool> (expr) \
      ? void (0) \
      : __assert_fail (#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))

static const bool kSurikoDebug = true;