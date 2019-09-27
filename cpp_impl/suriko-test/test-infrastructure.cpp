#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "test-helpers.h"
#include "suriko/rt-config.h"
#include "suriko/config-reader.h"
#include "suriko/obs-geom.h"

namespace suriko_test
{
using namespace suriko;
using namespace suriko::config;

class InfrastructureTest : public testing::Test
{
};

bool SideEffect(int* pv, int new_value, bool ret)
{
    *pv = new_value;
    return ret;
}

TEST_F(InfrastructureTest, TestSurikoAssert)
{
    SRK_ASSERT(false) << "chu";

    // assert can contain routines with side-effects, eg: assert(f->write(x) == 4)
    int i = 111;
    SRK_ASSERT(SideEffect(&i, 222, true));
    ASSERT_EQ(222, i) << "side effects are done";

    // assert can be used in if-else branches without curly braces
    if (i == 17)
        SRK_ASSERT(true);
    else
        SRK_ASSERT(true);
}

}
