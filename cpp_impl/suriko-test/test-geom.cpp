#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "suriko/obs-geom.h"
#include "suriko/rt-config.h"

namespace suriko_test
{
using namespace suriko;

using R = Recti;

constexpr R NullRect = R{ 0, 0, 0, 0 };

struct IntersectRectsTestData
{
    R lhs;
    R rhs;
    std::optional<R> result;
};

std::ostream& operator<<(std::ostream& os, const IntersectRectsTestData& d)
{
    os << d.lhs << " " << d.rhs << " ";
    if (d.result.has_value())
        os << d.result.value();
    else
        os << "std::nullopt";
    return os;
}

class GeomTest : public testing::TestWithParam<IntersectRectsTestData>
{
public:
	Scalar atol = (Scalar)1e-5;
};

auto GenIntersectRectTestData() -> std::vector <IntersectRectsTestData>
{
    std::vector < IntersectRectsTestData > items =
    {
        {R {12, 6, 10, 8}, R {9, 8, 6, 4},  R {12, 8, 3, 4}},
        {R {12, 6, 10, 8}, R {14, 8, 6, 4}, R {14, 8, 6, 4}},
        {R {12, 6, 10, 8}, R {19, 8, 6, 4}, R {19, 8, 3, 4}},
        {R {12, 6, 10, 8}, R {9, 8, 16, 4},  R {12, 8, 10, 4}},

        {R {12, 6, 10, 8}, R {9, 4, 6, 4},  R {12, 6, 3, 2}},
        {R {12, 6, 10, 8}, R {14, 4, 6, 4}, R {14, 6, 6, 2}},
        {R {12, 6, 10, 8}, R {19, 4, 6, 4}, R {19, 6, 3, 2}},
        {R {12, 6, 10, 8}, R {9, 4, 16, 4}, R {12, 6, 10, 2}},

        {R {12, 6, 10, 8}, R {9, 0, 6, 4}, std::nullopt},
        {R {12, 6, 10, 8}, R {14, 0, 6, 4}, std::nullopt},
        {R {12, 6, 10, 8}, R {19, 0, 6, 4}, std::nullopt},
        {R {12, 6, 10, 8}, R {9, 0, 16, 4}, std::nullopt}
    };
    return items;
}

INSTANTIATE_TEST_CASE_P(RectIntersectGen, GeomTest, testing::ValuesIn(GenIntersectRectTestData()));

TEST_P(GeomTest, RectIntersect)
{
    const IntersectRectsTestData& data = GetParam();
    std::optional<Recti> c1 = IntersectRects(data.lhs, data.rhs);
    EXPECT_EQ(data.result.value_or(NullRect), c1.value_or(NullRect));

    std::optional<Recti> c2 = IntersectRects(data.rhs, data.lhs); // exchanged
    EXPECT_EQ(data.result.value_or(NullRect), c2.value_or(NullRect));
}

}
