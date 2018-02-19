#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Eigen/Dense>
#include "suriko/eigen-helpers.hpp"

namespace suriko_test
{
using namespace suriko;
using namespace Eigen;

class EigenHelpersTest : public testing::Test
{
};

TEST_F(EigenHelpersTest, RemoveSimple1)
{
    Matrix<int, Dynamic, Dynamic> m;
    m.resize(3, 4);
    m << 1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12;

    RemoveRowsAndColsInplace(std::vector<size_t>{1}, std::vector<size_t>{2}, &m);

    Matrix<int, 2, 3> mtrunc;
    mtrunc << 1, 2, 4,
        9, 10, 12;
    ASSERT_EQ(mtrunc, m);
}

TEST_F(EigenHelpersTest, RemoveSimple2)
{
    Matrix<int, Dynamic, Dynamic> m;
    m.resize(4, 5);
    m << 1,   2,  3,  4,  5,
         6,   7,  8,  9, 10,
         11, 12, 13, 14, 15,
         16, 17, 18, 19, 20;

    RemoveRowsAndColsInplace(std::vector<size_t>{1, 2}, std::vector<size_t>{1, 2, 4}, &m);

    Matrix<int, 2, 2> mtrunc;
    mtrunc << 1, 4,
              16, 19;
    ASSERT_EQ(mtrunc, m);
}

TEST_F(EigenHelpersTest, RemoveJustColumns)
{
    Matrix<int, Dynamic, Dynamic> m;
    m.resize(2, 3);
    m << 1, 2, 3,
        4, 5, 6;
    RemoveRowsAndColsInplace(std::vector<size_t>{}, std::vector<size_t>{1}, &m);

    Matrix<int, 2, 2> m1_trunc;
    m1_trunc << 1, 3,
        4, 6;
    ASSERT_EQ(m1_trunc, m);
}

TEST_F(EigenHelpersTest, RemoveJustRows)
{
    Matrix<int, Dynamic, Dynamic> m;
    m.resize(3, 2);
    m << 1, 2,
        3, 4,
        5, 6;

    RemoveRowsAndColsInplace(std::vector<size_t>{1}, std::vector<size_t>{}, &m);

    Matrix<int, 2, 2> m_trunc;
    m_trunc << 1, 2,
        5, 6;
    ASSERT_EQ(m_trunc, m);
}
    
TEST_F(EigenHelpersTest, RemoveLeftColumns)
{
    Matrix<int, Dynamic, Dynamic> m;
    m.resize(2, 4);
    m << 1, 2, 3, 4,
          5, 6, 7, 8;
    RemoveRowsAndColsInplace(std::vector<size_t>{}, std::vector<size_t>{0, 1}, &m);

    Matrix<int, 2, 2> m1_trunc;
    m1_trunc << 3, 4,
                7, 8;
    ASSERT_EQ(m1_trunc, m);
}

TEST_F(EigenHelpersTest, RemoveRightColumns)
{
    Matrix<int, Dynamic, Dynamic> m;
    m.resize(2, 4);
    m << 1, 2, 3, 4,
          5, 6, 7, 8;
    RemoveRowsAndColsInplace(std::vector<size_t>{}, std::vector<size_t>{2, 3}, &m);

    Matrix<int, 2, 2> m1_trunc;
    m1_trunc << 1, 2,
                5, 6;
    ASSERT_EQ(m1_trunc, m);
}

TEST_F(EigenHelpersTest, RemoveTopRows)
{
    Matrix<int, Dynamic, Dynamic> m;
    m.resize(4, 2);
    m << 1, 2,
         3, 4,
         5, 6,
         7, 8;
    RemoveRowsAndColsInplace(std::vector<size_t>{0, 1}, std::vector<size_t>{}, &m);

    Matrix<int, 2, 2> m1_trunc;
    m1_trunc << 5, 6,
        7, 8;
    ASSERT_EQ(m1_trunc, m);
}

TEST_F(EigenHelpersTest, RemoveBottomRows)
{
    Matrix<int, Dynamic, Dynamic> m;
    m.resize(4, 2);
    m << 1, 2,
         3, 4,
         5, 6,
         7, 8;
    RemoveRowsAndColsInplace(std::vector<size_t>{2, 3}, std::vector<size_t>{}, &m);

    Matrix<int, 2, 2> m1_trunc;
    m1_trunc << 1, 2,
        3, 4;
    ASSERT_EQ(m1_trunc, m);
}

TEST_F(EigenHelpersTest, RemoveFirstLastRowsColumns)
{
    Matrix<int, Dynamic, Dynamic> m;
    m.resize(3, 4);
    m << 1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12;

    RemoveRowsAndColsInplace(std::vector<size_t>{0, 2}, std::vector<size_t>{0, 3}, &m);

    Matrix<int, 1, 2> mtrunc(6, 7);
    ASSERT_EQ(mtrunc, m);
}

TEST_F(EigenHelpersTest, RemoveNothing)
{
    Matrix<int, Dynamic, Dynamic> m1;
    m1.resize(2, 2);
    m1 << 1, 2,
        3, 4;
    auto m1_copy = m1;
    RemoveRowsAndColsInplace(std::vector<size_t>{}, std::vector<size_t>{}, &m1);
    ASSERT_EQ(m1_copy, m1);
}
    
TEST_F(EigenHelpersTest, RemoveAll)
{
    Matrix<int, Dynamic, Dynamic> m1;
    Matrix<int, 0, 0> m1_empty;

    // remove all rows
    m1.resize(2, 2);
    m1 << 1, 2,
        3, 4;
    RemoveRowsAndColsInplace(std::vector<size_t>{0, 1}, std::vector<size_t>{}, &m1);
	EXPECT_EQ(m1_empty, m1) << "all rows";

    // remove all columns
    m1.resize(2, 2);
    m1 << 1, 2,
        3, 4;
    RemoveRowsAndColsInplace(std::vector<size_t>{}, std::vector<size_t>{0, 1}, &m1);
	EXPECT_EQ(m1_empty, m1) << "all columns";

    // remove all rows and all columns
    m1.resize(2, 2);
    m1 << 1, 2,
        3, 4;
    RemoveRowsAndColsInplace(std::vector<size_t>{0, 1}, std::vector<size_t>{0, 1}, &m1);
	EXPECT_EQ(m1_empty, m1) << "both";
}
}
