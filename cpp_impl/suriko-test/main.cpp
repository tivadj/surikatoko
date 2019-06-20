#include <cstdlib>  // getenv
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "test-helpers.h"

DEFINE_string(test_data, "", "Root path to test data (eg: --test_data=c:/mydata)");

std::string GetTestData()
{
    std::string test_data = FLAGS_test_data;
    if (!test_data.empty())
        return test_data;

    const char* p_test_data = std::getenv("SRK_TEST_DATA");
    if (p_test_data != nullptr)
        return std::string(p_test_data);
    return std::string();
}

int main(int ac, char* av[])
{
    testing::InitGoogleTest(&ac, av);
    google::InitGoogleLogging(av[0]);

    std::string test_data = GetTestData();
    TestDataRoot() = test_data;
    LOG(INFO) << "test_data=" << test_data;

    return RUN_ALL_TESTS();
}
