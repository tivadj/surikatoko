#include <filesystem>
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

int main(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true); // parse flags first, as they may initialize the logger (eg: -logtostderr)
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);

    std::cout << "FLAGS_test_data=" << FLAGS_test_data <<std::endl;
    std::string test_data = GetTestData();
    if (test_data.empty())
    {
        LOG(ERROR) << "test_data param isn't set. Add --test_data=... or set SRK_TEST_DATA environment variable.";
        return 1;
    }

    TestDataRoot() = test_data;
    std::cout << "test_data=" << test_data <<std::endl;

    return RUN_ALL_TESTS();
}
