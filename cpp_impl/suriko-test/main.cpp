#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int ac, char* av[])
{
    testing::InitGoogleTest(&ac, av);
    google::InitGoogleLogging(av[0]);
    return RUN_ALL_TESTS();
}
