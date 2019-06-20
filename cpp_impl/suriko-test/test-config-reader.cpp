#include <cmath>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "test-helpers.h"
#include "suriko/rt-config.h"
#include "suriko/config-reader.h"

namespace suriko_test
{
using namespace suriko;
using namespace suriko::config;

class ConfigReaderTest : public testing::Test
{
};

TEST_F(ConfigReaderTest, ReadValue)
{
    auto file_path = TestData("config-1.json");
    ConfigReader config_reader{ file_path };
    ASSERT_FALSE(config_reader.HasErrors()) << config_reader.Error();

    double par1_f64 = config_reader.GetValue<double>("p1_float").value_or(-999);
    EXPECT_DOUBLE_EQ(3.3, par1_f64);

    int par2_int = config_reader.GetValue<int>("p2_int").value_or(-999);
    EXPECT_EQ(17, par2_int);

    // p2 may be initialized as int, but actually be a double
    double par2_f64 = config_reader.GetValue<double>("p2_int").value_or(-999);
    EXPECT_DOUBLE_EQ(17.0, par2_f64);

    bool par3_bool = config_reader.GetValue<bool>("p3_bool").value();
    EXPECT_EQ(true, par3_bool);
    bool par4_bool = config_reader.GetValue<bool>("p4_bool").value();
    EXPECT_EQ(false, par4_bool);

    std::string par5_str = config_reader.GetValue<std::string>("p5_str").value_or("<none>");
    EXPECT_EQ(std::string("Edifice"), par5_str);

    std::vector<double> par6_f64 = config_reader.GetSeq<double>("p6_array_float").value_or(std::vector<double>{});
    ASSERT_EQ(2, par6_f64.size());
    EXPECT_DOUBLE_EQ(11.1, par6_f64[0]);
    EXPECT_DOUBLE_EQ(22.2, par6_f64[1]);

    std::vector<int> par7_int = config_reader.GetSeq<int>("p7_array_int").value_or(std::vector<int>{});
    ASSERT_EQ(2, par7_int.size());
    EXPECT_DOUBLE_EQ(320, par7_int[0]);
    EXPECT_DOUBLE_EQ(200, par7_int[1]);

    std::optional<double> par_non_existent = config_reader.GetValue<double>("par_non_existent");
    EXPECT_FALSE(par_non_existent.has_value());

    EXPECT_FALSE(config_reader.HasErrors()) << config_reader.Error();
}

TEST_F(ConfigReaderTest, WarnUnusedParams)
{
    auto file_path = TestData("config-2.json");
    ConfigReader config_reader{ file_path };
    ASSERT_FALSE(config_reader.HasErrors()) << config_reader.Error();
    
    double ratio = config_reader.GetValue<double>("gold_ratio").value();

    std::vector<std::string_view> unused_params = config_reader.GetUnusedParams();
    ASSERT_EQ(1, unused_params.size());
    EXPECT_EQ("kelvin", unused_params[0]);
}

}
