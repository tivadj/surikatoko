project(suriko-test-prj)

add_executable(suriko-test
        main.cpp
        test-bundle-adj-kanatani.cpp
        test-eigen-helpers.cpp
        test-obs-geom.cpp)

# GTEST_HAS_TR1_TUPLE=0 says there is no std::tr1
# GTEST_HAS_STD_TUPLE_=1 says the std::tuple exist
target_compile_definitions(suriko-test PRIVATE GTEST_HAS_TR1_TUPLE=0 GTEST_HAS_STD_TUPLE_=1)

if (MSVC)
    target_compile_definitions(suriko-engine PRIVATE _USE_MATH_DEFINES) # allow M_PI in "cmath.h"
endif()

target_link_libraries(suriko-test Eigen3::Eigen)
target_link_libraries(suriko-test glog::glog)
target_link_libraries(suriko-test GTest::GTest)
target_link_libraries(suriko-test suriko-engine)
