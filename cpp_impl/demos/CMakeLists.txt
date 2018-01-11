project(demos)
cmake_minimum_required(VERSION 2.9.1)

message(STATUS "PROJECT_NAME=${PROJECT_NAME}")
message(STATUS "CMAKE_CURRENT_LIST_DIR=${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}") #
message(STATUS "CMAKE_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}") #
message(STATUS "PROJECT_ARCH=${PROJECT_ARCH}") #
message(STATUS "PLATFORM_SHORT_NAME=${PLATFORM_SHORT_NAME}") #

#find_package(engine REQUIRED)
#include_directories(${engine_INCLUDE_DIRS})
#message(STATUS "engine_INCLUDE_DIRS=${engine_INCLUDE_DIRS}") #

find_package(Eigen3 REQUIRED)
message(STATUS "EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}") #
include_directories(${EIGEN3_INCLUDE_DIR})


#
set(srcs main.cpp demo-bundle-adj-dinosaur.cpp)
add_executable(demos ${srcs})

target_link_libraries(demos engine)

#target_link_libraries(demos engine)
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(demos ${Boost_LIBRARIES})
