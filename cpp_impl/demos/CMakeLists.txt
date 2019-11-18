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

#find_package(suriko-engine REQUIRED)
#include_directories(${suriko-engine_INCLUDE_DIRS})
#message(STATUS "suriko-engine_INCLUDE_DIRS=${suriko-engine_INCLUDE_DIRS}") #


macro(AddDepsM dem)
    target_link_libraries(${dem} suriko-engine)
    target_include_directories(${dem} PRIVATE ${PROJECT_SOURCE_DIR}/../suriko-prj/include) # TODO: use variable

    #target_include_directories(dem PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries (${dem} Eigen3::Eigen)
    target_link_libraries (${dem} gflags)
    target_link_libraries (${dem} glog::glog)
endmacro()

function(AddDepsF dem)
    if (MSVC)
        target_compile_definitions(${dem} PRIVATE _USE_MATH_DEFINES) # allow M_PI in "cmath.h"
    endif()

    target_link_libraries(${dem} suriko-engine)
    target_include_directories(${dem} PRIVATE ${PROJECT_SOURCE_DIR}/../suriko-prj/include) # TODO: use variable

    #target_include_directories(dem PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_link_libraries (${dem} Eigen3::Eigen)
    target_link_libraries (${dem} gflags)
    target_link_libraries (${dem} glog::glog)
endfunction()

set (test-helpers 
    visualize-helpers.h
    visualize-helpers.cpp
    )

add_executable(demo-dino main.cpp demo-bundle-adj-dinosaur.cpp)
#AddDepsM(demo-dino)
AddDepsF(demo-dino)
target_link_libraries(demo-dino GuidelineGSL)

add_executable(demo-circle-grid demo-bundle-adj-circle-grid.cpp ${test-helpers})
AddDepsF(demo-circle-grid)
target_link_libraries(demo-circle-grid GuidelineGSL)
target_include_directories(demo-circle-grid PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demo-circle-grid ${OpenCV_LIBS})

add_executable(demo-multi-view-factorization demo-multi-view-factorization.cpp ${test-helpers})
AddDepsF(demo-multi-view-factorization)
target_link_libraries(demo-multi-view-factorization GuidelineGSL)
target_include_directories(demo-multi-view-factorization PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demo-multi-view-factorization ${OpenCV_LIBS})
target_include_directories(demo-multi-view-factorization PRIVATE ${Pangolin_INCLUDE_DIRS})
target_link_libraries(demo-multi-view-factorization ${Pangolin_LIBRARIES})

add_executable(demo-davison-mono-slam
    davison-mono-slam/demo-davison-mono-slam.cpp
    davison-mono-slam/demo-davison-mono-slam-ui.h
    davison-mono-slam/demo-davison-mono-slam-ui.cpp
    ${test-helpers})
AddDepsF(demo-davison-mono-slam)
target_link_libraries(demo-davison-mono-slam GuidelineGSL)
target_include_directories(demo-davison-mono-slam PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demo-davison-mono-slam ${OpenCV_LIBS})
target_include_directories(demo-davison-mono-slam PRIVATE ${Pangolin_INCLUDE_DIRS})
target_link_libraries(demo-davison-mono-slam ${Pangolin_LIBRARIES})

if(UNIX)
    target_link_libraries(demo-davison-mono-slam stdc++fs) # gcc 8.3 required this to use std::filesystem
endif()
