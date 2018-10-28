project(suriko-engine-prj)

message(STATUS "PROJECT_NAME=${PROJECT_NAME}")
message(STATUS "CMAKE_CURRENT_LIST_DIR=${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}") #
message(STATUS "CMAKE_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}") #
message(STATUS "PROJECT_ARCH=${PROJECT_ARCH}") #
message(STATUS "PLATFORM_SHORT_NAME=${PLATFORM_SHORT_NAME}") #
message(STATUS "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}") #

option (SURIKO_DEBUG "Adds extra run-time checks, slightly slowing the run-time speed" OFF)
set (SURIKO_SCALAR_TYPE_STRING "f64" CACHE STRING "f32 (for c++ float) or f64 (for c++ double)")
set_property(CACHE SURIKO_SCALAR_TYPE_STRING PROPERTY STRINGS f32 f64)
#
set(lib_hdrs
        ${PROJECT_SOURCE_DIR}/include/suriko/approx-alg.h
        ${PROJECT_SOURCE_DIR}/include/suriko/bundle-adj-kanatani.h
        ${PROJECT_SOURCE_DIR}/include/suriko/davison-mono-slam.h
        ${PROJECT_SOURCE_DIR}/include/suriko/eigen-helpers.hpp
        ${PROJECT_SOURCE_DIR}/include/suriko/obs-geom.h
        ${PROJECT_SOURCE_DIR}/include/suriko/opengl-helpers.h
        ${PROJECT_SOURCE_DIR}/include/suriko/mat-serialization.h
        ${PROJECT_SOURCE_DIR}/include/suriko/rt-config.h
        ${PROJECT_SOURCE_DIR}/include/suriko/lin-alg.h
        ${PROJECT_SOURCE_DIR}/include/suriko/quat.h
        ${PROJECT_SOURCE_DIR}/include/suriko/multi-view-factorization.h
        ${PROJECT_SOURCE_DIR}/include/suriko/rand-stuff.h
        ${PROJECT_SOURCE_DIR}/include/suriko/virt-world/scene-generator.h
        )
set(lib_srcs
        ${PROJECT_SOURCE_DIR}/src/bundle-adj-kanatani.cpp
        ${PROJECT_SOURCE_DIR}/src/davison-mono-slam.cpp
        ${PROJECT_SOURCE_DIR}/src/mat-serialization.cpp
        ${PROJECT_SOURCE_DIR}/src/obs-geom.cpp
        ${PROJECT_SOURCE_DIR}/src/opengl-helpers.cpp
        ${PROJECT_SOURCE_DIR}/src/quat.cpp
        ${PROJECT_SOURCE_DIR}/src/lin-alg.cpp
        ${PROJECT_SOURCE_DIR}/src/multi-view-factorization.cpp
        ${PROJECT_SOURCE_DIR}/src/virt-world/scene-generator.cpp
        )

add_library(suriko-engine)
target_sources(suriko-engine
        PRIVATE ${lib_srcs} ${lib_hdrs})

target_include_directories(suriko-engine
        PUBLIC ${PROJECT_SOURCE_DIR}/include)

if (MSVC)
    # C1128: number of sections exceeded object file format limit: compile with /bigobj
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
 
    target_compile_definitions(suriko-engine 
        PRIVATE _USE_MATH_DEFINES # allow M_PI in "cmath.h"
    )
endif()

set (SRK_DEBUG ${SURIKO_DEBUG})
if (SURIKO_SCALAR_TYPE_STRING STREQUAL "f32")
    set (SRK_SCALAR_TYPE "float")
elseif (SURIKO_SCALAR_TYPE_STRING STREQUAL "f64")
    set (SRK_SCALAR_TYPE "double")
else()
    message(SEND_ERROR "Unrecognized floating type SURIKO_SCALAR_TYPE_STRING=${SURIKO_SCALAR_TYPE_STRING}")
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/include/suriko/rt-config-autogen.h.in"
  "${PROJECT_BINARY_DIR}/rt-config-autogen.h"
  )

target_include_directories(suriko-engine PUBLIC "${PROJECT_BINARY_DIR}")

#target_include_directories(suriko-engine PRIVATE ${Boost_INCLUDE_DIRS})
#target_link_libraries(suriko-engine PRIVATE ${Boost_LIBRARIES})
target_link_libraries(suriko-engine PRIVATE Boost::filesystem Boost::diagnostic_definitions Boost::disable_autolinking)

#target_include_directories(suriko-engine PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries (suriko-engine PRIVATE Eigen3::Eigen)

target_link_libraries(suriko-engine PRIVATE GuidelineGSL)
target_link_libraries(suriko-engine PRIVATE glog::glog)

target_include_directories(suriko-engine PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(suriko-engine ${OpenCV_LIBS})