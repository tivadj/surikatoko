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

option (suriko_debug "Adds extra run-time checks, slightly slowing the run-time speed" OFF)
set (suriko_scalar_type_string "f64" CACHE STRING "f32 (for c++ float) or f64 (for c++ double)")
set_property(CACHE suriko_scalar_type_string PROPERTY STRINGS f32 f64)
set(suriko_monoslam_salient_point_repres "SphericalFirstCamInvDist" CACHE STRING "A represents for salient point, 'XYZ' for 3x1 vector, 'SphericalFirstCamInvDist' for 6x1 vector")
set_property(CACHE suriko_monoslam_salient_point_repres PROPERTY STRINGS XYZ SphericalFirstCamInvDist)

message(STATUS "suriko_debug=${suriko_debug}") #
message(STATUS "suriko_scalar_type_string=${suriko_scalar_type_string}") #
message(STATUS "suriko_monoslam_salient_point_repres=${suriko_monoslam_salient_point_repres}") #

#
set(lib_hdrs
        ${PROJECT_SOURCE_DIR}/include/suriko/approx-alg.h
        ${PROJECT_SOURCE_DIR}/include/suriko/bundle-adj-kanatani.h
        ${PROJECT_SOURCE_DIR}/include/suriko/config-reader.h
        ${PROJECT_SOURCE_DIR}/include/suriko/davison-mono-slam.h
        ${PROJECT_SOURCE_DIR}/include/suriko/eigen-helpers.hpp
        ${PROJECT_SOURCE_DIR}/include/suriko/obs-geom.h
        ${PROJECT_SOURCE_DIR}/include/suriko/opengl-helpers.h
        ${PROJECT_SOURCE_DIR}/include/suriko/image-proc.h
        ${PROJECT_SOURCE_DIR}/include/suriko/mat-serialization.h
        ${PROJECT_SOURCE_DIR}/include/suriko/templ-match.h
        ${PROJECT_SOURCE_DIR}/include/suriko/rt-config.h
        ${PROJECT_SOURCE_DIR}/include/suriko/lin-alg.h
        ${PROJECT_SOURCE_DIR}/include/suriko/quat.h
        ${PROJECT_SOURCE_DIR}/include/suriko/multi-view-factorization.h
        ${PROJECT_SOURCE_DIR}/include/suriko/rand-stuff.h
        ${PROJECT_SOURCE_DIR}/include/suriko/virt-world/scene-generator.h
        ${PROJECT_SOURCE_DIR}/include/suriko/rt-config-autogen.h.in
        )
set(lib_srcs
        ${PROJECT_SOURCE_DIR}/src/bundle-adj-kanatani.cpp
        ${PROJECT_SOURCE_DIR}/src/config-reader.cpp
        ${PROJECT_SOURCE_DIR}/src/davison-mono-slam.cpp
        ${PROJECT_SOURCE_DIR}/src/image-proc.cpp
        ${PROJECT_SOURCE_DIR}/src/mat-serialization.cpp
        ${PROJECT_SOURCE_DIR}/src/obs-geom.cpp
        ${PROJECT_SOURCE_DIR}/src/opengl-helpers.cpp
        ${PROJECT_SOURCE_DIR}/src/templ-match.cpp
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

if (suriko_scalar_type_string STREQUAL "f32")
    set (SRK_SCALAR_TYPE "float")
elseif (suriko_scalar_type_string STREQUAL "f64")
    set (SRK_SCALAR_TYPE "double")
else()
    message(SEND_ERROR "Unrecognized floating type suriko_scalar_type_string=${suriko_scalar_type_string}")
endif()

set (sal_pnt_repres_id 2) # 2 for 6-element vector, see "davison-mono-slam.h"
if (suriko_monoslam_salient_point_repres STREQUAL XYZ)
    set (sal_pnt_repres_id 1)
endif()

if (1)
    set (SRK_DEBUG ${suriko_debug}) # variable is used by configure
    configure_file(
      "${PROJECT_SOURCE_DIR}/include/suriko/rt-config-autogen.h.in"
      "${PROJECT_BINARY_DIR}/rt-config-autogen.h")
endif()

#set_target_properties(suriko-engine PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(suriko-engine PUBLIC SAL_PNT_REPRES=${sal_pnt_repres_id})
target_include_directories(suriko-engine PUBLIC "${PROJECT_BINARY_DIR}")

#target_include_directories(suriko-engine PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries (suriko-engine PRIVATE Eigen3::Eigen)

target_link_libraries(suriko-engine PRIVATE GuidelineGSL)
target_link_libraries(suriko-engine PRIVATE glog::glog)

target_include_directories(suriko-engine PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(suriko-engine PRIVATE ${OpenCV_LIBS})