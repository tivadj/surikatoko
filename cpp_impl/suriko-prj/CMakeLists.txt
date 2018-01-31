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

#
set(lib_hdrs
        ${PROJECT_SOURCE_DIR}/include/suriko/approx-alg.h
        ${PROJECT_SOURCE_DIR}/include/suriko/bundle-adj-kanatani.h
        ${PROJECT_SOURCE_DIR}/include/suriko/obs-geom.h
        ${PROJECT_SOURCE_DIR}/include/suriko/mat-serialization.h
        ${PROJECT_SOURCE_DIR}/include/suriko/rt-config.h
        )
set(lib_srcs
        ${PROJECT_SOURCE_DIR}/src/bundle-adj-kanatani.cpp
        ${PROJECT_SOURCE_DIR}/src/mat-serialization.cpp
        ${PROJECT_SOURCE_DIR}/src/obs-geom.cpp
        )

add_library(suriko-engine)
target_sources(suriko-engine
        PRIVATE ${lib_srcs} ${lib_hdrs})

target_include_directories(suriko-engine
        PUBLIC ${PROJECT_SOURCE_DIR}/include)

if (MSVC)
    target_compile_definitions(suriko-engine PRIVATE _USE_MATH_DEFINES) # allow M_PI in "cmath.h"
endif()

#target_include_directories(suriko-engine PRIVATE ${Boost_INCLUDE_DIRS})
#target_link_libraries(suriko-engine PRIVATE ${Boost_LIBRARIES})
target_link_libraries(suriko-engine PRIVATE Boost::filesystem Boost::diagnostic_definitions Boost::disable_autolinking)

#target_include_directories(suriko-engine PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries (suriko-engine PRIVATE Eigen3::Eigen)

target_link_libraries(suriko-engine PRIVATE GuidelineGSL)
target_link_libraries(suriko-engine PRIVATE glog::glog)


