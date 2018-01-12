project(engine)

message(STATUS "PROJECT_NAME=${PROJECT_NAME}")
message(STATUS "CMAKE_CURRENT_LIST_DIR=${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}") #
message(STATUS "CMAKE_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}") #
message(STATUS "PROJECT_ARCH=${PROJECT_ARCH}") #
message(STATUS "PLATFORM_SHORT_NAME=${PLATFORM_SHORT_NAME}") #
message("PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}") #

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

add_library(engine)
target_sources(engine
        PRIVATE ${lib_srcs}
        INTERFACE ${lib_hdrs})
#export(PACKAGE ${PROJECT_NAME})

target_include_directories(engine PRIVATE
        ${PROJECT_SOURCE_DIR}/include)

target_include_directories(engine PRIVATE ${EIGEN3_INCLUDE_DIR})
