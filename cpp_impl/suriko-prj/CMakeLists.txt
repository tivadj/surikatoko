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
        ${PROJECT_SOURCE_DIR}/include/suriko/approx-alg.hpp
        ${PROJECT_SOURCE_DIR}/include/suriko/bundle-adj-kanatani.hpp
        ${PROJECT_SOURCE_DIR}/include/suriko/eigen-helpers.hpp
        ${PROJECT_SOURCE_DIR}/include/suriko/obs-geom.hpp
        ${PROJECT_SOURCE_DIR}/include/suriko/mat-serialization.hpp
        ${PROJECT_SOURCE_DIR}/include/suriko/rt-config.h
        )
#set(src_headers
#        include/suriko
#        )

#add_library(engine ${src_headers}) # NOTE: we do not need library, because there are no *.cpp files
add_library(engine INTERFACE)
target_sources(engine INTERFACE ${lib_hdrs})
#export(PACKAGE ${PROJECT_NAME})

target_include_directories(engine INTERFACE
        ${PROJECT_SOURCE_DIR}/include)
#include_directories(${src_headers})
