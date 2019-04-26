# Qt5 requires CMake>2.8.9
cmake_minimum_required(VERSION 2.8.9)
project (SurikatOkoSolution)

#cmake_policy(SET CMP0020 NEW) # OLD=do not automatically link to 'qtmain.lib' when referencing QtCore target
# NOTE: the relative path after ${CMAKE_BINARY_DIR} must not be empty, otherwise the g++ link error appears:
# /usr/bin/ld: cannot open output file .: Is a directory
#set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "PROJECT_NAME=${PROJECT_NAME}")
message(STATUS "CMAKE_CURRENT_LIST_DIR=${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}")

#
#option(WITH_OPENCV "Enables visualization of images" ON)
# At the momoent, the picture object is implemented as OpenCV Mat structure. It allows easy visual debugging. For now, OpenCV is mandatory.
set (WITH_OPENCV ON)
option(WITH_PANGOLIN "Enables visualization of 3D scene" OFF)

# C++14 is required (because of std::make_unique)
# C++17 is required (because of [[noreturn]])
if (MSVC)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")

    # warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS") # suppress warning: strtok is unsafe
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING") # suppress warning: std::not1() is deprecated
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING") # suppress warning: STL4015: The std::iterator class template (used as a base class to provide typedefs) is deprecated in C++17
elseif(UNIX)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17") # tested on clang
endif()



# TODO: why libstc++?
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --verbose")

#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#    # require at least clang 3.2
#    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.9)
#        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
#    else()
#        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#    endif()
#endif()
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED NO_MODULE)
if (TARGET Eigen3::Eigen)
    message(STATUS "EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}") #
else()
    message(SEND_ERROR "Unable to find Eigen3.")
endif()

# GSL stands for Guideline Support Library
set (GuidelineGSL_INCLUDE_DIR "GuidelineGSL_INCLUDE_DIR-NOTFOUND" CACHE PATH "")

# try to find GSL
if (NOT EXISTS ${GuidelineGSL_INCLUDE_DIR})
    # set possible path candidates
    set (gsl_incl_dir_candidates "")
    if (UNIX)
        list(APPEND gsl_incl_dir_candidates "/usr/include;/usr/local/include")
    endif()
    foreach (incl_dir in ${gsl_incl_dir_candidates})
        set (anchor_file ${incl_dir}/gsl/gsl)
        if (EXISTS ${anchor_file})
            set(GuidelineGSL_INCLUDE_DIR ${incl_dir})
            break()
        endif()
    endforeach ()
endif()

if (EXISTS ${GuidelineGSL_INCLUDE_DIR})
    add_library(GuidelineGSL INTERFACE)
    target_include_directories(GuidelineGSL INTERFACE
            $<BUILD_INTERFACE:${GuidelineGSL_INCLUDE_DIR}>
            )
    message(STATUS GuidelineGSL_INCLUDE_DIR=${GuidelineGSL_INCLUDE_DIR})
else()
    message(SEND_ERROR "Unable to find GSL (Guideline Support Library) include folder. There is no directory: ${GuidelineGSL_INCLUDE_DIR}")
endif()

find_package(gflags REQUIRED)
message(STATUS "GFLAGS_INCLUDE_DIR=${GFLAGS_INCLUDE_DIR}")

find_package(glog REQUIRED)
get_target_property(glog_incl_dirs glog::glog INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "glog::glog.INTERFACE_INCLUDE_DIRECTORIES=${glog_incl_dirs}")

#enable_testing()
find_package(GTest REQUIRED)
message(STATUS "GTEST_INCLUDE_DIRS=${GTEST_INCLUDE_DIRS}")

if (WITH_OPENCV)
    find_package(OpenCV REQUIRED core highgui)
    message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
    add_definitions(-DSRK_HAS_OPENCV)
endif()

if (WITH_PANGOLIN)
    find_package(Pangolin REQUIRED)
    message(STATUS "Pangolin_INCLUDE_DIRS=${Pangolin_INCLUDE_DIRS}")
    message(STATUS "Pangolin_LIBRARIES=${Pangolin_LIBRARIES}")
    add_definitions(-DSRK_HAS_PANGOLIN)
endif()

add_subdirectory(suriko-engine)
add_subdirectory(suriko-test)
add_subdirectory(demos)
