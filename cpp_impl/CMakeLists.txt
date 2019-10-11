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
# At the momoent, the picture object is implemented as OpenCV Mat structure. It allows easy visual debugging. For now, OpenCV is mandatory.
#option(suriko_with_opencv "Enables visualization of images" ON)
set(suriko_with_opencv ON)
option(suriko_with_pangolin "Enables visualization of 3D scene" OFF)
option(suriko_build_tests "Include unit tests" OFF)

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
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

    # ask GCC to don't use PIC (position-independent code) to fix
    # linker error: relocation R_X86_64_32 against `.rodata' can not be used when making a PIE object; recompile with -fPIC
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-pie")
endif()

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

if (suriko_with_opencv)
    # features2d for cv::ORB
    find_package(OpenCV REQUIRED core imgcodecs imgproc highgui features2d calib3d)
    message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
    add_definitions(-DSRK_HAS_OPENCV)
endif()

if (suriko_with_pangolin)
    find_package(Pangolin REQUIRED)
    message(STATUS "Pangolin_INCLUDE_DIRS=${Pangolin_INCLUDE_DIRS}")
    message(STATUS "Pangolin_LIBRARIES=${Pangolin_LIBRARIES}")
    add_definitions(-DSRK_HAS_PANGOLIN)
endif()

if (suriko_build_tests)
    #enable_testing()
    find_package(GTest REQUIRED)
    message(STATUS "GTEST_INCLUDE_DIRS=${GTEST_INCLUDE_DIRS}")
endif()

#
add_subdirectory(suriko-engine)
add_subdirectory(demos)
if (suriko_build_tests)
    add_subdirectory(suriko-test)
endif()
