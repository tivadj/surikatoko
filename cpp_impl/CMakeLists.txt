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

# at least C++14 is required (eg std::make_unique)
if (MSVC)
	#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++latest")
elseif(UNIX)
	# PIC (position-independent code) is required by boost_filesystem (compile boost with this cxxflags=-fPIC too)

endif()



# TODO: why libstc++?
# break boost::filesystem: error: no member named 'move' in namespace 'std'
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --verbose")

#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#    # require at least clang 3.2
#    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.9)
#        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
#    else()
#        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#    endif()
#endif()
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(suriko-prj)
add_subdirectory(demos)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
