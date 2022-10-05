cmake_minimum_required(VERSION 3.0)

set(KERNEL_FILE src/reconstruction/gpu/kernel_recon)
file(READ ${KERNEL_FILE}.hpp CL_KERNEL)
string(CONCAT CL_KERNEL "R\"(\n" "${CL_KERNEL}" "\n)\"")
file(WRITE ${KERNEL_FILE}_str.hpp "${CL_KERNEL}")

#Setup project
project(XPRecon VERSION 1.0 LANGUAGES CXX)

#Setup environment 
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CL_HPP_TARGET_OPENCL_VERSION 300)

if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2)
endif()

#Setup dependencies
find_package(OpenMP)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "disable" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "disable" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "disable" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "disable" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "disable" FORCE)

set(TinyTIFF_BUILD_TESTS OFF)
set(TinyTIFF_BUILD_STATIC_LIBS ON)
set(TinyTIFF_BUILD_DECORATE_LIBNAMES_WITH_BUILDTYPE OFF)

set(OPENCL_SDK_BUILD_SAMPLES OFF CACHE BOOL "disable" FORCE)
set(OPENCL_SDK_TEST_SAMPLES OFF CACHE BOOL "disable" FORCE)

add_subdirectory(external/opencl-sdk)
add_subdirectory(external/tinytiff)

include_directories(external/json_struct/include)
include_directories(external/glm/glm)
include_directories(external/opencl-sdk/external/OpenCL-CLHPP/include)
include_directories(external/pocketfft)
include_directories(external/tinytiff/src)

#Setup source files
set(CPP_LIST src/main.cpp)

#Setup build
add_executable(${CMAKE_PROJECT_NAME} ${CPP_LIST})
target_link_libraries(${CMAKE_PROJECT_NAME} OpenMP::OpenMP_CXX OpenCL TinyTIFF)