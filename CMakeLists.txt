cmake_minimum_required(VERSION 3.0)

file(READ src/reconstruction/gpu/kernels/constants.cl CL_KERNEL_CONSTANTS)
file(READ src/reconstruction/gpu/kernels/utils.cl CL_KERNEL_UTILS)

file(GLOB CL_KERNELS "src/reconstruction/gpu/kernels/Kernel*.cl")
foreach(file ${CL_KERNELS})
    get_filename_component(FILENAME ${file} NAME_WE)
    file(READ ${file} FILE)
    string(REPLACE "#include \"utils.cl\"" "${CL_KERNEL_UTILS}" FILE "${FILE}")
    string(REPLACE "#include \"constants.cl\"" "${CL_KERNEL_CONSTANTS}" FILE "${FILE}")
    string(CONCAT FILE "R\"(\n" "${FILE}" "\n)\"")
    string(CONCAT OUTPUT_FILE src/reconstruction/gpu/kernels/headers/ ${FILENAME} .hpp)
    file(WRITE ${OUTPUT_FILE} "${FILE}")
endforeach()

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

#Setup dependencies
find_package(OpenMP)

set(TinyTIFF_BUILD_TESTS OFF)
set(TinyTIFF_BUILD_STATIC_LIBS ON)
set(TinyTIFF_BUILD_DECORATE_LIBNAMES_WITH_BUILDTYPE OFF)
set(BUILD_TESTS OFF CACHE BOOL "disable" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "disable" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "disable" FORCE)
set(OPENCL_SDK_BUILD_SAMPLES OFF CACHE BOOL "disable" FORCE)

add_subdirectory(external/tinytiff)
add_subdirectory(external/opencl-sdk)

include_directories(external/json_struct/include)
include_directories(external/tinytiff/src)
include_directories(external/glm/glm)
include_directories(external/opencl-sdk/external/OpenCL-CLHPP/include)

#Setup source files
set(CPP_LIST ${})
add_subdirectory(src)

#Setup build
add_executable(${CMAKE_PROJECT_NAME} ${CPP_LIST})
target_link_libraries(${CMAKE_PROJECT_NAME} OpenMP::OpenMP_CXX OpenCL TinyTIFF)