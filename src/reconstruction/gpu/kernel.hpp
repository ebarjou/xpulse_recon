/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

#include "kernels/constants.cl"

/**
 * @brief Wrapper class that represent a GPU kernel
 * 
 */
class Kernel {
    Ocl _ocl;
    cl::Program _program;
    cl::Kernel _kernel;
    uint32_t x = 1, y = 1, z = 1;
public:
    Kernel(Ocl &ocl, std::string source, const char *name) :
            _ocl(ocl)
    {
        cl_int error_program_creation;
        _program = cl::Program{_ocl.context, source, false, &error_program_creation};
        CHECK(error_program_creation)
        cl_int errorCode = _program.build("-cl-fast-relaxed-math\
                                           -cl-mad-enable\
                                           -cl-no-signed-zeros\
                                           -cl-unsafe-math-optimizations\
                                           -cl-finite-math-only\
                                         "); //-cl-std=CL3.0

        if (errorCode == CL_BUILD_PROGRAM_FAILURE) {
            std::string buildLog = _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(_ocl.device);
            std::cerr << "OpenCl kernel compile output : " << std::endl << buildLog << std::endl;
            throw std::runtime_error("kernel compilation error");
        }
        CHECK(errorCode);

        //-cl-uniform-work-group-size -cl-no-subgroup-ifp
        cl_int error_kernel_creation;
        _kernel = cl::Kernel{_program, name, &error_kernel_creation};
        CHECK(error_kernel_creation)
    }

    template <typename T>
    void setKernelArgument(int64_t index, T value) {
        CHECK(_kernel.setArg(cl_uint(index), value));
    }

    void setKernelArgument(int64_t index, Buffer buffer) {
        CHECK(_kernel.setArg(cl_uint(index), buffer.handle));
    }

    void setKernelSizeX(int64_t x) {
        this->x = uint32_t(x);
    }

    void setKernelSizeY(int64_t y) {
        this->y = uint32_t(y);
    }

    void setKernelSizeZ(int64_t z) {
        this->z = uint32_t(z);
    }

    void setKernelSize(int64_t x, int64_t y = 1, int64_t z = 1) {
        this->x = uint32_t(x);
        this->y = uint32_t(y);
        this->z = uint32_t(z);
    }

    void executeKernel() {
        uint32_t xGlobal = x + (WAVEFRONT_SIZE - x%WAVEFRONT_SIZE);
        CHECK(_ocl.queue.enqueueNDRangeKernel(_kernel, cl::NullRange, cl::NDRange{xGlobal, y, z}, cl::NDRange{WAVEFRONT_SIZE, 1, 1}));
    }

    void executeKernel(cl::CommandQueue &queue) {
        uint32_t xGlobal = x + (WAVEFRONT_SIZE - x%WAVEFRONT_SIZE);
        CHECK(queue.enqueueNDRangeKernel(_kernel, cl::NullRange, cl::NDRange{xGlobal, y, z}, cl::NDRange{WAVEFRONT_SIZE, 1, 1}));
    }
};