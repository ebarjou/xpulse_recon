/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

/**
 * @brief Wrapper class that represent a GPU program with a number of kernels
 * 
 */
class Program {
    reconstruction::gpu::Ocl _ocl;
    cl::Program _program;
    std::vector<cl::Kernel> _kernel;
    uint32_t x = 1, y = 1, z = 1;
    uint32_t offset_x = 0, offset_y = 0, offset_z = 0;
public:
    Program(reconstruction::gpu::Ocl &ocl, const std::string source, std::vector<const char *> names) :
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
        for(const char * name : names) {
            cl_int error_kernel_creation;
            _kernel.push_back(cl::Kernel{_program, name, &error_kernel_creation});
            CHECK(error_kernel_creation)
        }
    }

    template <typename T>
    void setKernelArgument(int64_t kid, int64_t index, T value) {
        CHECK(_kernel[kid].setArg(cl_uint(index), value));
    }

    void setKernelArgument(int64_t kid, int64_t index, reconstruction::gpu::Buffer buffer) {
        CHECK(_kernel[kid].setArg(cl_uint(index), buffer.handle));
    }

    void setProgramSizeX(int64_t x) {
        this->x = uint32_t(x);
    }

    void setProgramSizeY(int64_t y) {
        this->y = uint32_t(y);
    }

    void setProgramSizeZ(int64_t z) {
        this->z = uint32_t(z);
    }

    void setProgramSize(int64_t x, int64_t y = 1, int64_t z = 1) {
        this->x = uint32_t(x);
        this->y = uint32_t(y);
        this->z = uint32_t(z);
    }

    void setProgramOffset(int64_t x, int64_t y = 0, int64_t z = 0) {
        this->offset_x = uint32_t(x);
        this->offset_y = uint32_t(y);
        this->offset_z = uint32_t(z);
    }

    void executeKernel(cl::CommandQueue &queue, int64_t kid) {
        uint32_t xGlobal = x + (WAVEFRONT_SIZE - x%WAVEFRONT_SIZE);
        CHECK(queue.enqueueNDRangeKernel(_kernel[kid], cl::NDRange{offset_x, offset_y, offset_z}, cl::NDRange{xGlobal, y, z}, cl::NDRange{WAVEFRONT_SIZE, 1, 1}));
    }
};