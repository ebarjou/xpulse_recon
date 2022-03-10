/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

enum MEM_ACCESS {MEM_ACCESS_READ, MEM_ACCESS_WRITE, MEM_ACCESS_RW, MEM_ACCESS_NONE};

struct Ocl {
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::CommandQueue readQueue;
    cl::CommandQueue writeQueue;
    uint64_t memorySize = 0, maxAllocSize = 0;
};

struct Buffer {
    cl::Buffer handle;
    uint64_t elements;
    uint8_t bytePerSample;
    MEM_ACCESS kernelAccess, hostAccess;
};

/**
 * @brief Wrapper class the the OpenGL context
 * 
 */
class Context {
    Ocl _ocl;

public:
    Context() {
        int platformIndex = 0;
        int deviceIndex = 0;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        std::cout << "Compatible devices : " << std::endl;

        for(auto platform : platforms) {
            std::cout << "\t" <<platform.getInfo<CL_PLATFORM_NAME>().c_str() << std::endl;
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for(auto device : devices) {
                uint64_t memorySize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
                std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
                std::cout << "\t\t" << deviceName << " " << memorySize/(1024*1024) << "Mb" << std::endl;
                std::for_each(deviceName.begin(), deviceName.end(), [](char & c) { c = std::tolower(c); });
                if( (platformName.find("NVIDIA") != std::string::npos || platformName.find("AMD") != std::string::npos) 
                        && memorySize > _ocl.memorySize)
                {
                    _ocl.device = device;
                    _ocl.memorySize = memorySize;
                    _ocl.maxAllocSize = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
                }
            }
        }
        if(_ocl.memorySize == 0) {
            throw std::runtime_error("No compatible device found");
        }

        std::cout << "Loading OpenCL..." << std::flush;
        cl_int error_context_creation, error_queue_creation;
        _ocl.context = cl::Context{ _ocl.device, false, nullptr, nullptr, &error_context_creation};
        CHECK(error_context_creation)
        _ocl.queue = cl::CommandQueue{ _ocl.context, _ocl.device, CL_QUEUE_PROFILING_ENABLE, &error_queue_creation };
        CHECK(error_queue_creation)
        _ocl.readQueue = cl::CommandQueue{ _ocl.context, _ocl.device, CL_QUEUE_PROFILING_ENABLE, &error_queue_creation };
        CHECK(error_queue_creation)
        _ocl.writeQueue = cl::CommandQueue{ _ocl.context, _ocl.device, CL_QUEUE_PROFILING_ENABLE, &error_queue_creation };
        CHECK(error_queue_creation)
        std::cout << "Ok." << std::endl;
        //TODO : fix AMD memory detection
        //Get the real available size for our buffers
        //This method seems to work correctly on NVIDIA but not on AMD gpus
        //Otherwise, just take a percent of total memory (ex "_ocl.memorySize *= 0.8;")
        {
            _ocl.memorySize = 0;
            uint32_t sizeStep = 128*1024*1024; //256Mb
            std::vector<cl::Buffer> buffers;
            while(true) {
                try {
                    buffers.push_back(allocateBuffer(sizeStep, MEM_ACCESS_READ, MEM_ACCESS_READ));
                    wait();
                    _ocl.memorySize += sizeStep;
                } catch(...) {
                    break;
                }
            }
        }
        std::cout << "Selected " << _ocl.device.getInfo<CL_DEVICE_NAME>() << ": " << _ocl.memorySize/(1024.0*1024.0) << "Mb available, " << _ocl.maxAllocSize/(1024.0*1024.0) << "Mb max per alloc." << std::endl;
    }

    Ocl getOcl() {
        return _ocl;
    }

    /**
     * @brief Flush and finish all pending OpenCL call
     * 
     */
    void wait() {
        CHECK(_ocl.queue.flush());
        CHECK(_ocl.queue.finish());
    }

    /**
     * @brief Create a buffer of specific size
     * 
     * @param T type of the elements of the buffer
     * @param elements number of element, of type T
     * @param kernel specifies if the buffer will be read/written by a kernel
     * @param host specifies if the buffer will be read/written by the host
     */
    template <typename T, 
          typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Buffer createBuffer(uint64_t elements, const MEM_ACCESS kernel = MEM_ACCESS_RW, const MEM_ACCESS host = MEM_ACCESS_RW, const uint8_t sizeAlignment = 0) {
        if(elements == 0) elements = 1;
        if(sizeAlignment > 1) elements += sizeAlignment - elements%sizeAlignment;
        Buffer buffer;
        buffer.handle = allocateBuffer(elements*sizeof(T), kernel, host);
        buffer.elements = elements;
        buffer.bytePerSample = sizeof(T);
        buffer.kernelAccess = kernel;
        buffer.hostAccess = host;
        return buffer;
    }

    /**
     * @brief Create a buffer from a vector
     * 
     * @param storage vector containing the data to be stored in the buffer
     * @param kernel specifies if the buffer will be read/written by a kernel
     * @param host specifies if the buffer will be read/written by the host
     */
    template <typename T, 
          typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    Buffer createBuffer(std::vector<T> &storage, const MEM_ACCESS kernel = MEM_ACCESS_RW, const MEM_ACCESS host = MEM_ACCESS_RW) {
        Buffer buffer;
        buffer.handle = allocateBuffer(storage.size()*sizeof(T), kernel, host, (void*)storage.data());
        buffer.elements = storage.size();
        buffer.bytePerSample = sizeof(T);
        buffer.kernelAccess = kernel;
        buffer.hostAccess = host;
        return buffer;
    }

    /**
     * @brief Set the content of a buffer to the one contained in an std::vector
     * 
     * @param buffer buffer to process
     * @param storage vector containing the data to be copied to the buffer
     * @param blocking specifies if the copy is synchronous or not. If set to true, will use a queue dedicated to CPU->GPU transfert.
     */
    template <typename T, 
          typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    void setBuffer(const Buffer buffer, std::vector<T> &storage, const bool blocking = false) {
        if(storage.empty()) return;
        if(storage.size() > buffer.elements) return;
        if(sizeof(T) != buffer.bytePerSample) return;
        if(blocking) {
            CHECK(_ocl.writeQueue.enqueueWriteBuffer(buffer.handle, CL_TRUE, 0, storage.size()*sizeof(T), storage.data()));
        } else {
            CHECK(_ocl.queue.enqueueWriteBuffer(buffer.handle, CL_FALSE, 0, storage.size()*sizeof(T), storage.data()));
        }
    }

    /**
     * @brief Set the content of a buffer to the one contained in an std::vector
     * 
     * @param buffer buffer to process
     * @param value
     * @param elements
     */
    template <typename T, 
          typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    void setBuffer(const Buffer buffer, const T value) {
        if(sizeof(T) != buffer.bytePerSample) return;
        CHECK(_ocl.queue.enqueueFillBuffer(buffer.handle, value, 0, buffer.elements*sizeof(T)));
    }

    /**
     * @brief Set the content of a buffer to the one contained in an std::vector
     * 
     * @param dst
     * @param src
     * @param elements
     */
    template <typename T, 
          typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    void copyBuffer(const Buffer dst, const Buffer src) {
        if(elements == 0) return;
        if(dst.elements != src.elements) return;
        if(dst.bytePerSample != src.bytePerSample) return;
        CHECK(_ocl.queue.enqueueCopyBuffer(values.handle, buffer.handle, 0, 0, elements*sizeof(T)));
    }

    /**
     * @brief Cpopy the content of a buffer to an std::vector (which will be resized by the function)
     * 
     * @param buffer buffer to get data from
     * @param storage vector where the values are copied
     * @param blocking specifies if the copy is synchronous or not. If set to true, will use a queue dedicated to CPU->GPU transfert.
     */
    template <typename T, 
          typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    void getBuffer(const Buffer buffer, std::vector<T>& storage, const bool blocking = false) {
        storage.resize(buffer.elements);
        if(blocking) {
            CHECK(_ocl.readQueue.enqueueReadBuffer(buffer.handle, CL_TRUE, 0, buffer.elements*sizeof(T), storage.data()));
        } else {
            CHECK(_ocl.queue.enqueueReadBuffer(buffer.handle, CL_FALSE, 0, buffer.elements*sizeof(T), storage.data()));
        }
    }

    /**
     * @brief Set the content of a buffer to the one contained in an std::vector
     * 
     * @param buffer buffer to get data from
     * @param storage vector where the values are copied
     * @param offset number of elements to skip at the start of the buffer
     * @param elements number of elements to copy
     * @param blocking specifies if the copy is synchronous or not. If set to true, will use a queue dedicated to CPU->GPU transfert.
     */
    template <typename T, 
          typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    void getBuffer(const Buffer buffer, std::vector<T>& storage, uint64_t offset, uint64_t elements, const bool blocking = false) {
        storage.resize(buffer.elements);
        if(blocking) {
            CHECK(_ocl.readQueue.enqueueReadBuffer(buffer.handle, CL_TRUE, offset*sizeof(T), elements*sizeof(T), storage.data()));
        } else {
            CHECK(_ocl.queue.enqueueReadBuffer(buffer.handle, CL_FALSE, offset*sizeof(T), elements*sizeof(T), storage.data()));
        }
    }

private:
    /**
     * @brief Allocate a new OpenCL buffer
     * 
     * @param sizeInByte : size of the buffer, in bytes
     * @param kernel : Memory access from the kernel
     * @param host : Memory access from the host
     * @param hostPtr : if specified, fill the buffer with the content pointed by the pointer
     * @return handle to the created buffer
     */
    cl::Buffer allocateBuffer(uint64_t sizeInByte, MEM_ACCESS kernel, MEM_ACCESS host, void* hostPtr = nullptr) {
        cl_int buffer_allocation_error;
        cl_mem_flags kernelFlag, hostFlag, copyFlag;
        switch(kernel) {
            case MEM_ACCESS_WRITE: {kernelFlag = CL_MEM_WRITE_ONLY; break;}
            case MEM_ACCESS_READ: {kernelFlag = CL_MEM_READ_ONLY; break;}
            case MEM_ACCESS_RW: {kernelFlag = CL_MEM_READ_WRITE; break;}
            case MEM_ACCESS_NONE: {kernelFlag = CL_MEM_READ_ONLY; break;}
        }
        switch(host) {
            case MEM_ACCESS_WRITE: {hostFlag = CL_MEM_HOST_WRITE_ONLY; break;}
            case MEM_ACCESS_READ: {hostFlag = CL_MEM_HOST_READ_ONLY; break;}
            case MEM_ACCESS_RW: {hostFlag = 0; break;}
            case MEM_ACCESS_NONE: {hostFlag = CL_MEM_HOST_NO_ACCESS; break;}
        }
        copyFlag = hostPtr?CL_MEM_COPY_HOST_PTR:0;
        cl::Buffer buffer(_ocl.context, kernelFlag|hostFlag|copyFlag, cl::size_type(sizeInByte), hostPtr, &buffer_allocation_error);
        if(buffer_allocation_error != CL_SUCCESS) {
            throw std::bad_alloc();
        }
        //Buffer creation seemes to always return CL_SUCCESS since allocation is lazy, enqueueFillBuffer force the allocation and repport errors correctly
        uint32_t value = hostPtr ? ((uint32_t*)hostPtr)[0] : 0;
        if(_ocl.queue.enqueueFillBuffer(buffer, value, 0, 1*sizeof(uint32_t)) != CL_SUCCESS) {
            throw std::bad_alloc();
        }
        return buffer;
    }
};