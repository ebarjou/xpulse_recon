/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

enum MEM_ACCESS {MEM_ACCESS_READ, MEM_ACCESS_WRITE, MEM_ACCESS_RW, MEM_ACCESS_NONE};
enum VENDOR {VENDOR_NVIDIA, VENDOR_AMD, VENDOR_INTEL, VENDOR_OTHER};

struct Ocl {
    cl::Device device;
    cl::Context context;
    uint64_t memorySize = 0, maxAllocSize = 0;
    reconstruction::gpu::VENDOR vendor;
};

struct Buffer {
    cl::Buffer handle;
    uint64_t elements;
    uint8_t bytePerSample;
    reconstruction::gpu::MEM_ACCESS kernelAccess, hostAccess;
};

/**
 * @brief Wrapper class the the OpenGL context
 * 
 */
class Context {
    reconstruction::gpu::Ocl _ocl;

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
                std::cout << "\t\t" << device.get() << " - " << deviceName << " " << memorySize/(1024*1024) << "Mb" << std::endl;
                std::for_each(deviceName.begin(), deviceName.end(), [](char & c) { c = std::tolower(c); });
                if( (platformName.find("NVIDIA") != std::string::npos || platformName.find("AMD") != std::string::npos) 
                        && memorySize > _ocl.memorySize)
                {
                    _ocl.device = device;
                    _ocl.memorySize = memorySize;
                    _ocl.maxAllocSize = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
                    if(platformName.find("NVIDIA") != std::string::npos) {
                        _ocl.vendor = reconstruction::gpu::VENDOR_NVIDIA;
                    } else if(platformName.find("AMD") != std::string::npos) {
                        _ocl.vendor = reconstruction::gpu::VENDOR_AMD;
                    } else if(platformName.find("INTEL") != std::string::npos) {
                        _ocl.vendor = reconstruction::gpu::VENDOR_INTEL;
                    } else  {
                        _ocl.vendor = reconstruction::gpu::VENDOR_OTHER;
                    }
                }
            }
        }
        if(_ocl.memorySize == 0) {
            throw std::runtime_error("No compatible device found");
        }
        _ocl.memorySize -= 512*1024*1024; //512Mb of margin, to be improved

        std::cout << "Creating OpenCL context..." << std::flush;
        cl_int error_context_creation;
        _ocl.context = cl::Context{_ocl.device, false, nullptr, nullptr, &error_context_creation};
        CHECK(error_context_creation)
        std::cout << "Ok." << std::endl;
        
        std::cout << "Selected " << _ocl.device.getInfo<CL_DEVICE_NAME>() << ": " << _ocl.memorySize/(1024.0*1024.0) << "Mb available, " << _ocl.maxAllocSize/(1024.0*1024.0) << "Mb max per alloc." << std::endl;
    }

    reconstruction::gpu::Ocl *getOcl() {
        return &_ocl;
    }

    cl::CommandQueue createQueue() {
        cl_int error;
        cl_command_queue_properties props = 0;
        auto queue = cl::CommandQueue(_ocl.context, _ocl.device, props, &error);
        CHECK(error)
        return queue;
    }

    /**
     * @brief Flush and finish all pending OpenCL call
     * 
     */
    void wait(cl::CommandQueue *queue) {
        CHECK(queue->flush());
        CHECK(queue->finish());
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
    Buffer createBuffer(uint64_t elements, const MEM_ACCESS kernel = MEM_ACCESS_RW, const MEM_ACCESS host = MEM_ACCESS_RW) {
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
    template <class S>
    Buffer createBuffer(S &storage, const MEM_ACCESS kernel = MEM_ACCESS_RW, const MEM_ACCESS host = MEM_ACCESS_RW) {
        using T = typename S::value_type;
        Buffer buffer;
        buffer.handle = allocateBuffer(storage.size()*sizeof(T), kernel, host, (void*)&storage[0]);
        buffer.elements = storage.size();
        buffer.bytePerSample = sizeof(T);
        buffer.kernelAccess = kernel;
        buffer.hostAccess = host;
        return buffer;
    }

    template <class T>
    T* mapBuffer( const cl::CommandQueue *queue, 
                    const Buffer &buffer, const int64_t buffer_offset, 
                    const int64_t elements, cl_map_flags flags,
                    const bool blocking = false, cl::Event *event = nullptr)
    {
        cl_int error;
        T* host_ptr = (T*)queue->enqueueMapBuffer(buffer.handle, blocking, flags, buffer_offset*sizeof(T), elements*sizeof(T), nullptr, event, &error);
        CHECK(error);
        return host_ptr;
    }

    template <class T>
    void unmapBuffer(  const cl::CommandQueue *queue, 
                    const Buffer &buffer, T* host_ptr,
                    cl::Event *event = nullptr)
    {
        CHECK(queue->enqueueUnmapMemObject(buffer.handle, (void*)host_ptr, nullptr, event));
    }

    template <class S>
    void getBuffer( const cl::CommandQueue *queue, 
                    const Buffer &src, const int64_t src_offset, 
                    S &dst, const int64_t dst_offset,
                    const int64_t elements,
                    const bool blocking = false, cl::Event *event = nullptr)
    {
        using T = typename S::value_type;
        CHECK(queue->enqueueReadBuffer(src.handle, blocking, src_offset*sizeof(T), elements*sizeof(T), &dst[dst_offset], nullptr, event));
    }

    template <class S>
    void setBuffer( const cl::CommandQueue *queue, 
                    Buffer &dst, const uint64_t dst_offset, 
                    const S &src, const uint64_t src_offset, 
                    const int64_t elements, 
                    const bool blocking = false, cl::Event *event = nullptr)
    {
        using T = typename S::value_type;
        CHECK(queue->enqueueWriteBuffer(dst.handle, blocking, dst_offset*sizeof(T), elements*sizeof(T), &src[src_offset], nullptr, event));
    }

    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
    void setBuffer( const cl::CommandQueue *queue, 
                    const Buffer buffer, const T value,
                    cl::Event *event = nullptr)
    {
        CHECK(queue->enqueueFillBuffer(buffer.handle, value, 0, buffer.elements*sizeof(T), nullptr, event));
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
        return buffer;
    }
};