/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

namespace reconstruction {
    #include "reconstruction/gpu.hpp"
    #include "reconstruction/dataset.hpp"
    
    using gpu::Buffer;
    using dataset::Dataset;

    std::string kernel_backward_source = std::string(
        #include "reconstruction/gpu/kernels/headers/KernelBackward.hpp"
    );
    std::string kernel_forward_source = std::string(
        #include "reconstruction/gpu/kernels/headers/KernelForward.hpp"
    );
    std::string kernel_error_source = std::string(
        #include "reconstruction/gpu/kernels/headers/KernelError.hpp"
    );
    
    /**
     * @brief class representing a reconstruction algorithm.
     * All recon inherit from this class, that initialize the GPU kernels, the dataset, 
     *  and provide the necessary methods to configure the projections operators
     */
    class Reconstruction : protected gpu::Context {
        std::random_device _randomDevice;
        gpu::Kernel _kforward, _kbackward, _kerror;
    protected:
        dataset::Parameters *_parameters;
        Dataset _dataset;
        std::mt19937 _randomEngine = std::mt19937(_randomDevice());
        
    public:
        Reconstruction(dataset::Parameters *parameters) : gpu::Context(), 
                            _kforward(getOcl(), kernel_forward_source, "Forward"), 
                            _kbackward(getOcl(), kernel_backward_source, "Backward"), 
                            _kerror(getOcl(), kernel_error_source, "Error"),
                            _parameters(parameters),
                            _dataset(parameters)
        {
            
        }

        /**
         * @brief Return the amount of GPU memory requiered by the algorithm, in bytes.
         * 
         * @return uint64_t number of bytes
         */
        virtual uint64_t requieredGPUMemory() = 0;

        /**
         * @brief Main function of the reconstruction algorithm, where the forward, error and backward are called.
         * 
         */
        virtual void exec() = 0;

        /**
         * @brief Forward project the current volume and export the projections.
         * 
         */
        void forward_exec() {
            int32_t ASYNC_THREAD = 12;
            std::cout << "Allocating requiered buffers..." << std::flush;

            auto projData = prm_g.projection_matrices;
            auto sumImagesBuffer = createBuffer<float>(prm_g.dwidth * prm_g.dheight * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ);
            
            std::vector<reconstruction::gpu::Buffer> projDataBuffer, volumeBuffer;
            for(int sit = 0; sit < prm_r.sit; ++sit) {
                projDataBuffer.push_back(createBuffer(projData[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE));
            }
            for(int i = 0; i < ASYNC_THREAD; ++i) {
                volumeBuffer.push_back(createBuffer<float>(prm_g.vwidth * prm_g.vwidth, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE, WAVEFRONT_SIZE));
                setBuffer(volumeBuffer[i], 0.0f);
            }

            wait();
            std::cout << "Ok" << std::endl;
            
            setSumImagesBuffer(sumImagesBuffer);
            setImageParameters(prm_g.dwidth, prm_g.dheight, prm_g.concurrent_projections);
            setImageOffset({0, prm_g.dheight});
            setAngleNumber(prm_g.concurrent_projections, prm_d.module_number);
            
            float weight = prm_r.weight;
            int samples = 32;
            #pragma omp parallel num_threads(ASYNC_THREAD)
            {
                for(int sit = 0; sit < prm_r.sit; ++sit) {
                    setBuffer(sumImagesBuffer, FIXED_FRAC_ZERO);

                    #pragma omp single
                    {
                        setProjDataBuffer(projDataBuffer[sit]);
                    }

                    for(int mit = 0; mit < samples; ++mit) {

                        #pragma omp single
                        {
                            setOrigin(prm_g.orig, {prm_g.vx, prm_g.vx, prm_g.vx});
                            std::cout << "Executing forward projection..." << (100*(mit*prm_r.sit+sit))/(samples*prm_r.sit) << "%" << "\r" << std::flush;
                        }
                        #pragma omp barrier

                        #pragma omp for schedule(dynamic)
                        for(int l = 0; l < prm_g.vheight; ++l) {
                            const int tid = omp_get_thread_num();
                            std::vector<float> layer = _dataset.getLayers(l);
                            std::for_each(layer.begin(), layer.end(), [samples](float& v) { v /= samples;});
                            for(int j = 0; j < prm_g.vwidth; ++j) {
                                for(int i = 0; i < prm_g.vwidth; ++i) {
                                    if( std::sqrtf(std::pow(i-prm_g.vwidth*0.5f, 2.0f)+std::pow(j-prm_g.vwidth*0.5f, 2.0f)) >= prm_g.vwidth*0.5f - 1.0f ) {
                                        layer[j*prm_g.vwidth+i] = 0.0f;
                                    }
                                }
                            }

                            #pragma omp critical (write)
                            {
                                setBuffer(volumeBuffer[tid], layer, true);
                            }

                            #pragma omp critical (compute)
                            {
                                setVolumeBuffer(volumeBuffer[tid]);
                                setVolumeParameters({0, l, 0}, {prm_g.vwidth, l+1, prm_g.vwidth}, {prm_g.vx, prm_g.vx, prm_g.vx});
                                forward();
                                wait();
                            }
                        }
                    }

                    #pragma omp single
                    {
                        //Save projections
                        std::vector<fixed32> projections;
                        getBuffer(sumImagesBuffer, projections, true);
                        std::vector<float> processed_projections(projections.size());
                        for(int i = 0; i < projections.size(); ++i) {
                            processed_projections[i] = FIXED_TO_FLOAT(projections[i]);
                        }
                        _dataset.saveImages(processed_projections, sit, prm_g.projections, prm_r.sit);
                    }
                }
            }
            wait();
            std::cout << "Executing forward projection..." << " Done." << std::endl;
        }

    protected:
        /**
         * @brief Set the Buffer that will contains the images (should be initialized before the reconstruction).
         * 
         * @param buffer Buffer containing the projection images, as float
         */
        void setImagesBuffer(Buffer &buffer) {
            _kerror.setKernelArgument(gpu::INDEX_ERROR_IMAGE_BUFFER, buffer);
        }

        /**
         * @brief Set the Buffer that will store the projection of the volume (should be initialized before the reconstruction).
         * 
         * @param buffer Buffer of float. Must have the same size as the image buffer
         */
        void setSumImagesBuffer(Buffer &buffer) {
            _kforward.setKernelArgument(gpu::INDEX_FORWARD_SUM_IMAGE_BUFFER, buffer);
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_SUM_IMAGE_BUFFER, buffer);
            _kerror.setKernelArgument(gpu::INDEX_ERROR_SUM_IMAGE_BUFFER, buffer);
        }

        /**
         * @brief Set the Buffer that will be used to store the reconstructed volume (should be initialized before the reconstruction).
         * 
         * @param buffer Buffer that will contains the reconstructed volume, as float.
         */
        void setVolumeBuffer(Buffer &buffer) {
            _kforward.setKernelArgument(gpu::INDEX_FORWARD_VOLUME_BUFFER, buffer);
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_VOLUME_BUFFER, buffer);   
        }

        /**
         * @brief Set the projection matrix buffer (should be initialized before the reconstruction).
         * 
         * @param buffer buffer containing the MVP matrices as float16
         */
        void setProjDataBuffer(Buffer &buffer) {
            _kforward.setKernelArgument(gpu::INDEX_FORWARD_PROJDATA_BUFFER, buffer);
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_PROJDATA_BUFFER, buffer);   
        }

        /**
         * @brief Set the parameters of the images used in the reconstrution
         * 
         * @param width width of the images, in pixel
         * @param height height of the images, in pixel
         * @param number number of images
         */
        void setImageParameters(int64_t width, int64_t height, int64_t number) {
            _kforward.setKernelArgument(gpu::INDEX_FORWARD_SIZE_U4, cl_uint4{uint32_t(width), uint32_t(height), uint32_t(width), 1u});
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_SIZE_U4, cl_uint4{uint32_t(width), uint32_t(height), uint32_t(width), 1u});
            _kerror.setKernelArgument(gpu::INDEX_ERROR_SIZE_U4, cl_uint4{uint32_t(width), uint32_t(height), uint32_t(width), 1u});
            
            _kerror.setKernelSizeX(width*height*number);
            _kerror.setKernelArgument(gpu::INDEX_ERROR_ELEMENTS_U, uint32_t(width*height*number));
        }

        /**
         * @brief Set the parameters of the volume used in the reconstruction
         * 
         * @param min the minimal coordinate (in voxel) of the volume
         * @param max the maximal coordinate (in voxel) of the volume
         * @param voxelSize the size (in mm) of the volume voxels
         */
        void setVolumeParameters(glm::ivec3 min, glm::ivec3 max, glm::vec3 voxelSize) {
            _kforward.setKernelArgument(gpu::INDEX_FORWARD_VOLUME_MIN_I3, cl_int3{min.x, min.y, min.z});
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_VOLUME_MIN_I3, cl_int3{min.x, min.y, min.z});

            _kforward.setKernelArgument(gpu::INDEX_FORWARD_VOLUME_MAX_I3, cl_int3{max.x, max.y, max.z});
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_VOLUME_MAX_I3, cl_int3{max.x, max.y, max.z});

            glm::ivec3 size = max - min;
            _kforward.setKernelSizeX(size.x*size.y*size.z);
            _kbackward.setKernelSizeX(size.x*size.y*size.z);

            _kforward.setKernelArgument(gpu::INDEX_FORWARD_VOXEL_SIZE_F4, cl_float4{voxelSize.x, voxelSize.y, voxelSize.z, 1.0f});
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_VOXEL_SIZE_F4, cl_float4{voxelSize.x, voxelSize.y, voxelSize.z, 1.0f});
        }

        /**
         * @brief Configure the number of angles and modules per angle for the ProjData buffer
         * 
         * @param number the number of angles
         * @param modulePerAngle the number of module per angle
         */
        void setAngleNumber(int64_t number, int64_t modulePerAngle) {
            _kforward.setKernelArgument(gpu::INDEX_FORWARD_ANGLE_NUMBER_U, cl_uint(number));
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_ANGLE_NUMBER_U, cl_uint(number));

            _kforward.setKernelArgument(gpu::INDEX_FORWARD_MODULE_PER_ANGLE_U, cl_uint(modulePerAngle));
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_MODULE_PER_ANGLE_U, cl_uint(modulePerAngle));
        }

        /**
         * @brief Set the weight of the reconstruction iteration
         * 
         * @param weight
         */
        void setWeight(float weight) {
            _kerror.setKernelArgument(gpu::INDEX_ERROR_WEIGHT_F, weight);
        }
        
        /**
         * @brief Set the position of the volume and the offset of the first layer and image line
         * 
         * @param origin position of the origin of the volume (mm)
         * @param randomizedOffset if non zero, add a random offset to the volume between 0 and randomizedOffset for each axis
         */
        void setOrigin(glm::vec3 origin, glm::vec3 randomizedOffset = {0, 0, 0}) {
            for(int i = 0; i < 3; ++i) origin[i] += std::uniform_real_distribution<float>(0, randomizedOffset[i])(_randomEngine);
            _kforward.setKernelArgument(gpu::INDEX_FORWARD_ORIGIN_F4, cl_float4{origin.x, origin.y, origin.z, 0.0f});
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_ORIGIN_F4, cl_float4{origin.x, origin.y, origin.z, 0.0f});
        }

        /**
         * @brief Set the vertical offset of the image in pixel 
         * (ie. if the images are the lower half of the full images, the offset if half the height of the detector)
         */
        void setImageOffset(glm::ivec2 imageOffsetY) {
            _kforward.setKernelArgument(gpu::INDEX_FORWARD_LINE_OFFSET_I2, imageOffsetY);
            _kbackward.setKernelArgument(gpu::INDEX_BACKWARD_LINE_OFFSET_I2, imageOffsetY);
        }

        /**
         * @brief Set the number of element in the image buffer
         */
        void setImageElements(int32_t elements) {
            _kerror.setKernelSizeX(elements);
        }

        /**
         * @brief Execute the forward projection
         * 
         */
        void forward() {
            _kforward.executeKernel();
        }

        /**
         * @brief Compute the error
         * 
         */
        void error() {
            _kerror.executeKernel();
        }

        /**
         * @brief Execute the backward projection
         * 
         */
        void backward() {
            _kbackward.executeKernel();
        }

    private:

    };

    #include "reconstruction/reconstruction_n.hpp"
    #include "reconstruction/reconstruction_lpln.hpp"
    #include "reconstruction/reconstruction_lpla.hpp"
    #include "reconstruction/reconstruction_lplh.hpp"
    #include "reconstruction/reconstruction_chunk.hpp"
    #include "reconstruction/reconstruction_hartmann.hpp"
    #include "reconstruction/reconstruction_ipr.hpp"
}