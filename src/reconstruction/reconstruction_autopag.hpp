/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

std::string kernel_hforward_source = std::string(
    #include "reconstruction/gpu/kernels/headers/KernelForwardH.hpp"
);
std::string kernel_hbackward_source = std::string(
    #include "reconstruction/gpu/kernels/headers/KernelBackwardH.hpp"
);

class ReconstructionAutoPag : public reconstruction::Reconstruction {
    const uint32_t CPU_THREADS = 12;
    const uint32_t VOLUME_BUFFERS = 4;
    reconstruction::dataset::MvpPerLayer mvpPerLayer;

public:
    ReconstructionAutoPag(reconstruction::dataset::Parameters *parameters) : reconstruction::Reconstruction(parameters), mvpPerLayer(_dataset.getGeometry()->getMvpPerLayer())
    {
        if(requieredGPUMemory() > getOcl().memorySize) {
            throw std::runtime_error("Not enough GPU memory");
        }
        _dataset.initialize(false);  
    }

    uint64_t requieredGPUMemory() {
        int64_t indexSize = mvpPerLayer.maxAngles*CPU_THREADS*sizeof(uint16_t)*2;
        int64_t volumeSize = prm_g.vwidth*prm_g.vwidth*VOLUME_BUFFERS*sizeof(float);
        int64_t mvpSize = prm_g.projection_matrices[0].size()*sizeof(float);
        int64_t imageSize = prm_g.dheight*prm_g.dwidth*(prm_g.concurrent_projections+1)*sizeof(float);
        
        int64_t totalGPUSizeByte = indexSize+mvpSize+volumeSize+imageSize;
        std::cout << "Checking GPU memory requierments : " << totalGPUSizeByte/(1024*1024) << "Mo" << std::endl;
        return totalGPUSizeByte;
    }

    void exec() {
        std::vector<float> sumImages(prm_g.dwidth * prm_g.dheight * (prm_g.concurrent_projections+1));
        
        float weight = prm_r.weight;
        auto projDataBuffer = createBuffer<float>(mvpPerLayer.mvp, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE);
        auto sumImagesBuffer = createBuffer<float>(prm_g.dheight*prm_g.dwidth*(prm_g.concurrent_projections+1), reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_RW);
        
        struct layers{std::mutex m; reconstruction::gpu::Buffer b;};
        std::vector<layers> volumeBuffers(VOLUME_BUFFERS);
        for(int i = 0; i < VOLUME_BUFFERS; ++i) {
            volumeBuffers[i].b = createBuffer<float>(prm_g.vwidth * prm_g.vwidth, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_RW, WAVEFRONT_SIZE);
        }

        omp_set_nested(0);
        #pragma omp parallel num_threads(CPU_THREADS)
        {
            int tid = omp_get_thread_num();
            int vid = tid%VOLUME_BUFFERS;

            #pragma omp single
            std::cout << "[CPU_THREADS] : " << omp_get_num_threads() << std::endl;

            std::vector<float> volume(prm_g.vwidth * prm_g.vwidth);
            
            auto queue = createQueue();

            auto indexBuffer = createBuffer<uint16_t>(mvpPerLayer.maxAngles, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE);
            auto imgIndexBuffer = createBuffer<uint16_t>(mvpPerLayer.maxAngles, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE);
            
            reconstruction::gpu::Kernel backwardh(getOcl(), kernel_hbackward_source, "BackwardH");
            backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_VOLUME_BUFFER, volumeBuffers[vid].b);
            backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_SUMIMAGE_BUFFER, sumImagesBuffer);
            backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_INDEX_BUFFER, indexBuffer);
            backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_IMGINDEX_BUFFER, imgIndexBuffer);
            backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_PROJDATA_BUFFER, projDataBuffer);
            backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_DETWIDTH_U, uint32_t(prm_g.dwidth));
            backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_DETHEIGHT_U, uint32_t(prm_g.dheight));
            backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_VOLWIDTH_U, uint32_t(prm_g.vwidth));
            backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_VOXEL_SIZE_F, prm_g.vx);
            backwardh.setKernelSizeX(prm_g.vwidth * prm_g.vwidth);
            setOrigin(backwardh, reconstruction::gpu::INDEX_HBACKWARD_ORIGIN_F4, prm_g.orig, {prm_g.vx, prm_g.vx, prm_g.vx});

            reconstruction::gpu::Kernel forwardh(getOcl(), kernel_hforward_source, "ForwardH");
            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_VOLUME_BUFFER, volumeBuffers[vid].b);
            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_SUMIMAGE_BUFFER, sumImagesBuffer);
            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_INDEX_BUFFER, indexBuffer);
            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_IMGINDEX_BUFFER, imgIndexBuffer);
            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_PROJDATA_BUFFER, projDataBuffer);
            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_DETWIDTH_U, uint32_t(prm_g.dwidth));
            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_DETHEIGHT_U, uint32_t(prm_g.dheight));
            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_VOLWIDTH_U, uint32_t(prm_g.vwidth));
            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_VOXEL_SIZE_F, prm_g.vx);
            forwardh.setKernelSizeX(prm_g.vwidth * prm_g.vwidth);
            setOrigin(forwardh, reconstruction::gpu::INDEX_HFORWARD_ORIGIN_F4, prm_g.orig, {prm_g.vx, prm_g.vx, prm_g.vx});

            for(int mit = 0; mit < prm_r.it; ++mit) {
                #pragma omp single
                setWeight(weight);
                
                for(int sit = 0; sit < prm_r.sit; ++sit) {
                    #pragma omp single
                    {
                        setBuffer(queue, sumImagesBuffer, 0);
                        wait(queue);
                        std::cout << "Executing reconstruction..." << (100*(mit*prm_r.sit+sit))/(prm_r.it*prm_r.sit) << "%" << "\r" << std::flush;
                    }

                    //Forward
                    if(sit > 0 || mit > 0) {
                        #pragma omp for schedule(dynamic)
                        for(int l = 0; l < prm_g.vheight; ++l) {
                            volume = _dataset.getLayer(l);

                            setBuffer(queue, indexBuffer, mvpPerLayer.mvp_indexes[sit][l]); 
                            setBuffer(queue, imgIndexBuffer, mvpPerLayer.image_indexes[sit][l]); 
                            
                            volumeBuffers[vid].m.lock();

                            setBuffer(queue, volumeBuffers[vid].b, volume, true);
                            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_ANGLES_U, uint32_t(mvpPerLayer.mvp_indexes[sit][l].size()));
                            forwardh.setKernelArgument(reconstruction::gpu::INDEX_HFORWARD_YOFFSET_U, uint32_t(l));
                            forwardh.executeKernel(queue);
                            
                            queue.flush();
                            //Equivalent to finish, but not busy
                            setBuffer(queue, indexBuffer, std::vector<uint16_t>(2), true);

                            volumeBuffers[vid].m.unlock();   
                        }
                    }
                    wait(queue);

                    //

                    //Error
                    if(sit > 0 || mit > 0) {
                        #pragma omp for schedule(dynamic)
                        for(int j = sit; j < prm_g.concurrent_projections; j += prm_r.sit) {
                            std::vector<float> image = _dataset.getImage(j);
                            int id = (j-sit)/prm_r.sit;

                            float* dst = sumImages.data()+(id*prm_g.dwidth*prm_g.dheight);
                            uint32_t* src = (uint32_t*)sumImages.data()+(id*prm_g.dwidth*prm_g.dheight);
                            float* ref = image.data();

                            int64_t imageSize = int64_t(image.size());
                            for(int64_t i = 0; i < imageSize; ++i) {
                                float value = FIXED_TO_FLOAT(src[i]);
                                dst[i] = std::log(std::max(ref[i]/std::max(value,EPSILON), EPSILON))*weight;
                            }
                        }
                    } else {
                        #pragma omp for schedule(dynamic)
                        for(int j = sit; j < prm_g.concurrent_projections; j += prm_r.sit) {
                            std::vector<float> image = _dataset.getImage(j);
                            int id = (j-sit)/prm_r.sit;

                            float* dst = sumImages.data()+(id*prm_g.dwidth*prm_g.dheight);
                            float* ref = image.data();

                            int64_t imageSize = int64_t(image.size());
                            for(int64_t i = 0; i < imageSize; ++i) {
                                dst[i] = std::log(std::max(ref[i], EPSILON));
                            }
                        }
                    }

                    #pragma omp single
                    setBuffer(queue, sumImagesBuffer, sumImages, true);
                    
                    //Backward
                    #pragma omp for schedule(dynamic)
                    for(int l = 0; l < prm_g.vheight; ++l) {                        
                        if(sit > 0 || mit > 0) {
                            volume = _dataset.getLayer(l);
                            volumeBuffers[vid].m.lock();
                            setBuffer(queue, volumeBuffers[vid].b, volume);
                        } else {
                            volumeBuffers[vid].m.lock();
                            setBuffer(queue, volumeBuffers[vid].b, 1.0f/prm_g.concurrent_projections);
                        }

                        setBuffer(queue, indexBuffer, mvpPerLayer.mvp_indexes[sit][l]); 
                        setBuffer(queue, imgIndexBuffer, mvpPerLayer.image_indexes[sit][l]); 
                        backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_ANGLES_U, uint32_t(mvpPerLayer.mvp_indexes[sit][l].size()));
                        backwardh.setKernelArgument(reconstruction::gpu::INDEX_HBACKWARD_YOFFSET_U, uint32_t(l));
                        backwardh.executeKernel(queue);
                        queue.flush();

                        getBuffer(queue, volumeBuffers[vid].b, volume, true);
                        volumeBuffers[vid].m.unlock();

                        _dataset.saveLayer(volume, l, (sit==prm_r.sit-1) && (mit == prm_r.it-1));
                    }
                }

                #pragma omp single
                weight *= prm_r.weight_factor;
            }
        }
        wait();
        std::cout << "Executing reconstruction..." << " Done." << std::endl;
    }
    
private:
    void setOrigin(reconstruction::gpu::Kernel &kernel, int64_t param_index, glm::vec3 origin, glm::vec3 randomizedOffset = {0, 0, 0}) {
        for(int i = 0; i < 3; ++i) origin[i] += std::uniform_real_distribution<float>(0, randomizedOffset[i])(_randomEngine);
        kernel.setKernelArgument(param_index, cl_float4{origin.x, origin.y, origin.z, 1.0f});
    }
};