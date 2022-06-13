/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

#define ASYNC_THREAD 12

class ReconstructionLPLA : public reconstruction::Reconstruction {
public:
    ReconstructionLPLA(reconstruction::dataset::Parameters *parameters) : reconstruction::Reconstruction(parameters)
    {
        if(requieredGPUMemory() > getOcl().memorySize) {
            throw std::runtime_error("Not enough GPU memory");
        }
        _dataset.initialize(false);
    }

    uint64_t requieredGPUMemory() {
        int64_t volumeSize = prm_g.vwidth*prm_g.vwidth*ASYNC_THREAD;
        int64_t imageSize = prm_g.dwidth*prm_g.dheight*prm_g.concurrent_projections;
        int64_t sumImageSize = prm_g.dwidth*prm_g.dheight*prm_g.concurrent_projections;
        int64_t totalSizeByte = (volumeSize+imageSize+sumImageSize)*sizeof(float);
        std::cout << "Checking GPU memory requierments : " << totalSizeByte/(1024*1024) << "Mo" << std::endl;
        return totalSizeByte;
    }

    void exec() {
        std::cout << "Allocating requiered buffers..." << std::flush;
        auto projData = prm_g.projection_matrices;

        auto imagesBuffer = createBuffer<float>(prm_g.dwidth * prm_g.dheight * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE);
        auto sumImagesBuffer = createBuffer<float>(prm_g.dwidth * prm_g.dheight * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_NONE);
        
        std::vector<reconstruction::gpu::Buffer> projDataBuffer, volumeBuffer;
        for(int sit = 0; sit < prm_r.sit; ++sit) {
            projDataBuffer.push_back(createBuffer(projData[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE));
        }
        for(int i = 0; i < ASYNC_THREAD; ++i) {
            volumeBuffer.push_back(createBuffer<float>(prm_g.vwidth * prm_g.vwidth, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_RW, WAVEFRONT_SIZE));
            setBuffer(volumeBuffer[i], 1.0f);
        }

        wait();
        std::cout << "Ok" << std::endl;

        setSumImagesBuffer(sumImagesBuffer);
        setImagesBuffer(imagesBuffer);
        setImageParameters(prm_g.dwidth, prm_g.dheight, prm_g.concurrent_projections);
        setImageOffset({0, prm_g.dheight});
        setAngleNumber(prm_g.concurrent_projections, prm_d.module_number);

        setBuffer(sumImagesBuffer, FIXED_FRAC_ONE);
        
        auto asyncImageLoad = std::async(std::launch::async, [this]{
            return this->_dataset.getSitImages(0);
        });
        
        float weight = prm_r.weight;
        #pragma omp parallel num_threads(ASYNC_THREAD)
        {
            for(int mit = 0; mit < prm_r.it; ++mit) {

                #pragma omp single
                setWeight(weight);

                for(int sit = 0; sit < prm_r.sit; ++sit) {

                    #pragma omp single
                    {
                        setProjDataBuffer(projDataBuffer[sit]);
                        setOrigin(prm_g.orig, {prm_g.vx, prm_g.vx, prm_g.vx});
                        auto images = asyncImageLoad.get();
                        if(mit < prm_r.it-1 || sit < prm_r.sit-1) {
                            asyncImageLoad = std::async(std::launch::async, [this, sit]{ 
                                return this->_dataset.getSitImages((sit+1)%prm_r.sit);
                            });
                        }
                        setBuffer(imagesBuffer, images);
                        std::cout << "Executing reconstruction..." << (100*(mit*prm_r.sit+sit))/(prm_r.it*prm_r.sit) << "%" << "\r" << std::flush;
                    }
                    #pragma omp barrier

                    if(sit > 0 || mit > 0) {
                        #pragma omp for schedule(dynamic)
                        for(int l = 0; l < prm_g.vheight; ++l) {
                            const int tid = omp_get_thread_num();
                            std::vector<float> layer = _dataset.getLayer(l);

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
                    error();
                    #pragma omp single
                    wait();
                    #pragma omp barrier

                    #pragma omp for schedule(dynamic)
                    for(int l = 0; l < prm_g.vheight; ++l) {
                        const int tid = omp_get_thread_num();
                        std::vector<float> layer;

                        if(sit > 0 || mit > 0) {
                            layer = _dataset.getLayer(l);
                            #pragma omp critical (write)
                            {
                                setBuffer(volumeBuffer[tid], layer, true);
                            }
                        } else {
                            #pragma omp critical (compute)
                            {
                                setBuffer(volumeBuffer[tid], 1.0f);
                                wait();
                            }
                        }

                        #pragma omp critical (compute)
                        {
                            setVolumeBuffer(volumeBuffer[tid]);
                            setVolumeParameters({0, l, 0}, {prm_g.vwidth, l+1, prm_g.vwidth}, {prm_g.vx, prm_g.vx, prm_g.vx});
                            backward();
                            wait();
                        }

                        #pragma omp critical (read)
                        {
                            getBuffer(volumeBuffer[tid], layer, true);
                        }

                        _dataset.saveLayer(layer, l, ((sit==prm_r.sit-1) && (mit == prm_r.it-1)));
                    }

                    #pragma omp single
                    setBuffer(sumImagesBuffer, 0);
                }

                #pragma omp single
                weight *= prm_r.weight_factor;
            }
        }
        wait();
        std::cout << "Executing reconstruction..." << " Done." << std::endl;
    }
    
private:
    
};