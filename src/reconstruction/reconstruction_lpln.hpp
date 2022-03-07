/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

class ReconstructionLPLN : public reconstruction::Reconstruction {
public:
    ReconstructionLPLN(reconstruction::dataset::Parameters *parameters) : reconstruction::Reconstruction(parameters)
    {
        if(requieredGPUMemory() > getOcl().memorySize) {
            throw std::runtime_error("Not enough GPU memory");
        }
        _dataset.initialize();
    }

    uint64_t requieredGPUMemory() {
        int64_t volumeSize = prm_g.vwidth*prm_g.vwidth;
        int64_t imageSize = prm_g.dwidth*prm_g.dheight*prm_g.concurrent_projections;
        int64_t sumImageSize = prm_g.dwidth*prm_g.dheight*prm_g.concurrent_projections;
        int64_t totalSizeByte = (volumeSize+imageSize+sumImageSize)*sizeof(float);
        std::cout << "Checking GPU memory requierments : " << totalSizeByte/(1024*1024) << "Mo" << std::endl;
        return totalSizeByte;
    }

    void exec() {
        std::cout << "Allocating requiered buffers..." << std::flush;
        auto projData = prm_g.projection_matrices;

        auto volumeBuffer = createBuffer<float>(prm_g.vwidth * prm_g.vwidth, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_RW, WAVEFRONT_SIZE);
        auto imagesBuffer = createBuffer<float>(prm_g.dwidth * prm_g.dheight * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE);
        auto sumImagesBuffer = createBuffer<float>(prm_g.dwidth * prm_g.dheight * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_NONE);
        
        std::vector<reconstruction::gpu::Buffer> projDataBuffer;
        for(int sit = 0; sit < prm_r.sit; ++sit) {
            projDataBuffer.push_back(createBuffer(projData[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE));
        }

        wait();
        std::cout << "Ok" << std::endl;

        std::cout << "Executing reconstruction..." << std::flush;
        setSumImagesBuffer(sumImagesBuffer);
        setImagesBuffer(imagesBuffer);
        setVolumeBuffer(volumeBuffer);
        setImageParameters(prm_g.dwidth, prm_g.dheight, prm_g.concurrent_projections);
        setImageOffset({0, prm_g.dheight});
        setAngleNumber(prm_g.concurrent_projections, prm_md.size());

        setBuffer(volumeBuffer, 1.0f);
        setBuffer(sumImagesBuffer, FIXED_FRAC_ONE);
        
        float weight = prm_r.weight;
        for(int mit = 0; mit < prm_r.it; ++mit) {
            setWeight(weight);
            for(int sit = 0; sit < prm_r.sit; ++sit) {
                setProjDataBuffer(projDataBuffer[sit]);
                setOrigin(prm_g.orig, {prm_g.vx, prm_g.vx, prm_g.vx});
                auto images = _dataset.getImages(sit);
                setBuffer(imagesBuffer, images);
                if(sit > 0 || mit > 0) {
                    for(int l = 0; l < prm_g.vheight; ++l) {
                        std::vector<float> layer = _dataset.getLayers(l);
                        setBuffer(volumeBuffer, layer);
                        setVolumeParameters({0, l, 0}, {prm_g.vwidth, l+1, prm_g.vwidth}, {prm_g.vx, prm_g.vx, prm_g.vx});
                        forward();
                    }
                }

                error();

                for(int l = 0; l < prm_g.vheight; ++l) {
                    std::vector<float> layer;
                    if(sit > 0 || mit > 0) {
                        layer = _dataset.getLayers(l);
                        setBuffer(volumeBuffer, layer);
                    } else {
                        setBuffer(volumeBuffer, 1.0f);
                    }
                    setVolumeParameters({0, l, 0}, {prm_g.vwidth, l+1, prm_g.vwidth}, {prm_g.vx, prm_g.vx, prm_g.vx});
                    backward();
                    wait();
                    getBuffer<float>(volumeBuffer, layer);
                    wait();
                    _dataset.saveLayer(layer, l);
                }

                setBuffer(sumImagesBuffer, 0);
            }
            weight *= prm_r.weight_factor;
        }
        
        wait();
        std::cout << "Ok" << std::endl;
    }

private:
    
};