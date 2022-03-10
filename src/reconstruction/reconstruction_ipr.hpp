/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

class ReconstructionIPR : public reconstruction::Reconstruction {   
    int64_t width, height;

public:
    ReconstructionIPR(reconstruction::dataset::Parameters *parameters) : reconstruction::Reconstruction(parameters)
    {
        if(requieredGPUMemory() > getOcl().memorySize) {
            throw std::runtime_error("Not enough GPU memory");
        }

        _dataset.initialize();

        width = prm_g.dwidth;
        height = prm_g.dheight;
    }

    ~ReconstructionIPR() {
    }

    uint64_t requieredGPUMemory() {
        int64_t volumeSize = prm_g.vwidth*prm_g.vheight*prm_g.vwidth;
        int64_t imageSize = prm_g.dwidth*prm_g.dheight*prm_g.projections;
        int64_t sumImageSize = prm_g.dwidth*prm_g.dheight*prm_g.concurrent_projections;
        int64_t totalSizeByte = (volumeSize+imageSize+sumImageSize)*sizeof(float);
        std::cout << "Checking GPU memory requierments : " << totalSizeByte/(1024*1024) << "Mo" << std::endl;
        return totalSizeByte;
    }

    void exec() {
        std::cout << "Loading data..." << std::flush;
        std::vector<std::vector<float>> images;
        for(int sit = 0; sit < prm_r.sit; ++sit) {
            images.push_back(_dataset.getImages(sit));
        }
        auto projData = prm_g.projection_matrices;
        std::cout << "Ok" << std::endl;
         
        std::cout << "Allocating requiered buffers..." << std::flush;
        std::vector<reconstruction::gpu::Buffer> imagesBuffer;
        std::vector<reconstruction::gpu::Buffer> projDataBuffer;
        
        for(int sit = 0; sit < prm_r.sit; ++sit) {
            imagesBuffer.push_back(createBuffer(images[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE));
            projDataBuffer.push_back(createBuffer(projData[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE));
        }

        auto volumeBuffer = createBuffer<float>(prm_g.vwidth * prm_g.vwidth * prm_g.vheight, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ, WAVEFRONT_SIZE);
        auto sumImagesBuffer = createBuffer<int32_t>(prm_g.dwidth * prm_g.dheight * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_NONE);
        wait();
        std::cout << "Ok" << std::endl;
        
        std::cout << "Executing reconstruction..." << std::flush;
        setSumImagesBuffer(sumImagesBuffer);
        setVolumeBuffer(volumeBuffer);

        setImageParameters(prm_g.dwidth, prm_g.dheight, prm_g.concurrent_projections);
        setVolumeParameters({0, 0, 0}, {prm_g.vwidth, prm_g.vheight, prm_g.vwidth}, {prm_g.vx, prm_g.vx, prm_g.vx});
        setImageOffset({0, prm_g.dheight});
        setAngleNumber(prm_g.concurrent_projections, prm_md.size());
        
        setBuffer(volumeBuffer, 1.0f);
        setBuffer(sumImagesBuffer, FIXED_FRAC_ONE);

        float weight = prm_r.weight;
        for(int mit = 0; mit < prm_r.it; ++mit) {
            setWeight(weight);
            for(int sit = 0; sit < prm_r.sit; ++sit) {
                setOrigin(prm_g.orig, {prm_g.vx, prm_g.vx, prm_g.vx});
                setImagesBuffer(imagesBuffer[sit]);
                setProjDataBuffer(projDataBuffer[sit]);

                if(sit > 0 || mit > 0) {
                    forward();
                }
                error();
                backward();
                setBuffer(sumImagesBuffer, 0);
            }
            weight *= prm_r.weight_factor;
        }
        wait();
        std::cout << "Ok" << std::endl;

        std::cout << "Saving result..." << std::flush;
        std::vector<float> volume;
        getBuffer<float>(volumeBuffer, volume);
        wait();
        _dataset.saveLayers(volume, 0, prm_g.vheight);
        std::cout << "Ok" << std::endl;
    }

private:
};