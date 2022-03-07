/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

class ReconstructionChunk : public reconstruction::Reconstruction {
    static const uint32_t SUB_CHUNK = 4;
    
    reconstruction::dataset::GeometryChunk _geometryChunk;
public:
    ReconstructionChunk(reconstruction::dataset::Parameters *parameters) : reconstruction::Reconstruction(parameters),
        _geometryChunk(parameters, _dataset.getGeometry(), uint64_t(getOcl().memorySize*0.98f), getOcl().maxAllocSize)
    {
        std::cout << "Chunks : ";
        for(auto chunk : prm_m2.chunks) {
            std::cout << " [" << chunk.vOffset << "-" << chunk.vOffset+chunk.vSize << "],";
        }
        std::cout << std::endl;
        if(requieredGPUMemory() > getOcl().memorySize) {
            throw std::runtime_error("Not enough GPU memory");
        }
        _dataset.initialize();
    }

    uint64_t requieredGPUMemory() {
        uint64_t totalSizeByte = 0;
        for(auto chunk : prm_m2.chunks) {
            totalSizeByte = std::max(totalSizeByte, _geometryChunk.sizeInMemory(chunk));
        }
        std::cout << "Checking GPU memory requierments : " << totalSizeByte/(1024*1024) << "Mo" << std::endl;
        return totalSizeByte;
    }

    void exec() {
        std::future<void> asyncImageWrite;
        auto asyncImageLoad = std::async(std::launch::async, [this]{
            return this->_dataset.getImagesCropped(0);
        });

        auto projData = prm_g.projection_matrices;
        std::vector<reconstruction::gpu::Buffer> projDataBuffer;
        for(int sit = 0; sit < prm_r.sit; ++sit) {
            projDataBuffer.push_back(createBuffer(projData[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE));
        }
        
        setImageParameters(prm_g.dwidth, prm_g.dheight, prm_g.concurrent_projections);
        setAngleNumber(prm_g.concurrent_projections, prm_md.size());

        std::cout << "Processing chunks :" << std::endl;
        for(int c = 0; c < prm_m2.chunks.size(); ++c) {
            wait();
            auto chunk = prm_m2.chunks[c];
            std::cout << "[" << chunk.vOffset << "-" << chunk.vOffset+chunk.vSize << "]... " << std::flush;

            std::vector<reconstruction::gpu::Buffer> volumeBuffer(6);
            std::vector<glm::ivec2> volumeBufferOffset(6);
            
            volumeBuffer[0] = createBuffer<float>(prm_g.vwidth * prm_g.vwidth * chunk.vPadTop, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ, WAVEFRONT_SIZE);
            volumeBufferOffset[0] = {-chunk.vPadTop, chunk.vPadTop};
            for(int l = 1; l < SUB_CHUNK+1; ++l) {
                uint32_t size = uint32_t(l < SUB_CHUNK ? chunk.vSize/SUB_CHUNK : chunk.vSize - (SUB_CHUNK-1)*(chunk.vSize/SUB_CHUNK));
                volumeBuffer[l] = createBuffer<float>(prm_g.vwidth * prm_g.vwidth * size, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ, WAVEFRONT_SIZE);
                volumeBufferOffset[l] = {(l-1)*(chunk.vSize/4), size};
            }
            volumeBuffer[SUB_CHUNK+1] = createBuffer<float>(prm_g.vwidth * prm_g.vwidth * chunk.vPadBot, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ, WAVEFRONT_SIZE);
            volumeBufferOffset[SUB_CHUNK+1] = {chunk.vSize, chunk.vPadBot};

            for(int l = 0; l < SUB_CHUNK+2; ++l) {
                setBuffer(volumeBuffer[l], 1.0f);
            }

            { //Region where images and images buffers are kept in scope
                auto imagesBuffer = createBuffer<float>(prm_g.dwidth * chunk.iSize * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE);
                auto sumImagesBuffer = createBuffer<float>(prm_g.dwidth * chunk.iSize * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_NONE);

                setBuffer(sumImagesBuffer, FIXED_FRAC_ONE);

                setImagesBuffer(imagesBuffer);
                setSumImagesBuffer(sumImagesBuffer);
                setImageOffset({chunk.iOffset, chunk.iSize});
                setImageElements(uint32_t(prm_g.dwidth * chunk.iSize * prm_g.concurrent_projections));

                auto images = asyncImageLoad.get();
                if(c < prm_m2.chunks.size()-1) {
                    asyncImageLoad = std::async(std::launch::async, [this, c]{
                        return this->_dataset.getImagesCropped(c+1);
                    });
                }

                float weight = prm_r.weight;
                for(int mit = 0; mit < prm_r.it; ++mit) {
                    setWeight(weight);
                    for(int sit = 0; sit < prm_r.sit; ++sit) {
                        setBuffer(imagesBuffer, images[sit]);
                        setOrigin(prm_g.orig, {prm_g.vx, prm_g.vx, prm_g.vx});
                        setProjDataBuffer(projDataBuffer[sit]);
                        if(sit > 0 || mit > 0) {
                            for(int l = 0; l < volumeBuffer.size(); ++l) {
                                if(volumeBufferOffset[l].y == 0) continue; //Skip empty sub-chunks
                                setVolumeParameters({0, chunk.vOffset + volumeBufferOffset[l].x, 0},
                                                                {prm_g.vwidth, chunk.vOffset + volumeBufferOffset[l].x + volumeBufferOffset[l].y, prm_g.vwidth}, {prm_g.vx, prm_g.vx, prm_g.vx});
                                setVolumeBuffer(volumeBuffer[l]);
                                forward();
                            }
                        }
                        error();
                        for(int l = 0; l < volumeBuffer.size(); ++l) {
                            if(volumeBufferOffset[l].y == 0) continue; //Skip empty sub-chunks
                            setVolumeParameters({0, chunk.vOffset + volumeBufferOffset[l].x, 0},
                                                            {prm_g.vwidth, chunk.vOffset + volumeBufferOffset[l].x + volumeBufferOffset[l].y, prm_g.vwidth}, {prm_g.vx, prm_g.vx, prm_g.vx});
                            setVolumeBuffer(volumeBuffer[l]);
                            backward();
                        }
                        setBuffer(sumImagesBuffer, 0);
                        wait();
                    }
                    weight *= prm_r.weight_factor;
                }
            }
            //Récuperer que la partie utile
            for(int l = 1; l < SUB_CHUNK+1; ++l) {
                std::vector<float> volume;
                getBuffer<float>(volumeBuffer[l], volume, true);
                _dataset.saveLayers(volume, chunk.vOffset + volumeBufferOffset[l].x, chunk.vOffset +  volumeBufferOffset[l].x + volumeBufferOffset[l].y);
            }
            std::cout << "Done." << std::endl;
        }
        std::cout << "Ok" << std::endl;
    }

private:
    
};