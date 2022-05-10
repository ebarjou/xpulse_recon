/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

class ReconstructionHartmann : public reconstruction::Reconstruction {   
    std::vector<float> B;
    std::vector<uint8_t> G;
    int64_t width, height;
    float factor = 0.0f;
    float iterations;
    uint32_t total_iterations_done = 0;
    float scaling = 1.0f;
    float minimum = 0.0f; 

public:
    ReconstructionHartmann(reconstruction::dataset::Parameters *parameters) : reconstruction::Reconstruction(parameters),
            G(prm_g.dwidth*prm_g.dheight),
            B(prm_g.dwidth*prm_g.dheight*prm_g.concurrent_projections),
            iterations(prm_hm.iterations)
    {
        if(requieredGPUMemory() > getOcl().memorySize) {
            throw std::runtime_error("Not enough GPU memory");
        }

        _dataset.initialize();

        width = prm_g.dwidth;
        height = prm_g.dheight;
        std::vector<std::string> devx_folder = _dataset.collectFromDirectory(prm_hm.devx);
        std::vector<std::string> devy_folder = _dataset.collectFromDirectory(prm_hm.devy);

        #pragma omp parallel for
        for(int i = 0; i < devx_folder.size(); ++i) {
            std::vector<float> dx = _dataset.getImage(devx_folder[i]);
            std::vector<float> dy = _dataset.getImage(devy_folder[i]);
            std::vector<float> b(width*height);
            //Pre-compute B_jk (sum of the derivative of dy and dx) and G_jk (number of neighbors)
            for(int y = 0; y < height; ++y) {
                for(int x = 0; x < width; ++x) {
                    uint8_t neighboors = 4 - ((x==0 || x==(width-1))?1:0) - ((y==0 || y==(height-1))?1:0);
                    float diff = (((y > 0)?dy[x+width*(y-1)]:0) - dy[x+width*y]) + (((x > 0)?dx[(x-1)+width*y]:0) - dx[x+width*y]);
                    if(i == 0) {
                        G[y*width+x] = neighboors;
                    }
                    B[i*width*height+y*width+x] = (diff * parameters->detector.px) / neighboors;
                }
            }
        }
    }

    ~ReconstructionHartmann() {
        std::cout << "Total wavefront reconstruction iterations : " << total_iterations_done << std::endl;
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
            if(prm_hm.iterations > 0) {
                std::vector<float> projections(prm_g.dwidth*prm_g.dheight*prm_g.concurrent_projections);
                southwell(projections, sit, int32_t(prm_g.concurrent_projections));
                images.push_back(projections);
            } else {
                images.push_back(_dataset.getImages(sit));
            }
        }
        auto projData = prm_g.projection_matrices;
        std::cout << "Ok" << std::endl;
         
        std::cout << "Allocating requiered buffers..." << std::flush;
        std::vector<reconstruction::gpu::Buffer> imagesBuffer;
        std::vector<reconstruction::gpu::Buffer> projDataBuffer;
        
        for(int sit = 0; sit < prm_r.sit; ++sit) {
            imagesBuffer.push_back(createBuffer(images[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE));
            projDataBuffer.push_back(createBuffer(projData[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE));
        }

        auto volumeBuffer = createBuffer<float>(prm_g.vwidth * prm_g.vwidth * prm_g.vheight, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ, WAVEFRONT_SIZE);
        auto sumImagesBuffer = createBuffer<int32_t>(prm_g.dwidth * prm_g.dheight * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ);
        wait();
        std::cout << "Ok" << std::endl;
        
        std::cout << "Executing reconstruction..." << std::flush;
        setSumImagesBuffer(sumImagesBuffer);
        setVolumeBuffer(volumeBuffer);

        setImageParameters(prm_g.dwidth, prm_g.dheight, prm_g.concurrent_projections);
        setVolumeParameters({0, 0, 0}, {prm_g.vwidth, prm_g.vheight, prm_g.vwidth}, {prm_g.vx, prm_g.vx, prm_g.vx});
        setImageOffset({0, prm_g.dheight});
        setAngleNumber(prm_g.concurrent_projections, prm_d.module_number);
        
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
                    if(prm_hm.iterations > 0 && mit%prm_hm.iterations_step == 0) {
                        std::cout << "|" << mit << "|" << std::endl;
                        std::vector<int32_t> projections;
                        wait();
                        getBuffer(sumImagesBuffer, projections, true);

                        std::vector<float> new_projections(projections.size());
                        for(int i = 0; i < projections.size(); ++i) {new_projections[i] = FIXED_TO_FLOAT(projections[i]);}
                        southwell(new_projections, sit, int32_t(prm_g.concurrent_projections));//TODO : add step for sit
                        setBuffer(imagesBuffer[sit], new_projections, true);
                        wait();
                    }
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

    /**
     * @brief Execute the southwell wavefront reconsturction algorithm
     * 
     * @param ph 
     * @param start 
     * @param number 
     */
    void ReconstructionHartmann::southwell(std::vector<float> &ph, int start, int number) {//Ajouter step
        for(int i = 0; i < ph.size(); ++i) {
            ph[i] = -(ph[i]*scaling + minimum);
        }
        //Execute the iterations (Succecive Over-Relaxation method)
        //float w = 2.0f/(1.0f+std::sin(3.14159f/(std::max(width, height)+1.0f)));
        //Execute the iterations (Gauss-Seidel)
        #pragma omp parallel for
        for(int i = start; i < start+number; ++i) {
            int64_t offset = (i-start)*width*height;
            for(int k = 0; k < iterations*std::max(width, height); ++k) {
                for(int y = 0; y < height; ++y) {
                    for(int x = 0; x < width; ++x) {
                        float ph_nna_xy = ((x<width-1)?ph[offset+x+1+y*width]:0)
                                        + ((x>0)?ph[offset+x-1+y*width]:0)
                                        + ((y<height-1)?ph[offset+x+(y+1)*width]:0)
                                        + ((y>0)?ph[offset+x+(y-1)*width]:0);
                        ph[offset+x+y*width] = ph_nna_xy/G[x+y*width] + B[offset+x+y*width];
                    }
                }
            }
        }
        for(int i = 0; i < ph.size(); ++i) {
            ph[i] = -ph[i];
        }
        const auto [min, max] = std::minmax_element(std::begin(ph), std::end(ph));
        scaling = *max-*min;
        minimum = *min;
        for(int i = 0; i < ph.size(); ++i) {
            ph[i] = (ph[i]-minimum)/scaling;
        }

        total_iterations_done += uint32_t(iterations*std::max(width, height));
        iterations *= prm_hm.iterations_fct;
    }

private:
};