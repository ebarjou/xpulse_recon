/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

class ReconstructionIPR : public reconstruction::Reconstruction {   
    typedef std::complex<float> complex;
    std::vector<complex> fresnel_forward, fresnel_backward;
    float iterations = prm_ipr.iterations;
public:
    ReconstructionIPR(reconstruction::dataset::Parameters *parameters) : reconstruction::Reconstruction(parameters)
    {
        if(requieredGPUMemory() > getOcl().memorySize) {
            throw std::runtime_error("Not enough GPU memory");
        }

        _dataset.initialize();

        float wl = 1.2398e-6f / prm_ipr.energy_kev; //kev to mm
        float k = (2.0f*M_PI/wl);
        float mag = prm_g.so / prm_g.sd;
        float z = (prm_g.sd - prm_g.so)/mag;
        float px = prm_d.px/mag;
        fresnel_forward.resize(prm_g.dwidth*prm_g.dheight);
        fresnel_backward.resize(prm_g.dwidth*prm_g.dheight);
        using namespace std::complex_literals;
        for(int j = 0; j < prm_g.dheight; ++j) {
            for(int i = 0; i < prm_g.dwidth; ++i) {
                float kx = ((i < prm_g.dwidth/2) ? float(i) : float(prm_g.dwidth-i))*(1.0f/(prm_g.dwidth*px));
                float ky = ((j < prm_g.dheight/2) ? float(j) : float(prm_g.dheight-j))*(1.0f/(prm_g.dheight*px));
                /*fresnel_forward[j*prm_g.dwidth+i] = std::exp(1if*z*std::sqrt(k*k-kx*kx-ky*ky));
                fresnel_backward[j*prm_g.dwidth+i] = std::exp(1if*(-z)*std::sqrt(k*k-kx*kx-ky*ky));*/
                fresnel_forward[j*prm_g.dwidth+i] = std::exp(1if*k*z)*std::exp(-1if*z*((kx*kx+ky*ky)/(2.0f*k)));
                fresnel_backward[j*prm_g.dwidth+i] = std::exp(1if*k*(-z))*std::exp(-1if*(-z)*((kx*kx+ky*ky)/(2.0f*k)));
            }
        }
    }

    ~ReconstructionIPR() {
    }

    uint64_t requieredGPUMemory() {
        int64_t volumeSize = prm_g.vwidth*prm_g.vheight*prm_g.vwidth*2;
        int64_t imageSize = prm_g.dwidth*prm_g.dheight*prm_g.projections*2;
        int64_t sumImageSize = prm_g.dwidth*prm_g.dheight*prm_g.concurrent_projections*2;
        int64_t totalSizeByte = (volumeSize+imageSize+sumImageSize)*sizeof(float);
        std::cout << "Checking GPU memory requierments : " << totalSizeByte/(1024*1024) << "Mo" << std::endl;
        return totalSizeByte;
    }

    void exec() {
        std::cout << "Loading data..." << std::flush;
        std::vector<std::vector<float>> images(prm_r.sit);
        std::vector<std::vector<float>> images_abs(prm_r.sit);
        std::vector<std::vector<float>> images_arg(prm_r.sit);

        for(int sit = 0; sit < prm_r.sit; ++sit) {
            images[sit] = _dataset.getImages(sit);
            images_abs[sit].resize(images[sit].size(), 1.0f);
            images_arg[sit].resize(images[sit].size(), 0.0f);
            phase_retrieval(images_abs[sit], images_arg[sit], images[sit]);
            /*_dataset.saveImages(images_abs[sit], 0, prm_g.concurrent_projections);
            _dataset.saveImages(images_arg[sit], prm_g.concurrent_projections, prm_g.concurrent_projections*2);
            exit(0);*/
        }
        auto projData = prm_g.projection_matrices;
        std::cout << "Ok" << std::endl;
         
        std::cout << "Allocating requiered buffers..." << std::flush;
        std::vector<reconstruction::gpu::Buffer> imagesBuffer_abs;
        std::vector<reconstruction::gpu::Buffer> imagesBuffer_arg;
        std::vector<reconstruction::gpu::Buffer> projDataBuffer;
        
        for(int sit = 0; sit < prm_r.sit; ++sit) {
            imagesBuffer_abs.push_back(createBuffer(images_abs[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE));
            imagesBuffer_arg.push_back(createBuffer(images_arg[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE));
            projDataBuffer.push_back(createBuffer(projData[sit], reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE));
        }

        auto volumeBuffer_abs = createBuffer<float>(prm_g.vwidth * prm_g.vwidth * prm_g.vheight, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ, WAVEFRONT_SIZE);
        auto volumeBuffer_arg = createBuffer<float>(prm_g.vwidth * prm_g.vwidth * prm_g.vheight, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ, WAVEFRONT_SIZE);
        auto sumImagesBuffer_abs = createBuffer<int32_t>(prm_g.dwidth * prm_g.dheight * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ);
        auto sumImagesBuffer_arg = createBuffer<int32_t>(prm_g.dwidth * prm_g.dheight * prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_READ);
        wait();
        std::cout << "Ok" << std::endl;
        
        std::cout << "Executing reconstruction..." << std::flush;

        setImageParameters(prm_g.dwidth, prm_g.dheight, prm_g.concurrent_projections);
        setVolumeParameters({0, 0, 0}, {prm_g.vwidth, prm_g.vheight, prm_g.vwidth}, {prm_g.vx, prm_g.vx, prm_g.vx});
        setImageOffset({0, prm_g.dheight});
        setAngleNumber(prm_g.concurrent_projections, prm_d.module_number);
        
        setBuffer(volumeBuffer_abs, 1.0f);
        setBuffer(volumeBuffer_arg, 1.0f);
        setBuffer(sumImagesBuffer_abs, FIXED_FRAC_ONE);
        setBuffer(sumImagesBuffer_arg, FIXED_FRAC_ONE);

        float weight = prm_r.weight;
        for(int mit = 0; mit < prm_r.it; ++mit) {
            setWeight(weight);
            for(int sit = 0; sit < prm_r.sit; ++sit) {
                setOrigin(prm_g.orig, {prm_g.vx, prm_g.vx, prm_g.vx});
                setProjDataBuffer(projDataBuffer[sit]);

                if(sit > 0 || mit > 0) {
                    setImagesBuffer(imagesBuffer_abs[sit]);
                    setSumImagesBuffer(sumImagesBuffer_abs);
                    setVolumeBuffer(volumeBuffer_abs);
                    forward();
                    setImagesBuffer(imagesBuffer_arg[sit]);
                    setSumImagesBuffer(sumImagesBuffer_arg);
                    setVolumeBuffer(volumeBuffer_arg);
                    forward();
                    if(prm_ipr.iterations > 0 && mit%prm_ipr.iterations_step == 0) {
                        std::cout << "|" << mit << "|" << std::endl;
                        std::vector<int32_t> projections_abs, projections_arg;
                        wait();
                        getBuffer(sumImagesBuffer_abs, projections_abs, true);
                        getBuffer(sumImagesBuffer_arg, projections_arg, true);

                        std::vector<float> new_projections_abs(projections_abs.size()), new_projections_arg(projections_arg.size());
                        for(int i = 0; i < new_projections_abs.size(); ++i) {
                            new_projections_abs[i] = FIXED_TO_FLOAT(projections_abs[i]);
                            new_projections_arg[i] = FIXED_TO_FLOAT(projections_arg[i]);
                        }
                        phase_retrieval(new_projections_abs, new_projections_arg, images[sit]);

                        setBuffer(imagesBuffer_abs[sit], new_projections_abs, true);
                        setBuffer(imagesBuffer_arg[sit], new_projections_arg, true);
                        wait();
                    }
                }
                setImagesBuffer(imagesBuffer_abs[sit]);
                setSumImagesBuffer(sumImagesBuffer_abs);
                setVolumeBuffer(volumeBuffer_abs);
                error();
                setImagesBuffer(imagesBuffer_arg[sit]);
                setSumImagesBuffer(sumImagesBuffer_arg);
                setVolumeBuffer(volumeBuffer_arg);
                error();

                setImagesBuffer(imagesBuffer_abs[sit]);
                setSumImagesBuffer(sumImagesBuffer_abs);
                setVolumeBuffer(volumeBuffer_abs);
                backward();
                setImagesBuffer(imagesBuffer_arg[sit]);
                setSumImagesBuffer(sumImagesBuffer_arg);
                setVolumeBuffer(volumeBuffer_arg);
                backward();

                setBuffer(sumImagesBuffer_abs, 0);
                setBuffer(sumImagesBuffer_arg, 0);
            }
            weight *= prm_r.weight_factor;
        }
        wait();
        std::cout << "Ok" << std::endl;

        std::cout << "Saving result..." << std::flush;
        std::vector<float> volume;
        getBuffer<float>(volumeBuffer_arg, volume);
        wait();
        _dataset.saveLayers(volume, 0, prm_g.vheight);
        std::cout << "Ok" << std::endl;
    }

    void fresnel_propagator(std::vector<complex> &wavefront, bool direction) {
        std::vector<complex> fourrier_values((prm_g.dwidth*prm_g.dheight));
        pocketfft::c2c( std::vector<size_t>{size_t(prm_g.dwidth), size_t(prm_g.dheight)}, 
                        std::vector<ptrdiff_t>{ptrdiff_t(sizeof(float)*2), ptrdiff_t(sizeof(float)*2*prm_g.dwidth)}, 
                        std::vector<ptrdiff_t>{ptrdiff_t(sizeof(float)*2), ptrdiff_t(sizeof(float)*2*prm_g.dwidth)},
                        std::vector<size_t>{0,1},
                        pocketfft::FORWARD,
                        &wavefront[0],
                        &fourrier_values[0],
                        1.0f,
                        1//Threads
        );
        if(direction==pocketfft::FORWARD) {
            for(int j = 0; j < fourrier_values.size(); ++j) {
                fourrier_values[j] *= fresnel_forward[j];
            }
        } else {
            for(int j = 0; j < fourrier_values.size(); ++j) {
                fourrier_values[j] /= fresnel_forward[j];
            }
        }
        pocketfft::c2c( std::vector<size_t>{size_t(prm_g.dwidth), size_t(prm_g.dheight)}, 
                        std::vector<ptrdiff_t>{ptrdiff_t(sizeof(float)*2), ptrdiff_t(sizeof(float)*2*prm_g.dwidth)}, 
                        std::vector<ptrdiff_t>{ptrdiff_t(sizeof(float)*2), ptrdiff_t(sizeof(float)*2*prm_g.dwidth)},
                        std::vector<size_t>{0,1},
                        pocketfft::BACKWARD,
                        &fourrier_values[0],
                        &wavefront[0],
                        1.0f/float(prm_g.dwidth*prm_g.dheight),
                        1
        );
    }

    void phase_retrieval(std::vector<float> &guess_abs, std::vector<float> &guess_arg, std::vector<float> &images) {
        using namespace std::complex_literals;
        #pragma omp parallel for
        for(int64_t j = 0; j < images.size(); j+=(prm_g.dwidth*prm_g.dheight)) {
            std::vector<complex> guess(prm_g.dwidth*prm_g.dheight);
            for(int i = 0; i < guess.size(); ++i) {
                guess[i] = std::exp(-guess_abs[j+i])*std::exp(1if*guess_arg[j+i]);
            }
            for(int k = 0; k < int(iterations); ++k) {
                //forward propagation
                fresnel_propagator(guess, pocketfft::FORWARD);
                //detector constraint
                for(int i = 0; i < guess.size(); ++i) {
                    guess[i] = std::polar(images[j+i], std::arg(guess[i]));
                }
                //backward propagation
                fresnel_propagator(guess, pocketfft::BACKWARD);
                //detector constraint
                /*for(int i = 0; i < guess.size(); ++i) {
                    guess[i] = std::polar(std::abs(guess[i]), -std::abs(std::arg(guess[i])));
                }*/
            }
            //float min_arg = 0.0f;
            for(int i = 0; i < guess.size(); ++i) {
                guess_abs[j+i] = -std::log(std::min(1.0f,std::abs(guess[i])));
                guess_arg[j+i] = std::abs(std::arg(guess[i]));
                //min_arg = std::min(guess_arg[j+i], min_arg);
            }
            /*for(int i = 0; i < guess.size(); ++i) {
                guess_arg[i] = (guess_arg[i] - min_arg) + 1e-9f;
            }*/
        }
        iterations *= prm_ipr.iterations_fct;
    }

private:
};