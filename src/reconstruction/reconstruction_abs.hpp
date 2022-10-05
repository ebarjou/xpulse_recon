/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

static const std::string kernel_recon_source(
    #include "reconstruction/gpu/kernel_recon_str.hpp"
);

class ReconstructionAbs : public reconstruction::Reconstruction {
    struct Layer{std::mutex m; reconstruction::gpu::Buffer b;};
    static const uint32_t VOLUME_BUFFERS = 4;
    static const uint32_t CPU_THREADS = VOLUME_BUFFERS*2;
    static const int64_t FWD = 0;
    static const int64_t BWD = 1;
    reconstruction::gpu::Program _program;
    std::vector<cl::CommandQueue> queue;
    float _weight;
    //GPU Buffers
    reconstruction::gpu::Buffer imageBuffer, mvpBuffer, vpBuffer, mvpIndexBuffer, imageIndexBuffer;
    Layer volumeBuffers[VOLUME_BUFFERS];
    std::vector<reconstruction::gpu::Buffer> imageSubBuffer;
    //RAM Buffers
    std::valarray<float> imageArray[CPU_THREADS], refImageArray[CPU_THREADS], volumeArray[CPU_THREADS];
    std::vector<std::valarray<uint16_t>> sit_image_set;
    std::vector<std::vector<std::vector<uint16_t>>> layer_sit_mvp_indexes, layer_sit_image_indexes;

public:
    ReconstructionAbs(reconstruction::dataset::Parameters *parameters) : reconstruction::Reconstruction(parameters), 
        _program(getOcl(), kernel_recon_source, std::vector<const char*>{"Forward", "Backward"}), 
        _weight(prm_r.weight)
    {
        //RAM repartition
        int64_t remainingRAM = prm_r.usable_ram_go*1024*1024*1024;
        int64_t singleImageSizeRAM = prm_g.dheight*prm_g.dwidth*sizeof(float);
        int64_t imageSizeRAM = prm_g.concurrent_projections*singleImageSizeRAM;
        remainingRAM -= imageSizeRAM;

        int64_t layerSizeRAM = prm_g.vwidth*prm_g.vwidth*sizeof(uint16_t);
        if(remainingRAM > layerSizeRAM) {
            int64_t layersInRAM = std::min(remainingRAM/layerSizeRAM, prm_g.vheight);
            int64_t layersInRAMFrequency = prm_g.vheight/layersInRAM;
            layersInRAM = prm_g.vheight/layersInRAMFrequency;
            _dataset.initialize(layersInRAMFrequency);  
        } else {
            throw new std::exception("Not enough RAM allocated");
        }

        //VRAM Allocation
        imageBuffer = createBuffer<float>(prm_g.dheight*prm_g.dwidth*prm_g.concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_RW);
        mvpBuffer = createBuffer(_dataset.getGeometry()->mvpArray, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE);
        vpBuffer = createBuffer(_dataset.getGeometry()->vpArray, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE);
        imageIndexBuffer = createBuffer<uint32_t>(16384, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE);
        mvpIndexBuffer = createBuffer<uint32_t>(16384, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE);
        for(int64_t i = 0; i < VOLUME_BUFFERS; ++i) {
            volumeBuffers[i].b = createBuffer<float>(prm_g.vwidth*prm_g.vwidth, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_RW);
        }

        //RAM Allocation
        for(int64_t i = 0; i < CPU_THREADS; ++i) {
            imageArray[i].resize(prm_g.dwidth * prm_g.dheight);
            refImageArray[i].resize(prm_g.dwidth * prm_g.dheight);
            volumeArray[i].resize(prm_g.vwidth * prm_g.vwidth);
        }
        //TODO vérifier les tailles CPU_THREADS vs VOLUME_BUFFERS en accès tableaux
        //Allouer toute la memoire nécessaire ici

        for(int64_t i = 0; i < CPU_THREADS; ++i) {
            queue.push_back(createQueue());
        }

        for(int64_t i = 0; i < 2; ++i) {
            _program.setKernelArgument(i, reconstruction::gpu::INDEX_SUMIMAGE_BUFFER, imageBuffer);
            _program.setKernelArgument(i, reconstruction::gpu::INDEX_MVP_BUFFER, mvpBuffer);
            _program.setKernelArgument(i, reconstruction::gpu::INDEX_VP_BUFFER, vpBuffer);
            _program.setKernelArgument(i, reconstruction::gpu::INDEX_IMGINDEX_BUFFER, imageIndexBuffer);
            _program.setKernelArgument(i, reconstruction::gpu::INDEX_MVPINDEX_BUFFER, mvpIndexBuffer);
            
            _program.setKernelArgument(i, reconstruction::gpu::INDEX_VOXEL_SIZE_F, prm_g.vx);
            _program.setKernelArgument(i, reconstruction::gpu::INDEX_DETWIDTH_U, uint32_t(prm_g.dwidth));
            _program.setKernelArgument(i, reconstruction::gpu::INDEX_DETHEIGHT_U, uint32_t(prm_g.dheight));
            _program.setKernelArgument(i, reconstruction::gpu::INDEX_VOLWIDTH_U, uint32_t(prm_g.vwidth));
            _program.setProgramSize(prm_g.vwidth, prm_g.vwidth);
        }

        prepareImageSets();
    }

    void prepareImageSets() {
        sit_image_set.resize(prm_r.sit);
        for(int64_t sit = 0; sit < prm_r.sit; ++sit) {
            sit_image_set[sit].resize(prm_g.concurrent_projections);
            for(int64_t i = 0; i < prm_g.concurrent_projections; ++i) {
                //Ici : check si l'id rentre dans un short, sinon except ?
                sit_image_set[sit][i] = uint16_t((sit+i*prm_r.sit)%prm_g.projections);
            }
        }

        layer_sit_mvp_indexes.resize(prm_g.vheight);
        layer_sit_image_indexes.resize(prm_g.vheight);
        #pragma omp parallel for
        for(int l = 0; l < prm_g.vheight; ++l) {
            layer_sit_mvp_indexes[l].resize(prm_r.sit);
            layer_sit_image_indexes[l].resize(prm_r.sit);

            //mvp de la layer courante
            auto mvpIndexPerLayers = _dataset.getGeometry()->mvpIndexPerLayers[l];
            std::vector<uint16_t> current_layer_mvp_index(std::begin(mvpIndexPerLayers), std::end(mvpIndexPerLayers));

            for(int64_t sit = 0; sit < prm_r.sit; ++sit) {
                for(int64_t i = 0; i < int64_t(current_layer_mvp_index.size()); ++i) {
                    //Id de l'image correspondant au mvp
                    uint16_t global_image_id = _dataset.getGeometry()->imageIndexArray[current_layer_mvp_index[i]];
                    //Position de cet image dans le set d'image de la sit
                    auto index_in_current_set = std::find(std::begin(sit_image_set[sit]), std::end(sit_image_set[sit]), global_image_id);
                    //Si l'image est présente
                    if(index_in_current_set != std::end(sit_image_set[sit])) {
                        //On sauvegarde l'index du mvp et de l'image
                        uint16_t local_image_id = uint16_t(index_in_current_set - std::begin(sit_image_set[sit]));
                        layer_sit_mvp_indexes[l][sit].push_back(current_layer_mvp_index[i]);
                        layer_sit_image_indexes[l][sit].push_back(local_image_id);
                    }
                }
            }
        }
    }

    void exec() {
        //omp_set_nested(0);
        #pragma omp parallel num_threads(CPU_THREADS)
        {
            int tid = omp_get_thread_num();
            int vid = tid%VOLUME_BUFFERS;
            
            #pragma omp single
            std::cout << "[CPU_THREADS] : " << omp_get_num_threads() << std::endl;

            for(int64_t mit = 0; mit < prm_r.it; ++mit) {
                #pragma omp single
                {
                    _program.setKernelArgument(BWD, reconstruction::gpu::INDEX_WEIGHT_F, _weight);
                    _program.setKernelArgument(FWD, reconstruction::gpu::INDEX_WEIGHT_F, _weight);
                    _weight *= prm_r.weight_factor;
                }

                for(int64_t sit = 0; sit < prm_r.sit; ++sit) {
                    #pragma omp single
                    {
                        std::cout << "Executing reconstruction..." << (100*(mit*prm_r.sit+sit))/(prm_r.it*prm_r.sit) << "%" << "\r" << std::flush;
                        glm::vec3 offset = getRandomizedOffset();
                        _program.setKernelArgument(BWD, reconstruction::gpu::INDEX_ORIGIN_F4, glm::vec4(prm_g.orig+offset, 1.0f));
                        _program.setKernelArgument(FWD, reconstruction::gpu::INDEX_ORIGIN_F4, glm::vec4(prm_g.orig+offset, 1.0f));
                    }

                    //Forward
                    if(sit > 0 || mit > 0) {
                        #pragma omp single
                        setBuffer(queue[tid], imageBuffer, 0, true);

                        #pragma omp for schedule(dynamic)
                        for(int l = 0; l < prm_g.vheight; ++l) {
                            volumeArray[tid] = _dataset.getLayer(l);

                            volumeBuffers[vid].m.lock();
                            setBuffer(queue[tid], volumeBuffers[vid].b, 0, volumeArray[tid], 0, prm_g.dwidth*prm_g.dwidth, true);
                            #pragma omp critical
                            {
                                setBuffer(queue[tid], mvpIndexBuffer, 0, layer_sit_mvp_indexes[l][sit], 0, layer_sit_mvp_indexes[l][sit].size(), false);
                                setBuffer(queue[tid], imageIndexBuffer, 0, layer_sit_image_indexes[l][sit], 0, layer_sit_image_indexes[l][sit].size(), false);
                                _program.setKernelArgument(FWD, reconstruction::gpu::INDEX_VOLUME_BUFFER, volumeBuffers[vid].b);
                                _program.setKernelArgument(FWD, reconstruction::gpu::INDEX_ANGLES_U, uint32_t(layer_sit_mvp_indexes[l][sit].size()));
                                _program.setProgramOffset(0, 0, l);
                                _program.executeKernel(queue[tid], FWD);
                                queue[tid].flush();
                                //Equivalent to finish, but not busy
                                setBuffer(queue[tid], mvpIndexBuffer, 0, std::vector<uint16_t>(2), 0, 2, true);
                            }
                            volumeBuffers[vid].m.unlock();                               
                        }
                        #pragma omp barrier
                    }

                    #pragma omp single
                    std::cout << "[" << tid << "] - [" << vid << "] - " << "Start error" << std::endl;

                    //Error
                    //#pragma omp for schedule(dynamic)
                    #pragma omp single
                    for(int64_t i = 0; i < int64_t(sit_image_set[sit].size()); ++i) {    
                        if(sit > 0 || mit > 0) {
                            getBuffer(queue[tid], imageBuffer, i*prm_g.dwidth*prm_g.dheight, imageArray[tid], 0, prm_g.dwidth*prm_g.dheight, true);
                            float* dst = &imageArray[tid][0];
                            uint32_t* src = (uint32_t*)&imageArray[tid][0];
                            for(int64_t j = 0; j < int64_t(imageArray[tid].size()); ++j) {
                                dst[j] = FIXED_TO_FLOAT(src[j]);
                            }
                        } else {
                            imageArray[tid] = 1.0f;
                        }
                        _dataset.getImage(sit_image_set[sit][i], refImageArray[tid]);
                        
                        imageArray[tid][imageArray[tid] < EPSILON] = EPSILON;
                        imageArray[tid] = refImageArray[tid]/imageArray[tid];
                        setBuffer(queue[tid], imageBuffer, i*prm_g.dwidth*prm_g.dheight, imageArray[tid], 0, prm_g.dwidth*prm_g.dheight, true);
                    }
                    #pragma omp barrier

                    #pragma omp single
                    std::cout << "[" << tid << "] - [" << vid << "] - " << "End error" << std::endl;
                    
                    //Backward
                    #pragma omp for schedule(dynamic)
                    for(int64_t l = 0; l < prm_g.vheight; ++l) {    
                        #pragma omp critical (cout) 
                        std::cout << "[" << tid << "] - [" << vid << "] - " << "Start layer " << l << std::endl;                   
                        if(sit > 0 || mit > 0) {
                            volumeArray[tid] = _dataset.getLayer(l);
                        } else {
                            volumeArray[tid] = 1.0f/(prm_g.vwidth*prm_g.vwidth);
                        }

                        volumeBuffers[vid].m.lock();
                        #pragma omp critical (cout)
                        std::cout << "[" << tid << "] - [" << vid << "] - " << "locked layer" << std::endl;
                        setBuffer(queue[tid], volumeBuffers[vid].b, 0, volumeArray[tid], 0, prm_g.dwidth*prm_g.dwidth, true);
                        #pragma omp critical
                        {
                            #pragma omp critical (cout)
                            std::cout << "[" << tid << "] - [" << vid << "] - " << "critical section" << std::endl;
                            setBuffer(queue[tid], mvpIndexBuffer, 0, layer_sit_mvp_indexes[l][sit], 0, layer_sit_mvp_indexes[l][sit].size(), false);
                            setBuffer(queue[tid], imageIndexBuffer, 0, layer_sit_image_indexes[l][sit], 0, layer_sit_image_indexes[l][sit].size(), false);
                            _program.setKernelArgument(BWD, reconstruction::gpu::INDEX_VOLUME_BUFFER, volumeBuffers[vid].b);
                            _program.setKernelArgument(BWD, reconstruction::gpu::INDEX_ANGLES_U, uint32_t(layer_sit_mvp_indexes[l][sit].size()));
                            _program.setProgramOffset(0, 0, l);
                            _program.executeKernel(queue[tid], BWD);
                            queue[tid].flush();
                            queue[tid].finish();
                        }
                        getBuffer(queue[tid], volumeBuffers[vid].b, 0, volumeArray[tid], 0, prm_g.vwidth*prm_g.vwidth, true);
                        volumeBuffers[vid].m.unlock();
                        #pragma omp critical (cout)
                        std::cout << "[" << tid << "] - [" << vid << "] - " << "unlocked layer" << std::endl;

                        _dataset.saveLayer(volumeArray[tid], l, ((sit==prm_r.sit-1) && (mit == prm_r.it-1)));
                        #pragma omp critical (cout)
                        std::cout << "[" << tid << "] - [" << vid << "] - " << "layer done" << std::endl;
                    }
                    #pragma omp barrier
                }
            }
        }
        std::cout << "Executing reconstruction..." << " Done." << std::endl;
    }
};