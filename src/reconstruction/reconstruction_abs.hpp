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
    enum State{NONE, FORWARD, ERROR, BACKWARD};
    static const uint32_t CPU_THREADS = 6;
    static const uint32_t CONCURRENT_LAYERS = 3;
    static const uint32_t MAX_CONCURRENT_ANGLES = 400;//To use, to prevent constant buffer overflow
    static const int64_t FWD = 0;
    static const int64_t BWD = 1;
    int64_t sub_iterations, concurrent_projections;
    int64_t _sit, _mit;
    float _weight;
    //Threading
    msd::channel<int64_t> worker_channel;
    std::vector<std::thread> worker_thread;
    std::atomic<State> _state;
    std::atomic<int64_t> _remaining_task;
    std::vector<reconstruction::gpu::Program> program;
    //GPU Buffers
    reconstruction::gpu::Buffer imageBuffer, mvpBuffer, vpBuffer;
    //RAM Buffers
    std::vector<std::valarray<uint16_t>> sit_image_set;
    std::vector<std::vector<std::vector<uint16_t>>> layer_sit_mvp_indexes, layer_sit_image_indexes;

public:
    ReconstructionAbs(reconstruction::dataset::Parameters *parameters) : reconstruction::Reconstruction(parameters), 
        _weight(prm_r.weight)
    {
        //RAM repartition
        int64_t remainingRAM = prm_r.usable_ram_go*1024*1024*1024;
        remainingRAM -= prm_g.dheight*prm_g.dwidth*sizeof(float)*CPU_THREADS; //One image in ram per thread
        remainingRAM -= prm_g.vwidth*prm_g.vwidth*sizeof(float)*CPU_THREADS*CONCURRENT_LAYERS; //One layer in ram per thread
        //Fill free ram with layers
        int64_t max_layers_in_ram = remainingRAM/(prm_g.vwidth*prm_g.vwidth*sizeof(uint16_t));
        if(max_layers_in_ram < 0)
             throw new std::exception("Not enough RAM");
        int64_t layersInRAM = std::min(max_layers_in_ram, prm_g.vheight);
        int64_t layersInRAMFrequency = prm_g.vheight/layersInRAM;
        _dataset.initialize(layersInRAMFrequency);

        //VRAM Allocation
        int64_t remainingVRAM = getOcl()->memorySize;
        remainingVRAM -= prm_g.vwidth*prm_g.vwidth*sizeof(float)*CPU_THREADS*CONCURRENT_LAYERS; //One layer in vram per thread
        remainingVRAM -= _dataset.getGeometry()->mvpArray.size()*sizeof(float); //A unique mvp buffer in vram
        remainingVRAM -= _dataset.getGeometry()->vpArray.size()*sizeof(uint16_t); //A unique vp buffer in vram
        remainingVRAM -= 2*16384*sizeof(uint16_t)*CPU_THREADS; //Two constant index buffer in vram for each thread
        //Reduce concurrent_projections and change sit if needed
        int64_t max_concurrent_projections = std::min(remainingVRAM, int64_t(getOcl()->maxAllocSize))/(prm_g.dheight*prm_g.dwidth*sizeof(float));
        if(max_concurrent_projections < 1)
             throw new std::exception("Not enough VRAM");
        concurrent_projections = std::min(prm_g.projections, max_concurrent_projections);
        sub_iterations = prm_g.projections/concurrent_projections;
        std::cout << "Iterations will be divied in " << sub_iterations << " sub-iteration(s) of " << concurrent_projections << " angle(s)." << std::endl;

        imageBuffer = createBuffer<uint32_t>(prm_g.dheight*prm_g.dwidth*concurrent_projections, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_RW);
        mvpBuffer = createBuffer(_dataset.getGeometry()->mvpArray, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE);
        vpBuffer = createBuffer(_dataset.getGeometry()->vpArray, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_NONE);
        
        for(int64_t tid = 0; tid < CPU_THREADS; ++tid) {
            program.push_back(reconstruction::gpu::Program(getOcl(), kernel_recon_source, std::vector<const char*>{"Forward", "Backward"}));
            worker_thread.push_back(std::thread(&reconstruction::ReconstructionAbs::worker, this, tid));
        }
        prepareImageSets();
    }

    void prepareImageSets() {
        sit_image_set.resize(sub_iterations);
        for(int64_t sit = 0; sit < sub_iterations; ++sit) {
            sit_image_set[sit].resize(concurrent_projections);
            for(int64_t i = 0; i < concurrent_projections; ++i) {
                //Ici : check si l'id rentre dans un short, sinon except ?
                sit_image_set[sit][i] = uint16_t((sit+i*sub_iterations)%prm_g.projections);
            }
        }

        layer_sit_mvp_indexes.resize(prm_g.vheight);
        layer_sit_image_indexes.resize(prm_g.vheight);
        #pragma omp parallel for
        for(int l = 0; l < prm_g.vheight; ++l) {
            layer_sit_mvp_indexes[l].resize(sub_iterations);
            layer_sit_image_indexes[l].resize(sub_iterations);

            //mvp de la layer courante
            auto mvpIndexPerLayers = _dataset.getGeometry()->mvpIndexPerLayers[l];
            std::vector<uint16_t> current_layer_mvp_index(std::begin(mvpIndexPerLayers), std::end(mvpIndexPerLayers));

            for(int64_t sit = 0; sit < sub_iterations; ++sit) {
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

    void worker(int64_t tid) {
        auto queue = createQueue();
        cl::Event event;

        //VRAM allocation
        auto imageIndexBuffer = createBuffer<uint32_t>(16384, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE);
        auto mvpIndexBuffer = createBuffer<uint32_t>(16384, reconstruction::gpu::MEM_ACCESS_READ, reconstruction::gpu::MEM_ACCESS_WRITE);
        auto volumeBuffer = createBuffer<float>(prm_g.vwidth*prm_g.vwidth, reconstruction::gpu::MEM_ACCESS_RW, reconstruction::gpu::MEM_ACCESS_RW);
        //RAM allocation
        auto refImageArray = std::valarray<float>(prm_g.dwidth * prm_g.dheight);

        for(int64_t i = 0; i < 2; ++i) {
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_SUMIMAGE_BUFFER, imageBuffer);
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_MVP_BUFFER, mvpBuffer);
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_VP_BUFFER, vpBuffer);
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_IMGINDEX_BUFFER, imageIndexBuffer);
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_MVPINDEX_BUFFER, mvpIndexBuffer);
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_VOLUME_BUFFER, volumeBuffer);
            
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_VOXEL_SIZE_F, prm_g.vx);
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_DETWIDTH_U, uint32_t(prm_g.dwidth));
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_DETHEIGHT_U, uint32_t(prm_g.dheight));
            program[tid].setKernelArgument(i, reconstruction::gpu::INDEX_VOLWIDTH_U, uint32_t(prm_g.vwidth));
            program[tid].setProgramSize(prm_g.vwidth, prm_g.vwidth, CONCURRENT_LAYERS);
        } 

        for(const auto k : worker_channel) {
            switch(_state) {
                case FORWARD: {
                    float *mapped_layer = mapBuffer<float>(&queue, volumeBuffer, 0, prm_g.vwidth*prm_g.vwidth, CL_MAP_WRITE, false, &event);
                    setBuffer(&queue, mvpIndexBuffer, 0, layer_sit_mvp_indexes[k][_sit], 0, layer_sit_mvp_indexes[k][_sit].size(), false);
                    setBuffer(&queue, imageIndexBuffer, 0, layer_sit_image_indexes[k][_sit], 0, layer_sit_image_indexes[k][_sit].size(), false);
                    program[tid].setKernelArgument(FWD, reconstruction::gpu::INDEX_ANGLES_U, uint32_t(layer_sit_mvp_indexes[k][_sit].size()));
                    program[tid].setProgramOffset(0, 0, k);
                    event.wait();
                    _dataset.getLayer(k, mapped_layer);
                    unmapBuffer(&queue, volumeBuffer, mapped_layer, &event);
                    event.wait();
                    program[tid].executeKernel(&queue, FWD, &event);
                    event.wait();                         
                } break;
                case ERROR: {
                    if(_sit > 0 || _mit > 0) {
                        float *mapped_image = mapBuffer<float>(&queue, imageBuffer, k*prm_g.dwidth*prm_g.dheight, prm_g.dwidth*prm_g.dheight, CL_MAP_READ|CL_MAP_WRITE, false, &event);
                        _dataset.getImage(sit_image_set[_sit][k], refImageArray);
                        event.wait();
                        uint32_t* mapped_image_fxp = (uint32_t*)mapped_image;
                        for(int64_t j = 0; j < prm_g.dwidth*prm_g.dheight; ++j) {
                            if(mapped_image_fxp > 0) {
                                float value = FIXED_TO_FLOAT(mapped_image_fxp[j]);
                                mapped_image[j] = refImageArray[j]/std::max(value, EPSILON);
                            }
                        }
                        unmapBuffer(&queue, imageBuffer, mapped_image, &event);
                        event.wait();
                    } else {
                        float *mapped_image = mapBuffer<float>(&queue, imageBuffer, k*prm_g.dwidth*prm_g.dheight, prm_g.dwidth*prm_g.dheight, CL_MAP_WRITE, true);
                        _dataset.getImage(sit_image_set[_sit][k], mapped_image);
                        unmapBuffer(&queue, imageBuffer, mapped_image, &event);
                        event.wait();
                    }
                } break;
                case BACKWARD: {     
                    if(_sit > 0 || _mit > 0) {
                        float *mapped_layer = mapBuffer<float>(&queue, volumeBuffer, 0, prm_g.vwidth*prm_g.vwidth, CL_MAP_WRITE, true);
                        _dataset.getLayer(k, mapped_layer);
                        unmapBuffer(&queue, volumeBuffer, mapped_layer, &event);
                    } else {
                        setBuffer(&queue, volumeBuffer, 1.0f/std::sqrtf(prm_g.vwidth*prm_g.vwidth), &event);
                    }
                    setBuffer(&queue, mvpIndexBuffer, 0, layer_sit_mvp_indexes[k][_sit], 0, layer_sit_mvp_indexes[k][_sit].size(), false);
                    setBuffer(&queue, imageIndexBuffer, 0, layer_sit_image_indexes[k][_sit], 0, layer_sit_image_indexes[k][_sit].size(), false);
                    program[tid].setKernelArgument(BWD, reconstruction::gpu::INDEX_ANGLES_U, uint32_t(layer_sit_mvp_indexes[k][_sit].size()));
                    program[tid].setProgramOffset(0, 0, k);
                    event.wait();
                    program[tid].executeKernel(&queue, BWD, &event);
                    event.wait();
                    float *mapped_layer = mapBuffer<float>(&queue, volumeBuffer, 0, prm_g.vwidth*prm_g.vwidth, CL_MAP_READ, true);
                    _dataset.saveLayer(mapped_layer, k, ((_sit==sub_iterations-1) && (_mit == prm_r.it-1)));
                    unmapBuffer(&queue, volumeBuffer, mapped_layer, &event);
                    event.wait();
                } break;
            }
            --_remaining_task;
        }
    }

    void print_status() {
        float steps = BACKWARD;
        float total_iteration = prm_r.it*sub_iterations;
        float current_iteration = _mit*sub_iterations+_sit;
        float single_iteration_pc = 100.0f/total_iteration;
        float single_step_pc = single_iteration_pc/3.0f;
        float completion_pc = current_iteration*single_iteration_pc + (_state-1)*single_step_pc;
        std::cout << "Executing reconstruction : " << std::setw(3) << int(completion_pc) << "% completed, ";
        std::cout << "iteration "<< std::setw(5) << current_iteration << " of " << std::setw(5) << total_iteration;
        std::cout << " @ " << (_state==FORWARD?"Forward":"") << (_state==ERROR?"Error":"") << (_state==BACKWARD?"Backward":"");
        std::cout << "          \r" << std::flush;
    }

    void exec() {
        auto t1 = std::chrono::high_resolution_clock::now();

        auto queue = createQueue();
        cl::Event event;

        std::cout << "Starting reconstruction" << std::endl;

        for(_mit = 0; _mit < prm_r.it; ++_mit) {
            for(int tid = 0; tid < CPU_THREADS; ++tid) {
                program[tid].setKernelArgument(BWD, reconstruction::gpu::INDEX_WEIGHT_F, _weight);
                program[tid].setKernelArgument(FWD, reconstruction::gpu::INDEX_WEIGHT_F, _weight);
            }
            _weight *= prm_r.weight_factor;

            for(_sit = 0; _sit < sub_iterations; ++_sit) {
                glm::vec3 offset = getRandomizedOffset();
                for(int tid = 0; tid < CPU_THREADS; ++tid) {
                    program[tid].setKernelArgument(BWD, reconstruction::gpu::INDEX_ORIGIN_F4, glm::vec4(prm_g.orig+offset, 1.0f));
                    program[tid].setKernelArgument(FWD, reconstruction::gpu::INDEX_ORIGIN_F4, glm::vec4(prm_g.orig+offset, 1.0f));
                }

                //Forward
                if(_sit > 0 || _mit > 0) {
                    setBuffer(&queue, imageBuffer, 0.0f, &event);
                    event.wait();
                    _state = FORWARD;
                    _remaining_task = 0;
                    print_status();
                    for(int64_t i = 0; i < prm_g.vheight; i += CONCURRENT_LAYERS) {
                        ++_remaining_task;
                        i >> worker_channel;
                    }
                    while(_remaining_task > 0) {};
                }

                _state = ERROR;
                _remaining_task = 0;
                print_status();
                for(int64_t i = 0; i < int64_t(sit_image_set[_sit].size()); ++i) {
                    ++_remaining_task;
                    i >> worker_channel;
                }
                while(_remaining_task > 0) {};
            
                _state = BACKWARD;
                _remaining_task = 0;
                print_status();
                for(int64_t i = 0; i < prm_g.vheight; i += CONCURRENT_LAYERS) {
                    ++_remaining_task;
                    i >> worker_channel;
                }
                while(_remaining_task > 0) {};
            }
        }
        std::cout << std::endl;
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Reconstruction done in " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << "s." << std::endl;
    }
};