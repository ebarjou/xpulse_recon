/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

namespace reconstruction {
    #include "reconstruction/dataset.hpp"
    #include "reconstruction/gpu.hpp"
    
    using gpu::Buffer;
    using dataset::Dataset;
    
    /**
     * @brief class representing a reconstruction algorithm.
     * All recon inherit from this class, that initialize the GPU kernels, the dataset, 
     *  and provide the necessary methods to configure the projections operators
     */
    class Reconstruction : protected gpu::Context {
        std::random_device _randomDevice;
        std::mt19937 _randomEngine = std::mt19937(_randomDevice());
    
    protected:
        dataset::Parameters *_parameters;
        Dataset _dataset;
        
    public:
        Reconstruction(dataset::Parameters *parameters) : gpu::Context(),
                            _parameters(parameters),
                            _dataset(parameters)
        {
            
        }

        /**
         * @brief Return the amount of GPU memory requiered by the algorithm, in bytes.
         * 
         * @return uint64_t number of bytes
         */
        virtual uint64_t requieredGPUMemory() {
            return 0;
        };

        /**
         * @brief Main function of the reconstruction algorithm, where the forward, error and backward are called.
         * 
         */
        virtual void exec() = 0;

    protected:
        glm::vec3 getRandomizedOffset() {
            glm::vec3 offset{0,0,0};
            for(int i = 0; i < 3; ++i) {
                offset[i] += std::uniform_real_distribution<float>(0, prm_g.vx)(_randomEngine);
            }
            return offset;
        }
    };

    #include "reconstruction/reconstruction_abs.hpp"
    //#include "reconstruction/reconstruction_hartmann.hpp"
    //#include "reconstruction/reconstruction_ipr.hpp"
}