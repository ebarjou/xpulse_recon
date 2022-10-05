/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

namespace dataset {
    #include "dataset/parameters.hpp"
    #include "dataset/geometry.hpp"
    #include "dataset/data_loader.hpp"

    /**
     * @brief Manage the projections data and Input/ouput for the reconstruction. Need to be initialized before use.
     * 
     */
    class Dataset {
        reconstruction::dataset::Parameters *_parameters;
        std::vector<std::string> _tiff_files;
        reconstruction::dataset::Geometry _geometry;
        reconstruction::dataset::DataLoader _dataLoader;
        
    public:
        Dataset(reconstruction::dataset::Parameters *parameters) : 
            _parameters(parameters), 
            _tiff_files(collectFromDirectory(prm_r.input)), //Init prm_g.projections, prm_g.dwidth, _dheight
            _geometry(parameters), //Init all missing fields
            _dataLoader(parameters, _tiff_files)
        {
            std::cout << "Exploring directory..."<< std::flush;
            collectFromDirectory(prm_r.input);
            if(_tiff_files.size() == 0 || prm_g.dwidth == 0 || prm_g.dheight == 0) {
                throw std::runtime_error("No valid TIFF file found in the directory.");
            }
            std::cout << prm_g.projections << "x(" << prm_g.dwidth << "," << prm_g.dheight << ")." << std::endl;
        }

        /**
         * @brief Load all data and create temporary files if needed
         * 
         */
        void initialize(int64_t layersInRAMFrequency = 0) {
            _dataLoader.initialize(layersInRAMFrequency);
        }

        /**
         * @brief Get a single layer
         * 
         * @param layer index of the layer, from 0 (top layer) to volume height - 1 (bottem layer)
         * @return std::vector<float> 
         */
        std::valarray<float> getLayer(int64_t layer) {
            return _dataLoader.getLayer(layer).getFloatContent();
        }

        /**
         * @brief Save the layer contained in data to a single layer file
         * 
         * @param data of the layer, should be of size width*width
         * @param layer index of the layer, from 0 (top layer) to volume height - 1 (bottem layer)
         */
        void saveLayer(std::valarray<float> &data, int64_t layer, bool finalize = false) {
            _dataLoader.saveLayer(data, layer, finalize);
        }

        /**
         * @brief get a single image
         * 
         * @param id of the image
         * @return std::vector<float> 
         */
        std::valarray<float> getImage(int64_t id) {
            return _dataLoader.getImage(id).getFloatContent();
        }

        void getImage(int64_t id, std::valarray<float> &dst) {
            return _dataLoader.getImage(id).getFloatContent(&dst[0]);
        }

        /**
         * @brief get a single image
         * 
         * @param id of the image
         * @return std::vector<float> 
         */
        std::valarray<float> getImage(std::valarray<uint16_t> ids) {
            std::valarray<float> output(ids.size()*prm_g.dwidth*prm_g.dheight);
            for(int i = 0; i < ids.size(); ++i) {
                _dataLoader.getImage(ids[i]).getFloatContent(&output[i*prm_g.dwidth*prm_g.dheight]);
            }
            return output;
        }

        /**
         * @brief get a single image from a path
         * 
         * @param path of the image
         * @return std::vector<float> 
         */
        std::valarray<float> getImage(std::string path) {
            return _dataLoader.getImage(path).getFloatContent();
        }

        /**
         * @brief Get the Geometry object
         * 
         * @return Geometry* 
         */
        reconstruction::dataset::Geometry* getGeometry() {
            return &_geometry;
        }

private:
        /**
         * @brief Etablish the list of images that will be loaded when "initialize()" is called
         * 
         * @param directory 
         * @return std::vector<std::string> 
         */
        std::vector<std::string> collectFromDirectory(std::string directory) {
            std::vector<std::string> tiff_files;
            uint32_t width, height;
            for (const auto & entry : std::filesystem::directory_iterator(directory)) {
                if(entry.is_regular_file() && (entry.path().extension() == ".tif" || entry.path().extension() == ".tiff")){
                    if(tiff_files.size() == 0) {
                        _dataLoader.getTiffFileSize(entry.path().string(), width, height);
                    }
                    if(_dataLoader.checkTiffFile(entry.path().string(), TINYTIFF_SAMPLEFORMAT_FLOAT, width, height)) {
                        tiff_files.push_back(entry.path().string());
                    }
                }
            }
            if(tiff_files.size() == 0) {
                throw new std::exception("No valid TIFF file was found in the input folder");
            }
            //Fill parameters accordingly
            prm_g.projections = tiff_files.size();
            prm_g.concurrent_projections = int64_t(std::floor( float(prm_g.projections) / float(prm_r.sit) ));
            prm_g.dwidth = width;
            prm_g.dheight = height;
            return tiff_files;
        }
    };
}