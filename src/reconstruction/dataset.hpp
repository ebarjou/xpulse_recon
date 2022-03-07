/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

namespace dataset {
    #include "dataset/parameters.hpp"
    #include "dataset/geometry.hpp"
    #include "dataset/geometry_chunk.hpp"
    #include "dataset/data_loader.hpp"

    /**
     * @brief Manage the projections data and Input/ouput for the reconstruction. Need to be initialized before use.
     * 
     */
    class Dataset {
        Parameters *_parameters;
        int64_t _width, _height;
        std::vector<std::string> _tiff_files;
        Geometry _geometry;
        DataLoader _dataLoader;
        
    public:
        Dataset(Parameters *parameters) : 
            _parameters(parameters), 
            _width(0),
            _height(0),
            _tiff_files(collectFromDirectory(prm_r.input)), //Init prm_g.projections, prm_g.dwidth, _dheight
            _geometry(parameters), //Init all missing fields
            _dataLoader(parameters, _tiff_files)
        {
            std::cout << "Exploring directory..."<< std::flush;
            collectFromDirectory(prm_r.input);
            if(_tiff_files.size() == 0 || _width == 0 || _height == 0) {
                throw std::runtime_error("No valid TIFF file found in the directory.");
            }
            std::cout << getElements() << "x(" << getWidth() << "," << getHeight() << ")." << std::endl;
        }

        /**
         * @brief Load all data and create temporary files if needed
         * 
         */
        void initialize() {
            if(prm_r.images_preproc) {
                _dataLoader.initializeTempImages();
            }
        }

        /**
         * @brief Get the Width of the images
         * 
         * @return uint64_t 
         */
        uint64_t getWidth() {
            return _width;
        }

        /**
         * @brief Get the Height of the images
         * 
         * @return uint64_t 
         */
        uint64_t getHeight() {
            return _height;
        }

        /**
         * @brief Get the number of images
         * 
         * @return uint64_t 
         */
        uint64_t getElements() {
            return _tiff_files.size();
        }

        /**
         * @brief Get a single layer
         * 
         * @param layer index of the layer, from 0 (top layer) to volume height - 1 (bottem layer)
         * @return std::vector<float> 
         */
        std::vector<float> getLayers(int64_t layer) {
            return _dataLoader.getLayer(layer);
        }

        /**
         * @brief Save the layer contained in data to a single layer file
         * 
         * @param data of the layer, should be of size width*width
         * @param layer index of the layer, from 0 (top layer) to volume height - 1 (bottem layer)
         */
        void saveLayer(const std::vector<float> &data, int64_t layer) {
            _dataLoader.saveLayer(&data[0], layer);
        }

        /**
         * @brief Save multiple continuous layer contained in data
         * 
         * @param data of the layer, should be of size width*width*(end-start)
         * @param start first layer index
         * @param end last layer index, excluded
         */
        void saveLayers(const std::vector<float> &data, int64_t start, int64_t end) {
            _dataLoader.saveLayers(&data[0], start, end);
        }

        /**
         * @brief get a single image
         * 
         * @param id of the image
         * @return std::vector<float> 
         */
        std::vector<float> getImages(int64_t id) {
            return _dataLoader.getImages(id);
        }

        /**
         * @brief get a single image
         * 
         * @param path of the image
         * @return std::vector<float> 
         */
        std::vector<float> getImage(std::string path) {
            return _dataLoader.getImage(path);
        }

        /**
         * @brief Get all images, cropped to fit a chunk, and separated in sub-iterations
         * 
         * @param chunk 
         * @return std::vector<std::vector<float>> a vector for each sub-iterations containing the cropped images
         */
        std::vector<std::vector<float>> getImagesCropped(int64_t chunk) {
            return _dataLoader.getImagesCropped(chunk);
        }

        /**
         * @brief Save multiple images
         * 
         * @param data vector containing all images
         * @param start offset, in images from the vector
         * @param end offset, im images from the vector
         * @param step in image
         */
        void saveImages(const std::vector<float> &data, int64_t start, int64_t end, int64_t step = 1) {
            for(int64_t l = start; l < end; l += step) {
                _dataLoader.saveProjImage(&data[((l-start)/step)*prm_g.dwidth*prm_g.dheight], l);
            }
        }

        /**
         * @brief Get the Geometry object
         * 
         * @return Geometry* 
         */
        Geometry* getGeometry() {
            return &_geometry;
        }

        /**
         * @brief Etablish the list of images that will be loaded when "initialize()" is called
         * 
         * @param directory 
         * @return std::vector<std::string> 
         */
        std::vector<std::string> collectFromDirectory(std::string directory) {
            std::vector<std::string> tiff_files;
            for (const auto & entry : std::filesystem::directory_iterator(directory)) {
                if(entry.is_regular_file() && (entry.path().extension() == ".tif" || entry.path().extension() == ".tiff")){
                    if(checkTiffFileInfos(entry.path().string())) {
                        tiff_files.push_back(entry.path().string());
                    }
                }
            }
            //Fill parameters accordingly
            prm_g.projections = tiff_files.size();
            prm_g.concurrent_projections = int64_t(std::floor( float(prm_g.projections) / float(prm_r.sit) ));
            prm_g.dwidth = _width;
            prm_g.dheight = _height;
            for(int64_t i = 0; i < int64_t(prm_md.size()); ++i) {
                if(!(prm_md[i].start_x.assigned)) prm_md[i].start_x = 0;
                if(!(prm_md[i].end_x.assigned)) prm_md[i].end_x = prm_g.dwidth;
                if(!(prm_md[i].start_y.assigned)) prm_md[i].start_y = 0;
                if(!(prm_md[i].end_y.assigned)) prm_md[i].end_y = prm_g.dheight;
            }
            return tiff_files;
        }

    private:
        /**
         * @brief Check if a tiff file is valid and have the correct size
         * 
         * @param file 
         */
        bool checkTiffFileInfos(std::string file) {
            TinyTIFFReaderFile* tiffr=NULL;
            tiffr=TinyTIFFReader_open(file.c_str()); 
            if (tiffr) {
                const int32_t width = TinyTIFFReader_getWidth(tiffr); 
                const int32_t height = TinyTIFFReader_getHeight(tiffr);
                const uint16_t bitspersample = TinyTIFFReader_getBitsPerSample(tiffr, 0);
                const uint16_t format = TinyTIFFReader_getSampleFormat(tiffr);
                TinyTIFFReader_close(tiffr);

                if(format != TINYTIFF_SAMPLEFORMAT_FLOAT || bitspersample != 32) {
                    throw std::runtime_error("Trying to load a tiff with an incompatble format (should be floating-point 32bits)");
                }
                if(width == 0 || height == 0) {
                    throw std::runtime_error("Trying to load a tiff with an incorrect size.");
                }
                if(_width == 0 && _height == 0) {
                    _width = width;
                    _height = height;
                }
                if(_width == width && _height == height) {
                    return true;
                }
                return false;
            }
            return false;
        }
    };
}