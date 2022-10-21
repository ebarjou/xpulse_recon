/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

/**
 * @brief Handle the loading and writing of the images and layers from and to the hard-drive
 * Abstract the processing of the images (either pre-process and store in temp file or process at loading time)
 * 
 */
class DataLoader {
    reconstruction::dataset::Parameters *_parameters;
    std::vector<std::string> _tiff_files;
    std::string _proj_folder;

    int64_t layersInRAMFrequency;
    std::vector<reconstruction::dataset::ImageFXP> layerStorage;

public:
    DataLoader(reconstruction::dataset::Parameters *parameters, std::vector<std::string> tiff_files) : 
        _parameters(parameters),
        _tiff_files(tiff_files)
    {
        //Ensure output directory exist and is empty
        std::filesystem::remove_all(prm_r.output);
        std::filesystem::create_directories(prm_r.output);
        if(!prm_r.proj_output.empty()) {
            std::filesystem::create_directories(prm_r.proj_output);
        }
        _proj_folder = prm_r.output + "/projs/";
        std::filesystem::create_directories(_proj_folder);
    }

    ~DataLoader() {
        cleanTempImages();
    }

    /**
     * @brief Create and fill necesarry temporary files
     * 
     */
    void initialize(int64_t layersInRAMFrequency) {
        this->layersInRAMFrequency = layersInRAMFrequency;
        layerStorage.resize(prm_g.vheight);

        std::cout << "Pre-processing images..."<< std::flush;
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < prm_g.projections; ++i) {
            saveTIFF(getTempFilePath(_proj_folder, "p", i, "tif"), loadExternalTIFF(_tiff_files[i]));
        }
        std::cout << "Ok." << std::endl;
        std::cout << "One out of " << layersInRAMFrequency << " layers will be stored in RAM" << std::endl;
    }

    /**
     * @brief Clean all temporary files
     * 
     */
    void cleanTempImages() {
        std::cout << "Deleting temp files..." << std::flush;
        std::filesystem::remove_all(_proj_folder);
        std::cout << "Ok." << std::endl;
    }

    /**
     * @brief Get the Layer data of specified index
     * 
     * @param layer index of the layer, from top to bottom
     */
    reconstruction::dataset::ImageFXP getLayer(int64_t layer) {
        if(layerStorage[layer].is_valid()) {
            return layerStorage[layer];
        }
        return loadTIFF(getOutputFilePath("layer", layer, "tif"));
    }

    /**
     * @brief Save the given data to the layer file of specified index
     * 
     * @param data must be at least of size width*width
     * @param layer index of the layer, from top to bottom
     */
    void saveLayer(std::valarray<float> &image, int64_t layer, bool finalize = false) {
        if(finalize) {
            saveExternalTIFF(getOutputFilePath("layer", layer, "tif"), &image[0], prm_g.vwidth, prm_g.vwidth);
        } else if(layer%layersInRAMFrequency == 0) {
            layerStorage[layer] = reconstruction::dataset::ImageFXP(image, prm_g.vwidth, prm_g.vwidth);
        } else {
            saveTIFF(getOutputFilePath("layer", layer, "tif"), reconstruction::dataset::ImageFXP(image, prm_g.vwidth, prm_g.vwidth));
        }
    }

    void saveLayer(float *image, int64_t layer, bool finalize = false) {
        if(finalize) {
            saveExternalTIFF(getOutputFilePath("layer", layer, "tif"), image, prm_g.vwidth, prm_g.vwidth);
        } else if(layer%layersInRAMFrequency == 0) {
            layerStorage[layer] = reconstruction::dataset::ImageFXP(image, prm_g.vwidth, prm_g.vwidth);
        } else {
            saveTIFF(getOutputFilePath("layer", layer, "tif"), reconstruction::dataset::ImageFXP(image, prm_g.vwidth, prm_g.vwidth));
        }
    }

    /**
     * @brief Get the Images data
     * 
     * @param id index of the image
     * @return std::vector<float> 
     */
    reconstruction::dataset::ImageFXP getImage(int64_t id) {
        return loadTIFF(getTempFilePath(_proj_folder, "p", id, "tif"));
    }

    /**
     * @brief Get the Images data
     * 
     * @param id index of the image
     * @return std::vector<float> 
     */
    reconstruction::dataset::ImageFXP getImage(std::string path) {
        return loadTIFF(path);
    }

    void saveImage(float *image, int64_t index, int64_t width, int64_t height) {
        saveExternalTIFF(getOutputFilePath("image", index, "tif"), image, width, height);
    }

    bool checkTiffFile(std::string filename, uint16_t expected_format, uint32_t expected_width, uint32_t expected_height) {
        TinyTIFFReaderFile* tiffr=TinyTIFFReader_open(filename.c_str()); 
        if (!tiffr) { 
            throw new std::exception((std::string("TIFF file ") + filename + std::string(" does not exist.")).c_str());
        } 
        const uint32_t width = TinyTIFFReader_getWidth(tiffr); 
        const uint32_t height = TinyTIFFReader_getHeight(tiffr);
        const uint16_t format = TinyTIFFReader_getSampleFormat(tiffr);
        TinyTIFFReader_close(tiffr);
        return width == expected_width && height == expected_height && format == expected_format;
    }

    void getTiffFileSize(std::string filename, uint32_t &width, uint32_t &height) {
        TinyTIFFReaderFile* tiffr=TinyTIFFReader_open(filename.c_str()); 
        if (!tiffr) {
            throw new std::exception((std::string("TIFF file ") + filename + std::string(" does not exist.")).c_str());
        }
        width = TinyTIFFReader_getWidth(tiffr); 
        height = TinyTIFFReader_getHeight(tiffr);
        TinyTIFFReader_close(tiffr);
    }

private:
    /**
     * @brief Construct a file path from a prefix and a number, poiting to the temp folder
     * 
     * @param prefix name of the file
     * @param index number placed after the prefix
     */
    std::string getTempFilePath(std::string folder, std::string prefix, int64_t index, std::string extension) {
        std::ostringstream ss;
        ss << folder << "\\" << prefix << "_" << std::setw(10) << std::setfill('0') << index << "." << extension;
        return std::string(ss.str());
    }

    /**
     * @brief Construct a file path from a prefix and a number, poiting to the output folder
     * 
     * @param prefix name of the file
     * @param index number placed after the prefix, zero-padded with a constant size of five
     */
    std::string getOutputFilePath(std::string baseName, int64_t index, std::string extension) {
        std::ostringstream ss;
        ss << prm_r.output << "\\" << baseName << "_" << std::setw(5) << std::setfill('0') << index << "." << extension;
        return std::string(ss.str());
    }

    void saveTIFF(std::string filename, reconstruction::dataset::ImageFXP &image) {
        TinyTIFFWriterFile* tiffw=TinyTIFFWriter_open(filename.c_str(), 16, TinyTIFFWriter_UInt, 1, uint32_t(image.width), uint32_t(image.height), TinyTIFFWriter_Greyscale);
        if (tiffw) {
            auto coded_content = image.getCodedContent();
            TinyTIFFWriter_writeImage(tiffw, &coded_content[0]);
            TinyTIFFWriter_close(tiffw);
        }
    }

    reconstruction::dataset::ImageFXP loadTIFF(std::string filename) {
        TinyTIFFReaderFile* tiffr=TinyTIFFReader_open(filename.c_str()); 
        if (!tiffr) { 
            throw new std::exception((std::string("TIFF file ") + filename + std::string(" does not exist.")).c_str());
        } else { 
            const uint32_t width = TinyTIFFReader_getWidth(tiffr); 
            const uint32_t height = TinyTIFFReader_getHeight(tiffr);
            const uint16_t bps = TinyTIFFReader_getBitsPerSample(tiffr, 0);
            const uint16_t format = TinyTIFFReader_getSampleFormat(tiffr);
            if(format == TINYTIFF_SAMPLEFORMAT_UINT && bps == 16) {
                std::valarray<uint16_t> values(width*height);	
                TinyTIFFReader_getSampleData(tiffr, &values[0], 0);
                TinyTIFFReader_close(tiffr);
                return reconstruction::dataset::ImageFXP(values, width, height);
            } else {
                TinyTIFFReader_close(tiffr);
                throw new std::exception("Unsupported TIFF format, only SAMPLEFORMAT_IEEEFP is supported");
            }
        }
    }

    void saveExternalTIFF(std::string filename, float* image, int64_t width, int64_t height) {
        TinyTIFFWriterFile* tiffw=TinyTIFFWriter_open(filename.c_str(), 32, TinyTIFFWriter_Float, 1, uint32_t(width), uint32_t(height), TinyTIFFWriter_Greyscale);
        if (tiffw) {
            TinyTIFFWriter_writeImage(tiffw, image);
            TinyTIFFWriter_close(tiffw);
        }
    }

    reconstruction::dataset::ImageFXP loadExternalTIFF(std::string filename) {
        TinyTIFFReaderFile* tiffr=TinyTIFFReader_open(filename.c_str()); 
        if (!tiffr) { 
            throw new std::exception((std::string("TIFF file ") + filename + std::string(" does not exist.")).c_str());
        } else { 
            const uint32_t width = TinyTIFFReader_getWidth(tiffr); 
            const uint32_t height = TinyTIFFReader_getHeight(tiffr);
            const uint16_t bps = TinyTIFFReader_getBitsPerSample(tiffr, 0);
            const uint16_t format = TinyTIFFReader_getSampleFormat(tiffr);
            if(format == TINYTIFF_SAMPLEFORMAT_FLOAT && bps == 32) {
                std::valarray<float> values(width*height);	
                TinyTIFFReader_getSampleData(tiffr, &values[0], 0);
                TinyTIFFReader_close(tiffr);
                processExternalTIFF(values);
                return reconstruction::dataset::ImageFXP(values, width, height);
            } else {
                TinyTIFFReader_close(tiffr);
                throw new std::exception("Unsupported TIFF format, only SAMPLEFORMAT_IEEEFP is supported");
            }
        }
    }

    void processExternalTIFF(std::valarray<float> &data) {
        for(int i = 0; i < data.size(); ++i) {
            float value = std::clamp(data[i], 0.0f, 1.0f);
            if(prm_r.mlog) {
                value = -std::log(value);
            }
            //TODO : to improve, better to keep nan for min/max calculation
            if(value <= 0 || !(std::isfinite(value))) {
                value = 0;
            }
            data[i] = value;
        }
    }
};