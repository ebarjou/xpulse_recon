/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

struct ImageFXP {
    std::valarray<uint16_t> content;
    float min_value, max_value;
    uint32_t width, height;

    ImageFXP() : content(), min_value(0), max_value(0) {};

    ImageFXP(std::valarray<float> &float_content, uint32_t width, uint32_t height)
             : width(width), height(height) {
        
        const auto [min, max] = std::minmax_element(std::begin(float_content), std::end(float_content));
        min_value = *min;
        max_value = *max;
        content = std::valarray<uint16_t>(width*height);
        for(int i = 0; i < width*height; ++i) {
            content[i] = uint16_t( 65535.0f * ((float_content[i]-min_value)/(max_value-min_value)));
        }
    }

    ImageFXP(std::valarray<uint16_t> uint16_content, uint32_t width, uint32_t height, float min, float max)
             : width(width), height(height)  {
        min_value = min;
        max_value = max;
        content = std::valarray<uint16_t>(&uint16_content[0], width*height);
    }

    std::valarray<float> getFloatContent() {
        std::valarray<float> float_content(content.size());
        for(int i = 0; i < content.size(); ++i) {
            float_content[i] = float(content[i]);
        }
        return (float_content/65535.0f)*(max_value-min_value) + min_value;
    }

    uint16_t* data() {
        return &content[0];
    }

    uint32_t size() {
        return content.size();
    }

    bool is_valid() {
        return content.size() == width*height && content.size() > 0;
    }
};

/**
 * @brief Handle the loading and writing of the images and layers from and to the hard-drive
 * Abstract the processing of the images (either pre-process and store in temp file or process at loading time)
 * 
 */
class DataLoader {
    Parameters *_parameters;
    std::vector<std::string> _tiff_files;
    std::string _proj_folder;

    int64_t layersInRAMFrequency;
    std::vector<ImageFXP> layerStorage;

public:
    DataLoader(Parameters *parameters, std::vector<std::string> tiff_files) : 
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
    ImageFXP getLayer(int64_t layer) {
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
    void saveLayer(std::valarray<float> image, int64_t layer, bool finalize = false) {
        if(layer%layersInRAMFrequency == 0 && !finalize) {
            layerStorage[layer] = ImageFXP(image, prm_g.vwidth, prm_g.vwidth);
        } else {
            saveTIFF(getOutputFilePath("layer", layer, "tif"), ImageFXP(image, prm_g.vwidth, prm_g.vwidth));
        }
    }

    /**
     * @brief Get the Images data
     * 
     * @param id index of the image
     * @return std::vector<float> 
     */
    ImageFXP getImage(int64_t id) {
        return loadTIFF(getTempFilePath(_proj_folder, "p", id, "tif"));
    }

    /**
     * @brief Get the Images data
     * 
     * @param id index of the image
     * @return std::vector<float> 
     */
    ImageFXP getImage(std::string path) {
        return loadTIFF(path);
    }

    bool checkTiffFile(std::string filename, uint16_t expected_format, uint32_t expected_width, uint32_t expected_height) {
        TIFF* tiff = TIFFOpen(filename.c_str(), "r");
        if(!tiff) {
            throw new std::exception((std::string("TIFF file ") + filename + std::string(" does not exist.")).c_str());
        }
        uint32_t width, height;
        uint16_t format;
        TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tiff, TIFFTAG_SAMPLEFORMAT, &format);
        TIFFClose(tiff);
        return width == expected_width && height == expected_height && format == expected_format;
    }

    void getTiffFileSize(std::string filename, uint32_t &width, uint32_t &height) {
        TIFF* tiff = TIFFOpen(filename.c_str(), "r");
        if(!tiff) {
            throw new std::exception((std::string("TIFF file ") + filename + std::string(" does not exist.")).c_str());
        }
        TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
        TIFFClose(tiff);
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

    void saveTIFF(std::string filename, ImageFXP &image) {
        TIFF *tiff = TIFFOpen(filename.c_str(), "w");
        TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, image.width); 
        TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, image.height); 
        TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 16); 
        TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1); 
        TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, image.height);
        TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
        TIFFSetField(tiff, TIFFTAG_XPOSITION, image.max_value);
        TIFFSetField(tiff, TIFFTAG_YPOSITION, image.min_value);
        TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFWriteRawStrip(tiff, 0, image.data(), image.size()*sizeof(uint16_t));
        TIFFClose(tiff);
    }

    ImageFXP loadTIFF(std::string filename) {
        TIFF* tiff = TIFFOpen(filename.c_str(), "r");
        if(!tiff) {
            throw new std::exception((std::string("Internal TIFF file ") + filename + std::string(" does not exist.")).c_str());
        }
        float min_value, max_value;
        uint32_t width, height, rps;
        uint16_t bps, spp, format;
        TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bps); 
        TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &spp);
        TIFFGetField(tiff, TIFFTAG_ROWSPERSTRIP, &rps);
        TIFFGetField(tiff, TIFFTAG_SAMPLEFORMAT, &format);
        TIFFGetField(tiff, TIFFTAG_XPOSITION, &max_value);
        TIFFGetField(tiff, TIFFTAG_YPOSITION, &min_value);
        ImageFXP image;
        if(format == SAMPLEFORMAT_UINT && bps == 16) {
            std::valarray<uint16_t> values(width*height);
            TIFFReadRawStrip(tiff, 0, &values[0], width*TIFFTAG_ROWSPERSTRIP*bps/8);
            image = ImageFXP(values, width, height, min_value, max_value);
        } else {
            throw new std::exception("Unsupported TIFF format while attempting to read an internal TIFF file");
        }
        TIFFClose(tiff);
        return image;
    }

    ImageFXP loadExternalTIFF(std::string filename) {
        TIFF* tiff = TIFFOpen(filename.c_str(), "r");
        if(!tiff) {
            throw new std::exception((std::string("TIFF file ") + filename + std::string(" does not exist.")).c_str());
        }
        uint32_t width, height;
        uint16_t bps, spp, format;
        TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bps); 
        TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &spp);
        TIFFGetField(tiff, TIFFTAG_SAMPLEFORMAT, &format);
        ImageFXP image;
        if(format == SAMPLEFORMAT_IEEEFP && bps == 32) {
            std::valarray<float> values(width*height);
            for(int i = 0; i < height; i++) {
                TIFFReadScanline(tiff, &values[0]+i*width, i);
            }
            TIFFClose(tiff);
            processExternalTIFF(values);
            image = ImageFXP(values, width, height);
        } else {
            TIFFClose(tiff);
            throw new std::exception("Unsupported TIFF format, only SAMPLEFORMAT_IEEEFP is supported");
        }
        return image;
    }

    void processExternalTIFF(std::valarray<float> &data) {
        data[data<LARGE_EPSILON] = LARGE_EPSILON;
        data[data>(1-LARGE_EPSILON)] = 1-LARGE_EPSILON;
        if(prm_r.mlog) {
            data = -std::log(data);
        }
    }
};