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
    Parameters *_parameters;
    std::vector<std::string> _tiff_files;
    std::string _proj_folder;
    float _min_value = std::numeric_limits<float>::max();
    float _max_value = std::numeric_limits<float>::min();
    size_t _subit_images_elements = 0;
    double zfp_tolerance = 1.0/4096.0;
    int zstd_compression_level = 1;
    std::vector<float> _max_per_layer;
    float _layers_max = 0.0f;

public:
    DataLoader(Parameters *parameters, std::vector<std::string> tiff_files) : 
        _parameters(parameters),
        _tiff_files(tiff_files)
    {
        //Ensure output directory exist
        std::filesystem::create_directories(prm_r.output);
        if(!prm_r.proj_output.empty()) {
            std::filesystem::create_directories(prm_r.proj_output);
        }
        _subit_images_elements = (prm_g.projections/prm_r.sit)*prm_g.dwidth*prm_g.dheight;
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
    void initializeTempImages(bool chunks) {
        std::cout << "Pre-processing images..."<< std::flush;

        //Find min and max
        _max_per_layer.resize(prm_g.vheight);
        #pragma omp parallel
        {
            if(prm_r.normalize) {
                #pragma omp for schedule(dynamic)
                for(int i = 0; i < _tiff_files.size(); ++i) {
                    auto data = loadTIFF(_tiff_files[i], prm_g.dwidth, prm_g.dheight);
                    data.erase(std::remove_if(std::begin(data), std::end(data),
                            [](const auto& value) { return !std::isnormal(value); }),
                            std::end(data)
                    );
                    const auto [min, max] = std::minmax_element(data.begin(), data.end(),
                        [] (auto x, auto y) {
                            return x < y ? true : isnan(x);
                        }
                    );
                    #pragma omp critical
                    {
                        _min_value = std::min(_min_value, *min);
                        _max_value = std::max(_max_value, *max);
                    }
                }
            }
            #pragma omp for schedule(dynamic)
            for(int i = 0; i < prm_g.projections; ++i) {
                //Read
                auto data = loadTIFF(_tiff_files[i], prm_g.dwidth, prm_g.dheight);
                //Normalize
                if(prm_r.normalize) {
                    std::for_each(data.begin(), data.end(), [this](float& v) { v = std::isnormal(v)?(v-_min_value)/(_max_value-_min_value):(_max_value+_min_value)/2; });
                }
                //Log
                if(prm_r.mlog) {
                    std::for_each(data.begin(), data.end(), [](float& v) { v = -std::log(std::max(std::min(v, 1.0f-EPSILON), EPSILON)); });
                }
                //Write
                if(chunks) {
                    for(int c = 0; c < prm_m2.chunks.size(); ++c) {
                        saveZSTD(getTempFilePath(_proj_folder, "p", c*prm_g.projections+i, "zstd"), 
                                data.data()+prm_g.dwidth*prm_m2.chunks[c].iOffset, 
                                prm_g.dwidth, prm_m2.chunks[c].iSize
                        );
                    }
                } else {
                    saveZSTD(getTempFilePath(_proj_folder, "p", i, "zstd"), data.data(), prm_g.dwidth, prm_g.dheight);
                }
            }
        }
        std::cout << "Ok." << std::endl;
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
    std::vector<float> getLayer(int64_t layer) {
        //return loadZFP(getOutputFilePath("layer", layer, "zfp"), prm_g.vwidth, prm_g.vwidth);
        if(std::filesystem::exists(getOutputFilePath("layer", layer, "zstd"))) {
            return loadZSTD(getOutputFilePath("layer", layer, "zstd"), prm_g.vwidth, prm_g.vwidth);
        } else {
            return loadTIFF(getOutputFilePath("layer", layer, "tif"), prm_g.vwidth, prm_g.vwidth);
        }
    }

    /**
     * @brief Save the given data to the layer file of specified index
     * 
     * @param data must be at least of size width*width
     * @param layer index of the layer, from top to bottom
     */
    void saveLayer(float* data, int64_t layer, bool finalize = false) {
        if(finalize) {
            if(prm_r.normalize) {
                for(int i = 0; i < prm_g.vwidth*prm_g.vwidth; ++i) {
                    data[i] = data[i]*(_max_value-_min_value);
                }
            }
            saveTIFF(getOutputFilePath("layer", layer, "tif"), data, prm_g.vwidth, prm_g.vwidth);
            std::filesystem::remove(getOutputFilePath("layer", layer, "zstd"));
        } else {
            //saveZFP(getOutputFilePath("layer", layer, "zfp"), data, prm_g.vwidth, prm_g.vwidth);
            saveZSTD(getOutputFilePath("layer", layer, "zstd"), data, prm_g.vwidth, prm_g.vwidth);
        } 
    }

    /**
     * @brief Get the Images data
     * 
     * @param id index of the image
     * @return std::vector<float> 
     */
    std::vector<float> getImage(int64_t id) {
        auto image = loadZSTD(getTempFilePath(_proj_folder, "p", id, "zstd"), prm_g.dwidth, prm_g.dheight);
        return image;
    }

    /**
     * @brief REad a tiff image from a specified path
     * 
     * @param path to the file
     */
    std::vector<float> getImage(std::string path) {
        return loadTIFF(path, -1, -1);
    }

    void saveImage(const float *data, int width, int height, int id) {
        if(prm_r.proj_output.empty()) {
            saveTIFF(getOutputFilePath("projection", id, "tif"), data, width, height);
        } else {
            saveTIFF(getTempFilePath(prm_r.proj_output, "p", id, "tif"), data, width, height);
        }
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

    /**
     * @brief Save the content of image to the tiff file specified
     * 
     * @param file path of the tiff file
     * @param image data of the image
     * @param width of the image
     * @param height of the image
     */
    void saveTIFF(std::string file, const float *image, int64_t width, int64_t height) {
        auto tif = TinyTIFFWriter_open(file.c_str(), 32, TinyTIFFWriter_Float, 1, uint32_t(width), uint32_t(height), TinyTIFFWriter_Greyscale);
        if (tif) {
            TinyTIFFWriter_writeImage(tif, image);
            TinyTIFFWriter_close(tif);
        }
    }

    /**
     * @brief Load a tiff file
     * 
     * @param file path of the tiff file
     * @param width ignored
     * @param height ignored
     */
    std::vector<float> loadTIFF(std::string file, int64_t width, int64_t height) {
        auto tiffr = TinyTIFFReader_open(file.c_str()); 
        if (!tiffr || TinyTIFFReader_wasError(tiffr)) {
            throw std::runtime_error("Corrupted or missing TIFF file");
        }
        const int32_t twidth = TinyTIFFReader_getWidth(tiffr); 
        const int32_t theight = TinyTIFFReader_getHeight(tiffr);
        //TODO : change width/height behavior
        std::vector<float> image(twidth*theight);
        
        TinyTIFFReader_getSampleData(tiffr, image.data(), 0);
        TinyTIFFReader_close(tiffr);
        return image;
    }

    void saveZSTD(std::string filename, const float *image, int64_t width, int64_t height) {
        std::vector<char> output(width*height*sizeof(float));
        size_t const output_size = ZSTD_compress(output.data(), output.size(), image, width*height*sizeof(float), zstd_compression_level);
        
        auto file = std::ofstream(filename, std::ios::binary);
        file.write(output.data(), output_size);
        file.close();
    }

    std::vector<float> loadZSTD(std::string filename, int64_t width, int64_t height) {
        auto file = std::ifstream(filename, std::ios::binary);
        file.seekg(0, std::ios::end);
        size_t filesize=file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> compressed(filesize);
        std::vector<float> output(width*height);

        file.read(compressed.data(), filesize);
        file.close();

        size_t const dSize = ZSTD_decompress(output.data(), output.size()*sizeof(float), compressed.data(), filesize);
        return output;
    }

    void saveZSTD16(std::string filename, const float *image, int64_t width, int64_t height) {
        std::vector<uint16_t> input(width*height);
        std::vector<char> output(width*height*sizeof(uint16_t));
        float vmin = 1<<32, vmax = -vmin;
        for(int i = 0; i < width*height; ++i) {
            vmin = std::min(image[i], vmin);
            vmax = std::max(image[i], vmax);
        }
        for(int i = 0; i < width*height; ++i) {
            input[i] = uint16_t(std::floor(((image[i]-vmin)/(vmax-vmin))*65535.0f));
        }
        size_t const output_size = ZSTD_compress(output.data(), output.size(), input.data(), input.size()*sizeof(uint16_t), zstd_compression_level);
        
        auto file = std::ofstream(filename, std::ios::binary);
        file.write((char*)(&vmin), sizeof(float));
        file.write((char*)(&vmax), sizeof(float));
        file.write(output.data(), output_size);
        file.close();
    }

    std::vector<float> loadZSTD16(std::string filename, int64_t width, int64_t height) {
        auto file = std::ifstream(filename, std::ios::binary);

        float minmax[2];
        file.read((char*)&minmax, sizeof(float)*2);

        file.seekg(0, std::ios::end);
        size_t filesize = size_t(file.tellg()) - sizeof(float)*2;

        file.seekg(sizeof(float)*2, std::ios::beg);

        std::vector<char> compressed(filesize);
        std::vector<uint16_t> uncompressed(width*height);
        std::vector<float> output(width*height);

        file.read(compressed.data(), filesize);
        file.close();

        size_t const dSize = ZSTD_decompress(uncompressed.data(), uncompressed.size()*sizeof(uint16_t), compressed.data(), filesize);

        for(int i = 0; i < width*height; ++i) {
            output[i] = minmax[0]+(uncompressed[i]/65535.0f)*(minmax[1]-minmax[0]);
        }
        return output;
    }
};