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
    std::string _temp_folder;
    float _min_value = std::numeric_limits<float>::max();
    float _max_value = std::numeric_limits<float>::min();
    size_t _subit_images_elements = 0;

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
    }

    ~DataLoader() {
        if(prm_r.images_preproc) {
            cleanTempImages();
        }
    }

    /**
     * @brief Create and fill necesarry temporary files
     * 
     */
    void initializeTempImages() {
        //Create temp directory
        _temp_folder = prm_r.output + "/tmp/";
        std::filesystem::create_directories(_temp_folder);

        std::cout << "Pre-processing images..."<< std::flush;

        #pragma omp parallel for
        for(int i = 0; i < _tiff_files.size(); ++i) {
            auto data = loadTIFF(_tiff_files[i], prm_g.dwidth, prm_g.dheight);
            const auto [min, max] = std::minmax_element(data.begin(), data.end());
            #pragma omp critical
            {
                _min_value = std::min(_min_value, *min);
                _max_value = std::max(_max_value, *max);
            }
        }
        if(prm_m2.chunks.size() == 0) {
            #pragma omp parallel for if(prm_r.sit>1)
            for(int i = 0; i < prm_r.sit; ++i) {
                allocateTempFile("p", i , _subit_images_elements*sizeof(float));
                auto file = openWriteTrunc("p", i);
                for(int64_t j = i; j < int64_t(_tiff_files.size()); j += prm_r.sit) {
                    auto data = loadTIFF(_tiff_files[j], prm_g.dwidth, prm_g.dheight);
                    std::for_each(data.begin(), data.end(), [this](float& v) { v = (v-_min_value)/(_max_value-_min_value); });
                    if(prm_r.mlog) {
                        std::for_each(data.begin(), data.end(), [](float& v) { v = -std::log(std::max(std::min(v, 1.0f-EPSILON), EPSILON)); });
                    }
                    file.write((char*)&data[0], data.size()*sizeof(float));
                }
                file.close();
            }
        } else {
            //#pragma omp parallel for if(prm_r.sit>1)
            for(int i = 0; i < prm_r.sit; ++i) {
                std::vector<std::ofstream> files;
                for(int c = 0; c < prm_m2.chunks.size(); ++c) {
                    allocateTempFile("p", c*_tiff_files.size()+i , prm_g.dwidth*prm_m2.chunks[c].iSize*sizeof(float));
                    files.push_back(openWriteTrunc("p", c*_tiff_files.size()+i));
                }
                for(int64_t j = i; j < int64_t(_tiff_files.size()); j += prm_r.sit) {
                    auto data = loadTIFF(_tiff_files[j], prm_g.dwidth, prm_g.dheight);
                    std::for_each(data.begin(), data.end(), [this](float& v) { v = (v-_min_value)/(_max_value-_min_value); });
                    if(prm_r.mlog) {
                        std::for_each(data.begin(), data.end(), [](float& v) { v = -std::log(std::max(std::min(v, 1.0f-EPSILON), EPSILON)); });
                    }
                    for(int c = 0; c < prm_m2.chunks.size(); ++c) {
                        files[c].write((char*)&data[prm_g.dwidth*prm_m2.chunks[c].iOffset], prm_g.dwidth*prm_m2.chunks[c].iSize*sizeof(float));
                    }
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
        std::filesystem::remove_all(_temp_folder);
        #pragma omp parallel for
        for(int i = 0; i < prm_g.vheight; ++i) {
            auto data = getLayer(i);
            std::for_each(data.begin(), data.end(), [this](float& v) { v = v*(_max_value-_min_value); });
            saveLayer(&data[0], i);
        }
        std::cout << "Ok." << std::endl;
        
    }

    /**
     * @brief Get the Layer data of specified index
     * 
     * @param layer index of the layer, from top to bottom
     */
    std::vector<float> getLayer(int64_t layer) {
        return loadTIFF(getOutputFilePath("layer", layer), prm_g.vwidth, prm_g.vwidth);   
    }

    /**
     * @brief Save the given data to the layer file of specified index
     * 
     * @param data must be at least of size width*width
     * @param layer index of the layer, from top to bottom
     */
    void saveLayer(const float* data, int64_t layer) {
        saveTIFF(getOutputFilePath("layer", layer), data, prm_g.vwidth, prm_g.vwidth);
    }

    /**
     * @brief Save the given data to the layer files [istart,iend[ 
     * Consider the layer data as contiguous
     * 
     * @param data must be at least of size width*width*(iend-istart)
     * @param istart index of the first layer
     * @param iend index of the last layer + 1
     */
    void saveLayers(const float* data, int64_t istart, int64_t iend) {
        #pragma omp parallel for if((iend-istart) > 1)
        for(int64_t i = istart; i < iend; ++i) {
            saveLayer(&data[prm_g.vwidth*prm_g.vwidth*(i-istart)], i);
        }
    }

    /**
     * @brief Get the Images data for the specified sub-iteration
     * 
     * @param sit sub-iteration index
     * @return std::vector<float> 
     */
    std::vector<float> getImages(int64_t sit) {
        if(prm_r.images_preproc) {
            std::vector<float> data(_subit_images_elements);
            readTemp("p", sit, &data[0], _subit_images_elements*sizeof(float));
            return data;
        } else {
            std::vector<float> data;
            #pragma omp parallel for ordered
            for(int64_t i = sit; i < int64_t(_tiff_files.size()); i += prm_r.sit) {
                auto image = loadTIFF(_tiff_files[i], prm_g.dwidth, prm_g.dheight);
                if(prm_r.mlog) std::for_each(image.begin(), image.end(), [](float& v) { v = -std::log(std::max(std::min(v, 1.0f-EPSILON), EPSILON)); });
                #pragma omp ordered
                data.insert(data.end(), image.begin(), image.end());
            }
            return data;
        }
    }

    /**
     * @brief Get all images, cropped to fit a chunk, and separated in sub-iterations
     * 
     * @param chunk 
     * @return std::vector<std::vector<float>> a vector for each sub-iterations containing the cropped images
     */
    std::vector<std::vector<float>> getImagesCropped(int64_t chunk) {
        std::vector<std::vector<float>> data(prm_r.sit);
        if(prm_r.images_preproc) {
            #pragma omp parallel for if(prm_r.sit>1)
            for(int sit = 0; sit < prm_r.sit; ++sit) {
                data[sit].resize(prm_g.dwidth*prm_m2.chunks[chunk].iSize*prm_g.concurrent_projections);
                readTemp("p", chunk*_tiff_files.size()+sit, &data[sit][0], prm_g.dwidth*prm_m2.chunks[chunk].iSize*prm_g.concurrent_projections*sizeof(float));
            }
        } else {
            #pragma omp parallel for if(prm_r.sit>1)
            for(int sit = 0; sit < prm_r.sit; ++sit) {
                for(int64_t i = sit; i < int64_t(_tiff_files.size()); i += prm_r.sit) {
                    auto image = loadTIFF(_tiff_files[i], prm_g.dwidth, prm_g.dheight);
                    image = std::vector<float>(image.begin()+prm_m2.chunks[chunk].iOffset*prm_g.dwidth, image.begin()+(prm_m2.chunks[chunk].iOffset+prm_m2.chunks[chunk].iSize)*prm_g.dwidth);
                    if(prm_r.mlog) {
                        std::for_each(image.begin(), image.end(), [](float& v) { v = -std::log(std::max(std::min(v, 1.0f-EPSILON), EPSILON)); });
                    }
                    data[sit].insert(data[sit].end(), image.begin(), image.end());
                }
            }
        }
        return data;
    }

    /**
     * @brief Save the given data to the output image file of specified index
     * 
     * @param data must be at least of size width*height
     * @param layer index of the image
     */
    void saveProjImage(const float* data, int64_t index) {
        if(prm_r.images_preproc) {
            std::ostringstream ss;
            ss << prm_r.proj_output << "\\" << "image" << "_" << std::setw(5) << std::setfill('0') << index << ".tif";
            std::vector<float> denormalized(prm_g.dwidth*prm_g.dheight);
            for(int i = 0; i < prm_g.dwidth*prm_g.dheight; ++i) {
                denormalized[i] = data[i]*(_max_value-_min_value);
            }
            saveTIFF(std::string(ss.str()), denormalized.data(), prm_g.dwidth, prm_g.dheight);
        } else {
            std::ostringstream ss;
            ss << prm_r.proj_output << "\\" << "image" << "_" << std::setw(5) << std::setfill('0') << index << ".tif";
            saveTIFF(std::string(ss.str()), data, prm_g.dwidth, prm_g.dheight);
        }
        
    }

    /**
     * @brief Get the Image data from specified path
     * 
     * @param path to the file
     */
    std::vector<float> getImage(std::string path) {
        return loadTIFF(path, -1, -1);
    }

private:
    /**
     * @brief Construct a file path from a prefix and a number, poiting to the temp folder
     * 
     * @param prefix name of the file
     * @param index number placed after the prefix
     */
    std::string filename(std::string prefix, int64_t index) {
        return _temp_folder+prefix+std::to_string(index);
    }

    /**
     * @brief Construct a file path from a prefix and a number, poiting to the output folder
     * 
     * @param prefix name of the file
     * @param index number placed after the prefix, zero-padded with a constant size of five
     */
    std::string getOutputFilePath(std::string baseName, int64_t index) {
        std::ostringstream ss;
        ss << prm_r.output << "\\" << baseName << "_" << std::setw(5) << std::setfill('0') << index << ".tif";
        return std::string(ss.str());
    }

    /**
     * @brief Open a stream to a file located in the temporary folder, in truncate and write mode
     * 
     * @param prefix name of the file
     * @param id number placed after the prefix
     */
    std::ofstream openWriteTrunc(std::string prefix, int64_t id) {
        return std::ofstream(filename(prefix, id), std::ios::binary);
    }

    /**
     * @brief Open a stream to a file located in the temporary folder, in append and write mode
     * 
     * @param prefix name of the file
     * @param id number placed after the prefix
     */
    std::ofstream openWriteApp(std::string prefix, int64_t id) {
        return std::ofstream(filename(prefix, id), std::ios::binary || std::ios::app);
    }

    /**
     * @brief Open a stream to a file located in the temporary folder, in read-only mode
     * 
     * @param prefix name of the file
     * @param id number placed after the prefix
     */
    std::ifstream openRead(std::string prefix, int64_t id) {
        return std::ifstream(filename(prefix, id), std::ios::binary);
    }

    /**
     * @brief Open a stream to a file located in the temporary folder, truncate and write the content of data 
     * 
     * @param prefix name of the file
     * @param id number placed after the prefix
     * @param data content to write
     * @param size size of the data chunk that will be written, in byte
     */
    template<typename T>
    void writeTemp(std::string prefix, int64_t id, const T* data, size_t size) {
        auto file = openWriteTrunc(prefix, id);
        file.write((char*)data, size);
        file.close();
    }

    /**
     * @brief Open a stream to a file located in the temporary folder and read it 
     * 
     * @param prefix name of the file
     * @param id number placed after the prefix
     * @param data pre-allocated array where the data will be copied, with a size of at least 'size'
     * @param size size that will be read, in byte
     */
    template<typename T>
    void readTemp(std::string prefix, int64_t id, T* data, size_t size) {
        auto file = openRead(prefix, id);
        file.read((char*)data, size);
        file.close();
    }

    /**
     * @brief create a file of specified size
     * 
     * @param prefix name of the file
     * @param id number placed after the prefix
     * @param size size of the file, in byte
     */
    void allocateTempFile(std::string prefix, int64_t id, size_t size) {
        auto file = openWriteTrunc(prefix, id);
        size_t chunk_size = 64*1024*1024; //64Mo
        while(size > chunk_size) {
            std::vector<char> one_fill(chunk_size, 1);
            file.write(&one_fill[0], one_fill.size());
            size -= chunk_size;
        }
        std::vector<char> one_fill(size, 1);
        file.write(&one_fill[0], size);
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
};