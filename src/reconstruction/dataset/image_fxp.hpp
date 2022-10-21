/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

struct ImageFXP {
    const static uint16_t MAX_STORAGE_VALUE = 65534;
    std::valarray<uint16_t> content;
    float value_scaling;
    int64_t width, height;

    ImageFXP() : content(), value_scaling(0), width(0), height(0) {};
    
    ImageFXP(std::valarray<float> &float_content, int64_t width, int64_t height)
             : width(width), height(height) {
        float max_value = getMaxOf(&float_content[0], width*height);
        value_scaling = max_value/MAX_STORAGE_VALUE;
        content = std::valarray<uint16_t>(width*height);
        for(int i = 0; i < width*height; ++i) {
            content[i] = uint16_t(float_content[i]/value_scaling);
        }
    }

    ImageFXP(float *float_content, int64_t width, int64_t height)
             : width(width), height(height) {
        float max_value = getMaxOf(&float_content[0], width*height);
        value_scaling = max_value/MAX_STORAGE_VALUE;
        content = std::valarray<uint16_t>(width*height);
        for(int i = 0; i < width*height; ++i) {
            content[i] = std::max(uint16_t(float_content[i]/value_scaling), uint16_t(1));
        }
    }

    ImageFXP(std::valarray<uint16_t> &coded_content, int64_t width, int64_t height)
             : width(width), height(height) {
        content = std::valarray<uint16_t>(&coded_content[0], width*height);
        float* values_tag = (float*)&content[0];
        value_scaling = values_tag[0];
        values_tag[0] = 0.0;
    }

    float getMaxOf(float *data, int64_t size) {
        float max_value = std::numeric_limits<float>::min();
        for(int i = 0; i < size; ++i) {
            if(std::isfinite(data[i])) {
                max_value = std::max(max_value, data[i]);
            }
        }
        return max_value;
    }

    std::valarray<float> getFloatContent() {
        std::valarray<float> float_content(content.size());
        for(int i = 0; i < content.size(); ++i) {
            float_content[i] = float(content[i])*value_scaling;
        }
        return float_content;
    }

    void getFloatContent(float* dst) {
        for(int i = 0; i < content.size(); ++i) {
            dst[i] = float(content[i])*value_scaling;
        }
    }

    std::valarray<uint16_t> getCodedContent() {
        std::valarray<uint16_t> coded_content(&content[0], width*height);
        float* values_tag = (float*)&coded_content[0];
        values_tag[0] = value_scaling;
        return coded_content;
    }

    uint16_t* data() {
        return &content[0];
    }

    int64_t size() {
        return int64_t(content.size());
    }

    bool is_valid() {
        return content.size() == width*height && content.size() > 0;
    }
};