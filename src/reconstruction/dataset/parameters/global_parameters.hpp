/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

struct ReconstructionData {
    std::string input = "./";
    std::string output = "./recon/";
    std::string proj_output = "";
    Method method = Method::method_1b;
    
    int64_t it = 1, sit = 10;
    bool mlog = true;
    float weight = 1.0f, weight_factor = 0.9f;
    bool images_preproc = false;

    JS_OBJ(input, output, proj_output, method, it, sit, mlog, weight, weight_factor, images_preproc);
};

struct GeometryData {
    float so = 100.0f, sd = 1000.0f, angle = 360.0f;
    float heli_offset = 0.0f, heli_step = 1.0f;
    std::vector<float> angle_list;

    float vx;
    int64_t dwidth, dheight;
    int64_t vwidth, vheight;
    int64_t projections, concurrent_projections;
    glm::vec3 orig;
    std::vector<std::vector<float>> projection_matrices;

    JS_OBJ(so, sd, angle, heli_offset, heli_step, angle_list);
};

struct DetectorModuleData {
    float offset_x = 0.0f, offset_y = 0.0f, offset_z = 0.0f;
    float roll = 0.0f, pitch = 0.0f, yaw = 0.0f;
    ::JS::OptionalChecked<int64_t> start_x, end_x, start_y, end_y;

    JS_OBJ(offset_x, offset_y, offset_z, roll, pitch, yaw, start_x, end_x, start_y, end_y);
};

struct DetectorData {
    float px = 0.1f, rx = 0.0f, sx = 0.0f, sy = 0.0f;
    std::vector<DetectorModuleData> modules{DetectorModuleData()};

    JS_OBJ(px, rx, sx, sy, modules);
};

