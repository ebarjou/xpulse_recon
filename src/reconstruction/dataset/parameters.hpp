/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

struct ReconstructionData {
    std::string input = "./";
    std::string output = "./recon/";
    std::string proj_output = "";
    Method method = Method::abs;
    
    int64_t it = 1, sit = 10;
    bool mlog = true;
    float weight = 1.0f, weight_factor = 0.9f;
    bool normalize = false;
    bool simulation = false;
    int64_t usable_ram_go = 12;

    JS_OBJ(input, output, proj_output, method, it, sit, mlog, weight, weight_factor, normalize, simulation, usable_ram_go);
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
    int64_t module_number = 1;
    ::JS::OptionalChecked<float> module_angle_offset;
    float module_center_offset_z = 0.0f;

    JS_OBJ(px, rx, sx, sy, module_number, module_angle_offset, module_center_offset_z);
};

struct ReconHartmannData {
    std::string devx = "./";
    std::string devy = "./";
    float iterations = 0;
    float iterations_fct = 1;
    int64_t iterations_step = 1;
    MethodPhase method = MethodPhase::gs;

    JS_OBJ(devx, devy, iterations, iterations_fct, iterations_step, method);
};

struct ReconIPRData {
    float iterations = 0;
    float iterations_fct = 1;
    int64_t iterations_step = 1;
    float energy_kev = 1;

    JS_OBJ(iterations, iterations_fct, iterations_step, energy_kev);
};

#define prm_r _parameters->recon
#define prm_g _parameters->geometry
#define prm_d _parameters->detector
#define prm_hm _parameters->hartmann
#define prm_ipr _parameters->ipr

struct Parameters {
    struct ReconstructionData recon;
    struct GeometryData geometry;
    struct DetectorData detector;
    struct ReconHartmannData hartmann;
    struct ReconIPRData ipr;
    JS_OBJ(recon, geometry, detector, hartmann, ipr);
};

/**
 * @brief Fill and verify additionnal parameters fields
 * 
 * @param _parameters Parameter to fill
 */
void initializeParameters(Parameters *_parameters) {
    if(prm_g.heli_step < 1.0f) prm_g.heli_step = 1.0f;

    prm_d.rx *= prm_d.px;
}

/**
 * @brief Construst a Parameters struct from JSON data
 * 
 * @param file_or_content either the path to a json file, or directly json data (detected if it start with a "{")
 * @return Parameters Parameters struct 
 */
Parameters loadParameters(std::string file_or_content) {
    Parameters parameters;

    if(file_or_content[0] != '{') { // Read arg as file; If argument start with "{", assume it's json data and parse it
        std::ifstream t(file_or_content);
        std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        if(str.empty()) {
            throw std::exception("Can not read the parameters file");
        }
        JS::ParseContext context(str.c_str());
        if (context.parseTo(parameters) != JS::Error::NoError) {
            std::cout << context.makeErrorString();
            throw std::exception("Error while parsing the parameter file");
        }
    } else { // Load the file content and parse it
        JS::ParseContext context(file_or_content);
        if (context.parseTo(parameters) != JS::Error::NoError) {
            std::cout << context.makeErrorString();
            throw std::exception("Error while parsing the parameter string");
        }
    }

    initializeParameters(&parameters);

    return parameters;
}