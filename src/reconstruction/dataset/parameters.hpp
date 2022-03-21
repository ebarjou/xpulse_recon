/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

#include "parameters/global_parameters.hpp"
#include "parameters/specialized_parameters.hpp"

#define prm_r _parameters->recon
#define prm_g _parameters->geometry
#define prm_d _parameters->detector
#define prm_md _parameters->detector.modules
#define prm_m2 _parameters->method2
#define prm_hm _parameters->hartmann
#define prm_ipr _parameters->ipr

struct Parameters {
    struct ReconstructionData recon;
    struct GeometryData geometry;
    struct DetectorData detector;
    struct ReconMethod2Data method2;
    struct ReconHartmannData hartmann;
    struct ReconIPRData ipr;
    JS_OBJ(recon, geometry, detector, method2, hartmann, ipr);
};

/**
 * @brief Fill and verify additionnal parameters fields
 * 
 * @param _parameters Parameter to fill
 */
void initializeParameters(Parameters *_parameters) {
    if(prm_m2.overlap_fct > 1.0f ) prm_m2.overlap_fct = 1.0f;
    if(prm_m2.overlap_fct < 0.01f ) prm_m2.overlap_fct = 0.01f;
    if(prm_g.heli_step < 1.0f) prm_g.heli_step = 1.0f;
    
    prm_g.vx = (prm_d.px*(prm_g.so/prm_g.sd));
    prm_d.rx *= prm_d.px;
    for(int64_t i = 0; i < int64_t(prm_md.size()); ++i) {
        prm_d.sx *= prm_d.px;
        prm_d.sy *= prm_d.px;
        
        if((!prm_md[i].start_x.assigned || !prm_md[i].end_x.assigned || !prm_md[i].start_y.assigned || !prm_md[i].end_y.assigned) && prm_md.size() > 1) {
            std::cout << "Warning : No viewport specified with more than one module." << std::endl;
        }
    }
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
    if(parameters.detector.modules.size() == 0) {
        throw std::exception("No detector module given");
    }

    initializeParameters(&parameters);

    return parameters;
}