/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#define CL_HPP_TARGET_OPENCL_VERSION 300

/**
 * Ask the GPU driver to enable the high-performance GPU if available.
 */
#ifdef __cplusplus
extern "C" {
#endif
__declspec(dllexport) int NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#ifdef __cplusplus
}
#endif

#define RETURN_CODE_OK 0
#define RETURN_CODE_ERROR_ARGUMENTS -1
#define RETURN_CODE_ERROR_PARAMETERS -2
#define RETURN_CODE_ERROR_RECONSTRUCTION -3

#include <CL/opencl.hpp>
#include <glm.hpp>
#include <gtx/rotate_vector.hpp>
#include <gtc/type_ptr.hpp>
#include <gtx/range.hpp>
#include <tinytiffreader.h>
#include <tinytiffwriter.h>
#include <omp.h>
#include <json_struct.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <fstream>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <sstream>
#include <future>

//Should be in parameters.hpp, but json_struct does not handle namepaces nicely
JS_ENUM(Method, naive, method_1a, method_1b, method_2, hartmann, ipr);
JS_ENUM_DECLARE_STRING_PARSER(Method);

JS_ENUM(MethodPhase, jacobi, gs, sor, simpson);
JS_ENUM_DECLARE_STRING_PARSER(MethodPhase);

#include "reconstruction.hpp"

void usage() {
    std::cout << "./hr_tomorecon -template|<json_file>|<json>" << std::endl;
    std::cout << "\t" << "-template : Output in stdout a default JSON file" << std::endl;
    std::cout << "\t" << "<json_file> : path to the parameters file" << std::endl;
    std::cout << "\t" << "<json> : parameters as a JSON string (starting with a '{')" << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc < 2) {
        usage();
        return RETURN_CODE_ERROR_ARGUMENTS;
    }
    //If -template is given, generate the default JSON and print it to stdout
    if(std::string(argv[1]) == "-template") {
        reconstruction::dataset::Parameters parameters;
        std::cout << JS::serializeStruct(parameters) << std::endl;
        return RETURN_CODE_OK;
    }

    //Parse parameters
    reconstruction::dataset::Parameters parameters;
    try {
        parameters = reconstruction::dataset::loadParameters(argv[1]);
    } catch(std::runtime_error err) {
        std::cerr << "Error while creating the parameters object : " << err.what() << std::endl;
        return RETURN_CODE_ERROR_PARAMETERS;
    }

    //Choose the reconstruction algorithm
    reconstruction::Reconstruction *recon;
    try {
        switch (parameters.recon.method) {
            case Method::naive:
                recon = new reconstruction::ReconstructionN(&parameters);
                break;
            case Method::method_1a:
                recon = new reconstruction::ReconstructionLPLN(&parameters);
                break;
            case Method::method_1b:
                recon = new reconstruction::ReconstructionLPLA(&parameters);
                break;
            case Method::method_2:
                recon = new reconstruction::ReconstructionChunk(&parameters);
                break;
            case Method::hartmann:
                recon = new reconstruction::ReconstructionHartmann(&parameters);
                break;
            case Method::ipr:
                recon = new reconstruction::ReconstructionIPR(&parameters);
                break;
            default:
                break;
        }
    } catch(std::runtime_error err) {
        std::cerr << "Error while creating the reconstruction object : " << err.what() << std::endl;
        return RETURN_CODE_ERROR_RECONSTRUCTION;
    }

    //Execute the reconsturction
    recon->exec();
    
    //If needed, execute the forward projection to save the proejction of the volume
    if(!parameters.recon.proj_output.empty()) {
        recon->forward_exec();
    }

    //Clear reconstruction data
    delete recon;

    return RETURN_CODE_OK;
}