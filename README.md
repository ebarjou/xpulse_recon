XPR - XPulse Recon
==================
Fast out-of-core and external memory-driven GPU-based CBCT reconstruction of large tomograms
--------------------------------------------------------------------------------------------

# Description
[Linked paper](url)

[XPulse Project](http://www.atlas-onco.com/xpulse.html)

# Build
This project can be build using CMake. It requires a working OpenCL environment.

## Cloning
```cmd
git clone --recurse-submodules https://github.com/ebarjou/xpulse_recon
cd xpulse_recon
```
## Build (Windows, MSVC)
```cmd
mkdir build
cd build
cmake ..
MSBuild.exe .\ALL_BUILD.vcxproj /property:Configuration=Release
```
## Build (Unix, make)
```cmd
mkdir build
cd build
cmake ..
make
```

# Usage
```cmd
XPRecon -template|<json_file>|<json>
        -template : Output in stdout a default JSON file
        <json_file> : path to the parameters file
        <json> : parameters as a JSON string (starting with a '{')
```
This program takes a stack of FP-32bits TIFF (uncompressed) as input. Start the program with a configuration file as argument to execute the reconstruction.
## Configuration file
The program takes a JSON file containing all necessary parameters for the reconstruction as the unique argument.

You can output the default JSON structure with the '-template' argument.

Here is a minimal example of a parameter file :
```json
{
    "recon" : {
        "input" : "./", //Path to the folder containing the input TIFF images
        "output" : "./recon/", //Path to the folder that will contain the reconstruction layers, as a stack of TIFF
        "method" : "naive", //Type of memory managment (naive|method_1a|method1b|method2)
        "it" : 1, //Number of main iterations
        "sit" : 10, //Number of sub-iterations
        "mlog" : true, //If true, apply '-log' to each images
        "weight" : 1.0, //Initial iteration weight
        "weight_factor" : 0.9, //Iteration weight factor, applied after each main iterations
        "images_preproc" : false //If true, normalize (and -log if necessary) all images before the reconstruction
    },
    "geometry" : {
        "so" : 100.0, //Source-object distance (mm)
        "sd" : 1000.0, //Source-detector distance (mm)
        "angle" : 360.0, //Total span of the tomographic acquisition (degrees)
    },
    "detector" : {
        "px" : 0.1, //Pixel size (mm)
        "rx" : 0.0, //Offset of the rotation axis on the detector, from the center (px)
        "sx" : 0.0, //X offset of the normal source projection from the detector center (px)
        "sy" : 0.0, //Y offset of the normal source projection from the detector center (px)
        "modules" : [
            {
                "roll" : 0.0, //Rotation of the detector along the projection axis
                "pitch" : 0.0, //Rotation of the detector along the Y axis
                "yaw" : 0.0 //Rotation of the detector along the X axis
            }
        ]
    },
}
```

# Dependencies
This repository depends on four sub-modules : 
* [Json_struct](https://github.com/jorgen/json_struct) : Reading JSON parameter file
* [GLM](https://github.com/g-truc/glm) : Linear algebra
* [OpenCL-SDK](https://github.com/KhronosGroup/OpenCL-SDK) : OpenCL Binding
* [TinyTIFF](https://github.com/jkriege2/TinyTIFF) : TIFF image reading and writing
* [PocketFFT](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) : FFT for experimental Iterative FSP reconstruction
* 
The licenses can be found in [EXTERNAL_LICENSES.md](https://github.com/ebarjou/xpulse_recon/blob/master/EXTERNAL_LICENSES.md)