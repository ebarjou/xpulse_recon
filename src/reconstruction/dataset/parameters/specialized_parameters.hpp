/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#pragma once

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

typedef struct {
    int64_t vOffset, vSize;
    int64_t iOffset, iSize;
    int64_t vPadTop, vPadBot;
} Chunk;

struct ReconMethod2Data {
    float overlap_fct = 1.0f;

    std::vector<Chunk> chunks;

    JS_OBJ(overlap_fct);
};