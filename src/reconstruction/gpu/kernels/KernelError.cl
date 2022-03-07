/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#include "utils.cl"

kernel void Error(  global const float* restrict images, 
                    global fixed32* restrict sumImages,
                    const uint4 size, //Size of the total volume, in voxel
                    const uint elements,
                    const float weight)
{
    const uint id = get_global_id(0);
    if(id >= elements) {
        return;
    }
    const float ref = images[id];
    const float got = FIXED_TO_FLOAT(sumImages[id]);
    union { float f; fixed32 v; } result;
    result.f = log(max(ref/max(got,EPSILON), EPSILON))*weight;
    sumImages[id] = result.v;
}
