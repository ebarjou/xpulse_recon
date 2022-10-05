/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */
#define WAVEFRONT_SIZE 32

typedef unsigned int fixed32;
#define FIXED_FRAC_BITS 24
#define FIXED_FRAC_MAX = (1 << (32-FIXED_FRAC_BITS))
#define FIXED_FRAC_ONE (1 << FIXED_FRAC_BITS)
#define FIXED_FRAC_ZERO ((fixed32)0)
#define FIXED_TO_FLOAT(x) (((float)(x)) / FIXED_FRAC_ONE)
#define FLOAT_TO_FIXED(x) ((fixed32)((x) * FIXED_FRAC_ONE))
#define EPSILON FIXED_TO_FLOAT(1)
#define LARGE_EPSILON FIXED_TO_FLOAT(1)

enum INDEX_KERNEL_ARG{  INDEX_VOLUME_BUFFER = 0,
                        INDEX_SUMIMAGE_BUFFER,
                        INDEX_MVP_BUFFER,
                        INDEX_VP_BUFFER,
                        INDEX_IMGINDEX_BUFFER,
                        INDEX_MVPINDEX_BUFFER,

                        INDEX_WEIGHT_F,
                        INDEX_ANGLES_U,
                        INDEX_ORIGIN_F4,
                        INDEX_VOXEL_SIZE_F,
                        INDEX_DETWIDTH_U,
                        INDEX_DETHEIGHT_U,
                        INDEX_VOLWIDTH_U };

#ifdef __OPENCL_C_VERSION__

#define SUM4(v) (v.x + v.y + v.z + v.w)
#define SUM3(v) (v.x + v.y + v.z)
#define MVM_XYW(m,v) (float2)(dot(m.s0123, v), dot(m.s4567, v))/dot(m.sCDEF, v)

kernel void Forward(    global const float* restrict volume, 
                        global fixed32* restrict sumImages,
                        global const float16* restrict mvpArray,
                        global const ushort4* restrict vpArray,
                        constant const ushort* restrict image_indexes,
                        constant const ushort* restrict mvp_indexes,
                        const float weight,
                        const uint angles,
                        const float4 origin,
                        const float voxelSize,
                        const uint detectorWidth,
                        const uint detectorHeight,
                        const uint volumeWidth)
{
    const uint4 id = (uint4)(get_global_id(0), get_global_id(2), get_global_id(1), 0);
    const uint3 offset = (uint3)(get_global_offset(0), get_global_offset(2), get_global_offset(1));
    if(id.x > volumeWidth || id.y > volumeWidth)
        return;
    const uint3 id_lin = (id.xyz-offset)*((uint3)(1, volumeWidth*volumeWidth, volumeWidth));
    const float4 position3D = convert_float4(id)*voxelSize + origin;
    position3D.w = 1;
    
    const fixed32 value = FLOAT_TO_FIXED(volume[SUM3(id_lin)]);

    for(int i = 0; i < angles; ++i) {
        const ushort mvp_index = mvp_indexes[i];
        const ushort image_index = image_indexes[i];
        const float16 mvp = mvpArray[mvp_index];
        const ushort4 vp = vpArray[mvp_index];
        const float2 position2D = MVM_XYW(mvp, position3D);
        const ushort2 coordinate2D = convert_ushort2(round(position2D));
        const bool bounds = all( (coordinate2D >= vp.even) && (coordinate2D < vp.odd));
        if(bounds) {
            atomic_add(sumImages+image_index*detectorWidth*detectorHeight + coordinate2D.y*detectorWidth + coordinate2D.x, value);
        }
    }
}

kernel void Backward(   global float* restrict volume, 
                        global const float* restrict sumImages,
                        global const float16* restrict mvpArray,
                        global const ushort4* restrict vpArray,
                        constant const ushort* restrict image_indexes,
                        constant const ushort* restrict mvp_indexes,
                        const float weight,
                        const uint angles,
                        const float4 origin,
                        const float voxelSize,
                        const uint detectorWidth,
                        const uint detectorHeight,
                        const uint volumeWidth)
{
    const uint4 id = (uint4)(get_global_id(0), get_global_id(2), get_global_id(1), 0);
    const uint3 offset = (uint3)(get_global_offset(0), get_global_offset(2), get_global_offset(1));
    if(id.x >= volumeWidth || id.y >= volumeWidth)
        return;
    const uint3 id_lin = (id.xyz-offset)*((uint3)(1, volumeWidth*volumeWidth, volumeWidth));
    const float4 position3D = convert_float4(id)*voxelSize + origin;
    position3D.w = 1;
    
    float sum = 0;
    uint contributions = 0;
    for(int i = 0; i < angles; ++i) {
        const ushort mvp_index = mvp_indexes[i];
        const ushort image_index = image_indexes[i];
        const float16 mvp = mvpArray[mvp_index];
        const ushort4 vp = vpArray[mvp_index];
        const float2 position2D = MVM_XYW(mvp, position3D);
        const ushort2 coordinate2D = convert_ushort2(round(position2D));
        const bool bounds = all((coordinate2D >= vp.even) && (coordinate2D < vp.odd));
        if(bounds) {
            const float value = sumImages[image_index*detectorWidth*detectorHeight + coordinate2D.y*detectorWidth + coordinate2D.x];
            if(!isnan(value)) {
                sum += value;
                ++contributions;
            }
        }
    }
    if(contributions > 0)
        volume[SUM3(id_lin)] *= pow(sum/contributions, weight);
}

#endif