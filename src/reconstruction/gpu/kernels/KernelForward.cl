/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#include "utils.cl"

#define INTERP_ROUND

kernel void Forward(global const float* restrict volume, 
                    global fixed32* restrict sumImages,
                    constant const ProjData* restrict projData,
                    const uint4 size, //Size of the total volume, in voxel
                    const float4 origin, //Origin of the volume, in mm
                    const uint angleNumber,
                    const uint modulePerAngle,
                    const uint3 volumeMin, // | Bounding box of the current volume in memory, in voxel
                    const uint3 volumeMax, // |
                    const uint2 lineOffsetY,
                    const float4 voxelSize)
{
    const uint id = get_global_id(0);
    const uint3 volumeSize = volumeMax - volumeMin;
    const uint4 id3d = to3D(id, volumeSize.x, volumeSize.z)+(uint4)(volumeMin,0);
    const float4 position3D = convert_float4(id3d)*voxelSize + origin;
    const fixed32 value = FLOAT_TO_FIXED(volume[id]);

    for(uint i = 0, m = 0; m < angleNumber*modulePerAngle; ++m, i = m/modulePerAngle) {
        const ProjData _projData = projData[m];
        const float2 position2D = min(mvm_xy_w(_projData.mvp, position3D)-(float2)(0, lineOffsetY.x), (float2)(MAXFLOAT, lineOffsetY.y-1));
        #ifdef INTERP_ROUND
            const int2 coordinate2D = convert_int2(round(position2D));
            const bool bounds = all( isgreaterequal(convert_float2(coordinate2D), _projData.vp.even) && isless(convert_float2(coordinate2D), _projData.vp.odd) );
            if(bounds) { atomic_add(sumImages+IMAGE_OFFSET_2D(i, coordinate2D), value); }
        #endif
        #ifdef INTERP_BILIN
            const float2 coordinate2D = position2D;
            const bool bounds = coordinate2D.x >= 0 && coordinate2D.x < size.x-1 && coordinate2D.y >= 0 && coordinate2D.y < size.y-1;
            //TODO : bounds comme au dessus
            if(bounds) {
                float wx1 = coordinate2D.x - (int)(coordinate2D.x);
                float wx0 = 1.0f - wx1;
                float wy1 = coordinate2D.y - (int)(coordinate2D.y);
                float wy0 = 1.0f - wy1;

                atomic_add(sumImages+IMAGE_OFFSET_2D(i, convert_int2(coord) + (int2)(0,0)), FLOAT_TO_FIXED(FIXED_TO_FLOAT(value)*wx0*wy0) );
                atomic_add(sumImages+IMAGE_OFFSET_2D(i, convert_int2(coord) + (int2)(0,1)), FLOAT_TO_FIXED(FIXED_TO_FLOAT(value)*wx0*wy1) );
                atomic_add(sumImages+IMAGE_OFFSET_2D(i, convert_int2(coord) + (int2)(1,0)), FLOAT_TO_FIXED(FIXED_TO_FLOAT(value)*wx1*wy0) );
                atomic_add(sumImages+IMAGE_OFFSET_2D(i, convert_int2(coord) + (int2)(1,1)), FLOAT_TO_FIXED(FIXED_TO_FLOAT(value)*wx1*wy1) );
            }
        #endif
    }
}
