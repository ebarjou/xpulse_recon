/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#include "utils.cl"

kernel void Backward(   global float* restrict volume, 
                        global const float* restrict sumImages,
                        //constant const ProjData* restrict projData,
                        global const ProjData* restrict projData,
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
    const float value = volume[id];
    float sum = 0;
    uint contributions = 0;

    for(uint i = 0, m = 0; m < angleNumber*modulePerAngle; ++m, i = m/modulePerAngle) {
        const ProjData _projData = projData[m];
        //const float2 position2D = min(mvm_xy_w(_projData.mvp, position3D)-(float2)(0, lineOffsetY.x), (float2)(MAXFLOAT, lineOffsetY.y-1));
        //TODO: check ici et forward pour limites chunk
        const float2 position2D = mvm_xy_w(_projData.mvp, position3D)-(float2)(0, lineOffsetY.x);
        const int2 coordinate2D = convert_int2(round(position2D));
        
        const bool bounds = all( isgreaterequal(convert_float2(coordinate2D), _projData.vp.even) && isless(convert_float2(coordinate2D), _projData.vp.odd));
        sum += bounds?sumImages[IMAGE_OFFSET_2D(i, coordinate2D)]:0;
        contributions += bounds==true;
    }
    volume[id] = value * exp(sum/contributions);
}