/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#include "utils.cl"

kernel void ForwardH(   global const float* restrict volume, 
                        global fixed32* restrict sumImages,
                        global const ProjData* restrict projMat,
                        constant const ushort* restrict indexes,
                        constant const ushort* restrict img_indexes,

                        const uint angles,
                        const uint yOffset,
                        const float4 origin,
                        const float voxelSize,
                        const uint detectorWidth,
                        const uint detectorHeight,
                        const uint volumeWidth)
{
    const uint id = get_global_id(0);
    const uint4 iPosition3D = to3D(id, volumeWidth, volumeWidth)+(uint4)(0,yOffset,0,0);
    const float4 position3D = convert_float4(iPosition3D)*voxelSize + origin;
    position3D.w = 1;
    const fixed32 value = FLOAT_TO_FIXED(volume[id]);

    for(int i = 0; i < angles; ++i) {
        const ProjData _projMat = projMat[indexes[i]];
        const float2 position2D = mvm_xy_w(_projMat.mvp, position3D);
        const int2 coordinate2D = convert_int2(round(position2D));
        const bool bounds = all( isgreaterequal(convert_float2(coordinate2D), _projMat.vp.even) && isless(convert_float2(coordinate2D), _projMat.vp.odd));
        if(bounds) {
            atomic_add(sumImages+img_indexes[i]*detectorWidth*detectorHeight + coordinate2D.y*detectorWidth + coordinate2D.x, value);
        }
    }
}
