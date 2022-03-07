/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#include "constants.cl"

#define start(v) v.x
#define end(v) v.y
#define step(v) v.z
#define group(v) v.w

#define r0(m) m.lo.lo
#define r1(m) m.lo.hi
#define r2(m) m.hi.lo
#define r3(m) m.hi.hi
#define c0(m) m.even.even
#define c1(m) m.odd.even
#define c2(m) m.even.odd
#define c3(m) m.odd.odd

inline float sum_f4(float4 v) { return v.x + v.y + v.z + v.w; }
inline float uv3prod(uint3 v) { return v.x * v.y * v.z; }

#define mvm(m,v) (float4)(dot(m[0]*v), dot(m[1]*v), dot(m[2]*v), dot(m[3]*v))
#define mvm_xy_w(m,v) (float2)(dot(m[0], v), dot(m[1], v))/dot(m[3], v)

static inline float travelTime(float3 orig, float3 dir, float3 aabbMin, float3 aabbMax) {
    float3 invRay = 1.0f/dir;
    float3 v1 = (aabbMin - orig) * invRay;
    float3 v2 = (aabbMax - orig) * invRay;
    float3 n = min(v1, v2);
    float3 f = max(v1, v2);
    float enter = max(n.x, max(n.y, n.z));
    float exit = min(f.x, min(f.y, f.z));
    return (exit > 0.0f && enter < exit)?exit-enter:MAXFLOAT;
}

uint4 to3D(uint id, uint size0, uint size1) {
    uint z = id / (size0 * size1);
    id -= (z * size0 * size1);
    uint y = id / size0;
    uint x = id % size0;
    return (uint4)(x, z, y, 1);
}

#define IMAGE_OFFSET_2D(id, cd) (id*(size.x*lineOffsetY.y)+(cd).y*size.x+(cd).x)
#define IMAGE_OFFSET_1D(id, cd) (id*(size.x*size.y)+cd)
