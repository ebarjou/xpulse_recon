R"(
/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

/**
 * ©2020-2022 ALPhANOV (https://www.alphanov.com/) All Rights Reserved
 * Author: Barjou Emile
 */

#define WAVEFRONT_SIZE 32

#ifdef __cplusplus
typedef uint32_t uint;
typedef struct {float f[4];} float4;
#define fixed32 uint32_t
#else
typedef uint fixed32;
#endif


#define FIXED_FRAC_BITS 20
#define FIXED_FRAC_MAX = (1 << (32-FIXED_FRAC_BITS))
#define FIXED_FRAC_ONE (1 << FIXED_FRAC_BITS)
#define FIXED_FRAC_ZERO ((fixed32)0)
#define FIXED_TO_FLOAT(x) ((float)(x) / FIXED_FRAC_ONE)
#define FLOAT_TO_FIXED(x) ((fixed32)round((x) * FIXED_FRAC_ONE))
#define EPSILON FIXED_TO_FLOAT(1)

typedef struct {
    float4 mvp[4];
    float4 vp;
} ProjData;

enum INDEX_FORWARD{ INDEX_FORWARD_VOLUME_BUFFER = 0,
                    INDEX_FORWARD_SUM_IMAGE_BUFFER, 
                    INDEX_FORWARD_PROJDATA_BUFFER,
                    INDEX_FORWARD_SIZE_U4,
                    INDEX_FORWARD_ORIGIN_F4,
                    INDEX_FORWARD_ANGLE_NUMBER_U,
                    INDEX_FORWARD_MODULE_PER_ANGLE_U,
                    INDEX_FORWARD_VOLUME_MIN_I3,
                    INDEX_FORWARD_VOLUME_MAX_I3,
                    INDEX_FORWARD_LINE_OFFSET_I2,
                    INDEX_FORWARD_VOXEL_SIZE_F4,
            };

enum INDEX_BACKWARD{INDEX_BACKWARD_VOLUME_BUFFER = 0, 
                    INDEX_BACKWARD_SUM_IMAGE_BUFFER, 
                    INDEX_BACKWARD_PROJDATA_BUFFER, 
                    INDEX_BACKWARD_SIZE_U4,
                    INDEX_BACKWARD_ORIGIN_F4,
                    INDEX_BACKWARD_ANGLE_NUMBER_U,
                    INDEX_BACKWARD_MODULE_PER_ANGLE_U,
                    INDEX_BACKWARD_VOLUME_MIN_I3,
                    INDEX_BACKWARD_VOLUME_MAX_I3,
                    INDEX_BACKWARD_LINE_OFFSET_I2,
                    INDEX_BACKWARD_VOXEL_SIZE_F4,
                    };

enum INDEX_ERROR{   INDEX_ERROR_IMAGE_BUFFER = 0,
                    INDEX_ERROR_SUM_IMAGE_BUFFER,
                    INDEX_ERROR_SIZE_U4,
                    INDEX_ERROR_ELEMENTS_U,
                    INDEX_ERROR_WEIGHT_F, };

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

)"