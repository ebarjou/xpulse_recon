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


#define FIXED_FRAC_BITS 22
#define FIXED_FRAC_MAX = (1 << (32-FIXED_FRAC_BITS))
#define FIXED_FRAC_ONE (1 << FIXED_FRAC_BITS)
#define FIXED_FRAC_ZERO ((fixed32)0)
#define FIXED_TO_FLOAT(x) (((float)(x)) / FIXED_FRAC_ONE)
#define FLOAT_TO_FIXED(x) ((fixed32)((x) * FIXED_FRAC_ONE))
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

enum INDEX_HFORWARD{    INDEX_HFORWARD_VOLUME_BUFFER = 0, 
                        INDEX_HFORWARD_SUMIMAGE_BUFFER,
                        INDEX_HFORWARD_PROJDATA_BUFFER,
                        INDEX_HFORWARD_INDEX_BUFFER,
                        INDEX_HFORWARD_IMGINDEX_BUFFER,

                        INDEX_HFORWARD_ANGLES_U,
                        INDEX_HFORWARD_YOFFSET_U,
                        INDEX_HFORWARD_ORIGIN_F4,
                        INDEX_HFORWARD_VOXEL_SIZE_F,
                        INDEX_HFORWARD_DETWIDTH_U,
                        INDEX_HFORWARD_DETHEIGHT_U,
                        INDEX_HFORWARD_VOLWIDTH_U};

enum INDEX_HBACKWARD{   INDEX_HBACKWARD_VOLUME_BUFFER = 0, 
                        INDEX_HBACKWARD_SUMIMAGE_BUFFER,
                        INDEX_HBACKWARD_PROJDATA_BUFFER,
                        INDEX_HBACKWARD_INDEX_BUFFER,
                        INDEX_HBACKWARD_IMGINDEX_BUFFER,

                        INDEX_HBACKWARD_ANGLES_U,
                        INDEX_HBACKWARD_YOFFSET_U,
                        INDEX_HBACKWARD_ORIGIN_F4,
                        INDEX_HBACKWARD_VOXEL_SIZE_F,
                        INDEX_HBACKWARD_DETWIDTH_U,
                        INDEX_HBACKWARD_DETHEIGHT_U,
                        INDEX_HBACKWARD_VOLWIDTH_U};