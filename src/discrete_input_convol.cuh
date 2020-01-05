//#include <cufft.h>
#ifndef DISCRETE_INPUT_CONVOL_CUH
#define DISCRETE_INPUT_CONVOL_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <tuple>
#include <math_functions.h>         //
#include "DIRECTIVE.h"
#include "LGN_props.cuh"
#include "types.h"
#include "util/cuda_util.cuh"

// assuming viewing distance is a single unit length
__global__ 
void LGN_nonlinear(
        Size nLGN,
        Static_nonlinear &logistic,
        Float* __restrict__ max_convol,
        Float* __restrict__ current_convol,
        Float* __restrict__ LGN_fr
);

__global__
void store(// weights and max convolution
        Float* __restrict__ max_convol,

        Temporal_component &temporal,
        Float* __restrict__ TW_storage,
        SmallSize nKernelSample,
        Float kernelSampleDt,
        Float kernelSampleT0,

        Spatial_component &spatial,
        Float* __restrict__ SW_storage,
        Float* __restrict__ SC_storage,
        Float* __restrict__ dwdh_storage,
		Size nLGN_L,
		Float L_x0,
		Float L_y0,
		Float R_x0,
		Float R_y0,
		Float normViewDistance,
        Float nsig // span of spatialRF sample in units of std
);

__global__ 
void LGN_convol_c1s(
        Float* __restrict__ decayIn,
        Float* __restrict__ lastF,
        Float* __restrict__ SW_storage,
        Float* __restrict__ SC_storage,
        Float* __restrict__ dwdh_storage,
        Float* __restrict__ TW_storage,
        Float* __restrict__ current_convol,
        SmallSize* __restrict__ coneType,
        Spatial_component &spatial,
        Float nsig,
		Size nLGN_L,
		Float L_x0,
		Float L_y0,
		Float R_x0,
		Float R_y0,
		Float normViewDistance,
        SmallSize currentFrame,
        SmallSize maxFrame,
		Float tPerFrame,
        Float framePhase,
        Float Itau,
        Float kernelSampleDt,
        Size nKernelSample,
        Float dt
);

__host__
__device__
__forceinline__
void retina_to_plane(Float cosp, Float sinp, Float ecc, float &x, float &y, const Float normViewDistance, Float LR_x0, Float LR_y0) {
    double r = tan(static_cast<double>(ecc))*static_cast<double>(normViewDistance);
    x = LR_x0 + static_cast<float>(r*cosp);
    y = LR_y0 + static_cast<float>(r*sinp);
}
#endif
