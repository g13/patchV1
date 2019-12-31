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
        Static_nonlinear &logistic,
        Float* __restrict__ max_convol,
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
        Float* __restrict__ dxdy_storage,
        Float nsig, // span of spatialRF sample in units of std
        bool storeSpatial
);

__global__ 
void LGN_convol_c1s(
        Float* __restrict__ decayIn,
        Float* __restrict__ lastF,
        Float* __restrict__ SW_storage,
        Float* __restrict__ SC_storage,
        Float* __restrict__ dxdy_storage,
        Float* __restrict__ TW_storage,
        Float* __restrict__ LGNfr,
        SmallSize* __restrict__ coneType,
        Spatial_component &spatial,
        Float nsig,
        SmallSize currentFrame,
        SmallSize maxFrame,
		Float tPerFrame,
        Float framePhase,
        Float Itau,
        Float kernelSampleDt,
        Size nKernelSample,
        Float dt,
        bool spatialStored
);
#endif
