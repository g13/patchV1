//#include <cufft.h>
#ifndef DISCRETE_INPUT_H
#define DISCRETE_INPUT_H

#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <math_functions.h>         //
#include "DIRECTIVE.h"
#include "cuda_util.h"
#include "LGN_props.cuh"

extern texture<float, cudaTextureType2DLayered> L_retinaConSig;
extern texture<float, cudaTextureType2DLayered> M_retinaConSig;
extern texture<float, cudaTextureType2DLayered> S_retinaConSig;
extern const float sqrt2;

#ifdef SINGLE_PRECISION
	#define square_root sqrtf
	#define atan atan2f
	#define uniform curand_uniform
	#define expp expf
	#define power powf
	#define abs fabsf 
    #define copy copysignf
#else
	#define square_root sqrt
	#define atan atan2
	#define uniform curand_uniform_double
	#define expp exp 
	#define power pow
	#define abs fabs 
    #define copy copysign
#endif

// assuming viewing distance is a single unit length
__global__ 
void LGN_nonlinear(
        Static_nonlinear &logistic,
        Float* __restrict__ max_convol,
        Float* __restrict__ LGN_fr
);

__global__
void store_weight(
        Temporal_component* __restrict__ temporal,
        Float* __restrict__ TW_storage,
        SmallSize nKernelSample,
        Float kernelSampleDt,
        Float kernelSampleT0,

        Spatial_component* __restrict__ spatial,
        Float* __restrict__ SW_storage,
        Float* __restrict__ SC_storage,
        Float* __restrict__ dxdy_storage,
        Float nsig, // span of spatialRF sample in units of std
        SmallSize nSpatialSample_1D, 
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
        PosInt nsig,
        Float framePhase,
        Float ave_tau,
        Float kernelSampleDt,
        Size nKernelSample,
        Float dt,
        bool spatialStored
);

#endif
