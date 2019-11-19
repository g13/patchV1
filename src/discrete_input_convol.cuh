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
__device__
__inline__
void plane_to_retina(float x0, float y0, Float &x, Float &y) {
    Float r = sqrt(x0*x0 + y0*y0);
    Float atanr = -atan(r);
    Float xr = x0/r;
    Float yr = x0/r;
    x = xr*atanr;
    y = yr*atanr;
}

__device__
__inline__
void retina_to_plane(Float x0, Float y0, float &x, float &y) {
    Float r = sqrt(x0*x0 + y0*y0);
    Float tanr = -tan(r);
    Float xr = x0/r;
    Float yr = x0/r;
    x = xr*tanr;
    y = yr*tanr;
}

/*
 * 2D-block for spatial convolution and sum
 * loop to sum time convol
 * 1D-grid for different LGN
*/

__global__
__global__ void LGN_convol(_float* __restrict__ LGNfr,
                           LGN_parameter pLGN, // consider pointer
                           unsigned int iSample0,
                           _float samplePhase, unsigned int nKernelSample, _float kernelSampleDt, unsigned int nsig, unsigned int npixel_1D);

__global__ 
void LGN_nonlinear(_float* __restrict__ LGN_fr, static_nonlinear logistic, _float* __restrict__ max_convol);

__global__
__global__ void LGN_maxResponse(_float* __restrict__ max_convol,
                                LGN_parameter pLGN, // consider pointer
                                unsigned int nKernelSample, _float kernelSampleDt, unsigned int nsig, unsigned int npixel_1D);

#endif
