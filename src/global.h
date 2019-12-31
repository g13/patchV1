#ifndef GLOBAL
#define GLOBAL
#include <cuda.h>
#include <cuda_runtime.h>
// texture for kernel convolution with the input
texture<float, cudaTextureType2DLayered> L_retinaConSig;
texture<float, cudaTextureType2DLayered> M_retinaConSig;
texture<float, cudaTextureType2DLayered> S_retinaConSig;

__device__ __constant__ float sqrt2 = 1.4142135623730951;

#endif
