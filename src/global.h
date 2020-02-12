#ifndef GLOBAL
#define GLOBAL
#include <cuda.h>
#include <cuda_runtime.h>
// texture for kernel convolution with the input
// Texture reference defined at file scope
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> L_retinaInput;
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> M_retinaInput;
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> S_retinaInput;

__device__ __constant__ float sqrt2 = 1.4142135623730951;

#endif
