#ifndef GLOBAL
#define GLOBAL
#include <cuda.h>
#include <cuda_runtime.h>
#include "types.h"
// texture for kernel convolution with the input
// Texture reference defined at file scope
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> L_retinaInput;
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> M_retinaInput;
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> S_retinaInput;

surface<void, cudaSurfaceType2DLayered> LGNspikeSurface;

#endif
