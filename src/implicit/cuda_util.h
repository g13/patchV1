#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include "DIRECTIVE.h"
#include "CUDA_MACRO.h"

template <typename T>
__global__ void init(T *array, T value) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    array[id] = value;
}

__device__ void warp0_min(_float array[], unsigned int id[]);

__device__ void warps_min(_float array[], _float data, unsigned int id[]);

__device__ void find_min(_float array[], _float data, unsigned int id[]);

__device__ void warps_reduce(unsigned int array[], unsigned int data);

__device__ void warp0_reduce(unsigned int array[]);

__device__ void block_reduce(unsigned int array[], unsigned int data);

#endif
