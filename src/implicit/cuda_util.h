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

template <typename T>
__device__ void warps_reduce(T array[], T data) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(FULL_MASK, data, offset);
    }
    __syncthreads();
    if (threadIdx.x % warpSize == 0) {
        array[threadIdx.x/warpSize] = data;
    }
}

template <typename T>
__device__ void warp0_reduce(T array[]) {
    T data = array[threadIdx.x];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(FULL_MASK, data, offset);
    }
    if (threadIdx.x == 0) {
        array[0] = data;
    }
}

template <typename T>
__device__ void block_reduce(T array[], T data) {
	warps_reduce<T>(array, data);
    __syncthreads();
    if (threadIdx.x < warpSize) {
        warp0_reduce<T>(array);
    }
    __syncthreads();
}

#endif
