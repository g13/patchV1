#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H
#include <cuda.h>
#include "../DIRECTIVE.h"
#include "../CUDA_MACRO.h"
#include "../types.h"

// 1D
template <typename T>
__global__ void init(T *array, T value, PosInt nData) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nData) {
        array[id] = value;
    }
}

__device__ void warp0_min(Float array[], PosInt id[]);

__device__ void warps_min(Float array[], Float data, PosInt id[]);

__device__ void find_min(Float array[], Float data, PosInt id[]);

// only reduce is extensively tested: cuda_full_min.cu

template <typename T>
__device__ void warps_reduce(T array[], T data) {
    PosInt tid = blockDim.x*threadIdx.y + threadIdx.x;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(FULL_MASK, data, offset);
        //if (tid % warpSize == 0) {
        //    printf("#%i at %i: %f\n", tid, offset, data);
        //}
    }
    __syncthreads();
    if (tid % warpSize == 0) {
        array[tid/warpSize] = data;
    }
}

template <typename T>
__device__ void warp0_reduce(T array[]) {
    PosInt tid = blockDim.x*threadIdx.y + threadIdx.x;
    T data = array[tid];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(FULL_MASK, data, offset);
        //if (tid == 0) {
        //    printf("##%i at %i: %f\n", tid, offset, data);
        //}
    }
    if (tid == 0) {
        array[0] = data;
    }
}

template <typename T>
__device__ void block_reduce(T array[], T data) {
	warps_reduce<T>(array, data);
    __syncthreads();
    //if (blockDim.x*threadIdx.y + threadIdx.x < nWarp) {
    if (blockDim.x*threadIdx.y + threadIdx.x < warpSize) {
        warp0_reduce<T>(array);
    }
    __syncthreads();
}

#endif
