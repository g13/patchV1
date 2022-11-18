#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H
#include <stdio.h>
#include <cuda.h>
#include "../CUDA_MACRO.h"
#include "../types.h"


// 1D
template <typename T>
__global__ void init(T *array, T value, PosInt nData) {
    PosIntL id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nData) {
        array[id] = value;
    }
}

__device__ void find_min(Float array[], Float data, PosInt id[], Size n);

// only reduce is extensively tested: cuda_full_min.cu

template <typename T>
__device__ void warps_reduce(T array[], T data, PosInt tid, Size n) {
    Size width = WARP_SIZE; // ceil power of 2
    Size iWarp = tid/WARP_SIZE;
    if (n % WARP_SIZE != 0 && iWarp == n/WARP_SIZE) {
        // get the ceil power of 2 for the last warp
        n = n % WARP_SIZE; // remaining element in the last warp
        //Size width = 2;
        width = 2;
        n--;
        while(n >>= 1) width <<=1;
    }
    for (int offset = width/2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(FULL_MASK, data, offset, width);
    }
    __syncthreads();
    if (tid % WARP_SIZE == 0) {
        array[tid/WARP_SIZE] = data;
    }
}

template <typename T>
__device__ void warp0_reduce(T array[], PosInt tid, Size n) {
    Size width = 2;
    Size m = n-1;
    while (m >>= 1) width <<= 1;
	unsigned MASK = __ballot_sync(FULL_MASK, tid < n);
	if (tid < n) {
        T data = array[tid];
	    for (int offset = width / 2; offset > 0; offset /= 2) {
	    	data += __shfl_down_sync(MASK, data, offset, width);
	    }
	    if (tid == 0) {
	    	array[0] = data;
	    }
    }
}

template <typename T>
__device__ void block_reduce(T array[], T data) {
    PosInt tid = blockDim.x*(blockDim.y*threadIdx.z + threadIdx.y) + threadIdx.x;
    Size n = blockDim.x*blockDim.y*blockDim.z;
	warps_reduce<T>(array, data, tid, n);
    __syncthreads();
    n = (n + WARP_SIZE-1)/WARP_SIZE;
    if (tid < WARP_SIZE && n > 1) {
        warp0_reduce<T>(array, tid, n);
    }
    __syncthreads();
}

template <typename T>
__device__ void warps_reduce2(T array1[], T array2[], T data[], PosInt tid, Size n) {
    Size width = WARP_SIZE; // ceil power of 2
    Size iWarp = tid/WARP_SIZE;
    if (n % WARP_SIZE != 0 && iWarp == n/WARP_SIZE) {
        // get the ceil power of 2 for the last warp
        n = n % WARP_SIZE; // remaining element in the last warp
        //Size width = 2;
        width = 2;
        n--;
        while(n >>= 1) width <<=1;
    }
    for (int offset = width/2; offset > 0; offset /= 2) {
        data[0] += __shfl_down_sync(FULL_MASK, data[0], offset, width);
        data[1] += __shfl_down_sync(FULL_MASK, data[1], offset, width);
    }
    __syncthreads();
    if (tid % WARP_SIZE == 0) {
        array1[tid/WARP_SIZE] = data[0];
        array2[tid/WARP_SIZE] = data[1];
    }
}

template <typename T>
__device__ void warp0_reduce2(T array1[], T array2[], PosInt tid, Size n) {
    Size width = 2;
    Size m = n-1;
    while (m >>= 1) width <<= 1;
	unsigned MASK = __ballot_sync(FULL_MASK, tid < n);
	if (tid < n) {
        T data1 = array1[tid];
        T data2 = array2[tid];
	    for (int offset = width / 2; offset > 0; offset /= 2) {
	    	data1 += __shfl_down_sync(MASK, data1, offset, width);
	    	data2 += __shfl_down_sync(MASK, data2, offset, width);
	    }
	    if (tid == 0) {
	    	array1[0] = data1;
	    	array2[0] = data2;
	    }
    }
}

template <typename T>
__device__ void block_reduce2(T array1[], T array2[], T data[]) {
    PosInt tid = blockDim.x*(blockDim.y*threadIdx.z + threadIdx.y) + threadIdx.x;
    Size n = blockDim.x*blockDim.y*blockDim.z;
	warps_reduce2<T>(array1, array2, data, tid, n);
    __syncthreads();
    n = (n + WARP_SIZE-1)/WARP_SIZE;
    if (tid < WARP_SIZE && n > 1) {
        warp0_reduce2<T>(array1, array2, tid, n);
    }
    __syncthreads();
}

#endif
