#include <cassert>
#include <curand_kernel.h>
#include <cuda.h>
#include <stdio.h>
#include "cuda_util.h"
//#include <helper_functions.h>
#include <helper_cuda.h>
#define warpSize 32
#define blockSize 1024 
#define FULL_MASK 0xffffffff

__global__ void test_min(curandStateMRG32k3a *state, unsigned int seed) {
    __shared__ float array[1024];
    __shared__ unsigned int index[1024];
	index[threadIdx.x] = threadIdx.x;
    curandStateMRG32k3a localState = state[threadIdx.x];
    curand_init(seed, threadIdx.x, 0, &localState);
    float data = curand_uniform(&localState);
	array[threadIdx.x] = data;
	//printf("%u: %f\n", index[threadIdx.x], array[threadIdx.x]);
    find_min(array, data, index);
    if (threadIdx.x == 0) {
        __syncwarp();
        printf("final: %u: %f\n", index[0], array[0]);
        float min = 1.0;
        int imin = -1;
        for (int i=0; i<1024; i++) {
            if (array[i] < min) {
                min = array[i];
                imin = index[i];
            }
        }
        printf("serial: %i: %f\n", imin, min);
    }
}

__global__ void test_sum(curandStateMRG32k3a *state, unsigned int seed, unsigned int nThreads) {
    __shared__ double array[warpSize];
    __shared__ double serial[blockSize];
    unsigned int tid = threadIdx.x + threadIdx.y*blockDim.x;
    //printf("%i, %i = %i\n", threadIdx.y, threadIdx.x, tid);

    curandStateMRG32k3a localState = state[tid];
    curand_init(seed, tid, 0, &localState);
	double data = curand_uniform_double(&localState);
	//double data = 1.0;
    serial[tid] = data;
    //if (tid < warpSize) {
    //    array[tid] = 0;
    //}
    __syncthreads();
    if (tid == 0) {
        printf("data generated\n");
    }

    __syncwarp();
    block_reduce<double>(array, data);

    if (tid == 0) {
        printf("reduce: %f\n", array[0]);
        data = 0.0;
        for (int i=0; i<nThreads; i++) {
            //printf("d[%i]: %f\n", i, serial[i]);
            data += serial[i];
        }
        printf("sum: %f\n", data);
    }
}

int main(int argc, char *argv[]) {
    curandStateMRG32k3a *state;
    cudaMalloc((void **)&state, blockSize * sizeof(curandStateMRG32k3a));
	unsigned int seed = 65763895;
    sscanf(argv[argc-1],"%u",&seed);
    //test_min<<<1,1024,1024*8+1024*4>>>(state, seed);
    unsigned int width, height;
    sscanf(argv[argc-2],"%u",&height);
    sscanf(argv[argc-3],"%u",&width);
    printf("block layout:  %i x %i\n", width, height);
    dim3 block(width,height,1); 
    dim3 grid(1,1,1); 
    unsigned int nThreads = block.x * block.y;
    test_sum<<<grid, block>>>(state, seed, nThreads);
    getLastCudaError("sum failed");
    return EXIT_SUCCESS;
}
