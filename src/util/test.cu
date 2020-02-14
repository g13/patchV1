#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "cuda_util.cuh"
#include <helper_cuda.h>

__global__ void test_min(curandStateMRG32k3a *state, PosInt seed) {
    __shared__ Float array[blockSize];
    __shared__ PosInt index[blockSize];
	index[threadIdx.x] = threadIdx.x;
    curandStateMRG32k3a localState = state[threadIdx.x];
    curand_init(seed, threadIdx.x, 0, &localState);
    Float data = curand_uniform(&localState);
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

__global__ void test_sum(curandStateMRG32k3a *state, PosInt seed, PosInt nThreads) {
    __shared__ Float array[warpSize];
    __shared__ Float serial[blockSize];
    Size tid = threadIdx.x + threadIdx.y*blockDim.x;
    //printf("%i, %i = %i\n", threadIdx.y, threadIdx.x, tid);

    curandStateMRG32k3a localState = state[tid];
    curand_init(seed, tid, 0, &localState);
	Float data = curand_uniform_double(&localState);
    serial[tid] = data;
    if (tid == 0) {
        printf("data generated\n");
    }

    block_reduce<Float>(array, data);

    if (tid == 0) {
        printf("reduce1: %f\n", array[0]);
        data = 0.0;
        for (Size i=0; i<nThreads; i++) {
            //printf("d[%i]: %f\n", i, serial[i]);
            data += serial[i];
        }
        printf("sum: %f\n", data);
    }

	data = curand_uniform_double(&localState);
    serial[tid] = data;
    block_reduce<Float>(array, data);

    if (tid == 0) {
        printf("reduce2: %f\n", array[0]);
        data = 0.0;
        for (Size i=0; i<nThreads; i++) {
            //printf("d[%i]: %f\n", i, serial[i]);
            data += serial[i];
        }
        printf("sum: %f\n", data);
    }
}

int main(int argc, char *argv[]) {
    curandStateMRG32k3a *state;
    cudaMalloc((void **)&state, blockSize * sizeof(curandStateMRG32k3a));
	PosInt seed = 65763895;
    sscanf(argv[argc-1],"%u",&seed);
    //test_min<<<1,1024,1024*8+1024*4>>>(state, seed);
    PosInt width, height;
    sscanf(argv[argc-2],"%u",&height);
    sscanf(argv[argc-3],"%u",&width);
    printf("block layout:  %u x %d\n", width, height);
    dim3 block(width,height,1); 
    dim3 grid(1,1,1); 
    PosInt nThreads = block.x * block.y;
    test_sum<<<grid, block>>>(state, seed, nThreads);
    getLastCudaError("sum failed");
	cudaDeviceSynchronize();
    return EXIT_SUCCESS;
}
