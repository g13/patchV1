#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "cuda_util.cuh"
#include <helper_cuda.h>

__launch_bounds__(1024,1)
__global__ 
void test_min(curandStateMRG32k3a *state, PosIntL seed, Size n) {
    __shared__ Float array[warpSize];
    __shared__ PosInt id[warpSize];
    __shared__ Float serial[blockSize];
    Size tid = threadIdx.x + threadIdx.y*blockDim.x;
    curandStateMRG32k3a localState = state[tid];
    curand_init(seed, tid, 0, &localState);
    Float data;
    if (tid < n) {
        data = curand_uniform(&localState);
	    serial[tid] = data;
    }
    if (tid == 0) {
        printf("data generated\n");
    }
    find_min(array, data, id, n);

    if (tid == 0) {
        printf("pmin: %u: %e\n", id[0], array[0]);
        Float min = 1.0;
        PosInt imin;
        for (int i=0; i<n; i++) {
            //printf("%f, ", serial[i]);
            if (serial[i] < min) {
                min = serial[i];
                imin = i;
            }
        }
        printf("smin: %u: %e\n", imin, min);
    }
}

__launch_bounds__(1024,1)
__global__ 
void test_sum(curandStateMRG32k3a *state, PosIntL seed) {
    __shared__ Float array[warpSize];
    __shared__ Float serial[blockSize];
    Size tid = threadIdx.x + threadIdx.y*blockDim.x;

    curandStateMRG32k3a localState = state[tid];
    curand_init(seed, tid, 0, &localState);
	Float data = curand_uniform(&localState);
    serial[tid] = data;

    if (tid == 0) {
        for (PosInt i = 0; i<warpSize; i++) {
            array[i] = 100;
        }
    }
    __syncthreads();
    block_reduce<Float>(array, data);

    if (tid == 0) {
        printf("reduce: %e\n", array[0]);
        data = 0;
        for (Size i=0; i<blockDim.x*blockDim.y; i++) {
            //printf("%e, ", serial[i]);
            data += serial[i];
        }
        printf("sum: %e\n", data);
        array[0] = 0;
    }

    data = serial[tid];
    __syncthreads();
    atomicAdd(array, data);
    __syncthreads();
    if (tid == 0) {
        printf("atomicAdd: %e\n", array[0]);
    }
}

int main(int argc, char *argv[]) {
    curandStateMRG32k3a *state;
	checkCudaErrors(cudaMalloc((void **)&state, blockSize * sizeof(curandStateMRG32k3a)));
	PosIntL seed;
	Size n;
    sscanf(argv[argc-1],"%u",&n);
    sscanf(argv[argc-2],"%lu",&seed);
    Size width, height;
    sscanf(argv[argc-3],"%u",&height);
    sscanf(argv[argc-4],"%u",&width);
    printf("block layout:  %u x %d\n", width, height);
    printf("seed =  %lu\n", seed);
    dim3 block(width,height,1); 
    dim3 grid(1,1,1); 
    test_sum<<<grid, block>>>(state, seed);
    getLastCudaError("sum failed");
    test_min<<<grid, block>>>(state, seed, n);
    getLastCudaError("min failed");
	cudaDeviceSynchronize();
	checkCudaErrors(cudaFree(state));
    return EXIT_SUCCESS;
}
