#include <cassert>
#include <curand_kernel.h>
#include <cuda.h>
#include <stdio.h>

#define warpSize 32
#define FULL_MASK 0xffffffff

__device__ void warp_min(double* array, unsigned int* id) {
    double value = array[threadIdx.x];
    double index = id[threadIdx.x];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        double compare = __shfl_down_sync(FULL_MASK, value, offset);
        unsigned int comp_id = __shfl_down_sync(FULL_MASK, index, offset);
        if (value > compare) {
            value = compare;
            index = comp_id;
        }
    }
    if (threadIdx.x % warpSize == 0) {
        unsigned int head = threadIdx.x/warpSize;
        array[head] = value;
        id[head] = index;
    }
}

__device__ void find_min(double* array, unsigned int* id) { 
    warp_min(array, id);
    __syncthreads();
    if (threadIdx.x < warpSize) {
        warp_min(array, id);
    }
    __syncthreads();
}

__global__ void test(curandStateMRG32k3a *state, unsigned int seed) {
    extern __shared__ double array[1024];
    extern __shared__ unsigned int index[1024];
	index[threadIdx.x] = threadIdx.x;
    curandStateMRG32k3a localState = state[threadIdx.x];
    curand_init(seed, threadIdx.x, 0, &localState);
	array[threadIdx.x] = curand_uniform_double(&localState);
	//printf("%u: %f\n", index[threadIdx.x], array[threadIdx.x]);
    find_min(array, index);
    if (threadIdx.x == 0) {
        __syncwarp();
        printf("final: %u: %f\n", index[0], array[0]);
        double min = 1.0;
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

int main(int argc, char *argv[]) {
    curandStateMRG32k3a *state;
    cudaMalloc((void **)&state, 1024 * sizeof(curandStateMRG32k3a));
	unsigned int seed = 65763895;
    sscanf(argv[argc-1],"%u",&seed);
    test<<<1,1024,1024*8+1024*4>>>(state, seed);
    return EXIT_SUCCESS;
}
