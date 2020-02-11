#include "cuda_util.cuh"

__device__ void warp0_min(Float array[], PosInt id[]) {
    Float value = array[threadIdx.x];
    Float index = id[threadIdx.x];
    for (Size offset = warpSize/2; offset > 0; offset /= 2) {
        Float compare = __shfl_down_sync(FULL_MASK, value, offset);
        PosInt comp_id = __shfl_down_sync(FULL_MASK, index, offset);
        if (value > compare) {
            value = compare;
            index = comp_id;
        }
        __syncwarp();
    }
    if (threadIdx.x == 0) {
        array[0] = value;
        id[0] = index;
    }
}

__device__ void warps_min(Float array[], Float data, PosInt id[]) {
	Float index = threadIdx.x;
    for (Size offset = warpSize/2; offset > 0; offset /= 2) {
        Float comp_data = __shfl_down_sync(FULL_MASK, data, offset);
        PosInt comp_id = __shfl_down_sync(FULL_MASK, index, offset);
        if (data > comp_data) {
            data = comp_data;
            index = comp_id;
        }
        __syncwarp();
    }
    __syncthreads();
    if (threadIdx.x % warpSize == 0) {
        PosInt head = threadIdx.x/warpSize;
        array[head] = data;
        id[head] = index;
    }
}

__device__ void find_min(Float array[], Float data, PosInt id[]) { 
	warps_min(array, data, id);
    __syncthreads();
    if (threadIdx.x < warpSize) {
        warp0_min(array, id);
    }
    __syncthreads();
}


