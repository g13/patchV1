#include "cuda_util.cuh"

__device__ void warps_min(Float array[], Float data, PosInt id[]) {
    PosInt tid = blockDim.x*threadIdx.x + threadIdx.x;
	PosInt index = tid;
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
    if (tid % warpSize == 0) {
        PosInt head = tid/warpSize;
        array[head] = data;
        id[head] = index;
    }
}

__device__ void warp0_min(Float array[], PosInt id[]) {
    PosInt tid = blockDim.x*threadIdx.y + threadIdx.x;
	Size n = (blockDim.x * blockDim.y * blockDim.z + warpSize - 1)/warpSize; // may not be a full warp, avoid access uninitialized shared memory
	unsigned MASK = __ballot_sync(FULL_MASK, tid < n);
    if (tid < n) {
        Float value = array[tid];
        PosInt index = id[tid];
        for (Size offset = warpSize/2; offset > 0; offset /= 2) {
            Float compare = __shfl_down_sync(MASK, value, offset);
            PosInt comp_id = __shfl_down_sync(MASK, index, offset);
            if (value > compare) {
                value = compare;
                index = comp_id;
            }
            __syncwarp();
        }
        if (tid == 0) {
            array[0] = value;
            id[0] = index;
        }
    }
}

__device__ void find_min(Float array[], Float data, PosInt id[]) { 
	warps_min(array, data, id);
    __syncthreads();
    if (blockDim.x*threadIdx.y + threadIdx.x < warpSize) {
        warp0_min(array, id);
    }
    __syncthreads();
}


