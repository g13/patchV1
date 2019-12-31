#include "cuda_util.cuh"

__device__ void warp0_min(_float array[], unsigned int id[]) {
    _float value = array[threadIdx.x];
    _float index = id[threadIdx.x];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        _float compare = __shfl_down_sync(FULL_MASK, value, offset);
        unsigned int comp_id = __shfl_down_sync(FULL_MASK, index, offset);
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

__device__ void warps_min(_float array[], _float data, unsigned int id[]) {
	_float index = threadIdx.x;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        _float comp_data = __shfl_down_sync(FULL_MASK, data, offset);
        unsigned int comp_id = __shfl_down_sync(FULL_MASK, index, offset);
        if (data > comp_data) {
            data = comp_data;
            index = comp_id;
        }
        __syncwarp();
    }
    __syncthreads();
    if (threadIdx.x % warpSize == 0) {
        unsigned int head = threadIdx.x/warpSize;
        array[head] = data;
        id[head] = index;
    }
}

__device__ void find_min(_float array[], _float data, unsigned int id[]) { 
	warps_min(array, data, id);
    __syncthreads();
    if (threadIdx.x < warpSize) {
        warp0_min(array, id);
    }
    __syncthreads();
}
