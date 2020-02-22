#include "cuda_util.cuh"

__device__ void warps_min(Float array[], Float data, PosInt id[], PosInt tid, Size n) {
	PosInt index = tid;
    unsigned MASK = __ballot_sync(FULL_MASK, tid < n);
    if (tid < n) {
        Size half_width = warpSize/2; // ceil power of 2
        Size remain = 0;
        if (n % warpSize != 0 && tid/warpSize == n/warpSize) {
            // get the ceil power of 2 for the last warp
            remain = n % warpSize; // remaining element in the last warp
            Size m = remain;
            half_width = 1;
            while(m >>= 1) half_width <<=1;
            remain = half_width*2 - remain; // inactive element in the ceil power of 2 threads
        }
        /*DEBUG
            if (blockIdx.x == 2 && tid == 0) {
                printf("width = %u, remain = %u\n", 2*half_width, remain);
            }
        */
 
        for (Size offset = half_width; offset > 0; offset /= 2) {
            Float comp_data = __shfl_down_sync(MASK, data, offset, 2*half_width);
            PosInt comp_id = __shfl_down_sync(MASK, index, offset, 2 *half_width);
            PosInt ttid = __shfl_down_sync(MASK, tid, offset, 2*half_width);
            if (data > comp_data && (tid%warpSize < half_width - remain || offset < half_width)) {
                /* DEBUG
                    if (blockIdx.x == 2) {
                        printf("%u/%u, exchange %u#%e -> %u#%e\n", offset, 2*half_width, tid, data, ttid, comp_data);
                    }
                */
                data = comp_data;
                index = comp_id;
            }
        }
    }
    __syncthreads();
    if (tid < n && tid%warpSize == 0) {
        PosInt head = tid/warpSize;
        array[head] = data;
        id[head] = index;
        /*DEBUG
            if (blockIdx.x == 2) {
                printf("#%u, sorted = %e\n", index, data);
            }
        */
    }
}

__device__ void warp0_min(Float array[], PosInt id[], Size tid, PosInt n) {
	unsigned MASK = __ballot_sync(FULL_MASK, tid < n);
    if (tid < n) {
        Size half_width = 1;
        Size m = n;
        while (m >>= 1) half_width <<= 1;
        Float data = array[tid];
        PosInt index = id[tid];
        for (Size offset = half_width; offset > 0; offset /= 2) {
            Float comp_data = __shfl_down_sync(MASK, data, offset, 2*half_width);
            PosInt comp_id = __shfl_down_sync(MASK, index, offset, 2*half_width);
            PosInt ttid = __shfl_down_sync(MASK, tid, offset, 2*half_width);
            if (data > comp_data && (tid < n - half_width || offset < half_width)) { // tid < width/2 - (width - n)
                data = comp_data;
                index = comp_id;
            }
        }
        if (tid == 0) {
            array[0] = data;
            id[0] = index;
        }
    }
}

__device__ void find_min(Float array[], Float data, PosInt id[], Size n) { 
    PosInt tid = blockDim.x*(blockDim.y*threadIdx.z + threadIdx.y) + threadIdx.x;
    Size nThreads = blockDim.x*blockDim.y*blockDim.z;
    if (n > nThreads) n = nThreads;
    /*DEBUG
        if (tid < n && blockIdx.x == 2) {
            printf("#%u,data = %e\n", tid, data);
        }
    */
	warps_min(array, data, id, tid, n);
    __syncthreads();
    n = (n + warpSize-1)/warpSize;
    if (tid < warpSize && n > 1) {
        warp0_min(array, id, tid, n);
    }
    __syncthreads();
}
