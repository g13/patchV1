#include <cuda.h>

template <typename X>
__device__ void warpReduce(volatile X* data, int id) {
    data[id] += data[id + 32];
    data[id] += data[id + 16];
    data[id] += data[id + 8];
    data[id] += data[id + 4];
    data[id] += data[id + 2];
    data[id] += data[id + 1];
}
template <typename X, typename Y>
__global__ void dot1d(X* x, Y* y1, Y* y2, Y* g, Y* h) {
    extern __shared__ Y product_g[];
    extern __shared__ Y product_h[];
    unsigned int blockLen = blockDim.x;
    // thread index
    unsigned int block_id = threadIdx.x;
    unsigned int global_id = blockIdx.x*(2*blockLen) + block_id;
    // elmenent-wise product
    product_g[block_id] = x[global_id]*y1[global_id] + x[global_id + blockLen] * y1[global_id+blockLen];
    product_h[block_id] = x[global_id]*y2[global_id] + x[global_id + blockLen] * y2[global_id+blockLen];
    __syncthreads();
    // reduction within block
    for (unsigned int i=blockDim.x/2; i>32; i>>=1) { // keep data stored sequentially
        if (block_id < i) {
            product_g[block_id] += product_g[block_id + i];
            product_h[block_id] += product_h[block_id + i];
        }
        __syncthreads();
    }

    if (block_id < 32) {
        warpReduce(product_g,block_id);
        warpReduce(product_h,block_id);
    }

    if (block_id == 0) {
        *g = product_g[0];
        *h = product_h[0];
    }
}
