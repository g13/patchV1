#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include <cassert>
#include <ctime>
#include <cmath>
#include <fenv.h>
//#include <cuda.h>
//#include <curand_kernel.h>
//#include "cuda_runtime.h"

#include "connect.h"

template <typename T, typename I>
void check_statistics(T* array, I n, T &max, T &min, T &mean, T &std) {

}

int main() {
    unsigned long int i = (unsigned long int) -1;
    printf("%lu\n", i);
    unsigned long long seed = 7548637;
    std::ifstream pos_file;
    std::ofstream conMat_file, conVec_file;
    unsigned int nblock = 1024;
    unsigned int nPotentialNeigbor = 20*blockSize;
    unsigned int networkSize = nblock*blockSize;
    initialize_package init_pack();
    _float speedOfThought = 1.0f; // mm/ms
    
    std::string theme = "1-10th_macaque_fovea_v1";
    pos_file.open("1-10th_macaque_fovea_v1_3d_pos.bin", std::ios::in|std::ios::binary);
    conMat_file.open(theme + "_conMat.bin", std::ios::in|std::ios::binary);
    conVec_file.open(theme + "_conVec.bin", std::ios::in|std::ios::binary);
    unsigned long d_memorySize = 0;
    // non-inits device only
    unsigned long memorySize = 0;

    _float *block_x, *block_y; // nblock
        memorySize += nblock*2*sizeof(_float);

    _float *gpu_chunk;
    CUDA_CALL(cudaMalloc((void**)&gpu_chunk, memorySize));
    d_memorySize += memorySize;

    block_x = gpu_chunk; block_y = block_x + nblock;

    memorySize = 0;

    unsigned long outputSize = 0;
    // to receive from device
    _float *pos;
        memorySize += 3*networkSize*sizeof(_float);

    unsigned int *preType;
        memorySize += networkSize*sizeof(unsigned int);

    _float *conMat, *delayMat;
        memorySize += 2*blockSize*blockSize*nblock*sizeof(_float);
        outputSize += 2*blockSize*blockSize*nblock*sizeof(_float);

    _float  *conVec, *delayVec;
        memorySize += 2*blockSize*nPotentialNeigbor*sizeof(_float);
        outputSize += 2*blockSize*nPotentialNeigbor*sizeof(_float);

    unsigned int *neighborBlockId, *nNeighborBlock;
        memorySize += (nPotentialNeigbor + 1)*nblock*sizeof(unsigned int);
        outputSize += (nPotentialNeigbor + 1)*nblock*sizeof(unsigned int);

    unsigned int *preTypeConnected, *preTypeAvail; // NTYPE*networkSize
        memorySize += NTYPE*networkSize*2*sizeof(unsigned int);
        outputSize += NTYPE*networkSize*2*sizeof(unsigned int);

    _float *preTypeStr, *preTypeStrSum;
        memorySize += NTYPE*(NTYPE + networkSize)*sizeof(_float);
        outputSize += NTYPE*(NTYPE + networkSize)*sizeof(_float);

    _float *cpu_chunk;
    CUDA_CALL(cudaMallocHost((void**)&cpu_chunk, memorySize));
    printf("memory allocated on host: %lu\n", memorySize);

    pos = cpu_chunk;
    preType = (*unsigned int) (pos + 3*networkSize);
    conMat = (*_float) (preType + networkSize*NTYPE); delayMat = conMat + blockSize*blockSize*nblock;
    conVec = delayMat + blockSize*blockSize*nblock; delayVec = conVec + blockSize*nPotentialNeigbor;
    neighborBlockId = (*unsigned int) (delayVec + blockSize*nPotentialNeigbor); nNeighborBlock = neighborBlockId + nPotentialNeigbor*nblock;
    preTypeConnected = nNeighborBlock + nblock; preTypeAvail = preTypeConnected + NTYPE*networkSize;
    preTypeStr = (*_float) (preTypeAvail + NTYPE*networkSize);
    preTypeStrSum = preTypeStr + NTYPE*networkSize;

    // init by host, one-way trip to device
    _float *rden, *raxn; // NTYPE
        memorySize += NTYPE*2*networkSize*sizeof(_float);

    _float *preTypeDaxn;
        memorySize += NTYPE*networkSize*sizeof(_float);

    unsigned int *preTypeN;
        memorySize += NTYPE*networkSize*sizeof(unsigned int);
    // output to host
    _float *d_pos; // initialized by cudaMemcpy
    unsigned int *d_preType;
    _float *d_conMat, *d_conVec;
    _float *d_delayMat, *d_delayVec;
    unsigned int *d_neighborBlockId, *d_nNeighborBlock;
    unsigned int *d_preTypeConnected, *d_preTypeAvail;
    _float *d_preTypeStr, d_preTypeStrSum;
    _float *d_chunk;
    d_memorySize += memorySize;
    CUDA_CALL(cudaMalloc((void**)&d_chunk, memorySize));
    printf("memory allocated on device: %lu\n", d_memorySize);

    rden = d_chunk; raxn = rden + NTYPE*networkSize;
    preTypeDaxn = raxn + NTYPE*networkSize;
    preTypeN = (*unsigned int) (preTypeDaxn + NTYPE*networkSize); 

    d_pos = (*_float) (preTypeN + NTYPE*networkSize);
    d_preType = (*unsigned int) (d_pos + 3*networkSize);
    d_conMat = (*_float) (d_preType + networkSize); d_delayMat = d_conMat + blockSize*blockSize*nblock;
    d_conVec = d_delayMat + blockSize*blockSize*nblock; d_delayVec = d_conVec + blockSize*nPotentialNeigbor;
    d_neighborBlockId = (*unsigned int) (d_delayVec + blockSize*nPotentialNeigbor); d_nNeighborBlock = d_neighborBlockId + nPotentialNeigbor*nblock;
    d_preTypeConnected = d_nNeighborBlock + nblock; d_preTypeAvail = d_preTypeConnected + NTYPE*networkSize;
    d_preTypeStr = (*_float) (d_preTypeAvail + NTYPE*networkSize);
    d_preTypeStrSum = d_preTypeStr + NTYPE*networkSize;

    pos_file.seekg(0, std::ios::end);
    const size_t precision = pos_file.tellg() / (3*networkSize);
    pos_file.seekg(0, std::ios::beg);
    printf("pos precision: %u\n", precision);
    if (precision == sizeof(double)) {
        double *tmp = new double[networkSize*3];
        pos_file.read(reinterpret_cast<char*>(&tmp[0]), 3*networkSize*sizeof(double));
        for (unsigned int i=0; i<networkSize; i++) {
            pos[i] = static_cast<float>(tmp[i]);
        }
        delete []tmp;
    } else {
        assert(precision == sizeof(float));
        pos_file.read(reinterpret_cast<char*>(&pos[0]), 3*networkSize*sizeof(float));
    }

    cudaStream_t s0, s1, s2;
    cudaEvent_t i0, i1, i2;
    cudaEventCreate(&i0);
    cudaEventCreate(&i1);
    cudaEventCreate(&i2);
    CUDA_CALL(cudaStreamCreate(&s0));
    CUDA_CALL(cudaStreamCreate(&s1));
    CUDA_CALL(cudaStreamCreate(&s2));
    CUDA_CALL(cudaMemcpyAsync(d_pos, pos, nblock*blockSize*blockSize*sizeof(_float), cudaMemcpyHostToDevice, s0));
    init<<<nblock, blockSize, 0, s0>>>(d_preType, rden, raxn, preTypeDaxn, preTypeN, d_pos, init_pack, seed);
	CUDA_CALL(cudaEventRecord(i0, s0));
    unsigned int shared_mem;
    shared_mem = 2*warpSize*sizeof(_float);
    cal_blockPos<<<nblock, blockSize, shared_mem, s1>>>(d_pos, block_x, block_y);
    shared_mem = sizeof(unsigned int);
    get_neighbor_blockId<<<nblock, blockSize, shared_mem, s1>>>(block_x, block_y, d_neighborBlockId, d_nNeighborBlock, max_radius, nPotentialNeigbor);
	CUDA_CALL(cudaEventRecord(i1, s1));
	CUDA_CALL(cudaEventSynchronize(i0));
	CUDA_CALL(cudaEventSynchronize(i1));
    shared_mem = blockSize*sizeof(_float) * blockSize*sizeof(_float);
    generate_connections<<<nblock, blockSize, shared_mem, s0>>>(d_pos, d_neighborBlockId, d_nNeighborBlock, rden, axn, d_conMat, d_delayMat, d_conVec, d_delayVec, d_preTypeConnected, d_preTypeAvail, preTypeN, d_preTypeStrSum, d_preTypeStr, d_preType, preTypeDaxn, state, networkSize, nNeighborMax, speedOfThought);
    CUDA_CALL(cudaMemcpyAsync(conMat, d_conMat, outputSize, cudaMemcpyDeviceToHost, s0)); // the whole chunk of output
    CUDA_CALL(cudaStreamDestroy(s0));
    CUDA_CALL(cudaStreamDestroy(s1));
    CUDA_CALL(cudaStreamDestroy(s2));

    /*unsigned long preSumN[NTYPE][NTYPE];
	unsigned long preSumStr[NTYPE][NTYPE];
    for (unsigned int i=0; i<networkSize; i++) {
        for (unsigned int j=0; j<NTYPE; j++) {
        }
    }*/

    CUDA_CALL(cudaFreeHost(cpu_chunk));
    CUDA_CALL(cudaFree(d_chunk));
    CUDA_CALL(cudaFree(gpu_chunk));
    delete []pos;
}
