#include <stdio.h>
#include <fstream>
#include <sstream>
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

int main(int argc, char *argv[])
{
    unsigned long int i = (unsigned long int) -1;
    printf("max unsigned long = %lu\n", i);
    unsigned long long seed = 7548637;
    std::ifstream pos_file;
    std::string dir(argv[1]);
    std::string theme(argv[2]);
    std::ofstream conMat_file, conVec_file;
    unsigned int nblock = 1024;
    unsigned int nPotentialNeighbor = 24;
    unsigned int networkSize = nblock*blockSize;
    _float radius[NTYPE][2];
    _float neuron_type_acc_count[NTYPE+1];
	unsigned int den_axn[NTYPE];
    // E
    radius[0][0] = 0.080;
    radius[0][1] = 0.150;
    // I
    radius[1][0] = 0.050;
    radius[1][1] = 0.100;

    // NTYPE
    neuron_type_acc_count[0] = 0;
    neuron_type_acc_count[1] = 768;
    neuron_type_acc_count[2] = 1024;
    assert(neuron_type_acc_count[NTYPE] == blockSize);

    den_axn[0] = 1;
    den_axn[1] = 2;

    initialize_package init_pack(radius, neuron_type_acc_count, den_axn);
    _float speedOfThought = 1.0f; // mm/ms
    _float max_radius = 0.4f;
    
	std::string posfn = dir + theme + "_3d_pos.bin";
    pos_file.open(posfn, std::ios::in|std::ios::binary);
	if (!pos_file.is_open()) {
		std::cout << "failed to open pos file:" << posfn << "\n";
		return EXIT_FAILURE;
	}
    conMat_file.open(theme + "_conMat.bin", std::ios::out | std::ios::binary);
    conVec_file.open(theme + "_conVec.bin", std::ios::out | std::ios::binary);
    size_t d_memorySize = 0;
    // non-inits device only
    size_t memorySize = 0;

    _float *block_x, *block_y; // nblock
        memorySize += 2*nblock*sizeof(_float);

    void *gpu_chunk;
    CUDA_CALL(cudaMalloc((void**)&gpu_chunk, memorySize));
    d_memorySize += memorySize;

    block_x = (_float*) gpu_chunk; block_y = block_x + nblock;

    memorySize = 0;

    _float *pos;
        memorySize += 3*networkSize*sizeof(_float);
	
	// to receive from device
    unsigned long outputSize = 0;
    unsigned int *preType;
        memorySize += networkSize*sizeof(unsigned int);
        outputSize += networkSize*sizeof(unsigned int);

    _float *conMat, *delayMat;
        memorySize += 2*blockSize*blockSize*nblock*sizeof(_float);
        outputSize += 2*blockSize*blockSize*nblock*sizeof(_float);

    _float *conVec, *delayVec;
        memorySize += 2*blockSize*nPotentialNeighbor*sizeof(_float);
        outputSize += 2*blockSize*nPotentialNeighbor*sizeof(_float);

    unsigned int *neighborBlockId, *nNeighborBlock;
        memorySize += (nPotentialNeighbor + 1)*nblock*sizeof(unsigned int);
        outputSize += (nPotentialNeighbor + 1)*nblock*sizeof(unsigned int);

    unsigned int *preTypeConnected, *preTypeAvail; // NTYPE*networkSize
        memorySize += 2*NTYPE*networkSize*sizeof(unsigned int);
        outputSize += 2*NTYPE*networkSize*sizeof(unsigned int);

    _float *preTypeStr, *preTypeStrSum;
        memorySize += NTYPE*(NTYPE + networkSize)*sizeof(_float);
        outputSize += NTYPE*(NTYPE + networkSize)*sizeof(_float);

    
	printf("need to allocate %f MB memory on host\n", static_cast<float>(memorySize)/1024/1024);
	void *cpu_chunk = malloc(memorySize);
	assert(cpu_chunk);

    pos = (_float*) cpu_chunk;
    preType = (unsigned int*) (pos + 3*networkSize);
    conMat = (_float*) (preType + networkSize); delayMat = conMat + blockSize*blockSize*nblock;
    conVec = delayMat + blockSize*blockSize*nblock; delayVec = conVec + blockSize*nPotentialNeighbor;
    neighborBlockId = (unsigned int*) (delayVec + blockSize*nPotentialNeighbor); nNeighborBlock = neighborBlockId + nPotentialNeighbor*nblock;
    preTypeConnected = nNeighborBlock + nblock; preTypeAvail = preTypeConnected + NTYPE*networkSize;
    preTypeStr = (_float*) (preTypeAvail + NTYPE*networkSize);
    preTypeStrSum = preTypeStr + NTYPE*NTYPE;

	assert(static_cast<void*>((char*)cpu_chunk + memorySize) == static_cast<void*>(preTypeStrSum + NTYPE * networkSize));

    // init by kernel, reside on device only
    _float *rden, *raxn; // NTYPE
		memorySize += 2*networkSize*sizeof(_float);

    unsigned int *preTypeDaxn;
        memorySize += networkSize*sizeof(unsigned int);

    unsigned int *preTypeN;
        memorySize += NTYPE*networkSize*sizeof(unsigned int);

    curandStateMRG32k3a* state;
        memorySize += networkSize*sizeof(curandStateMRG32k3a);

    // initialized by cudaMemcpy
    _float *d_pos;
    // output to host
    unsigned int *d_preType;
    _float *d_conMat, *d_conVec;
    _float *d_delayMat, *d_delayVec;
    unsigned int *d_neighborBlockId, *d_nNeighborBlock;
    unsigned int *d_preTypeConnected, *d_preTypeAvail;
    _float *d_preTypeStr, *d_preTypeStrSum;
    void *d_chunk;
    d_memorySize += memorySize;
	printf("need to allocate %f MB memory on device\n", static_cast<float>(d_memorySize) / 1024 / 1024);
    CUDA_CALL(cudaMalloc((void**)&d_chunk, memorySize));

    rden = (_float*) d_chunk; raxn = rden + networkSize;
	preTypeDaxn = (unsigned int*)(raxn + networkSize);
    preTypeN =  preTypeDaxn + networkSize;
    state = (curandStateMRG32k3a*) (preTypeN + NTYPE*networkSize);

    d_pos = (_float*) (state + networkSize);

    d_preType = (unsigned int*) (d_pos + 3*networkSize);
    d_conMat = (_float*) (d_preType + networkSize); d_delayMat = d_conMat + blockSize*blockSize*nblock;
    d_conVec = d_delayMat + blockSize*blockSize*nblock; d_delayVec = d_conVec + blockSize*nPotentialNeighbor;
    d_neighborBlockId = (unsigned int*) (d_delayVec + blockSize*nPotentialNeighbor); d_nNeighborBlock = d_neighborBlockId + nPotentialNeighbor*nblock;
    d_preTypeConnected = d_nNeighborBlock + nblock; d_preTypeAvail = d_preTypeConnected + NTYPE*networkSize;
    d_preTypeStr = (_float*) (d_preTypeAvail + NTYPE*networkSize);
    d_preTypeStrSum = d_preTypeStr + NTYPE*NTYPE;

	//std::cout << static_cast<void*>((char*)d_chunk + memorySize) << "\n";
	//std::cout << static_cast<void*>(preTypeStrSum + NTYPE * networkSize) << "\n";
	//assert(static_cast<void*>((char*)d_chunk + memorySize) == static_cast<void*>(preTypeStrSum + NTYPE * networkSize));

    //pos_file.seekg(0, std::ios::end);
	//size_t file_size = pos_file.tellg();
	//printf("pos file size: %zu kB\n", file_size/1024);
    //size_t precision =  file_size / (3*networkSize);
    //pos_file.seekg(0, std::ios::beg);
    //printf("pos precision: %zu Bytes\n", precision);
	//void *tmp;
    //if (precision == sizeof(double)) {
		//typedef double pos_type;
        double* tmp = new double[networkSize*3];
        pos_file.read(reinterpret_cast<char*>(tmp), 3*networkSize*sizeof(double));
		for (unsigned int i = 0; i < networkSize * 3; i++) {
			pos[i] = static_cast<_float>(reinterpret_cast<double*>(tmp)[i]);
		}
		delete[]tmp;
    /*} else {
        assert(precision == sizeof(float));
		typedef float pos_type;
		tmp = new float[networkSize * 3];
        pos_file.read(reinterpret_cast<char*>(tmp), 3*networkSize*sizeof(float));
		for (unsigned int i = 0; i < networkSize * 3; i++) {
			pos[i] = static_cast<_float>(reinterpret_cast<float*>(tmp)[i]);
		}
		delete[]tmp;
    }*/

	
    cudaStream_t s0, s1, s2;
    cudaEvent_t i0, i1, i2;
    cudaEventCreate(&i0);
    cudaEventCreate(&i1);
    cudaEventCreate(&i2);
    CUDA_CALL(cudaStreamCreate(&s0));
    CUDA_CALL(cudaStreamCreate(&s1));
    CUDA_CALL(cudaStreamCreate(&s2));
    CUDA_CALL(cudaMemcpyAsync(d_pos, pos, networkSize*3*sizeof(_float), cudaMemcpyHostToDevice, s0));
    initialize<<<nblock, blockSize, 0, s1>>>(state, 
											 d_preType, 
											 rden, 
											 raxn, 
											 preTypeDaxn, 
											 preTypeN, 
											 init_pack, seed);
	CUDA_CHECK();
	CUDA_CALL(cudaEventRecord(i1, s1));
    unsigned int shared_mem;
    shared_mem = 2*warpSize*sizeof(_float);
    cal_blockPos<<<nblock, blockSize, shared_mem, s0>>>(d_pos, 
														block_x, 
														block_y, 
														networkSize);
	CUDA_CHECK();
	shared_mem = sizeof(unsigned int);
    get_neighbor_blockId<<<nblock, blockSize, shared_mem, s0>>>(block_x, 
																block_y, 
																d_neighborBlockId, 
																d_nNeighborBlock, 
																max_radius, nPotentialNeighbor);
	CUDA_CHECK();
	CUDA_CALL(cudaEventRecord(i0, s0));
	CUDA_CALL(cudaEventSynchronize(i0));
	CUDA_CALL(cudaEventSynchronize(i1));
	//CUDA_CALL(cudaEventSynchronize(i2));
    shared_mem = blockSize*sizeof(_float) + blockSize*sizeof(_float) + blockSize*sizeof(unsigned int);
    generate_connections<<<nblock, blockSize, shared_mem, s0>>>(d_pos, 
																d_neighborBlockId, 
																d_nNeighborBlock, 
																rden, 
																raxn, 
																d_conMat, 
																d_delayMat, 
																d_conVec, 
																d_delayVec, 
																d_preTypeConnected, 
																d_preTypeAvail, 
																preTypeN, 
																d_preTypeStrSum, 
																d_preTypeStr, 
																d_preType, 
																preTypeDaxn, 
																state, 
																networkSize, speedOfThought);
	CUDA_CHECK();
	CUDA_CALL(cudaMemcpyAsync(preType, d_preType, outputSize, cudaMemcpyDeviceToHost, s0)); // the whole chunk of output
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
