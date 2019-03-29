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
    unsigned int nblock = 48;
    unsigned int nPotentialNeighbor = 8;
    unsigned int networkSize = nblock*blockSize;
    _float radius[NTYPE][2];
    _float neuron_type_acc_count[NTYPE+1];
	_float den_axn[NTYPE];
	_float den_den[NTYPE];
    // E
    radius[0][0] = 0.080;
    radius[0][1] = 0.150;
    // I
    radius[1][0] = 0.050;
    radius[1][1] = 0.100;

    // row <- column
    _float sTypeMat[NTYPE][NTYPE];
    sTypeMat[0][0] = 1.0;
    sTypeMat[0][1] = 4.0;
    sTypeMat[1][0] = 1.0;
    sTypeMat[1][1] = 4.0;

    //upper limit of sparsity stricted to NTYPE
    _float pTypeMat[NTYPE][NTYPE];
    pTypeMat[0][0] = 0.15;
    pTypeMat[0][1] = 0.15;
    pTypeMat[1][0] = 0.5;
    pTypeMat[1][1] = 0.5;

    // 
    unsigned int cTypeMat[NTYPE][NTYPE];
    pTypeMat[0][0] = 400+20*5; // mean + std*5
    pTypeMat[0][1] = 100+10*5;
    pTypeMat[1][0] = 400+20*5;
    pTypeMat[1][1] = 100+10*5;
    
    unsigned int neighborSize = 400;

    // NTYPE
    neuron_type_acc_count[0] = 0;
    neuron_type_acc_count[1] = 768;
    neuron_type_acc_count[2] = 1024;
    assert(neuron_type_acc_count[NTYPE] == blockSize);

    den_axn[0] = 1.0;
    den_axn[1] = 2.0;

    den_den[0] = 1.0;
    den_den[1] = 2.0;

    initialize_package init_pack(radius, neuron_type_acc_count, den_axn, den_den);
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
        memorySize += 2*networkSize*neighborSize*sizeof(_float);
        outputSize += 2*networkSize*neighborSize*sizeof(_float);

    unsigned int *vecID;
        memorySize += networkSize*neighborSize*sizeof(unsigned int);
        outputSize += networkSize*neighborSize*sizeof(unsigned int);

    unsigned int *neighborBlockId, *nNeighborBlock;
        memorySize += (nPotentialNeighbor + 1)*nblock*sizeof(unsigned int);
        outputSize += (nPotentialNeighbor + 1)*nblock*sizeof(unsigned int);

    unsigned int *preTypeConnected, *preTypeAvail; // NTYPE*networkSize
        memorySize += 2*NTYPE*networkSize*sizeof(unsigned int);
        outputSize += 2*NTYPE*networkSize*sizeof(unsigned int);

    _float *preTypeStrSum;
        memorySize += NTYPE*(NTYPE + networkSize)*sizeof(_float);
        outputSize += NTYPE*(NTYPE + networkSize)*sizeof(_float);

    
	printf("need to allocate %f MB memory on host\n", static_cast<float>(memorySize)/1024/1024);
	void *cpu_chunk = malloc(memorySize);
	assert(cpu_chunk);

    pos = (_float*) cpu_chunk;
    preType = (unsigned int*) (pos + 3*networkSize);
    conMat = (_float*) (preType + networkSize); 
    delayMat = conMat + blockSize*blockSize*nblock;
    conVec = delayMat + blockSize*blockSize*nblock; 
    delayVec = conVec + networkSize*neighborSize;
    vecID = (unsigned int*) (delayVec + networkSize*neighborSize);
    neighborBlockId = vecID + networkSize*neighborSize;
    nNeighborBlock = neighborBlockId + nPotentialNeighbor*nblock;
    preTypeConnected = nNeighborBlock + nblock; 
    preTypeAvail = preTypeConnected + NTYPE*networkSize;
    preTypeStrSum = (_float*) (preTypeAvail + NTYPE*networkSize);

	assert(static_cast<void*>((char*)cpu_chunk + memorySize) == static_cast<void*>(preTypeStrSum + NTYPE * networkSize));

    // ========== GPU mem ============
    // init by kernel, reside on device only
    _float *rden, *raxn; // NTYPE
		memorySize += 2*networkSize*sizeof(_float);

    _float *preTypeDaxn;
        memorySize += networkSize*sizeof(_float);

    _float *preTypeDend;
        memorySize += networkSize*sizeof(_float);

    _float *preTypeS;
        memorySize += NTYPE*networkSize*sizeof(_float);

    _float *preTypeP;
        memorySize += NTYPE*networkSize*sizeof(_float);

    unsigned int *preTypeN;
        memorySize += NTYPE*networkSize*sizeof(unsigned int);

    curandStateMRG32k3a* state;
        memorySize += networkSize*sizeof(curandStateMRG32k3a);

    // init by cudaMemcpy, reside on device only
    _float *d_sTypeMat;
        memorySize += NTYPE*NTYPE*sizeof(_float);

    _float *d_pTypeMat;
        memorySize += NTYPE*NTYPE*sizeof(_float);

    unsigned int *d_cTypeMat;
        memorySize += NTYPE*NTYPE*sizeof(unsigned int);

    // init by cudaMemcpy
    _float *d_pos;
    // output to host
    unsigned int *d_preType;
    _float *d_conMat, *d_conVec;
    _float *d_delayMat, *d_delayVec;
    unsigned int *d_vecID;
    unsigned int *d_neighborBlockId, *d_nNeighborBlock;
    unsigned int *d_preTypeConnected, *d_preTypeAvail;
    _float *d_preTypeStrSum;
    void *d_chunk;
    d_memorySize += memorySize;
	printf("need to allocate %f MB memory on device\n", static_cast<float>(d_memorySize) / 1024 / 1024);
    CUDA_CALL(cudaMalloc((void**)&d_chunk, memorySize));

    rden = (_float*) d_chunk; raxn = rden + networkSize;
	preTypeDaxn = raxn + networkSize;
	preTypeDend = preTypeDaxn + networkSize;
    preTypeS = preTypeDend + networkSize;
    preTypeP = preTypeS + NTYPE*networkSize;
    preTypeN = (unsigned int*) preTypeP + NTYPE*networkSize;
    state = (curandStateMRG32k3a*) (preTypeN + NTYPE*networkSize);
    d_sTypeMat = (_float*) (state + networkSize);
    d_pTypeMat = d_sTypeMat +NTYPE*NTYPE;
    d_cTypeMat = (unsigned int*) (d_pTypeMat + NTYPE*NTYPE);

    d_pos = (_float*) (d_cTypeMat + NTYPE*NTYPE);

    d_preType = (unsigned int*) (d_pos + 3*networkSize);
    d_conMat = (_float*) (d_preType + networkSize); 
    d_delayMat = d_conMat + blockSize*blockSize*nblock;
    d_conVec = d_delayMat + blockSize*blockSize*nblock; 
    d_delayVec = d_conVec + networkSize*neighborSize;
    d_vecID = (unsigned int*) d_delayVec + networkSize*neighborSize;
    d_neighborBlockId = d_vecID + networkSize*neighborSize;
    d_nNeighborBlock = d_neighborBlockId + nPotentialNeighbor*nblock;
    d_preTypeConnected = d_nNeighborBlock + nblock;
    d_preTypeAvail = d_preTypeConnected + NTYPE*networkSize;
    d_preTypeStrSum = (_float*) (d_preTypeAvail + NTYPE*networkSize);

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
    CUDA_CALL(cudaMemcpy(d_pos, pos, networkSize*3*sizeof(_float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_sTypeMat, sTypeMat, NTYPE*NTYPE*sizeof(_float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pTypeMat, pTypeMat, NTYPE*NTYPE*sizeof(_float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_cTypeMat, cTypeMat, NTYPE*NTYPE*sizeof(unsigned int), cudaMemcpyHostToDevice));
    initialize<<<nblock, blockSize, 0, s1>>>(state, 
											 d_preType, 
											 rden, 
											 raxn, 
											 preTypeDaxn, 
											 preTypeDend, 
											 d_sTypeMat,
											 d_pTypeMat,
											 d_cTypeMat,
											 preTypeS, 
											 preTypeP, 
											 preTypeN, 
											 init_pack, seed, networkSize);
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
																preTypeS,
																preTypeP,
																preTypeN,
																d_neighborBlockId, 
																d_nNeighborBlock, 
																rden, 
																raxn, 
																d_conMat, 
																d_delayMat, 
																d_conVec, 
																d_delayVec, 
																d_vecID,
																d_preTypeConnected, 
																d_preTypeAvail, 
																d_preTypeStrSum, 
																d_preType, 
																preTypeDaxn, 
																preTypeDend, 
																state, 
																networkSize, neighborSize, nPotentialNeighbor, speedOfThought);
	CUDA_CHECK();
	CUDA_CALL(cudaMemcpy(preType, d_preType, outputSize, cudaMemcpyDeviceToHost)); // the whole chunk of output
	//CUDA_CALL(cudaMemcpy(preType, d_preType, 1, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaStreamDestroy(s0));
    CUDA_CALL(cudaStreamDestroy(s1));
    CUDA_CALL(cudaStreamDestroy(s2));

    /*unsigned long preSumN[NTYPE][NTYPE];
	unsigned long preSumStr[NTYPE][NTYPE];
    for (unsigned int i=0; i<networkSize; i++) {
        for (unsigned int j=0; j<NTYPE; j++) {
        }
    }*/

    CUDA_CALL(cudaFree(d_chunk));
    CUDA_CALL(cudaFree(gpu_chunk));
	free(cpu_chunk);
}
