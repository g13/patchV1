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
    std::ofstream conMat_file, conVec_file, stats_file;
    unsigned int nblock = 48;
    unsigned int nPotentialNeighbor = 8;
    unsigned int networkSize = nblock*blockSize;
    unsigned int neighborSize = 400;
    unsigned int usingPosDim = 2;

    _float radius[NTYPE][2];
    _float neuron_type_acc_count[NTYPE+1];
	_float den_axn[NTYPE];
	_float den_den[NTYPE];
    // E
    _float scale = 5.0f;
    radius[0][0] = 0.08f*scale;
    radius[0][1] = 0.15f*scale;
    // I
    radius[1][0] = 0.05f*scale;
    radius[1][1] = 0.1f*scale;

    // row <- column
    _float sTypeMat[NTYPE][NTYPE];
    sTypeMat[0][0] = 1.0f;
    sTypeMat[0][1] = 4.0f;
    sTypeMat[1][0] = 1.0f;
    sTypeMat[1][1] = 4.0f;

    //upper limit of sparsity stricted to NTYPE
    _float pTypeMat[NTYPE][NTYPE];
    pTypeMat[0][0] = 0.15f;
    pTypeMat[0][1] = 0.15f;
    pTypeMat[1][0] = 0.5f;
    pTypeMat[1][1] = 0.5f;

    // 
    unsigned int nTypeMat[NTYPE][NTYPE];
    nTypeMat[0][0] = 400+20*5; // mean + std*5
    nTypeMat[0][1] = 100+10*5;
    nTypeMat[1][0] = 400+20*5;
    nTypeMat[1][1] = 100+10*5;
    
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
    _float max_radius = 0.4f*scale;
    
	std::string posfn = dir + theme + "_3d_pos.bin";
    pos_file.open(posfn, std::ios::in|std::ios::binary);
	if (!pos_file.is_open()) {
		std::cout << "failed to open pos file:" << posfn << "\n";
		return EXIT_FAILURE;
	}
    conMat_file.open(theme + "_conMat.bin", std::ios::out | std::ios::binary);
    conVec_file.open(theme + "_conVec.bin", std::ios::out | std::ios::binary);
    size_t d_memorySize, memorySize = 0;

    // read from file cudaMemcpy to device
    _float *pos;
        memorySize += usingPosDim*networkSize*sizeof(_float);
	
	// to receive from device
    unsigned long outputSize = 0;

    _float *block_x, *block_y; // nblock
        memorySize += 2*nblock*sizeof(_float);
        outputSize += 2*nblock*sizeof(_float);

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
        memorySize += NTYPE*networkSize*sizeof(_float);
        outputSize += NTYPE*networkSize*sizeof(_float);

    
	printf("need to allocate %f MB memory on host\n", static_cast<float>(memorySize)/1024/1024);
	void *cpu_chunk = malloc(memorySize);
	assert(cpu_chunk);

    pos = (_float*) cpu_chunk;
    block_x = pos + usingPosDim*networkSize;
    block_y = block_x + nblock;
    preType = (unsigned int*) (block_y + nblock);
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
    d_memorySize = memorySize;
    // init by kernel, reside on device only
    _float *rden, *raxn; // NTYPE
		d_memorySize += 2*networkSize*sizeof(_float);

    _float *dden, *daxn;
        d_memorySize += 2*networkSize*sizeof(_float);

    _float *preTypeS;
        d_memorySize += NTYPE*networkSize*sizeof(_float);

    _float *preTypeP;
        d_memorySize += NTYPE*networkSize*sizeof(_float);

    unsigned int *preTypeN;
        d_memorySize += NTYPE*networkSize*sizeof(unsigned int);

    curandStateMRG32k3a* state;
        d_memorySize += networkSize*sizeof(curandStateMRG32k3a);

    // init by cudaMemcpy for kernel , reside on device only
    _float *d_sTypeMat;
        d_memorySize += NTYPE*NTYPE*sizeof(_float);

    _float *d_pTypeMat;
        d_memorySize += NTYPE*NTYPE*sizeof(_float);

    unsigned int *d_nTypeMat;
        d_memorySize += NTYPE*NTYPE*sizeof(unsigned int);

    // init by cudaMemcpy
    _float *d_pos;

    // output to host
    _float *d_block_x, *d_block_y;
    unsigned int *d_preType;
    _float *d_conMat, *d_conVec;
    _float *d_delayMat, *d_delayVec;
    unsigned int *d_vecID;
    unsigned int *d_neighborBlockId, *d_nNeighborBlock;
    unsigned int *d_preTypeConnected, *d_preTypeAvail;
    _float *d_preTypeStrSum;
    void *gpu_chunk;
	printf("need to allocate %f MB memory on device\n", static_cast<float>(d_memorySize) / 1024 / 1024);
    CUDA_CALL(cudaMalloc((void**)&gpu_chunk, d_memorySize));

    rden = (_float*) gpu_chunk; 
    raxn = rden + networkSize;
	dden = raxn + networkSize;
	daxn = dden + networkSize;
    preTypeS = daxn + networkSize;
    preTypeP = preTypeS + NTYPE*networkSize;
    preTypeN = (unsigned int*) preTypeP + NTYPE*networkSize;
    state = (curandStateMRG32k3a*) (preTypeN + NTYPE*networkSize);
    d_sTypeMat = (_float*) (state + networkSize);
    d_pTypeMat = d_sTypeMat +NTYPE*NTYPE;
    d_nTypeMat = (unsigned int*) (d_pTypeMat + NTYPE*NTYPE);

    d_pos = (_float*) (d_nTypeMat + NTYPE*NTYPE);

    d_block_x = d_pos + usingPosDim*networkSize; 
    d_block_y = d_block_x + nblock;
    d_preType = (unsigned int*) (d_block_y + nblock);
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

	assert(static_cast<void*>((char*)gpu_chunk + d_memorySize) == static_cast<void*>(d_preTypeStrSum + NTYPE * networkSize));

    double* tmp = new double[networkSize*usingPosDim];
    pos_file.read(reinterpret_cast<char*>(tmp), usingPosDim*networkSize*sizeof(double));
	for (unsigned int i = 0; i < networkSize * usingPosDim; i++) {
		pos[i] = static_cast<_float>(reinterpret_cast<double*>(tmp)[i]);
	}
	delete[]tmp;
    unsigned int localHeapSize = sizeof(_float)*networkSize*nPotentialNeighbor*blockSize;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, localHeapSize*1.5);
    printf("heap size preserved %f Mb\n", localHeapSize*1.5/1024/1024);
    cudaStream_t s0, s1, s2;
    cudaEvent_t i0, i1, i2;
    cudaEventCreate(&i0);
    cudaEventCreate(&i1);
    cudaEventCreate(&i2);
    CUDA_CALL(cudaStreamCreate(&s0));
    CUDA_CALL(cudaStreamCreate(&s1));
    CUDA_CALL(cudaStreamCreate(&s2));
    CUDA_CALL(cudaMemcpy(d_pos, pos, usingPosDim*networkSize*sizeof(_float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_sTypeMat, sTypeMat, NTYPE*NTYPE*sizeof(_float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pTypeMat, pTypeMat, NTYPE*NTYPE*sizeof(_float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_nTypeMat, nTypeMat, NTYPE*NTYPE*sizeof(_float), cudaMemcpyHostToDevice));
    initialize<<<nblock, blockSize, 0, s0>>>(state, 
											 d_preType, 
											 rden, 
											 raxn, 
											 dden, 
											 daxn, 
											 d_sTypeMat,
											 d_pTypeMat,
											 d_nTypeMat,
											 preTypeS, 
											 preTypeP, 
											 preTypeN, 
											 init_pack, seed, networkSize);
	CUDA_CHECK();
	//CUDA_CALL(cudaEventRecord(i1, s1));
	//CUDA_CALL(cudaEventSynchronize(i1));
    printf("initialzied\n");
    unsigned int shared_mem;
    shared_mem = 2*warpSize*sizeof(_float);
    cal_blockPos<<<nblock, blockSize, shared_mem, s1>>>(d_pos, 
														d_block_x, 
														d_block_y, 
														networkSize);
	CUDA_CHECK();
	CUDA_CALL(cudaEventRecord(i1, s1));
	CUDA_CALL(cudaEventSynchronize(i1));
    printf("block centers calculated\n");
	shared_mem = sizeof(unsigned int);
    get_neighbor_blockId<<<nblock, blockSize, shared_mem, s0>>>(d_block_x, 
																d_block_y, 
																d_neighborBlockId, 
																d_nNeighborBlock, 
																max_radius, nPotentialNeighbor);
	CUDA_CHECK();
	CUDA_CALL(cudaEventRecord(i1, s1));
	CUDA_CALL(cudaEventSynchronize(i1));
    printf("neighbor blocks acquired\n");
	//CUDA_CALL(cudaEventRecord(i0, s0));
	//CUDA_CALL(cudaEventSynchronize(i0));
	//CUDA_CALL(cudaEventSynchronize(i1));
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
																dden, 
																daxn, 
																state, 
																networkSize, neighborSize, nPotentialNeighbor, speedOfThought);
	CUDA_CHECK();
	CUDA_CALL(cudaEventRecord(i0, s0));
	CUDA_CALL(cudaEventSynchronize(i0));
    printf("connectivity constructed\n");
	CUDA_CALL(cudaMemcpy(block_x, d_block_x, outputSize, cudaMemcpyDeviceToHost)); // the whole chunk of output
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
    CUDA_CALL(cudaFree(gpu_chunk));
	free(cpu_chunk);
    return 0;
}
