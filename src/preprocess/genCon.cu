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
    if (argc < 7) {
        printf("usage: arg #1: the directory to read neuron positions\n");
        printf("     : arg #2: theme name to identify newly created data files\n");
        printf("     : arg #3: number of blocks (1024 neurons per block)\n");
        printf("     : arg #4: scaling ratio of the neurites' lengths\n");
        printf("     : arg #5: max radius (center to center) to include neighboring blocks\n");
        printf("     : arg #6: max neighboring blocks allowed\n");
    }
    _float scale;
    _float max_radius;
    unsigned int nblock;
    unsigned int nPotentialNeighbor;
    std::string dir(argv[1]);
    std::string theme(argv[2]);
	sscanf(argv[3], "%f", &nblock);
	sscanf(argv[4], "%f", &scale);
	sscanf(argv[5], "%f", &max_radius);
	sscanf(argv[6], "%u", &nPotentialNeighbor);
    std::ofstream mat_file, vec_file;
    std::ofstream blockPos_file, neighborBlock_file;
    std::ofstream stats_file;
    std::ofstream posR_file;
    unsigned int networkSize = nblock*blockSize;
    unsigned int neighborSize = 100;
    unsigned int usingPosDim = 2;

    _float radius[NTYPE][2];
    _float neuron_type_acc_count[NTYPE+1];
	_float den_axn[NTYPE];
	_float den_den[NTYPE];
    // E
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

    // fixed number of connections
    unsigned int nTypeMat[NTYPE][NTYPE];
    //nTypeMat[0][0] = 400+20*5; // mean + std*5
    //nTypeMat[0][1] = 100+10*5;
    //nTypeMat[1][0] = 400+20*5;
    //nTypeMat[1][1] = 100+10*5;
    nTypeMat[0][0] = 40; // mean + std*5
    nTypeMat[0][1] = 10;
    nTypeMat[1][0] = 40;
    nTypeMat[1][1] = 10;
    
    // NTYPE
    neuron_type_acc_count[0] = 0;
    neuron_type_acc_count[1] = 768;
    neuron_type_acc_count[2] = 1024;
    assert(neuron_type_acc_count[NTYPE] == blockSize);

	// density of axons and dendrites
    den_axn[0] = 1.0;
    den_axn[1] = 2.0;

    den_den[0] = 1.0;
    den_den[1] = 2.0;

    initialize_package init_pack(radius, neuron_type_acc_count, den_axn, den_den);
    _float speedOfThought = 1.0f; // mm/ms
    
	std::string posfn = dir + theme + "_3d_pos.bin";
    pos_file.open(posfn, std::ios::in|std::ios::binary);
	if (!pos_file.is_open()) {
		std::cout << "failed to open pos file:" << posfn << "\n";
		return EXIT_FAILURE;
	}
    mat_file.open(dir + theme + "_mat.bin", std::ios::out | std::ios::binary);
    vec_file.open(dir + theme + "_vec.bin", std::ios::out | std::ios::binary);
    blockPos_file.open(dir + theme + "_blkPos.bin", std::ios::out | std::ios::binary);
    neighborBlock_file.open(dir + theme + "_neighborBlk.bin", std::ios::out | std::ios::binary);
    stats_file.open(dir + theme + "_stats.bin", std::ios::out | std::ios::binary);
    posR_file.open(dir + theme + "_reshaped_pos.bin", std::ios::out|std::ios::binary);
    size_t d_memorySize, memorySize = 0;

    // read from file cudaMemcpy to device
    _float* __restrict__ pos;
        memorySize += usingPosDim*networkSize*sizeof(_float);
	
	// to receive from device
    unsigned long outputSize = 0;

    _float* __restrict__ block_x;
    _float* __restrict__ block_y; // nblock
        memorySize += 2*nblock*sizeof(_float);
        outputSize += 2*nblock*sizeof(_float);

    unsigned int* __restrict__ preType;
        memorySize += networkSize*sizeof(unsigned int);
        outputSize += networkSize*sizeof(unsigned int);

    _float* __restrict__ conMat;
    _float* __restrict__ delayMat;
        memorySize += 2*blockSize*blockSize*nblock*sizeof(_float);
        outputSize += 2*blockSize*blockSize*nblock*sizeof(_float);

    _float* __restrict__ conVec;
    _float* __restrict__ delayVec;
        memorySize += 2*networkSize*neighborSize*sizeof(_float);
        outputSize += 2*networkSize*neighborSize*sizeof(_float);

    unsigned int* __restrict__ vecID;
        memorySize += networkSize*neighborSize*sizeof(unsigned int);
        outputSize += networkSize*neighborSize*sizeof(unsigned int);

    unsigned int* __restrict__ nVec;
        memorySize += networkSize*sizeof(unsigned int);
        outputSize += networkSize*sizeof(unsigned int);

    unsigned int* __restrict__ neighborBlockId;
    unsigned int* __restrict__ nNeighborBlock;
        memorySize += (nPotentialNeighbor + 1)*nblock*sizeof(unsigned int);
        outputSize += (nPotentialNeighbor + 1)*nblock*sizeof(unsigned int);

    unsigned int* __restrict__ preTypeConnected;
    unsigned int* __restrict__ preTypeAvail; // NTYPE*networkSize
        memorySize += 2*NTYPE*networkSize*sizeof(unsigned int);
        outputSize += 2*NTYPE*networkSize*sizeof(unsigned int);

    _float* __restrict__ preTypeStrSum;
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
    nVec = vecID + networkSize*neighborSize;
    neighborBlockId = nVec + networkSize;
    nNeighborBlock = neighborBlockId + nPotentialNeighbor*nblock;
    preTypeConnected = nNeighborBlock + nblock; 
    preTypeAvail = preTypeConnected + NTYPE*networkSize;
    preTypeStrSum = (_float*) (preTypeAvail + NTYPE*networkSize);

	assert(static_cast<void*>((char*)cpu_chunk + memorySize) == static_cast<void*>(preTypeStrSum + NTYPE * networkSize));

    // ========== GPU mem ============
    d_memorySize = memorySize;
    // init by kernel, reside on device only
    _float* __restrict__ rden;
    _float* __restrict__ raxn; // NTYPE
		d_memorySize += 2*networkSize*sizeof(_float);

    _float* __restrict__ dden;
    _float* __restrict__ daxn;
        d_memorySize += 2*networkSize*sizeof(_float);

    _float* __restrict__ preTypeS;
        d_memorySize += NTYPE*networkSize*sizeof(_float);

    _float* __restrict__ preTypeP;
        d_memorySize += NTYPE*networkSize*sizeof(_float);

    unsigned int* __restrict__ preTypeN;
        d_memorySize += NTYPE*networkSize*sizeof(unsigned int);

    curandStateMRG32k3a* __restrict__ state;
        d_memorySize += networkSize*sizeof(curandStateMRG32k3a);

    // init by cudaMemcpy for kernel , reside on device only
    _float* __restrict__ d_sTypeMat;
        d_memorySize += NTYPE*NTYPE*sizeof(_float);

    _float* __restrict__ d_pTypeMat;
        d_memorySize += NTYPE*NTYPE*sizeof(_float);

    unsigned int* __restrict__ d_nTypeMat;
        d_memorySize += NTYPE*NTYPE*sizeof(unsigned int);

    // init by cudaMemcpy
    _float* __restrict__ d_pos;

    // output to host
    _float* __restrict__ d_block_x;
    _float* __restrict__ d_block_y;
    unsigned int* __restrict__ d_preType;
    _float* __restrict__ d_conMat;
    _float* __restrict__ d_conVec;
    _float* __restrict__ d_delayMat;
    _float* __restrict__ d_delayVec;
    unsigned int* __restrict__ d_vecID;
    unsigned int* __restrict__ d_nVec;
    unsigned int* __restrict__ d_neighborBlockId;
    unsigned int* __restrict__ d_nNeighborBlock;
    unsigned int* __restrict__ d_preTypeConnected;
    unsigned int* __restrict__ d_preTypeAvail;
    _float* __restrict__ d_preTypeStrSum;
    void* __restrict__ gpu_chunk;
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
    d_nVec = d_vecID + networkSize*neighborSize;
    d_neighborBlockId = d_nVec + networkSize;
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
	//CUDA_CALL(cudaEventSynchronizeudaEventRecord(i1, s1));
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
                                                                d_nVec,
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
    // output to binary data files
    mat_file.write((char*)conMat, nblock*blockSize*blockSize*sizeof(_float));
    mat_file.write((char*)delayMat, nblock*blockSize*blockSize*sizeof(_float));
    mat_file.close();
    
    vec_file.write((char*)nVec, networkSize*sizeof(unsigned int));
    for (unsigned int i=0; i<networkSize; i++) {
        vec_file.write((char*)&(vecID[i*neighborSize]), nVec[i]*sizeof(unsigned int));
        vec_file.write((char*)&(conVec[i*neighborSize]), nVec[i]*sizeof(_float));
        vec_file.write((char*)&(delayVec[i*neighborSize]), nVec[i]*sizeof(_float));
    }
    vec_file.close();

    blockPos_file.write((char*)block_x, nblock*sizeof(_float));
    blockPos_file.write((char*)block_y, nblock*sizeof(_float));
    blockPos_file.close();

    neighborBlock_file.write((char*)nNeighborBlock, nblock*sizeof(unsigned int));
    for (unsigned int i=0; i<nblock; i++) {
        neighborBlock_file.write((char*)&(neighborBlockId[i*nPotentialNeighbor]), nNeighborBlock[i]*sizeof(unsigned int));
    }
    neighborBlock_file.close();

    stats_file.write((char*)preTypeConnected, NTYPE*networkSize*sizeof(unsigned int));
    stats_file.write((char*)preTypeAvail, NTYPE*networkSize*sizeof(unsigned int));
    stats_file.write((char*)preTypeStrSum, NTYPE*networkSize*sizeof(_float));
    stats_file.close();

    posR_file.write((char*)pos, networkSize*usingPosDim*sizeof(_float));
    posR_file.close();

    CUDA_CALL(cudaFree(gpu_chunk));
	free(cpu_chunk);
    return 0;
}
