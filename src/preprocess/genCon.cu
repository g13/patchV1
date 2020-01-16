#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>
#include <ctime>
#include <cmath>
#include <fenv.h>
#include <boost/program_options.hpp>

#include "connect.h"
#include "../types.h"
#include "../util/po.h" // custom validator for reading vector in the configuration file
#include "../util/util.h"
// TODO: polish the IO and file 
// TODO: L, R separation

/* TODO:
template <typename T, typename I>
void check_statistics(T* array, I n, T &max, T &min, T &mean, T &std) {

}
*/

int main(int argc, char *argv[])
{
	namespace po = boost::program_options;
    using namespace std;
	using std::string;

    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
    printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);
    printf("total global memory: %f Mb.\n", deviceProps.totalGlobalMem/1024.0/1024.0);

    BigSize seed
    Size nPotentialNeighbor, nType;
    string V1_pos_filename, V1_V1_idList_filename, V1_V1_sList_filename, theme;
    Float dscale, blockROI;
    vector<Float> radius;
    vector<Size> typeAccCount;
	vector<Float> dDend, dAxon;
    vector<Float> sTypeMat, pTypeMat;
    vector<Size> nTypeMat;

	po::options_description generic_opt("Generic options");
	generic_opt.add_options()
        ("seed", po::value<BigSize>(&seed)->default_value(7641807), "seed for RNG")
		("cfg_file,c", po::value<string>()->default_value("patchV1.cfg"), "filename for configuration file")
		("help,h", "print usage");
	po::options_description input_opt("output options");
	input_opt.add_options()
        ("nType", po::value<Size>(&nType), "types of V1 neurons, e.g., excitatory, inhibitory")
        ("radius", po::value<vector<Float>>(&raidus),  "a matrix of dendrites and axons' radii, size of [nType,2]")
        ("dScale",po::value<Float>(&dScale)->default_value(1.0),"a scaling ratio of all the neurites' lengths <radius>")
        ("typeAccCount",po::value<vector<Size>>(&typeAccCount), "neuronal types' discrete accumulative distribution, size of [nType+1]")
        ("dDend", po::value<vector<Float>>(&dDend), "dendrites' densities, vector of [nType]")
        ("dAxon", po::value<vector<Float>>(&dAxon), "axons' densities, vector of [nType]")
        ("sTypeMat", po::value<vector<Float>>(&sTypeMat), "connection matrix between neuronal types in strength, size of [nType, nType]")
        ("pTypeMat", po::value<vector<Float>>(&pTypeMat), "connection matrix between neuronal types in connection prob, size of [nType, nType]")
        ("nTypeMat", po::value<vector<Size>>(&nTypeMat), "connection matrix between neuronal types in number of neurons, size of [nType, nType]")
        ("blockROI", po::value<Float>(&blockROI), "max radius (center to center) to include neighboring blocks");
        ("fV1_pos", po::vlaue<string>(&V1_pos_filename)->default_value("V1_pos.bin"), "the directory to read neuron positions");

	po::options_description output_opt("output options");
	output_opt.add_options()
        ("theme", po::value<string>(&theme)->default_value(""), "a name to be associated with the generated connection profile")
        ("fV1_V1_ID", po::value<string>(&V1_V1_idList_filename)->default_value("V1_V1_idList.bin"), "file that stores V1 to V1 connection by ID")
        ("fV1_V1_s", po::value<string>(&V1_V1_sList_filename)->default_value("V1_V1_sList.bin"), "file that stores V1 to V1 connection by strength")
        ("nPotentialNeighbor", po::value<Size>(&nPotentialNeighbor), "the preserved size of the array that store the neighboring blocks ID");

	po::options_description cmdline_options;
	cmdline_options.add(generic_opt).add(input_opt).add(output_opt);

	po::options_description config_file_options;
	config_file_options.add(generic_opt).add(input_opt).add(output_opt);

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
	if (vm.count("help")) {
		cout << cmdline_options << "\n";
		return EXIT_SUCCESS;
	}
	string cfg_filename = vm["cfg_file"].as<string>();
	ifstream cfg_file{cfg_filename.c_str()};
	if (cfg_file) {
		po::store(po::parse_config_file(cfg_file, config_file_options), vm);
		cout << "Using configuration file: " << cfg_filename << "\n";
	} else {
		cout << "No configuration file is given, default values are used for non-specified parameters\n";
	}
	po::notify(vm);

    ifstream fV1_pos;
    ofstream fV1_V1_ID, fV1_V1_s;
    Size nblock;
    ofstream mat_file, vec_file;
    ofstream blockPos_file, neighborBlock_file;
    ofstream stats_file;
    ofstream posR_file;
    Size networkSize = nblock*blockSize;
    Size neighborSize = 100;
    Size usingPosDim = 2;

    if (radius.size() != nType*2) {
        cout << "size of radius: " << radius.size() << " cannot be made into the shape of [" << nType << ",2]\n"; 
        return EXIT_FAILURE;
    } else {
        // adjust the scale
        for (Float &r: radius){
            r *= dScale;
        }
    }
    // row <- column

    if (typeAccCount.size() != nType + 1) {
        cout << "the accumulative distribution of neuronal type <typeAccCount> has size of " typeAccCount.size() << ", should be " << nType + 1 << ",  <nType> + 1\n";
        return EXIT_FAILURE;
    }
    // TODO: types that shares a smaller portion than 1/1024
    if (typeAccCount.end() != blockSize) {

        return EXIT_FAILURE;
    }


    initialize_package init_pack(radius, typeAccCount, dAxon, dDend);
    Float speedOfThought = 1.0f; // mm/ms
    
    fV1_pos.open(V1_pos_filename, ios::in|ios::binary);
	if (!fV1_pos) {
		cout << "failed to open pos file:" << V1_pos_filename << "\n";
		return EXIT_FAILURE;
	}
    mat_file.open(dir + theme + "_mat.bin", ios::out | ios::binary);
    vec_file.open(dir + theme + "_vec.bin", ios::out | ios::binary);
    blockPos_file.open(dir + theme + "_blkPos.bin", ios::out | ios::binary);
    neighborBlock_file.open(dir + theme + "_neighborBlk.bin", ios::out | ios::binary);
    stats_file.open(dir + theme + "_stats.bin", ios::out | ios::binary);
    posR_file.open(dir + theme + "_reshaped_pos.bin", ios::out|ios::binary);
    size_t d_memorySize, memorySize = 0;

    // read from file cudaMemcpy to device
    Float* __restrict__ pos;
        memorySize += usingPosDim*networkSize*sizeof(Float);
	
	// to receive from device
    unsigned long outputSize = 0;

    Float* __restrict__ block_x;
    Float* __restrict__ block_y; // nblock
        memorySize += 2*nblock*sizeof(Float);
        outputSize += 2*nblock*sizeof(Float);

    Size* __restrict__ preType;
        memorySize += networkSize*sizeof(Size);
        outputSize += networkSize*sizeof(Size);

    Float* __restrict__ conMat;
    Float* __restrict__ delayMat;
        memorySize += 2*blockSize*blockSize*nblock*sizeof(Float);
        outputSize += 2*blockSize*blockSize*nblock*sizeof(Float);

    Float* __restrict__ conVec;
    Float* __restrict__ delayVec;
        memorySize += 2*networkSize*neighborSize*sizeof(Float);
        outputSize += 2*networkSize*neighborSize*sizeof(Float);

    Size* __restrict__ vecID;
        memorySize += networkSize*neighborSize*sizeof(Size);
        outputSize += networkSize*neighborSize*sizeof(Size);

    Size* __restrict__ nVec;
        memorySize += networkSize*sizeof(Size);
        outputSize += networkSize*sizeof(Size);

    Size* __restrict__ neighborBlockId;
    Size* __restrict__ nNeighborBlock;
        memorySize += (nPotentialNeighbor + 1)*nblock*sizeof(Size);
        outputSize += (nPotentialNeighbor + 1)*nblock*sizeof(Size);

    Size* __restrict__ preTypeConnected;
    Size* __restrict__ preTypeAvail; // nType*networkSize
        memorySize += 2*nType*networkSize*sizeof(Size);
        outputSize += 2*nType*networkSize*sizeof(Size);

    Float* __restrict__ preTypeStrSum;
        memorySize += nType*networkSize*sizeof(Float);
        outputSize += nType*networkSize*sizeof(Float);

    
	printf("need to allocate %f MB memory on host\n", static_cast<float>(memorySize)/1024/1024);
	void *cpu_chunk = malloc(memorySize);
	assert(cpu_chunk);

    pos = (Float*) cpu_chunk;
    block_x = pos + usingPosDim*networkSize;
    block_y = block_x + nblock;
    preType = (Size*) (block_y + nblock);
    conMat = (Float*) (preType + networkSize); 
    delayMat = conMat + blockSize*blockSize*nblock;
    conVec = delayMat + blockSize*blockSize*nblock; 
    delayVec = conVec + networkSize*neighborSize;
    vecID = (Size*) (delayVec + networkSize*neighborSize);
    nVec = vecID + networkSize*neighborSize;
    neighborBlockId = nVec + networkSize;
    nNeighborBlock = neighborBlockId + nPotentialNeighbor*nblock;
    preTypeConnected = nNeighborBlock + nblock; 
    preTypeAvail = preTypeConnected + nType*networkSize;
    preTypeStrSum = (Float*) (preTypeAvail + nType*networkSize);

	assert(static_cast<void*>((char*)cpu_chunk + memorySize) == static_cast<void*>(preTypeStrSum + nType * networkSize));

    // ========== GPU mem ============
    d_memorySize = memorySize;
    // init by kernel, reside on device only
    Float* __restrict__ rden;
    Float* __restrict__ raxn; // nType
		d_memorySize += 2*networkSize*sizeof(Float);

    Float* __restrict__ dden;
    Float* __restrict__ daxn;
        d_memorySize += 2*networkSize*sizeof(Float);

    Float* __restrict__ preTypeS;
        d_memorySize += nType*networkSize*sizeof(Float);

    Float* __restrict__ preTypeP;
        d_memorySize += nType*networkSize*sizeof(Float);

    Size* __restrict__ preTypeN;
        d_memorySize += nType*networkSize*sizeof(Size);

    curandStateMRG32k3a* __restrict__ state;
        d_memorySize += networkSize*sizeof(curandStateMRG32k3a);

    // init by cudaMemcpy for kernel , reside on device only
    Float* __restrict__ d_sTypeMat;
        d_memorySize += nType*nType*sizeof(Float);

    Float* __restrict__ d_pTypeMat;
        d_memorySize += nType*nType*sizeof(Float);

    Size* __restrict__ d_nTypeMat;
        d_memorySize += nType*nType*sizeof(Size);

    // init by cudaMemcpy
    Float* __restrict__ d_pos;

    // output to host
    Float* __restrict__ d_block_x;
    Float* __restrict__ d_block_y;
    Size* __restrict__ d_preType;
    Float* __restrict__ d_conMat;
    Float* __restrict__ d_conVec;
    Float* __restrict__ d_delayMat;
    Float* __restrict__ d_delayVec;
    Size* __restrict__ d_vecID;
    Size* __restrict__ d_nVec;
    Size* __restrict__ d_neighborBlockId;
    Size* __restrict__ d_nNeighborBlock;
    Size* __restrict__ d_preTypeConnected;
    Size* __restrict__ d_preTypeAvail;
    Float* __restrict__ d_preTypeStrSum;
    void* __restrict__ gpu_chunk;
	printf("need to allocate %f MB memory on device\n", static_cast<float>(d_memorySize) / 1024 / 1024);
    CUDA_CALL(cudaMalloc((void**)&gpu_chunk, d_memorySize));

    rden = (Float*) gpu_chunk; 
    raxn = rden + networkSize;
	dden = raxn + networkSize;
	daxn = dden + networkSize;
    preTypeS = daxn + networkSize;
    preTypeP = preTypeS + nType*networkSize;
    preTypeN = (Size*) preTypeP + nType*networkSize;
    state = (curandStateMRG32k3a*) (preTypeN + nType*networkSize);
    d_sTypeMat = (Float*) (state + networkSize);
    d_pTypeMat = d_sTypeMat +nType*nType;
    d_nTypeMat = (Size*) (d_pTypeMat + nType*nType);

    d_pos = (Float*) (d_nTypeMat + nType*nType);

    d_block_x = d_pos + usingPosDim*networkSize; 
    d_block_y = d_block_x + nblock;
    d_preType = (Size*) (d_block_y + nblock);
    d_conMat = (Float*) (d_preType + networkSize); 
    d_delayMat = d_conMat + blockSize*blockSize*nblock;
    d_conVec = d_delayMat + blockSize*blockSize*nblock; 
    d_delayVec = d_conVec + networkSize*neighborSize;
    d_vecID = (Size*) d_delayVec + networkSize*neighborSize;
    d_nVec = d_vecID + networkSize*neighborSize;
    d_neighborBlockId = d_nVec + networkSize;
    d_nNeighborBlock = d_neighborBlockId + nPotentialNeighbor*nblock;
    d_preTypeConnected = d_nNeighborBlock + nblock;
    d_preTypeAvail = d_preTypeConnected + nType*networkSize;
    d_preTypeStrSum = (Float*) (d_preTypeAvail + nType*networkSize);

	assert(static_cast<void*>((char*)gpu_chunk + d_memorySize) == static_cast<void*>(d_preTypeStrSum + nType * networkSize));

    double* tmp = new double[networkSize*usingPosDim];
    pos_file.read(reinterpret_cast<char*>(tmp), usingPosDim*networkSize*sizeof(double));
	for (Size i = 0; i < networkSize * usingPosDim; i++) {
		pos[i] = static_cast<Float>(reinterpret_cast<double*>(tmp)[i]);
	}
	delete[]tmp;
    Size localHeapSize = sizeof(Float)*networkSize*nPotentialNeighbor*blockSize;
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
    CUDA_CALL(cudaMemcpy(d_pos, pos, usingPosDim*networkSize*sizeof(Float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_sTypeMat, sTypeMat, nType*nType*sizeof(Float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pTypeMat, pTypeMat, nType*nType*sizeof(Float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_nTypeMat, nTypeMat, nType*nType*sizeof(Float), cudaMemcpyHostToDevice));
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
    Size shared_mem;
    shared_mem = 2*warpSize*sizeof(Float);
    cal_blockPos<<<nblock, blockSize, shared_mem, s1>>>(d_pos, 
														d_block_x, 
														d_block_y, 
														networkSize);
	CUDA_CHECK();
	CUDA_CALL(cudaEventRecord(i1, s1));
	CUDA_CALL(cudaEventSynchronize(i1));
    printf("block centers calculated\n");
	shared_mem = sizeof(Size);
    get_neighbor_blockId<<<nblock, blockSize, shared_mem, s0>>>(d_block_x, 
																d_block_y, 
																d_neighborBlockId, 
																d_nNeighborBlock, 
																blockROI, nPotentialNeighbor);
	CUDA_CHECK();
	CUDA_CALL(cudaEventRecord(i1, s1));
	CUDA_CALL(cudaEventSynchronize(i1));
    printf("neighbor blocks acquired\n");
	//CUDA_CALL(cudaEventRecord(i0, s0));
	//CUDA_CALL(cudaEventSynchronize(i0));
	//CUDA_CALL(cudaEventSynchronize(i1));
	//CUDA_CALL(cudaEventSynchronize(i2));
    shared_mem = blockSize*sizeof(Float) + blockSize*sizeof(Float) + blockSize*sizeof(Size);
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
    mat_file.write((char*)conMat, nblock*blockSize*blockSize*sizeof(Float));
    mat_file.write((char*)delayMat, nblock*blockSize*blockSize*sizeof(Float));
    mat_file.close();
    
    vec_file.write((char*)nVec, networkSize*sizeof(Size));
    for (Size i=0; i<networkSize; i++) {
        vec_file.write((char*)&(vecID[i*neighborSize]), nVec[i]*sizeof(Size));
        vec_file.write((char*)&(conVec[i*neighborSize]), nVec[i]*sizeof(Float));
        vec_file.write((char*)&(delayVec[i*neighborSize]), nVec[i]*sizeof(Float));
    }
    vec_file.close();

    blockPos_file.write((char*)block_x, nblock*sizeof(Float));
    blockPos_file.write((char*)block_y, nblock*sizeof(Float));
    blockPos_file.close();

    neighborBlock_file.write((char*)nNeighborBlock, nblock*sizeof(Size));
    for (Size i=0; i<nblock; i++) {
        neighborBlock_file.write((char*)&(neighborBlockId[i*nPotentialNeighbor]), nNeighborBlock[i]*sizeof(Size));
    }
    neighborBlock_file.close();

    stats_file.write((char*)preTypeConnected, nType*networkSize*sizeof(Size));
    stats_file.write((char*)preTypeAvail, nType*networkSize*sizeof(Size));
    stats_file.write((char*)preTypeStrSum, nType*networkSize*sizeof(Float));
    stats_file.close();

    posR_file.write((char*)pos, networkSize*usingPosDim*sizeof(Float));
    posR_file.close();

    CUDA_CALL(cudaFree(gpu_chunk));
	free(cpu_chunk);
    return 0;
}
