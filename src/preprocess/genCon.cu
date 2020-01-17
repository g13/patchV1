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
// TODO***: L, R separation

/* TODO*:
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
    string V1_pos_filename, theme;
	string V1_conMat_filename, V1_delayMat_filename;
	string V1_vec_filename;
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
        ("rDend", po::value<vector<Float>>(&rDend),  "a vector of dendritic extensions' radius, size of [nType]")
        ("rAxon", po::value<vector<Float>>(&rAxon),  "a vector of axonic extensions' radius, size of [nType]")
        ("dScale",po::value<Float>(&dScale)->default_value(1.0),"a scaling ratio of all the neurites' lengths <radius>")
        ("typeAccCount",po::value<vector<Size>>(&typeAccCount), "neuronal types' discrete accumulative distribution, size of [nType+1]")
        ("dDend", po::value<vector<Float>>(&dDend), "dendrites' densities, vector of [nType]")
        ("dAxon", po::value<vector<Float>>(&dAxon), "axons' densities, vector of [nType]")
        ("sTypeMat", po::value<vector<Float>>(&sTypeMat), "connection matrix between neuronal types in strength, size of [nType, nType], row(post) <- column(pre)")
        ("pTypeMat", po::value<vector<Float>>(&pTypeMat), "connection prob matrix between neuronal types based on the type only (not including the density of dendrites and axons or the shared physical space) size of [nType, nType], row(post) <- column(pre)")
        ("nTypeMat", po::value<vector<Size>>(&nTypeMat), "connection matrix between neuronal types in number of neurons, size of [nType, nType], row(post) <- column(pre)")
        ("blockROI", po::value<Float>(&blockROI), "max radius (center to center) to include neighboring blocks");
    	("usingPosDim", po::value<Size>(&usingPosDim)->default_value(2), "using <2>D coord. or <3>D coord. when calculating distance between neurons, influencing how the position data is read") 
        ("fV1_pos", po::vlaue<string>(&V1_pos_filename)->default_value("V1_pos.bin"), "the directory to read neuron positions");

	po::options_description output_opt("output options");
	output_opt.add_options()
        ("theme", po::value<string>(&theme)->default_value(""), "a name to be associated with the generated connection profile")
        ("fV1_conMat", po::value<string>(&V1_conMat_filename)->default_value("V1_conMat.bin"), "file that stores V1 to V1 connection within the neighboring blocks")
        ("fV1_delayMat", po::value<string>(&V1_delayMat_filename)->default_value("V1_delayMat.bin"), "file that stores V1 to V1 transmission delay within the neighboring blocks")
        ("fV1_vec", po::value<string>(&V1_vec_filename)->default_value("V1_vec.bin"), "file that stores V1 to V1 connection ID, strength and transmission delay outside the neighboring blocks")
		("fBlock_pos", po::value<string>(&block_pos_filename)->default_value("block_pos.bin"), "file that stores the center coord of each block")
		("fNeighborBlock", po::value<string>(&neighborBlock_filename)->default_value("neighborBlock.bin"), "file that stores the neighboring blocks' ID for each block")
		("fStats", po::value<string>(&stats_filename)->default_value("conStats.bin"), "file that stores the statistics of connections")
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
    ofstream fV1_conMat, fV1_delayMat, fV1_vec;
    ofstream fBlock_pos, fNeighborBlock;
    ofstream fStats;
    Size neighborSize = 100;

    if (rAxon.size() != nType) {
        cout << "size of rAxon: " << rAxon.size() << " should be " << nType << "\n"; 
        return EXIT_FAILURE;
    } else {
        // adjust the scale
        for (Float &r: rAxon){
            r *= dScale;
        }
    }
    if (rDend.size() != nType) {
    	cout << "size of rDend: " << rDend.size() << " should be " << nType << "\n"; 
        return EXIT_FAILURE;
    } else {
        // adjust the scale
        for (Float &r: rDend){
            r *= dScale;
        }
    }

	// TODO: slide in std. for simple and complex
    if (nTypeMat.size() != nType*nType) {
		cout << "nTypeMat has size of " << nTypeMat.size() << ", should be " << nType*nType << "\n";
		return EXIT_FAILURE;
	}

    if (pTypeMat.size() != nType*nType) {
		cout << "pTypeMat has size of " << pTypeMat.size() << ", should be " << nType*nType << "\n";
		return EXIT_FAILURE;
	}

    if (sTypeMat.size() != nType*nType) {
		cout << "sTypeMat has size of " << sTypeMat.size() << ", should be " << nType*nType << "\n";
		return EXIT_FAILURE;
	}

    if (typeAccCount.size() != nType + 1) {
        cout << "the accumulative distribution of neuronal type <typeAccCount> has size of " typeAccCount.size() << ", should be " << nType + 1 << ",  <nType> + 1\n";
        return EXIT_FAILURE;
    }

    hInitialize_package hInit_pack(nType, typeAccCount, rAxon, rDend, dAxon, dDend, sTypeMat, pTypeMat, nTypeMat);
	initialize_package init_pack(nType, hInit_pack);
	hInit_pack.freeMem();
    Float speedOfThought = 1.0f; // mm/ms
    
    fV1_pos.open(V1_pos_filename, ios::in|ios::binary);
	if (!fV1_pos) {
		cout << "failed to open pos file:" << V1_pos_filename << "\n";
		return EXIT_FAILURE;
	}
    Size nblock, neuronPerBlock, FP_pos, dataDim;
    // read from file cudaMemcpy to device
	
    fV1_pos.read(reinterpret_cast<char*>(&nblock), sizeof(Size));
    fV1_pos.read(reinterpret_cast<char*>(&neuronPerBlock), sizeof(Size));
    Size networkSize = nblock*blockSize;
	fV1_pos.read(reinterpret_cast<char*>(&dataDim), sizeof(Size))
	if (dataDim != usingPosDim) {
		cout << "the dimension of position coord intended is " << usingPosDim << ", data provided from " << V1_pos_filename << " gives " << dataDim << "\n";
		return EXIT_FAILURE;
	}
	fV1_pos.read(reinterpret_cast<char*>(&FP_pos), sizeof(Size))
	if (FP_pos == 8) {
		typedef double pos_dType;
	} else {
		if (FP_pos != 4) {
			cout << "unrecognized floating precision byte size: " << FP_pos << "\n";
			return EXIT_FAILURE;
		}
		typedef float pos_dType;
	}
    vector<pos_dType> pos;
    fV1_pos.read(reinterpret_cast<char*>(&pos), usingPosDim*networkSize*sizeof(pos_dType));

    // TODO: types that shares a smaller portion than 1/neuronPerBlock
    if (typeAccCount.end() != neuronPerBlock) {
		cout << "type acc. dist. end with " << typeAccCount.end() << " should be " << neuronPerBlock << "\n";
        return EXIT_FAILURE;
    }

    fV1_conMat.open(theme + "-" + V1_conMat_filename, ios::out | ios::binary);
	if (!fV1_conMat) {
		cout << "cannot open " << theme + "-" + V1_conMat_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fV1_delayMat.open(theme + "-" + V1_delayMat_filename, ios::out | ios::binary);
	if (!fV1_delayMat) {
		cout << "cannot open " << theme + "-" + V1_delayMat_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fV1_vec.open(theme + "-" + V1_vec_filename, ios::out | ios::binary);
	if (!fV1_vec) {
		cout << "cannot open " << theme + "-" + V1_vec_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fBlock_pos.open(theme + "-" + block_pos_filename, ios::out | ios::binary);
	if (!fBlock_pos) {
		cout << "cannot open " << theme + "-" + block_pos_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fNeighborBlock.open(theme + "-" + neighborBlock_filename, ios::out | ios::binary);
	if (!fNeighborBlock) {
		cout << "cannot open " << theme + "-" + neighborBlock_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fStats.open(theme + "-" + stats_filename, ios::out | ios::binary);
	if (!fStats) {
		cout << "cannot open " << theme + "-" + stats_filename << " to write.\n";
		return EXIT_FAILURE;
	}

	size_t memorySize = 2*nblock*sizeof(Float) + // block_x and y
        				networkSize*sizeof(Size) + // preType
        				2*blockSize*blockSize*nblock*sizeof(Float) + // con and delayMat
        				2*networkSize*neighborSize*sizeof(Float) + // con and delayVec
        				networkSize*neighborSize*sizeof(Size) + // vecID
        				networkSize*sizeof(Size) + // nVec
        				(nPotentialNeighbor + 1)*nblock*sizeof(Size) + // neighborBlockId and nNeighborBlock
        				2*nType*networkSize*sizeof(Size) + // preTypeConnected and *Avail
        				nType*networkSize*sizeof(Float); // preTypeStrSum

	// to receive from device
    size_t outputSize = memorySize;
	printf("need to allocate %f MB memory on host\n", static_cast<float>(memorySize)/1024/1024);
	void *cpu_chunk = malloc(memorySize);
	assert(cpu_chunk);

    block_x = (Float*) cpu_chunk;
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
	size_t host2deviceMem = 2*networkSize*sizeof(Float) + // rden and raxn
     						2*networkSize*sizeof(Float) + // dden and daxn
     						nType*networkSize*sizeof(Float) + // preTypeS
     						nType*networkSize*sizeof(Float) + // preTypeP
     						nType*networkSize*sizeof(Size) + // preTypeN
     						networkSize*sizeof(curandStateMRG32k3a) + //state
     						nType*nType*sizeof(Float) + // d_sTypeMat
     						nType*nType*sizeof(Float) + // d_pTypeMat
     						nType*nType*sizeof(Size); // d_nTypeMat
    size_t d_memorySize = memorySize + usingPosDim*networkSize*sizeof(pos_dType) + host2deviceMem;
    void* __restrict__ gpu_chunk;
	printf("need to allocate %f MB memory on device\n", static_cast<float>(d_memorySize) / 1024 / 1024);
    checkCudaErrors(cudaMalloc((void**)&gpu_chunk, d_memorySize));

    // init by kernel, reside on device only
	// copy and init by the whole chunk from host to device
    Float * __restrict__ rden = (Float*) gpu_chunk; 
    Float * __restrict__ raxn = rden + networkSize;
	Float * __restrict__ dden = raxn + networkSize;
	Float * __restrict__ daxn = dden + networkSize;
    Float * __restrict__ preTypeS = daxn + networkSize;
    Float * __restrict__ preTypeP = preTypeS + nType*networkSize;
    Float * __restrict__ preTypeN = (Size*) preTypeP + nType*networkSize;
    curandStateMRG32k3a* __restrict__ state = (curandStateMRG32k3a*) (preTypeN + nType*networkSize);
    // init by cudaMemcpy , reside on device only
    Float* __restrict__ d_sTypeMat = (Float*) (state + networkSize);
    Float* __restrict__ d_pTypeMat = d_sTypeMat +nType*nType;
    Size*  __restrict__ d_nTypeMat = (Size*) (d_pTypeMat + nType*nType);
	// copy from host to device indivdual chunk
    Float* __restrict__ d_pos = (Float*) (d_nTypeMat + nType*nType);
	// copy by the whole chunk
    // device to host
    Float* __restrict__ d_block_x = (pos_dType*) (d_pos + usingPosDim*networkSize); 
    Float* __restrict__ d_block_y = d_block_x + nblock;
    Size*  __restrict__ d_preType = (Size*) (d_block_y + nblock);
    Float* __restrict__ d_conMat = (Float*) (d_preType + networkSize); 
    Float* __restrict__ d_delayMat = d_conMat + blockSize*blockSize*nblock;
    Float* __restrict__ d_conVec = d_delayMat + blockSize*blockSize*nblock; 
    Float* __restrict__ d_delayVec = d_conVec + networkSize*neighborSize;
    Size*  __restrict__ d_vecID = (Size*) d_delayVec + networkSize*neighborSize;
    Size*  __restrict__ d_nVec = d_vecID + networkSize*neighborSize;
    Size*  __restrict__ d_neighborBlockId = d_nVec + networkSize;
    Size*  __restrict__ d_nNeighborBlock = d_neighborBlockId + nPotentialNeighbor*nblock;
    Size*  __restrict__ d_preTypeConnected = d_nNeighborBlock + nblock;
    Size*  __restrict__ d_preTypeAvail = d_preTypeConnected + nType*networkSize;
    Float* __restrict__ d_preTypeStrSum = (Float*) (d_preTypeAvail + nType*networkSize);

	// check memory address consistency
	assert(static_cast<void*>((char*)gpu_chunk + d_memorySize) == static_cast<void*>(d_preTypeStrSum + nType * networkSize));

	// TODO: check the usage of this
    Size localHeapSize = sizeof(Float)*networkSize*nPotentialNeighbor*blockSize;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, localHeapSize*1.5);
    printf("heap size preserved %f Mb\n", localHeapSize*1.5/1024/1024);

    cudaStream_t s0, s1, s2;
    cudaEvent_t i0, i1, i2;
    cudaEventCreate(&i0);
    cudaEventCreate(&i1);
    cudaEventCreate(&i2);
    checkCudaErrors(cudaStreamCreate(&s0));
    checkCudaErrors(cudaStreamCreate(&s1));
    checkCudaErrors(cudaStreamCreate(&s2));
    checkCudaErrors(cudaMemcpy(d_pos, pos, usingPosDim*networkSize*sizeof(pos_dType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sTypeMat, sTypeMat, nType*nType*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pTypeMat, pTypeMat, nType*nType*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_nTypeMat, nTypeMat, nType*nType*sizeof(Float), cudaMemcpyHostToDevice));
    initialize<<<nblock, neuronPerBlock, 0, s0>>>(state,
											      d_preType,
											 	  rden, raxn, dden, daxn,
											 	  preTypeS, preTypeP, preTypeN,
											 	  init_pack, seed, networkSize, nType);
	getLastCudaError();
	//checkCudaErrors(cudaEventSynchronizeudaEventRecord(i1, s1));
	//checkCudaErrors(cudaEventSynchronize(i1));
    printf("initialzied\n");
    Size shared_mem;
    shared_mem = 2*warpSize*sizeof(Float);
    cal_blockPos<<<nblock, blockSize, shared_mem, s1>>>(d_pos, 
														d_block_x, d_block_y, 
														networkSize);
	getLastCudaError();
	checkCudaErrors(cudaEventRecord(i1, s1));
	checkCudaErrors(cudaEventSynchronize(i1));
    printf("block centers calculated\n");
	shared_mem = sizeof(Size);
    get_neighbor_blockId<<<nblock, blockSize, shared_mem, s0>>>(d_block_x, d_block_y, 
																d_neighborBlockId, d_nNeighborBlock, 
																blockROI, nPotentialNeighbor);
	getLastCudaError();
	checkCudaErrors(cudaEventRecord(i1, s1));
	checkCudaErrors(cudaEventSynchronize(i1));
    printf("neighbor blocks acquired\n");
	//checkCudaErrors(cudaEventRecord(i0, s0));
	//checkCudaErrors(cudaEventSynchronize(i0));
	//checkCudaErrors(cudaEventSynchronize(i1));
	//checkCudaErrors(cudaEventSynchronize(i2));
    shared_mem = blockSize*sizeof(Float) + blockSize*sizeof(Float) + blockSize*sizeof(Size);
    generate_connections<<<nblock, neuronPerBlock, shared_mem, s0>>>(d_pos,
																	 preTypeS, preTypeP, preTypeN,
																	 d_neighborBlockId, d_nNeighborBlock,
																	 rden, raxn,
																	 d_conMat, d_delayMat,
																	 d_conVec, d_delayVec,
																	 d_vecID, d_nVec,
																	 d_preTypeConnected, d_preTypeAvail, d_preTypeStrSum,
																	 d_preType,
																	 dden, daxn,
																	 state,
																	 networkSize, neighborSize, nPotentialNeighbor, speedOfThought, nType);
	getLastCudaError();
	checkCudaErrors(cudaEventRecord(i0, s0));
	checkCudaErrors(cudaEventSynchronize(i0));
    printf("connectivity constructed\n");
	checkCudaErrors(cudaMemcpy(block_x, d_block_x, outputSize, cudaMemcpyDeviceToHost)); // the whole chunk of output
	//checkCudaErrors(cudaMemcpy(preType, d_preType, 1, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaStreamDestroy(s0));
    checkCudaErrors(cudaStreamDestroy(s1));
    checkCudaErrors(cudaStreamDestroy(s2));
    // output to binary data files
    fV1_conMat.write((char*)conMat, nblock*blockSize*blockSize*sizeof(Float));
    fV1_conMat.close();
    fV1_delayMat.write((char*)delayMat, nblock*blockSize*blockSize*sizeof(Float));
    fV1_delayMat.close();
    
    fV1_vec.write((char*)nVec, networkSize*sizeof(Size));
    for (Size i=0; i<networkSize; i++) {
        fV1_vec.write((char*)&(vecID[i*neighborSize]), nVec[i]*sizeof(Size));
        fV1_vec.write((char*)&(conVec[i*neighborSize]), nVec[i]*sizeof(Float));
        fV1_vec.write((char*)&(delayVec[i*neighborSize]), nVec[i]*sizeof(Float));
    }
    fV1_vec.close();

    fBlock_pos.write((char*)block_x, nblock*sizeof(Float));
    fBlock_pos.write((char*)block_y, nblock*sizeof(Float));
    fBlock_pos.close();

    fNeighborBlock.write((char*)nNeighborBlock, nblock*sizeof(Size));
    for (Size i=0; i<nblock; i++) {
        fNeighborBlock.write((char*)&(neighborBlockId[i*nPotentialNeighbor]), nNeighborBlock[i]*sizeof(Size));
    }
    fNeighborBlock.close();

    fStats.write((char*)preTypeConnected, nType*networkSize*sizeof(Size));
    fStats.write((char*)preTypeAvail, nType*networkSize*sizeof(Size));
    fStats.write((char*)preTypeStrSum, nType*networkSize*sizeof(Float));
    fStats.close();

    checkCudaErrors(cudaFree(gpu_chunk));
	free(cpu_chunk);
    return 0;
}
