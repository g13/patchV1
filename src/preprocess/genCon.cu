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

	BigSize seed;
    Size maxNeighborBlock, maxDistantNeighbor;
	vector<Size> nTypeHierarchy;
    vector<Size> preTypeN;
    string V1_prop_filename, V1_pos_filename, theme;
	string V1_conMat_filename, V1_delayMat_filename;
	string V1_vec_filename, type_filename;
	string block_pos_filename, neighborBlock_filename, stats_filename;
    Float dscale, blockROI;
	bool gaussian_profile;
	Float dScale;
	Size usingPosDim;
    vector<Size> typeAccCount;
	vector<Float> rDend, rAxon;
	vector<Float> dDend, dAxon;
    vector<Float> sTypeMat, pTypeMat;

	po::options_description generic_opt("Generic options");
	generic_opt.add_options()
        ("seed", po::value<BigSize>(&seed)->default_value(7641807), "seed for RNG")
		("cfg_file,c", po::value<string>()->default_value("connectome.cfg"), "filename for configuration file")
		("help,h", "print usage");
	po::options_description input_opt("output options");
	input_opt.add_options()
        ("DisGauss", po::value<bool>(&gaussian_profile), "if set true, conn. prob. based on distance will follow a 2D gaussian with a variance. of (raxn*raxn + rden*rden)/2, otherwise will based on the overlap of the area specified by raxn and rden")
        ("rDend", po::value<vector<Float>>(&rDend),  "a vector of dendritic extensions' radius, size of nArchtype = nTypeHierarchy[0]")
        ("rAxon", po::value<vector<Float>>(&rAxon),  "a vector of axonic extensions' radius, size of nArchtype = nTypeHierarchy[0]")
        ("dScale",po::value<Float>(&dScale)->default_value(1.0),"a scaling ratio of all the neurites' lengths <radius>")
        ("typeAccCount",po::value<vector<Size>>(&typeAccCount), "neuronal types' discrete accumulative distribution, size of [nArchtype+1], nArchtype = nTypeHierarchy[0]")
        ("dDend", po::value<vector<Float>>(&dDend), "vector of dendrites' densities, size of nArchtype = nTypeHierarchy[0]")
        ("dAxon", po::value<vector<Float>>(&dAxon), "vector of axons' densities, size of nArchtype = nTypeHierarchy[0]")
		("nTypeHierarchy", po::value<vector<Size>>(&nTypeHierarchy), "a vector of hierarchical types, e.g., Exc and Inh at top level, sublevel Left Right, then the vector would be {2, 2}, resulting in a type ID sheet: 1, 2, 3, 4 being, Exc|Left, Exc|Right, Inh|Left, Inh|Right")
        ("sTypeMat", po::value<vector<Float>>(&sTypeMat), "connection strength matrix between neuronal types, size of [nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("pTypeMat", po::value<vector<Float>>(&pTypeMat), "connection prob. matrix between neuronal types, size of [nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("preTypeN", po::value<vector<Size>>(&preTypeN), "a vector of total number of presynaptic connection based on neuronal types size of nArchtype = nTypeHierarchy[0]")
        ("blockROI", po::value<Float>(&blockROI), "max radius (center to center) to include neighboring blocks in mm")
    	("usingPosDim", po::value<Size>(&usingPosDim)->default_value(2), "using <2>D coord. or <3>D coord. when calculating distance between neurons, influencing how the position data is read") 
        ("maxDistantNeighbor", po::value<Size>(&maxDistantNeighbor), "the preserved size of the array that store the presynaptic neurons' ID, who are not in the neighboring blocks")
        ("maxNeighborBlock", po::value<Size>(&maxNeighborBlock), "the preserved size of the array that store the neighboring blocks ID")
		("fType", po::value<string>(&type_filename)->default_value(""), "read nTypeHierarchy, pTypeMat, sTypeMat from this file, not implemented")
        ("fV1_prop", po::value<string>(&V1_prop_filename)->default_value("V1_prop.bin"), "the directory to read spatially predetermined functional neuronal types")
        ("fV1_pos", po::value<string>(&V1_pos_filename)->default_value("V1_pos.bin"), "the directory to read neuron positions");

	po::options_description output_opt("output options");
	output_opt.add_options()
        ("theme", po::value<string>(&theme)->default_value(""), "a name to be associated with the generated connection profile")
        ("fV1_conMat", po::value<string>(&V1_conMat_filename)->default_value("V1_conMat.bin"), "file that stores V1 to V1 connection within the neighboring blocks")
        ("fV1_delayMat", po::value<string>(&V1_delayMat_filename)->default_value("V1_delayMat.bin"), "file that stores V1 to V1 transmission delay within the neighboring blocks")
        ("fV1_vec", po::value<string>(&V1_vec_filename)->default_value("V1_vec.bin"), "file that stores V1 to V1 connection ID, strength and transmission delay outside the neighboring blocks")
		("fBlock_pos", po::value<string>(&block_pos_filename)->default_value("block_pos.bin"), "file that stores the center coord of each block")
		("fNeighborBlock", po::value<string>(&neighborBlock_filename)->default_value("neighborBlock.bin"), "file that stores the neighboring blocks' ID for each block")
		("fStats", po::value<string>(&stats_filename)->default_value("conStats.bin"), "file that stores the statistics of connections");

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

    ifstream fV1_pos, fType, fV1_prop;
    ofstream fV1_conMat, fV1_delayMat, fV1_vec;
    ofstream fBlock_pos, fNeighborBlock;
    ofstream fStats;

	if (gaussian_profile) {
		cout << "Using gaussian profile over distance\n";
	} else {
		cout << "probability over distance depends on dendrite and axon overlap\n";
	}
    cout << "maxDistantNeighbors = " << maxDistantNeighbor << " outside the blocks per neuron.\n";
    cout << "blockROI = " << blockROI << " mm.\n";
	Size nType;
	Size nHierarchy = nTypeHierarchy.size();
	if (type_filename.empty()) {
		if (nTypeHierarchy.size() < 1) {
			cout << "at least define one type of neuron with nTypeHierarchy.\n";
			return EXIT_FAILURE;
		} else {
			auto product = [](Size a, Size b) {
				return a*b;
			};
			nType = accumulate(nTypeHierarchy.begin(), nTypeHierarchy.end(), 1, product);
		}

    	if (pTypeMat.size() != nType*nType) {
			cout << "pTypeMat has size of " << pTypeMat.size() << ", should be " << nType*nType << "\n";
			return EXIT_FAILURE;
		}
    	if (sTypeMat.size() != nType*nType) {
			cout << "sTypeMat has size of " << sTypeMat.size() << ", should be " << nType*nType << "\n";
			return EXIT_FAILURE;
		}
	} else {
		fType.open(type_filename, ios::in|ios::binary);
		if (!fType) {
			cout << "failed to open neurnal type file:" << type_filename << "\n";
			return EXIT_FAILURE;
		}
	    // TODO: implement reading type conn. matrices from file
	}

	// TODO: std.
	Size nArchtype = nTypeHierarchy[0];
	if (preTypeN.size() != nArchtype) {
        cout << "size of preTypeN: " << preTypeN.size() << " should be consistent with the number of neuronal types at the hierarchical top: " << nArchtype << "\n"; 
        return EXIT_FAILURE;
	}
	{// dendrites and axons
    	if (dAxon.size() != nArchtype) {
    	    cout << "size of dAxon: " << dAxon.size() << " should be consistent with the number of neuronal types at the hierarchical top: " << nArchtype << "\n"; 
    	    return EXIT_FAILURE;
    	}
    	if (dDend.size() != nArchtype) {
    	    cout << "size of dDend: " << dDend.size() << " should be consistent with the number of neuronal types at the hierarchical top: " << nArchtype << "\n"; 
    	    return EXIT_FAILURE;
    	}

    	if (rAxon.size() != nArchtype) {
    	    cout << "size of rAxon: " << rAxon.size() << " should be consistent with the number of neuronal types at the hierarchical top: " << nArchtype << "\n"; 
    	    return EXIT_FAILURE;
    	} else {
    	    // adjust the scale
    	    for (Float &r: rAxon){
    	        r *= dScale;
    	    }
    	}
    	if (rDend.size() != nArchtype) {
    	    cout << "size of rDend: " << rDend.size() << " should be consistent with the number of neuronal types at the hierarchical top: " << nArchtype << "\n"; 
    	    return EXIT_FAILURE;
    	} else {
    	    // adjust the scale
    	    for (Float &r: rDend){
    	        r *= dScale;
    	    }
    	}
	}

    if (typeAccCount.size() != nArchtype + 1) {
        cout << "the accumulative distribution of neuronal type <typeAccCount> has size of " << typeAccCount.size() << ", should be " << nArchtype + 1 << ",  <nArchtype> + 1\n";
        return EXIT_FAILURE;
    }

    fV1_pos.open(V1_pos_filename, ios::in|ios::binary);
	if (!fV1_pos) {
		cout << "failed to open pos file:" << V1_pos_filename << "\n";
		return EXIT_FAILURE;
	}
    Size nblock, neuronPerBlock, dataDim;
    // read from file cudaMemcpy to device
	
    fV1_pos.read(reinterpret_cast<char*>(&nblock), sizeof(Size));
    fV1_pos.read(reinterpret_cast<char*>(&neuronPerBlock), sizeof(Size));
    Size networkSize = nblock*blockSize;
	cout << "networkSize = " << networkSize << "\n";
	fV1_pos.read(reinterpret_cast<char*>(&dataDim), sizeof(Size));
	// TODO: implement 3D, usingPosDim=3
	if (dataDim != usingPosDim) {
		cout << "the dimension of position coord intended is " << usingPosDim << ", data provided from " << V1_pos_filename << " gives " << dataDim << "\n";
		return EXIT_FAILURE;
	}
    vector<double> pos(usingPosDim*networkSize);
    fV1_pos.read(reinterpret_cast<char*>(&pos[0]), usingPosDim*networkSize*sizeof(double));
	fV1_pos.close();
	
	// read functional neuronal subtypes based on spatial location.
    fV1_prop.open(V1_prop_filename, ios::in|ios::binary);
	if (!fV1_prop) {
		cout << "failed to open pos file:" << V1_prop_filename << "\n";
		return EXIT_FAILURE;
	}
	Size nSubHierarchy;
    fV1_prop.read(reinterpret_cast<char*>(&nSubHierarchy), sizeof(Size));
	if (nSubHierarchy != nHierarchy-1) {
		cout << "inconsistent nSubHierarchy: " << nSubHierarchy << " should be " << nHierarchy - 1<< "\n";
		return EXIT_FAILURE;
	}
	for (Size i=0; i<nSubHierarchy; i++) {
		Size nSubType;
    	fV1_prop.read(reinterpret_cast<char*>(&nSubType), sizeof(Size));
		if (nSubType != nTypeHierarchy[i+1]) {
			cout << "inconsistent nSubType: " << nSubType << " at " << i+1 << "th level, should be " << nTypeHierarchy[i+1] << "\n";
			return EXIT_FAILURE;
		}
	}
	auto check_max = [](Int a, Int b) {
		return b > a? b: a;
	};
	vector<Int> preFixType(nSubHierarchy*networkSize);
    fV1_prop.read(reinterpret_cast<char*>(&preFixType[0]), sizeof(Int)*nSubHierarchy*networkSize);
	fV1_prop.close();

    hInitialize_package hInit_pack(nArchtype, nType, nHierarchy, nTypeHierarchy, typeAccCount, rAxon, rDend, dAxon, dDend, sTypeMat, pTypeMat, preTypeN);
	initialize_package init_pack(nArchtype, nType, nHierarchy, hInit_pack);
	hInit_pack.freeMem();
    Float speedOfThought = 1.0f; // mm/ms

    // TODO: types that shares a smaller portion than 1/neuronPerBlock
    if (typeAccCount.back() != neuronPerBlock) {
		cout << "type acc. dist. end with " << typeAccCount.back() << " should be " << neuronPerBlock << "\n";
        return EXIT_FAILURE;
    }

    if (!theme.empty()) {
        theme = theme + '-';
    }
    fV1_conMat.open(theme + V1_conMat_filename, ios::out | ios::binary);
	if (!fV1_conMat) {
		cout << "cannot open " << theme + V1_conMat_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fV1_delayMat.open(theme + V1_delayMat_filename, ios::out | ios::binary);
	if (!fV1_delayMat) {
		cout << "cannot open " << theme + V1_delayMat_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fV1_vec.open(theme + V1_vec_filename, ios::out | ios::binary);
	if (!fV1_vec) {
		cout << "cannot open " << theme + V1_vec_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fBlock_pos.open(theme + block_pos_filename, ios::out | ios::binary);
	if (!fBlock_pos) {
		cout << "cannot open " << theme + block_pos_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fNeighborBlock.open(theme + neighborBlock_filename, ios::out | ios::binary);
	if (!fNeighborBlock) {
		cout << "cannot open " << theme + neighborBlock_filename << " to write.\n";
		return EXIT_FAILURE;
	}
    fStats.open(theme + stats_filename, ios::out | ios::binary);
	if (!fStats) {
		cout << "cannot open " << theme + stats_filename << " to write.\n";
		return EXIT_FAILURE;
	}

	size_t memorySize = 2*nblock*sizeof(Float) + // block_x and y
        				networkSize*sizeof(Size) + // preType
        				2*blockSize*blockSize*nblock*sizeof(Float) + // con and delayMat
        				2*networkSize*maxDistantNeighbor*sizeof(Float) + // con and delayVec
        				networkSize*maxDistantNeighbor*sizeof(Size) + // vecID
        				networkSize*sizeof(Size) + // nVec
        				(maxNeighborBlock + 1)*nblock*sizeof(Size) + // neighborBlockId and nNeighborBlock
        				2*nType*networkSize*sizeof(Size) + // preTypeConnected and *Avail
        				nType*networkSize*sizeof(Float); // preTypeStrSum

	// to receive from device
    size_t outputSize = memorySize;
	printf("need to allocate %f MB memory on host\n", static_cast<float>(memorySize)/1024/1024);
	void *cpu_chunk = malloc(memorySize);
	assert(cpu_chunk);

    Float* block_x = (Float*) cpu_chunk;
    Float* block_y = block_x + nblock;
    Size* preType = (Size*) (block_y + nblock);
    Float* conMat = (Float*) (preType + networkSize); 
    Float* delayMat = conMat + blockSize*blockSize*nblock;
    Float* conVec = delayMat + blockSize*blockSize*nblock; 
    Float* delayVec = conVec + networkSize*maxDistantNeighbor;
    Size* vecID = (Size*) (delayVec + networkSize*maxDistantNeighbor);
    Size* nVec = vecID + networkSize*maxDistantNeighbor;
    Size* neighborBlockId = nVec + networkSize;
    Size* nNeighborBlock = neighborBlockId + maxNeighborBlock*nblock;
    Size* preTypeConnected = nNeighborBlock + nblock; 
    Size* preTypeAvail = preTypeConnected + nType*networkSize;
    Float* preTypeStrSum = (Float*) (preTypeAvail + nType*networkSize);

	assert(static_cast<void*>((char*)cpu_chunk + memorySize) == static_cast<void*>(preTypeStrSum + nType * networkSize));

    // ========== GPU mem ============
	size_t deviceOnlyMemSize = 2*networkSize*sizeof(Float) + // rden and raxn
     						   2*networkSize*sizeof(Float) + // dden and daxn
     						   nType*networkSize*sizeof(Float) + // preS_type
     						   nType*networkSize*sizeof(Float) + // preP_type
     						   networkSize*sizeof(Size) + // preN
     						   networkSize*sizeof(curandStateMRG32k3a); //state
    size_t d_memorySize = memorySize + nSubHierarchy*networkSize*sizeof(Int) + usingPosDim*networkSize*sizeof(double) + deviceOnlyMemSize;
    void* __restrict__ gpu_chunk;
	printf("need to allocate %f MB memory on device\n", static_cast<float>(d_memorySize) / 1024 / 1024);
    checkCudaErrors(cudaMalloc((void**)&gpu_chunk, d_memorySize));

    // init by kernel, reside on device only
	// copy and init by the whole chunk from host to device
    Float * __restrict__ rden = (Float*) gpu_chunk; 
    Float * __restrict__ raxn = rden + networkSize;
	Float * __restrict__ dden = raxn + networkSize;
	Float * __restrict__ daxn = dden + networkSize;
    Float * __restrict__ preS_type = daxn + networkSize;
    Float * __restrict__ preP_type = preS_type + nType*networkSize;
    Size * __restrict__ preN = (Size*) (preP_type + nType*networkSize);
    curandStateMRG32k3a* __restrict__ state = (curandStateMRG32k3a*) (preN + networkSize);
	// copy from host to device indivdual chunk
    Int* __restrict__ d_preFixType = (Int*) (state + networkSize);
    double* __restrict__ d_pos = (double*) (d_preFixType + nSubHierarchy*networkSize);
	// copy by the whole chunk
    // device to host
    Float* __restrict__ d_block_x = (Float*) (d_pos + usingPosDim*networkSize); 
    Float* __restrict__ d_block_y = d_block_x + nblock;
    Size*  __restrict__ d_preType = (Size*) (d_block_y + nblock);
    Float* __restrict__ d_conMat = (Float*) (d_preType + networkSize); 
    Float* __restrict__ d_delayMat = d_conMat + blockSize*blockSize*nblock;
    Float* __restrict__ d_conVec = d_delayMat + blockSize*blockSize*nblock; 
    Float* __restrict__ d_delayVec = d_conVec + networkSize*maxDistantNeighbor;
    Size*  __restrict__ d_vecID = (Size*) (d_delayVec + networkSize*maxDistantNeighbor);
    Size*  __restrict__ d_nVec = d_vecID + networkSize*maxDistantNeighbor;
    Size*  __restrict__ d_neighborBlockId = d_nVec + networkSize;
    Size*  __restrict__ d_nNeighborBlock = d_neighborBlockId + maxNeighborBlock*nblock;
    Size*  __restrict__ d_preTypeConnected = d_nNeighborBlock + nblock;
    Size*  __restrict__ d_preTypeAvail = d_preTypeConnected + nType*networkSize;
    Float* __restrict__ d_preTypeStrSum = (Float*) (d_preTypeAvail + nType*networkSize);

	// check memory address consistency
	assert(static_cast<void*>((char*)gpu_chunk + d_memorySize) == static_cast<void*>(d_preTypeStrSum + nType * networkSize));

    // for array usage on the device in function "generate_connections"
    Size localHeapSize = sizeof(Float)*networkSize*maxNeighborBlock*blockSize;
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
    checkCudaErrors(cudaMemcpy(d_preFixType, &preFixType[0], nSubHierarchy*networkSize*sizeof(Int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pos, &pos[0], usingPosDim*networkSize*sizeof(double), cudaMemcpyHostToDevice));
    initialize<<<nblock, neuronPerBlock, 0, s0>>>(state,
											      d_preType,
											 	  rden, raxn, dden, daxn,
											 	  preS_type, preP_type, preN, d_preFixType,
											 	  init_pack, seed, networkSize, nType, nArchtype, nSubHierarchy);
	getLastCudaError("initialize failed");
	//checkCudaErrors(cudaEventSynchronizeudaEventRecord(i1, s1));
	//checkCudaErrors(cudaEventSynchronize(i1));
    init_pack.freeMem();
    printf("initialzied\n");
    Size shared_mem;
    shared_mem = 2*warpSize*sizeof(Float);
    cal_blockPos<<<nblock, blockSize, shared_mem, s1>>>(d_pos, 
														d_block_x, d_block_y, 
														networkSize);
	getLastCudaError("cal_blockPos failed");
	checkCudaErrors(cudaEventRecord(i1, s1));
	checkCudaErrors(cudaEventSynchronize(i1));
    printf("block centers calculated\n");
	shared_mem = sizeof(Size);
    get_neighbor_blockId<<<nblock, blockSize, shared_mem, s0>>>(d_block_x, d_block_y, 
																d_neighborBlockId, d_nNeighborBlock, 
																blockROI, maxNeighborBlock);
	getLastCudaError("get_neighbor_blockId failed");
	checkCudaErrors(cudaEventRecord(i1, s1));
	checkCudaErrors(cudaEventSynchronize(i1));
    printf("neighbor blocks acquired\n");
	//checkCudaErrors(cudaEventRecord(i0, s0));
	//checkCudaErrors(cudaEventSynchronize(i0));
	//checkCudaErrors(cudaEventSynchronize(i1));
	//checkCudaErrors(cudaEventSynchronize(i2));
    shared_mem = blockSize*sizeof(Float) + blockSize*sizeof(Float) + blockSize*sizeof(Size);
    generate_connections<<<nblock, neuronPerBlock, shared_mem, s0>>>(d_pos,
																	 preS_type, preP_type, preN,
																	 d_neighborBlockId, d_nNeighborBlock,
																	 rden, raxn,
																	 d_conMat, d_delayMat,
																	 d_conVec, d_delayVec,
																	 d_vecID, d_nVec,
																	 d_preTypeConnected, d_preTypeAvail, d_preTypeStrSum,
																	 d_preType,
																	 dden, daxn,
																	 state,
																	 networkSize, maxDistantNeighbor, maxNeighborBlock, speedOfThought, nType, gaussian_profile);
	getLastCudaError("generate_connections failed");
	checkCudaErrors(cudaEventRecord(i0, s0));
	checkCudaErrors(cudaEventSynchronize(i0));
    printf("connectivity constructed\n");
	// the whole chunk of output
	checkCudaErrors(cudaMemcpy(block_x, d_block_x, outputSize, cudaMemcpyDeviceToHost)); 	
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
        fV1_vec.write((char*)&(vecID[i*maxDistantNeighbor]), nVec[i]*sizeof(Size));
        fV1_vec.write((char*)&(conVec[i*maxDistantNeighbor]), nVec[i]*sizeof(Float));
        fV1_vec.write((char*)&(delayVec[i*maxDistantNeighbor]), nVec[i]*sizeof(Float));
    }
    fV1_vec.close();

    fBlock_pos.write((char*)block_x, nblock*sizeof(Float));
    fBlock_pos.write((char*)block_y, nblock*sizeof(Float));
    fBlock_pos.close();

    fNeighborBlock.write((char*)nNeighborBlock, nblock*sizeof(Size));
    for (Size i=0; i<nblock; i++) {
        fNeighborBlock.write((char*)&(neighborBlockId[i*maxNeighborBlock]), nNeighborBlock[i]*sizeof(Size));
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
