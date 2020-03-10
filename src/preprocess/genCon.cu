#include "genCon.h"
// TODO: check preType, V1_type.bin

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
    Size maxNeighborBlock, nearNeighborBlock;
	Size maxDistantNeighbor;
	vector<Size> nTypeHierarchy;
    vector<Size> preTypeN;
    string V1_type_filename, V1_feature_filename, V1_pos_filename, theme;
	string V1_conMat_filename, V1_delayMat_filename;
	string V1_vec_filename, typeMat_filename;
	string block_pos_filename, neighborBlock_filename, stats_filename;
    Float dScale, blockROI;
	bool gaussian_profile;
	Size usingPosDim;
    vector<Size> archtypeAccCount;
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
        ("archtypeAccCount",po::value<vector<Size>>(&archtypeAccCount), "neuronal types' discrete accumulative distribution, size of [nArchtype], nArchtype = nTypeHierarchy[0]")
        ("dDend", po::value<vector<Float>>(&dDend), "vector of dendrites' densities, size of nArchtype = nTypeHierarchy[0]")
        ("dAxon", po::value<vector<Float>>(&dAxon), "vector of axons' densities, size of nArchtype = nTypeHierarchy[0]")
		("nTypeHierarchy", po::value<vector<Size>>(&nTypeHierarchy), "a vector of hierarchical types, e.g., Exc and Inh at top level, sublevel Left Right, then the vector would be {2, 2}, resulting in a type ID sheet: 1, 2, 3, 4 being, Exc|Left, Exc|Right, Inh|Left, Inh|Right")
        ("sTypeMat", po::value<vector<Float>>(&sTypeMat), "connection strength matrix between neuronal types, size of [nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("pTypeMat", po::value<vector<Float>>(&pTypeMat), "connection prob. matrix between neuronal types, size of [nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("preTypeN", po::value<vector<Size>>(&preTypeN), "a vector of total number of presynaptic connection based on the neurons' archtypes, size of nArchtype = nTypeHierarchy[0]")
        ("blockROI", po::value<Float>(&blockROI), "max radius (center to center) to include neighboring blocks in mm")
    	("usingPosDim", po::value<Size>(&usingPosDim)->default_value(2), "using <2>D coord. or <3>D coord. when calculating distance between neurons, influencing how the position data is read") 
        ("maxDistantNeighbor", po::value<Size>(&maxDistantNeighbor), "the preserved size of the array that store the presynaptic neurons' ID, who are not in the neighboring blocks")
        ("maxNeighborBlock", po::value<Size>(&maxNeighborBlock)->default_value(12), "the preserved size (minus the nearNeighborBlock) of the array that store the neighboring blocks ID that goes into conVec")
        ("nearNeighborBlock", po::value<Size>(&nearNeighborBlock)->default_value(8), "the preserved size of the array that store the neighboring blocks ID that goes into conMat, excluding the self block")
		("fV1_typeMat", po::value<string>(&typeMat_filename)->default_value(""), "read nTypeHierarchy, pTypeMat, sTypeMat from this file, not implemented")
        ("fV1_type", po::value<string>(&V1_type_filename)->default_value("V1_type.bin"), "file to read predetermined neuronal types based on nTypeHierarchy")
        ("fV1_feature", po::value<string>(&V1_feature_filename)->default_value("V1_feature.bin"), "file to read spatially predetermined functional features of neurons")
        ("fV1_pos", po::value<string>(&V1_pos_filename)->default_value("V1_allpos.bin"), "the directory to read neuron positions");

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

    ifstream fV1_pos, fV1_typeMat, fV1_type, fV1_feature;
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
	if (typeMat_filename.empty()) {
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
		fV1_typeMat.open(typeMat_filename, ios::in|ios::binary);
		if (!fV1_typeMat) {
			cout << "failed to open neurnal type file:" << typeMat_filename << "\n";
			return EXIT_FAILURE;
		} else {
			cout << "not implemented\n";
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

    if (archtypeAccCount.size() != nArchtype) {
        cout << "the accumulative distribution of neuronal type <archtypeAccCount> has size of " << archtypeAccCount.size() << ", should be " << nArchtype << ",  <nArchtype>\n";
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
    if (neuronPerBlock > blockSize) {
        cout << "neuron per block (" << neuronPerBlock << ") cannot be larger than cuda block size: " << blockSize << "\n";
    }
    Size networkSize = nblock*neuronPerBlock;
	cout << "networkSize = " << networkSize << "\n";
	fV1_pos.read(reinterpret_cast<char*>(&dataDim), sizeof(Size));
	// TODO: implement 3D, usingPosDim=3
	if (dataDim != usingPosDim) {
		cout << "the dimension of position coord intended is " << usingPosDim << ", data provided from " << V1_pos_filename << " gives " << dataDim << "\n";
		return EXIT_FAILURE;
	}
	
	{// not used
		double tmp;	
		fV1_pos.read(reinterpret_cast<char*>(&tmp), sizeof(double));
		fV1_pos.read(reinterpret_cast<char*>(&tmp), sizeof(double));
		fV1_pos.read(reinterpret_cast<char*>(&tmp), sizeof(double));
		fV1_pos.read(reinterpret_cast<char*>(&tmp), sizeof(double));
	}
	
    vector<double> pos(usingPosDim*networkSize);
    fV1_pos.read(reinterpret_cast<char*>(&pos[0]), usingPosDim*networkSize*sizeof(double));
	fV1_pos.close();
	
	// read predetermined neuronal subtypes.
	Size nSubHierarchy;
	vector<Size> preFixType;
	if (nHierarchy - 1 > 0) {
		fV1_type.open(V1_type_filename, ios::in | ios::binary);
		if (!fV1_type) {
			cout << "failed to open V1type file:" << V1_type_filename << ", note if nHierarchy: " << nHierarchy << " > 1.\n";
			return EXIT_FAILURE;
		}
		fV1_type.read(reinterpret_cast<char*>(&nSubHierarchy), sizeof(Size));
		if (nSubHierarchy != nHierarchy - 1) {
			cout << "inconsistent nSubHierarchy: " << nSubHierarchy << " should be " << nHierarchy - 1 << "\n";
			return EXIT_FAILURE;
		}
		for (Size i = 0; i < nSubHierarchy; i++) {
			Size nSubType;
			fV1_type.read(reinterpret_cast<char*>(&nSubType), sizeof(Size));
			if (nSubType != nTypeHierarchy[i + 1]) {
				cout << "inconsistent nSubType: " << nSubType << " at " << i + 1 << "th level, should be " << nTypeHierarchy[i + 1] << "\n";
				return EXIT_FAILURE;
			}
		}
        // TODO: check if vector::reserve can work.
		preFixType.assign(nSubHierarchy*networkSize,0);
		fV1_type.read(reinterpret_cast<char*>(&preFixType[0]), sizeof(Size)*nSubHierarchy*networkSize);
		fV1_type.close();
	} else {
        nSubHierarchy = 0;
		cout << "no subtypes\n";
    }
	// read predetermined functional response features of neurons (use as starting seed if involve learning).
    fV1_feature.open(V1_feature_filename, ios::in|ios::binary);
	if (!fV1_feature) {
		cout << "failed to open feature file:" << V1_feature_filename << "\n";
		return EXIT_FAILURE;
	}
	Size nFeature;
    fV1_feature.read(reinterpret_cast<char*>(&nFeature), sizeof(Size));
	vector<Float> featureValue(nFeature*networkSize); // [OD, OP, ..]
    fV1_feature.read(reinterpret_cast<char*>(&featureValue[0]), sizeof(Float)*nFeature*networkSize);
	fV1_feature.close();
    initializePreferenceFunctions(nFeature);

    hInitialize_package hInit_pack(nArchtype, nType, nHierarchy, nTypeHierarchy, archtypeAccCount, rAxon, rDend, dAxon, dDend, sTypeMat, pTypeMat, preTypeN);
	initialize_package init_pack(nArchtype, nType, nHierarchy, hInit_pack);
	hInit_pack.freeMem();
    Float speedOfThought = 1.0f; // mm/ms

    // TODO: types that shares a smaller portion than 1/neuronPerBlock
    if (archtypeAccCount.back() != neuronPerBlock) {
		cout << "type acc. dist. end with " << archtypeAccCount.back() << " should be " << neuronPerBlock << "\n";
        return EXIT_FAILURE;
    }

    if (!theme.empty()) {
        theme = theme + '-';
    }
    nearNeighborBlock += 1; // including self
    fV1_conMat.open(theme + V1_conMat_filename, ios::out | ios::binary);
	if (!fV1_conMat) {
		cout << "cannot open " << theme + V1_conMat_filename << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fV1_conMat.write((char*) &nearNeighborBlock, sizeof(Size));
    }
    fV1_delayMat.open(theme + V1_delayMat_filename, ios::out | ios::binary);
	if (!fV1_delayMat) {
		cout << "cannot open " << theme + V1_delayMat_filename << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fV1_delayMat.write((char*) &nearNeighborBlock, sizeof(Size));
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

    maxNeighborBlock += 1; // including self
    // check memory availability
    size_t memorySize, d_memorySize, matSize;

    size_t neighborSize = 2*nblock*sizeof(Float) + // block_x and y
        			      (maxNeighborBlock + 1)*nblock*sizeof(Size); // neighborBlockId and nNeighborBlock

    size_t statSize = 2*nType*networkSize*sizeof(Size) + // preTypeConnected and *Avail
        	          nType*networkSize*sizeof(Float); // preTypeStrSum
    size_t vecSize = 2*maxDistantNeighbor*networkSize*sizeof(Float) + // con and delayVec
        		     maxDistantNeighbor*networkSize*sizeof(Size) + // vecID
        		     networkSize*sizeof(Size); // nVec

	size_t deviceOnlyMemSize = 2*networkSize*sizeof(Float) + // rden and raxn
         					   2*networkSize*sizeof(Float) + // dden and daxn
         					   nType*networkSize*sizeof(Float) + // preS_type
         					   nType*networkSize*sizeof(Float) + // preP_type
         					   networkSize*sizeof(Size) + // preN
                               networkSize*sizeof(Size) + // preType
         					   networkSize*sizeof(curandStateMRG32k3a); //state
	void *cpu_chunk;
    Int half = 1;
    Size maxChunkSize = nblock;
	do { 
        if (half > 1) {
            Size half0 = maxChunkSize/half;
            Size half1 = maxChunkSize - half0;
            maxChunkSize = (half0 > half1)? half0: half1;
        }
        matSize = 2*nearNeighborBlock*neuronPerBlock*neuronPerBlock*maxChunkSize*sizeof(Float); // con and delayMat

        memorySize = matSize + vecSize + statSize + neighborSize;

        d_memorySize = memorySize + deviceOnlyMemSize +
                       nSubHierarchy*networkSize*sizeof(Size) + 
                       nFeature*networkSize*sizeof(Float) + 
                       usingPosDim*networkSize*sizeof(double); 

        half *= 2;
	    cpu_chunk = malloc(memorySize);
    } while ((cpu_chunk == NULL || d_memorySize > deviceProps.totalGlobalMem*0.8) && nblock > 1);
    Size nChunk = (nblock + maxChunkSize-1) /maxChunkSize - 1;
    Size remainChunkSize = nblock%maxChunkSize;
	if (remainChunkSize == 0) {
		remainChunkSize = maxChunkSize;
	}
	printf("need to allocate %f MB memory on host\n", static_cast<float>(memorySize)/1024/1024);
	// to receive from device
    void* __restrict__ gpu_chunk;
	printf("need to allocate %f MB memory on device\n", static_cast<float>(d_memorySize) / 1024 / 1024);
    checkCudaErrors(cudaMalloc((void**)&gpu_chunk, d_memorySize));
    if (nChunk > 0) {
        cout << "due to memory limit, connection matrix, vectors and their statistics will come in "<< nChunk << " chunks of " << maxChunkSize << " blocks and a single chunk of " << remainChunkSize << " blocks.\n";
    } else {
        cout << "the connectome will be  generated in whole\n";
    }

    // ============ CPU MEM ============
    // blocks
    Float* block_x = (Float*) cpu_chunk;
    Float* block_y = block_x + nblock;
    PosInt* neighborBlockId = (Size*) (block_y + nblock);
    Size* nNeighborBlock = neighborBlockId + maxNeighborBlock*nblock;

    // connectome
    Float* conMat = (Float*) (nNeighborBlock + nblock);
    Float* delayMat = conMat + nearNeighborBlock*neuronPerBlock*neuronPerBlock*maxChunkSize;
    Float* conVec = delayMat + nearNeighborBlock*neuronPerBlock*neuronPerBlock*maxChunkSize; 
    Float* delayVec = conVec + maxDistantNeighbor*networkSize;
    Size* vecID = (Size*) (delayVec + maxDistantNeighbor*networkSize);
    Size* nVec = vecID + maxDistantNeighbor*networkSize;

    // stats
    Size* preTypeConnected = nVec + networkSize; 
    Size* preTypeAvail = preTypeConnected + nType*networkSize;
    Float* preTypeStrSum = (Float*) (preTypeAvail + nType*networkSize);

	assert(static_cast<void*>((char*)cpu_chunk + memorySize) == static_cast<void*>(preTypeStrSum + nType * networkSize));

    // ========== GPU mem ============
    // init by kernel, reside on device only
    Float* __restrict__ rden = (Float*) gpu_chunk; 
    Float* __restrict__ raxn = rden + networkSize;
	Float* __restrict__ dden = raxn + networkSize;
	Float* __restrict__ daxn = dden + networkSize;
    Float* __restrict__ preS_type = daxn + networkSize;
    Float* __restrict__ preP_type = preS_type + nType*networkSize;
    Size*  __restrict__ preN = (Size*) (preP_type + nType*networkSize);
    Size*  __restrict__ d_preType = preN + networkSize;
    curandStateMRG32k3a* __restrict__ state = (curandStateMRG32k3a*) (d_preType + networkSize);

	// copy from host to device indivdual chunk
    Size* __restrict__ d_preFixType = (Size*) (state + networkSize);
    Float* __restrict__ d_feature = (Float*) (d_preFixType + nSubHierarchy*networkSize);
    double* __restrict__ d_pos = (double*) (d_feature + nFeature*networkSize);
	// copy by the whole chunk
    // device to host
    Float* __restrict__ d_block_x = (Float*) (d_pos + usingPosDim*networkSize); 
    Float* __restrict__ d_block_y = d_block_x + nblock;
    PosInt*  __restrict__ d_neighborBlockId = (Size*) (d_block_y + nblock);
    Size*  __restrict__ d_nNeighborBlock = d_neighborBlockId + maxNeighborBlock*nblock;

    Float* __restrict__ d_conMat = (Float*) (d_nNeighborBlock + nblock);
    Float* __restrict__ d_delayMat = d_conMat + nearNeighborBlock*neuronPerBlock*neuronPerBlock*maxChunkSize;
    Float* __restrict__ d_conVec = d_delayMat + nearNeighborBlock*neuronPerBlock*neuronPerBlock*maxChunkSize; 
    Float* __restrict__ d_delayVec = d_conVec + networkSize*maxDistantNeighbor;
    Size*  __restrict__ d_vecID = (Size*) (d_delayVec + networkSize*maxDistantNeighbor);
    Size*  __restrict__ d_nVec = d_vecID + networkSize*maxDistantNeighbor;

    // stats
    Size*  __restrict__ d_preTypeConnected = d_nVec + networkSize;
    Size*  __restrict__ d_preTypeAvail = d_preTypeConnected + nType*networkSize;
    Float* __restrict__ d_preTypeStrSum = (Float*) (d_preTypeAvail + nType*networkSize);

	// check memory address consistency
	assert(static_cast<void*>((char*)gpu_chunk + d_memorySize) == static_cast<void*>(d_preTypeStrSum + nType * networkSize));

    // for array usage on the device in function "generate_connections"
    Size localHeapSize = (sizeof(Float)*maxNeighborBlock*neuronPerBlock + sizeof(Size)*nType*3)*neuronPerBlock*deviceProps.multiProcessorCount;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, localHeapSize);
    printf("heap size preserved %f Mb\n", localHeapSize*1.5/1024/1024);

    //cudaStream_t s0, s1, s2;
    //cudaEvent_t i0, i1, i2;
    //cudaEventCreate(&i0);
    //cudaEventCreate(&i1);
    //cudaEventCreate(&i2);
    //checkCudaErrors(cudaStreamCreate(&s0));
    //checkCudaErrors(cudaStreamCreate(&s1));
    //checkCudaErrors(cudaStreamCreate(&s2));
    if (nSubHierarchy > 0) {
        checkCudaErrors(cudaMemcpy(d_preFixType, &preFixType[0], nSubHierarchy*networkSize*sizeof(Size), cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMemcpy(d_feature, &featureValue[0], nFeature*networkSize*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pos, &pos[0], usingPosDim*networkSize*sizeof(double), cudaMemcpyHostToDevice));
    initialize<<<nblock, neuronPerBlock>>>(
        state,
	    d_preType,
		rden, raxn, dden, daxn,
		preS_type, preP_type, preN, d_preFixType,
		init_pack, seed, networkSize, nType, nArchtype, nSubHierarchy);
	getLastCudaError("initialize failed");
	checkCudaErrors(cudaDeviceSynchronize());
    init_pack.freeMem();
    printf("initialzied\n");
    //Size shared_mem;
    cal_blockPos<<<nblock, neuronPerBlock>>>(
        d_pos, 
		d_block_x, d_block_y, 
		networkSize);
	getLastCudaError("cal_blockPos failed");
    printf("block centers calculated\n");
	//shared_mem = sizeof(Size);
    get_neighbor_blockId<<<nblock, neuronPerBlock, maxNeighborBlock*(sizeof(PosInt)+sizeof(Float))>>>(
        d_block_x, d_block_y, 
		d_neighborBlockId, d_nNeighborBlock, 
		nblock, blockROI, maxNeighborBlock);
	getLastCudaError("get_neighbor_blockId failed");
    printf("neighbor blocks acquired\n");

	checkCudaErrors(cudaMemcpy(block_x, d_block_x, neighborSize, cudaMemcpyDeviceToHost)); 	
    fBlock_pos.write((char*)block_x, nblock*sizeof(Float));
    fBlock_pos.write((char*)block_y, nblock*sizeof(Float));
    fBlock_pos.close();

    fNeighborBlock.write((char*)nNeighborBlock, nblock*sizeof(Size));
    //cout << "number of neighbors: ";
    for (Size i=0; i<nblock; i++) {
        //cout << nNeighborBlock[i] <<  ", ";
        fNeighborBlock.write((char*)&neighborBlockId[i*maxNeighborBlock], nNeighborBlock[i]*sizeof(PosInt));
    }
    //cout << "\n";
    fNeighborBlock.close();

    //shared_mem = neuronPerBlock*sizeof(Float) + neuronPerBlock*sizeof(Float) + neuronPerBlock*sizeof(Size);
    Size current_nblock;
    PosInt offset = 0; // memory offset
    for (PosInt iChunk = 0; iChunk < nChunk+1; iChunk++) {
        if (iChunk < nChunk) current_nblock = maxChunkSize;
        else current_nblock = remainChunkSize;
		cout << "generate_connections<<<" << current_nblock << ", " << neuronPerBlock << ">>>\n";
        generate_connections<<<current_nblock, neuronPerBlock>>>(
            d_pos,
	    	preS_type, preP_type, preN,
	    	d_neighborBlockId, d_nNeighborBlock,
	    	rden, raxn,
	    	d_conMat, d_delayMat,
	    	d_conVec, d_delayVec,
	    	d_vecID, d_nVec,
	    	d_preTypeConnected, d_preTypeAvail, d_preTypeStrSum,
	    	d_preType, d_feature,
	    	dden, daxn,
	    	state,
	    	offset, networkSize, maxDistantNeighbor, nearNeighborBlock, maxNeighborBlock, speedOfThought, nType, nFeature, gaussian_profile);
	    getLastCudaError("generate_connections failed");
        offset += current_nblock*neuronPerBlock;

        size_t current_matSize = current_nblock*nearNeighborBlock*neuronPerBlock*neuronPerBlock*sizeof(Float);
	    checkCudaErrors(cudaMemcpy(conMat, d_conMat, 2*current_matSize, cudaMemcpyDeviceToHost)); 	
        // output connectome data
        fV1_conMat.write((char*)conMat, current_matSize);
        fV1_delayMat.write((char*)delayMat, current_matSize);
    }
    fV1_conMat.close();
    fV1_delayMat.close();
    printf("connectivity constructed\n");
	//checkCudaErrors(cudaStreamDestroy(s0));
    //checkCudaErrors(cudaStreamDestroy(s1));
    //checkCudaErrors(cudaStreamDestroy(s2));
	checkCudaErrors(cudaMemcpy(conVec, d_conVec, vecSize+statSize, cudaMemcpyDeviceToHost)); 	
    
    fV1_vec.write((char*)nVec, networkSize*sizeof(Size));
    for (Size i=0; i<networkSize; i++) {
        fV1_vec.write((char*)&(vecID[i*maxDistantNeighbor]), nVec[i]*sizeof(Size));
        fV1_vec.write((char*)&(conVec[i*maxDistantNeighbor]), nVec[i]*sizeof(Float));
        fV1_vec.write((char*)&(delayVec[i*maxDistantNeighbor]), nVec[i]*sizeof(Float));
    }
    fV1_vec.close();

    fStats.write((char*)&nType,sizeof(Size));
    fStats.write((char*)&networkSize,sizeof(Size));
    fStats.write((char*)preTypeConnected, nType*networkSize*sizeof(Size));
    fStats.write((char*)preTypeAvail, nType*networkSize*sizeof(Size));
    fStats.write((char*)preTypeStrSum, nType*networkSize*sizeof(Float));
    fStats.close();

	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(gpu_chunk));
	free(cpu_chunk);
    return 0;
}
