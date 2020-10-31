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
    string V1_type_filename, V1_feature_filename, V1_pos_filename, LGN_V1_s_filename, suffix, conLGN_suffix, LGN_V1_cfg_filename, output_cfg_filename;
	string V1_conMat_filename, V1_delayMat_filename;
	string V1_vec_filename;
	string block_pos_filename, neighborBlock_filename, stats_filename;
    Float dScale, blockROI;
	vector<Float> extExcRatio;
    //vector<Float> targetFR;
    ///Float FF_FB_ratio;
    Float min_FB_ratio;
    //Float LGN_targetFR;
	bool gaussian_profile;
    bool strictStrength;
    bool CmoreN;
	Size usingPosDim;
	Float longRangeROI;
	Float longRangeCorr;
	vector<Float> rDend, rAxon;
	vector<Float> dDend, dAxon;
    vector<Float> sTypeMat;
    vector<Float> typeFeatureMat;
    vector<Size> nTypeMat;
    vector<Float> synPerConFF;
    vector<Float> synPerCon;

	po::options_description generic_opt("Generic options");
	generic_opt.add_options()
        ("seed", po::value<BigSize>(&seed)->default_value(7641807), "seed for RNG")
		("cfg_file,c", po::value<string>()->default_value("connectome.cfg"), "filename for configuration file")
		("help,h", "print usage");

	Float minConTol;
	po::options_description input_opt("output options");
	input_opt.add_options()
        ("DisGauss", po::value<bool>(&gaussian_profile), "if set true, conn. prob. based on distance will follow a 2D gaussian with a variance. of (raxn*raxn + rden*rden)/2, otherwise will based on the overlap of the area specified by raxn and rden")
        ("strictStrength", po::value<bool>(&strictStrength), "strictly match preset summed connection")
        ("CmoreN", po::value<bool>(&CmoreN), "if true complex gets more connections otherwise stronger strength")
        ("rDend", po::value<vector<Float>>(&rDend),  "a vector of dendritic extensions' radius, size of nType ")
        ("rAxon", po::value<vector<Float>>(&rAxon),  "a vector of axonic extensions' radius, size of nType")
        ("dScale",po::value<Float>(&dScale)->default_value(1.0),"a scaling ratio of all the neurites' lengths <radius>")
        ("longRangeROI", po::value<Float>(&longRangeROI), "ROI of long-range cortical input")
        ("longRangeCorr", po::value<Float>(&longRangeCorr), "correlation between long-range cortical inputs that cortical cells receives")
        ("dDend", po::value<vector<Float>>(&dDend), "vector of dendrites' densities, size of nType")
        ("dAxon", po::value<vector<Float>>(&dAxon), "vector of axons' densities, size of nType")
		("nTypeHierarchy", po::value<vector<Size>>(&nTypeHierarchy), "a vector of hierarchical types differs in non-functional properties: reversal potentials, characteristic lengths of dendrite and axons, e.g. in size of nArchtype, {Exc-Pyramidal, Exc-stellate; Inh-PV, Inh-SOM, Inh-LTS} then the vector would be {3, 2}, with Exc and Inh being arch type")
		//("LGN_targetFR", po::value<Float>(&LGN_targetFR), "target firing rate of a LGN cell")
		//("targetFR", po::value<vector<Float>>(&targetFR), "a vector of target firing rate of different neuronal types")
		//("FF_FB_ratio", po::value<Float>(&FF_FB_ratio), "excitation ratio of FF over total excitation")
		("extExcRatio", po::value<vector<Float>>(&extExcRatio), "minimum cortical excitation ratio to E and I with predefined mean value")
		("min_FB_ratio", po::value<Float>(&min_FB_ratio), "minimum cortical excitation ratio of predefined mean value")
		("minConTol", po::value<Float>(&minConTol), "minimum cortical connection ratio of the predefined value")
        ("sTypeMat", po::value<vector<Float>>(&sTypeMat), "connection strength matrix between neuronal types, size of [nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("nTypeMat", po::value<vector<Size>>(&nTypeMat), "#connection matrix between neuronal types, size of [nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("synPerConFF",  po::value<vector<Float>>(&synPerConFF), "synpases per feedforward connection, size of nType")
        ("synPerCon",  po::value<vector<Float>>(&synPerCon), "synpases per cortical connection size of [nType, nType]")
        ("typeFeatureMat", po::value<vector<Float>>(&typeFeatureMat), "feature parameter of neuronal types, size of [nFeature, nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("blockROI", po::value<Float>(&blockROI), "max radius (center to center) to include neighboring blocks in mm")
    	("usingPosDim", po::value<Size>(&usingPosDim)->default_value(2), "using <2>D coord. or <3>D coord. when calculating distance between neurons, influencing how the position data is read") 
        ("maxDistantNeighbor", po::value<Size>(&maxDistantNeighbor), "the preserved size of the array that store the presynaptic neurons' ID, who are not in the neighboring blocks")
        ("maxNeighborBlock", po::value<Size>(&maxNeighborBlock)->default_value(12), "the preserved size (minus the nearNeighborBlock) of the array that store the neighboring blocks ID that goes into conVec")
        ("nearNeighborBlock", po::value<Size>(&nearNeighborBlock)->default_value(8), "the preserved size of the array that store the neighboring blocks ID that goes into conMat, excluding the self block, self will be added later")
        ("fV1_feature", po::value<string>(&V1_feature_filename)->default_value("V1_feature.bin"), "file to read spatially predetermined functional features of neurons")
        ("fV1_pos", po::value<string>(&V1_pos_filename)->default_value("V1_allpos.bin"), "the directory to read neuron positions")
        ("conLGN_suffix", po::value<string>(&conLGN_suffix)->default_value(""), "suffix associated with fLGN_V1_s")
		("fLGN_V1_cfg", po::value<string>(&LGN_V1_cfg_filename)->default_value("LGN_V1_cfg"),"file stores LGN_V1.cfg parameters")
		("fLGN_V1_s", po::value<string>(&LGN_V1_s_filename)->default_value("LGN_V1_sList"),"file stores LGN to V1 connection strengths, use conLGN_suffix");

	po::options_description output_opt("output options");
	output_opt.add_options()
        ("suffix", po::value<string>(&suffix)->default_value(""), "a suffix to be associated with the generated connection profile")
        ("fV1_conMat", po::value<string>(&V1_conMat_filename)->default_value("V1_conMat"), "file that stores V1 to V1 connection within the neighboring blocks")
        ("fV1_delayMat", po::value<string>(&V1_delayMat_filename)->default_value("V1_delayMat"), "file that stores V1 to V1 transmission delay within the neighboring blocks")
        ("fV1_vec", po::value<string>(&V1_vec_filename)->default_value("V1_vec"), "file that stores V1 to V1 connection ID, strength and transmission delay outside the neighboring blocks")
		("fBlock_pos", po::value<string>(&block_pos_filename)->default_value("block_pos"), "file that stores the center coord of each block")
		("fNeighborBlock", po::value<string>(&neighborBlock_filename)->default_value("neighborBlock"), "file that stores the neighboring blocks' ID for each block")
		("fConnectome_cfg", po::value<string>(&output_cfg_filename)->default_value("connectome_cfg"), "file stores parameters in current cfg_file")
		("fStats", po::value<string>(&stats_filename)->default_value("conStats"), "file that stores the statistics of connections");

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

    ifstream fV1_pos, fV1_feature;
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

	if (!conLGN_suffix.empty()) {
        conLGN_suffix = '_' + conLGN_suffix;
    }
    conLGN_suffix = conLGN_suffix + ".bin";

	Float p_n_LGNeff;
	Float max_LGNeff;
	Size nType;
	vector<Size> typeAccCount;
	ifstream fLGN_V1_cfg(LGN_V1_cfg_filename + conLGN_suffix, fstream::in | fstream::binary);
	if (!fLGN_V1_cfg) {
		cout << "Cannot open or find " << LGN_V1_cfg_filename + conLGN_suffix <<"\n";
		return EXIT_FAILURE;
	} else {
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&p_n_LGNeff), sizeof(Float));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&max_LGNeff), sizeof(Float));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&nType), sizeof(Size));
		typeAccCount.assign(nType, 0);
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&typeAccCount[0]), nType*sizeof(Size));
	}
	cout << "type accumulate to 1024:\n";
	vector<Size> typeAcc0(1,0);
	for (PosInt i=0; i<nType; i++) {
		cout << typeAccCount[i];
		if (i < nType-1) cout << ",";
		else cout << "\n";
		typeAcc0.push_back(typeAccCount[i]);
	}
	Size* d_typeAcc0;
    checkCudaErrors(cudaMalloc((void**)&d_typeAcc0, (nType+1)*sizeof(Size)));
    checkCudaErrors(cudaMemcpy(d_typeAcc0, &(typeAcc0[0]), (nType+1)*sizeof(Size), cudaMemcpyHostToDevice));

	Size nType0;
	Size nArchtype = nTypeHierarchy.size();
    cout << nArchtype << " archtypes\n";
	if (nArchtype < 1) {
		cout << "at least define one type of neuron with nTypeHierarchy.\n";
		return EXIT_FAILURE;
	} else {
		if (nArchtype > 2) {
        	cout << "nTypeHierarchy has more than 2 elements, only E and I types are implemented\n"; 
			return EXIT_FAILURE;
		} else {
			nType0 = accumulate(nTypeHierarchy.begin(), nTypeHierarchy.end(), 0);
			if (nType0 != nType) {
				cout << "nTypeHierarchy inconsistent with LGN_V1 connection built with suffix " << conLGN_suffix << "\n";
				return EXIT_FAILURE;
			}
		}
	}

    if (nTypeMat.size() != nType*nType) {
		cout << "nTypeMat has size of " << nTypeMat.size() << ", should be " << nType*nType << "\n";
		return EXIT_FAILURE;
	}
    if (sTypeMat.size() != nType*nType) {
		cout << "sTypeMat has size of " << sTypeMat.size() << ", should be " << nType*nType << "\n";
		return EXIT_FAILURE;
	}
	// TODO: std.
	{// dendrites and axons
    	if (dAxon.size() != nType) {
    	    cout << "size of dAxon: " << dAxon.size() << " should be consistent with the number of neuronal types at the hierarchical top: " << nType << "\n"; 
    	    return EXIT_FAILURE;
    	}
    	if (dDend.size() != nType) {
    	    cout << "size of dDend: " << dDend.size() << " should be consistent with the number of neuronal types at the hierarchical top: " << nType << "\n"; 
    	    return EXIT_FAILURE;
    	}

    	if (rAxon.size() != nType) {
    	    cout << "size of rAxon: " << rAxon.size() << " should be consistent with the number of neuronal types at the hierarchical top: " << nType << "\n"; 
    	    return EXIT_FAILURE;
    	} else {
    	    // adjust the scale
    	    for (Float &r: rAxon){
    	        r *= dScale;
    	    }
    	}
    	if (rDend.size() != nType) {
    	    cout << "size of rDend: " << rDend.size() << " should be consistent with the number of neuronal types at the hierarchical top: " << nType << "\n"; 
    	    return EXIT_FAILURE;
    	} else {
    	    // adjust the scale
    	    for (Float &r: rDend){
    	        r *= dScale;
    	    }
    	}
	}

    if (typeAccCount.size() != nType) {
        cout << "the accumulative distribution of neuronal type <typeAccCount> has size of " << typeAccCount.size() << ", should be " << nType << ",  <nType>\n";
        return EXIT_FAILURE;
    }

    //if (FF_FB_ratio == 0) {
    //    cout << "FF_FB_ratio cannot be zero\n";
    //}
    //if (targetFR.size() != nType) {
    //    cout << "target firing rates of different neuronal types <targetFR> has size of " << targetFR.size() << ", should be " << nType << ",  <nType>\n";
    //    return EXIT_FAILURE;
    //}

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
    for (PosInt i = 0; i<networkSize; i++) {
        featureValue[networkSize+i] = (featureValue[networkSize+i] - 0.5)*M_PI;
    }
	fV1_feature.close();

    if (typeFeatureMat.size()/(nType*nType) != nFeature) {
		cout << "typeFeatureMat has " << typeFeatureMat.size()/(nType*nType) << " feature mats, should be " << nFeature << "\n";
		return EXIT_FAILURE;
	}

    initializePreferenceFunctions(nFeature);

    //hInitialize_package hInit_pack(nArchtype, nType, nFeature, nTypeHierarchy, typeAccCount, targetFR, rAxon, rDend, dAxon, dDend, typeFeatureMat, sTypeMat, nTypeMat);
    hInitialize_package hInit_pack(nArchtype, nType, nFeature, nTypeHierarchy, typeAccCount, rAxon, rDend, dAxon, dDend, typeFeatureMat, sTypeMat, nTypeMat);
	initialize_package init_pack(nArchtype, nType, nFeature, hInit_pack);

    //Float speedOfThought = 1.0f; specify instead in patch.cu through patchV1.cfg

    // TODO: types that shares a smaller portion than 1/neuronPerBlock
    if (typeAccCount.back() != neuronPerBlock) {
		cout << "type acc. dist. end with " << typeAccCount.back() << " should be " << neuronPerBlock << "\n";
        return EXIT_FAILURE;
    }

    if (!suffix.empty()) {
		cout << " suffix = " << suffix << "\n";
        suffix = '_' + suffix;
    }
    suffix = suffix + ".bin";
    nearNeighborBlock += 1; // including self
    fV1_conMat.open(V1_conMat_filename + suffix, ios::out | ios::binary);
	if (!fV1_conMat) {
		cout << "cannot open " << V1_conMat_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fV1_conMat.write((char*) &nearNeighborBlock, sizeof(Size));
    }
    fV1_delayMat.open(V1_delayMat_filename + suffix, ios::out | ios::binary);
	if (!fV1_delayMat) {
		cout << "cannot open " << V1_delayMat_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fV1_delayMat.write((char*) &nearNeighborBlock, sizeof(Size));
    }
    fV1_vec.open(V1_vec_filename + suffix, ios::out | ios::binary);
	if (!fV1_vec) {
		cout << "cannot open " << V1_vec_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	}
    fBlock_pos.open(block_pos_filename + suffix, ios::out | ios::binary);
	if (!fBlock_pos) {
		cout << "cannot open " << block_pos_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	}
    fNeighborBlock.open(neighborBlock_filename + suffix, ios::out | ios::binary);
	if (!fNeighborBlock) {
		cout << "cannot open " << neighborBlock_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	}
    fStats.open(stats_filename + suffix, ios::out | ios::binary);
	if (!fStats) {
		cout << "cannot open " << stats_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	}

    maxNeighborBlock += 1; // including self
    // check memory availability
    size_t memorySize, d_memorySize, matSize;

    size_t neighborSize = 2*nblock*sizeof(Float) + // block_x and y
        			      (maxNeighborBlock + 1)*nblock*sizeof(Size); // neighborBlockId and nNeighborBlock

    size_t statSize = 2*nType*networkSize*sizeof(Size) + // preTypeConnected and *Avail
        	          nType*networkSize*sizeof(Float); // preTypeStrSum
    size_t vecSize = 2*static_cast<size_t>(maxDistantNeighbor)*networkSize*sizeof(Float) + // con and delayVec
        		     static_cast<size_t>(maxDistantNeighbor)*networkSize*sizeof(Size) + // vecID
        		     networkSize*sizeof(Size); // nVec

	size_t deviceOnlyMemSize = 2*networkSize*sizeof(Float) + // rden and raxn
         					   2*networkSize*sizeof(Float) + // dden and daxn
         					   nType*nFeature*networkSize*sizeof(Float) + // preF_type
         					   nType*networkSize*sizeof(Float) + // preS_type
         					   nType*networkSize*sizeof(Float) + // preN_type
                               networkSize*sizeof(Size) + // preType
         					   networkSize*sizeof(curandStateMRG32k3a); //state



	if (synPerCon.size() != nType*nType) {
		cout << "the size of synPerCon has size of " << synPerCon.size() << " != " << nType << " x " << nType << "\n";
		return EXIT_FAILURE;
	} 

	if (synPerConFF.size() != nType && synPerConFF.size() != 1) {
		cout << "the size of synPerConFF has size of " << synPerConFF.size() << " != " << nType << " or 1.\n";
		return EXIT_FAILURE;
	}
	
    Float *d_synPerConFF;
    Float *d_synPerCon;
    checkCudaErrors(cudaMalloc((void**)&d_synPerConFF, nType*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void**)&d_synPerCon, nType*nType*sizeof(Float)));
	checkCudaErrors(cudaMemcpy(d_synPerConFF, &(synPerConFF[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_synPerCon, &(synPerCon[0]), nType*nType*sizeof(Float), cudaMemcpyHostToDevice));

	Float *d_extExcRatio;
	Size* max_N = new Size[nType];
	Size* d_max_N;
    checkCudaErrors(cudaMalloc((void**)&d_max_N, nType*sizeof(Size)));
    checkCudaErrors(cudaMalloc((void**)&d_extExcRatio, nType*sizeof(Float)));
	Size sum_max_N = 0;
	for (PosInt i=0; i<nType; i++) {
		max_N[i] = 0;
		for (PosInt j=0; j<nType; j++) {
			if (hInit_pack.nTypeMat[i*nType + j]*(1-extExcRatio[j]) > max_N[i]) {
				max_N[i] = static_cast<Size>(hInit_pack.nTypeMat[i*nType + j] * (1-extExcRatio[j]));
			}
		}
		sum_max_N += max_N[i];
	}
	cout << " sum_max_N = " << sum_max_N << "\n";
    checkCudaErrors(cudaMemcpy(d_max_N, max_N, nType*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_extExcRatio, &(extExcRatio[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	delete []max_N;
	deviceOnlyMemSize += networkSize*sizeof(Size)*sum_max_N; // tmp_vecID

	void *cpu_chunk;
    Int half = 1;
    Size maxChunkSize = nblock;
    //Size maxChunkSize = 1;
	do { 
        if (half > 1) {
            Size half0 = maxChunkSize/half;
            Size half1 = maxChunkSize - half0;
            maxChunkSize = (half0 > half1)? half0: half1;
        }
        matSize = 2*nearNeighborBlock*neuronPerBlock*neuronPerBlock*maxChunkSize*sizeof(Float); // con and delayMat

        memorySize = matSize + vecSize + statSize + neighborSize;

        d_memorySize = memorySize + deviceOnlyMemSize +
                       nFeature*networkSize*sizeof(Float) + 
                       usingPosDim*networkSize*sizeof(double); 

        half *= 2;
	    cpu_chunk = malloc(memorySize);
    } while ((cpu_chunk == NULL || d_memorySize > deviceProps.totalGlobalMem*0.8) && nblock > 1);
    
    Size nChunk = (nblock + maxChunkSize-1) /maxChunkSize - 1;
    cout << "nChunk = " << nChunk+1 << ", with maxChunkSize: " << maxChunkSize << " in total " << nblock << " blocks.\n";
    Size remainChunkSize = nblock%maxChunkSize;
	if (remainChunkSize == 0) {
		remainChunkSize = maxChunkSize;
	}
    assert(maxChunkSize * nChunk + remainChunkSize == nblock);
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
    Float* __restrict__ preF_type = daxn + networkSize;
    Float* __restrict__ preS_type = preF_type + nType*nFeature*networkSize;
    Size*  __restrict__ preN_type = (Size*) (preS_type + nType*networkSize);
    Size*  __restrict__ d_preType = preN_type + nType*networkSize;
    curandStateMRG32k3a* __restrict__ state = (curandStateMRG32k3a*) (d_preType + networkSize);
    PosInt* __restrict__ tmp_vecID = (PosInt*) (state + networkSize);

	// copy from host to device indivdual chunk
    Float* __restrict__ d_feature = (Float*) (tmp_vecID + networkSize*sum_max_N);
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
	assert(static_cast<void*>((char*)gpu_chunk + d_memorySize) == static_cast<void*>(d_preTypeStrSum + nType*networkSize));

    // for array usage on the device in function "generate_connections"
    Size localHeapSize = (sizeof(Float)*maxNeighborBlock*neuronPerBlock + sizeof(Size)*nType*3)*neuronPerBlock*deviceProps.multiProcessorCount;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, localHeapSize);
    printf("heap size preserved %f Mb\n", localHeapSize*1.5/1024/1024);

	/*
    Float* LGN_V1_sSumMax = new Float[nType];
    Float* LGN_V1_sSumMean = new Float[nType];
    Float* LGN_V1_sSum = new Float[networkSize];
    
    read_LGN_sSum(LGN_V1_s_filename + conLGN_suffix, LGN_V1_sSum, LGN_V1_sSumMax, LGN_V1_sSumMean, &(typeAccCount[0]), nType, nblock, false);

    Float* d_LGN_V1_sSum;
    checkCudaErrors(cudaMalloc((void**)&d_LGN_V1_sSum, networkSize*sizeof(Float)));
    checkCudaErrors(cudaMemcpy(d_LGN_V1_sSum, LGN_V1_sSum, networkSize*sizeof(Float), cudaMemcpyHostToDevice));

	Float* presetConstExc = new Float[nType];
	for (PosInt i=0; i<nType; i++) {
		presetConstExc[i] = p_n_LGNeff + hInit_pack.sumType[i];
		if (presetConstExc[i] - LGN_V1_sSumMax[i] < hInit_pack.sumType[i]*min_FB_ratio) {
			cout << "Exc won't be constant for type " << i << " with current parameter set, min_FB_ratio will be utilized, presetConstExc = " << presetConstExc[i] << ", LGN_V1_sSumMax = " << LGN_V1_sSumMax[i] << ", sumType = " << hInit_pack.sumType[i] << "\n";
		}
	}

    //Float* d_LGN_V1_sSumMax;
    //Float* d_LGN_V1_sSumMean;
    //checkCudaErrors(cudaMalloc((void**)&d_LGN_V1_sSumMax, nType*sizeof(Float)));
    //checkCudaErrors(cudaMalloc((void**)&d_LGN_V1_sSumMean, nType*sizeof(Float)));
    //checkCudaErrors(cudaMemcpy(d_LGN_V1_sSumMax, LGN_V1_sSumMax, nType*sizeof(Float), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_LGN_V1_sSumMean, LGN_V1_sSumMean, nType*sizeof(Float), cudaMemcpyHostToDevice));
    delete []LGN_V1_sSumMax;
    delete []LGN_V1_sSumMean;
	delete []presetConstExc;
	*/
	Float* presetConstExcSyn = new Float[nType];
	Size* nLGN_V1 = new Size[networkSize];
	Size* nLGN_V1_Max = new Size[nType];
    
    read_LGN_V1(LGN_V1_s_filename + conLGN_suffix, nLGN_V1, nLGN_V1_Max, &(typeAccCount[0]), nType);

    Size* d_nLGN_V1;
    checkCudaErrors(cudaMalloc((void**)&d_nLGN_V1, networkSize*sizeof(Size)));
    checkCudaErrors(cudaMemcpy(d_nLGN_V1, nLGN_V1, networkSize*sizeof(Size), cudaMemcpyHostToDevice));

	for (PosInt i=0; i<nType; i++) {
		presetConstExcSyn[i] = p_n_LGNeff*synPerConFF[i] + hInit_pack.nTypeMat[i]*synPerCon[i];
		if (presetConstExcSyn[i] - nLGN_V1_Max[i]*synPerConFF[i] < hInit_pack.nTypeMat[i]*synPerCon[i]*min_FB_ratio) {
			cout << "Exc won't be constant for type " << i << " with current parameter set, min_FB_ratio will be utilized, presetConstExcSyn = " << presetConstExcSyn[i] << ", max ffExcSyn = " << nLGN_V1_Max[i]*synPerConFF[i] << ", corticalExcSyn = " << hInit_pack.nTypeMat[i]*synPerCon[i] << "\n";
		}
	}
	delete []presetConstExcSyn;
    delete []nLGN_V1;
    delete []nLGN_V1_Max;

	
    //cudaStream_t s0, s1, s2;
    //cudaEvent_t i0, i1, i2;
    //cudaEventCreate(&i0);
    //cudaEventCreate(&i1);
    //cudaEventCreate(&i2);
    //checkCudaErrors(cudaStreamCreate(&s0));
    //checkCudaErrors(cudaStreamCreate(&s1));
    //checkCudaErrors(cudaStreamCreate(&s2));
    checkCudaErrors(cudaMemcpy(d_feature, &featureValue[0], nFeature*networkSize*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pos, &pos[0], usingPosDim*networkSize*sizeof(double), cudaMemcpyHostToDevice));
	Float* ExcRatio = new Float[networkSize];
	Float* d_ExcRatio;
    checkCudaErrors(cudaMalloc((void**)&d_ExcRatio, networkSize*sizeof(Float)));

    initialize<<<nblock, neuronPerBlock>>>(
        state,
	    d_preType,
		rden, raxn, dden, daxn,
		//preF_type, preS_type, preN_type, d_LGN_V1_sSum, d_ExcRatio, d_extExcRatio, min_FB_ratio,
		preF_type, preS_type, preN_type, d_nLGN_V1, d_ExcRatio, d_extExcRatio, d_synPerCon, d_synPerConFF, min_FB_ratio,
		init_pack, seed, networkSize, nType, nArchtype, nFeature, CmoreN, p_n_LGNeff);
	getLastCudaError("initialize failed");
	checkCudaErrors(cudaDeviceSynchronize());
    printf("initialzied\n");
    checkCudaErrors(cudaMemcpy(ExcRatio, d_ExcRatio, networkSize*sizeof(Float), cudaMemcpyDeviceToHost));
    //Size shared_mem;
    cal_blockPos<<<nblock, neuronPerBlock>>>(
        d_pos, 
		d_block_x, d_block_y, 
		networkSize);
	getLastCudaError("cal_blockPos failed");
    printf("block centers calculated\n");
	//shared_mem = sizeof(Size);
    // blocks -> blocks, threads -> cal neighbor blocks
    get_neighbor_blockId<<<nblock, neuronPerBlock, maxNeighborBlock*(sizeof(PosInt)+sizeof(Float))>>>(
        d_block_x, d_block_y, 
		d_neighborBlockId, d_nNeighborBlock, 
		nblock, blockROI, maxNeighborBlock);
	getLastCudaError("get_neighbor_blockId failed");
    printf("neighbor blocks acquired\n");
    
	checkCudaErrors(cudaMemcpy(block_x, d_block_x, neighborSize, cudaMemcpyDeviceToHost)); 	
    //TODO: generate conMat concurrently
    //shared_mem = neuronPerBlock*sizeof(Float) + neuronPerBlock*sizeof(Float) + neuronPerBlock*sizeof(Size);
    Size current_nblock;
    PosInt offset = 0; // memory offset

    for (PosInt iChunk = 0; iChunk < nChunk+1; iChunk++) {
        if (iChunk < nChunk) current_nblock = maxChunkSize;
        else current_nblock = remainChunkSize;
        cout << iChunk << ": current_nblock = " << current_nblock << "\n";
        size_t current_matSize = static_cast<size_t>(current_nblock)*nearNeighborBlock*neuronPerBlock*neuronPerBlock*sizeof(Float);
	    checkCudaErrors(cudaMemset(d_conMat, 0, current_matSize)); // initialize for each chunk
		cout << "generate_connections<<<" << current_nblock << ", " << neuronPerBlock << ">>>\n";
        generate_connections<<<current_nblock, neuronPerBlock>>>(
            d_pos,
	    	preF_type, preS_type, preN_type,
	    	d_neighborBlockId, d_nNeighborBlock,
	    	rden, raxn,
	    	d_conMat, d_delayMat,
	    	d_conVec, d_delayVec,
			d_max_N, tmp_vecID,
	    	d_vecID, d_nVec,
	    	d_preTypeConnected, d_preTypeAvail, d_preTypeStrSum,
	    	d_preType, d_feature,
	    	dden, daxn,
			d_typeAcc0,
	    	state,
	    	sum_max_N, offset, networkSize, maxDistantNeighbor, nearNeighborBlock, maxNeighborBlock, nType, nFeature, gaussian_profile, strictStrength, minConTol);
	    checkCudaErrors(cudaDeviceSynchronize());
	    getLastCudaError("generate_connections failed");
        //offset += current_nblock*neuronPerBlock;
        offset += current_nblock; // offset is block_offset

	    checkCudaErrors(cudaMemcpy(conMat, d_conMat, 2*current_matSize, cudaMemcpyDeviceToHost)); // con and delay both copied
        // output connectome data
        fV1_conMat.write((char*)conMat, current_matSize);
        fV1_delayMat.write((char*)delayMat, current_matSize);
    }
    fV1_conMat.close();
    fV1_delayMat.close();
    init_pack.freeMem();

	checkCudaErrors(cudaMemcpy(conVec, d_conVec, vecSize+statSize, cudaMemcpyDeviceToHost)); 	

	hInit_pack.freeMem();

    printf("connectivity constructed\n");
	//checkCudaErrors(cudaStreamDestroy(s0));
    //checkCudaErrors(cudaStreamDestroy(s1));
    //checkCudaErrors(cudaStreamDestroy(s2));

    fBlock_pos.write((char*)block_x, nblock*sizeof(Float));
    fBlock_pos.write((char*)block_y, nblock*sizeof(Float));
    fBlock_pos.close();

    cout << "number of neighbors (self-included): ";
    for (Size i=0; i<nblock; i++) {
        cout << nNeighborBlock[i]; 
        if (nNeighborBlock[i] > nearNeighborBlock) { // sorted according to distance, only choose the nearest nearNeighborBlock blocks
            nNeighborBlock[i] = nearNeighborBlock;
            cout << " (" << nearNeighborBlock << ")";
        }
        cout << ", ";
    }
    cout << "\n";
    fNeighborBlock.write((char*)nNeighborBlock, nblock*sizeof(Size));
    cout << "neighbors blockId: ";
    for (Size i=0; i<nblock; i++) {
        fNeighborBlock.write((char*)&neighborBlockId[i*maxNeighborBlock], nNeighborBlock[i]*sizeof(PosInt));
        cout << i << ": ";
        for (PosInt j=0; j<nNeighborBlock[i]; j++) {
            cout << neighborBlockId[i*maxNeighborBlock + j] << ",";
        }
        cout << "\n";
    }
    fNeighborBlock.close();

    fV1_vec.write((char*)nVec, networkSize*sizeof(Size));
    for (Size i=0; i<networkSize; i++) {
        fV1_vec.write((char*)&(vecID[i*maxDistantNeighbor]), nVec[i]*sizeof(Size));
        fV1_vec.write((char*)&(conVec[i*maxDistantNeighbor]), nVec[i]*sizeof(Float));
        fV1_vec.write((char*)&(delayVec[i*maxDistantNeighbor]), nVec[i]*sizeof(Float));
    }
    fV1_vec.close();

    fStats.write((char*)&nType,sizeof(Size));
    fStats.write((char*)&networkSize,sizeof(Size));
    fStats.write((char*)ExcRatio, networkSize*sizeof(Float));
    fStats.write((char*)preTypeConnected, nType*networkSize*sizeof(Size));
    fStats.write((char*)preTypeAvail, nType*networkSize*sizeof(Size));
    fStats.write((char*)preTypeStrSum, nType*networkSize*sizeof(Float));
    fStats.close();
    
    cout << "stats in  mean: \n";
    Size *preConn = new Size[nType*nType];
    Size *preAvail = new Size[nType*nType];
    Float *preStr = new Float[nType*nType];
    Size *nnType = new Size[nType];
    for (PosInt i=0; i<nType; i++) {
        for (PosInt j=0; j<nType; j++) {
            preConn[i*nType + j] = 0;
            preAvail[i*nType + j] = 0;
            preStr[i*nType + j] = 0.0;
        }
        nnType[i] = 0; 
    }
    for (PosInt i=0; i<nType; i++) {
        for (PosInt j=0; j<networkSize; j++) {
            for (PosInt k=0; k<nType; k++) {
                if (j%blockSize < typeAccCount[k])  {
                    preConn[i*nType + k] += preTypeConnected[i*networkSize + j];
                    preAvail[i*nType + k] += preTypeAvail[i*networkSize + j];
                    preStr[i*nType + k] += preTypeStrSum[i*networkSize + j];
                    if (i == 0) {
                        nnType[k]++;
                    }
                    break;
                }
            }
        }
    }

    for (PosInt i=0; i<nType; i++) {
        for (PosInt j=0; j<nType; j++) {
            preConn[i*nType + j] /= nnType[j];
            preAvail[i*nType + j] /= nnType[j];
            preStr[i*nType + j] /= nnType[j];
        }
        cout << "type " << i << " has " << nnType[i] << " neurons\n";
    }
    delete [] nnType;

    cout << "mean Type:    ";
    for (PosInt i = 0; i < nType; i++) {
        cout << i << ",    ";
    }
    cout << "\n";
    for (PosInt i=0; i<nType; i++) {
        for (PosInt j=0; j<nType; j++) {
            if (j==0) {
                cout << i << ": ";
            }
            cout << "[" << preConn[i*nType +j] << "/" << preAvail[i*nType +j] << ", " << preStr[i*nType + j] << "]";
            if (j==nType-1) {
                cout << "\n";
            }
        }
    }

	// in max
    for (PosInt i=0; i<nType; i++) {
        for (PosInt j=0; j<nType; j++) {
            preConn[i*nType + j] = 0;
            preAvail[i*nType + j] = 0;
            preStr[i*nType + j] = 0;
        }
    }
    for (PosInt i=0; i<nType; i++) {
        for (PosInt j=0; j<networkSize; j++) {
            for (PosInt k=0; k<nType; k++) {
                if (j%blockSize < typeAccCount[k])  {
					if (preConn[i*nType + k] < preTypeConnected[i*networkSize + j]) {
						preConn[i*nType + k] = preTypeConnected[i*networkSize + j];
					}
					if (preAvail[i*nType + k] < preTypeAvail[i*networkSize + j]) {
                    	preAvail[i*nType + k] = preTypeAvail[i*networkSize + j];
					}
					if (preStr[i*nType + k] < preTypeStrSum[i*networkSize + j]) {
                    	preStr[i*nType + k] = preTypeStrSum[i*networkSize + j];
					}
                    break;
                }
            }
        }
    }
	cout << "max Type:    ";
    for (PosInt i = 0; i < nType; i++) {
        cout << i << ",    ";
    }
    cout << "\n";
    for (PosInt i=0; i<nType; i++) {
        for (PosInt j=0; j<nType; j++) {
            if (j==0) {
                cout << i << ": ";
            }
            cout << "[" << preConn[i*nType +j] << "/" << preAvail[i*nType +j] << ", " << preStr[i*nType + j] << "]";
            if (j==nType-1) {
                cout << "\n";
            }
        }
    }

	// in min
    for (PosInt i=0; i<nType; i++) {
        for (PosInt j=0; j<networkSize; j++) {
            for (PosInt k=0; k<nType; k++) {
                if (j%blockSize < typeAccCount[k])  {
					if (preConn[i*nType + k] > preTypeConnected[i*networkSize + j]) {
						preConn[i*nType + k] = preTypeConnected[i*networkSize + j];
					}
					if (preAvail[i*nType + k] > preTypeAvail[i*networkSize + j]) {
                    	preAvail[i*nType + k] = preTypeAvail[i*networkSize + j];
					}
					if (preStr[i*nType + k] > preTypeStrSum[i*networkSize + j]) {
                    	preStr[i*nType + k] = preTypeStrSum[i*networkSize + j];
					}
                    break;
                }
            }
        }
    }
	cout << "min Type:    ";
    for (PosInt i = 0; i < nType; i++) {
        cout << i << ",    ";
    }
    cout << "\n";
    for (PosInt i=0; i<nType; i++) {
        for (PosInt j=0; j<nType; j++) {
            if (j==0) {
                cout << i << ": ";
            }
            cout << "[" << preConn[i*nType +j] << "/" << preAvail[i*nType +j] << ", " << preStr[i*nType + j] << "]";
            if (j==nType-1) {
                cout << "\n";
            }
        }
    }
    delete [] preConn;
    delete [] preAvail;
    delete [] preStr;


	ofstream fConnectome_cfg(output_cfg_filename + suffix, fstream::out | fstream::binary);
	if (!fConnectome_cfg) {
		cout << "Cannot open or find " << output_cfg_filename + suffix <<"\n";
		return EXIT_FAILURE;
	} else {
		fConnectome_cfg.write((char*) &nType, sizeof(Size));
		fConnectome_cfg.write((char*) (&typeAccCount[0]), nType*sizeof(Size));
		fConnectome_cfg.write((char*) (&synPerCon[0]), nType*nType*sizeof(Size));
		fConnectome_cfg.write((char*) (&synPerConFF[0]), nType*sizeof(Size));
		fConnectome_cfg.write((char*) (&typeAccCount[0]), nType*sizeof(Size));
		fConnectome_cfg.write((char*) (&nTypeMat[0]), nType*nType*sizeof(Size));
		fConnectome_cfg.write((char*) (&sTypeMat[0]), nType*nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&rDend[0]), nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&rAxon[0]), nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&dDend[0]), nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&dAxon[0]), nType*sizeof(Float));
		fConnectome_cfg.close();
	}

    
	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(gpu_chunk));
    checkCudaErrors(cudaFree(d_typeAcc0));
    //checkCudaErrors(cudaFree(d_LGN_V1_sSum));
    checkCudaErrors(cudaFree(d_nLGN_V1));
    checkCudaErrors(cudaFree(d_ExcRatio));
    checkCudaErrors(cudaFree(d_max_N));
    checkCudaErrors(cudaFree(d_extExcRatio));
    checkCudaErrors(cudaFree(d_synPerConFF));
    checkCudaErrors(cudaFree(d_synPerCon));
	free(cpu_chunk);
    return 0;
}
