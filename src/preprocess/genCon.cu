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
	Size maxDistantNeighbor, gap_maxDistantNeighbor;
	vector<Size> nTypeHierarchy;
    string V1_type_filename, V1_feature_filename, V1_pos_filename, LGN_V1_s_filename, suffix, conLGN_suffix, LGN_V1_cfg_filename, output_cfg_filename;
	string V1_conMat_filename, V1_delayMat_filename, V1_gapMat_filename;
	string V1_vec_filename, V1_gapVec_filename;
	string block_pos_filename, neighborBlock_filename, stats_filename;
    Float dScale, blockROI, blockROI_max;
	vector<Float> extExcRatio;
	vector<Float> extSynRatio;
    //vector<Float> targetFR;
    ///Float FF_FB_ratio;
    Float min_FB_ratio;
    Float C_InhRatio;
    //Float LGN_targetFR;
    bool strictStrength;
    bool CmoreN;
    bool ClessI;
	Size usingPosDim;
	Float longRangeROI;
	Float longRangeCorr;
	Float disGauss;
	vector<Float> rDend, rAxon;
	vector<Float> dDend, dAxon;
    vector<Float> synapticLoc;
    vector<Float> sTypeMat, gap_sTypeMat;
    vector<Float> typeFeatureMat, gap_fTypeMat;
    vector<Size> nTypeMat, nInhGap;
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
        ("DisGauss", po::value<Float>(&disGauss), "if set true, conn. prob. based on distance will follow a 2D gaussian with a variance. of (raxn*raxn + rden*rden)/(2*ln(2))*disGauss, otherwise 0 will based on the overlap of the area specified by raxn and rden")
        ("strictStrength", po::value<bool>(&strictStrength), "strictly match preset summed connection")
        ("CmoreN", po::value<bool>(&CmoreN), "if true complex gets more connections otherwise stronger strength")
        ("rDend", po::value<vector<Float>>(&rDend),  "a vector of dendritic extensions' radius, size of nType ")
        ("rAxon", po::value<vector<Float>>(&rAxon),  "a vector of axonic extensions' radius, size of nType")
        ("dScale",po::value<Float>(&dScale)->default_value(1.0),"a scaling ratio of all the neurites' lengths <radius>")
        ("longRangeROI", po::value<Float>(&longRangeROI), "ROI of long-range cortical input")
        ("longRangeCorr", po::value<Float>(&longRangeCorr), "correlation between long-range cortical inputs that cortical cells receives")
        ("dDend", po::value<vector<Float>>(&dDend), "vector of dendrites' densities, size of nType")
        ("dAxon", po::value<vector<Float>>(&dAxon), "vector of axons' densities, size of nType")
        ("synapticLoc", po::value<vector<Float>>(&synapticLoc), " maximal synaptic location relative to the soma, percentage of dendrite, of different presynaptic type, size of [nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
		("nTypeHierarchy", po::value<vector<Size>>(&nTypeHierarchy), "a vector of hierarchical types differs in non-functional properties: reversal potentials, characteristic lengths of dendrite and axons, e.g. in size of nArchtype, {Exc-Pyramidal, Exc-stellate; Inh-PV, Inh-SOM, Inh-LTS} then the vector would be {3, 2}, with Exc and Inh being arch type")
		//("LGN_targetFR", po::value<Float>(&LGN_targetFR), "target firing rate of a LGN cell")
		//("targetFR", po::value<vector<Float>>(&targetFR), "a vector of target firing rate of different neuronal types")
		//("FF_FB_ratio", po::value<Float>(&FF_FB_ratio), "excitation ratio of FF over total excitation")
		("extExcRatio", po::value<vector<Float>>(&extExcRatio), "external cortical excitation ratio to E and I")
		("extSynRatio", po::value<vector<Float>>(&extSynRatio), "external cortical exc connection #synapse ratio to E and I")
		("min_FB_ratio", po::value<Float>(&min_FB_ratio), "minimum external cortical excitation ratio")
		("C_InhRatio", po::value<Float>(&C_InhRatio), "minimum inhibition for complex cells")
		("ClessI", po::value<bool>(&ClessI), "lesser inhibition for complex cell")
		("minConTol", po::value<Float>(&minConTol), "minimum difference tolerance of the number of preset cortical connections")
        ("sTypeMat", po::value<vector<Float>>(&sTypeMat), "connection strength matrix between neuronal types, size of [nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("gap_sTypeMat", po::value<vector<Float>>(&gap_sTypeMat), "gap junction strength matrix between inhibitory neuronal types, size of [nTypeI, nTypeI], nTypeI = nTypeHierarchy[1], row_id -> postsynaptic, column_id -> presynaptic")
        ("nTypeMat", po::value<vector<Size>>(&nTypeMat), "#connection matrix between neuronal types, size of [nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("nInhGap", po::value<vector<Size>>(&nInhGap), "#gap junction matrix between inhibitory neuronal types, size of [nTypeHierarchy[1], nTypeHierarchy[1]], symmetric")
        ("synPerConFF",  po::value<vector<Float>>(&synPerConFF), "synpases per feedforward connection, size of nType")
        ("synPerCon",  po::value<vector<Float>>(&synPerCon), "synpases per cortical connection size of [nType, nType]")
        ("typeFeatureMat", po::value<vector<Float>>(&typeFeatureMat), "feature parameter of neuronal types, size of [nFeature, nType, nType], nType = sum(nTypeHierarchy), row_id -> postsynaptic, column_id -> presynaptic")
        ("gap_fTypeMat", po::value<vector<Float>>(&gap_fTypeMat), "feature parameter of neuronal types, size of [nFeature, nTypeI, nTypeI], nTypeI = nTypeHierarchy[1], row_id -> postsynaptic, column_id -> presynaptic")
        ("blockROI", po::value<Float>(&blockROI), "garaunteed radius (center to center) to include neighboring blocks in mm")
        ("blockROI_max", po::value<Float>(&blockROI_max), "max radius (center to center) to include neighboring blocks in mm")
    	("usingPosDim", po::value<Size>(&usingPosDim)->default_value(2), "using <2>D coord. or <3>D coord. when calculating distance between neurons, influencing how the position data is read") 
        ("maxDistantNeighbor", po::value<Size>(&maxDistantNeighbor), "the preserved size of the array that store the presynaptic neurons' ID, who are not in the neighboring blocks")
        ("gap_maxDistantNeighbor", po::value<Size>(&gap_maxDistantNeighbor), "the preserved size of the array that store the pre-junction neurons' ID, who are not in the neighboring blocks")
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
        ("fV1_gapMat", po::value<string>(&V1_gapMat_filename)->default_value("V1_gapMat"), "file that stores V1 to V1 gap junction within the neighboring blocks")
        ("fV1_vec", po::value<string>(&V1_vec_filename)->default_value("V1_vec"), "file that stores V1 to V1 connection ID, strength and transmission delay outside the neighboring blocks")
        ("fV1_gapVec", po::value<string>(&V1_gapVec_filename)->default_value("V1_gapVec"), "file that stores V1 to V1 gap-junction ID, strength and transmission delay outside the neighboring blocks")
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
    ofstream fV1_conMat, fV1_delayMat, fV1_gapMat, fV1_vec, fV1_gapVec;
    ofstream fBlock_pos, fNeighborBlock;
    ofstream fStats;

	if (disGauss > 0) {
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
	Size max_LGNeff;
	Float mean_nLGN;
	Float mean_sLGN;
	Float preset_nLGN;
	Size nType;
	vector<Size> typeAccCount;
	ifstream fLGN_V1_cfg(LGN_V1_cfg_filename + conLGN_suffix, fstream::in | fstream::binary);
	if (!fLGN_V1_cfg) {
		cout << "Cannot open or find " << LGN_V1_cfg_filename + conLGN_suffix <<"\n";
		return EXIT_FAILURE;
	} else {
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&p_n_LGNeff), sizeof(Float));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&max_LGNeff), sizeof(Size));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&nType), sizeof(Size));
		typeAccCount.assign(nType, 0);
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&typeAccCount[0]), nType*sizeof(Size));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&mean_nLGN), sizeof(Float));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&mean_sLGN), sizeof(Float));
	}
	preset_nLGN = mean_nLGN;
	cout << "type accumulate to 1024:\n";
	vector<Size> typeAcc0(1,0);
	for (PosInt i=0; i<nType; i++) {
		cout << typeAccCount[i];
		if (i < nType-1) cout << ",";
		else cout << "\n";
		typeAcc0.push_back(typeAccCount[i]);
	}
	assert(typeAccCount.back() == blockSize);
	Size nTypeE = nTypeHierarchy[0];
	Size nTypeI = nTypeHierarchy[1];
	Size nE = typeAccCount[nTypeE-1];
	Size nI = blockSize - typeAccCount[nTypeE-1];
	cout << "nTypeE = " << nTypeE << ", nTypeI = " << nTypeI << ", nE = " << nE << ", nI = " << nI << "\n";
	Size* d_typeAcc0;
    checkCudaErrors(cudaMalloc((void**)&d_typeAcc0, (nType+1)*sizeof(Size)));
    checkCudaErrors(cudaMemcpy(d_typeAcc0, &(typeAcc0[0]), (nType+1)*sizeof(Size), cudaMemcpyHostToDevice));

    if (synapticLoc.size() != nType*nType) {
		cout << "synapticLoc has size of " << synapticLoc.size() << ", should be " << nType*nType << "\n";
		return EXIT_FAILURE;
	}
	Float* d_synloc;
    checkCudaErrors(cudaMalloc((void**)&d_synloc, (nType*nType)*sizeof(Float)));
    checkCudaErrors(cudaMemcpy(d_synloc, &(synapticLoc[0]), (nType*nType)*sizeof(Float), cudaMemcpyHostToDevice));

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
			cout << "nType0 = " << nType0 << "\n";
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

    if (nInhGap.size() != nTypeI*nTypeI) {
		cout << "nInhGap has size of " << nInhGap.size() << ", should be " << nTypeI*nTypeI << "\n";
		return EXIT_FAILURE;
	} else {
		for (PosInt i=0; i<nTypeI; i++) {
			for (PosInt j=0; j<i; j++) {
				if (nInhGap[i*nType + j] != nInhGap[j*nType + i]) {
					cout << "nInhGap is not symmetric\n";
				}
			}
		}
	}

    if (gap_sTypeMat.size() != nTypeI*nTypeI) {
		cout << "gap_sTypeMat has size of " << gap_sTypeMat.size() << ", should be " << nTypeI*nTypeI << "\n";
		return EXIT_FAILURE;
	} else {
		for (PosInt i=0; i<nTypeI; i++) {
			for (PosInt j=0; j<i; j++) {
				if (gap_sTypeMat[i*nType + j] != gap_sTypeMat[j*nType + i]) {
					cout << "gap_sTypeMat is not symmetric\n";
				}
			}
		}
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
	Size mI = nblock*nI;

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
	for (PosInt iF = 0; iF<nFeature; iF++) {
		cout << iF << "th featureValue range: [" << *min_element(featureValue.begin()+iF*networkSize, featureValue.begin()+(iF+1)*networkSize) << ", " << *max_element(featureValue.begin()+iF*networkSize, featureValue.begin()+(iF+1)*networkSize) << "]\n";
	}
    for (PosInt i = 0; i<networkSize; i++) {
        featureValue[networkSize+i] = (featureValue[networkSize+i] - 0.5)*M_PI;
    }
	fV1_feature.close();

    if (typeFeatureMat.size()/(nType*nType) != nFeature) {
		cout << "typeFeatureMat has " << typeFeatureMat.size()/(nType*nType) << " features, should be " << nFeature << "\n";
		return EXIT_FAILURE;
	}

    if (gap_fTypeMat.size()/(nTypeI*nTypeI) != nFeature) {
		cout << "gap_fTypeMat has " << gap_fTypeMat.size()/(nTypeI*nTypeI) << " features, should be " << nFeature << "\n";
		return EXIT_FAILURE;
	} else {
		for (PosInt k=0; k<nFeature; k++) {
			for (PosInt i=0; i<nTypeI; i++) {
				for (PosInt j=0; j<i; j++) {
					if (gap_fTypeMat[k*nTypeI*nTypeI + i*nType + j] != gap_fTypeMat[k*nTypeI*nTypeI + j*nType + i]) {
						cout << "gap_fTypeMat is not symmetric for feature " << k << "\n";
					}
				}
			}
		}
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
    fV1_gapMat.open(V1_gapMat_filename + suffix, ios::out | ios::binary);
	if (!fV1_gapMat) {
		cout << "cannot open " << V1_gapMat_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fV1_gapMat.write((char*) &nearNeighborBlock, sizeof(Size));
    }
    fV1_vec.open(V1_vec_filename + suffix, ios::out | ios::binary);
	if (!fV1_vec) {
		cout << "cannot open " << V1_vec_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	}
    fV1_gapVec.open(V1_gapVec_filename + suffix, ios::out | ios::binary);
	if (!fV1_gapVec) {
		cout << "cannot open " << V1_gapVec_filename + suffix << " to write.\n";
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

    size_t gap_statSize = nTypeI*mI*sizeof(Size) + // preTypeGapped
        	              nTypeI*mI*sizeof(Float); // preTypeStrGapped

    size_t vecSize = 2*static_cast<size_t>(maxDistantNeighbor)*networkSize*sizeof(Float) + // con and delayVec
        		     static_cast<size_t>(maxDistantNeighbor)*networkSize*sizeof(Size) + // vecID
        		     networkSize*sizeof(Size); // nVec

    size_t gap_vecSize = 2*static_cast<size_t>(gap_maxDistantNeighbor)*mI*sizeof(Float) + // con and delayVec
        		     static_cast<size_t>(gap_maxDistantNeighbor)*mI*sizeof(Size) + // vecID
        		     mI*sizeof(Size); // nVec

	size_t deviceOnlyMemSize = 2*networkSize*sizeof(Float) + // rden and raxn
         					   2*networkSize*sizeof(Float) + // dden and daxn
         					   nFeature*nType*networkSize*sizeof(Float) + // preF_type
         					   nType*networkSize*sizeof(Float) + // preS_type
         					   nType*networkSize*sizeof(Float) + // preN_type
                               networkSize*sizeof(Size) + // preType
         					   networkSize*sizeof(curandStateMRG32k3a); //state



	if (synPerCon.size() != nType*nType) {
		cout << "the size of synPerCon has size of " << synPerCon.size() << " != " << nType << " x " << nType << "\n";
		return EXIT_FAILURE;
	} else {
		for (PosInt i=0; i<nType; i++) {
			cout << "synPerCon["<<i <<"]: " << synPerCon[i] << "-> ";
			synPerCon[i] /= (1-extExcRatio[i]) + extExcRatio[i]*extSynRatio[i];
			cout << synPerCon[i] << "\n";
		}
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
        cout << "max_N[" << i << "] = " << max_N[i] << "\n";
	}
	cout << " sum_max_N = " << sum_max_N << "\n";

	Size gap_sum_max_N = accumulate(nInhGap.begin(), nInhGap.end(), 0);
	cout << "gap sum_max_N = " << gap_sum_max_N << "\n";

    checkCudaErrors(cudaMemcpy(d_max_N, max_N, nType*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_extExcRatio, &(extExcRatio[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	delete []max_N;

    Float* gap_preF_type;
    checkCudaErrors(cudaMalloc((void**)&gap_preF_type, nFeature*nTypeI*nTypeI*sizeof(Float)));
    checkCudaErrors(cudaMemcpy(gap_preF_type, &(gap_fTypeMat[0]), nFeature*nTypeI*nTypeI*sizeof(Float), cudaMemcpyHostToDevice));
    Float* gap_preS_type;
    checkCudaErrors(cudaMalloc((void**)&gap_preS_type, nTypeI*nTypeI*sizeof(Float)));
    checkCudaErrors(cudaMemcpy(gap_preS_type, &(gap_sTypeMat[0]), nTypeI*nTypeI*sizeof(Float), cudaMemcpyHostToDevice));
    Size*  gap_preN_type;
    checkCudaErrors(cudaMalloc((void**)&gap_preN_type, nTypeI*nTypeI*sizeof(Size)));
    checkCudaErrors(cudaMemcpy(gap_preN_type, &(nInhGap[0]), nTypeI*nTypeI*sizeof(Size), cudaMemcpyHostToDevice));

	void *cpu_chunk;
    Int half = 1;
    Size maxChunkSize = nblock;
    size_t disNeighborSize;
    size_t gap_disNeighborSize;
    size_t tmpVecSize;
    size_t localHeapSize;

	do { 
        //half *= 2;
        if (half > 1) {
            Size half0 = maxChunkSize/half;
            Size half1 = maxChunkSize - half0;
            maxChunkSize = (half0 > half1)? half0: half1;
        }
        matSize = static_cast<size_t>(2*nearNeighborBlock*neuronPerBlock)*neuronPerBlock*maxChunkSize*sizeof(Float); // con and delayMat

        memorySize = matSize + vecSize + gap_vecSize + statSize + gap_statSize + neighborSize;

        disNeighborSize = sizeof(Float)*static_cast<size_t>((maxNeighborBlock-nearNeighborBlock)*neuronPerBlock)*maxChunkSize*neuronPerBlock; 
        gap_disNeighborSize = sizeof(Float)*static_cast<size_t>((maxNeighborBlock-nearNeighborBlock)*nI)*maxChunkSize*nI; 

	    tmpVecSize = static_cast<size_t>(maxChunkSize*neuronPerBlock)*sizeof(Size); // tmp_vecID
		if (sum_max_N > gap_sum_max_N) {
			tmpVecSize *= sum_max_N;
		} else {
			tmpVecSize *= gap_sum_max_N;
		}

		// share: qid, ratio, typeConnected, postSynLoc, fV
		// nType: sumP, availType, sumType, sumStrType, pN, pS, pF, __vecID, nid
		// nTypeI: ...
        localHeapSize = ((4*sizeof(Size) + 3*sizeof(Float) + sizeof(PosInt*) + nFeature*sizeof(Float))*(nType + nTypeI) + (sizeof(PosInt) + 2*sizeof(Float) + sizeof(bool))*nType + nFeature*sizeof(Float))*static_cast<size_t>(maxChunkSize*neuronPerBlock*deviceProps.multiProcessorCount);
		localHeapSize *= 1.1; // leave some extra room
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, localHeapSize));
        d_memorySize = memorySize + deviceOnlyMemSize + disNeighborSize + gap_disNeighborSize + tmpVecSize  +
                       nFeature*networkSize*sizeof(Float) + 
                       usingPosDim*networkSize*sizeof(double); 

        if (half > 2) {
            free(cpu_chunk);
        }
	    cpu_chunk = malloc(memorySize);
        half *= 2;
        cout << localHeapSize/1024/1024 << "Mb heap size\n";
        cout << memorySize/1024/1024 << "Mb cpu mem\n";
        cout << d_memorySize/1024/1024 << "Mb gpu mem request\n";
        cout << deviceProps.totalGlobalMem/1024/1024 << "Mb gpu mem in total\n";
    } while ((cpu_chunk == NULL || d_memorySize + localHeapSize > deviceProps.totalGlobalMem*0.8) && nblock > 1);
    
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

	Float* gapMat = new Float[static_cast<size_t>(nI*nI*nearNeighborBlock)*nblock];
    Float* d_gapMat; 
    checkCudaErrors(cudaMalloc((void**)&d_gapMat, static_cast<size_t>(nI*nI*nearNeighborBlock)*maxChunkSize*sizeof(Float)));

    // ============ CPU MEM ============
    // blocks
    Float* block_x = (Float*) cpu_chunk;
    Float* block_y = block_x + nblock;
    PosInt* neighborBlockId = (Size*) (block_y + nblock);
    Size* nNeighborBlock = neighborBlockId + maxNeighborBlock*nblock;

    // connectome
    Float* conMat = (Float*) (nNeighborBlock + nblock);
    Float* delayMat = conMat + static_cast<size_t>(nearNeighborBlock*neuronPerBlock)*neuronPerBlock*maxChunkSize;

    Float* conVec = delayMat + static_cast<size_t>(nearNeighborBlock*neuronPerBlock)*neuronPerBlock*maxChunkSize;
    Float* delayVec = conVec + maxDistantNeighbor*networkSize;
    Size* vecID = (Size*) (delayVec + maxDistantNeighbor*networkSize);
    Size* nVec = vecID + maxDistantNeighbor*networkSize;

    // stats
    Size* preTypeConnected = nVec + networkSize;
    Size* preTypeAvail = preTypeConnected + nType*networkSize;
    Float* preTypeStrSum = (Float*) (preTypeAvail + nType*networkSize);

	// gapVec
    Float* gapVec = preTypeStrSum + nType*networkSize;
    Float* gapDelayVec = gapVec + mI*gap_maxDistantNeighbor;
    Size*  gapVecID = (Size*) (gapDelayVec + mI*gap_maxDistantNeighbor);
    Size*  nGapVec = gapVecID + mI*gap_maxDistantNeighbor;

	// gap stats
    Size* preTypeGapped = nGapVec + mI;
    Float* preTypeStrGapped = (Float*) (preTypeGapped + nTypeI*mI);

	assert(static_cast<void*>((char*)cpu_chunk + memorySize) == static_cast<void*>(preTypeStrGapped + nTypeI*mI));

    // ========== GPU mem ============
    // init by kernel, reside on device only
    Float* __restrict__ rden = (Float*) gpu_chunk; 
    Float* __restrict__ raxn = rden + networkSize;
	Float* __restrict__ dden = raxn + networkSize;
	Float* __restrict__ daxn = dden + networkSize;

    Float* __restrict__ preF_type = daxn + networkSize;
    Float* __restrict__ preS_type = preF_type + nFeature*nType*networkSize;
    Size*  __restrict__ preN_type = (Size*) (preS_type + nType*networkSize);
    Size*  __restrict__ d_preType = preN_type + nType*networkSize;
    curandStateMRG32k3a* __restrict__ state = (curandStateMRG32k3a*) (d_preType + networkSize);
    PosInt* __restrict__ tmp_vecID = (PosInt*) (state + networkSize);
    Float* __restrict__ disNeighborP = (Float*) (tmp_vecID + tmpVecSize/sizeof(Size));
    Float* __restrict__ gap_disNeighborP = disNeighborP + disNeighborSize/sizeof(Float);

	// copy from host to device indivdual chunk
    Float* __restrict__ d_feature = gap_disNeighborP + gap_disNeighborSize/sizeof(Float);
    double* __restrict__ d_pos = (double*) (d_feature + nFeature*networkSize);
	// copy by the whole chunk
    // device to host
    Float* __restrict__ d_block_x = (Float*) (d_pos + usingPosDim*networkSize); 
    Float* __restrict__ d_block_y = d_block_x + nblock;
    PosInt*  __restrict__ d_neighborBlockId = (Size*) (d_block_y + nblock);
    Size*  __restrict__ d_nNeighborBlock = d_neighborBlockId + maxNeighborBlock*nblock;

    Float* __restrict__ d_conMat = (Float*) (d_nNeighborBlock + nblock);
    Float* __restrict__ d_delayMat = d_conMat + static_cast<size_t>(nearNeighborBlock*neuronPerBlock)*neuronPerBlock*maxChunkSize;

    Float* __restrict__ d_conVec = d_delayMat + static_cast<size_t>(nearNeighborBlock*neuronPerBlock)*neuronPerBlock*maxChunkSize;
    Float* __restrict__ d_delayVec = d_conVec + networkSize*maxDistantNeighbor;
    Size*  __restrict__ d_vecID = (Size*) (d_delayVec + networkSize*maxDistantNeighbor);
    Size*  __restrict__ d_nVec = d_vecID + networkSize*maxDistantNeighbor;

    // stats
    Size*  __restrict__ d_preTypeConnected = d_nVec + networkSize;
    Size*  __restrict__ d_preTypeAvail = d_preTypeConnected + nType*networkSize;
    Float* __restrict__ d_preTypeStrSum = (Float*) (d_preTypeAvail + nType*networkSize);

	// gapVec
    Float* __restrict__ d_gapVec = d_preTypeStrSum + nType*networkSize;
    Float* __restrict__ d_gapDelayVec = d_gapVec + mI*gap_maxDistantNeighbor;
    Size*  __restrict__ d_gapVecID = (Size*) (d_gapDelayVec + mI*gap_maxDistantNeighbor);
    Size*  __restrict__ d_nGapVec = d_gapVecID + mI*gap_maxDistantNeighbor;

	// gap stats
    Size*  __restrict__ d_preTypeGapped = d_nGapVec + mI;
    Float* __restrict__ d_preTypeStrGapped = (Float*) (d_preTypeGapped + nTypeI*mI);

	// check memory address consistency
	assert(static_cast<void*>((char*)gpu_chunk + d_memorySize) == static_cast<void*>(d_preTypeStrGapped + nTypeI*mI));

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
		presetConstExcSyn[i] = preset_nLGN*synPerConFF[i] + hInit_pack.nTypeMat[i]*synPerCon[i];
		if (presetConstExcSyn[i] - nLGN_V1_Max[i]*synPerConFF[i] < hInit_pack.nTypeMat[i]*synPerCon[i]*min_FB_ratio) {
			cout << "Exc won't be constant for type " << i << " with current parameter set, min_FB_ratio will be utilized, presetConstExcSyn = " << presetConstExcSyn[i] << ", max ffExcSyn = " << nLGN_V1_Max[i]*synPerConFF[i] << ", corticalExcSyn = " << hInit_pack.nTypeMat[i]*synPerCon[i] << "\n";
		}
	}
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
		preF_type, preS_type, preN_type, d_nLGN_V1, d_ExcRatio, d_extExcRatio, d_synPerCon, d_synPerConFF, min_FB_ratio, C_InhRatio,
		init_pack, seed, networkSize, nType, nArchtype, nFeature, CmoreN, ClessI, preset_nLGN);
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
	Size* d_nNearNeighborBlock;
    checkCudaErrors(cudaMalloc((void**)&d_nNearNeighborBlock, nblock*sizeof(Size)));

    get_neighbor_blockId<<<nblock, neuronPerBlock, maxNeighborBlock*(sizeof(PosInt)+sizeof(Float))>>>(
        d_block_x, d_block_y, 
		d_neighborBlockId, d_nNeighborBlock, d_nNearNeighborBlock, 
		nblock, blockROI, blockROI_max, maxNeighborBlock);
	getLastCudaError("get_neighbor_blockId failed");
    printf("neighbor blocks acquired\n");
    
	Size* nNearNeighborBlock = new Size[nblock];
    checkCudaErrors(cudaMemcpy(nNearNeighborBlock, d_nNearNeighborBlock, nblock*sizeof(Size), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(block_x, d_block_x, neighborSize, cudaMemcpyDeviceToHost)); 	
	Size _nearNeighborBlock = *max_element(nNearNeighborBlock, nNearNeighborBlock + nblock);
	if (_nearNeighborBlock > nearNeighborBlock) {
		cout << "increase nearNeighborBlock " << _nearNeighborBlock << "/" << nearNeighborBlock << "\n";
		return EXIT_FAILURE;
	} else {
		cout << "min nearNeighborBlock = " << *min_element(nNearNeighborBlock, nNearNeighborBlock + nblock) << ", max nearNeighborBlock = " << *max_element(nNearNeighborBlock, nNearNeighborBlock + nblock) << " < " << nearNeighborBlock << " < " << maxNeighborBlock << "\n";
	}
	for (PosInt i=0; i<nblock; i++) {
		if (nNeighborBlock[i] - nNearNeighborBlock[i] > maxNeighborBlock - nearNeighborBlock) {
			cout << "increase maxNeighborBlock block "<< i << " non-sense: " << nNeighborBlock[i] << "-" << nNearNeighborBlock[i] << " = " << nNeighborBlock[i] - nNearNeighborBlock[i] << "/ " << maxNeighborBlock - nearNeighborBlock << ".\n";
			return EXIT_FAILURE;
		}
	}
    //TODO: generate conMat concurrently
    //shared_mem = neuronPerBlock*sizeof(Float) + neuronPerBlock*sizeof(Float) + neuronPerBlock*sizeof(Size);
    Size current_nblock;
    PosInt offset = 0; // memory offset

    for (PosInt iChunk = 0; iChunk < nChunk+1; iChunk++) {
        if (iChunk < nChunk) current_nblock = maxChunkSize;
        else current_nblock = remainChunkSize;
        cout << iChunk << ": current_nblock = " << current_nblock << "\n";
        size_t current_matSize = static_cast<size_t>(current_nblock*nearNeighborBlock*neuronPerBlock)*neuronPerBlock*sizeof(Float);
	    checkCudaErrors(cudaMemset(d_conMat, 0, current_matSize)); // initialize for each chunk
		size_t gap_matSize = nearNeighborBlock*nI*nI;
	    checkCudaErrors(cudaMemset(d_gapMat, 0, gap_matSize*current_nblock*sizeof(Float))); // initialize for each chunk
		cout << "generate_connections<<<" << current_nblock << ", " << neuronPerBlock << ">>>\n";
        generate_connections<<<current_nblock, neuronPerBlock>>>(
            d_pos,
	    	preF_type, gap_preF_type,
			preS_type, gap_preS_type,
			preN_type, gap_preN_type,
	    	d_neighborBlockId, d_nNeighborBlock, d_nNearNeighborBlock,
	    	rden, raxn,
	    	d_conMat, d_delayMat, d_gapMat,
	    	d_conVec, d_delayVec, d_gapVec, d_gapDelayVec,
			d_max_N,
			tmp_vecID,
			disNeighborP, gap_disNeighborP,
	    	d_vecID, d_nVec,
			d_gapVecID, d_nGapVec,
	    	d_preTypeConnected, d_preTypeAvail, d_preTypeStrSum,
			d_preTypeGapped, d_preTypeStrGapped,
	    	d_preType, d_feature,
	    	dden, daxn, d_synloc,
			d_typeAcc0,
	    	state,
	    	sum_max_N, gap_sum_max_N, offset, networkSize, mI, maxDistantNeighbor, gap_maxDistantNeighbor, nearNeighborBlock, maxNeighborBlock, nType, nTypeE, nTypeI, nE, nI, nFeature, disGauss, strictStrength, minConTol);
	    checkCudaErrors(cudaDeviceSynchronize());
	    getLastCudaError("generate_connections failed");
        //offset += current_nblock*neuronPerBlock;

	    checkCudaErrors(cudaMemcpy(conMat, d_conMat, 2*current_matSize, cudaMemcpyDeviceToHost)); // con and delay both copied
	    checkCudaErrors(cudaMemcpy(gapMat + offset*gap_matSize, d_gapMat, gap_matSize*current_nblock*sizeof(Float), cudaMemcpyDeviceToHost)); // con and delay both copied
        // output connectome data
        fV1_conMat.write((char*)conMat, current_matSize);
        fV1_delayMat.write((char*)delayMat, current_matSize);
        offset += current_nblock; // offset is block_offset
    }
    fV1_conMat.close();
    fV1_delayMat.close();
	
    init_pack.freeMem();

	checkCudaErrors(cudaMemcpy(conVec, d_conVec, vecSize+statSize, cudaMemcpyDeviceToHost)); 	
	checkCudaErrors(cudaMemcpy(gapVec, d_gapVec, gap_vecSize+gap_statSize, cudaMemcpyDeviceToHost)); 	

	/*===========
	{
		for	(PosInt ib=0; ib<nblock; ib++) {
			cout << "before block#" << ib << " gap stats in  mean: \n";
    		Size* preConn = new Size[nTypeI*nTypeI];
    		Size* preAvail = new Size[nTypeI*nTypeI];
    		Float* preStr = new Float[nTypeI*nTypeI];
    		Size* nnTypeI = new Size[nTypeI];
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=0; j<nTypeI; j++) {
    		        preConn[i*nTypeI + j] = 0;
    		        preAvail[i*nTypeI + j] = 0;
    		        preStr[i*nTypeI + j] = 0.0;
    		    }
    		    nnTypeI[i] = 0; 
    		}
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=ib*nI; j<(ib+1)*nI; j++) {
    		        for (PosInt k=0; k<nTypeI; k++) {
    		            if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
    		                preConn[i*nTypeI + k] += preTypeGapped[i*mI + j];
    		                preAvail[i*nTypeI + k] += preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
    		                preStr[i*nTypeI + k] += preTypeStrGapped[i*mI + j];
    		                if (i == 0) {
    		                    nnTypeI[k]++;
    		                }
    		                break;
    		            }
    		        }
    		    }
    		}

    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=0; j<nTypeI; j++) {
    		        preConn[i*nTypeI + j] /= nnTypeI[j];
    		        preAvail[i*nTypeI + j] /= nnTypeI[j];
    		        preStr[i*nTypeI + j] /= nnTypeI[j];
    		    }
    		    cout << "type " << i << " has " << nnTypeI[i] << " neurons\n";
    		}
    		delete [] nnTypeI;

    		cout << "mean Type:    ";
    		for (PosInt i = 0; i < nTypeI; i++) {
    		    cout << i << ",    ";
    		}
    		cout << "\n";
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=0; j<nTypeI; j++) {
    		        if (j==0) {
    		            cout << i << ": ";
    		        }
    		        cout << "[" << preConn[i*nTypeI +j] << "/" << preAvail[i*nTypeI +j] << ", " << preStr[i*nTypeI + j] << "]";
    		        if (j==nTypeI-1) {
    		            cout << "\n";
    		        }
    		    }
    		}

			// in max
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=0; j<nTypeI; j++) {
    		        preConn[i*nTypeI + j] = 0;
    		        preAvail[i*nTypeI + j] = 0;
    		        preStr[i*nTypeI + j] = 0;
    		    }
    		}
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=ib*nI; j<(ib+1)*nI; j++) {
    		        for (PosInt k=0; k<nTypeI; k++) {
    		            if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
							if (preConn[i*nTypeI + k] < preTypeGapped[i*mI + j]) {
								preConn[i*nTypeI + k] = preTypeGapped[i*mI + j];
							}
							if (preAvail[i*nTypeI + k] < preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j]) {
    		                	preAvail[i*nTypeI + k] = preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
							}
							if (preStr[i*nTypeI + k] < preTypeStrGapped[i*mI + j]) {
    		                	preStr[i*nTypeI + k] = preTypeStrGapped[i*mI + j];
							}
    		                break;
    		            }
    		        }
    		    }
    		}
			cout << "max Type:    ";
    		for (PosInt i = 0; i < nTypeI; i++) {
    		    cout << i << ",    ";
    		}
    		cout << "\n";
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=0; j<nTypeI; j++) {
    		        if (j==0) {
    		            cout << i << ": ";
    		        }
    		        cout << "[" << preConn[i*nTypeI +j] << "/" << preAvail[i*nTypeI +j] << ", " << preStr[i*nTypeI + j] << "]";
    		        if (j==nTypeI-1) {
    		            cout << "\n";
    		        }
    		    }
    		}

			// in min
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=ib*nI; j<(ib+1)*nI; j++) {
    		        for (PosInt k=0; k<nTypeI; k++) {
    		            if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
							if (preConn[i*nTypeI + k] > preTypeGapped[i*mI + j]) {
								preConn[i*nTypeI + k] = preTypeGapped[i*mI + j];
							}
							if (preAvail[i*nTypeI + k] > preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j]) {
    		                	preAvail[i*nTypeI + k] = preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
							}
							if (preStr[i*nTypeI + k] > preTypeStrGapped[i*mI + j]) {
    		                	preStr[i*nTypeI + k] = preTypeStrGapped[i*mI + j];
							}
    		                break;
    		            }
    		        }
    		    }
    		}
			cout << "min Type:    ";
    		for (PosInt i = 0; i < nTypeI; i++) {
    		    cout << i << ",    ";
    		}
    		cout << "\n";
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=0; j<nTypeI; j++) {
    		        if (j==0) {
    		            cout << i << ": ";
    		        }
    		        cout << "[" << preConn[i*nTypeI +j] << "/" << preAvail[i*nTypeI +j] << ", " << preStr[i*nTypeI + j] << "]";
    		        if (j==nTypeI-1) {
    		            cout << "\n";
    		        }
    		    }
    		}
    		delete [] preConn;
    		delete [] preAvail;
    		delete [] preStr;
		}
	}
	//===========*/
	// initialize neighborMat (-1, no connections, 0<=i<nn) for pair-checking
	vector<int> neighborMat(nblock*nblock, -1);
	for (PosInt i=0; i<nblock; i++) {
		cout << "block " << i << ":\n";
		Size *blockId = neighborBlockId + i*maxNeighborBlock;
		Size nn = nNearNeighborBlock[i];
		assert(blockId[0] == i);
		for (PosInt j=0; j<nn; j++) {
			neighborMat[blockId[j]*nblock + i] = j;
			cout << blockId[j]  << ", ";
		}
		cout << "\n";
	}
	for (PosInt i=0; i<nblock; i++) {
		assert(neighborMat[i*nblock + i] == 0);
		for (PosInt j=0; j<i; j++) {
			if ((neighborMat[i*nblock + j] >=0 && neighborMat[j*nblock + i] < 0) || (neighborMat[i*nblock + j] < 0 && neighborMat[j*nblock + i] >= 0)) {
				cout << "neighborMat is not symmetric\n";
				return EXIT_FAILURE;
			}
		}
	}
	int* d_neighborMat;
    checkCudaErrors(cudaMalloc((void**)&d_neighborMat, nblock*nblock*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_neighborMat, &(neighborMat[0]), nblock*nblock*sizeof(int), cudaMemcpyHostToDevice));
	Float* v_outstanding;
    checkCudaErrors(cudaMalloc((void**)&v_outstanding, nearNeighborBlock*nI*nI*sizeof(Float)));
	PosInt* i_outstanding;
    checkCudaErrors(cudaMalloc((void**)&i_outstanding, nearNeighborBlock*nI*nI*sizeof(PosInt)));
	
	size_t gap_neighborMatSize = static_cast<size_t>(nearNeighborBlock*nI)*nI;
	for (PosInt i=0; i<nblock; i++) {
	//for (PosInt i=0; i<1; i++) {
		Size nc = 0; 
		// gather pairs in ith block
		vector<PosInt> clusterID;
		for (PosInt j=0; j<nblock; j++) {
			if (neighborMat[j*nblock + i] >= 0) {
				clusterID.push_back(neighborMat[j*nblock + i]);
				nc++;
			}
		}

		Size *blockId = neighborBlockId + i*maxNeighborBlock;
		//cout << "cluster " << i << "\n";
		assert(clusterID[0] == 0);
		assert(blockId[clusterID[0]] == i);

		if (nc > 0) {
			PosInt *d_clusterID;
    		checkCudaErrors(cudaMalloc((void**)&d_clusterID, nc*sizeof(PosInt)));
			checkCudaErrors(cudaMemcpy(d_clusterID, &(clusterID[0]), nc*sizeof(PosInt), cudaMemcpyHostToDevice));

			checkCudaErrors(cudaMemcpy(d_neighborMat, &(neighborMat[0]), nblock*nblock*sizeof(int), cudaMemcpyHostToDevice));

			/*{ // m
				PosInt i0 = 0;
				size_t gap_nNS = nearNeighborBlock*nI*nI;
				vector<int> neighborMat0(nblock*nblock, -1);
				for (PosInt ii=0; ii<nblock; ii++) {
					PosInt *blockId0 = neighborBlockId + ii*maxNeighborBlock;
					Size nn = nNearNeighborBlock[ii];
					assert(blockId0[0] == ii);
					for (PosInt j=0; j<nn; j++) {
						neighborMat0[blockId0[j]*nblock + ii] = j;
					}
				}
				for (PosInt ii=0; ii<nI; ii++) {
					// postsynaptic
					for (PosInt in=0; in<nNearNeighborBlock[i0]; in++) {
						PosInt bid = neighborBlockId[i0*maxNeighborBlock + in];
						for (PosInt j=0; j<nI; j++) {
							Float home_v = gapMat[i0*gap_nNS + in*nI*nI + j*nI + ii];
							Float guest_v = gapMat[bid*gap_nNS + neighborMat0[i0*nblock+bid]*nI*nI + ii*nI + j];
							if (i0 == 0 && ii == 0 && bid == 2) {
								cout << i << " m-" <<j << ": " << home_v << ", " << guest_v << "(" << i0*gap_nNS + in*nI*nI + j*nI + ii << ", " << bid*gap_nNS + neighborMat0[i0*nblock+bid]*nI*nI + ii*nI + j << ")\n";
							}
						}
					}
				}
			}*/

			// make a mem cluster and send to gpu
			Float* d_clusterGapMat;
    		checkCudaErrors(cudaMalloc((void**)&d_clusterGapMat, gap_neighborMatSize*nc*sizeof(Float)));
			//cout << "uploading " << i << "\n";
			for (PosInt j=0; j<nc; j++) {
				checkCudaErrors(cudaMemcpy(d_clusterGapMat + j*gap_neighborMatSize, gapMat + blockId[clusterID[j]]*gap_neighborMatSize, gap_neighborMatSize*sizeof(Float), cudaMemcpyHostToDevice));
				//cout << j << ":" << gapMat + blockId[clusterID[j]]*gap_neighborMatSize << "to" << d_clusterGapMat + j*gap_neighborMatSize << "\n";
			}
			checkCudaErrors(cudaDeviceSynchronize());

			// make symmteric, use original neighborMat
			//cout << "<<<" << nc << " x " << nI << ">>>\n";
			generate_symmetry<<<nc,nI>>>(
					d_clusterID,
					d_neighborBlockId,
					d_neighborMat,
					d_clusterGapMat,
					d_preTypeGapped, d_preTypeStrGapped, d_preType, state,
					i_outstanding, v_outstanding,
					i, nblock, nearNeighborBlock, maxNeighborBlock, mI, nE, nI, nTypeE, nTypeI);
			// unwind the cluster back to cpu
			checkCudaErrors(cudaDeviceSynchronize());
			//cout << "downloading " << i << "\n";
			for (PosInt j=0; j<nc; j++) {
				checkCudaErrors(cudaMemcpy(gapMat + blockId[clusterID[j]]*gap_neighborMatSize, d_clusterGapMat + j*gap_neighborMatSize, gap_neighborMatSize*sizeof(Float), cudaMemcpyDeviceToHost));
				//cout << j << ":" << d_clusterGapMat + j*gap_neighborMatSize << "to" <<  gapMat + blockId[clusterID[j]]*gap_neighborMatSize << "\n";

				/*{ // u
					PosInt i0 = 0;
					size_t gap_nNS = nearNeighborBlock*nI*nI;
					vector<int> neighborMat0(nblock*nblock, -1);
					for (PosInt ii=0; ii<nblock; ii++) {
						PosInt *blockId0 = neighborBlockId + ii*maxNeighborBlock;
						Size nn = nNearNeighborBlock[ii];
						assert(blockId0[0] == ii);
						for (PosInt jj=0; jj<nn; jj++) {
							neighborMat0[blockId0[jj]*nblock + ii] = jj;
						}
					}
					for (PosInt ii=0; ii<nI; ii++) {
						// postsynaptic
						for (PosInt in=0; in<nNearNeighborBlock[i0]; in++) {
							PosInt bid = neighborBlockId[i0*maxNeighborBlock + in];
							bool pass = true;
							for (PosInt jj=0; jj<nI; jj++) {
								PosInt h_id = i0*gap_nNS + in*nI*nI + jj*nI + ii;
								PosInt g_id = bid*gap_nNS + neighborMat0[i0*nblock+bid]*nI*nI + ii*nI + jj;
								Float home_v = gapMat[h_id];
								Float guest_v = gapMat[g_id];
								if (i0 == 0 && ii == 0 && bid == 2) {
									cout << blockId[clusterID[j]] << "~" << i << " u-" <<jj << ": " << home_v << ", " << guest_v << "(" << h_id << ", " << g_id << ")\n";
									if ((home_v > 0  && guest_v <= 0) || (home_v <= 0  && guest_v > 0)) {
										pass = false;
									}
								//} else {
								}
							}
							if (!pass && !(i==0 && j<2)) {
								cout << i << "-" << blockId[clusterID[j]] << "destroyed gMat\n";
								return EXIT_FAILURE;
							}
						}
					}
				}*/
			}
			checkCudaErrors(cudaDeviceSynchronize());
    		checkCudaErrors(cudaFree(d_clusterGapMat));
    		checkCudaErrors(cudaFree(d_clusterID));

			//cout << "pair " << i << ": ";
			for (PosInt j=0; j<nc; j++) {
				neighborMat[i*nblock + blockId[clusterID[j]]] = -1;
				neighborMat[blockId[clusterID[j]]*nblock + i] = -1;
				//cout << blockId[clusterID[j]] << ", ";
			}
			//cout << "\n";
			clusterID.clear();
		} else {
			cout << "no more to change\n";
		}
		// update the counterparts in the pairs
	}
	checkCudaErrors(cudaMemcpy(preTypeGapped, d_preTypeGapped, gap_statSize, cudaMemcpyDeviceToHost)); 	
    checkCudaErrors(cudaFree(i_outstanding));
    checkCudaErrors(cudaFree(v_outstanding));

	/*===========
	for	(PosInt ib=0; ib<nblock; ib++) {
		cout << "after block#" << ib << " gap stats in  mean: \n";
    	Size* preConn = new Size[nTypeI*nTypeI];
    	Size* preAvail = new Size[nTypeI*nTypeI];
    	Float* preStr = new Float[nTypeI*nTypeI];
    	Size* nnTypeI = new Size[nTypeI];
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=0; j<nTypeI; j++) {
    	        preConn[i*nTypeI + j] = 0;
    	        preAvail[i*nTypeI + j] = 0;
    	        preStr[i*nTypeI + j] = 0.0;
    	    }
    	    nnTypeI[i] = 0; 
    	}
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=ib*nI; j<(ib+1)*nI; j++) {
    	        for (PosInt k=0; k<nTypeI; k++) {
    	            if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
    	                preConn[i*nTypeI + k] += preTypeGapped[i*mI + j];
    	                preAvail[i*nTypeI + k] += preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
    	                preStr[i*nTypeI + k] += preTypeStrGapped[i*mI + j];
    	                if (i == 0) {
    	                    nnTypeI[k]++;
    	                }
    	                break;
    	            }
    	        }
    	    }
    	}

    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=0; j<nTypeI; j++) {
    	        preConn[i*nTypeI + j] /= nnTypeI[j];
    	        preAvail[i*nTypeI + j] /= nnTypeI[j];
    	        preStr[i*nTypeI + j] /= nnTypeI[j];
    	    }
    	    cout << "type " << i << " has " << nnTypeI[i] << " neurons\n";
    	}
    	delete [] nnTypeI;

    	cout << "mean Type:    ";
    	for (PosInt i = 0; i < nTypeI; i++) {
    	    cout << i << ",    ";
    	}
    	cout << "\n";
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=0; j<nTypeI; j++) {
    	        if (j==0) {
    	            cout << i << ": ";
    	        }
    	        cout << "[" << preConn[i*nTypeI +j] << "/" << preAvail[i*nTypeI +j] << ", " << preStr[i*nTypeI + j] << "]";
    	        if (j==nTypeI-1) {
    	            cout << "\n";
    	        }
    	    }
    	}

		// in max
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=0; j<nTypeI; j++) {
    	        preConn[i*nTypeI + j] = 0;
    	        preAvail[i*nTypeI + j] = 0;
    	        preStr[i*nTypeI + j] = 0;
    	    }
    	}
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=ib*nI; j<(ib+1)*nI; j++) {
    	        for (PosInt k=0; k<nTypeI; k++) {
    	            if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
						if (preConn[i*nTypeI + k] < preTypeGapped[i*mI + j]) {
							preConn[i*nTypeI + k] = preTypeGapped[i*mI + j];
						}
						if (preAvail[i*nTypeI + k] < preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j]) {
    	                	preAvail[i*nTypeI + k] = preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
						}
						if (preStr[i*nTypeI + k] < preTypeStrGapped[i*mI + j]) {
    	                	preStr[i*nTypeI + k] = preTypeStrGapped[i*mI + j];
						}
    	                break;
    	            }
    	        }
    	    }
    	}
		cout << "max Type:    ";
    	for (PosInt i = 0; i < nTypeI; i++) {
    	    cout << i << ",    ";
    	}
    	cout << "\n";
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=0; j<nTypeI; j++) {
    	        if (j==0) {
    	            cout << i << ": ";
    	        }
    	        cout << "[" << preConn[i*nTypeI +j] << "/" << preAvail[i*nTypeI +j] << ", " << preStr[i*nTypeI + j] << "]";
    	        if (j==nTypeI-1) {
    	            cout << "\n";
    	        }
    	    }
    	}

		// in min
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=ib*nI; j<(ib+1)*nI; j++) {
    	        for (PosInt k=0; k<nTypeI; k++) {
    	            if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
						if (preConn[i*nTypeI + k] > preTypeGapped[i*mI + j]) {
							preConn[i*nTypeI + k] = preTypeGapped[i*mI + j];
						}
						if (preAvail[i*nTypeI + k] > preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j]) {
    	                	preAvail[i*nTypeI + k] = preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
						}
						if (preStr[i*nTypeI + k] > preTypeStrGapped[i*mI + j]) {
    	                	preStr[i*nTypeI + k] = preTypeStrGapped[i*mI + j];
						}
    	                break;
    	            }
    	        }
    	    }
    	}
		cout << "min Type:    ";
    	for (PosInt i = 0; i < nTypeI; i++) {
    	    cout << i << ",    ";
    	}
    	cout << "\n";
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=0; j<nTypeI; j++) {
    	        if (j==0) {
    	            cout << i << ": ";
    	        }
    	        cout << "[" << preConn[i*nTypeI +j] << "/" << preAvail[i*nTypeI +j] << ", " << preStr[i*nTypeI + j] << "]";
    	        if (j==nTypeI-1) {
    	            cout << "\n";
    	        }
    	    }
    	}
    	delete [] preConn;
    	delete [] preAvail;
    	delete [] preStr;
	}
	//===============*/

	cout << "gap mat ready\n";
	// make reciprocal gap junctions in vectors
	//cout << "mI = " << mI << "\n";
	for (PosInt i=0; i<mI; i++) {
		PosInt itype;
		for (PosInt k=0; k<nTypeI; k++) {
			if (i%nI < typeAccCount[k+nTypeE] - nE) itype = k;
		}
		PosInt host_tid = i/nI*blockSize + nE + i%nI;
		for (PosInt j=0; j<nGapVec[i]; j++) {
			PosInt id = gapVecID[i*gap_maxDistantNeighbor + j];
			PosInt guest_id = id/blockSize * nI + id%blockSize-nE;
			assert(guest_id < mI);
			bool gapped = false;
			for (PosInt k=0; k<nGapVec[guest_id]; k++) {
				if (gapVecID[guest_id*gap_maxDistantNeighbor + k] == i) {
					gapped = true;
					break;
				}
			}
			if (!gapped) {
				PosInt guest_itype;
				for (PosInt k=0; k<nTypeI; k++) {
					if (guest_id%nI < typeAccCount[k+nTypeE] - nE) guest_itype = k;
				}
				if (nGapVec[i] < gap_maxDistantNeighbor && preTypeGapped[itype*mI + guest_id] < nInhGap[itype*nTypeI + guest_itype]*(1+minConTol)) {
					gapVecID[guest_id*gap_maxDistantNeighbor + nGapVec[guest_id]] = host_tid;
					gapVec[guest_id*gap_maxDistantNeighbor + nGapVec[guest_id]] = gapVec[i*gap_maxDistantNeighbor + j];
					gapDelayVec[guest_id*gap_maxDistantNeighbor + nGapVec[guest_id]] = gapDelayVec[i*gap_maxDistantNeighbor + j];
					nGapVec[guest_id]++;
					preTypeGapped[itype*mI + guest_id]++;
					preTypeStrGapped[itype*mI + guest_id] += gapVec[i*gap_maxDistantNeighbor + j];
					//cout << "gapVec[ " << guest_id << "] added 1 gap from  " << gapVecID[guest_id*gap_maxDistantNeighbor + nGapVec[guest_id]-1]  << "\n";
				} else {
					preTypeGapped[itype*mI + guest_id]--;
					preTypeStrGapped[itype*mI + guest_id] -= gapVec[i*gap_maxDistantNeighbor + j];
					//if (j < nGapVec[i]-1) { // advance the array elements by 1
						PosInt i0 = i*gap_maxDistantNeighbor ;
						PosInt i1 = i0 + nGapVec[i];
						i0 += j+1;
					//cout << "gapVec[ " << i << "] removed 1 gap from  " << gapVecID[i0-1]  << "\n";
						vector<Float> tmpS(gapVec+i0, gapVec+i1);
						vector<Float> tmpD(gapDelayVec+i0, gapDelayVec+i1);
						vector<PosInt> tmpID(gapVecID+i0, gapVecID+i1);
						for (PosInt k = 0; k<tmpS.size(); k++) {
							gapVec[i0-1 + k] = tmpS[k];
							gapDelayVec[i0-1 + k] = tmpD[k];
							gapVecID[i0-1 + k] = tmpID[k];
						}
					//}
					nGapVec[i]--;
					j--;
				}
			}
		}
	}
	for (PosInt i=0; i<mI; i++) {
		if (nGapVec[i]) {
			cout << i << "th inh gaps:\n";
			for (PosInt j=0; j<nGapVec[i]; j++) {
				cout << gapVecID[i*gap_maxDistantNeighbor+j] << ", ";
			}
			cout << "\n";
			cout << i << "th inh gap dis:\n";
			for (PosInt j=0; j<nGapVec[i]; j++) {
				cout << gapDelayVec[i*gap_maxDistantNeighbor+j] << ", ";
			}
			cout << "\n";
		}
	}
	cout << "gap vec ready\n";

	if (strictStrength) {
		cout << "apply strict strength\n";		
		vector<int> neighborMat0(nblock*nblock, -1);
		for (PosInt i=0; i<nblock; i++) {
			PosInt *blockId = neighborBlockId + i*maxNeighborBlock;
			Size nn = nNearNeighborBlock[i];
			assert(blockId[0] == i);
			for (PosInt j=0; j<nn; j++) {
				neighborMat0[blockId[j]*nblock + i] = j;
			}
		}
		cout << "built neighborMat\n";		
		size_t gap_nNS = nearNeighborBlock*nI*nI;
		for (PosInt ib=0; ib<nblock; ib++) {
			for (PosInt i=0; i<nI; i++) {
				// postsynaptic
				PosInt itype;
				for (PosInt j=0; j<nTypeI; j++) {
					if (i < typeAccCount[j + nTypeE] - nE) {
						itype = j;
						break;
					}
				}
				vector<Float> ratio;
				for (PosInt j=0; j<nTypeI; j++) {
					ratio.push_back(gap_sTypeMat[j*nTypeI+ itype]*nInhGap[j*nTypeI+ itype]/preTypeStrGapped[j*mI + ib*nI+i]);
					assert(ratio[j] > 0);
				}
				for (PosInt in=0; in<nNearNeighborBlock[ib]; in++) {
					PosInt bid = neighborBlockId[ib*maxNeighborBlock + in];
					bool pass = true;
					for (PosInt j=0; j<nI; j++) {
						Float home_v = gapMat[ib*gap_nNS + in*nI*nI + j*nI + i];
						Float guest_v = gapMat[bid*gap_nNS + neighborMat0[ib*nblock+bid]*nI*nI + i*nI + j];
						if (home_v > 0) {
							for (PosInt k=0; k<nTypeI; k++) {
								if (j < typeAccCount[k + nTypeE] - nE) {
									gapMat[ib*gap_nNS + in*nI*nI + j*nI + i] *= ratio[k];
									break;
								}
							}
							if (guest_v <= 0) {
								cout << ib << ", " << bid << "(" << in << "): " << j << "<->" << i <<" = " << home_v << " ~ " << guest_v << "\n";
								pass = false;
								//assert(guest_v > 0);
							}
						} else {
							if (guest_v > 0) {
								
								cout << ib << ", " << bid << "(" << in << "): " << j << "<->" << i <<" = " << home_v << " ~ " << guest_v << "\n";
								pass = false;
								//assert(guest_v <= 0);
							}
						}
					}
					if (!pass) {
						assert(pass);
					}
				}
				for (PosInt j=0; j<nGapVec[i]; j++) {
					PosInt ktype;
					for (PosInt k=0; k<nTypeI; k++) {
						if (gapVecID[i*gap_maxDistantNeighbor + j]%blockSize < typeAccCount[j + nTypeE]) {
							ktype = j;
							break;
						}
					}
					gapVec[i*gap_maxDistantNeighbor + j] *= ratio[ktype];
				}
			}
		}
	}

    fV1_gapMat.write((char*)gapMat, static_cast<size_t>(nearNeighborBlock*nI)*nI*nblock*sizeof(Float));
    fV1_gapMat.close();
	delete []gapMat;

    printf("connectivity constructed\n");
	//checkCudaErrors(cudaStreamDestroy(s0));
    //checkCudaErrors(cudaStreamDestroy(s1));
    //checkCudaErrors(cudaStreamDestroy(s2));

    fBlock_pos.write((char*)block_x, nblock*sizeof(Float));
    fBlock_pos.write((char*)block_y, nblock*sizeof(Float));
    fBlock_pos.close();

    cout << "number of neighbors (self-included): ";
    for (Size i=0; i<nblock; i++) {
        cout << nNeighborBlock[i] << " (" << nNearNeighborBlock[i] << "), ";
    }
    cout << "\n";
    fNeighborBlock.write((char*)nNearNeighborBlock, nblock*sizeof(Size));
    fNeighborBlock.write((char*)nNeighborBlock, nblock*sizeof(Size));
    cout << "neighbors blockId: \n";
    for (Size i=0; i<nblock; i++) {
        fNeighborBlock.write((char*)&neighborBlockId[i*maxNeighborBlock], nNeighborBlock[i]*sizeof(PosInt));
        cout << i << ": ";
        for (PosInt j=0; j<nNeighborBlock[i]; j++) {
            cout << neighborBlockId[i*maxNeighborBlock + j] << ",";
        }
        cout << "\n";
    }
    fNeighborBlock.close();
	cout << "neighbors written\n";

    fV1_vec.write((char*)nVec, networkSize*sizeof(Size));
    for (Size i=0; i<networkSize; i++) {
        fV1_vec.write((char*)&(vecID[i*maxDistantNeighbor]), nVec[i]*sizeof(Size));
        fV1_vec.write((char*)&(conVec[i*maxDistantNeighbor]), nVec[i]*sizeof(Float));
        fV1_vec.write((char*)&(delayVec[i*maxDistantNeighbor]), nVec[i]*sizeof(Float));
    }
    fV1_vec.close();
	cout << "conVec written\n";

    fV1_gapVec.write((char*)nGapVec, mI*sizeof(Size));
    for (Size i=0; i<mI; i++) {
        fV1_gapVec.write((char*)&(gapVecID[i*gap_maxDistantNeighbor]), nGapVec[i]*sizeof(PosInt));
        fV1_gapVec.write((char*)&(gapVec[i*gap_maxDistantNeighbor]), nGapVec[i]*sizeof(Float));
        fV1_gapVec.write((char*)&(gapDelayVec[i*gap_maxDistantNeighbor]), nGapVec[i]*sizeof(Float));
    }
    fV1_gapVec.close();
	cout << "gapVec written\n";

    fStats.write((char*)&nType,sizeof(Size));
    fStats.write((char*)&networkSize,sizeof(Size));
    fStats.write((char*)ExcRatio, networkSize*sizeof(Float));
    fStats.write((char*)preTypeConnected, nType*networkSize*sizeof(Size));
    fStats.write((char*)preTypeAvail, nType*networkSize*sizeof(Size));
    fStats.write((char*)preTypeStrSum, nType*networkSize*sizeof(Float));
    fStats.write((char*)&nTypeI,sizeof(Size));
    fStats.write((char*)&mI,sizeof(Size));
    fStats.write((char*)preTypeGapped, nTypeI*mI*sizeof(Size));
    fStats.write((char*)preTypeStrGapped, nTypeI*mI*sizeof(Float));
    fStats.close();

	for (PosInt i=0; i<nType; i++) {
		cout << "cortical excitation percentage of total, including extra laminar cortical input: " << extExcRatio[i] << "\n";
		Float presetCorticalRatio = hInit_pack.nTypeMat[i]*synPerCon[i] / presetConstExcSyn[i];
		Float init = 0.0;
		cout << "[" << *min_element(ExcRatio, ExcRatio+networkSize) * presetCorticalRatio << ", " << accumulate(ExcRatio, ExcRatio+networkSize, init)/networkSize * presetCorticalRatio << ", " << *max_element(ExcRatio, ExcRatio+networkSize) * presetCorticalRatio << "]\n";
	}
	delete []ExcRatio;
	delete []presetConstExcSyn;
	hInit_pack.freeMem();

    cout << "connection stats in  mean: \n";
    //Size *preConn = new Size[nType*nType];
    //Size *preAvail = new Size[nType*nType];
    //Float *preStr = new Float[nType*nType];
    Size* preConn = new Size[nType*nType];
    Size* preAvail = new Size[nType*nType];
    Float* preStr = new Float[nType*nType];
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
    delete [] preStr;
	delete [] preAvail;


	cout << "gap stats in  mean: \n";
    preConn = new Size[nTypeI*nTypeI];
    preAvail = new Size[nTypeI*nTypeI];
    preStr = new Float[nTypeI*nTypeI];
    //Size *nnTypeI = new Size[nTypeI];
    Size* nnTypeI = new Size[nTypeI];
    for (PosInt i=0; i<nTypeI; i++) {
        for (PosInt j=0; j<nTypeI; j++) {
            preConn[i*nTypeI + j] = 0;
            preAvail[i*nTypeI + j] = 0;
            preStr[i*nTypeI + j] = 0.0;
        }
        nnTypeI[i] = 0; 
    }
    for (PosInt i=0; i<nTypeI; i++) {
        for (PosInt j=0; j<mI; j++) {
            for (PosInt k=0; k<nTypeI; k++) {
                if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
                    preConn[i*nTypeI + k] += preTypeGapped[i*mI + j];
                    preAvail[i*nTypeI + k] += preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
                    preStr[i*nTypeI + k] += preTypeStrGapped[i*mI + j];
                    if (i == 0) {
                        nnTypeI[k]++;
                    }
                    break;
                }
            }
        }
    }

    for (PosInt i=0; i<nTypeI; i++) {
        for (PosInt j=0; j<nTypeI; j++) {
            preConn[i*nTypeI + j] /= nnTypeI[j];
            preAvail[i*nTypeI + j] /= nnTypeI[j];
            preStr[i*nTypeI + j] /= nnTypeI[j];
        }
        cout << "type " << i << " has " << nnTypeI[i] << " neurons\n";
    }
    delete [] nnTypeI;

    cout << "mean Type:    ";
    for (PosInt i = 0; i < nTypeI; i++) {
        cout << i << ",    ";
    }
    cout << "\n";
    for (PosInt i=0; i<nTypeI; i++) {
        for (PosInt j=0; j<nTypeI; j++) {
            if (j==0) {
                cout << i << ": ";
            }
            cout << "[" << preConn[i*nTypeI +j] << "/" << preAvail[i*nTypeI +j] << ", " << preStr[i*nTypeI + j] << "]";
            if (j==nTypeI-1) {
                cout << "\n";
            }
        }
    }

	// in max
    for (PosInt i=0; i<nTypeI; i++) {
        for (PosInt j=0; j<nTypeI; j++) {
            preConn[i*nTypeI + j] = 0;
            preAvail[i*nTypeI + j] = 0;
            preStr[i*nTypeI + j] = 0;
        }
    }
    for (PosInt i=0; i<nTypeI; i++) {
        for (PosInt j=0; j<mI; j++) {
            for (PosInt k=0; k<nTypeI; k++) {
                if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
					if (preConn[i*nTypeI + k] < preTypeGapped[i*mI + j]) {
						preConn[i*nTypeI + k] = preTypeGapped[i*mI + j];
					}
					if (preAvail[i*nTypeI + k] < preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j]) {
                    	preAvail[i*nTypeI + k] = preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
					}
					if (preStr[i*nTypeI + k] < preTypeStrGapped[i*mI + j]) {
                    	preStr[i*nTypeI + k] = preTypeStrGapped[i*mI + j];
					}
                    break;
                }
            }
        }
    }
	cout << "max Type:    ";
    for (PosInt i = 0; i < nTypeI; i++) {
        cout << i << ",    ";
    }
    cout << "\n";
    for (PosInt i=0; i<nTypeI; i++) {
        for (PosInt j=0; j<nTypeI; j++) {
            if (j==0) {
                cout << i << ": ";
            }
            cout << "[" << preConn[i*nTypeI +j] << "/" << preAvail[i*nTypeI +j] << ", " << preStr[i*nTypeI + j] << "]";
            if (j==nTypeI-1) {
                cout << "\n";
            }
        }
    }

	// in min
    for (PosInt i=0; i<nTypeI; i++) {
        for (PosInt j=0; j<mI; j++) {
            for (PosInt k=0; k<nTypeI; k++) {
                if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
					if (preConn[i*nTypeI + k] > preTypeGapped[i*mI + j]) {
						preConn[i*nTypeI + k] = preTypeGapped[i*mI + j];
					}
					if (preAvail[i*nTypeI + k] > preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j]) {
                    	preAvail[i*nTypeI + k] = preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
					}
					if (preStr[i*nTypeI + k] > preTypeStrGapped[i*mI + j]) {
                    	preStr[i*nTypeI + k] = preTypeStrGapped[i*mI + j];
					}
                    break;
                }
            }
        }
    }
	cout << "min Type:    ";
    for (PosInt i = 0; i < nTypeI; i++) {
        cout << i << ",    ";
    }
    cout << "\n";
    for (PosInt i=0; i<nTypeI; i++) {
        for (PosInt j=0; j<nTypeI; j++) {
            if (j==0) {
                cout << i << ": ";
            }
            cout << "[" << preConn[i*nTypeI +j] << "/" << preAvail[i*nTypeI +j] << ", " << preStr[i*nTypeI + j] << "]";
            if (j==nTypeI-1) {
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
		fConnectome_cfg.write((char*) &nTypeI, sizeof(Size));
		fConnectome_cfg.write((char*) (&typeAccCount[0]), nType*sizeof(Size));
		fConnectome_cfg.write((char*) (&synPerCon[0]), nType*nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&synPerConFF[0]), nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&nTypeMat[0]), nType*nType*sizeof(Size));
		fConnectome_cfg.write((char*) (&sTypeMat[0]), nType*nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&nInhGap[0]), nTypeI*nTypeI*sizeof(Size));
		fConnectome_cfg.write((char*) (&gap_sTypeMat[0]), nTypeI*nTypeI*sizeof(Size));
		fConnectome_cfg.write((char*) (&gap_fTypeMat[0]), nTypeI*nTypeI*sizeof(Size));
		fConnectome_cfg.write((char*) (&rDend[0]), nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&rAxon[0]), nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&dDend[0]), nType*sizeof(Float));
		fConnectome_cfg.write((char*) (&dAxon[0]), nType*sizeof(Float));
		fConnectome_cfg.close();
	}

    
	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(gpu_chunk));
    checkCudaErrors(cudaFree(d_gapMat));
    checkCudaErrors(cudaFree(d_typeAcc0));
    checkCudaErrors(cudaFree(d_synloc));
    //checkCudaErrors(cudaFree(d_LGN_V1_sSum));
    checkCudaErrors(cudaFree(d_nLGN_V1));
    checkCudaErrors(cudaFree(d_ExcRatio));
    checkCudaErrors(cudaFree(d_max_N));
    checkCudaErrors(cudaFree(gap_preF_type));
    checkCudaErrors(cudaFree(gap_preS_type));
    checkCudaErrors(cudaFree(gap_preN_type));
    checkCudaErrors(cudaFree(d_extExcRatio));
    checkCudaErrors(cudaFree(d_synPerConFF));
    checkCudaErrors(cudaFree(d_synPerCon));
    checkCudaErrors(cudaFree(d_neighborMat));
	free(cpu_chunk);
    return 0;
}
