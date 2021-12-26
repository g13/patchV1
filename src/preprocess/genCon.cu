#include "genCon.h"
// TODO: check preType, V1_type.bin

int main(int argc, char *argv[])
{
	namespace po = boost::program_options;
	namespace bf = boost::filesystem;
    using namespace std;
	using std::string;

    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
    printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);
    printf("total global memory: %f Mb.\n", deviceProps.totalGlobalMem/1024.0/1024.0);

	BigSize seed;
    vector<Size> maxNeighborBlock;
    vector<Size> nearNeighborBlock;
	vector<Size> maxDistantNeighbor;
    vector<Size> gap_maxDistantNeighbor;
	vector<Size> nonInputTypeEI;
    string V1_type_filename, V1_feature_filename, V1_allpos_filename, LGN_V1_s_filename, suffix, conLGN_suffix, LGN_V1_cfg_filename, output_cfg_filename;
	string V1_conMat_filename, V1_delayMat_filename, V1_gapMat_filename;
	string V1_vec_filename, V1_gapVec_filename;
	string block_pos_filename, neighborBlock_filename, stats_filename;
    Float dScale;
    vector<Float> blockROI_max, blockROI;
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
	Float longRangeROI;
	Float longRangeCorr;
	vector<Float> disGauss;
	vector<Float> rDend, rAxon;
	vector<Float> dDend, dAxon;
    vector<Float> synapticLoc;
    vector<Float> sTypeMat, gap_sTypeMatE, gap_sTypeMatI;
    vector<Float> typeFeatureMat, gap_fTypeMatE, gap_fTypeMatI;
    vector<Size> nTypeMat, gap_nTypeMatE, gap_nTypeMatI;
    vector<PosInt> nonInputLayer;
    vector<PosInt> nonInputTypeAccCount;

	po::options_description generic_opt("Generic options");
	generic_opt.add_options()
        ("seed", po::value<BigSize>(&seed)->default_value(7641807), "seed for RNG")
		("cfg_file,c", po::value<string>()->default_value("connectome.cfg"), "filename for configuration file")
		("help,h", "print usage");

	Float minConTol;
	po::options_description input_opt("output options");
	input_opt.add_options()
		("inputFolder", po::value<string>(&inputFolder)->default_value(""), "where the input data files at, must end with /")
		("outputFolder", po::value<string>(&outputFolder)->default_value(""), "where the output data files at must end with /")
		("res_suffix", po::value<string>(&res_suffix)->default_value(""), "suffix for resource files")
        ("nonInputLayer", po::value<vector<PosInt>>(&nonInputLayer), "layer id of those cortical layers that do not receive external inputs")
		("nonInputTypeEI", po::value<vector<Size>>(&nonInputTypeEI), "a vector of hierarchical types differs in non-functional properties: reversal potentials, characteristic lengths of dendrite and axons, e.g. in the form of nArchtypeE, nArchtypeI for each nonInput layer (2 x nonInputLayer). {Exc-Pyramidal, Exc-stellate; Inh-PV, Inh-SOM, Inh-LTS} then for that layer the element would be {3, 2}")
        ("nonInputTypeAccCount", po::value<vector<PosInt>>(&nonInputTypeAccCount), "accumulative arrays for number of types for each nonInputLayer")
        ("DisGauss", po::value<vector<Float>>(&disGauss), "if set true, conn. prob. based on distance will follow a 2D gaussian with a variance. of (raxn*raxn + rden*rden)/(2*ln(2))*disGauss, otherwise 0 will based on the overlap of the area specified by raxn and rden")
        ("strictStrength", po::value<bool>(&strictStrength), "strictly match preset summed connection")
        ("CmoreN", po::value<bool>(&CmoreN), "if true complex gets more connections otherwise stronger strength")
        ("rDend", po::value<vector<Float>>(&rDend),  "a vector of dendritic extensions' radius, size of nType x nLayer ")
        ("rAxon", po::value<vector<Float>>(&rAxon),  "a vector of axonic extensions' radius, size of nType x nLayer")
        ("dScale",po::value<Float>(&dScale)->default_value(1.0),"a scaling ratio of all the neurites' lengths <radius>")
        ("longRangeROI", po::value<Float>(&longRangeROI), "ROI of long-range cortical input")
        ("longRangeCorr", po::value<Float>(&longRangeCorr), "correlation between long-range cortical inputs that cortical cells receives")
        ("dDend", po::value<vector<Float>>(&dDend), "vector of dendrites' densities, size of nType x nLayer")
        ("dAxon", po::value<vector<Float>>(&dAxon), "vector of axons' densities, size of nType x nLayer")
        ("synapticLoc", po::value<vector<Float>>(&synapticLoc), " maximal synaptic location relative to the soma, percentage of dendrite, of different presynaptic type, size of [nType, nType], nType = sum(nTypeEI), row_id -> postsynaptic, column_id -> presynaptic")
		//("LGN_targetFR", po::value<Float>(&LGN_targetFR), "target firing rate of a LGN cell")
		//("targetFR", po::value<vector<Float>>(&targetFR), "a vector of target firing rate of different neuronal types")
		//("FF_FB_ratio", po::value<Float>(&FF_FB_ratio), "excitation ratio of FF over total excitation")
		("extExcRatio", po::value<vector<Float>>(&extExcRatio), "external cortical excitation ratio to E and I")
		("extSynRatio", po::value<vector<Float>>(&extSynRatio), "external cortical exc connection #synapse ratio to E and I")
		("min_FB_ratio", po::value<Float>(&min_FB_ratio), "minimum external cortical excitation ratio")
		("C_InhRatio", po::value<Float>(&C_InhRatio), "minimum inhibition for complex cells")
		("ClessI", po::value<bool>(&ClessI), "lesser inhibition for complex cell")
		("minConTol", po::value<Float>(&minConTol), "minimum difference tolerance of the number of preset cortical connections")
        ("sTypeMat", po::value<vector<Float>>(&sTypeMat), "connection strength matrix between neuronal types, size of [nType, nType], nType = sum(nTypeEI), row_id -> postsynaptic, column_id -> presynaptic")
        ("gap_sTypeMatE", po::value<vector<Float>>(&gap_sTypeMatE), "gap junction strength matrix between neuronal types, size of [nTypeI/E, nTypeI/E], within each layer, row_id -> postsynaptic, column_id -> presynaptic")
        ("gap_nTypeMatE", po::value<vector<Size>>(&gap_nTypeMatE), "connection numbers' matrix for gap junctio between exc neuronal types, size of [totalTypeE, totalTypeE], across each layer, row_id -> postsynaptic, column_id -> presynaptic")
        ("gap_sTypeMatI", po::value<vector<Float>>(&gap_sTypeMatI), "gap junction strength matrix between neuronal types, size of [nTypeI/E, nTypeI/E], within each layer, row_id -> postsynaptic, column_id -> presynaptic")
        ("gap_nTypeMatI", po::value<vector<Size>>(&gap_nTypeMatI), "connection numbers' matrix for gap junctio between inh neuronal types, size of [totalTypeI, totalTypeI], across each layer, row_id -> postsynaptic, column_id -> presynaptic")
        ("nTypeMat", po::value<vector<Size>>(&nTypeMat), "#connection matrix between neuronal types, size of [nType, nType], nType = sum(nTypeEI), row_id -> postsynaptic, column_id -> presynaptic")
        ("typeFeatureMat", po::value<vector<Float>>(&typeFeatureMat), "feature-specific matrixi for parameters of connection preference over neuronal types, size of [nFeature, totalType, totalType], row: postsynaptic-id, column: presynaptic-id")
        ("gap_fTypeMat", po::value<vector<Float>>(&gap_fTypeMat), "feature parameter of neuronal types, size of [nFeature, nTypeI, nTypeI], nTypeI = nTypeEI[1], row_id -> postsynaptic, column_id -> presynaptic")
        ("blockROI", po::value<vector<Float>>(&blockROI), "garaunteed radius (center to center) to include neighboring blocks in mm")
        ("blockROI_max", po::value<vector<Float>>(&blockROI_max), "max radius (center to center) to include neighboring blocks in mm")
        ("maxDistantNeighbor", po::value<vector<Size>>(&maxDistantNeighbor), "the preserved size of the array that store the presynaptic neurons' ID, who are not in the neighboring blocks")
        ("gap_maxDistantNeighbor", po::value<Size>(&gap_maxDistantNeighbor), "the preserved size of the array that store the pre-junction neurons' ID, who are not in the neighboring blocks")
        ("maxNeighborBlock", po::value<vector<Size>>(&maxNeighborBlock), "the preserved size (minus the nearNeighborBlock) of the array that store the neighboring blocks ID that goes into conVec")
        ("nearNeighborBlock", po::value<vector<Size>>(&nearNeighborBlock), "the preserved size of the array that store the neighboring blocks ID that goes into conMat, excluding the self block, self will be added later, nLayer x nLayer")
        ("fV1_feature", po::value<string>(&V1_feature_filename)->default_value("V1_feature.bin"), "file to read spatially predetermined functional features of neurons")
        ("fV1_allpos", po::value<string>(&V1_allpos_filename)->default_value("V1_allpos"), "the directory to read neuron positions")
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
	ifstream cfg_file;
	if (!cfg_filename.empty()) {
        cfg_file.open(cfg_filename.c_str(), fstream::in);
	    if (cfg_file) {
	    	po::store(po::parse_config_file(cfg_file, config_file_options), vm);
	    	cout << "Using configuration file: " << cfg_filename << "\n";
	        po::notify(vm);
            cfg_file.close();
	    } else {
	    	cout << "No configuration file is given, default values are used for non-specified parameters\n";
			return EXIT_FAILURE;
	    }
    } else {
		cout << "No configuration file is given, default values are used for non-specified parameters\n";
	    po::notify(vm);
    }

	if (!vm["inputFolder"].defaulted()) {
		cout << "unspecified input files will be read from " << inputFolder << "\n";
		if (vm["fLGN_V1_cfg"].defaulted()){
			LGN_V1_cfg_filename = inputFolder + LGN_V1_cfg_filename;
		}
		if (vm["fV1_delayMat"].defaulted()){
			V1_delayMat_filename = inputFolder + V1_delayMat_filename;
		}
		if (vm["fV1_gapMat"].defaulted()){
			V1_gapMat_filename = inputFolder + V1_gapMat_filename;
		}
		if (vm["fConStats"].defaulted()){
			conStats_filename = inputFolder + conStats_filename;
		}
		if (vm["fV1_vec"].defaulted()){
			V1_vec_filename = inputFolder + V1_vec_filename;
		}
		if (vm["fV1_gapVec"].defaulted()){
			V1_gapVec_filename = inputFolder + V1_gapVec_filename;
		}
		if (vm["fNeighborBlock"].defaulted()){
			neighborBlock_filename = inputFolder + neighborBlock_filename;
		}
		// not in use
		//if (!vm["fV1_RF"].defaulted(){
		//	V1_RF_filename = inputFolder + V1_RF_filename;
		//}
	} else {
		cout << "inputFolder defaulted to current folder\n";
	}

	if (!vm["outputFolder"].defaulted()) {
		bf::path outputPath(outputFolder);
		if (!bf::is_directory(outputPath)) {
			cout << "creating output folder: " << outputFolder << "\n";
			bf::create_directory(outputPath);
		}
		if (vm["fConnectome_cfg"].defaulted()){
			connectome_cfg_filename = outputFolder + connectome_cfg_filename;
		}
        if (vm["fV1_conMat"].defaulted()){
			V1_conMat_filename = outputFolder + V1_conMat_filename;
		}
		if (vm["fV1_delayMat"].defaulted()){
			V1_delayMat_filename = outputFolder + V1_delayMat_filename;
		}
		if (vm["fV1_gapMat"].defaulted()){
			V1_gapMat_filename = outputFolder + V1_gapMat_filename;
		}
		if (vm["fConStats"].defaulted()){
			conStats_filename = outputFolder + conStats_filename;
		}
		if (vm["fV1_vec"].defaulted()){
			V1_vec_filename = outputFolder + V1_vec_filename;
		}
		if (vm["fV1_gapVec"].defaulted()){
			V1_gapVec_filename = outputFolder + V1_gapVec_filename;
		}
		if (vm["fBlock_pos"].defaulted()){
            block_pos_filename = outputFolder + block_pos_filename;
        }
		if (vm["fNeighborBlock"].defaulted()){
			neighborBlock_filename = outputFolder + neighborBlock_filename;
		}
        if (vm["fStats"].defaulted()) {
            stats_filename = outputFolder + stats_filename;
        }
	}

    ifstream fV1_allpos, fV1_feature;
    ofstream fV1_conMat, fV1_delayMat, fV1_gapMat, fV1_vec, fV1_gapVec;
    ofstream fBlock_pos, fNeighborBlock;
    ofstream fStats;

	if (!conLGN_suffix.empty()) {
        conLGN_suffix = '_' + conLGN_suffix;
    }
    conLGN_suffix = conLGN_suffix + ".bin";

    if (!res_suffix.empty())  {
        res_suffix = "_" + res_suffix;
    }
    res_suffix = res_suffix + ".bin";

    if (!suffix.empty()) {
		cout << " suffix = " << suffix << "\n";
        suffix = '_' + suffix;
    }
    suffix = suffix + ".bin";

    Size nLayer;
    vector<Size> nblock;
    vector<Size> neuronPerBlock; 
    vector<Size> networkSize; 
	vector<Size> mI;
	vector<Size> mE;
	vector<vector<Float>> pos;;
    fV1_allpos.open(V1_allpos_filename + res_suffix, ios::in|ios::binary);
	if (!fV1_allpos) {
		cout << "failed to open pos file:" <<  V1_allpos_filename + res_suffix << "\n";
		return EXIT_FAILURE;
	} else {
        int size_of_precision;
		fV1_allpos.read(reinterpret_cast<char*>(&size_of_precision), sizeof(int));
		fV1_allpos.read(reinterpret_cast<char*>(&nLayer), sizeof(Size));
        pos.resize(nLayer);
	    nblock.resize(nLayer);
        neuronPerBlock.resize(nLayer);
        networkSize.resize(nLayer);
        fV1_allpos.read(reinterpret_cast<char*>(&nblock), nLayer*sizeof(Size));
        fV1_allpos.read(reinterpret_cast<char*>(&neuronPerBlock), nLayer*sizeof(Size));
        for (PosInt iLayer = 0; iLayer < nLayer; iLayer++) {
            networkSize[iLayer] = nblock[iLayer]*neuronPerBlock[iLayer];
            if (neuronPerBlock[iLayer] > blockSize) {
                cout << "neuron per block (" << neuronPerBlock[iLayer] << ") cannot be larger than cuda block size: " << blockSize << "\n";
            }
            cout << networkSize[iLayer] << " neurons for V1 layer " << iLayer << "\n";
            if (size_of_precision == 8) {
                vector<double> _pos(2*networkSize[iLayer]);
	            fV1_allpos.read(reinterpret_cast<char*>(&_pos[0]), 2*networkSize[iLayer] * sizeof(double)); 
                pos[iLayer].assign(_pos.begin(), _pos.end());
	            fV1_allpos.seekg(2*networkSize[iLayer]*sizeof(double), fV1_allpos.cur);  // skip vpos
            } else {
                assert(size_of_precision == 4);
                vector<float> _pos(2*networkSize[iLayer]);
	            fV1_allpos.read(reinterpret_cast<char*>(&_pos[0]), 2*networkSize[iLayer] * sizeof(float)); 
                pos[iLayer].assign(_pos.begin(), _pos.end());
	            fV1_allpos.seekg(2*networkSize[iLayer]*sizeof(float), fV1_allpos.cur);  // skip vpos
            }
        }
    }
	fV1_allpos.close();
	
    Size nInputLayer;
	vector<PosInt> inputLayer;
	vector<Float> p_n_LGNeff;
	vector<Size> max_LGNeff;
	vector<Float> mean_nLGN;
	vector<Float> mean_sLGN;
	vector<Size> nInputType;
	vector<Size> nInputTypeEI;
	vector<Size> inputTypeAccCount;
    Size totalInputType = 0;
	ifstream fLGN_V1_cfg(LGN_V1_cfg_filename + conLGN_suffix, fstream::in | fstream::binary);
	if (!fLGN_V1_cfg) {
		cout << "Cannot open or find " << LGN_V1_cfg_filename + conLGN_suffix <<"\n";
		return EXIT_FAILURE;
	} else {
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&nInputLayer), sizeof(Size));
        inputLayer.resize(nInputLayer);
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&inputLayer), nInputLayer*sizeof(Size));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&nInputTypeEI), 2*nInputLayer*sizeof(Size));
        for (PosInt i=0; i<nInputLayer; i++) {
            nInputType[i] = nInputTypeEI[2*i] + nInputTypeEI[2*i+1];
            totalInputType += nInputType[i];
        }
		inputTypeAccCount.resize(totalInputType);
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&inputTypeAccCount[0]), totalInputType*sizeof(Size));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&p_n_LGNeff[0]), totalInputType*sizeof(Float));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&max_LGNeff[0]), totalInputType*sizeof(Size));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&mean_nLGN[0]), nInputLayer*sizeof(Float));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&mean_sLGN[0]), nInputLayer*sizeof(Float));
	}

    vector<Size> nTypeEI(2*nLayer);
    vector<Size> nType(nLayer, 0);
    vector<vector<Size>> typeAccLayered(nLayer);
    PosInt qType = 0;
    for (PosInt iLayer=0; iLayer<nInputLayer; iLayer++) {
        PosInt jLayer = inputLayer[iLayer];
        cout << "layer " << jLayer << "(input layer):\n";
	    for (PosInt i=0; i<nInputType[iLayer]; i++) {
	    	cout << inputTypeAccCount[qType + i];
	    	if (i < nInputType[iLayer]-1) cout << ",";
	    	else cout << "\n";
            typeAccLayered[jLayer].push_back(inputTypeAccCount[qType+i]);
	    }
        nType[jLayer] = nInputType[iLayer];
        nTypeEI[2*jLayer] = nInputTypeEI[2*iLayer];
        nTypeEI[2*jLayer+1] = nInputTypeEI[2*iLayer+1];
        qType += nInputType[iLayer];
    }
    bool typeConsistent = true;
    qType = 0;
    for (PosInt iLayer=0; iLayer<nonInputLayer.size(); iLayer++) {
        PosInt jLayer = nonInputLayer[iLayer];
        cout << "layer " << jLayer << "(non-input layer):\n";
	    for (PosInt i=0; i<nonInputType[iLayer]; i++) {
	    	cout << nonInputTypeAccCount[qType + i];
	    	if (i < nonInputType[iLayer]-1) cout << ",";
	    	else cout << "\n";
            typeAccLayered[jLayer].push_back(inputTypeAccCount[qType+i]);
	    }
        if (nType[jLayer] > 0) typeConsistent = false;
        nType[jLayer] = nonInputType[iLayer];
        nTypeEI[2*jLayer] = nonInputTypeEI[2*iLayer];
        nTypeEI[2*jLayer+1] = nonInputTypeEI[2*iLayer+1];
        qType += nonInputType[iLayer];
    }
	vector<Size> typeAcc0;
    vector<Size>typeLayerID(1,0);
    PosInt totalType = 0;
    for (PosInt iLayer=0; iLayer<nLayer; iLayer++) {
        if (nType[jLayer] == 0) typeConsistent = false;
        typeAcc0.push_back(0);
        typeAcc0.insert(typeAcc0.end(), typeAccLayered[iLayer].begin(), typeAccLayered[iLayer].end());
        totalType += nType[iLayer];
        typeLayerAcc.push_back(totalType);
    }
    assert(totalType + nLayer == typeAcc0.size());
    assert(typeLayerID.back() == totalType);
    typeLayerID.pop_back();
    if (!typeConsistent) {
        cout << "# types for input/non-input layers is not consistent\n";
        return EXIT_FAILURE;
    }

    mI.resize(nLayer);
    mE.resize(nLayer);
    vector<Size> nI(nLayer);
    vector<Size> nE(nLayer);
    vector<Size> nTypeI(nLayer);
    vector<Size> nTypeE(nLayer);
    Size totalTypeE, totalTypeI;
    cout << "layer, nTypeE, nTypeI, nE, nI\n";
    for (PosInt iLayer=0; iLayer<nLayer; iLayer++) {
	    if (typeAccLayered[iLayer].back() != neuronPerBlock[iLayer]) {
            cout << "inconsistent typeAcc with neuronPerBlock\n";
            return EXIT_FAILURE;
        }
        nTypeE[iLayer] = nTypeEI[2*iLayer];
        nTypeI[iLayer] = nTypeEI[2*iLayer+1];
        nE[iLayer] = typeAccLayered[iLayer][nTypeE[iLayer]-1];
        nI[iLayer] = neuronPerBlock[iLayer] - nE[iLayer];
        mE[iLayer] = nE[iLayer] * nblock[iLayer];
        mI[iLayer] = nI[iLayer] * nblock[iLayer];
	    cout << iLayer << ", " << nTypeE[iLayer] << ", " << nTypeI[iLayer] << ", " << nE[iLayer] << ", " << nI[iLayer] << "\n";
        totalTypeE += nTypeE[iLayer];
        totalTypeI += nTypeI[iLayer];
    }

    if (synapticLoc.size() != totalType*totalType) {
		cout << "synapticLoc has size of " << synapticLoc.size() << ", should be " << totalType*totalType << "\n";
		return EXIT_FAILURE;
	}
	Float* d_synloc; // self matrix of row, column [typeE, typeI] x nLayer 
    checkCudaErrors(cudaMalloc((void**)&d_synloc, (totalType*totalType)*sizeof(Float)));
    checkCudaErrors(cudaMemcpy(d_synloc, &(synapticLoc[0]), (totalType*totalType)*sizeof(Float), cudaMemcpyHostToDevice));

	Size nType0;
    const Size nArchtype = 2;

    if (nTypeMat.size() != totalType*totalType) {
		cout << "nTypeMat has size of " << nTypeMat.size() << ", should be " << totalType*totalType << "\n";
		return EXIT_FAILURE;
	}
    if (sTypeMat.size() != totalType*totalType) {
		cout << "sTypeMat has size of " << sTypeMat.size() << ", should be " << totalType*totalType << "\n";
		return EXIT_FAILURE;
	}

    Size totalTypeE = 0;
    Size totalTypeI = 0;
    for (PosInt i=0; i<nLayer; i++) {
        totalTypeE += nTypeE[i];
        totalTypeI += nTypeI[i];
    }

    if (gap_nTypeMatE.size() != totalTypeE*totalTypeE && gap_nTypeMatE.size() != 1 && gap_nTypeMatE.size() != 0) { // gapE matrix each layer then gapI
		cout << "array gap_nTypeMatE have size of " << gap_nTypeMatE.size() << ", should be " << totalTypeE << "x" << totalTypeE << ", 0(none) or 1(for all)\n";
		return EXIT_FAILURE;
	} else {
        if (gap_nTypeMatE.size() == 0) {
            cout << "no exc gap junctions\n";
        } else {
            if (gap_nTypeMatE.size() == 1) {
                gap_nTypeMatE.insert(gap_nTypeMatE.end(), totalTypeE*totalTypeE-1, gap_nTypeMatE[0]);
            } else {
		        for (PosInt i=0; i<totalTypeE; i++) {
		        	for (PosInt j=0; j<i; j++) {
		        		if (gap_nTypeMatE[i*totalTypeE + j] != gap_nTypeMatE[j*totalTypeE + i]) {
		        		    	cout << "gap_nTypeMat is not symmetric for exc-type neurons in layer " << i << "\n";
                                return EXIT_FAILURE;
		        		}
                    }
		        }
            }
        }
	}

    if (gap_nTypeMatE.size() > 0) {
        if (gap_sTypeMatE.size() != totalTypeE*totalTypeE && gap_sTypeMatE.size() != 1) { // gapE matrix each layer then gapI
	    	cout << "array gap_sTypeMatE have size of " << gap_sTypeMatE.size() << ", should be " << totalTypeE << "x" << totalTypeE << ", 0(none) or 1(for all)\n";
	    	return EXIT_FAILURE;
	    } else {
            if (gap_sTypeMatE.size() == 1) {
                gap_sTypeMatE.insert(gap_sTypeMatE.end(), totalTypeE*totalTypeE-1, gap_sTypeMatE[0]);
            } else {
	    	    for (PosInt i=0; i<totalTypeE; i++) {
	    	    	for (PosInt j=0; j<i; j++) {
	    	    		if (gap_sTypeMatE[i*totalTypeE + j] != gap_sTypeMatE[j*totalTypeE + i]) {
	    	    		    	cout << "gap_sTypeMat is not symmetric for exc-type neurons in layer " << i << "\n";
                                return EXIT_FAILURE;
	    	    		}
                    }
	    	    }
            }
	    }
    } else {
        if (gap_sTypeMatE.size() != 0) {
            cout << "size of gap_nTypeMatE is zero, ignoring gap_sTypeMatE\n";
        }
    }

    if (gap_nTypeMatI.size() != totalTypeI*totalTypeI && gap_nTypeMatI.size() != 1 && gap_nTypeMatI.size() != 0) { // gapE matrix each layer then gapI
		cout << "array gap_nTypeMatI have size of " << gap_nTypeMatI.size() << ", should be " << totalTypeI << "x" << totalTypeI << ", 0(none) or 1(for all)\n";
		return EXIT_FAILURE;
	} else {
        if (gap_nTypeMatI.size() == 0) {
            cout << "no inh gap junctions\n";
        } else {
            if (gap_nTypeMatI.size() == 1) {
                gap_nTypeMatI.insert(gap_nTypeMatI.end(), totalTypeI*totalTypeI-1, gap_nTypeMatI[0]);
            } else {
		        for (PosInt i=0; i<totalTypeI; i++) {
		        	for (PosInt j=0; j<i; j++) {
		        		if (gap_nTypeMatI[i*totalTypeI + j] != gap_nTypeMatI[j*totalTypeI + i]) {
		        		    	cout << "gap_nTypeMat is not symmetric for inh-type neurons in layer " << i << "\n";
                                return EXIT_FAILURE;
		        		}
                    }
		        }
            }
        }
	}

    if (gap_nTypeMatI.size() > 0) {
        if (gap_sTypeMatI.size() != totalTypeI*totalTypeI && gap_sTypeMatI.size() != 1) { // gapE matrix each layer then gapI
	    	cout << "array gap_sTypeMatI have size of " << gap_sTypeMatI.size() << ", should be " << totalTypeI << "x" << totalTypeI << ", 0(none) or 1(for all)\n";
	    	return EXIT_FAILURE;
	    } else {
            if (gap_sTypeMatI.size() == 1) {
                gap_sTypeMatI.insert(gap_sTypeMatI.end(), totalTypeI*totalTypeI-1, gap_sTypeMatI[0]);
            } else {
	    	    for (PosInt i=0; i<totalTypeI; i++) {
	    	    	for (PosInt j=0; j<i; j++) {
	    	    		if (gap_sTypeMatI[i*totalTypeI + j] != gap_sTypeMatI[j*totalTypeI + i]) {
	    	    		    	cout << "gap_sTypeMat is not symmetric for inh-type neurons in layer " << i << "\n";
                                return EXIT_FAILURE;
	    	    		}
                    }
	    	    }
            }
	    }
    } else {
        if (gap_sTypeMatI.size() != 0) {
            cout << "size of gap_nTypeMatI is zero, ignoring gap_sTypeMatI\n";
        }
    }

    // read predetermined neuronal subtypes.
	// read predetermined functional response features of neurons (use as starting seed if involve learning).
    fV1_feature.open(V1_feature_filename, ios::in|ios::binary);
	if (!fV1_feature) {
		cout << "failed to open feature file:" << V1_feature_filename << "\n";
		return EXIT_FAILURE;
	}
	Size nFeature;
    fV1_feature.read(reinterpret_cast<char*>(&nFeature), sizeof(Size));
    vector<vector<PosInt>> featureLayer(nFeature);
	vector<vector<vector<Float>>> featureValue(nFeature); // [OD, OP, ..]
    Size nfl;
    fV1_feature.read(reinterpret_cast<char*>(&nfl), sizeof(Size));
	for (PosInt iF = 0; iF<nFeature; iF++) {
        featureLayer[iF].resize(nfl);
        fV1_feature.read(reinterpret_cast<char*>(&(featureLayer[iF][0])), nfl*sizeof(PosInt));
        vector<Size> _n(nfl);
        fV1_feature.read(reinterpret_cast<char*>(&_n[0]), nfl*sizeof(Size));
        for (PosInt i = 0; i<nfl; i++) {
            Size jLayer = featureLayer[iF][i];
            featureValue[iF][jLayer].resize(_n[i]);
	        fV1_feature.read(reinterpret_cast<char*>(&(featureValue[iF][jLayer][0])), _n[i] * sizeof(Float));
	        cout << iF << "th featureValue range in layer " << jLayer << ": [" << *min_element(featureValue[iF][jLayer].begin(), featureValue[iF][jLayer].end()) << ", " << *max_element(featureValue[iF][jLayer].begin(), featureValue[iF][jLayer].end()) << "]\n";
	        }
        }
    }
    for (PosInt i = 0; i<nLayer; i++) {
        for (PosInt j=0; j<featureValue[1][i].size(); j++) {
            featureValue[1][i][j] = (featureValue[1][i][j] - 0.5)*M_PI;
        }
    }
	fV1_feature.close();

    cout << "defaultFeatureValue:\n";
    for (PosInt i=0; i<nFeature; i++) {
        cout << defaultFeatureValue[i];
        if (i==nFeature-1) {
            cout << "\n";
        } else {
            cout << ", ";
        }
    }

    if (typeFeatureMat.size()/(totalType*totalType*pPerFeature) != nFeature) {
		cout << "typeFeatureMat has " << typeFeatureMat.size()/(totalType*totalType*pPerFeature) << " features, should be " << nFeature << "\n";
		return EXIT_FAILURE;
	}

    if (gap_nTypeMatE.size() > 0) {
        if (gap_fTypeMatE.size() != totalTypeE*totalTypeE*nFeature*pPerFeature && gap_fTypeMatE.size() != 0 && gap_fTypeMatE != pPerFeature*nFeature) { // gapE matrix each layer then gapE
	    	cout << "array gap_fTypeMatE have size of " << gap_fTypeMatE.size() << ", should be " << pPerFeature << "x" << nFeature << "x" << totalTypeE << "x" << totalTypeE << ", 0(defaut) or nFeature x pPerFeature (for each feature)\n";
	    	return EXIT_FAILURE;
	    } else {
            if (gap_fTypeMatE.size() == 0) {
                for (PosInt i=0; i<pPerFeature*nFeature; i++) {
                    gap_fTypeMatE.insert(gap_nTypeMatE.end(), totalTypeE*totalTypeE, defaultFeatureValue[i]);
                }
            } else {
                if (gap_fTypeMatE.size() == pPerFeature*nFeature) {
                    vector<Float>val(gap_fTypeMatE.begin(), gap_fTypeMatE.end());
                    gap_fTypeMatE.clear();
                    for (PosInt i=0; i<pPerFeature*nFeature; i++) {
                        gap_fTypeMatE.insert(gap_nTypeMatE.end(), totalTypeE*totalTypeE, val[i]);
                    }
                } else {
                    for (PosInt f=0; f<pPerFeature*nFeature; f++) {
	    	            for (PosInt i=0; i<totalTypeE; i++) {
	    	            	for (PosInt j=0; j<i; j++) {
	    	            		if (gap_fTypeMatE[f*totalTypeE*totalTypeE + i*totalTypeE + j] != gap_fTypeMatE[f*totalTypeE*totalTypeE + j*totalTypeE + i]) {
	    	            		    	cout << "gap_fTypeMat is not symmetric for exc-type neurons in layer " << i << " for feature " << f/2 << "-" << f%2 << "\n";
                                        return EXIT_FAILURE;
	    	            		}
                            }
	    	            }
                    }
                }
            }
	    }
    } else {
        if (gap_fTypeMatE.size() != 0) {
            cout << "size of gap_nTypeMatE is zero, ignoring gap_fTypeMatE\n";
        }
    }

    if (gap_nTypeMatI.size() > 0) {
        if (gap_fTypeMatI.size() != totalTypeI*totalTypeI*nFeature*pPerFeature && gap_fTypeMatI.size() != 0 && gap_fTypeMatI != pPerFeature*nFeature) { // gapE matrix each layer then gapE
	    	cout << "array gap_fTypeMatI have size of " << gap_fTypeMatI.size() << ", should be " << pPerFeature << "x" << nFeature << "x" << totalTypeI << "x" << totalTypeI << ", 0(defaut) or nFeature x pPerFeature(for each feature)\n";
	    	return EXIT_FAILURE;
	    } else {
            if (gap_fTypeMatI.size() == 0) {
                for (PosInt i=0; i<pPerFeature*nFeature; i++) {
                    gap_fTypeMatI.insert(gap_nTypeMatI.end(), totalTypeI*totalTypeI, defaultFeatureValue[i]);
                }
            } else {
                if (gap_fTypeMatI.size() == pPerFeature*nFeature) {
                    vector<Float>val(gap_fTypeMatI.begin(), gap_fTypeMatI.end());
                    gap_fTypeMatI.clear();
                    for (PosInt i=0; i<pPerFeature*nFeature; i++) {
                        gap_fTypeMatI.insert(gap_nTypeMatI.end(), totalTypeI*totalTypeI, val[i]);
                    }
                } else {
                    for (PosInt f=0; f<pPerFeature*nFeature; f++) {
	    	            for (PosInt i=0; i<totalTypeI; i++) {
	    	            	for (PosInt j=0; j<i; j++) {
	    	            		if (gap_fTypeMatI[f*totalTypeI*totalTypeI + i*totalTypeI + j] != gap_fTypeMatI[f*totalTypeI*totalTypeI + j*totalTypeI + i]) {
	    	            		    	cout << "gap_fTypeMat is not symmetric for inh-type neurons in layer " << i << " for feature " << f/2 << "-" << f%2 << "\n";
                                        return EXIT_FAILURE;
	    	            		}
                            }
	    	            }
                    }
                }
            }
	    }
    } else {
        if (gap_fTypeMatI.size() != 0) {
            cout << "size of gap_nTypeMatI is zero, ignoring gap_fTypeMatI\n";
        }
    }

    if (nearNeighborBlock.size() != nLayer*nLayer || nearNeighborBlock != 1) {
        cout << "nearNeighborBlock must have a size of " <<nLayer << " x " << nLayer << "(nLayer) or 1\n";
    } else {// including self
        if (nearNeighborBlock.size() == 1) {
            nearNeighborBlock.insert(nearNeighborBlock.end(), nLayer*nLayer-1, nearNeighborBlock[0]);
        }
        for (PosInt i=0; i<nLayer; i++) {
            for (PosInt j=0; j<nLayer; j++) {
                if (j == 0) {
                    nearNeighborBlock[i*nLayer+j] += 1;
                }
            }
        }
    }

    if (maxNeighborBlock.size() != nLayer*nLayer || maxNeighborBlock != 1) {
        cout << "maxNeighborBlock must have a size of " << nLayer << " x " << nLayer << "(nLayer) or 1\n";
        return EXIT_FAILURE;
    } else {// including self
        if (maxNeighborBlock.size() == 1) {
            maxNeighborBlock.insert(maxNeighborBlock.end(), nLayer*nLayer-1, maxNeighborBlock[0]);
        }
        for (PosInt i=0; i<nLayer; i++) {
            for (PosInt j=0; j<nLayer; j++) {
                if (j == 0) {
                    maxNeighborBlock[i*nLayer+j] += 1;
                }
            }
        }
    }

    if (disGauss.size() != nLayer*nLayer || disGauss != 1) {
        cout << "disGauss must have a size of " << nLayer << " x " << nLayer << "(nLayer) or 1\n";
        return EXIT_FAILURE;
    } else {
        if (disGauss.size() == 1) {
            disGauss.insert(disGauss.end(), nLayer*nLayer-1, disGauss[0]);
        }
    }

    fV1_conMat.open(V1_conMat_filename + suffix, ios::out | ios::binary);
	if (!fV1_conMat) {
		cout << "cannot open " << V1_conMat_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fV1_conMat.write((char*) &nearNeighborBlock[0], nLayer*sizeof(Size));
    }

    fV1_delayMat.open(V1_delayMat_filename + suffix, ios::out | ios::binary);
	if (!fV1_delayMat) {
		cout << "cannot open " << V1_delayMat_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fV1_delayMat.write((char*) &nearNeighborBlock[0], nLayer*sizeof(Size));
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

    size_t sizeof_block = 0;
    size_t sizeof_neighbor = 0;
    vector<PosInt> blockAcc(nLayer+1, 0);
    vector<PosInt> neighborBlockAcc(nLayer+1, 0);
    vector<PosInt> networkSizeAcc(nLayer+1, 0);
    for (PosInt i=0; i<nLayer; i++) {
        sizeof_block += 2*nblock[i]*sizeof(Float); // size of block_pos
        blockAcc[i+1] = block_pos_acc[i] + nblock[i];
        networkSizeAcc[i+1] = networkSizeAcc[i] + networkSize[i];
        Size qMaxNeighborBlock = 0;
        for (PosInt j=0; j<nLayer; j++) {
            sizeof_neighbor += nblock[i]*maxNeighborBlock[i*nLayer+j]*sizeof(PosInt); // size of neighborBlockId
            qMaxNeighborBlock += maxNeighborBlock[i*nLayer+j];
        }
        neighborBlockAcc[i+1] = neighborBlockAcc[i] + nblock[i]*qMaxNeighborBlock;
    }
    sizeof_neighbor += 2*nLayer*blockAcc.back()*sizeof(Size); // n(Near)NeighborBlock
    size_t sizeof_pos = 2*networkSizeAcc.back()*sizeof(Float);
    void* __restrict__ block_chunk;
    checkCudaErrors(cudaMalloc((void**)&block_chunk, sizeof_block + sizeof_neighbor + sizeof_pos + (3*nLayer+1)*sizeof(PosInt)));
    Float* __restrict__ d_pos = (Float*) block_chunk;
    Float* __restrict__ d_block_x = d_pos + 2*networkSizeAcc.back();
    Float* __restrict__ d_block_y = d_block_x + blockAcc.back();
    PosInt*  __restrict__ d_neighborBlockId = (Size*) (d_block_y + blockAcc.back());
    Size*  __restrict__ d_nNeighborBlock = d_neighborBlockId + neighborBlockAcc.back();
	Size* __restrict__ d_nNearNeighborBlock = d_nNeighborBlock + nLayer*blockAcc.back();
    PosInt* __restrict__ d_blockAcc = d_nNearNeighborBlock + nLayer*blockAcc.back();
    PosInt* __restrict__ d_neighborBlockAcc = d_blockAcc + nLayer+1; 
    PosInt* __restrict__ d_networkSizeAcc = d_neighborBlockAcc + nLayer+1; 

	cudaStream_t *layerStream = new cudaStream_t[nLayer];
    for (PosInt i=0; i<nLayer; i++) {
		checkCudaErrors(cudaStreamCreate(&layerStream[i]));
    }
        
    for (PosInt i=0; i<nLayer; i++) {
        checkCudaErrors(cudaMemcpyAsync(d_pos+2*networkSizeAcc[i], &(pos[i][0]), 2*networkSize[i]*sizeof(Float), cudaMemcpyHostToDevice, layerStream[i]));
        cal_blockPos<<<nblock[i], neuronPerBlock[i],0,layerStream[i]>>>(
            d_pos + 2*networkSizeAcc[i], 
	    	d_block_x + blockAcc[i], d_block_y + blockAcc[i], 
	    	networkSize[i]);
	    getLastCudaError("cal_blockPos failed");
    }
    printf("block centers calculated\n");

	//shared_mem = sizeof(Size);
    // blocks -> blocks, threads -> cal neighbor blocks
    for (PosInt i=0; i<nLayer; i++) {
        Size qMaxNeighborBlock = 0;
        for (PosInt j=0; j<nLayer; j++) {
            get_neighbor_blockId<<<nblock[i], blockSize, maxNeighborBlock*(sizeof(PosInt)+sizeof(Float)),0,layerStream[i]>>>(
                d_block_x, d_block_y, d_blockAcc,
	    	    d_neighborBlockId + neighborBlockAcc[i] + qMaxNeighborBlock,
                d_nNeighborBlock + nLayer*blockAcc[i] + j*nblock[i],
                d_nNearNeighborBlock + nLayer*blockAcc[i] + j*nblock[i],
	    	    nblock[j], blockROI[i*nLayer+j], blockROI_max[i*nLayer+j], maxNeighborBlock[i*nLayer+j], i, j);
            qMaxNeighborBlock += nblock[i] * maxNeighborBlock[i*nLayer+j];
        }
    }
	getLastCudaError("get_neighbor_blockId failed");
    printf("neighbor blocks acquired\n");
    // n(Near/Max)NeighborBlock: [recv_layer, send_layer, recv_nblock]
    // neighborBlockId: [recv_layer, send_layer, recv_nblock]
    fBlock_pos.write((char*)block_x, nblock*sizeof(Float));
    fBlock_pos.write((char*)block_y, nblock*sizeof(Float));
    fBlock_pos.close();


	{// dendrites and axons
    	if (dAxon.size() != nLayer*totalType) {
    	    cout << "size of dAxon: " << dAxon.size() << " should be consistent with the number of neuronal types: " << nLayer << "x" << totalType << "\n"; 
    	    return EXIT_FAILURE;
    	}
    	if (dDend.size() != nLayer*totalType) {
    	    cout << "size of dDend: " << dDend.size() << " should be consistent with the number of neuronal types: " << nLayer << "x" << totalType << "\n"; 
    	    return EXIT_FAILURE;
    	}

    	if (rAxon.size() != nLayer*totalType) {
    	    cout << "size of rAxon: " << rAxon.size() << " should be consistent with the number of neuronal types: " << nLayer << "x" << totalType << "\n"; 
    	    return EXIT_FAILURE;
    	} else {
    	    // adjust the scale
    	    for (Float &r: rAxon){
    	        r *= dScale;
    	    }
    	}
    	if (rDend.size() != nLayer*totalType) {
    	    cout << "size of rDend: " << rDend.size() << " should be consistent with the number of neuronal types: " << nLayer << "x" << totalType << "\n"; 
    	    return EXIT_FAILURE;
    	} else {
    	    // adjust the scale
    	    for (Float &r: rDend){
    	        r *= dScale;
    	    }
    	}
	}
    initializePreferenceFunctions(nFeature);
    //Float speedOfThought = 1.0f; specify instead in patch.cu through patchV1.cfg

    void* __restrict__ preset_chunk;
    size_t morph_size = nLayer*totalType*4*sizeof(Float);
    size_t matSize = (totalType*totalType + totalTypeE*totalTypeE + totalTypeI*totalTypeI)*(sizeof(Float) + sizeof(Size)) + pPerFeature*nFeature*(totalType*totalType + totalTypeE*totalTypeE + totalTypeI*totalTypeI)*sizeof(Float);
    size_t  = networkSize.back()*sizeof(Float);
    checkCudaErrors(cudaMalloc((void**)&preset_chunk, morph_size + matSize + ));
    Float* rden = (Float*) preset_chunk; 
    Float* raxn = rden + nLayer*totalType;
	Float* dden = raxn + nLayer*totalType;
	Float* daxn = dden + nLayer*totalType;
    Size*  d_nTypeMat = (Size*) (daxn + nLayer*totalType);
    Float* d_sTypeMat = (Float*) (d_nTypeMat + totalType*totalType);
    Float* d_fTypeMat = d_sTypeMat + totalType*totalType;

    Size*  d_gap_nTypeMatE = (Size*) (d_fTypeMat + pPerFeature*nFeature*totalType*totalType);
    Float* d_gap_sTypeMatE = (Float*) (d_gap_nTypeMatE + totalTypeE*totalTypeE);
    Float* d_gap_fTypeMatE = d_gap_sTypeMatE + totalTypeE*totalTypeE;

    Size*  d_gap_nTypeMatI = (Size*) (d_gap_fTypeMatE + pPerFeature*nFeature*totalTypeE*totalTypeE);
    Float* d_gap_sTypeMatI = (Float*) (d_gap_nTypeMatI + totalTypeI*totalTypeI);
    Float* d_gap_fTypeMatI = d_gap_sTypeMatI + totalTypeI*totalTypeI;

	Float* d_ExcRatio = d_gap_fTypeMatI + pPerFeature*nFeature*totalTypeI*totalTypeI;
    checkCudaErrors(cudaMalloc((void**)&d_ExcRatio, networkSize*sizeof(Float)));

	Float* ExcRatio = new Float[networkSize];
	Size* nLGN_V1_Max = new Size[totalType];
	Float* sLGN_V1_Max = new Float[totalType];
    
    Float* sLGN = new Float[networkSizeAcc.back()]{};
	Size* nLGN = new Size[networkSizeAcc.back()]{};
    read_LGN_V1_stats(LGN_V1_s_filename + conLGN_suffix, sLGN, nLGN, inputLayer, networkSizeAcc);

    void* LGN_chunk;
    size_t LGN_size = networkSizeAcc.back()*(sizeof(Size) + sizeof(Float));
    checkCudaErrors(cudaMalloc((void**)&LGN_chunk, LGN_size));
    Float* d_sLGN = (Float*) LGN_chunk;
    Float* d_nLGN = (Size*) (d_sLGN + networkSize.back());

    checkCudaErrors(cudaMemcpy(d_nLGN, nLGN, networkSize.back()*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_sLGN, sLGN, networkSize.back()*sizeof(Float), cudaMemcpyHostToDevice));

	for (PosInt i=0; i<nType; i++) {
		presetConstExcSyn[i] = mean_nLGN*synPerConFF[i] + hInit_pack.nTypeMat[i]*synPerCon[i];
		if (presetConstExcSyn[i] - nLGN_V1_Max[i]*synPerConFF[i] < hInit_pack.nTypeMat[i]*synPerCon[i]*min_FB_ratio) {
			cout << "Exc won't be constant for type " << i << " with current parameter set, min_FB_ratio will be utilized, presetConstExcSyn = " << presetConstExcSyn[i] << ", max ffExcSyn = " << nLGN_V1_Max[i]*synPerConFF[i] << ", corticalExcSyn = " << hInit_pack.nTypeMat[i]*synPerCon[i] << "\n";
		}
	}
    delete []nLGN_V1;
    delete []nLGN_V1_Max;

    initialize<<<nblock, neuronPerBlock>>>(
        state,
	    d_preType,
		rden, raxn, dden, daxn,
		//preF_type, preS_type, preN_type, d_LGN_V1_sSum, d_ExcRatio, d_extExcRatio, min_FB_ratio,
		preF_type, preS_type, preN_type, d_nLGN_V1, d_ExcRatio, d_extExcRatio, d_synPerCon, d_synPerConFF, min_FB_ratio, C_InhRatio,
		init_pack, seed, networkSize, nType, nArchtype, nFeature, CmoreN, ClessI, mean_nLGN);
	getLastCudaError("initialize failed");
	checkCudaErrors(cudaDeviceSynchronize());
    printf("initialzied\n");
    checkCudaErrors(cudaMemcpy(ExcRatio, d_ExcRatio, networkSize*sizeof(Float), cudaMemcpyDeviceToHost));

    // check memory availability
    size_t memorySize, d_memorySize, matSize;

    size_t statSize = 0;
    for (PosInt i=0; i<nLayer; i++) {
        statsSize += 2*nType[i]*networkSize[i]*sizeof(Size) + // preTypeConnected and *Avail
        	        nType[i]*networkSize[i]*sizeof(Float); // preTypeStrSum
    }

    size_t gap_statSizeE = 0;
    for (PosInt i=0; i<nLayer; i++) {
        gap_statSizeE += nTypeE[i]*mE[i]*sizeof(Size) + // preTypeGapped
        	              nTypeE[i]*mE[i]*sizeof(Float); // preTypeStrGapped
    }

    size_t gap_statSizeI = 0;
    for (PosInt i=0; i<nLayer; i++) {
        gap_statSizeI += nTypeI[i]*mI[i]*sizeof(Size) + // preTypeGapped
        	              nTypeI[i]*mI[i]*sizeof(Float); // preTypeStrGapped
    }

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
                       2*networkSize*sizeof(double); 

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
	
	Size* d_typeAcc0;
    checkCudaErrors(cudaMalloc((void**)&d_typeAcc0, (totalType+nLayer)*sizeof(Size)));
    checkCudaErrors(cudaMemcpy(d_typeAcc0, &(typeAcc0[0]), (totalType+nLayer)*sizeof(Size), cudaMemcpyHostToDevice));
	
    //cudaStream_t s0, s1, s2;
    //cudaEvent_t i0, i1, i2;
    //cudaEventCreate(&i0);
    //cudaEventCreate(&i1);
    //cudaEventCreate(&i2);
    //checkCudaErrors(cudaStreamCreate(&s0));
    //checkCudaErrors(cudaStreamCreate(&s1));
    //checkCudaErrors(cudaStreamCreate(&s2));
    checkCudaErrors(cudaMemcpy(d_feature, &featureValue[0], nFeature*networkSize*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pos, &pos[0], 2*networkSize*sizeof(double), cudaMemcpyHostToDevice));

    //Size shared_mem;

    
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
    		Size* totalTypeI = new Size[nTypeI];
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=0; j<nTypeI; j++) {
    		        preConn[i*nTypeI + j] = 0;
    		        preAvail[i*nTypeI + j] = 0;
    		        preStr[i*nTypeI + j] = 0.0;
    		    }
    		    totalTypeI[i] = 0; 
    		}
    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=ib*nI; j<(ib+1)*nI; j++) {
    		        for (PosInt k=0; k<nTypeI; k++) {
    		            if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
    		                preConn[i*nTypeI + k] += preTypeGapped[i*mI + j];
    		                preAvail[i*nTypeI + k] += preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
    		                preStr[i*nTypeI + k] += preTypeStrGapped[i*mI + j];
    		                if (i == 0) {
    		                    totalTypeI[k]++;
    		                }
    		                break;
    		            }
    		        }
    		    }
    		}

    		for (PosInt i=0; i<nTypeI; i++) {
    		    for (PosInt j=0; j<nTypeI; j++) {
    		        preConn[i*nTypeI + j] /= totalTypeI[j];
    		        preAvail[i*nTypeI + j] /= totalTypeI[j];
    		        preStr[i*nTypeI + j] /= totalTypeI[j];
    		    }
    		    cout << "type " << i << " has " << totalTypeI[i] << " neurons\n";
    		}
    		delete [] totalTypeI;

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
    	Size* totalTypeI = new Size[nTypeI];
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=0; j<nTypeI; j++) {
    	        preConn[i*nTypeI + j] = 0;
    	        preAvail[i*nTypeI + j] = 0;
    	        preStr[i*nTypeI + j] = 0.0;
    	    }
    	    totalTypeI[i] = 0; 
    	}
    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=ib*nI; j<(ib+1)*nI; j++) {
    	        for (PosInt k=0; k<nTypeI; k++) {
    	            if (j%nI < (typeAccCount[k+nTypeE]-nE))  {
    	                preConn[i*nTypeI + k] += preTypeGapped[i*mI + j];
    	                preAvail[i*nTypeI + k] += preTypeAvail[(i+nTypeE)*networkSize + (j/nI+1)*nE + j];
    	                preStr[i*nTypeI + k] += preTypeStrGapped[i*mI + j];
    	                if (i == 0) {
    	                    totalTypeI[k]++;
    	                }
    	                break;
    	            }
    	        }
    	    }
    	}

    	for (PosInt i=0; i<nTypeI; i++) {
    	    for (PosInt j=0; j<nTypeI; j++) {
    	        preConn[i*nTypeI + j] /= totalTypeI[j];
    	        preAvail[i*nTypeI + j] /= totalTypeI[j];
    	        preStr[i*nTypeI + j] /= totalTypeI[j];
    	    }
    	    cout << "type " << i << " has " << totalTypeI[i] << " neurons\n";
    	}
    	delete [] totalTypeI;

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
    checkCudaErrors(cudaFree(d_neighborMat));

    checkCudaErrors(cudaFree(block_chunk));
	for (PosInt i=0; i<nLayer; i++) {
		checkCudaErrors(cudaStreamDestroy(layerStream[i]));
	}

	free(cpu_chunk);
    return 0;
}
