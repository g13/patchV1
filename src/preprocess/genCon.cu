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
    vector<Size> maxNeighborBlock, nearNeighborBlock, maxDistantNeighbor;
    vector<Size> gap_maxDistantNeighbor;
	vector<Size> nonInputTypeEI;
    string inputFolder, outputFolder, res_suffix;
    string V1_type_filename, V1_feature_filename, V1_allpos_filename, LGN_V1_s_filename, LGN_V1_ID_filename, suffix, conLGN_suffix, LGN_V1_cfg_filename, output_cfg_filename;
	string V1_conMat_filename, V1_delayMat_filename, V1_gapMatE_filename, V1_gapMatI_filename;
	string V1_vec_filename;
	string block_pos_filename, neighborBlock_filename, stats_filename, gapStats_filename;
    Float dScale;
	vector<Float> ffSynOccupyRatio, synOccupyRatioE, inhRatio;
	Float longRangeROI;
	Float longRangeCorr;
    bool strictStrength, CmoreN;
	vector<Float> disGauss;
	vector<Float> rDend, rAxon;
    vector<Float> gapJunctDis;
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
        ("DisGauss", po::value<vector<Float>>(&disGauss), "if set true, conn. prob. based on distance will follow a 2D gaussian with a variance. of (rden*disGauss)^2, otherwise 0 will based on the overlap of the area specified by raxn and rden")
        ("strictStrength", po::value<bool>(&strictStrength), "strictly match preset summed connection")
        ("CmoreN", po::value<bool>(&CmoreN), "if true complex gets more connections otherwise stronger strength")
        ("rDend", po::value<vector<Float>>(&rDend),  "a vector of dendritic extensions' radius, size of totalType x totalType")
        ("rAxon", po::value<vector<Float>>(&rAxon),  "a vector of axonic extensions' radius, size of totalType x totalType")
        ("dScale",po::value<Float>(&dScale)->default_value(1.0),"a scaling ratio of all the neurites' lengths <radius>")
        ("longRangeROI", po::value<Float>(&longRangeROI), "ROI of long-range cortical input")
        ("longRangeCorr", po::value<Float>(&longRangeCorr), "correlation between long-range cortical inputs that cortical cells receives")
        ("gapJunctDis", po::value<vector<Float>>(&gapJunctDis), " maximal gap junction location relative to the soma, size of totalType")
		//("LGN_targetFR", po::value<Float>(&LGN_targetFR), "target firing rate of a LGN cell")
		//("targetFR", po::value<vector<Float>>(&targetFR), "a vector of target firing rate of different neuronal types")
		//("FF_FB_ratio", po::value<Float>(&FF_FB_ratio), "excitation ratio of FF over total excitation")
		("synOccupyRatioE", po::value<vector<Float>>(&synOccupyRatioE), "the resource occupying ratio of cortical excitatory connection from each type of neuron, set at least one to 1, size of nTypeE")
		("ffSynOccupyRatio", po::value<vector<Float>>(&ffSynOccupyRatio), "the resource occupying ratio of feedforward excitatory connection from each type of LGN, compared with unity ratio in cortical excitation, size of mLayer")
		("inhRatio", po::value<vector<Float>>(&inhRatio), "a gain or loss ratio of inhibition for each type of neuron, trying to balancing different ff fb and cort excitation, #inh = nTypeMat*(1+inhRatio*(1-ffRatio))")
		("minConTol", po::value<Float>(&minConTol), "minimum difference tolerance of the number of preset cortical connections")
        ("gap_nTypeMatE", po::value<vector<Size>>(&gap_nTypeMatE), "connection numbers' matrix for gap junctio between exc neuronal types, size of [totalTypeE, totalTypeE], across each layer, row_id -> postsynaptic, column_id -> presynaptic")
        ("gap_nTypeMatI", po::value<vector<Size>>(&gap_nTypeMatI), "connection numbers' matrix for gap junctio between inh neuronal types, size of [totalTypeI, totalTypeI], across each layer, row_id -> postsynaptic, column_id -> presynaptic")
        ("nTypeMat", po::value<vector<Size>>(&nTypeMat), "#connection matrix between neuronal types, size of [nType, nType], nType = sum(nTypeEI), row_id -> postsynaptic, column_id -> presynaptic")
        ("typeFeatureMat", po::value<vector<Float>>(&typeFeatureMat), "feature-specific matrixi for parameters of connection preference over neuronal types, size of [nFeature, totalType, totalType], row: postsynaptic-id, column: presynaptic-id")
        ("gap_fTypeMatE", po::value<vector<Float>>(&gap_fTypeMatE), "feature parameter of neuronal types, size of [nFeature, nTypeI, nTypeI], nTypeI = nTypeEI[1], row_id -> postsynaptic, column_id -> presynaptic")
        ("gap_fTypeMatI", po::value<vector<Float>>(&gap_fTypeMatI), "feature parameter of neuronal types, size of [nFeature, nTypeE, nTypeE], nTypeE = nTypeEI[0], row_id -> postsynaptic, column_id -> presynaptic")
        ("maxDistantNeighbor", po::value<vector<Size>>(&maxDistantNeighbor), "the preserved size of the array that store the presynaptic neurons' ID, who are not in the neighboring blocks")
        ("gap_maxDistantNeighbor", po::value<vector<Size>>(&gap_maxDistantNeighbor), "the preserved size of the array that store the pre-junction neurons' ID, who are not in the neighboring blocks")
        ("maxNeighborBlock", po::value<vector<Size>>(&maxNeighborBlock), "the preserved size (minus the nearNeighborBlock) of the array that store the neighboring blocks ID that goes into conVec")
        ("nearNeighborBlock", po::value<vector<Size>>(&nearNeighborBlock), "the preserved size of the array that store the neighboring blocks ID that goes into conMat, excluding the self block, self will be added later, nLayer x nLayer")
        ("fV1_feature", po::value<string>(&V1_feature_filename)->default_value("V1_feature.bin"), "file to read spatially predetermined functional features of neurons")
        ("fV1_allpos", po::value<string>(&V1_allpos_filename)->default_value("V1_allpos"), "the directory to read neuron positions")
        ("conLGN_suffix", po::value<string>(&conLGN_suffix)->default_value(""), "suffix associated with fLGN_V1_s")
		("fLGN_V1_cfg", po::value<string>(&LGN_V1_cfg_filename)->default_value("LGN_V1_cfg"),"file stores LGN_V1.cfg parameters")
		("fLGN_V1_s", po::value<string>(&LGN_V1_s_filename)->default_value("LGN_V1_sList"),"file stores LGN to V1 connection strengths, use conLGN_suffix")
		("fLGN_V1_ID", po::value<string>(&LGN_V1_ID_filename)->default_value("LGN_V1_idList"),"file stores LGN to V1 connection ID, use conLGN_suffix");

	po::options_description output_opt("output options");
	output_opt.add_options()
        ("suffix", po::value<string>(&suffix)->default_value(""), "a suffix to be associated with the generated connection profile")
        ("fV1_conMat", po::value<string>(&V1_conMat_filename)->default_value("V1_conMat"), "file that stores V1 to V1 connection within the neighboring blocks")
        ("fV1_delayMat", po::value<string>(&V1_delayMat_filename)->default_value("V1_delayMat"), "file that stores V1 to V1 transmission delay within the neighboring blocks")
        ("fV1_gapMatE", po::value<string>(&V1_gapMatE_filename)->default_value("V1_gapMatE"), "file that stores cortical exc-exc gap junction within the neighboring blocks")
        ("fV1_gapMatI", po::value<string>(&V1_gapMatI_filename)->default_value("V1_gapMatI"), "file that stores cortical inh-inh gap junction within the neighboring blocks")
        ("fV1_vec", po::value<string>(&V1_vec_filename)->default_value("V1_vec"), "file that stores V1 to V1 connection ID, strength and transmission delay outside the neighboring blocks")
		("fBlock_pos", po::value<string>(&block_pos_filename)->default_value("block_pos"), "file that stores the center coord of each block")
		("fNeighborBlock", po::value<string>(&neighborBlock_filename)->default_value("neighborBlock"), "file that stores the neighboring blocks' ID for each block")
		("fConnectome_cfg", po::value<string>(&output_cfg_filename)->default_value("connectome_cfg"), "file stores parameters in current cfg_file")
		("fStats", po::value<string>(&stats_filename)->default_value("conStats"), "file that stores the statistics of synaptic connections")
		("fGapStats", po::value<string>(&gapStats_filename)->default_value("gapStats"), "file that stores the statistics of gap connections");

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
			output_cfg_filename = outputFolder + output_cfg_filename;
		}
        if (vm["fV1_conMat"].defaulted()){
			V1_conMat_filename = outputFolder + V1_conMat_filename;
		}
		if (vm["fV1_delayMat"].defaulted()){
			V1_delayMat_filename = outputFolder + V1_delayMat_filename;
		}
		if (vm["fV1_gapMatE"].defaulted()){
			V1_gapMatE_filename = outputFolder + V1_gapMatE_filename;
		}
		if (vm["fV1_gapMatI"].defaulted()){
			V1_gapMatI_filename = outputFolder + V1_gapMatI_filename;
		}
		if (vm["fV1_vec"].defaulted()){
			V1_vec_filename = outputFolder + V1_vec_filename;
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
        if (vm["fGapStats"].defaulted()) {
            gapStats_filename = outputFolder + gapStats_filename;
        }
	}

    ifstream fV1_allpos, fV1_feature;
    ofstream fV1_conMat, fV1_delayMat, fV1_vec;
    ofstream fBlock_pos, fNeighborBlock;
    ofstream fStats;
    fstream fV1_gapMatE, fV1_gapMatI, fGapStats;

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
	vector<vector<Float>> pos;;
    vector<Float> layerDensity;
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
        fV1_allpos.read(reinterpret_cast<char*>(&nblock[0]), nLayer*sizeof(Size));
        fV1_allpos.read(reinterpret_cast<char*>(&neuronPerBlock[0]), nLayer*sizeof(Size));
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
        fV1_allpos.read(reinterpret_cast<char*>(&layerDensity), nLayer*sizeof(Float));
    }
	fV1_allpos.close();
	
    Size nInputLayer;
	vector<PosInt> inputLayer;
	vector<PosInt> inputLayerPick(nLayer,0);
	vector<Float> p_n_LGNeff;
	vector<Size> max_LGNeff;
	vector<Float> mean_nLGN;
	vector<Float> mean_sLGN;
    Size mLayer;
	vector<Size> mL;
	vector<Size> mR;
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
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&mLayer), sizeof(Size));

    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&mL[0]), mLayer*sizeof(Size));
    	fLGN_V1_cfg.read(reinterpret_cast<char*>(&mR[0]), mLayer*sizeof(Size));
        for (PosInt i=0; i<nInputLayer; i++) {
            nInputType[i] = nInputTypeEI[2*i] + nInputTypeEI[2*i+1];
            totalInputType += nInputType[i];
            inputLayerPick[inputLayer[i]] = 1;
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
    vector<Size> nonInputType(nLayer);
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
        nonInputType.push_back(nonInputTypeEI[2*iLayer] + nonInputTypeEI[2*iLayer+1]);
	    for (PosInt i=0; i<nonInputType[iLayer]; i++) {
	    	cout << nonInputTypeAccCount[qType + i];
	    	if (i < nonInputType[iLayer]-1) cout << ",";
	    	else cout << "\n";
            typeAccLayered[jLayer].push_back(nonInputTypeAccCount[qType+i]);
	    }
        if (nType[jLayer] > 0) typeConsistent = false;
        nType[jLayer] = nonInputType[iLayer];
        nTypeEI[2*jLayer] = nonInputTypeEI[2*iLayer];
        nTypeEI[2*jLayer+1] = nonInputTypeEI[2*iLayer+1];
        qType += nonInputType[iLayer];
    }

	vector<Size> mI(nLayer);
	vector<Size> mE(nLayer);
    vector<Size> nI(nLayer);
    vector<Size> nE(nLayer);
    vector<Size> nTypeI(nLayer);
    vector<Size> nTypeE(nLayer);
	vector<Size> typeAcc0;
    vector<Size> typeLayerID(1,0);
    vector<Size> typeLayerEID(1,0);
    vector<Size> typeLayerIID(1,0);
    PosInt totalType = 0;
    Size totalTypeE = 0;
    Size totalTypeI = 0;
    cout << "layer, nTypeE, nTypeI, nE, nI\n";
    for (PosInt iLayer=0; iLayer<nLayer; iLayer++) {
	    if (typeAccLayered[iLayer].back() != neuronPerBlock[iLayer]) {
            cout << "inconsistent typeAcc with neuronPerBlock\n";
            return EXIT_FAILURE;
        }
        if (nType[iLayer] == 0) typeConsistent = false;
        typeAcc0.push_back(0);
        typeAcc0.insert(typeAcc0.end(), typeAccLayered[iLayer].begin(), typeAccLayered[iLayer].end());
        totalType += nType[iLayer];
        typeLayerID.push_back(totalType);
        nTypeE[iLayer] = nTypeEI[2*iLayer];
        nTypeI[iLayer] = nTypeEI[2*iLayer+1];
        nE[iLayer] = typeAccLayered[iLayer][nTypeE[iLayer]-1];
        nI[iLayer] = neuronPerBlock[iLayer] - nE[iLayer];
        mE[iLayer] = nE[iLayer] * nblock[iLayer];
        mI[iLayer] = nI[iLayer] * nblock[iLayer];
	    cout << iLayer << ", " << nTypeE[iLayer] << ", " << nTypeI[iLayer] << ", " << nE[iLayer] << ", " << nI[iLayer] << "\n";
        totalTypeE += nTypeE[iLayer];
        typeLayerEID.push_back(totalTypeE);
        totalTypeI += nTypeI[iLayer];
        typeLayerIID.push_back(totalTypeI);
    }
    assert(totalType + nLayer == typeAcc0.size());
    assert(typeLayerID.back() == totalType);
    typeLayerID.pop_back();
    typeLayerEID.pop_back();
    typeLayerIID.pop_back();
    if (!typeConsistent) {
        cout << "# types for input/non-input layers is not consistent\n";
        return EXIT_FAILURE;
    }

    if (nTypeMat.size() != totalType*totalType) {
		cout << "nTypeMat has size of " << nTypeMat.size() << ", should be " << totalType*totalType << "\n";
		return EXIT_FAILURE;
	}

    cout << "preliminary check:\n";
	for (PosInt i=0; i<totalType; i++) {
        cout << "\n";
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
        featureValue[iF].resize(nLayer);
        for (PosInt i = 0; i<nfl; i++) {
            Size jLayer = featureLayer[iF][i];
            featureValue[iF][jLayer].resize(_n[i]);
	        fV1_feature.read(reinterpret_cast<char*>(&(featureValue[iF][jLayer][0])), _n[i] * sizeof(Float));
	        cout << iF << "th featureValue range in layer " << jLayer << ": [" << *min_element(featureValue[iF][jLayer].begin(), featureValue[iF][jLayer].end()) << ", " << *max_element(featureValue[iF][jLayer].begin(), featureValue[iF][jLayer].end()) << "]\n";
	    }
    }
	fV1_feature.close();
    for (PosInt i = 0; i<nLayer; i++) {
        for (PosInt j=0; j<featureValue[1][i].size(); j++) {
            featureValue[1][i][j] = static_cast<Float>((featureValue[1][i][j] - 0.5)*M_PI);
        }
        for (PosInt iF=0;iF<nFeature;iF++) {
            if (featureValue[iF][i].size() == 0) {
                featureValue[iF][i].assign(networkSize[i], 0);
                cout << iF <<"th feature of layer " << i << " is not set, default parameters will lead to non-specific connections\n";
            } else {
                assert(featureValue[iF][i].size() == networkSize[i]);
            }
        }
    }

    cout << "defaultFeatureParameter:\n";
    for (PosInt i=0; i<nFeature; i++) {
        cout << defaultFeatureParameter[i];
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
        if (gap_fTypeMatE.size() != totalTypeE*totalTypeE*nFeature*pPerFeature && gap_fTypeMatE.size() != 0 && gap_fTypeMatE.size() != pPerFeature*nFeature) { // gapE matrix each layer then gapE
	    	cout << "array gap_fTypeMatE have size of " << gap_fTypeMatE.size() << ", should be " << pPerFeature << "x" << nFeature << "x" << totalTypeE << "x" << totalTypeE << ", 0(defaut) or nFeature x pPerFeature (for each feature)\n";
	    	return EXIT_FAILURE;
	    } else {
            if (gap_fTypeMatE.size() == 0) {
                for (PosInt i=0; i<pPerFeature*nFeature; i++) {
                    gap_fTypeMatE.insert(gap_fTypeMatE.end(), totalTypeE*totalTypeE, defaultFeatureParameter[i]);
                }
            } else {
                if (gap_fTypeMatE.size() == pPerFeature*nFeature) {
                    vector<Float>val(gap_fTypeMatE.begin(), gap_fTypeMatE.end());
                    gap_fTypeMatE.clear();
                    for (PosInt i=0; i<pPerFeature*nFeature; i++) {
                        gap_fTypeMatE.insert(gap_fTypeMatE.end(), totalTypeE*totalTypeE, val[i]);
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
        if (gap_fTypeMatI.size() != totalTypeI*totalTypeI*nFeature*pPerFeature && gap_fTypeMatI.size() != 0 && gap_fTypeMatI.size() != pPerFeature*nFeature) { // gapE matrix each layer then gapE
	    	cout << "array gap_fTypeMatI have size of " << gap_fTypeMatI.size() << ", should be " << pPerFeature << "x" << nFeature << "x" << totalTypeI << "x" << totalTypeI << ", 0(defaut) or nFeature x pPerFeature(for each feature)\n";
	    	return EXIT_FAILURE;
	    } else {
            if (gap_fTypeMatI.size() == 0) {
                for (PosInt i=0; i<pPerFeature*nFeature; i++) {
                    gap_fTypeMatI.insert(gap_fTypeMatI.end(), totalTypeI*totalTypeI, defaultFeatureParameter[i]);
                }
            } else {
                if (gap_fTypeMatI.size() == pPerFeature*nFeature) {
                    vector<Float>val(gap_fTypeMatI.begin(), gap_fTypeMatI.end());
                    gap_fTypeMatI.clear();
                    for (PosInt i=0; i<pPerFeature*nFeature; i++) {
                        gap_fTypeMatI.insert(gap_fTypeMatI.end(), totalTypeI*totalTypeI, val[i]);
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

    if (nearNeighborBlock.size() != nLayer*nLayer || nearNeighborBlock.size() != 1) {
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

    if (maxNeighborBlock.size() != nLayer*nLayer || maxNeighborBlock.size() != 1) {
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

    if (disGauss.size() != nLayer*nLayer || disGauss.size() != 1) {
        cout << "disGauss must have a size of " << nLayer << " x " << nLayer << "(nLayer) or 1\n";
        return EXIT_FAILURE;
    } else {
        if (disGauss.size() == 1) {
            disGauss.insert(disGauss.end(), nLayer*nLayer-1, disGauss[0]);
        }
    }

	{// dendrites and axons, gap junction locs
    	if (rAxon.size() != totalType*totalType) {
    	    cout << "size of rAxon: " << rAxon.size() << " should be consistent with the number of neuronal types: " << totalType << "x" << totalType << "\n"; 
    	    return EXIT_FAILURE;
    	} else {
    	    // adjust the scale
    	    for (Float &r: rAxon){
    	        r *= dScale;
    	    }
    	}
    	if (rDend.size() != totalType*totalType) {
    	    cout << "size of rDend: " << rDend.size() << " should be consistent with the number of neuronal types: " << totalType << "x" << totalType << "\n"; 
    	    return EXIT_FAILURE;
    	} else {
    	    // adjust the scale
    	    for (Float &r: rDend){
    	        r *= dScale;
    	    }
    	}
        if (gapJunctDis.size() != totalType || gapJunctDis.size() != 1 || gapJunctDis.size() != 2) {
	    	cout << "gapJunctDis has size of " << gapJunctDis.size() << ", should be " << totalType << ", 1 or 2\n";
	    	return EXIT_FAILURE;
	    } else {
            if (gapJunctDis.size() == 1) {
                gapJunctDis.insert(gapJunctDis.end(), totalType-1, gapJunctDis[0]);
            }
            if (gapJunctDis.size() == 2) {
                vector<Float> val(gapJunctDis.begin(), gapJunctDis.end());
                gapJunctDis.clear();
                for (PosInt i=0; i<nLayer; i++) {
                    for (PosInt it=0; it<nTypeE[i]; it++) {
                        gapJunctDis.push_back(val[0]);
                    }
                    for (PosInt it=0; it<nTypeI[i]; it++) {
                        gapJunctDis.push_back(val[1]);
                    }
                }
            }
            assert(gapJunctDis.size() == totalType);
            for (PosInt i=0; i<totalType; i++) {
                if (gapJunctDis[i] < 0 || gapJunctDis[i] > *max_element(rDend.begin() + i*totalType, rDend.begin() + (i+1)*totalType)) {
	    	        cout << "entries of gapJunctDis should be smaller than the max of rDend of the same type " << i << "\n";
	    	        return EXIT_FAILURE;
                }
            }
        }
	}


    fNeighborBlock.open(neighborBlock_filename + suffix, ios::out | ios::binary);
	if (!fNeighborBlock) {
		cout << "cannot open " << neighborBlock_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fNeighborBlock.write((char*)&nLayer, sizeof(Size));
        fNeighborBlock.write((char*)&nblock[0], nLayer*sizeof(Size));
    }
    fV1_conMat.open(V1_conMat_filename + suffix, ios::out | ios::binary);
	if (!fV1_conMat) {
		cout << "cannot open " << V1_conMat_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	}
    fV1_delayMat.open(V1_delayMat_filename + suffix, ios::out | ios::binary);
	if (!fV1_delayMat) {
		cout << "cannot open " << V1_delayMat_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	}
    fV1_gapMatE.open(V1_gapMatE_filename + suffix, ios::app | ios::in | ios::out| ios::binary);
	if (!fV1_gapMatE) {
		cout << "cannot open " << V1_gapMatE_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	}
    fV1_gapMatI.open(V1_gapMatI_filename + suffix, ios::app | ios::in | ios::out| ios::binary);
	if (!fV1_gapMatI) {
		cout << "cannot open " << V1_gapMatI_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	}

    fV1_vec.open(V1_vec_filename + suffix, ios::out | ios::binary);
	if (!fV1_vec) {
		cout << "cannot open " << V1_vec_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fV1_vec.write((char*)&nLayer, sizeof(Size));
        fV1_vec.write((char*)&(networkSize[0]), nLayer*sizeof(Size));
    }

    fBlock_pos.open(block_pos_filename + suffix, ios::out | ios::binary);
	if (!fBlock_pos) {
		cout << "cannot open " << block_pos_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fBlock_pos.write((char*)&nLayer, sizeof(Size));
        fBlock_pos.write((char*)&nblock[0], nLayer*sizeof(Size));
    }

    fStats.open(stats_filename + suffix, ios::out | ios::binary);
	if (!fStats) {
		cout << "cannot open " << stats_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fStats.write((char*)&nType, nLayer*sizeof(Size));
        fStats.write((char*)&networkSize, nLayer*sizeof(Size));
    }

    fGapStats.open(gapStats_filename + suffix, ios::app | ios::in | ios::out | ios::binary);
	if (!fGapStats) {
		cout << "cannot open " << gapStats_filename + suffix << " to write.\n";
		return EXIT_FAILURE;
	} else {
        fGapStats.write((char*)&nTypeE[0], nLayer*sizeof(Size));
        fGapStats.write((char*)&mE[0], nLayer*sizeof(Size));
        fGapStats.write((char*)&nTypeI[0], nLayer*sizeof(Size));
        fGapStats.write((char*)&mI[0], nLayer*sizeof(Size));
    }

    size_t sizeof_block = 0;
    size_t sizeof_neighbor = 0;
    vector<PosInt> blockAcc(nLayer+1, 0);
    vector<PosInt> neighborBlockAcc(nLayer+1, 0);
    vector<PosInt> networkSizeAcc(nLayer+1, 0);
    for (PosInt i=0; i<nLayer; i++) {
        sizeof_block += 3*nblock[i]*sizeof(Float); // size of block_pos
        blockAcc[i+1] = blockAcc[i] + nblock[i];
        networkSizeAcc[i+1] = networkSizeAcc[i] + networkSize[i];
        Size qMaxNeighborBlock = 0;
        for (PosInt j=0; j<nLayer; j++) {
            sizeof_neighbor += nblock[i]*maxNeighborBlock[j*nLayer+i]*sizeof(PosInt); // size of neighborBlockId
            qMaxNeighborBlock += maxNeighborBlock[j*nLayer+i];
        }
        neighborBlockAcc[i+1] = neighborBlockAcc[i] + nblock[i]*qMaxNeighborBlock;
    }
    sizeof_neighbor += 4*nLayer*blockAcc.back()*sizeof(Size); // n(Near,Gap)NeighborBlock(E/I)
    size_t sizeof_pos = 2*networkSizeAcc.back()*sizeof(Float);
    size_t morph_size = totalType*totalType*2*sizeof(Float) + totalType*sizeof(Float);
    size_t sizeof_block_chunk = sizeof_block + sizeof_neighbor + sizeof_pos + 2*(nLayer+1)*sizeof(PosInt) + morph_size;
    void* __restrict__ block_chunk;
    checkCudaErrors(cudaMalloc((void**)&block_chunk, sizeof_block_chunk));
    Float*  d_pos = (Float*) block_chunk;
    Float*  d_block_x = d_pos + 2*networkSizeAcc.back();
    Float*  d_block_y = d_block_x + blockAcc.back();
    Float*  d_block_r = d_block_y + blockAcc.back();
    PosInt* d_neighborBlockId = (Size*) (d_block_r + blockAcc.back());
    Size*   d_nNeighborBlock = d_neighborBlockId + neighborBlockAcc.back();
	Size*   d_nNearNeighborBlock = d_nNeighborBlock + nLayer*blockAcc.back();
	Size*   d_nGapNeighborBlockE = d_nNearNeighborBlock + nLayer*blockAcc.back();
	Size*   d_nGapNeighborBlockI = d_nGapNeighborBlockE + nLayer*blockAcc.back();
    // accumulative
    PosInt* d_blockAcc = d_nNearNeighborBlock + nLayer*blockAcc.back();
    PosInt* d_networkSizeAcc = d_blockAcc + nLayer+1; 
    // morph
    Float* rden = (Float*) (d_networkSizeAcc + nLayer+1); 
    Float* raxn = rden + totalType*totalType;
	Float* d_gapDis = raxn + totalType*totalType;

	cudaStream_t *layerStream = new cudaStream_t[nLayer];
    for (PosInt i=0; i<nLayer; i++) {
		checkCudaErrors(cudaStreamCreate(&layerStream[i]));
    }
        
    for (PosInt i=0; i<nLayer; i++) {
        checkCudaErrors(cudaMemcpyAsync(d_pos+2*networkSizeAcc[i], &(pos[i][0]), 2*networkSize[i]*sizeof(Float), cudaMemcpyHostToDevice, layerStream[i]));
        cal_blockPos<<<nblock[i], neuronPerBlock[i],0,layerStream[i]>>>(
            d_pos + 2*networkSizeAcc[i], 
	    	d_block_x + blockAcc[i], d_block_y + blockAcc[i], d_block_r + blockAcc[i],
	    	networkSize[i]);
	    getLastCudaError("cal_blockPos failed");
    }
    printf("block centers calculated\n");

    Float* block_pos = new Float[blockAcc.back()*3];
    Float* block_x = block_pos;
    Float* block_y = block_x + blockAcc.back();
    Float* block_r = block_y + blockAcc.back();
	checkCudaErrors(cudaMemcpy(block_pos, d_block_x, sizeof_block, cudaMemcpyDeviceToHost)); 	
    fBlock_pos.write((char*)block_pos, 3*blockAcc.back()*sizeof(Float));
    fBlock_pos.close();
    delete []block_pos;

    checkCudaErrors(cudaMemcpy(d_blockAcc, &(blockAcc[0]), (nLayer+1)*sizeof(PosInt), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_networkSizeAcc, &(networkSizeAcc[0]), (nLayer+1)*sizeof(PosInt), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(rden, &(rDend[0]), (totalType*totalType)*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(raxn, &(rAxon[0]), (totalType*totalType)*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_gapDis, &(gapJunctDis[0]), totalType*sizeof(Float), cudaMemcpyHostToDevice));

    for (PosInt i=0; i<nLayer; i++) {
        Size qMaxNeighborBlock = 0;
        Float postGapJunctDisE = *max_element(gapJunctDis.begin() + typeLayerID[i], gapJunctDis.begin() + typeLayerID[i] + nTypeE[i]);
        Float postGapJunctDisI = *max_element(gapJunctDis.begin() + typeLayerID[i] + nTypeE[i], gapJunctDis.begin() + typeLayerID[i] + nType[i]);
        for (PosInt j=0; j<nLayer; j++) {
            Float preGapJunctDisE = *max_element(gapJunctDis.begin() + typeLayerID[j], gapJunctDis.begin() + typeLayerID[i] + nTypeE[j]);
            Float preGapJunctDisI = *max_element(gapJunctDis.begin() + typeLayerID[j] + nTypeE[j], gapJunctDis.begin() + typeLayerID[j] + nType[j]);
            Float max_rden = *max_element(rDend.begin() + i*totalType+typeLayerID[j], rDend.begin() + i*totalType+typeLayerID[j] + nType[j]);
            Float max_raxn = *max_element(rAxon.begin() + j*totalType+typeLayerID[i], rAxon.begin() + j*totalType+typeLayerID[i] + nType[i]);
            // blocks -> blocks, threads -> cal neighbor blocks
            get_neighbor_blockId<<<nblock[i], blockSize, maxNeighborBlock[j*nLayer+i]*sizeof(PosInt) + (2*sizeof(Float) + sizeof(Int))*nblock[j],layerStream[j]>>>(
                d_block_x, d_block_y, d_block_r + blockAcc[j], d_blockAcc,
	    	    d_neighborBlockId + neighborBlockAcc[i] + qMaxNeighborBlock,
                d_nNeighborBlock + nLayer*blockAcc[i] + j*nblock[i],
                d_nNearNeighborBlock + nLayer*blockAcc[i] + j*nblock[i],
                d_nGapNeighborBlockE + nLayer*blockAcc[i] + j*nblock[i],
                d_nGapNeighborBlockI + nLayer*blockAcc[i] + j*nblock[i],
	    	    nblock[j], max_rden, max_raxn, maxNeighborBlock[j*nLayer+i], postGapJunctDisE, preGapJunctDisE, postGapJunctDisI, preGapJunctDisI, i, j);
	        getLastCudaError("get_neighbor_blockId failed");
            qMaxNeighborBlock += nblock[i] * maxNeighborBlock[j*nLayer+i];
        }
    }
    
    printf("neighbor blocks acquired\n");
    
    // n(Near/Max)NeighborBlock: [recv_layer, send_layer, recv_nblock]
    // neighborBlockId: [recv_layer, send_layer, recv_nblock]
    void* neighbors = malloc(sizeof_neighbor);
    PosInt* neighborBlockId = (PosInt*) neighbors;
    Size* nNeighborBlock = (Size*) (neighborBlockId + neighborBlockAcc.back());
    Size* nNearNeighborBlock = nNeighborBlock + nLayer*blockAcc.back();
    Size* nGapNeighborBlockE = nNearNeighborBlock + nLayer*blockAcc.back();
    Size* nGapNeighborBlockI = nGapNeighborBlockE + nLayer*blockAcc.back();
    checkCudaErrors(cudaMemcpy(neighborBlockId, d_neighborBlockId, sizeof_neighbor, cudaMemcpyDeviceToHost));

    vector<Size> gapNeighborBlockE(nLayer*nLayer);
    vector<Size> gapNeighborBlockI(nLayer*nLayer);
    vector<Size> maxNeighborBlockNew(nLayer*nLayer);
    for (PosInt i=0; i<nLayer; i++) {
        for (PosInt j=0; j<nLayer; j++) {
            maxNeighborBlockNew[j*nLayer+i] = *max_element(nNeighborBlock+nLayer*blockAcc[i] + j*nblock[i], nNeighborBlock+nLayer*blockAcc[i] + (j+1)*nblock[i]);
            nearNeighborBlock[j*nLayer+i] = *max_element(nNearNeighborBlock+nLayer*blockAcc[i] + j*nblock[i], nNearNeighborBlock+nLayer*blockAcc[i] + (j+1)*nblock[i]);
            gapNeighborBlockE[j*nLayer+i] = *max_element(nGapNeighborBlockE+nLayer*blockAcc[i] + j*nblock[i], nGapNeighborBlockE+nLayer*blockAcc[i] + (j+1)*nblock[i]);
            gapNeighborBlockI[j*nLayer+i] = *max_element(nGapNeighborBlockI+nLayer*blockAcc[i] + j*nblock[i], nGapNeighborBlockI+nLayer*blockAcc[i] + (j+1)*nblock[i]);
            assert (maxNeighborBlockNew[j*nLayer+i] < maxNeighborBlock[j*nLayer+i]);
            assert (nearNeighborBlock[j*nLayer+i] < maxNeighborBlockNew[j*nLayer+i]);
            assert (gapNeighborBlockE[j*nLayer+i] < nearNeighborBlock[j*nLayer+i]);
            assert (gapNeighborBlockI[j*nLayer+i] < nearNeighborBlock[j*nLayer+i]);
        }
    }

    // rebuild Accs after maxima recalibration
    vector<size_t> gapMatAccE(nLayer+1, 0);
    vector<size_t> gapMatAccI(nLayer+1, 0);
    for (PosInt i=0; i<nLayer; i++) {
        Size qGapNeighborBlockE = 0;
        Size qGapNeighborBlockI = 0;
        for (PosInt j=0; j<nLayer; j++) {
            qGapNeighborBlockE += gapNeighborBlockE[j*nLayer+i]*nE[j];
            qGapNeighborBlockI += gapNeighborBlockI[j*nLayer+i]*nI[j];
        }
        gapMatAccE[i+1] = gapMatAccE[i] + static_cast<size_t>(nblock[i]*nE[i]*qGapNeighborBlockE);
        gapMatAccI[i+1] = gapMatAccI[i] + static_cast<size_t>(nblock[i]*nI[i]*qGapNeighborBlockI);
    }

    cout << "# neighbors (self-included) from ";
    for (PosInt i=0; i<nLayer; i++) {
        cout << "layer " << i;
        if (i == nLayer-1) cout << "\n";
        else cout << ", ";
    }

    for (PosInt i=0; i<nLayer; i++) {
        cout << " layer " << i << " <- (";
        for (PosInt j=0; j<nLayer; j++) {
            Float mean_neighbor = 0.0;
            for (PosInt k=0; k<nblock[i]; k++) {
                mean_neighbor += nNeighborBlock[nLayer*blockAcc[i] + j*nblock[i] + k];
            }
            cout << mean_neighbor/nblock[i];
            if (j == nLayer-1) cout << "\n";
            else cout << ", ";
        }
    }
    cout << "\n";
    fV1_gapMatI.write((char*) &gapNeighborBlockI[0], nLayer*nLayer*sizeof(Size));
    fV1_gapMatE.write((char*) &gapNeighborBlockE[0], nLayer*nLayer*sizeof(Size));
    fV1_conMat.write((char*) &nearNeighborBlock[0], nLayer*nLayer*sizeof(Size));
    fV1_delayMat.write((char*) &nearNeighborBlock[0], nLayer*nLayer*sizeof(Size));

    fNeighborBlock.write((char*)&nNearNeighborBlock[0], nLayer*blockAcc.back()*sizeof(Size));
    fNeighborBlock.write((char*)&nNeighborBlock[0], nLayer*blockAcc.back()*sizeof(Size));
    fNeighborBlock.write((char*)&neighborBlockId, neighborBlockAcc.back()*sizeof(PosInt));
    fNeighborBlock.close();
	cout << "neighbors written\n";
    free(neighbors);

    if (inhRatio.size() != totalType && inhRatio.size() != 1) {
        cout << "size of inhRatio: " << inhRatio.size() << " should be consistent with the number of neuronal types: " << totalType << " or 1\n";
        return EXIT_FAILURE;
    } else {
        if (inhRatio.size() == 1) {
            inhRatio.insert(inhRatio.end(), totalType-1, inhRatio[0]);
        }
    }

    if (synOccupyRatioE.size() != totalTypeE && synOccupyRatioE.size() != 1) {
        cout << "size of synOccupyRatioE: " << synOccupyRatioE.size() << " should be consistent with the number of exc neuronal types: " << totalTypeE << " or 1\n";
        return EXIT_FAILURE;
    } else {
        if (synOccupyRatioE.size() == 1) {
            synOccupyRatioE.insert(synOccupyRatioE.end(), totalTypeE-1, synOccupyRatioE[0]);
        }
    }

    if (ffSynOccupyRatio.size() != mLayer && ffSynOccupyRatio.size() != 1) {
        cout << "size of ffSynOccupyRatio: " << ffSynOccupyRatio.size() << " should be consistent with the number of LGN types: " << mLayer << " or 1\n";
        return EXIT_FAILURE;
    } else {
        if (ffSynOccupyRatio.size() == 1) {
            ffSynOccupyRatio.insert(ffSynOccupyRatio.end(), mLayer-1, ffSynOccupyRatio[0]);
        }
    }

    initializePreferenceFunctions(nFeature);
    //Float speedOfThought = 1.0f; specify instead in patch.cu through patchV1.cfg

    void* __restrict__ preset_chunk;

    size_t typeMatSize = (totalType*totalType + totalTypeE*totalTypeE + totalTypeI*totalTypeI)*(sizeof(Float) + sizeof(Size)) + pPerFeature*nFeature*(totalType*totalType + totalTypeE*totalTypeE + totalTypeI*totalTypeI)*sizeof(Float);
    // inits: ffRatio, featureValue, typeAcc0, nTypeEI, nType, nTypeE, nTypeI, typeAcc
    size_t init_size = networkSizeAcc.back()*sizeof(Float) + 2*totalType*sizeof(Float) + (totalType+nLayer)*sizeof(Size) + 5*nLayer*sizeof(Size) + nLayer*sizeof(Size*) + totalType*(sizeof(Float) + sizeof(PosInt)) + nFeature*networkSizeAcc.back()*sizeof(Float); 
    checkCudaErrors(cudaMalloc((void**)&preset_chunk, typeMatSize + init_size));
    // mats
    Size*  d_nTypeMat = (Size*) preset_chunk;
    Float* d_fTypeMat = (Float*) (d_nTypeMat + totalType*totalType);

    Size*  d_gap_nTypeMatE = (Size*) (d_fTypeMat + pPerFeature*nFeature*totalType*totalType);
    Float* d_gap_fTypeMatE = (Float*) (d_gap_nTypeMatE + totalTypeE*totalTypeE);

    Size*  d_gap_nTypeMatI = (Size*) (d_gap_fTypeMatE + pPerFeature*nFeature*totalTypeE*totalTypeE);
    Float* d_gap_fTypeMatI = (Float*) (d_gap_nTypeMatI + totalTypeI*totalTypeI);

    // inits
	Float* d_ffRatio = d_gap_fTypeMatI + pPerFeature*nFeature*totalTypeI*totalTypeI;
	Float* d_feature = d_ffRatio + networkSizeAcc.back();
	Float* d_inhRatio = d_feature + nFeature*networkSizeAcc.back();
    Size** d_typeAcc = new Size*[nLayer]; 
	d_typeAcc[0] = (Size*) (d_inhRatio + totalType);
    for (PosInt i=1; i<nLayer; i++) {
        d_typeAcc[i] = d_typeAcc[i-1] + 1 + nType[i-1];
    }
	Size* d_nTypeEI =  d_typeAcc[nLayer-1] + 1 + nType[nLayer-1];
	Size* d_nType = d_nTypeEI + 2*nLayer;
	Size* d_nTypeE = d_nType + nLayer;
	Size* d_nTypeI = d_nTypeE + nLayer;
    Size** dd_typeAcc = (Size**) (d_nTypeI + nLayer);
    Float* d_maxCortExc = (Float*) (dd_typeAcc + nLayer);

    Float* maxCortExc = new Float[totalType]{};
    for (PosInt k=0; k<totalType; k++) {
        PosInt iq = 0;
        PosInt iqE = 0;
        for (PosInt i=0; i<nLayer; i++) {
            for (PosInt j=0; j<nTypeEI[2*i]; j++) { // exc only
                maxCortExc[k] += nTypeMat[(iq + j)*totalType + k]*synOccupyRatioE[iqE + j];
            }
            iq += nType[i];
            iqE += nTypeEI[2*i];
        }
    }

	Float* nLGN_eff = new Float[networkSizeAcc.back()]{};
    read_LGN_V1(LGN_V1_s_filename + conLGN_suffix, LGN_V1_ID_filename + conLGN_suffix, nLGN_eff, inputLayer, networkSizeAcc, ffSynOccupyRatio, mL, mR, mLayer, false);

    Float* meanFF = new Float[totalType]{};
    Float* maxFF = new Float[totalType]{};
    Float* minFF = new Float[totalType]{};
    qType = 0;
    for (PosInt i=0; i<nLayer; i++) {
        for (PosInt j=0; j<nblock[i]; j++) {
            PosInt iType = 0;
            for (PosInt k=0; k<neuronPerBlock[i]; k++) {
                if (k > typeAccLayered[i][iType+1]) {
                    iType++;
                }
                PosInt id = networkSizeAcc[i] + j*neuronPerBlock[i] + k;
                Float FF = nLGN_eff[id];
                meanFF[qType + iType] += FF;
                if (maxFF[qType + iType] < FF) {
                    maxFF[qType + iType] = FF;
                }
                if (minFF[qType + iType] > FF) {
                    minFF[qType + iType] = FF;
                }
            }
        }
        for (PosInt j=0; j<nType[i]; j++) {
            meanFF[j] /= (typeAccLayered[i][j+1] - typeAccLayered[i][j])*nblock[i];
        }
        qType += nType[i];
    }

    vector<bool> inhType(totalType, false);
    vector<bool> excType(totalType, false);
    qType = 0;
    for (PosInt i=0; i<nLayer; i++) {
        for (PosInt j=0; j<nTypeEI[2*i]; j++) {
            excType[qType] = true;
            qType++;
        }
        for (PosInt j=0; j<nTypeEI[2*i+1]; j++) {
            inhType[qType] = true;
            qType++;
        }
    }
    for (PosInt i=0; i<totalType; i++) {
        if (maxCortExc[i] < maxFF[i]) {
            cout << "max FF excitation need to be smaller than max cortical excitation, type " << i << " conflicted.\n";
            return EXIT_FAILURE;
        }
    }
    	
    checkCudaErrors(cudaMemcpy(d_typeAcc[0], &(typeAcc0[0]), (totalType+nLayer)*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dd_typeAcc, d_typeAcc, nLayer*sizeof(Size*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_nType, &(nType[0]), (nLayer)*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_nTypeEI, &(nTypeEI[0]), (2*nLayer)*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_nTypeE, &(nTypeE[0]), nLayer*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_nTypeI, &(nTypeI[0]), nLayer*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_inhRatio, &(inhRatio[0]), totalType*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_maxCortExc, maxCortExc, totalType*sizeof(Float), cudaMemcpyHostToDevice));

	Float* ffRatio = new Float[networkSizeAcc.back()];

    void* LGN_chunk;
    size_t LGN_size = networkSizeAcc.back()*sizeof(Float) + totalType*sizeof(Float);
    checkCudaErrors(cudaMalloc((void**)&LGN_chunk, LGN_size));
    Float* d_nLGN_eff = (Float*) LGN_chunk;
    checkCudaErrors(cudaMemcpy(d_nLGN_eff, nLGN_eff, networkSizeAcc.back()*sizeof(Float), cudaMemcpyHostToDevice));

    for (PosInt iLayer=0; iLayer<nLayer; iLayer++) {
        if (inputLayerPick[iLayer] == 0) {
            for (PosInt i=0; i<networkSize[iLayer]; i++) {
                ffRatio[networkSizeAcc[iLayer] + i] = 0;
            }
            checkCudaErrors(cudaMemcpyAsync(d_ffRatio + networkSizeAcc[iLayer], ffRatio + networkSizeAcc[iLayer], networkSize[iLayer]*sizeof(Float), cudaMemcpyHostToDevice, layerStream[iLayer]));
        } else {
            initialize<<<nblock[iLayer], neuronPerBlock[iLayer], 0, layerStream[iLayer]>>>(d_nLGN_eff + networkSizeAcc[iLayer], d_ffRatio + networkSizeAcc[iLayer], d_typeAcc[iLayer], d_maxCortExc, nType[iLayer]);
	        getLastCudaError("initialize failed");
            checkCudaErrors(cudaMemcpyAsync(ffRatio + networkSizeAcc[iLayer], d_ffRatio + networkSizeAcc[iLayer], networkSize[iLayer]*sizeof(Float), cudaMemcpyDeviceToHost, layerStream[iLayer]));
        }
    }
    printf("initialzied\n");
    
    checkCudaErrors(cudaMemcpy(d_nTypeMat, &(nTypeMat[0]), (totalType*totalType)*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fTypeMat, &(typeFeatureMat[0]), (nFeature*pPerFeature*totalType*totalType)*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_gap_nTypeMatE, &(gap_nTypeMatE[0]), (totalTypeE*totalTypeE)*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_gap_fTypeMatE, &(gap_fTypeMatE[0]), (nFeature*pPerFeature*totalTypeE*totalTypeE)*sizeof(Float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_gap_nTypeMatI, &(gap_nTypeMatI[0]), (totalTypeI*totalTypeI)*sizeof(Size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_gap_fTypeMatI, &(gap_fTypeMatI[0]), (nFeature*pPerFeature*totalTypeI*totalTypeI)*sizeof(Float), cudaMemcpyHostToDevice));
    for (PosInt iF=0; iF<nFeature; iF++) {
        for (PosInt i=0; i<nLayer; i++) {
            checkCudaErrors(cudaMemcpy(d_feature + iF*networkSizeAcc.back() + networkSizeAcc[i], &(featureValue[iF][i][0]), networkSize[i]*sizeof(Float), cudaMemcpyHostToDevice));
        }
    }
    

    size_t used_mem = sizeof_block_chunk + LGN_size + morph_size + typeMatSize + init_size;
    size_t availableMem0 = deviceProps.totalGlobalMem - used_mem;
    for (PosInt i = 0; i<nLayer; i++) {
        // deviceOnly
        curandStateMRG32k3a* state;
        size_t randSize = sizeof(curandStateMRG32k3a)*networkSize[i];
        checkCudaErrors(cudaMalloc((void**)&state, randSize));

        //TODO, low priority: generate conMat concurrently
        Size current_nblock;
        Size maxChunkSize = nblock[i];
        PosInt qMaxNeighborBlock = 0; // neighborBlock offset
        for (PosInt j = 0; j<nLayer; j++) {
            void *gpu_ij;
            size_t vecSize = nType[j]*sizeof(Size) + static_cast<size_t>(maxDistantNeighbor[j*nLayer+i])*networkSize[i]*(2*sizeof(Float) + sizeof(Size)) + networkSize[i]*sizeof(Size); // con, delayVec, vecID and nVec
            size_t statSize = nType[j]*networkSize[i]*(2*sizeof(Size)/*preTypeConnected and Avail*/ +sizeof(Float)/*preTypeEffSum*/);
            size_t gap_statSize = (nTypeE[j]*mE[i] + nTypeI[j]*mI[i])*(2*sizeof(Size) + sizeof(Float)); // preTypeGapE/I, preTypeStrGapE/I
            checkCudaErrors(cudaMalloc((void**)&gpu_ij, vecSize + statSize + gap_statSize));
            // vec
	        Size*   d_max_N = (Size*) gpu_ij;
            Float*  d_conVec = (Float*) (d_max_N + nType[j]); 
            Float*  d_delayVec = d_conVec + static_cast<size_t>(maxDistantNeighbor[j*nLayer+i])*networkSize[i];
            Size*   d_vecID = (Size*) (d_delayVec + static_cast<size_t>(maxDistantNeighbor[j*nLayer+i])*networkSize[i]);
            Size*   d_nVec = d_vecID + static_cast<size_t>(maxDistantNeighbor[j*nLayer+i])*networkSize[i];
            // stats
            Size*   d_preTypeConnected = d_nVec + networkSize[i];
            Size*   d_preTypeAvail = d_preTypeConnected + nType[j]*networkSize[i];
            Float*  d_preTypeStrSum = (Float*) (d_preTypeAvail + nType[j]*networkSize[i]);
	        // gapStats
            Size*   d_preTypeGapE = (Size*) (d_preTypeStrSum + nType[j]*networkSize[i]);
            Size*   d_preTypeAvailGapE = d_preTypeGapE + nTypeE[j]*mE[i];
            Float*  d_preTypeStrGapE = (Float*) (d_preTypeAvailGapE + nTypeE[j]*mE[i]);
            Size*   d_preTypeGapI = (Size*) (d_preTypeStrGapE + nTypeE[j]*mE[i]);
            Size*   d_preTypeAvailGapI = d_preTypeGapI + nTypeI[j]*mI[i];
            Float*  d_preTypeStrGapI = (Float*) (d_preTypeAvailGapI + nTypeI[j]*mI[i]);
            //
	        assert(static_cast<void*>((char*)gpu_ij + vecSize + statSize + gap_statSize) == static_cast<void*>(d_preTypeStrGapI + nTypeI[j]*mI[i]));

            void *cpu_ij = malloc(vecSize + statSize + gap_statSize);
            // vec
            Size*   max_N = (Size*) cpu_ij;
            Float*  conVec = (Float*) (max_N + nType[j]); 
            Float*  delayVec = conVec + static_cast<size_t>(maxDistantNeighbor[j*nLayer+i])*networkSize[i];
            Size*   vecID = (Size*) (delayVec + static_cast<size_t>(maxDistantNeighbor[j*nLayer+i])*networkSize[i]);
            Size*   nVec = vecID + static_cast<size_t>(maxDistantNeighbor[j*nLayer+i])*networkSize[i];
            // stats
            Size*   preTypeConnected = nVec + networkSize[i];
            Size*   preTypeAvail = preTypeConnected + nType[j]*networkSize[i];
            Float*  preTypeStrSum = (Float*) (preTypeAvail + nType[j]*networkSize[i]);
	        // gapSts
            Size*   preTypeGapE = (Size*) (preTypeStrSum + nType[j]*networkSize[i]);
            Size*   preTypeAvailGapE = preTypeGapE + nTypeE[j]*mE[i];
            Float*  preTypeStrGapE = (Float*) (preTypeAvailGapE + nTypeE[j]*mE[i]);
            Size*   preTypeGapI = (Size*) (preTypeStrGapE + nTypeE[j]*mE[i]);
            Size*   preTypeAvailGapI = preTypeGapI + nTypeI[j]*mI[i];
            Float*  preTypeStrGapI = (Float*) (preTypeAvailGapI + nTypeI[j]*mI[i]);
            //
	        assert(static_cast<void*>((char*)cpu_ij + vecSize + statSize + gap_statSize) == static_cast<void*>(preTypeStrGapI + nTypeI[j]*mI[i]));
            size_t availableMem = availableMem0 - randSize - vecSize - statSize - gap_statSize;

            void *gpu_ijChunk, *cpu_ijChunk;
            size_t tmpVecSize, disNeighborSize, matSize, gapMatSize;
            size_t d_memorySize, memorySize, deviceOnlyMemSize, localHeapSize; 

	        Size sum_max_N = 0;
	        for (PosInt jt=0; jt<nType[j]; jt++) {
	        	max_N[jt] = 0;
	        	for (PosInt it=0; it<nType[i]; it++) {
                    Size nmax = nTypeMat[(typeLayerID[j]+jt)*totalType+typeLayerID[i]+it];
	        		if (nmax > max_N[jt]) {
	        			max_N[jt] = nmax;
	        		}
	        	}
	        	sum_max_N += max_N[jt];
	        }

            checkCudaErrors(cudaMemcpy(d_max_N, max_N, nType[j]*sizeof(Size), cudaMemcpyHostToDevice));
	        delete []max_N;

            Int half = 1;
	        do { 
                //half *= 2;
                if (half > 1) {
                    Size half0 = maxChunkSize/half;
                    Size half1 = maxChunkSize - half0;
                    maxChunkSize = (half0 > half1)? half0: half1;
                }
                matSize = static_cast<size_t>(2*nearNeighborBlock[j*nLayer+i]*neuronPerBlock[i])*neuronPerBlock[j]*maxChunkSize*sizeof(Float); // con and delayMat

                gapMatSize = static_cast<size_t>(gapNeighborBlockE[j*nLayer+i]*nE[j])*maxChunkSize*nE[i]*sizeof(Float)
                           + static_cast<size_t>(gapNeighborBlockI[j*nLayer+i]*nI[j])*maxChunkSize*nI[i]*sizeof(Float);
                memorySize = matSize + gapMatSize;

                disNeighborSize = static_cast<size_t>((maxNeighborBlockNew[j*nLayer+i]-nearNeighborBlock[j*nLayer+i])*neuronPerBlock[j])*maxChunkSize*neuronPerBlock[i]*sizeof(Float);
	            tmpVecSize = static_cast<size_t>(maxChunkSize*neuronPerBlock[i])*sizeof(Size) * sum_max_N; // tmp_vecID
                deviceOnlyMemSize = tmpVecSize + disNeighborSize;

	        	// nType: sumP, availType, sumType, sumStrType, pN, __vecID, nid
	        	// nTypeI: ...
                Size max_gapType = (nTypeE[j]>nTypeI[j]?nTypeE[j]:nTypeI[j]);
                localHeapSize = (nType[j] + max_gapType) * (2*sizeof(Size) + sizeof(Float)) + nFeature*sizeof(Float) + nType[j]*(sizeof(bool) + sizeof(PosInt*)) + nType[j]*(2*sizeof(Size) + 2*sizeof(Float) + sizeof(PosInt)) + max_gapType*(sizeof(Size) + 2*sizeof(Float));
                localHeapSize *= static_cast<size_t>(maxChunkSize*neuronPerBlock[i]*deviceProps.multiProcessorCount);
                checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, localHeapSize));
                d_memorySize = memorySize + deviceOnlyMemSize;

                if (half > 2) {
                    free(cpu_ijChunk);
                }
	            cpu_ijChunk = malloc(memorySize);
                if (cpu_ijChunk == NULL) cout << "RAM not enough?!\n";
                half *= 2;
                cout << localHeapSize/1024/1024 << "Mb heap size\n";
                cout << memorySize/1024/1024 << "Mb cpu mem\n";
                cout << d_memorySize/1024/1024 << "Mb gpu mem request\n";
                cout << availableMem/1024/1024 << "Mb gpu mem available\n";
            } while ((cpu_ijChunk == NULL || d_memorySize + localHeapSize > availableMem) || nblock[i] <= 1);
            
            if (cpu_ijChunk == NULL) {
                cout << "RAM not enough even for one block, layer " << j << "->" << i << "\n";
            }
            if (d_memorySize + localHeapSize > availableMem) {
                cout << "global gpu memory not enough even for one block, layer " << j << "->" << i << "\n";
            }
            size_t shared_size = (nType[i]*nType[j] + nTypeE[i]*nTypeE[j] + nTypeI[i]*nTypeI[j])*sizeof(Size) /* nMats */
                                + nFeature*pPerFeature*(nType[i]*nType[j] + nTypeE[i]*nTypeE[j] + nTypeI[i]*nTypeI[j])*sizeof(Float) /* fMats */
                                +(2+nFeature)*neuronPerBlock[j]*sizeof(Float) /* x1, y1 and pre_feat */
                                + nType[i]*nType[j]*sizeof(Float) /* ra */
                                + nType[j]*(sizeof(Float) + sizeof(PosInt)); /*typeAcc and gapDis */
            if (shared_size > deviceProps.sharedMemPerBlock) {
                cout << "shared memory not enough for layer " << j << "->" << i << "\n";
            }

            Size nChunk = (nblock[i] + maxChunkSize-1) /maxChunkSize - 1;
            cout << "nChunk = " << nChunk+1 << ", with maxChunkSize: " << maxChunkSize << " in total " << nblock[i] << " blocks.\n";
            Size remainChunkSize = nblock[i]%maxChunkSize;
	        if (remainChunkSize == 0) {
	        	remainChunkSize = maxChunkSize;
	        }
            assert(maxChunkSize * nChunk + remainChunkSize == nblock[i]);
	        printf("need to allocate %f MB memory on host\n", static_cast<float>(memorySize)/1024/1024);
            // ========= set the gpu mem

            checkCudaErrors(cudaMalloc((void**)&gpu_ijChunk, d_memorySize));
            // deviceOnly
            PosInt* tmp_vecID = (PosInt*) gpu_ijChunk;
            Float*  disNeighborP = (Float*) (tmp_vecID + static_cast<size_t>(maxChunkSize*neuronPerBlock[i]) * sum_max_N);
	        // mat
            Float*  d_conMat = disNeighborP + disNeighborSize/sizeof(Float);
            Float*  d_gapMatE = d_conMat + static_cast<size_t>(nearNeighborBlock[j*nLayer+i]*neuronPerBlock[i])*neuronPerBlock[j]*maxChunkSize;
            Float*  d_gapMatI = d_gapMatE + static_cast<size_t>(gapNeighborBlockE[j*nLayer+i]*nE[i])*nE[j]*maxChunkSize;
            Float*  d_delayMat = d_gapMatI + static_cast<size_t>(gapNeighborBlockI[j*nLayer+i]*nI[i])*nI[j]*maxChunkSize;
	        // check
	        assert(static_cast<void*>((char*)gpu_ijChunk + d_memorySize) == static_cast<void*>(d_delayMat + static_cast<size_t>(nearNeighborBlock[j*nLayer+i]*neuronPerBlock[i])*neuronPerBlock[j]*maxChunkSize));

            // ========= set the cpu mem

            Float*  conMat = (Float*) cpu_ijChunk;
            Float*  gapMatE = conMat + static_cast<size_t>(nearNeighborBlock[j*nLayer+i]*neuronPerBlock[i])*neuronPerBlock[j]*maxChunkSize;
            Float*  gapMatI = gapMatE + static_cast<size_t>(gapNeighborBlockE[j*nLayer+i]*nE[i])*nE[j]*maxChunkSize;
            Float*  delayMat = gapMatI + static_cast<size_t>(gapNeighborBlockI[j*nLayer+i]*nI[i])*nI[j]*maxChunkSize;
	                    
            // check
	        assert(static_cast<void*>((char*)cpu_ijChunk + memorySize) == static_cast<void*>(delayMat + static_cast<size_t>(nearNeighborBlock[j*nLayer+i]*neuronPerBlock[i])*neuronPerBlock[j]*maxChunkSize));

            PosInt offset = 0; // memory offset
            for (PosInt iChunk = 0; iChunk < nChunk+1; iChunk++) {
                if (iChunk < nChunk) current_nblock = maxChunkSize;
                else current_nblock = remainChunkSize;
                cout << iChunk << ": current_nblock = " << current_nblock << "\n";
                // initialize for each chunk
                size_t current_matSize = static_cast<size_t>(current_nblock*nearNeighborBlock[j*nLayer+i]*neuronPerBlock[i])*neuronPerBlock[j]*sizeof(Float);
	            checkCudaErrors(cudaMemset(d_conMat, 0, current_matSize)); 
	        	size_t gap_matSizeE = static_cast<size_t>(current_nblock*gapNeighborBlockE[j*nLayer+i]*nE[i])*nE[j]*sizeof(Float);
	            checkCudaErrors(cudaMemset(d_gapMatE, 0, gap_matSizeE));
	        	size_t gap_matSizeI = static_cast<size_t>(current_nblock*gapNeighborBlockI[j*nLayer+i]*nI[i])*nI[j]*sizeof(Float);
	            checkCudaErrors(cudaMemset(d_gapMatI, 0, gap_matSizeI));

	        	cout << "generate_connections<<<" << current_nblock << ", " << neuronPerBlock[i] << ">>>\n";
                PosInt preOffset = nLayer*blockAcc[i] + j*nblock[i] + offset;
                PosInt vecOffset = maxDistantNeighbor[j*nLayer+i]*neuronPerBlock[i]*offset;
                generate_connections<<<current_nblock, neuronPerBlock[i], shared_size, 0>>>(
                    state + networkSizeAcc[i],
                    d_pos + networkSizeAcc[i]*2,
                    d_pos + networkSizeAcc[j]*2,
	            	d_neighborBlockId + neighborBlockAcc[i] + qMaxNeighborBlock,
                    d_nNeighborBlock + preOffset, d_nNearNeighborBlock + preOffset, d_nGapNeighborBlockE + preOffset, d_nGapNeighborBlockI + preOffset,
                    d_feature,
	            	rden, raxn, d_gapDis,
                    d_ffRatio + networkSize[i] + offset*neuronPerBlock[i], d_inhRatio + typeLayerID[i],
                    d_nTypeMat, d_gap_nTypeMatE, d_gap_nTypeMatI,
                    d_fTypeMat, d_gap_fTypeMatE, d_gap_fTypeMatI,
	            	d_conMat, d_delayMat, d_gapMatE, d_gapMatI,
	            	d_conVec + vecOffset, d_delayVec + vecOffset, d_max_N, tmp_vecID, disNeighborP, d_vecID + vecOffset, d_nVec + offset*neuronPerBlock[i],
	            	d_preTypeConnected, d_preTypeAvail, d_preTypeStrSum,
	        		d_preTypeGapE, d_preTypeAvailGapE, d_preTypeStrGapE, d_preTypeGapI, d_preTypeAvailGapI, d_preTypeStrGapI,
	        		dd_typeAcc,
                    i, j, networkSizeAcc[nLayer], networkSizeAcc[i], networkSizeAcc[j], totalType, totalTypeE, totalTypeI, typeLayerID[i], typeLayerID[j], typeLayerEID[i], typeLayerEID[j], typeLayerIID[i], typeLayerIID[j], nFeature, pPerFeature, neuronPerBlock[j], sum_max_N, offset, networkSize[i], networkSize[j], nType[i], nType[j], nTypeE[i], nTypeE[j], nTypeI[i], nTypeI[j], mE[i], mI[i], maxDistantNeighbor[j*nLayer + i], nearNeighborBlock[j*nLayer + i], maxNeighborBlock[j*nLayer + i], maxNeighborBlockNew[j*nLayer+i]-nearNeighborBlock[j*nLayer+i], gapNeighborBlockE[j*nLayer+i], gapNeighborBlockI[j*nLayer+i], nE[i], nI[i], nE[j], nI[j], disGauss[j*nLayer+i], strictStrength, CmoreN, seed);
	            checkCudaErrors(cudaDeviceSynchronize());
	            getLastCudaError("generate_connections failed");

	            checkCudaErrors(cudaMemcpy(conMat, d_conMat, memorySize, cudaMemcpyDeviceToHost));
                // output connectome data
                fV1_conMat.write((char*)conMat, current_matSize);
                fV1_delayMat.write((char*)delayMat, current_matSize);
                fV1_gapMatE.write((char*)gapMatE, gap_matSizeE);
                fV1_gapMatI.write((char*)gapMatI, gap_matSizeI);
                offset += current_nblock; // offset is block_offset
                qMaxNeighborBlock += current_nblock * maxNeighborBlock[j*nLayer+i];
            }
	        checkCudaErrors(cudaMemcpy(conVec, d_conVec, vecSize + statSize + gap_statSize - nType[j]*sizeof(Size), cudaMemcpyDeviceToHost));

            fV1_vec.write((char*)&(nVec[0]), networkSize[i]*sizeof(Size));
            for (Size in=0; in<networkSize[i]; in++) {
                fV1_vec.write((char*)&(vecID[in*maxDistantNeighbor[j*nLayer+i]]), nVec[i]*sizeof(Size));
                fV1_vec.write((char*)&(conVec[in*maxDistantNeighbor[j*nLayer+i]]), nVec[i]*sizeof(Float));
                fV1_vec.write((char*)&(delayVec[in*maxDistantNeighbor[j*nLayer+i]]), nVec[i]*sizeof(Float));
            }

            checkCudaErrors(cudaFree(gpu_ijChunk));
	        free(cpu_ijChunk);
            checkCudaErrors(cudaFree(gpu_ij));
	        free(cpu_ij);
            {
                cout << "connection stats in  mean: \n";
                Size* preConn = new Size[nType[i]*nType[j]];
                Size* preAvail = new Size[nType[i]*nType[j]];
                Float* preStr = new Float[nType[i]*nType[j]];
                for (PosInt it=0; it<nType[i]; it++) {
                    for (PosInt jt=0; jt<nType[j]; jt++) {
                        preConn[jt*nType[i] + it] = 0;
                        preAvail[jt*nType[i] + it] = 0;
                        preStr[jt*nType[i] + it] = 0.0;
                    }
                }
                for (PosInt it=0; it<nType[j]; it++) {
                    for (PosInt jn=0; jn<networkSize[i]; jn++) {
                        for (PosInt k=0; k<nType[i]; k++) {
                            if (jn%neuronPerBlock[i] < typeAccLayered[i][k+1])  {
                                preConn[it*nType[i] + k] += preTypeConnected[it*networkSize[i] + jn];
                                preAvail[it*nType[i] + k] += preTypeAvail[it*networkSize[i] + jn];
                                preStr[it*nType[i] + k] += preTypeStrSum[it*networkSize[i] + jn];
                                break;
                            }
                        }
                    }
                }

                for (PosInt it=0; it<nType[i]; it++) {
                    Size iTypeN = (typeAccLayered[i][it+1] - typeAccLayered[i][it])*nblock[i];
                    for (PosInt jt=0; jt<nType[j]; jt++) {
                        preConn[jt*nType[i] + it] /= iTypeN;
                        preAvail[jt*nType[i] + it] /= iTypeN;
                        preStr[jt*nType[i] + it] /= iTypeN;
                    }
                }

                cout << "MEAN type:    ";
                for (PosInt it = 0; it < nType[i]; it++) {
                    cout << it;
                    if (it == nType[i]-1) cout << "\n";
                    else cout << ",    ";
                }
                cout << "\n";
                for (PosInt jt=0; jt<nType[j]; jt++) {
                    for (PosInt it=0; it<nType[i]; it++) {
                        if (it==0) {
                            cout << jt << ": ";
                        }
                        PosInt id = jt*nType[i] +it;
                        cout << "[" << preConn[id] << "/" << preAvail[id] << ", " << preStr[id] << "]";
                        if (it==nType[i]-1) cout << "\n";
                        else cout << ", ";
                    }
                }

	            // in max
                for (PosInt it=0; it<nType[i]; it++) {
                    for (PosInt jt=0; jt<nType[j]; jt++) {
                        preConn[jt*nType[i] + it] = 0;
                        preAvail[jt*nType[i] + it] = 0;
                        preStr[jt*nType[i] + it] = 0.0;
                    }
                }
                for (PosInt it=0; it<nType[j]; it++) {
                    for (PosInt jn=0; jn<networkSize[i]; jn++) {
                        for (PosInt k=0; k<nType[i]; k++) {
                            PosInt typeID = it*nType[i] + k;
                            PosInt id = it*networkSize[i] + jn;
                            if (jn%neuronPerBlock[i] < typeAccLayered[i][k+1])  {
					            if (preConn[typeID] < preTypeConnected[id]) {
					            	preConn[typeID] = preTypeConnected[id];
					            }
					            if (preAvail[typeID] < preTypeAvail[id]) {
                                	preAvail[typeID] = preTypeAvail[id];
					            }
					            if (preStr[typeID] < preTypeStrSum[id]) {
                                	preStr[typeID] = preTypeStrSum[id];
					            }
                                break;
                            }
                        }
                    }
                }
	            cout << "MAX  type:    ";
                for (PosInt it = 0; it < nType[i]; it++) {
                    cout << it;
                    if (it == nType[i]-1) cout << "\n";
                    else cout << ",    ";
                }
                cout << "\n";
                for (PosInt jt=0; jt<nType[j]; jt++) {
                    for (PosInt it=0; it<nType[i]; it++) {
                        if (it==0) {
                            cout << jt << ": ";
                        }
                        PosInt id = jt*nType[i] +it;
                        cout << "[" << preConn[id] << "/" << preAvail[id] << ", " << preStr[id] << "]";
                        if (it==nType[i]-1) cout << "\n";
                        else cout << ", ";
                    }
                }

	            // in min
                for (PosInt it=0; it<nType[j]; it++) {
                    for (PosInt jn=0; jn<networkSize[i]; jn++) {
                        for (PosInt k=0; k<nType[i]; k++) {
                            PosInt typeID = it*nType[i] + k;
                            PosInt id = it*networkSize[i] + jn;
                            if (jn%neuronPerBlock[i] < typeAccLayered[i][k+1])  {
					            if (preConn[typeID] > preTypeConnected[id]) {
					            	preConn[typeID] = preTypeConnected[id];
					            }
					            if (preAvail[typeID] > preTypeAvail[id]) {
                                	preAvail[typeID] = preTypeAvail[id];
					            }
					            if (preStr[typeID] > preTypeStrSum[id]) {
                                	preStr[typeID] = preTypeStrSum[id];
					            }
                                break;
                            }
                        }
                    }
                }
	            cout << "MIN  type:    ";
                for (PosInt it = 0; it < nType[i]; it++) {
                    cout << it;
                    if (it == nType[i]-1) cout << "\n";
                    else cout << ",    ";
                }
                cout << "\n";
                for (PosInt jt=0; jt<nType[jt]; jt++) {
                    for (PosInt it=0; it<nType[i]; it++) {
                        if (it==0) {
                            cout << jt << ": ";
                        }
                        PosInt id = jt*nType[i] +it;
                        cout << "[" << preConn[id] << "/" << preAvail[id] << ", " << preStr[id] << "]";
                        if (it==nType[i]-1) cout << "\n";
                        else cout << ", ";
                    }
                }
                delete [] preConn;
                delete [] preStr;
	            delete [] preAvail;
            }
            fStats.write((char*)preTypeConnected, nType[j]*networkSize[i]*sizeof(Size));
            fStats.write((char*)preTypeAvail, nType[j]*networkSize[i]*sizeof(Size));
            fStats.write((char*)preTypeStrSum, nType[j]*networkSize[i]*sizeof(Float));

            fGapStats.write((char*)preTypeGapE, nTypeE[j]*mE[i]*sizeof(Size));
            fGapStats.write((char*)preTypeAvailGapE, nTypeE[j]*mE[i]*sizeof(Size));
            fGapStats.write((char*)preTypeStrGapE, nTypeE[j]*mE[i]*sizeof(Float));
            fGapStats.write((char*)preTypeGapI, nTypeI[j]*mI[i]*sizeof(Size));
            fGapStats.write((char*)preTypeAvailGapI, nTypeE[j]*mE[i]*sizeof(Size));
            fGapStats.write((char*)preTypeStrGapI, nTypeI[j]*mI[i]*sizeof(Float));
        }
        checkCudaErrors(cudaFree(state));
    }
    fV1_conMat.close();
    fV1_delayMat.close();
    fV1_vec.close();
    fStats.close();
    
    // Make gapMats symmetric
    // build a blockNeighborMat of all blocks in all layers
    // [blockAcc.back() x blockAcc.back()] (-1, no connections, 0<=i<nn) for pair-checking
	Int *d_neighborMatE, *d_neighborMatI;
	vector<Int> neighborMatE(blockAcc.back()*blockAcc.back(), -1);
	vector<Int> neighborMatI(blockAcc.back()*blockAcc.back(), -1);
    for (PosInt it=0; it<2; it++) {
	    Int *neighborMat;
        Size *nGapNeighborBlock;
        if (it == 0) {
            neighborMat = &neighborMatE[0];
            nGapNeighborBlock = &nGapNeighborBlockE[0];
        } else {
            neighborMat = &neighborMatI[0];
            nGapNeighborBlock = &nGapNeighborBlockI[0];
        }
        PosInt qMaxNeighborBlock = 0;
        for (PosInt i = 0; i<nLayer; i++) {
            for (PosInt j = 0; j<nLayer; j++) {
	            for (PosInt ib=0; ib<nblock[i]; ib++) {
	            	Size *blockId = neighborBlockId + qMaxNeighborBlock + ib*maxNeighborBlock[j*nLayer+i];
	            	Size nn = nGapNeighborBlock[nLayer*blockAcc[i] + j*nblock[i] + ib];
	            	for (PosInt jb=0; jb<nn; jb++) {
	            		neighborMat[(blockAcc[j] + blockId[jb])*blockAcc.back() + blockAcc[i] + ib] = jb;
	            		cout << blockId[j]  << ", ";
	            	}
	            	cout << "\n";
	            }
                qMaxNeighborBlock += nblock[i]*maxNeighborBlock[j*nLayer+i];
            }
        }
        // assert symmetric
	    for (PosInt i=0; i<blockAcc.back(); i++) {
	    	assert(neighborMat[i*blockAcc.back() + i] == 0); // self-neighbor is the first in blockId 
	    	for (PosInt j=0; j<i; j++) {
	    		if ((neighborMat[i*blockAcc.back() + j] >=0 && neighborMat[j*blockAcc.back() + i] < 0) || (neighborMat[i*blockAcc.back() + j] < 0 && neighborMat[j*blockAcc.back() + i] >= 0)) {
	    			cout << "neighborMat is not symmetric\n";
	    			return EXIT_FAILURE;
	    		}
	    	}
	    }
        if (it == 0) {
            checkCudaErrors(cudaMalloc((void**)&d_neighborMatE, blockAcc.back()*blockAcc.back()*sizeof(Int)));
	        checkCudaErrors(cudaMemcpy(d_neighborMatE, &(neighborMatE[0]), blockAcc.back()*blockAcc.back()*sizeof(Int), cudaMemcpyHostToDevice));
        } else {
            checkCudaErrors(cudaMalloc((void**)&d_neighborMatI, blockAcc.back()*blockAcc.back()*sizeof(Int)));
	        checkCudaErrors(cudaMemcpy(d_neighborMatI, &(neighborMatI[0]), blockAcc.back()*blockAcc.back()*sizeof(Int), cudaMemcpyHostToDevice));
        }
    }
    size_t f0 = nLayer*nLayer*sizeof(Size);

    vector<size_t> gapStatAcc(1,0);
    for (PosInt i = 0; i<nLayer; i++) {
        for (PosInt j = 0; j<nLayer; j++) {
            gapStatAcc.push_back(gapStatAcc.back() + (nTypeE[j]*mE[i] + nTypeI[j]*mI[i])*(sizeof(Float) + sizeof(Size)));
        }
    }

    size_t g0 = 4*nLayer*sizeof(Size);
    for (PosInt i = 0; i<nLayer; i++) {
        size_t fj[2] = {0, 0};
        PosInt qj=0;
        size_t fi[2] = {0, 0};
        PosInt qi=0;
        curandStateMRG32k3a* state;
        size_t randSize = sizeof(curandStateMRG32k3a)*networkSize[i];
        checkCudaErrors(cudaMalloc((void**)&state, randSize));
        for (PosInt j = 0; j<nLayer; j++) {
            void* gpu_stats;
            size_t gap_statSize = (nTypeE[j]*mE[i] + nTypeE[i]*mE[j] + nTypeI[j]*mI[i] + nTypeI[i]*mI[j])*(sizeof(Size) + sizeof(Float));
        	checkCudaErrors(cudaMalloc((void**)&gpu_stats, gap_statSize));
            Size* d_preTypeGapE = (Size*) gpu_stats;
            Float* d_preTypeStrGapE = (Float*) d_preTypeGapE + nTypeE[j]*mE[i];
            Size* d_preTypeGapI = (Size*) d_preTypeStrGapE + nTypeE[j]*mE[i];
            Float* d_preTypeStrGapI = (Float*) d_preTypeGapI + nTypeI[j]*mI[i];
            Size* d_postTypeGapE = (Size*) d_preTypeStrGapI + nTypeI[j]*mI[i];
            Float* d_postTypeStrGapE = (Float*) d_postTypeGapE + nTypeE[i]*mE[j];
            Size* d_postTypeGapI = (Size*) d_postTypeStrGapE + nTypeE[i]*mE[j];
            Float* d_postTypeStrGapI = (Float*) d_postTypeGapI + nTypeI[i]*mI[j];

            void* cpu_stats = malloc(gap_statSize);
            Size* preTypeGapE = (Size*) cpu_stats;
            Float* preTypeStrGapE = (Float*) preTypeGapE + nTypeE[j]*mE[i];
            Size* preTypeGapI = (Size*) preTypeStrGapE + nTypeE[j]*mE[i];
            Float* preTypeStrGapI = (Float*) preTypeGapI + nTypeI[j]*mI[i];
            Size* postTypeGapE = (Size*) preTypeStrGapI + nTypeI[j]*mI[i];
            Float* postTypeStrGapE = (Float*) postTypeGapE + nTypeE[i]*mE[j];
            Size* postTypeGapI = (Size*) postTypeStrGapE + nTypeE[i]*mE[j];
            Float* postTypeStrGapI = (Float*) postTypeGapI + nTypeI[i]*mI[j];

            fGapStats.seekg(g0 + gapStatAcc[i*nLayer + j]);
            fGapStats.read(reinterpret_cast<char*>(postTypeGapE), nTypeE[j]*mE[i]*sizeof(Size));
            fGapStats.read(reinterpret_cast<char*>(postTypeStrGapE), nTypeE[j]*mE[i]*sizeof(Float));
            fGapStats.seekg(nTypeE[j]*mE[i]*sizeof(Size), fGapStats.cur);
            fGapStats.read(reinterpret_cast<char*>(postTypeGapI), nTypeI[j]*mI[i]*sizeof(Size));
            fGapStats.read(reinterpret_cast<char*>(postTypeStrGapI), nTypeI[j]*mI[i]*sizeof(Float));
            fGapStats.seekg(g0 + gapStatAcc[j*nLayer + i]);
            fGapStats.read(reinterpret_cast<char*>(preTypeGapE), nTypeE[i]*mE[j]*sizeof(Size));
            fGapStats.read(reinterpret_cast<char*>(preTypeStrGapE), nTypeE[i]*mE[j]*sizeof(Float));
            fGapStats.seekg(nTypeE[i]*mE[j]*sizeof(Size), fGapStats.cur);
            fGapStats.read(reinterpret_cast<char*>(preTypeGapI), nTypeI[i]*mI[j]*sizeof(Size));
            fGapStats.read(reinterpret_cast<char*>(preTypeStrGapI), nTypeI[i]*mI[j]*sizeof(Float));
	    	checkCudaErrors(cudaMemcpy(gpu_stats, cpu_stats, gap_statSize, cudaMemcpyHostToDevice));

            for (PosInt it=0; it<2; it++) {
                Int *neighborMat, *d_neighborMat;
                size_t *gapMatAcc;
                fstream *fV1_gapMat;
                Size gapNeighborBlock, totalGapType;
                Size post_nType, pre_nType, post_nType0, pre_nType0;
                Size prePerBlock, postPerBlock, prePerBlock0, postPerBlock0;
                Size *d_preTypeGap, *d_postTypeGap, *d_gap_nTypeMat;
                Float *d_preTypeStrGap, *d_postTypeStrGap;
                PosInt preTypeID, postTypeID;
                if (it == 0) {
                    neighborMat = &neighborMatE[0];
                    d_neighborMat = d_neighborMatE;
                    gapNeighborBlock = gapNeighborBlockE[j*nLayer + i];
                    fV1_gapMat = &fV1_gapMatE;
                    gapMatAcc = &gapMatAccE[0];
                    post_nType = nTypeE[i];
                    pre_nType = nTypeE[j];
                    d_postTypeGap = d_postTypeGapE;
                    d_preTypeGap = d_preTypeGapE;
                    d_postTypeStrGap = d_postTypeStrGapE;
                    d_preTypeStrGap = d_preTypeStrGapE;
                    post_nType0 = 0;
                    pre_nType0 = 0;
                    postPerBlock = nE[i];
                    prePerBlock = nE[j];
                    postPerBlock0 = 0;
                    prePerBlock0 = 0;
                    d_gap_nTypeMat = d_gap_nTypeMatE;
                    preTypeID = typeLayerEID[j];
                    postTypeID = typeLayerEID[i];
                    totalGapType = totalTypeE;
                } else {
                    neighborMat = &neighborMatI[0];
                    d_neighborMat = d_neighborMatI;
                    gapNeighborBlock = gapNeighborBlockI[j*nLayer + i];
                    fV1_gapMat = &fV1_gapMatI;
                    gapMatAcc = &gapMatAccI[0];
                    post_nType = nTypeI[i];
                    pre_nType = nTypeI[j];
                    d_postTypeGap = d_postTypeGapI;
                    d_preTypeGap = d_preTypeGapI;
                    d_postTypeStrGap = d_postTypeStrGapI;
                    d_preTypeStrGap = d_preTypeStrGapI;
                    post_nType0 = nTypeE[i];
                    pre_nType0 = nTypeE[j];
                    postPerBlock = nI[i];
                    prePerBlock = nI[j];
                    postPerBlock0 = nE[i];
                    prePerBlock0 = nE[j];
                    d_gap_nTypeMat = d_gap_nTypeMatI;
                    preTypeID = typeLayerIID[j];
                    postTypeID = typeLayerIID[i];
                    totalGapType = totalTypeI;
                }
	            Float* v_outstanding;
	            PosInt* i_outstanding;
                checkCudaErrors(cudaMalloc((void**)&v_outstanding, gapNeighborBlock*postPerBlock*prePerBlock*sizeof(Float)));
                checkCudaErrors(cudaMalloc((void**)&i_outstanding, gapNeighborBlock*postPerBlock*prePerBlock*sizeof(PosInt)));

	            for (PosInt ib=0; ib<nblock[i]; ib++) {
	    	        Size nc = 0; 
	    	        // gather pairs in ith block
	    	        vector<PosInt> clusterID;
	    	        for (PosInt jb=0; jb<nblock[j]; jb++) {
                        Int preID = neighborMat[(blockAcc[j]+jb)*blockAcc.back() + blockAcc[i]+ib];
	    	        	if (preID >= 0) {
	    	        		clusterID.push_back(preID);
	    	        		nc++;
	    	                Size *blockId = neighborBlockId + neighborBlockAcc[j] + qi + jb*maxNeighborBlock[i*nLayer+j]; // for j
                            Int _preID = neighborMat[(blockAcc[i]+ib)*blockAcc.back() + blockAcc[j]+jb];
                            assert(blockId[_preID] == ib);
	    	        	}
	    	        }
	    	        if (nc > 0) {
	    	        	PosInt *d_clusterID;
        	        	checkCudaErrors(cudaMalloc((void**)&d_clusterID, nc*sizeof(PosInt)));
	    	        	checkCudaErrors(cudaMemcpy(d_clusterID, &(clusterID[0]), nc*sizeof(PosInt), cudaMemcpyHostToDevice));
                        // make a mem cluster and send to gpu
	    	        	Float* d_postGapMat;
	                    size_t gapNeighborMatSize = static_cast<size_t>(gapNeighborBlock*prePerBlock)*postPerBlock;// layer i-j symmetric
	    	        	Float* d_clusterGapMat;
        	        	checkCudaErrors(cudaMalloc((void**)&d_postGapMat, gapNeighborMatSize*sizeof(Float)));
        	        	checkCudaErrors(cudaMalloc((void**)&d_clusterGapMat, gapNeighborMatSize*nc*sizeof(Float)));
	    	        	//cout << "uploading " << i << "\n";
                        Float* postGapMat = new Float[gapNeighborMatSize];
                        (*fV1_gapMat).seekg(f0 + (gapMatAcc[i] + fj[it])*sizeof(Float));
                        (*fV1_gapMat).read(reinterpret_cast<char*>(postGapMat), gapNeighborMatSize*sizeof(Float));
	    	        	checkCudaErrors(cudaMemcpy(d_postGapMat, postGapMat, gapNeighborMatSize*sizeof(Float), cudaMemcpyHostToDevice));
                        Float* gapMat_ij = new Float[gapNeighborMatSize];
	    	            Size *blockId = neighborBlockId + neighborBlockAcc[i] + qj + ib*maxNeighborBlock[j*nLayer+i];
	    	        	for (PosInt jb=0; jb<nc; jb++) {
                            (*fV1_gapMat).seekg(f0 + (gapMatAcc[j] + fi[it] + blockId[clusterID[jb]]*gapNeighborMatSize)*sizeof(Float));
                            (*fV1_gapMat).read(reinterpret_cast<char*>(gapMat_ij), gapNeighborMatSize*sizeof(Float));
	    	        		checkCudaErrors(cudaMemcpy(d_clusterGapMat + jb*gapNeighborMatSize, gapMat_ij, gapNeighborMatSize*sizeof(Float), cudaMemcpyHostToDevice));
	    	        	}
                        delete []gapMat_ij;
                        
	    	        	/*{ // m
	    	        		PosInt i0 = 0;
	    	        		size_t gap_nNS = nearNeighborBlock*nI*nI;
	    	        		vector<Int> neighborMat0(nblock*nblock, -1);
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
	    	        	// make symmteric, use original neighborMat
	    	        	//cout << "<<<" << nc << " x " << nI << ">>>\n";
	    	        	generate_symmetry<<<nc, postPerBlock, pre_nType*sizeof(Size) + pre_nType*post_nType*sizeof(Size)>>>(
                                state + ib*neuronPerBlock[i] + postPerBlock,
	    	        			d_clusterID,
                                d_gap_nTypeMat,
	    	        			d_neighborBlockId + neighborBlockAcc[i] + qj + ib*maxNeighborBlock[j*nLayer+i],
	    	        			d_neighborMat, d_blockAcc, d_postGapMat, d_clusterGapMat,
	    	        			d_preTypeGap, d_preTypeStrGap, d_postTypeGap, d_postTypeStrGap,
	    	        			i_outstanding, v_outstanding, dd_typeAcc,
	    	        			ib, i, j, nLayer, gapNeighborBlock, postPerBlock*nblock[i], prePerBlock*nblock[j], postPerBlock0, prePerBlock0, prePerBlock, preTypeID, postTypeID, totalGapType, pre_nType, pre_nType0, post_nType, post_nType0, strictStrength, seed+networkSizeAcc[nLayer]);
	    	        	// unwind the cluster back to cpu
	    	        	checkCudaErrors(cudaDeviceSynchronize());
	    	        	//cout << "downloading " << i << "\n";

                        (*fV1_gapMat).seekp(f0 + (gapMatAcc[i] + fj[it])*sizeof(Float));
	    	        	checkCudaErrors(cudaMemcpy(d_postGapMat, postGapMat, gapNeighborMatSize*sizeof(Float), cudaMemcpyDeviceToHost));
                        (*fV1_gapMat).write((char*)postGapMat, gapNeighborMatSize*sizeof(Float));
                        gapMat_ij = new Float[nc*gapNeighborMatSize];
	    	        	checkCudaErrors(cudaMemcpy(gapMat_ij, d_clusterGapMat, nc*gapNeighborMatSize*sizeof(Float), cudaMemcpyHostToDevice));
	    	        	for (PosInt jb=0; jb<nc; jb++) {
                            (*fV1_gapMat).seekp(f0 + (gapMatAcc[j] + fi[it] + blockId[clusterID[jb]]*gapNeighborMatSize)*sizeof(Float));
                            (*fV1_gapMat).write((char*)(gapMat_ij + jb*gapNeighborMatSize), gapNeighborMatSize*sizeof(Float));
	    	        		/*{ // u
	    	        			PosInt i0 = 0;
	    	        			size_t gap_nNS = nearNeighborBlock*nI*nI;
	    	        			vector<Int> neighborMat0(nblock*nblock, -1);
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
                        delete []gapMat_ij;
        	        	checkCudaErrors(cudaFree(d_clusterGapMat));
        	        	checkCudaErrors(cudaFree(d_clusterID));
	    	        	// checkCudaErrors(cudaDeviceSynchronize());

	    	            // update the counterparts in the pairs
	    	        	//cout << "pair " << i << ": ";
	    	        	for (PosInt ic=0; ic<nc; ic++) {
                            // blockId now gives the neighbor id of ib in layer j
	    	        		neighborMat[(blockAcc[i] + ib)*blockAcc.back() + blockAcc[j] + blockId[clusterID[ic]]] = -1;
	    	        		neighborMat[(blockAcc[j] + blockId[clusterID[ic]])*blockAcc.back() + blockAcc[i] + ib] = -1;
	    	        	}
	    	        	//cout << "\n";
	    	        	clusterID.clear();
	    	        } else {
	    	        	cout << "no more to change\n";
	    	        }
                    fj[it] += gapNeighborBlock*prePerBlock*postPerBlock;
	            }
                checkCudaErrors(cudaFree(i_outstanding));
                checkCudaErrors(cudaFree(v_outstanding));
            }
            for (PosInt iq=0;iq<i;iq++) {
                fi[0] += nblock[j]*gapNeighborBlockE[i*nLayer+j]*nE[iq]*nE[j];
                fi[1] += nblock[j]*gapNeighborBlockI[i*nLayer+j]*nI[iq]*nI[j];
            }
	    	checkCudaErrors(cudaMemcpy(cpu_stats, gpu_stats, gap_statSize, cudaMemcpyHostToDevice));
            fGapStats.seekp(g0 + gapStatAcc[i*nLayer + j]);
            fGapStats.write((char*)postTypeGapE, nTypeE[j]*mE[i]*sizeof(Size));
            fGapStats.write((char*)postTypeStrGapE, nTypeE[j]*mE[i]*sizeof(Float));
            fGapStats.seekp(nTypeE[j]*mE[i]*sizeof(Size), fGapStats.cur);
            fGapStats.write((char*)postTypeGapI, nTypeI[j]*mI[i]*sizeof(Size));
            fGapStats.write((char*)postTypeStrGapI, nTypeI[j]*mI[i]*sizeof(Float));
            fGapStats.seekp(g0 + gapStatAcc[j*nLayer + i]);
            fGapStats.write((char*)preTypeGapE, nTypeE[i]*mE[j]*sizeof(Size));
            fGapStats.write((char*)preTypeStrGapE, nTypeE[i]*mE[j]*sizeof(Float));
            fGapStats.seekp(nTypeE[i]*mE[j]*sizeof(Size), fGapStats.cur);
            fGapStats.write((char*)preTypeGapI, nTypeI[i]*mI[j]*sizeof(Size));
            fGapStats.write((char*)preTypeStrGapI, nTypeI[i]*mI[j]*sizeof(Float));

        	checkCudaErrors(cudaFree(gpu_stats));

            qj += nblock[i]*maxNeighborBlock[j*nLayer+i];
            for (PosInt iq=0;iq<i;iq++) {
                qi += nblock[j]*maxNeighborBlock[i*nLayer+j];
            }
        }
        checkCudaErrors(cudaFree(state));
    }

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
    fGapStats.close();
    fV1_gapMatE.close();
    fV1_gapMatI.close();

    cout << "gap junction stats in  mean: \n";
    fGapStats.seekg(4*nLayer*sizeof(Size), fGapStats.beg);
    for (PosInt i = 0; i<nLayer; i++) {
        for (PosInt j = 0; j<nLayer; j++) {
            Size pre_nType, post_nType, post_nType0;
            Size  postPerBlock0,  postPerBlock;
            for (PosInt iEI=0; iEI<2; iEI++) {
                if (iEI == 0) {
                    pre_nType = nTypeE[j];
                    post_nType = nTypeE[i];
                    postPerBlock = nE[i];
                    post_nType0 = 0;
                    postPerBlock0 = 0;
                } else {
                    pre_nType = nTypeI[j];
                    post_nType = nTypeI[i];
                    postPerBlock = nI[i];
                    post_nType0 = nTypeE[i];
                    postPerBlock0 = nE[i];
                }
                Size postN = nblock[i]*postPerBlock;
                Size* preTypeAvail = new Size[pre_nType*postN];
                Size* preTypeConnected = new Size[pre_nType*postN];
                Float* preTypeStrSum = new Float[pre_nType*postN];
                fGapStats.read(reinterpret_cast<char*>(preTypeConnected), pre_nType*postN*sizeof(Size));
                fGapStats.read(reinterpret_cast<char*>(preTypeAvail), pre_nType*postN*sizeof(Size));
                fGapStats.read(reinterpret_cast<char*>(preTypeStrSum), pre_nType*postN*sizeof(Float));

                Size* preConn = new Size[pre_nType*post_nType];
                Size* preAvail = new Size[pre_nType*post_nType];
                Float* preStr = new Float[pre_nType*post_nType];
                for (PosInt it=0; it<post_nType; it++) {
                    for (PosInt jt=0; jt<pre_nType; jt++) {
                        preConn[jt*post_nType + it] = 0;
                        preAvail[jt*post_nType + it] = 0;
                        preStr[jt*post_nType + it] = 0.0;
                    }
                }
                for (PosInt it=0; it<pre_nType; it++) {
                    for (PosInt jn=0; jn<postN; jn++) {
                        for (PosInt k=0; k<post_nType; k++) {
                            if (jn%postPerBlock + postPerBlock0 < typeAccLayered[i][post_nType0 + k+1])  {
                                preConn[it*post_nType + k] += preTypeConnected[it*postN + jn];
                                preAvail[it*post_nType + k] += preTypeAvail[it*postN + jn];
                                preStr[it*post_nType + k] += preTypeStrSum[it*postN + jn];
                                break;
                            }
                        }
                    }
                }

                for (PosInt it=0; it<post_nType; it++) {
                    Size iTypeN = (typeAccLayered[i][post_nType0 + it+1] - typeAccLayered[i][post_nType0 + it])*nblock[i];
                    for (PosInt jt=0; jt<pre_nType; jt++) {
                        preConn[jt*post_nType + it] /= iTypeN;
                        preAvail[jt*post_nType + it] /= iTypeN;
                        preStr[jt*post_nType + it] /= iTypeN;
                    }
                }

                cout << "MEAN type:    ";
                for (PosInt it = 0; it < post_nType; it++) {
                    cout << it;
                    if (it == post_nType-1) cout << "\n";
                    else cout << ",    ";
                }
                cout << "\n";
                for (PosInt jt=0; jt<pre_nType; jt++) {
                    for (PosInt it=0; it<post_nType; it++) {
                        if (it==0) {
                            cout << jt << ": ";
                        }
                        PosInt id = jt*post_nType +it;
                        cout << "[" << preConn[id] << "/" << preAvail[id] << ", " << preStr[id] << "]";
                        if (it==post_nType-1) cout << "\n";
                        else cout << ", ";
                    }
                }

	            // in max
                for (PosInt it=0; it<post_nType; it++) {
                    for (PosInt jt=0; jt<pre_nType; jt++) {
                        preConn[jt*post_nType + it] = 0;
                        preAvail[jt*post_nType + it] = 0;
                        preStr[jt*post_nType + it] = 0.0;
                    }
                }
                for (PosInt it=0; it<pre_nType; it++) {
                    for (PosInt jn=0; jn<postN; jn++) {
                        for (PosInt k=0; k<post_nType; k++) {
                            PosInt typeID = it*post_nType + k;
                            PosInt id = it*postN + jn;
                            if (jn%postPerBlock + postPerBlock0 < typeAccLayered[i][post_nType0 + k+1])  {
			    	            if (preConn[typeID] < preTypeConnected[id]) {
			    	            	preConn[typeID] = preTypeConnected[id];
			    	            }
			    	            if (preAvail[typeID] < preTypeAvail[id]) {
                                	preAvail[typeID] = preTypeAvail[id];
			    	            }
			    	            if (preStr[typeID] < preTypeStrSum[id]) {
                                	preStr[typeID] = preTypeStrSum[id];
			    	            }
                                break;
                            }
                        }
                    }
                }
	            cout << "MAX  type:    ";
                for (PosInt it = 0; it < post_nType; it++) {
                    cout << it;
                    if (it == post_nType-1) cout << "\n";
                    else cout << ",    ";
                }
                cout << "\n";
                for (PosInt jt=0; jt<pre_nType; jt++) {
                    for (PosInt it=0; it<post_nType; it++) {
                        if (it==0) {
                            cout << jt << ": ";
                        }
                        PosInt id = jt*post_nType +it;
                        cout << "[" << preConn[id] << "/" << preAvail[id] << ", " << preStr[id] << "]";
                        if (it==post_nType-1) cout << "\n";
                        else cout << ", ";
                    }
                }

	            // in min
                for (PosInt it=0; it<pre_nType; it++) {
                    for (PosInt jn=0; jn<postN; jn++) {
                        for (PosInt k=0; k<post_nType; k++) {
                            PosInt typeID = it*post_nType + k;
                            PosInt id = it*postN + jn;
                            if (jn%postPerBlock + postPerBlock0 < typeAccLayered[i][post_nType0 + k+1])  {
			    	            if (preConn[typeID] > preTypeConnected[id]) {
			    	            	preConn[typeID] = preTypeConnected[id];
			    	            }
			    	            if (preAvail[typeID] > preTypeAvail[id]) {
                                	preAvail[typeID] = preTypeAvail[id];
			    	            }
			    	            if (preStr[typeID] > preTypeStrSum[id]) {
                                	preStr[typeID] = preTypeStrSum[id];
			    	            }
                                break;
                            }
                        }
                    }
                }
	            cout << "MIN  type:    ";
                for (PosInt it = 0; it < post_nType; it++) {
                    cout << it;
                    if (it == post_nType-1) cout << "\n";
                    else cout << ",    ";
                }
                cout << "\n";
                for (PosInt jt=0; jt<pre_nType; jt++) {
                    for (PosInt it=0; it<post_nType; it++) {
                        if (it==0) {
                            cout << jt << ": ";
                        }
                        PosInt id = jt*post_nType +it;
                        cout << "[" << preConn[id] << "/" << preAvail[id] << ", " << preStr[id] << "]";
                        if (it==post_nType-1) cout << "\n";
                        else cout << ", ";
                    }
                }
                delete [] preConn;
                delete [] preStr;
	            delete [] preAvail;
            }
        }
    }
    printf("connectivity constructed\n");
    
	ofstream fConnectome_cfg(output_cfg_filename + suffix, fstream::out | fstream::binary);
	if (!fConnectome_cfg) {
		cout << "Cannot open or find " << output_cfg_filename + suffix <<"\n";
		return EXIT_FAILURE;
	} else {
		fConnectome_cfg.write((char*) &nLayer, sizeof(Size));
		fConnectome_cfg.write((char*) &(nType[0]), nLayer*sizeof(Size));
		fConnectome_cfg.write((char*) &(nTypeE[0]), nLayer*sizeof(Size));
		fConnectome_cfg.write((char*) &(nTypeI[0]), nLayer*sizeof(Size));
		fConnectome_cfg.write((char*) (&typeAcc0[0]), (nLayer + totalType)*sizeof(Size));
		fConnectome_cfg.write((char*) (&nTypeMat[0]), totalType*totalType*sizeof(Size));
		fConnectome_cfg.write((char*) (&gap_nTypeMatE[0]), totalTypeE*totalTypeE*sizeof(Size));
		fConnectome_cfg.write((char*) (&gap_nTypeMatI[0]), totalTypeI*totalTypeI*sizeof(Size));
		fConnectome_cfg.write((char*) (&gap_fTypeMatE[0]), pPerFeature*nFeature*totalTypeE*totalTypeE*sizeof(Float));
		fConnectome_cfg.write((char*) (&gap_fTypeMatI[0]), pPerFeature*nFeature*totalTypeI*totalTypeI*sizeof(Float));
		fConnectome_cfg.write((char*) (&rDend[0]), totalType*totalType*sizeof(Float));
		fConnectome_cfg.write((char*) (&rAxon[0]), totalType*totalType*sizeof(Float));
		fConnectome_cfg.close();
	}

	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_neighborMatE));
    checkCudaErrors(cudaFree(d_neighborMatI));

    checkCudaErrors(cudaFree(preset_chunk));
    checkCudaErrors(cudaFree(block_chunk));
	for (PosInt i=0; i<nLayer; i++) {
		checkCudaErrors(cudaStreamDestroy(layerStream[i]));
	}
    delete []maxCortExc;
    delete []ffRatio;
	delete []nLGN_eff;
    return 0;
}
