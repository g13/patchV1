#include "patch.h"
//TODO: gap junction and learning in cortex
int main(int argc, char **argv) {
	namespace po = boost::program_options;
	using namespace std;
	using std::string;
    int nDevice;
    checkCudaErrors(cudaGetDeviceCount(&nDevice));
    cout << nDevice << " gpu on the node\n";
    int iDevice;
    size_t maxFree = 0;
    for (PosInt i=0; i<nDevice; i++) {
        checkCudaErrors(cudaSetDevice(i));
        size_t free;
        size_t total;
        checkCudaErrors(cudaMemGetInfo(&free, &total));
        if (free > maxFree) {
            iDevice = i; 
            maxFree = free;
        }
    }
    checkCudaErrors(cudaSetDevice(iDevice));
    cout << "using gpu device " << iDevice << ", with " << maxFree/1024.0/1024.0 << "Mb memory\n";
	cudaDeviceProp deviceProps;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
	printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
	printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);
	printf("total global memory: %f Mb.\n", deviceProps.totalGlobalMem/1024.0/1024.0);
	size_t GMemAvail = deviceProps.totalGlobalMem;

#ifdef SINGLE_PRECISION
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
#else
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
	printf("maximum threads per MP: %d.\n", deviceProps.maxThreadsPerMultiProcessor);
	printf("shared memory per block: %zu bytes.\n", deviceProps.sharedMemPerBlock);
	printf("registers per block: %d.\n", deviceProps.regsPerBlock);
	cout << "\n";

	bool storeSpatial = true;
	Float dt; // in ms, better in fractions of binary 
	Float dot;
    bool print_log;
	bool useNewLGN, readFeature;
	bool saveLGN_fr, saveLGN_gallery, saveOutputB4V1;
	bool frameVisV1output, frameVisLGNoutput, framePhyV1output;
    bool hWrite, rawData;
    bool ignoreRetinogeniculateDelay;
    bool learnData_FF;
    bool learnData_V1;
    bool manual;
    bool symQ;
    bool flat_retina;
    bool uniform_LGN;
    bool LGN_switch;
    bool getLGN_sp;
	bool delPrevSnapshot;
	bool asInit;
	bool use_v0;
    int rebound;
    int learning;
    int iModel;
	Size snapshotInterval;
	Size nChunk;
	Size matConcurrency;
	Float phyWidth_scale;
	Float visWidth_scale;
    Size nLearnTypeFF_E, nLearnTypeFF_I, nLearnTypeE;
	SmallSize nSpatialSample1D;
	SmallSize mSpatialSample1D;
	SmallSize nKernelSample;
	SmallSize mKernelSample;
    Size SCsplit;
    vector<Float> spE0;
    vector<Float> spI0;
    vector<Float> noisyDep;
    vector<Float> tonicDep;
    vector<Float> synFailFF;
    vector<Float> synFail;
    // TODO: specify proportion of different types of conductances
	vector<Float> grFF;
	vector<Float> grE;
	vector<Float> grI;
	vector<Float> gdFF;
	vector<Float> gdE;
	vector<Float> gdI;
	vector<Float> A_LGN;
	vector<Float> A_V1;
	vector<Float> A_Q;
	vector<Float> tauQ;
	vector<Float> tauLTP;
	vector<Float> tauLTD;
	vector<Float> tauTrip;
	vector<Float> tauAvg;
	vector<Float> targetFR;
	vector<Float> gmaxLGN;
	vector<Float> gmaxE;
	vector<Float> gmaxQ;
	vector<Float> gminLGN;
	vector<Float> gminE;
	vector<Float> gminQ;
	vector<Float> v0;
	vector<Float> w0;
	vector<Float> gFF0;
	vector<Float> gE0;
	vector<Float> gI0;
	vector<Float> pFF;
	vector<Float> pE;
	vector<Float> pI;
	vector<Float> vR;
	vector<Float> vThres;
	vector<Float> gL;
	vector<Float> C;
	vector<Float> tRef;
	vector<Float> vT;
	vector<Float> deltaT;
	vector<Float> tau_w;
	vector<Float> a;
	vector<Float> b;
    vector<PosInt> preList, postList;
    vector<Float> sList;
	vector<Float> sRatioLGN;
	vector<Size> nTypeHierarchy;
	vector<Float> fbROI, fbSamplingRate, fbEI_ratio, fbFR;
	vector<Size> nFBperColumn;
	Float nsig; // extent of spatial RF sampling in units of std
	Float tau, mau;
	Float Itau; // in ms .. cone adaptation at 300ms https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003289
    Float sRatioV1;
	Float convolRatio;
	Float frRatioLGN;
	Float speedOfThought;
	PosInt frameRate; // Hz
	Size nt; // number of steps
	PosIntL seed;

	po::options_description generic_opt("Generic options");
	generic_opt.add_options()
		("seed,s", po::value<PosIntL>(&seed),"seed for trial")
		("cfg_file,c", po::value<string>()->default_value("patchV1.cfg"), "filename for configuration file")
		("print,p", po::value<bool>(&print_log)->default_value(false), "print outputs")
		("help,h", "print usage");
	po::options_description top_opt("top-level configuration");
	// non-files
	top_opt.add_options()
		("dt", po::value<Float>(&dt)->default_value(0.0625), "simulatoin time step in ms") 
		("nChunk,n", po::value<Size>(&nChunk)->default_value(10),"simulation in chunks, empricial")
		("matConcurrency,n", po::value<Size>(&matConcurrency)->default_value(10),"sum presynaptic inputs from connection matrices in parallel, depends on the availability of device memory")
		("speedOfThought", po::value<Float>(&speedOfThought)->default_value(1.0), "velocity of conduction, mm/ms") 
		("nt", po::value<Size>(&nt)->default_value(8000), "total simulatoin time in units of time step")
		("fbROI", po::value<vector<Float>>(&fbROI), "total simulatoin time in units of time step")
		("nFBperColumn", po::value<vector<Size>>(&nFBperColumn), "total simulatoin time in units of time step")
		("fbSamplingRate", po::value<vector<Float>>(&fbSamplingRate), "total simulatoin time in units of time step")
		("fbEI_ratio", po::value<vector<Float>>(&fbEI_ratio), "total simulatoin time in units of time step")
		("fbFR", po::value<vector<Float>>(&fbFR), "total simulatoin time in units of time step")
		("snapshotInterval", po::value<Size>(&snapshotInterval)->default_value(10000), "snapshotInterval for resume or initialization")
		("nsig", po::value<Float>(&nsig)->default_value(3), "extent of spatial RF sampling in units of std")
		("nSpatialSample1D", po::value<SmallSize>(&nSpatialSample1D)->default_value(warpSize), "number of samples per x,y direction for a parvo-LGN spatial RF")
		("mSpatialSample1D", po::value<SmallSize>(&mSpatialSample1D)->default_value(warpSize), "number of samples per x,y direction for a magno-LGN spatial RF")
        ("phyWidth_scale", po::value<Float>(&phyWidth_scale)->default_value(1), "pixel width of the physical V1 sheet frame output")
        ("visWidth_scale", po::value<Float>(&visWidth_scale)->default_value(1), "pixel width of the half visual field frame output")
		("tau", po::value<Float>(&tau)->default_value(250.0), "the backward time interval that a parvo-LGN temporal RF should cover")
		("mau", po::value<Float>(&mau)->default_value(250.0), "the backward time interval that a magno-LGN temporal RF should cover")
		("Itau", po::value<Float>(&Itau)->default_value(300.0), "the light intensity adaptation time-scale of a cone")
		("sRatioLGN", po::value<vector<Float>>(&sRatioLGN), "scale connection strength from LGN in array of size nType")
		("sRatioV1", po::value<Float>(&sRatioV1)->default_value(1), "scale connection strength from V1")
		("convolRatio", po::value<Float>(&convolRatio)->default_value(1), "scale convol value")
		("frRatioLGN", po::value<Float>(&frRatioLGN)->default_value(1), "scale LGN firing rate")
		("nKernelSample", po::value<Size>(&nKernelSample)->default_value(500), "number of samples per parvo-temporal kernel")
		("mKernelSample", po::value<Size>(&mKernelSample)->default_value(500), "number of samples per magno-temporal kernel")
		("frameRate", po::value<PosInt>(&frameRate)->default_value(120), "frame rate of the input stimulus")
		("riseTimeFF", po::value<vector<Float>>(&grFF), "array for rise time of the feed-forward excitatory conductances, size should be consistent with decayTimeFF")
		("riseTimeE", po::value<vector<Float>>(&grE), "array for rise time of the cortical excitatory conductances, size should be consistent with decayTimeE")
		("riseTimeI", po::value<vector<Float>>(&grI), "array for rise time of the inhibitory conductances, size should be consistent with decayTimeI")
		("decayTimeFF", po::value<vector<Float>>(&gdFF), "array for decay time of the feed-forward excitatory conductances, size should be consistent with riseTimeFF")
		("decayTimeE", po::value<vector<Float>>(&gdE), "array for decay time of the cortical excitatory conductances, size should be consistent with riseTimeE")
		("decayTimeI", po::value<vector<Float>>(&gdI), "array for decay time of the inhibitory conductances, size should be consistent with riseTimeI")
		("v0", po::value<vector<Float>>(&v0), "array for initial dist [nType, mean, std]")
		("w0", po::value<vector<Float>>(&w0), "AdEx model's initial dist of w [nType, mean, std]")
		("gFF0", po::value<vector<Float>>(&gFF0), "array for initial dist [nType, ngTypeFF, mean, std]")
		("gE0", po::value<vector<Float>>(&gE0), "array for initial dist [nType, ngTypeE, mean, std]")
		("gI0", po::value<vector<Float>>(&gI0), "array for initial dist [nType, ngTypeI, mean, std]")
		("pFF", po::value<vector<Float>>(&pFF), "array for proportions of [nType, ngTypeFF]")
		("pE", po::value<vector<Float>>(&pE), "array for proportions of [nType, ngTypeE]")
		("pI", po::value<vector<Float>>(&pI), "array for proportions of [nType, ngTypeI]")
        ("learning", po::value<int>(&learning)->default_value(1), "trip rule learning, from Jennifer, 0 no learning, 1 default learning")
		("rebound", po::value<int>(&rebound)->default_value(0), "after connection strength hits min or max can they retract") // TODO: replace "continue;" in compute_V 
		("iModel", po::value<int>(&iModel)->default_value(0), "0: LIF, 1: AdEx")
        ("A_LGN", po::value<vector<Float>>(&A_LGN), "array of learning rate for feedforward connections")
        ("A_V1", po::value<vector<Float>>(&A_V1), "array of learning rate for coritcal connections")
        ("A_Q", po::value<vector<Float>>(&A_Q), "array of learning rate ofr inhibitory connecitons")
        ("symQ", po::value<bool>(&symQ)->default_value(true), "if pre and post of STDP Qlearning is symmetric")
        ("nLearnTypeFF_I", po::value<Size>(&nLearnTypeFF_I)->default_value(1), " number of types of triplet rule learning for feedforward LGN connections to inhibitory neurons")
        ("nLearnTypeFF_E", po::value<Size>(&nLearnTypeFF_E)->default_value(1), " number of types of triplet rule learning for feedforward LGN connections to excitatory neurons")
        ("nLearnTypeE", po::value<Size>(&nLearnTypeE)->default_value(1), " number of types of triplet rule learning that involves only excitatory coritcal cells")
		("tauQ", po::value<vector<Float>>(&tauQ), "array for the decay timescale of iSTDP variable")
		("tauLTP", po::value<vector<Float>>(&tauLTP), "array for the decay timescale of LTP variable of the triplet rule")
		("tauLTD", po::value<vector<Float>>(&tauLTD), "array for the decay timescale of LTD variable of the triplet rule")
		("tauTrip", po::value<vector<Float>>(&tauTrip), "array for the decay timescale of the triplet variable of the triplet rule")
		("tauAvg", po::value<vector<Float>>(&tauAvg), "the decay timescale of the filtered firing rate for the triplet rule, size of 2, (E, I)")
		("targetFR", po::value<vector<Float>>(&targetFR), "the target firing rate for the triplet rule LTD, size of 2, (E, I)")
		("gmaxLGN", po::value<vector<Float>>(&gmaxLGN), "maximum connection strength for LGN->V1, size of nLearnTypeFF")
		("gmaxE", po::value<vector<Float>>(&gmaxE), "maximum connection strength for E->E, size of nLearnTypeE")
		("gmaxQ", po::value<vector<Float>>(&gmaxQ), "maximum connection strength for I->E, size of nLearnTypeQ")
		("gminLGN", po::value<vector<Float>>(&gminLGN), "minimum connection strength for LGN->V1, size of nLearnTypeFF")
		("gminE", po::value<vector<Float>>(&gminE), "minimum connection strength for E->E, size of nLearnTypeE")
		("gminQ", po::value<vector<Float>>(&gminQ), "minimum connection strength for I->E, size of nLearnTypeQ")
		("vR",po::value<vector<Float>>(&vR), "single neuron model's reseting voltage of size nType")
		("vThres",po::value<vector<Float>>(&vThres), "single neuron model's reset threshold of size nType")
		("gL",po::value<vector<Float>>(&gL), "single neuron model's leaky conductance of size nType")
		("C",po::value<vector<Float>>(&C), "single neuron model's capacitance of size nType")
		("tRef",po::value<vector<Float>>(&tRef), "single neuron model's refactory period of size nType")
		("vT",po::value<vector<Float>>(&vT), "AdEx model's spiking threshold (parameter) of size nType")
		("deltaT",po::value<vector<Float>>(&deltaT), "AdEx model's deltaT of size nType")
		("tau_w",po::value<vector<Float>>(&tau_w), "AdEx model's time scale of adaptive variable w of size nType")
		("a",po::value<vector<Float>>(&a), "AdEx model's parameter a of size nType")
		("b",po::value<vector<Float>>(&b), "AdEx model's parameter b of size nType")
        ("nTypeHierarchy",  po::value<vector<Size>>(&nTypeHierarchy), "types of excitatory neurons and inhibtory neurons")
        ("spE0",  po::value<vector<Float>>(&spE0), "Exc. initial spike dist. mean. of size [nTypeHierarchy[0], 2] (s->c) ")
        ("spI0",  po::value<vector<Float>>(&spI0), "Inh. initial spike dist. mean. of size [nTypeHierarchy[1], 2] (s->c) ")
        ("SCsplit",  po::value<Size>(&SCsplit)->default_value(0), "simple complex split at nLGN > SCsplit (simple)")
		("noisyDep", po::value<vector<Float>>(&noisyDep), "noisy depolarization borrow from NMC model")
		("tonicDep", po::value<vector<Float>>(&tonicDep), "tonic depolarization borrow from NMC model")
        ("synFailFF",  po::value<vector<Float>>(&synFailFF), "FF synpase failure rate of size nType")
        ("synFail",  po::value<vector<Float>>(&synFail), "synpase failure rate of size [nType, nType]")
        ("manual", po::value<bool>(&manual)->default_value(false), "manually connect neurons, modify on top of conMat")
        ("preList", po::value<vector<PosInt>>(&preList), "the presynaptic neurons of the manual connections")
        ("postList", po::value<vector<PosInt>>(&postList), "the postynaptic neurons of the manual connections")
        ("sList", po::value<vector<Float>>(&sList), "the strength of the manual connections, can be an scalar")
		("readFeature", po::value<bool>(&readFeature)->default_value(true), "read features, OD, OP rather than learning them")
		("hWrite", po::value<bool>(&hWrite)->default_value(true), "write h data to fRawData output")
		("saveLGN_fr", po::value<bool>(&saveLGN_fr)->default_value(true),"write LGN firing rates to disk, specify filename through LGN_fr_filename")
		("rawData", po::value<bool>(&rawData)->default_value(true), "to save V1 response (spike, v, g, (h, depends on hWrite)) over time")
		("learnData_FF", po::value<bool>(&learnData_FF)->default_value(false), "to save LGN->V1 connection strength plasticity over time")
		("learnData_V1", po::value<bool>(&learnData_V1)->default_value(false), "to save V1->V1 connection strength plasticity over time")
		("framePhyV1output", po::value<bool>(&framePhyV1output)->default_value(false),"get response stats frame for cortical sheet of V1")
		("frameVisV1output", po::value<bool>(&frameVisV1output)->default_value(false),"get response stats frame for visual field of V1")
		("frameVisLGNoutput", po::value<bool>(&frameVisLGNoutput)->default_value(false),"get response stats frame for visual field of LGN")
		("dot", po::value<Float>(&dot)->default_value(16), "outputFrame interval in ms") 
		("saveLGN_gallery", po::value<bool>(&saveLGN_gallery)->default_value(true), "check convolution kernels and maximum convolution values, write data to disk, specify filename through LGN_gallery_filename")
		("saveOutputB4V1", po::value<bool>(&saveOutputB4V1)->default_value(true), "check adapted luminance values, write data to disk, specify filename through outputB4V1_filename")
		("ignoreRetinogeniculateDelay", po::value<bool>(&ignoreRetinogeniculateDelay)->default_value(true), "ignore the delay")
		("flat_retina", po::value<bool>(&flat_retina)->default_value(false), "flat retina to flat stimulus, no special transformations")
		("uniform_LGN", po::value<bool>(&uniform_LGN)->default_value(false), "uniform LGN properties within cell type")
		("LGN_switch", po::value<bool>(&LGN_switch)->default_value(false), "control LGN activation during retinal waves, make sure LGN_switch file is ready")
		("getLGN_sp", po::value<bool>(&getLGN_sp)->default_value(false), "if write LGN spikes to file")
		("delPrevSnapshot", po::value<bool>(&delPrevSnapshot)->default_value(true), "delete old snapshot")
		("asInit", po::value<bool>(&asInit)->default_value(true), "use snapshot for initialization not to resume previous simulation")
		("use_v0", po::value<bool>(&use_v0)->default_value(false), "use v0 to initialize membrane potential, otherwise is set according to depC")
		("useNewLGN", po::value<bool>(&useNewLGN)->default_value(true), "regenerate the a new ensemble of LGN parameters according to their distribution");

	// files
	string connectome_cfg_filename, patchV1_cfg_filename, restore;
    string output_suffix, output_suffix0; // suffix to be added to all output filename
    string conV1_suffix; // suffix of the input filenames, if suffix is not the same, set f*
    string conLGN_suffix; // suffix of the input filenames, if suffix is not the same, set f*
    string snapshot_suffix; // suffix of the input filenames, if suffix is not the same, set f*
	string conStats_filename;
	string stimulus_filename, LGN_switch_filename;
	string V1_RF_filename, V1_feature_filename, V1_pos_filename;
	string neighborBlock_filename;
	string V1_vec_filename, V1_delayMat_filename, V1_conMat_filename;
	string LGN_surfaceID_filename;
	string LGN_filename, LGN_vpos_filename, LGN_V1_s_filename, LGN_V1_ID_filename; // inputs
	string LGN_fr_filename, outputFrame_filename; // outputs
	string LGN_convol_filename, LGN_gallery_filename, outputB4V1_filename, rawData_filename, learnData_FF_filename, learnData_V1_filename, sLGN_filename, LGN_sp_filename;
	top_opt.add_options()
		("output_suffix", po::value<string>(&output_suffix0)->default_value(""),"output file suffix")
		("conV1_suffix", po::value<string>(&conV1_suffix)->default_value(""),"suffix for V1 connectome files")
		("conLGN_suffix", po::value<string>(&conLGN_suffix)->default_value(""),"suffix for LGN to V1 connectome files")
		("fConnectome_cfg", po::value<string>(&connectome_cfg_filename)->default_value("connectome_cfg"),"file that stores connectome cfg parameters")
		("snapshot_suffix", po::value<string>(&snapshot_suffix),"suffix of the snapshot")
		("fSnapshot", po::value<string>(&restore)->default_value(""),"file that can be used to restore previous simulation status")
		("fPatchV1_cfg", po::value<string>(&patchV1_cfg_filename)->default_value("patchV1_cfg"),"file that stores patchV1 cfg parameters")
		("fLGN_switch", po::value<string>(&LGN_switch_filename)->default_value("LGN_switch"),"file that stores which types of LGN to turn on and off over time, ints of size: (nInputType, nStatus)")
		("fStimulus", po::value<string>(&stimulus_filename)->default_value("stimulus.bin"),"file that stores LGN firing rates, array of size (nframes,width,height,3)")
		("fLGN_vpos", po::value<string>(&LGN_vpos_filename)->default_value("LGN_vpos.bin"),"file that stores LGN neurons information")
		("fLGN_V1_ID", po::value<string>(&LGN_V1_ID_filename)->default_value("LGN_V1_idList"),"file stores LGN to V1 connections")
		("fLGN_V1_s", po::value<string>(&LGN_V1_s_filename)->default_value("LGN_V1_sList"),"file stores LGN to V1 connection strengths")
		("fLGN_surfaceID", po::value<string>(&LGN_surfaceID_filename)->default_value("LGN_surfaceID.bin"),"file stores LGN position ID on surface memory")
		("fV1_pos", po::value<string>(&V1_pos_filename)->default_value("V1_allpos.bin"), "file that stores V1 coritcal position and visual field position")
		("fV1_feature", po::value<string>(&V1_feature_filename)->default_value("V1_feature.bin"), "file to read spatially predetermined functional features of neurons")
		("fV1_conMat", po::value<string>(&V1_conMat_filename)->default_value("V1_conMat"), "file that stores V1 to V1 connection within the neighboring blocks")
		("fV1_delayMat", po::value<string>(&V1_delayMat_filename)->default_value("V1_delayMat"), "file that stores V1 to V1 transmission delay within the neighboring blocks")
		("fConStats", po::value<string>(&conStats_filename)->default_value("conStats"),"file that stores connection stats")
		("fV1_vec", po::value<string>(&V1_vec_filename)->default_value("V1_vec"), "file that stores V1 to V1 connection ID, strength and transmission delay far the neighboring blocks")
		("fNeighborBlock", po::value<string>(&neighborBlock_filename)->default_value("neighborBlock"), "file that stores V1 to V1 connection ID, strength and transmission delay far the neighboring blocks")
		("fV1_RF", po::value<string>(&V1_RF_filename)->default_value("V1_RF"), "file that stores V1 RF properties, (orientation info is in fV1_feature)")
		("fLGN", po::value<string>(&LGN_filename)->default_value("LGN"),"file that stores all the information of LGN neurons")
		("fLGN_fr", po::value<string>(&LGN_fr_filename)->default_value("LGN_fr"),"file stores LGN firing rates")
		("fRawData", po::value<string>(&rawData_filename)->default_value("rawData"), "file that stores V1 response (spike, v, g) over time")
		("fLearnData_FF", po::value<string>(&learnData_FF_filename)->default_value("learnData_FF"), "file that stores LGN->V1 connection strength and the related variables over time, make sure learnData_FF is set")
		("f_sLGN", po::value<string>(&sLGN_filename)->default_value("sLGN"), "file that stores the LGN->V1 connection strength over time, make sure learnData_FF is set")
		("fLGN_sp", po::value<string>(&LGN_sp_filename)->default_value("LGN_sp"), "write LGN spikes to file")
		("fOutputFrame", po::value<string>(&outputFrame_filename)->default_value("outputFrame"),"file that stores firing rate from LGN and/or V1 (in physical location or visual field) spatially to be ready for frame production") // TEST 
		("fOutputB4V1", po::value<string>(&outputB4V1_filename)->default_value("outputB4V1"),"file that stores luminance values, contrasts, LGN convolution and their firing rates") // TEST 
		("fLGN_gallery", po::value<string>(&LGN_gallery_filename)->default_value("LGN_gallery"),"file that stores spatial and temporal convolution parameters"); // TEST 

	po::options_description cmdline_options;
	cmdline_options.add(generic_opt).add(top_opt);

	po::options_description config_file_options;
	config_file_options.add(generic_opt).add(top_opt);

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
			cout << "cannot find specified config file " << cfg_filename << "\n";
			return EXIT_FAILURE;
		}
	} else {
		cout << "No configuration file is given, default values are used for non-specified parameters\n";
	}

	if (sizeof(Float) == 4) {
		printf("simulating for %u steps, t = %f ms\n", nt, nt*dt);
	} else {
		printf("simulating for %u steps, t = %lf ms\n", nt, nt*dt);
	}

    Size ot;
    if (frameVisV1output||framePhyV1output||frameVisLGNoutput) {
        ot = static_cast<Size>(dot/dt);
        dot = ot*dt;
        cout << " collects framed output every " << dot << "ms, (" << ot << " time steps)\n";
    }

    if (!output_suffix0.empty())  {
        output_suffix = "_" + output_suffix0;
    } else output_suffix = "";
    output_suffix = output_suffix + ".bin";

    if (!conLGN_suffix.empty())  {
        conLGN_suffix = "_" + conLGN_suffix;
    }
    conLGN_suffix = conLGN_suffix + ".bin";

    if (!conV1_suffix.empty())  {
        conV1_suffix = "_" + conV1_suffix;
    }
    conV1_suffix = conV1_suffix + ".bin";

    cout << "conLGN_suffix: " << conLGN_suffix << "\n";
    cout << "conV1_suffix: " << conV1_suffix << "\n";

	if (!restore.empty()) {
		restore = restore + "_" + snapshot_suffix + ".bin";
		if (!asInit) {
			if (snapshot_suffix != output_suffix0) {
				cout << "to use snapshot to resume simulation, it needs to have the same suffix as the output_suffix to work\n";
				return EXIT_FAILURE;
			}
		}
	}

	if (nSpatialSample1D > 32) {
		cout << "nSpatialSample1D has to be smaller than 32 (1024 threads per block).\n";
		return EXIT_FAILURE;
	}

	if (mSpatialSample1D > 32) {
		cout << "mSpatialSample1D has to be smaller than 32 (1024 threads per block).\n";
		return EXIT_FAILURE;
	}

	Size nType;
	vector<Size> typeAccCount;
    vector<Float> synPerConFF;
    vector<Float> synPerCon;
	ifstream fConnectome_cfg(connectome_cfg_filename + conV1_suffix, fstream::in | fstream::binary);
	if (!fConnectome_cfg) {
		cout << "Cannot open or find " << connectome_cfg_filename + conV1_suffix <<"\n";
		return EXIT_FAILURE;
	} else {
		fConnectome_cfg.read(reinterpret_cast<char*>(&nType), sizeof(Size));
		typeAccCount.assign(nType,0);
		synPerCon.assign(nType*nType,0);
		synPerConFF.assign(nType,0);
		fConnectome_cfg.read(reinterpret_cast<char*>(&typeAccCount[0]), nType*sizeof(Size));
		fConnectome_cfg.read(reinterpret_cast<char*>(&synPerCon[0]), nType*nType*sizeof(Float));
		fConnectome_cfg.read(reinterpret_cast<char*>(&synPerConFF[0]), nType*sizeof(Float));

		fConnectome_cfg.close();
	}

	Size nType0;
    Size nArchType = nTypeHierarchy.size();
	if (nArchType > 2) {
		cout << "at least define one type of neuron with nTypeHierarchy.\n";
		return EXIT_FAILURE;
	} else {
		if (nArchType > 2) {
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

    Size nE = typeAccCount[nTypeHierarchy[0] - 1];
	Size nI = blockSize-nE;

	if (nType > max_nType) {
		cout << "the accumulative distribution of neuronal type <typeAccCount> has size of " << nType << " > " << max_nType << "\n";
		return EXIT_FAILURE;
	}

    vector<Size> typeAcc0;
    typeAcc0.push_back(0);
    for (PosInt i=0; i<nType; i++) {
        typeAcc0.push_back(typeAccCount[i]);
    }

    Size ngTypeFF = static_cast<Size>(grFF.size());
	if (ngTypeFF > max_ngTypeFF) {
		cout << "too many types of feed-foward(FF) excitatory conductances, change max_ngTypeFF in CONST.h accordingly and recompile\n"; 
		return EXIT_FAILURE;
	} else {
		if (gdFF.size() != ngTypeFF) {
			cout << "size of decayTimeFF is not consistent with riseTimeFF\n";
			return EXIT_FAILURE;
		}
		if (pFF.size() != nType*ngTypeFF) {
			cout << "size of pFF is not nType*ngTypeFF (" << nType*ngTypeFF << ")\n";
			return EXIT_FAILURE;
		}
		if (gFF0.size() != nType*ngTypeFF*4) {
			cout << "size of gFF0 is not nType*ngTypeFF*4 (g,h,mean,std), (" << nType*ngTypeFF*4 << ")\n";
			return EXIT_FAILURE;
		}
	}

	Size ngTypeE = static_cast<Size>(grE.size());
	if (ngTypeE > max_ngTypeE) {
		cout << "too many types of cortical excitatory conductances, change max_ngTypeE in CONST.h accordingly and recompile\n"; 
		return EXIT_FAILURE;
	} else {
		if (gdE.size() != ngTypeE) {
			cout << "size of decayTimeE is not consistent with riseTimeE\n";
			return EXIT_FAILURE;
		}
		if (pE.size() != nType*ngTypeE) {
			cout << "size of pE is not nType*ngTypeE (" << nType*ngTypeE << ")\n";
			return EXIT_FAILURE;
		}
		if (gE0.size() != nType*ngTypeE*4) {
			cout << "size of gE0 is not nType*ngTypeE*4 (g,h,mean,std), (" << nType*ngTypeE*4 << ")\n";
			return EXIT_FAILURE;
		}
	}

	Size ngTypeI = static_cast<Size>(grI.size());
	if (ngTypeI > max_ngTypeI) {
		cout << "too many types of inhibitory conductances, change max_ngTypeI in CONST.h accordingly and recompile\n"; 
		return EXIT_FAILURE;
	} else {
		if (gdI.size() != ngTypeI) {
			cout << "size of decayTimeI is not consistent with riseTimeI\n";
			return EXIT_FAILURE;
		}
		if (pI.size() != nType*ngTypeI) {
			cout << "size of pI is not nType*ngTypeI (" << nType*ngTypeI << ")\n";
			return EXIT_FAILURE;
		}
		if (gI0.size() != nType*ngTypeI*4) {
			cout << "size of gI0 is not nType*ngTypeI*4 (g,h,mean,std), (" << nType*ngTypeI*4 << ")\n";
			return EXIT_FAILURE;
		}
	}

    bool has_sp0 = false;
	if (!vm.count("spE0")) {
        for (PosInt i=0; i<2*nTypeHierarchy[0]; i++) {
            spE0.push_back(0);
        }
    } else if (spE0.size() != 2*nTypeHierarchy[0]) {
		cout << "the size of spE0 has size of " << spE0.size() << " != " << nTypeHierarchy[0] << "x 2\n";
        return EXIT_FAILURE;
    } else {
        for (PosInt i=0; i<nTypeHierarchy[0]*2; i++) {
            spE0[i] *= dt/1000.0;
        }
        has_sp0 = true;
    }

	if (!vm.count("spI0")) {
        for (PosInt i=0; i<2*nTypeHierarchy[1]; i++) {
            spI0.push_back(0);
        }
    } else if (spI0.size() != 2*nTypeHierarchy[1]) {
		cout << "the size of spI0 has size of " << spI0.size() << " != " << nTypeHierarchy[1] << "x 2\n";
        return EXIT_FAILURE;
    } else {
        for (PosInt i=0; i<nTypeHierarchy[1]*2; i++) {
            spI0[i] *= dt/1000.0;
        }
        has_sp0 = true;
    }

    if (iModel > 1) {
        cout << "model " << iModel << " is not implemented\n";
        return EXIT_FAILURE;
    }

	if (tonicDep.size() == 1) {
		for (PosInt i=1; i<nType; i++) {
			tonicDep.push_back(tonicDep[0]);
		}
	} else {
		if (tonicDep.size() != nType) {
			cout << "size of tonic depolarization (tonicDep) is " << tonicDep.size() << " != " << nType << "\n";
			return EXIT_FAILURE;
		}
	}

	if (noisyDep.size() == 1) {
		for (PosInt i=1; i<nType; i++) {
			noisyDep.push_back(noisyDep[0]);
		}
	} else {
		if (noisyDep.size() != nType) {
			cout << "size of noisy level of the tonic depolarization (noisyDep) is " << noisyDep.size() << " != " << nType << "\n";
			return EXIT_FAILURE;
		}
	}

	if (synFailFF.size() != nType && synFailFF.size() != 1) {
		cout << "the size of synFailFF has size of " << synFailFF.size() << " != " << nType << " or 1.\n";
		return EXIT_FAILURE;
	} else {
		if (synFailFF.size() != nType) {
        	for (PosInt i=1; i<nType; i++) {
				synFailFF.push_back(synFailFF[0]);
			}
		}
	}

	if (synFail.size() != nType*nType) {
		cout << "the size of synFail has size of " << synFail.size() << " != " << nType << " x " << nType << "\n";
		return EXIT_FAILURE;
	} else {
        for (PosInt i=0; i<nType; i++) {
        	for (PosInt j=0; j<nType; j++) {
            	if (synFail[i*nType + j] >= 1.0) cout << "synapse on neuronal type from " << i  << " to " << j << " has been set to always fail\n";
            	if (synFail[i*nType + j] < 0.0) synFail[i*nType + j] = 0.0;
			}
        }
    }

	if (synPerCon.size() != nType*nType) {
		cout << "the size of synPerCon has size of " << synPerCon.size() << " != " << nType << " x " << nType << "\n";
		return EXIT_FAILURE;
	} 

	if (synPerConFF.size() != nType && synPerConFF.size() != 1) {
		cout << "the size of synPerConFF has size of " << synPerConFF.size() << " != " << nType << " or 1.\n";
		return EXIT_FAILURE;
	}

    ConductanceShape condFF(&(grFF[0]), &(gdFF[0]), ngTypeFF);
	ConductanceShape condE(&(grE[0]), &(gdE[0]), ngTypeE);
	ConductanceShape condI(&(grI[0]), &(gdI[0]), ngTypeI);

    if (vR.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            vR.push_back(vR[0]);
        }
    } else {
        if (vR.size() != nType) {
            cout << "vR need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }

    if (vT.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            vT.push_back(vT[0]);
        }
    } else {
        if (vT.size() != nType) {
            cout << "vT need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }

    if (vThres.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            vThres.push_back(vThres[0]);
        }
    } else {
        if (vThres.size() != nType) {
            cout << "vThres need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }

    if (iModel == 0) {
        cout << "using LIF model vT(ignored) is equivalent with vThres.\n";
		vT.clear();
        for (PosInt i=0; i<nType; i++) {
            vT.push_back(vThres[i]);
        }
    }

    if (C.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            C.push_back(C[0]);
        }
    } else {
        if (C.size() != nType) {
            cout << "C need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }

    if (gL.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            gL.push_back(gL[0]);
        }
    } else {
        if (gL.size() != nType) {
            cout << "gL need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }

    if (tRef.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            tRef.push_back(tRef[0]);
        }
    } else {
        if (tRef.size() != nType) {
            cout << "tRef need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }
	if (print_log) {
    	cout << "tRef: ";
        for (PosInt i=0; i<nType; i++) {
			cout << tRef[i];
			if (i == nType -1) cout << "\n";
			else cout << ", ";
        }
	}

    if (deltaT.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            deltaT.push_back(deltaT[0]);
        }
    } else {
        if (deltaT.size() != nType) {
            cout << "deltaT need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }

    if (tau_w.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            tau_w.push_back(tau_w[0]);
        }
    } else {
        if (tau_w.size() != nType) {
            cout << "tau_w (parameter) need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }

    if (a.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            a.push_back(a[0]);
        }
    } else {
        if (a.size() != nType) {
            cout << "a (parameter) need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }
    
    if (b.size() == 1) {
        for (PosInt i=1; i<nType; i++) {
            b.push_back(b[0]);
        }
    } else {
        if (b.size() != nType) {
            cout << "b (parameter) need to has size of " << nType << "\n";
			return EXIT_FAILURE;
        }
    }

    Float *d_vR;
    Float *d_vThres;
    Float *d_gL;
    Float *d_C;
    Float *d_tRef;
    Float *d_vT;
    Float *d_deltaT;
    Float *d_tau_w;
    Float *d_a;
    Float *d_b;
	checkCudaErrors(cudaMalloc((void **) &d_vR, nType*sizeof(Float)));
	checkCudaErrors(cudaMalloc((void **) &d_vThres, nType*sizeof(Float)));
	checkCudaErrors(cudaMalloc((void **) &d_gL, nType*sizeof(Float)));
	checkCudaErrors(cudaMalloc((void **) &d_C, nType*sizeof(Float)));
	checkCudaErrors(cudaMalloc((void **) &d_tRef, nType*sizeof(Float)));
	checkCudaErrors(cudaMalloc((void **) &d_vT, nType*sizeof(Float)));
	checkCudaErrors(cudaMalloc((void **) &d_deltaT, nType*sizeof(Float)));
	checkCudaErrors(cudaMalloc((void **) &d_tau_w, nType*sizeof(Float)));
	checkCudaErrors(cudaMalloc((void **) &d_a, nType*sizeof(Float)));
	checkCudaErrors(cudaMalloc((void **) &d_b, nType*sizeof(Float)));
	checkCudaErrors(cudaMemcpy(d_vR, &(vR[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vThres, &(vThres[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_gL, &(gL[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_C, &(C[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tRef, &(tRef[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_vT, &(vT[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_deltaT, &(deltaT[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tau_w, &(tau_w[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_a, &(a[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, &(b[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));

	size_t usingGMem = nType*10*sizeof(Float);

    // LearnType
    Size nLearnTypeFF;
    LearnVarShapeFF_E_pre lFF_E_pre;
    LearnVarShapeFF_I_pre lFF_I_pre;
    LearnVarShapeFF_E_post lFF_E_post;
    LearnVarShapeFF_I_post lFF_I_post;
    LearnVarShapeE lE;
    LearnVarShapeQ lQ;
	Size nLearnTypeQ;
    if (learning) {
        if (tauAvg.size() > 2) {
            cout << "only E and I having different tauAvg for filtered spike response is implemented\n";
            return EXIT_FAILURE;
        } else {
            if (tauAvg.size() == 1) {
                tauAvg.push_back(tauAvg[0]);
            } 
            if (tauAvg.size() != 2) {
                cout << "tauAvg is not given\n";
                return EXIT_FAILURE;
            }
        }

        if (targetFR.size() > 2) {
            cout << "only E and I having different target firing rate is implemented\n";
            return EXIT_FAILURE;
        } else {
            if (targetFR.size() == 1) {
                targetFR.push_back(targetFR[0]);
            } 
            if (targetFR.size() != 2) {
                cout << "targetFR is not given\n";
                return EXIT_FAILURE;
            }
        }
        // Q
        if (tauQ.size() != A_Q.size() || tauQ.size() != gmaxQ.size() || tauQ.size() != gminQ.size()) {
            cout << "size of tauQ: " << tauQ.size() << " should be the same as A_Q: " << A_Q.size() << "\n";
            return EXIT_FAILURE;
        }
        if (symQ) { // double the size
            nLearnTypeQ = A_Q.size();
            vector<Float> tmpA(A_Q.begin(), A_Q.end());
            vector<Float> tmpTau(tauQ.begin(),tauQ.end());
            vector<Float> tmpGmax(gmaxQ.begin(),gmaxQ.end());
            vector<Float> tmpGmin(gminQ.begin(),gminQ.end());
            A_Q.clear();
            tauQ.clear();
            gmaxQ.clear();
            gminQ.clear();
            for (PosInt i=0; i<nLearnTypeQ; i++) {
                A_Q.push_back(tmpA[i]);
                A_Q.push_back(tmpA[i]);
                tauQ.push_back(tmpTau[i]);
                tauQ.push_back(tmpTau[i]);
                gmaxQ.push_back(tmpGmax[i]);
                gminQ.push_back(tmpGmin[i]);
            }
        } else {
            if (tauQ.size()%2 != 0) {
                cout << "size of array tauQ: " << tauQ.size() << " must be even (or single-value)\n";
                return EXIT_FAILURE;
            } else {
                nLearnTypeQ = static_cast<Size>(tauQ.size())/2;
            }
        }
        // learning
        if (learning == 1) { // default
            cout << "only FF_E, E and Q learning is active, setting nLearnTypeFF_I to 0\n";
            nLearnTypeFF_I = 0;
        }
        // learning == 2 then all learning active
        if (learning == 3) {
            cout << "only FF_E learning is active, setting others to 0\n";
            nLearnTypeFF_I = 0;
            nLearnTypeE = 0;
            nLearnTypeQ = 0;
        }
        if (learning == 4) {
            cout << "only E and Q learning is active, setting others to 0\n";
            nLearnTypeFF_E = 0;
            nLearnTypeFF_I = 0;
        }
        // determine nLearnTypes
        if (nLearnTypeQ > max_nLearnTypeQ) {
            cout << "The size of tauQ (" << nLearnTypeQ << ") should not be larger than max_nLearnTypeQ: " << max_nLearnTypeQ << ", change it in CONST.h and recompile\n";
            return EXIT_FAILURE;
        }
        nLearnTypeFF = nLearnTypeFF_E + nLearnTypeFF_I;
        if (nLearnTypeFF_I > max_nLearnTypeFF_I) {
            cout << "nLearnTypeFF_I: " << nLearnTypeFF_I << " must be smaller than max_nLearnTypeFF_I: " << nLearnTypeFF << " defined in CONST.h, change and recompile\n";
            return EXIT_FAILURE;
        }
        if (nLearnTypeFF_E > max_nLearnTypeFF_E) {
            cout << "nLearnTypeFF_E: " << nLearnTypeFF_E << " must be smaller than max_nLearnTypeFF_I: " << nLearnTypeFF << " defined in CONST.h, change and recompile\n";
            return EXIT_FAILURE;
        }
	    Size nTau_trip = static_cast<Size>(tauTrip.size());
        if (tauLTD.size() != nTau_trip || tauLTP.size() != nTau_trip) {
            cout << "The size of both tauLTD (" << tauLTD.size() << "),and tauLTP (" << tauLTP.size() << ") should be the same as tauTrip: " << nTau_trip << "\n";
            return EXIT_FAILURE;
        }

        if (nTau_trip == 1) { // identical timescale for all types
            if (nLearnTypeFF_E > 1 || nLearnTypeFF_I > 1 || nLearnTypeE > 1) {
                cout << "tauLTP, tauLTD, tauTrip has size of 1, but nLearnTypeFF_E: " << nLearnTypeFF_E << ", nLearnTypeFF_I: " << nLearnTypeFF_I << ", nLearnTypeE: " << nLearnTypeE << " are inconsistent.\n";
                return EXIT_FAILURE;
            }
            for (PosInt i=1; i<nLearnTypeFF + nLearnTypeE; i++) {
                tauLTP.push_back(tauLTP[0]);
                tauLTD.push_back(tauLTD[0]);
                tauTrip.push_back(tauTrip[0]);
            }
        } else {
            if (nLearnTypeE + nLearnTypeFF > nTau_trip) {
                cout << "number of types of cortical triplet learning rules for excitatory neurons, nLearnTypeE: " << nLearnTypeE << " added to the FF triplet ones, nLearnTypeFF " << nLearnTypeFF << " should not be larger than the total number of triplet learning rules " << nTau_trip << "\n";
                return EXIT_FAILURE;
            }
        }

        if (nLearnTypeE > 0) {
            if (gmaxE.size() > nLearnTypeE) {
                cout << "array size of gmaxE cannot be bigger than nLearnTypeE\n";
                return EXIT_FAILURE;
            } else {
                if (gmaxE.size() < nLearnTypeE) {
                    cout << "fill gmaxE with the same value " << gmaxE.back()<< "\n";
                    while(gmaxE.size() < nLearnTypeFF) {
                        gmaxE.push_back(gmaxE.back());
                    }
                }
            }

            if (gmaxE.size() < gminE.size()) {
                cout << "array size of gminE cannot be bigger than gmaxE\n";
                return EXIT_FAILURE;
            } else {
                if (gmaxE.size() > gminE.size()) {
                    Float value = gminE.size() == 1? gminE.back(): 0;
                    cout << "fill gminE with the same value" << value << "\n";
                    while (gminE.size() < gmaxE.size()) {
                        gminE.push_back(0);
                    }
                }
            }
        }

        if (nLearnTypeFF > 0) {
            if (gmaxLGN.size() > nLearnTypeFF) {
                cout << "array size of gmaxLGN cannot be bigger than nLearnTypeFF\n";
                return EXIT_FAILURE;
            } else {
                if (gmaxLGN.size() < nLearnTypeFF) {
                    cout << "fill gmaxLGN with the same value " << gmaxLGN.back()<< "\n";
                    while(gmaxLGN.size() < nLearnTypeFF) {
                        gmaxLGN.push_back(gmaxLGN.back());
                    }
                }
            }

            if (gmaxLGN.size() < gminLGN.size()) {
                cout << "array size of gminLGN cannot be bigger than gmaxLGN\n";
                return EXIT_FAILURE;
            } else {
                if (gmaxLGN.size() > gminLGN.size()) {
                    Float value = gminLGN.size() == 1? gminLGN.back(): 0;
                    cout << "fill gminLGN with the same value" << value << "\n";
                    while (gminLGN.size() < gmaxLGN.size()) {
                        gminLGN.push_back(0);
                    }
                }
            }
        }

        if (nLearnTypeFF_E) learnFF_pre<LearnVarShapeFF_E_pre>(lFF_E_pre, &(tauLTP[0]), nLearnTypeFF_E);
        if (nLearnTypeFF_I) learnFF_pre<LearnVarShapeFF_I_pre>(lFF_I_pre, &(tauLTP[nLearnTypeFF_E]), nLearnTypeFF_I);
        if (nLearnTypeFF_E) learnFF_post<LearnVarShapeFF_E_post>(lFF_E_post, &(tauLTD[0]), &(tauTrip[0]), tauAvg[0], targetFR[0], &(A_LGN[0]), gmaxLGN[0], gminLGN[0], nLearnTypeFF_E, sRatioLGN[0]);
        if (nLearnTypeFF_I) learnFF_post<LearnVarShapeFF_I_post>(lFF_I_post, &(tauLTD[nLearnTypeFF_E]), &(tauTrip[nLearnTypeFF_E]), tauAvg[1], targetFR[1], &(A_LGN[nLearnTypeFF_E]), gmaxLGN[nLearnTypeFF_E], gminLGN[nLearnTypeFF_E], nLearnTypeFF_I, sRatioLGN[1]);

        if (nLearnTypeFF_E) printFF_pre<LearnVarShapeFF_E_pre>(lFF_E_pre, 1);
        if (nLearnTypeFF_I) printFF_pre<LearnVarShapeFF_I_pre>(lFF_I_pre, 0);
        if (nLearnTypeFF_E) printFF_post<LearnVarShapeFF_E_post>(lFF_E_post, 1);
        if (nLearnTypeFF_I) printFF_post<LearnVarShapeFF_I_post>(lFF_I_post, 0);

        if (nLearnTypeE) learnE(lE, &(tauLTP[nLearnTypeFF]), &(tauLTD[nLearnTypeFF]), &(tauTrip[nLearnTypeFF]), tauAvg[0], targetFR[0], &(A_V1[0]), gmaxE[0], gminE[0], nLearnTypeE, sRatioV1);
        if (nLearnTypeQ) learnQ(lQ, &(tauQ[0]), &(A_Q[0]), gmaxQ[0], gminQ[0], nLearnTypeQ, sRatioV1);

        if (nLearnTypeE) printE(lE);
        if (nLearnTypeQ) printQ(lQ);
    } else {
        nLearnTypeFF_E = 0;
        nLearnTypeFF_I = 0;
        nLearnTypeFF = 0;
        nLearnTypeE = 0;
        nLearnTypeQ = 0;
    }

	// precheck
	bool simpleContrast = true; // TODO : implement this for comparison no cone adaptation if set to true
	Size stepRate = std::round(1000/dt);
	if (std::round(1000/dt) != 1000/dt) {
		cout << "stepRate = " << 1000/dt << "Hz, but it has to be a integer.\n";
		return EXIT_FAILURE;
	}
	if (frameRate > stepRate) {
		cout << "stepRate cannot be smaller than frameRate, 1000/dt.\n";
		return EXIT_FAILURE;
	}

	Size nRetrace = static_cast<Size>(std::round(tau/dt));
	if (std::round(tau/dt) != tau/dt) {
		cout << "tau: " << tau << " should be divisible by dt: " << tau << ".\n"; 
		return EXIT_FAILURE;
	}
	if (nRetrace % nKernelSample != 0) {
		cout << "parvo tau: " << nRetrace << " in #dt should be divisible by nKernelSample: " << nKernelSample << ".\n"; 
		return EXIT_FAILURE;
	}

	Size mRetrace = static_cast<Size>(std::round(mau/dt));
	if (std::round(mau/dt) != mau/dt) {
		cout << "magno tau (mau): " << mau << " should be divisible by dt: " << mau << ".\n"; 
		return EXIT_FAILURE;
	}
	if (mRetrace % mKernelSample != 0) {
		cout << "magno tau: " << mRetrace << " in #dt should be divisible by mKernelSample: " << mKernelSample << ".\n"; 
		return EXIT_FAILURE;
	}

	// from the retina facing out
	ifstream fStimulus, fLGN_switch; // ext. inputs
	ifstream fV1_RF, fV1_feature; // V1 related
	ifstream fLGN_vpos; // LGN VF pos 
	fstream fLGN; // LGN properties
	ofstream fLGN_fr; // outputs
	ofstream fLGN_sp;
	ofstream fLGN_gallery, fOutputB4V1;
	ofstream fRawData, fOutputFrame, fLearnData_FF, f_sLGN;

	float init_L, init_M, init_S;
	Size width;
	Size height;
	Size nFrame;
	fStimulus.open(stimulus_filename, fstream::in | fstream::binary);
	Size nPixelPerFrame;
	Float stimulus_buffer, stimulus_range; // see below 
	if (!fStimulus) {
		cout << "cannot open " << stimulus_filename << "\n";
		return EXIT_FAILURE;
	} else {
		vector<Size> stimulus_dimensions(3, 0); 
		fStimulus.read(reinterpret_cast<char*>(&stimulus_dimensions[0]), 3 * sizeof(Size));
		nFrame = stimulus_dimensions[0];
		height = stimulus_dimensions[1];
		width = stimulus_dimensions[2];
		cout << "stimulus: " << nFrame << " frames of " << width << "x" << height << "\n";
		if (width > 16384 || height > 16384) {
			cout << "a single layered texture object is limited to 16384x16384\n";
			return EXIT_FAILURE;
		}
		fStimulus.read(reinterpret_cast<char*>(&init_L), sizeof(float));
		fStimulus.read(reinterpret_cast<char*>(&init_M), sizeof(float));
		fStimulus.read(reinterpret_cast<char*>(&init_S), sizeof(float));
	}
	vector<float> domain(2, 0); 
	nPixelPerFrame = width*height;
	fStimulus.read(reinterpret_cast<char*>(&domain[0]), 2 * sizeof(float));
	stimulus_buffer = static_cast<Float>(domain[0]);
	stimulus_range = static_cast<Float>(domain[1]);
	streampos sofStimulus = fStimulus.tellg();
	fStimulus.seekg(0, fStimulus.end);
	streampos eofStimulus = fStimulus.tellg();
	fStimulus.seekg(sofStimulus);
	// in each frame
	// stimulus start from 0 -> ecc
	/*			
				left-eye		right-eye
				1  _______________ ______________
				^ |b-------------|b-------------| <- buffer(b)  
				| |b|			 |b|			|  
				| |b|			 |b|			| 
				| |b|			 |b|			| 
				| |b|			 |b|			| 
				| |b|			 |b|			| 
				| |b|*		     |b|*		  	|  2 x range
				| |b|			 |b|			| 
				| |b|			 |b|			| 
				| |b|			 |b|			| 
				| |b|			 |b|			| 
				| |b|____________|b|____________|
				0 |b-------------|b-------------| <- buffer(b)
				0------------------------------>1
				  |b| <- range ->|b| <- range ->|				 
	 */
    /* flat_retina:
				            left-eye (cyclop)
				1  ______________________________
				^ |b--------------------------|b| <- buffer(b)  
				| |b|			 			  |b|  
				| |b|			 			  |b| 
				| |b|			 			  |b| 
				| |b|			 			  |b| 
				| |b|			 			  |b| 
				| |b|		     *		  	  |b|  range
				| |b|			 			  |b| 
				| |b|			 			  |b| 
				| |b|			 			  |b| 
				| |b|			 			  |b| 
				| |b|_________________________|b|
				0 |b---------------------------b| <- buffer(b)
				0------------------------------>1
				  |b|    <--   range   -->    |b|		 
    */

	// DON'T close file, still being read during simulation

	Float deg2rad = M_PI/180.0;
	// Float rad2deg = 180.0/M_PI;
	Size nLGN_I, nLGN_C, nLGN;
	Float max_ecc, L_x0, L_y0, R_x0, R_y0, normViewDistance;
	fLGN_vpos.open(LGN_vpos_filename, fstream::in | fstream::binary);
	if (!fLGN_vpos) {
		cout << "Cannot open or find " << LGN_vpos_filename <<" to read in LGN properties.\n";
		return EXIT_FAILURE;
	} else {
		fLGN_vpos.read(reinterpret_cast<char*>(&nLGN_I), sizeof(Size));
		fLGN_vpos.read(reinterpret_cast<char*>(&nLGN_C), sizeof(Size));
		float max_ecc0;
		fLGN_vpos.read(reinterpret_cast<char*>(&max_ecc0), sizeof(float)); // in rad
		max_ecc = static_cast<Float>(max_ecc0);
	}
	nLGN = nLGN_I + nLGN_C;
    Float normEccMaxStimulus_extent;
	if (max_ecc > stimulus_range) {
		cout << "max_ecc: " << max_ecc << "must not be larger than " << stimulus_range << "\n";
        return EXIT_FAILURE;
	}
    
    Size nStatus;
    vector<Float> LGN_status;
    vector<Float> LGN_sDur;
    vector<Size> LGN_switchIt;
    if (LGN_switch) {
        fLGN_switch.open(LGN_switch_filename + conLGN_suffix, fstream::in | fstream::binary);
        if (!fLGN_switch) {
		    cout << "Cannot open or find " << LGN_switch_filename + conLGN_suffix <<" to read in LGN properties.\n";
		    return EXIT_FAILURE;
        } else {
            fLGN_switch.read(reinterpret_cast<char*>(&nStatus), sizeof(Size));
            LGN_status.assign(nStatus*nInputType,0);
            LGN_sDur.assign(nStatus,0.0);
            LGN_switchIt.assign(nStatus,0);
			vector<float> LGN_status0(nInputType*nStatus);
		    fLGN_switch.read(reinterpret_cast<char*>(&LGN_status0[0]), nInputType*nStatus*sizeof(float));
			for (PosInt i=0; i<nInputType*nStatus; i++) {
				LGN_status[i] = static_cast<Float>(LGN_status0[i]);
			}
			vector<float> LGN_sDur0(nStatus);
		    fLGN_switch.read(reinterpret_cast<char*>(&LGN_sDur0[0]), nStatus*sizeof(float));
			for (PosInt i=0; i<nStatus; i++) {
				LGN_sDur[i] = static_cast<Float>(LGN_sDur0[i]);
			}
			vector<float>().swap(LGN_sDur0);
			vector<float>().swap(LGN_status0);
            Float totalDur = accumulate(LGN_sDur.begin(), LGN_sDur.end(), 0.0);
            for (PosInt i = 1; i < nStatus; i++) {
                LGN_switchIt[i] = static_cast<Size>(std::floor(LGN_sDur[i]/totalDur*nt)) + LGN_switchIt[i-1];
            }
            cout << "number of status change: " << nStatus << ": " << totalDur << "\n";
            for (PosInt i = 0; i < nStatus; i++) {
                cout << "starting tstep: " << LGN_switchIt[i] << "\n";
                for (PosInt j=0; j<nInputType; j++) {
                    cout << LGN_status[i*nInputType + j] << "; " << LGN_sDur[i];
                    if (j<nInputType-1) cout << ", ";
                    else  cout << "\n";
                }
            }
        }
    }

    if (!flat_retina) {
	    Float stimulus_extent = stimulus_range + 2*stimulus_buffer;
	    normEccMaxStimulus_extent = max_ecc/(2*stimulus_extent); // origin at left boundary, just the ecc at VF center, its surround can be much bigger, normalized for texture coordinates
	    // normalized stimulus reading points for stimulus access
        // (L_x0, L_y0) is the visual center of the left eye 
        // (R_x0, R_y0) .. right..
	    L_x0 = stimulus_buffer/(2*stimulus_extent);
	    L_y0 = 0.5;
	    R_x0 = 0.5 + L_x0;
	    R_y0 = 0.5;
	    normViewDistance = normEccMaxStimulus_extent/tan(max_ecc*deg2rad);
    } else {
	    Float stimulus_extent = 2*stimulus_range + 2*stimulus_buffer;
        normEccMaxStimulus_extent = max_ecc/stimulus_extent; // origin at center, just the ecc at VF center, its surround can be much bigger, normalized for texture coordinates
	    L_x0 = 0.5;
	    L_y0 = 0.5;
	    R_x0 = 1.0; // can be arbitrary, no right eye LGN
	    R_y0 = 1.0;
	    normViewDistance = normEccMaxStimulus_extent/max_ecc/deg2rad; // since uniform, just a scale
    }

	cout << nLGN << " LGN neurons, " << nLGN_I << " from the ipsilateral eye, " << nLGN_C << " from the contralateral eye, LGN center positions are within the eccentricity of " << max_ecc << " deg, reaching normalized stimulus radius of " << normEccMaxStimulus_extent << "\n";
	cout << "normalized view distance: " << normViewDistance << "\n";

	Size nRegion = 2;
	Size mRegion = 1;
	Size nSample = nSpatialSample1D * nSpatialSample1D;
	Size mSample = mSpatialSample1D * mSpatialSample1D;
	if (nKernelSample == 0) {
		cout << "kernel sampling points: " << nKernelSample << " must be positive integer.\n"; 
		return EXIT_FAILURE;
	}
	// TODO: reduce memory usage if no distribution
	printf("=== temporal storage memory required: %dx%dx%d = %fMB\n", nKernelSample,nLGN,nRegion,nKernelSample*nLGN*nRegion*sizeof(Float)/1024.0/1024.0);
	printf("=== spatial storage memory required: %dx%d = %fMB\n", nSpatialSample1D, nSpatialSample1D, nSample*sizeof(Float)/1024.0/1024.0);
	printf("=== texture coord storage memory required: %dx%dx%d = %fMB\n", nSample, nLGN, nRegion, 2*nSample*nRegion*nLGN*sizeof(float)/1024.0/1024.0);

	// vectors to initialize LGN
	// center surround, thus the x2
	vector<InputType> LGNtype(nLGN);
	vector<Float> LGN_polar(nLGN*2);
	vector<Float> LGN_ecc(nLGN*2);
	vector<Float> LGN_rw(nLGN*2);
	vector<Float> LGN_rh(nLGN*2);
	vector<Float> LGN_orient(nLGN*2);

	vector<Float> LGN_k(nLGN*2);
	vector<Float> ratio(nLGN*2);
	vector<Float> tauR(nLGN*2);
	vector<Float> tauD(nLGN*2);
	vector<Float> nR(nLGN*2);
	vector<Float> nD(nLGN*2);
	vector<Float> delay(nLGN*2);

	vector<Float> c50(nLGN);
	vector<Float> sharpness(nLGN);
	vector<Float> spont(nLGN);
	vector<PosInt> coneType(nLGN*2);
	vector<Float> covariant(nLGN);

	// 
	vector<float> LGN_xx(nLGN);
	vector<float> LGN_yy(nLGN);
	float _LGN_x0, _LGN_xspan;
	float _LGN_y0, _LGN_yspan;

	vector<Float> LGN_x(nLGN);
	vector<Float> LGN_y(nLGN);
	Float LGN_x0, LGN_xspan;
	Float LGN_y0, LGN_yspan;
	fLGN_vpos.read(reinterpret_cast<char*>(&_LGN_x0), sizeof(float));
	fLGN_vpos.read(reinterpret_cast<char*>(&_LGN_xspan), sizeof(float));
	fLGN_vpos.read(reinterpret_cast<char*>(&_LGN_y0), sizeof(float));
	fLGN_vpos.read(reinterpret_cast<char*>(&_LGN_yspan), sizeof(float));

	fLGN_vpos.read(reinterpret_cast<char*>(&LGN_xx[0]), nLGN*sizeof(float));
	fLGN_vpos.read(reinterpret_cast<char*>(&LGN_yy[0]), nLGN*sizeof(float));

	LGN_x0 = static_cast<Float>(_LGN_x0);
	LGN_xspan = static_cast<Float>(_LGN_xspan);
	LGN_y0 = static_cast<Float>(_LGN_y0);
	LGN_yspan = static_cast<Float>(_LGN_yspan);
	for (PosInt i=0; i<nLGN; i++) {
		LGN_x[i] = static_cast<Float>(LGN_xx[i]);
		LGN_y[i] = static_cast<Float>(LGN_yy[i]);
	}
	vector<float>().swap(LGN_xx);
	vector<float>().swap(LGN_yy);

    cout << "LGN_x: [" << *min_element(LGN_x.begin(), LGN_x.end()) << ", " << *max_element(LGN_x.begin(), LGN_x.end()) << "]\n";
    cout << "LGN_y: [" << *min_element(LGN_y.begin(), LGN_y.end()) << ", " << *max_element(LGN_y.begin(), LGN_y.end()) << "]\n";

	fLGN_vpos.read(reinterpret_cast<char*>(&LGNtype[0]), nLGN*sizeof(InputType_t));

	auto get_incLowBound = [](Float thres) {
		function<bool(Float)> bound = [thres] (Float value) {
			return value < thres;
		};
		return bound;
	};

	Size nParvo = 0;
    Size nMagno = 0;
    Size nParvo_I = 0;
    Size nParvo_C = 0;
    Size nMagno_I = 0;
    Size nMagno_C = 0;
	// parvo_I, magno_I, parvo_C, magno_C
    for (PosInt i=0; i<nLGN; i++) {
        if (static_cast<InputType_t>(LGNtype[i]) < 4) {
            nParvo++;
            if (i < nLGN_I) {
                nParvo_I++;
                assert(nParvo_I == i+1);
            } else {
                nParvo_C++;
                assert(nParvo_C == i+1-nLGN_I);
            }
        } else {
            nMagno++;
            if (i < nLGN_I) {
                nMagno_I++;
                assert(nMagno_I == i+1-nParvo_I);
            } else {
                nMagno_C++;
                assert(nMagno_C == i+1-nLGN_I-nParvo_C);
            }
        }
    }
    cout << nParvo << " parvo-cellular LGN (" << nParvo_I << ", " << nParvo_C << ") and " << nMagno << " (" << nMagno_I << ", " << nMagno_C << ") magno-cellular LGN\n";


	if (useNewLGN) { // Setup LGN here 
		cout << "initializing LGN spatial parameters...\n";
		// TODO: k, rx, ry, surround_x, surround_y or their distribution parameters readable from file

		vector<float> LGN_polar0(nLGN);
		vector<float> LGN_ecc0(nLGN);
		fLGN_vpos.read(reinterpret_cast<char*>(&LGN_polar0[0]), nLGN*sizeof(float));
		// polar is in rad
		fLGN_vpos.read(reinterpret_cast<char*>(&LGN_ecc0[0]), nLGN*sizeof(float));
		for (PosInt i=0; i<nLGN; i++) {
			LGN_polar[i] = static_cast<Float>(LGN_polar0[i]);
			LGN_ecc[i] = static_cast<Float>(LGN_ecc0[i]);
		}

		vector<float>().swap(LGN_polar0);
		vector<float>().swap(LGN_ecc0);
		// ecc of center is in deg [0, nLGN), surround to generate in rad [nLGN, 2*nLGN)
		auto transform_deg2rad = [deg2rad] (Float ecc) {return ecc*deg2rad;};
		transform(LGN_ecc.begin(), LGN_ecc.begin()+nLGN, LGN_ecc.begin(), transform_deg2rad);
		fLGN_vpos.close();

        // Magno-cellular have no sustain response, std is arbitarily set
        Float K_onMagno[2] = {60.0f * frRatioLGN, 0.1f};
        Float ratio_onMagno[2] = {1.0f, 0.1f};
        Float tauR_onMagno[2] = {1.5f, 0.1f};
        Float tauD_onMagno[2] = {4.5f, 0.1f};
        Float nR_onMagno[2] = {19.0f, 0.1f};
        Float nD_onMagno[2] = {9.5f, 0.1f};
        Float delay_onMagno[2] = {14.0f, 0.1f};

        Float K_offMagno[2] = {60.0f * frRatioLGN, 0.1f};
        Float ratio_offMagno[2] = {1.0f, 0.1f};
        Float tauR_offMagno[2] = {1.5f, 0.1f};
        Float tauD_offMagno[2] = {4.5f, 0.1f};
        Float nR_offMagno[2] = {19.0f, 0.1f};
        Float nD_offMagno[2] = {9.5f, 0.1f};
        Float delay_offMagno[2] = {14.0f, 0.1f};

        // ON-CENTER temporal mu and c.v.
		// LGN_on = LGN_all * P_on/P_all
		Float K_onC[2] = {29.13289963f*frRatioLGN, 0.60*0.52/0.48}; //A spatially-integrated K
		Float ratio_onC[2] = {0.56882181, 0.18*0.11/0.13}; //Hs

		Float nR_onC[2] = {23.20564706, 0.28*0.18/0.23};
		Float nR_onClowBound = 1-(1-20/36.8)*0.28/0.23;
		Float nR_onCupBound =  1+(46/36.8-1)*0.28/0.23;
		Float tauR_onC = 1.89874285; // controlled by tspR/(nR_onC-1)
		Float tspR_onC[2]; 
		tspR_onC[0] = (nR_onC[0]-1) * tauR_onC;
		tspR_onC[1] = 0.08*0.09/0.09; // NL*TauL, table2, table 5

		Float tauD_onC[2] = {6.34054054, 0.52*0.45/0.59}; //tauS
		// nD'distribution will be controlled by tspD/tauD, with tspD from tspD_dist
		//Float nD_onC[2] = {12.56210526, 0.28*0.18/0.23}; //NL

		// OFF-CENTER temporal mu and c.v.
		Float K_offC[2] = {22.71914498f*frRatioLGN, 0.60*0.37/0.48}; // A [Hz per unit contrast (100% or 1%?)]
		Float ratio_offC[2] = {0.60905210, 0.18*0.14/0.13}; //Hs

		Float nR_offC[2] = {16.70023529, 0.28*0.20/0.23};
		Float nR_offClowBound = 1-(1-21/27.64)*0.28/0.23;
		Float nR_offCupBound =  1+(41/27.64-1)*0.28/0.23;
		Float tauR_offC = 2.78125714;
		Float tspR_offC[2]; 	
		tspR_offC[0] = (nR_offC[0]-1) * tauR_offC;
		tspR_offC[1] = 0.08*0.09/0.09; // NL*TauL, table 5

		Float tauD_offC[2] = {7.71891892, 0.52*0.62/0.59}; //tauS
		// nD'distribution will be controlled by tspD/tauD, with tspD from tspD_dist
		//Float nD_offC[2] = {11.33052632, 0.28*0.20/0.23}; //NL

		// delay from stimulus onset to LGN with mu and std.
		Float delay_onC[2] = {13.45, square_root(2.39*2.39+0.51*0.51)};
		Float delay_offC[2] = {13.45, square_root(2.39*2.39+0.61*0.61)};

		// ON-SURROUND temporal mu and c.v.
		Float K_onS[2] = {23.17378917f*frRatioLGN,  1.45*1.53/1.64}; // A
		Float ratio_onS[2] = {0.36090931, 0.57*0.74/0.64}; //Hs

		Float nR_onS[2] = {50.29602888, 0.35*0.12/0.38};
		Float nR_onSlowBound = 1-(1-98/111.5)*0.35/0.38;
		Float nR_onSupBound = 1+(133/111.5-1)*0.35/0.38;
		Float tauR_onS = 0.90455076;
		Float tspR_onS[2];
		tspR_onS[0] = (nR_onS[0]-1) * tauR_onS;
		tspR_onS[1] = 0.09*0.10/0.08; // NL*TauL, table2, table 5

		//Float nD_onS[2] = {22.54098361, 0.35*0.12/0.38}; //NL
		Float tauD_onS[2] = {3.05084745, 0.90*1.08/0.94}; //tauS

		// OFF-surround temporal mu and c.v.
		Float K_offS[2] = {12.53276353f*frRatioLGN, 1.45*1.84/1.64}; // A
		Float ratio_offS[2] = {0.33004402, 0.57*0.59/0.64}; //Hs

		Float nR_offS[2] = {26.23465704, 0.35*0.53/0.38};
		Float nR_offSlowBound = 1-(1-45/78.88)*0.35/0.38;
		Float nR_offSupBound = 1+(168/78.88-1)*0.35/0.38;
		Float tauR_offS = 1.83570595;
		Float tspR_offS[2]; 
		tspR_offS[0] = (nR_offS[0]-1) * tauR_offS;
		tspR_offS[1] = 0.09*0.06/0.08; // NL*TauL, table2, table 5

		//Float nD_offS[2] = {8.29508197, 0.35*0.53/0.38}; //NL
		Float tauD_offS[2] = {9.66101695, 0.90*0.71/0.94}; //tauS

		// delay from stimulus onset to LGN with mu and std.
		Float delay_onS[2] = {22.06611570, square_root(1.48*1.48+0.51*0.51)};
		Float delay_offS[2] = {26.68388430, square_root(1.48*1.48+0.61*0.61)};
		{ // c.v. to std
			K_onC[1] = K_onC[0] * K_onC[1];
			ratio_onC[1] = ratio_onC[0] * ratio_onC[1];
			nR_onC[1] = nR_onC[0] * nR_onC[1];
			tspR_onC[1] = tspR_onC[0] * tspR_onC[1];
			tauD_onC[1] = tauD_onC[0] * tauD_onC[1];
			K_offC[1] = K_offC[0] * K_offC[1];
			ratio_offC[1] = ratio_offC[0] * ratio_offC[1];
			nR_offC[1] = nR_offC[0] * nR_offC[1];
			tspR_offC[1] = tspR_offC[0] * tspR_offC[1];
			tauD_offC[1] = tauD_offC[0] * tauD_offC[1];

			K_onS[1] = K_onS[0] * K_onS[1];
			ratio_onS[1] = ratio_onS[0] * ratio_onS[1];
			nR_onS[1] = nR_onS[0] * nR_onS[1];
			tspR_onS[1] = tspR_onS[0] * tspR_onS[1];
			tauD_onS[1] = tauD_onS[0] * tauD_onS[1];
			K_offS[1] = K_offS[0] * K_offS[1];
			ratio_offS[1] = ratio_offS[0] * ratio_offS[1];
			nR_offS[1] = nR_offS[0] * nR_offS[1];
			tspR_offS[1] = tspR_offS[0] * tspR_offS[1];
			tauD_offS[1] = tauD_offS[0] * tauD_offS[1];
		}
		
		auto tspD_dist = [](Float n, Float tau) {
			Float tsp = (n-1)*tau;
			Float stride = n*tau/logrithm(n);
			Float mean = tsp + stride * 1.5;
			Float std = stride/6.0;
			return make_pair(mean, std);
		};
		//     sigma*sqrt(2)
		Float acuityC[2] = {0.03f*deg2rad, 0.01f*deg2rad/1.349f}; // interquartile/1.349 = std 
		Float acuityS[2] = {0.18f*deg2rad, 0.07f*deg2rad/1.349f};
		Float acuityC_M[2] = {0.06f*deg2rad, 0.02f*deg2rad/1.349f}; // interquartile/1.349 = std 
		Float acuityS_M[2] = {0.18f*deg2rad, 0.06f*deg2rad/1.349f};
		Float pTspD[2];
		Float tspR, tspD;
        Float sharpness_dist[2] = {1.0, 0.1};
        Float sharpness_distM[2] = {10.0, 1.0};
        Float c50_dist[2] = {0.5, 0.06}; //
        Float c50_distM[2] = {0.25, 0.03}; //
        //Float sharpness_dist[2] = {1.0, 1.0};
        //Float c50_dist[2] = {0.5, 0.03}; //
		Float spontPercentUL = 0.15; // 0.15
		Float spontPercent = 0.1; // 0.05
        //============
	    auto get_excLowBound = [](Float thres) {
	    	function<bool(Float)> bound = [thres] (Float value) {
	    		return value <= thres;
	    	};
	    	return bound;
	    };
	    auto positiveBound = get_excLowBound(0.0);
        default_random_engine rGen_LGNsetup(seed);
		seed++; // so that next random_engine use a different seed;
	    // lambda to return a function that generates random numbers from a given distribution
	    auto get_rand_from_gauss0 = [](default_random_engine &rGen, normal_distribution<Float> &dist, function<bool(Float)> &outOfBound) {
	    	function<Float()> get_rand = [&rGen, &dist, &outOfBound] () {
	    		Float rand;
	    		Size count = 0;
	    		do {
	    			rand = dist(rGen);
	    			count++;
	    			if (count > 10) {
	    				cout << rand << "\n";
	    			}
	    			if (count > 20) {
	    				assert(count <= 20);
	    			}
	    		} while (outOfBound(rand));
	    		return rand;
	    	};
	    	return get_rand;
	    };

        function<bool(Float, Float)> larger = [](Float v1, Float v2) {
            return v1<=v2 || v2<0.1*v1; // i.e., requring 0.1*v1 <= v2 < v1
        };
        function<bool(Float, Float)> smaller = [](Float v1, Float v2) {
            return v1>=v2;
        };
		// set test param for LGN subregion RF spatial kernel 
		// Croner and Kaplan 1995
		// center and surround RF size correlates
        Float maxSecc, minSecc, maxSpolar, minSpolar;
        Float maxCecc, minCecc, maxCpolar, minCpolar;
        if (!uniform_LGN) {
		    Float rho_SC = 0.8; // empirical guess
		    Float rho_SC_comp = sqrt(1.0-rho_SC*rho_SC);
            for (PosInt i=0; i<nLGN; i++) {
                switch (LGNtype[i]) {
		    		case InputType::MonLoff: case InputType::MoffLon: case InputType::LonMoff: case InputType::LoffMon:
		                tie(LGN_rw[i], LGN_rw[i+nLGN]) = get_rands_from_correlated_gauss(acuityC, acuityS, rho_SC, rho_SC_comp, rGen_LGNsetup, rGen_LGNsetup, positiveBound, positiveBound, smaller);
		    			break;
                    case InputType::OnOff: case InputType::OffOn:
		                tie(LGN_rw[i], LGN_rw[i+nLGN]) = get_rands_from_correlated_gauss(acuityC_M, acuityS_M, rho_SC, rho_SC_comp, rGen_LGNsetup, rGen_LGNsetup, positiveBound, positiveBound, smaller);
                        break;
                    default:
		    			throw("There's no implementation of this type with non-uniform LGN");
                }
            }
            // ry ~ rx circular sub RF, not necessary
	        normal_distribution<Float> norm(1.0, 1.0/30.0);
		    auto add_noise = [&norm, &rGen_LGNsetup](Float input) {
		    	return input * norm(rGen_LGNsetup);
		    };
		    transform(LGN_rw.begin(), LGN_rw.end(), LGN_rh.begin(), add_noise);
            // orientation of LGN RF
		    auto uniform = uniform_real_distribution<Float>(0, M_PI);
		    std::function<Float()> get_rand_from_uniform = [&rGen_LGNsetup, &uniform] () {
		    	return uniform(rGen_LGNsetup);
		    };
		    generate(LGN_orient.begin(), LGN_orient.end(), get_rand_from_uniform);

            // surround origin shift
		    function<bool(Float)> noBound = [] (Float value) {
		    	return false;
		    };

		    norm = normal_distribution<Float>(0.0, 1.0/3.0/sqrt(2)); // within a 1/3 of the std of the gauss0ian
		    auto get_rand = get_rand_from_gauss0(rGen_LGNsetup, norm, noBound);
		    for (Size i=0; i<nLGN; i++) {
		    	// not your usual transform 
		    	Float eta = LGN_rw[i]*get_rand();
		    	Float intermediateEcc = LGN_ecc[i]+LGN_rh[i]*get_rand();
                if (!flat_retina) {
		    	    orthPhiRotate3D_arc(LGN_polar[i], intermediateEcc, eta, LGN_polar[i+nLGN], LGN_ecc[i+nLGN]);
                } else {
                    LGN_ecc[i+nLGN] = intermediateEcc;    
                    LGN_polar[i+nLGN] = LGN_polar[i] + eta/intermediateEcc;
                }
            }
        } else {
            // no orientation prefence
            // no surrond origin shift;
            for (PosInt i=0; i<nLGN; i++) {
                switch (LGNtype[i]) {
                    case InputType::MonLoff: case InputType::MoffLon: case InputType::LonMoff: case InputType::LoffMon:
                        LGN_rw[i] = acuityC[0];
                        LGN_rw[i+nLGN] = acuityS[0];
                        LGN_rh[i] = acuityC[0];
                        LGN_rh[i+nLGN] = acuityS[0];
                        break;
                    case InputType::OnOff: case InputType::OffOn:
                        LGN_rw[i] = acuityC_M[0];
                        LGN_rw[i+nLGN] = acuityS_M[0];
                        LGN_rh[i] = acuityC_M[0];
                        LGN_rh[i+nLGN] = acuityS_M[0];
                        break;
                }
                LGN_ecc[i+nLGN] = LGN_ecc[i];
                LGN_polar[i+nLGN] = LGN_polar[i];
            }
        }
		for (Size i=0; i<nLGN; i++) {
            if (i==0) {
                minSecc = LGN_ecc[i+nLGN];
                maxSecc = LGN_ecc[i+nLGN];
                minSpolar = LGN_polar[i+nLGN];
                maxSpolar = LGN_polar[i+nLGN];
                minCecc = LGN_ecc[i];
                maxCecc = LGN_ecc[i];
                minCpolar = LGN_polar[i];
                maxCpolar = LGN_polar[i];
            } else {
                if (LGN_ecc[i+nLGN] < minSecc) minSecc = LGN_ecc[i+nLGN];
                if (LGN_polar[i+nLGN] < minSpolar) minSpolar = LGN_polar[i+nLGN];
                if (LGN_ecc[i+nLGN] > maxSecc) maxSecc = LGN_ecc[i+nLGN];
                if (LGN_polar[i+nLGN] > maxSpolar) maxSpolar = LGN_polar[i+nLGN];
                if (LGN_ecc[i] < minCecc) minCecc = LGN_ecc[i];
                if (LGN_polar[i] < minCpolar) minCpolar = LGN_polar[i];
                if (LGN_ecc[i] > maxCecc) maxCecc = LGN_ecc[i];
                if (LGN_polar[i] > maxCpolar) maxCpolar = LGN_polar[i];
            }
		}
        cout << "Cen ecc: " <<  minCecc*180/M_PI << ", " << maxCecc*180/M_PI << "\n";
        cout << "Cen polar: " <<  minCpolar*180/M_PI << ", " << maxCpolar*180/M_PI << "\n";
        cout << "Sur ecc: " << minSecc*180/M_PI << ", " <<  maxSecc*180/M_PI << "\n";
        cout << "Sur polar: " << minSpolar*180/M_PI << ", " << maxSpolar*180/M_PI << "\n";

        auto get_rand_from_clipped_gauss = [](Float param[2], Float lb, Float ub) {
			assert(lb < 1);
			assert(ub > 1);
			//function<Float(RandomEngine)> get_rand = [param, lb, ub](RandomEngine &rGen) {
			auto get_rand = [param, lb, ub](RandomEngine &rGen) {
				static normal_distribution<Float> norm(0.0, 1.0);
				Size count = 0;
				Float v;
				do {
					v = norm(rGen)*param[1] + param[0];
					count++;
					if (count > 10) {
						std::cout << "mean: " << param[0] << ", std: " << param[1] << "\n";
						std::cout << "lb: " << lb << ", ub: " << ub << "\n";
						std::cout << count << ": " << v << "\n";
					}
					if (count > 20) {
						assert(lb < 1);
						assert(ub > 1);
						assert(count <= 20);
					}
				} while (v < lb*param[0] || v > ub*param[0]);
				return v; 
			};
			return get_rand;
		};

		// TODO: Group the same cone-specific types together for warp performance
		// set test param for LGN subregion RF temporal kernel, also the K_c and K_s
		// Benardete and Kaplan 1997 (source fitted by LGN_kernel.ipynb (mean))
		// empirical value
		
        if (!uniform_LGN) {
		    auto get_nRonC = get_rand_from_clipped_gauss(nR_onC, nR_onClowBound, nR_onCupBound);
		    auto get_nRoffC = get_rand_from_clipped_gauss(nR_offC, nR_offClowBound, nR_offCupBound);
		    auto get_nRonS = get_rand_from_clipped_gauss(nR_onS, nR_onSlowBound, nR_onSupBound);
		    auto get_nRoffS = get_rand_from_clipped_gauss(nR_offS, nR_offSlowBound, nR_offSupBound);

		    Float rho_Kc_Ks = 0.5;
		    Float rho_Kc_Ks_comp = sqrt(1.0-rho_Kc_Ks*rho_Kc_Ks);
		    // reset norm to standard normal distribution
		    normal_distribution<Float> norm(0.0, 1.0);

		    cout << "initializing LGN temporal parameters...\n";
		    // for spontaneous firing rate
		    Float log_mean, log_std;
            assert(spontPercent > 0);
		    tie(log_mean, log_std) = lognstats<Float>(spontPercent, 0.01);

            auto halfBound_onC = get_incLowBound(0.5*K_onC[0]);
            //auto halfBound_offS = get_incLowBound(0.5*K_offS[0]);
            auto halfBound_offC = get_incLowBound(0.5*K_offC[0]);
            //auto halfBound_onS = get_incLowBound(0.5*K_onS[0]);

		    auto nonNegativeBound = get_incLowBound(0.0);

            function<bool(Float)> tauD_Cbound;
            function<bool(Float)> tauD_Sbound;
		    for (unsigned int i=0; i<nLGN; i++) {
		    	// using median from table 2,3  (on,off)/all * table 5,6 with matching c.v.# Benardete and Kaplan 1997
		    	// fit with difference of exponentials in LGN_kernel.ipynb
		    	// cones' differences are not differentiated
                switch (LGNtype[i]) {
                    case InputType::MonLoff: case InputType::LonMoff: 
		    		    // on-center, off-surround
		    		    // k
                        tie(LGN_k[i], LGN_k[i+nLGN]) = get_rands_from_correlated_gauss(K_onC, K_offS, rho_Kc_Ks, rho_Kc_Ks_comp, rGen_LGNsetup, rGen_LGNsetup, halfBound_onC, positiveBound, larger);
		    		    LGN_k[i+nLGN] *= -1; //off-surround !!!IMPORTANT sign change here

		    		    // centers' tau, n 
		    		    nR[i] = get_nRonC(rGen_LGNsetup);
		    		    tspR = get_rand_from_gauss(tspR_onC, rGen_LGNsetup, positiveBound);
		    		    tauR[i] = tspR/(nR[i]-1);

		    		    tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i], tauR[i]);
		    		    tauD_Cbound = get_excLowBound(tauR[i]);
                        tauD[i] = get_rand_from_gauss(tauD_onC, rGen_LGNsetup, tauD_Cbound);
		    		    tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
		    		    nD[i] = tspD/tauD[i]+1;

		    		    // surround' tau, n 
		    		    nR[i+nLGN] = get_nRoffS(rGen_LGNsetup);
		    		    tspR = get_rand_from_gauss(tspR_offS, rGen_LGNsetup, positiveBound);
		    		    tauR[i+nLGN] = tspR/(nR[i+nLGN]-1);

		    		    tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i+nLGN], tauR[i+nLGN]);
		    		    tauD_Sbound = get_excLowBound(tauR[i+nLGN]);
		    		    tauD[i+nLGN] = get_rand_from_gauss(tauD_offS, rGen_LGNsetup, tauD_Sbound);
		    		    tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
		    		    nD[i+nLGN] = tspD/tauD[i+nLGN];

		    		    // ratio
		    		    ratio[i] = get_rand_from_gauss(ratio_onC, rGen_LGNsetup, nonNegativeBound);
		    		    ratio[i+nLGN] = get_rand_from_gauss(ratio_offS, rGen_LGNsetup, nonNegativeBound);

		    		    // delay
		    		    delay[i] = get_rand_from_gauss(delay_onC, rGen_LGNsetup, nonNegativeBound);
		    		    delay[i+nLGN] = get_rand_from_gauss(delay_offS, rGen_LGNsetup, nonNegativeBound);
		    	        // NOTE: proportion between L and M, significantly overlap in cone response curve, is implemented to calculate maxConvol
		    	        //covariant[i] = 0.53753461391295254; 
		    	        covariant[i] = 1;
                        break;
                    case InputType::MoffLon: case InputType::LoffMon:
                        // off-centers
		    			// k
                        tie(LGN_k[i], LGN_k[i+nLGN]) = get_rands_from_correlated_gauss(K_offC, K_onS, rho_Kc_Ks, rho_Kc_Ks_comp, rGen_LGNsetup, rGen_LGNsetup, halfBound_offC, positiveBound, larger);
		    			LGN_k[i] *= -1; //off-center

		    			// centers' tau, n 
		    			nR[i] = get_nRoffC(rGen_LGNsetup);
		    			tspR = get_rand_from_gauss(tspR_offC, rGen_LGNsetup, positiveBound);
		    			tauR[i] = tspR/(nR[i]-1);

		    			tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i], tauR[i]);
		    			tauD_Cbound = get_excLowBound(tauR[i]);
		    			tauD[i] = get_rand_from_gauss(tauD_offC, rGen_LGNsetup, tauD_Cbound);
		    			tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
		    			nD[i] = tspD/tauD[i];

		    			// surround' tau, n 
		    			nR[i+nLGN] = get_nRonS(rGen_LGNsetup);
		    			tspR = get_rand_from_gauss(tspR_onS, rGen_LGNsetup, positiveBound);
		    			tauR[i+nLGN] = tspR/(nR[i+nLGN]-1);

		    			tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i+nLGN], tauR[i+nLGN]);
		    			tauD_Sbound = get_excLowBound(tauR[i+nLGN]);
		    			tauD[i+nLGN] = get_rand_from_gauss(tauD_onS, rGen_LGNsetup, tauD_Sbound);
		    			tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
		    			nD[i+nLGN] = tspD/tauD[i+nLGN];

		    		    // ratio
		    			ratio[i] = get_rand_from_gauss(ratio_offC, rGen_LGNsetup, nonNegativeBound);
		    			ratio[i+nLGN] = get_rand_from_gauss(ratio_onS, rGen_LGNsetup, nonNegativeBound);

                        // delay
		    			delay[i] = get_rand_from_gauss(delay_offC, rGen_LGNsetup, nonNegativeBound);
		    			delay[i+nLGN] = get_rand_from_gauss(delay_onS, rGen_LGNsetup, nonNegativeBound);
		    	        // NOTE: proportion between L and M, significantly overlap in cone response curve, is implemented to calculate maxConvol
		    	        //covariant[i] = 0.53753461391295254; 
		    	        covariant[i] = 1;
                        break;
                    /* TODO: magno dist
                    case InputType::OnOff:
		    		    // ratio
		    		    ratio[i] = get_rand_from_gauss(ratioMagno, rGen_LGNsetup, nonNegativeBound);
		    		    ratio[i+nLGN] = get_rand_from_gauss(ratioMagno, rGen_LGNsetup, nonNegativeBound);
		    		    // delay
                    case InputType::OffOn:
		    		    // ratio
		    		    ratio[i] = get_rand_from_gauss(ratioMagno, rGen_LGNsetup, nonNegativeBound);
		    		    ratio[i+nLGN] = get_rand_from_gauss(ratioMagno, rGen_LGNsetup, nonNegativeBound);
		    		    // delay
                    */
		    	    default:	
		    			throw("There's no implementation of this type with non-uniform LGN");
		    	} 		    			
		    	// c-s coneType, LGN_props.h
		    	switch (LGNtype[i]) {
		    		case InputType::MonLoff: case InputType::MoffLon:
		    			coneType[i] = 1;
		    			coneType[i+nLGN] = 0;
		    			break;
		    		case InputType::LonMoff: case InputType::LoffMon:
		    			coneType[i] = 0;
		    			coneType[i+nLGN] = 1;
		    			break;
                    case InputType::OnOff: case InputType::OffOn:
		    			coneType[i] = 3;
		    			coneType[i+nLGN] = 3;
		    			break;
		    		default: throw("There's no implementation of such RF for parvo LGN");
		    	}
		    	// non-linearity
		    	Float spontTmp = exp(log_mean + norm(rGen_LGNsetup) * log_std);
		    	while (spontTmp > spontPercentUL) {
		    		spontTmp = exp(log_mean + norm(rGen_LGNsetup) * log_std);
		    	}
		    	spont[i] =  spontTmp;
		    	//cout << "\r" << i << "/" << nLGN;
		    }
		    //cout << "\n";

		    norm = normal_distribution<Float>(c50_dist[0], c50_dist[1]);
		    auto normM = normal_distribution<Float>(c50_distM[0], c50_distM[1]);
		    auto get_c50 = get_rand_from_gauss0(rGen_LGNsetup, norm, positiveBound);
		    auto get_c50M = get_rand_from_gauss0(rGen_LGNsetup, normM, positiveBound);
		    generate(c50.begin(), c50.begin()+nParvo_I, get_c50);
		    generate(c50.begin()+nLGN_I, c50.begin()+nParvo_C, get_c50);
		    generate(c50.begin()+nParvo_I, c50.begin()+nLGN_I, get_c50M);
		    generate(c50.begin()+nLGN_I+nParvo_C, c50.end(), get_c50M);

		    auto unityIncBound = get_incLowBound(1.0);
		    norm = normal_distribution<Float>(sharpness_dist[0], sharpness_dist[1]);
		   	normM = normal_distribution<Float>(sharpness_distM[0], sharpness_distM[1]);
		   	auto get_sharpness = get_rand_from_gauss0(rGen_LGNsetup, norm, unityIncBound);
		   	auto get_sharpnessM = get_rand_from_gauss0(rGen_LGNsetup, normM, unityIncBound);
		    generate(sharpness.begin(), sharpness.begin()+nParvo_I, get_sharpness);
		    generate(sharpness.begin()+nLGN_I, sharpness.begin()+nParvo_C, get_sharpness);
		    generate(sharpness.begin()+nParvo_I, sharpness.begin()+nLGN_I, get_sharpnessM);
		    generate(sharpness.begin()+nLGN_I+nParvo_C, sharpness.end(), get_sharpnessM);
		   	for (Size j = 0; j<nLGN; j++) {
		   		assert(sharpness[j] >= 1.0);
		   	}    
        } else {
		    for (unsigned int i=0; i<nLGN; i++) {
		    	// using median from table 2,3  (on,off)/all * table 5,6 with matching c.v.# Benardete and Kaplan 1997
		    	// fit with difference of exponentials in LGN_kernel.ipynb
		    	// cones' differences are not differentiated
                switch (LGNtype[i]) {
                    case InputType::MonLoff: case InputType::LonMoff:
		    		    // on-center, off-surround
		    		    // k
                        LGN_k[i] = K_onC[0];
                        LGN_k[i+nLGN] = -K_offS[0]; //off-surround !!!IMPORTANT sign change here

		    		    // centers' tau, n 
		    		    nR[i] = nR_onC[0];
		    		    tspR = tspR_onC[0];
		    		    tauR[i] = tspR/(nR[i]-1);

		    		    tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i], tauR[i]);
		    		    tauD[i] = tauD_onC[0];
		    		    tspD = pTspD[0];
		    		    nD[i] = tspD/tauD[i]+1;

		    		    // surround' tau, n 
		    		    nR[i+nLGN] = nR_offS[0];
		    		    tspR = tspR_offS[0];
		    		    tauR[i+nLGN] = tspR/(nR[i+nLGN]-1);

		    		    tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i+nLGN], tauR[i+nLGN]);
		    		    tauD[i+nLGN] = tauD_offS[0];
		    		    tspD = pTspD[0];
		    		    nD[i+nLGN] = tspD/tauD[i+nLGN];

		    		    // ratio
		    		    ratio[i] = ratio_onC[0];
		    		    ratio[i+nLGN] = ratio_offS[0];
		    		    // delay
		    		    delay[i] = delay_onC[0];
		    		    delay[i+nLGN] = delay_offS[0];
		    	        // NOTE: proportion between L and M, significantly overlap in cone response curve, is implemented to calculate maxConvol
		    	        //covariant[i] = 0.53753461391295254; 
		    	        covariant[i] = 1;
                        break;
                    case InputType::MoffLon: case InputType::LoffMon:
		    			// off-centers
		    			// k
                        LGN_k[i] = -K_offC[0]; //off-center
                        LGN_k[i+nLGN] = K_onS[0];

		    			// centers' tau, n 
		    			nR[i] = nR_offC[0];
		    			tspR = tspR_offC[0];
		    			tauR[i] = tspR/(nR[i]-1);
		    			tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i], tauR[i]);
		    			tauD[i] = tauD_offC[0];
		    			tspD = pTspD[0];
		    			nD[i] = tspD/tauD[i];

		    			// surround' tau, n 
		    			nR[i+nLGN] = nR_onS[0];
		    			tspR = tspR_onS[0];
		    			tauR[i+nLGN] = tspR/(nR[i+nLGN]-1);

		    			tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i+nLGN], tauR[i+nLGN]);
		    			tauD[i+nLGN] = tauD_onS[0];
		    			tspD = pTspD[0];
		    			nD[i+nLGN] = tspD/tauD[i+nLGN];

		    			// ratio
		    		    ratio[i] = ratio_offC[0];
		    		    ratio[i+nLGN] = ratio_onS[0];
		    			// delay
		    			delay[i] = delay_offC[0];
		    			delay[i+nLGN] = delay_onS[0];
		    	        // NOTE: proportion between L and M, significantly overlap in cone response curve, is implemented to calculate maxConvol
		    	        //covariant[i] = 0.53753461391295254; 
		    	        covariant[i] = 1.0;
                        break;
                    case InputType::OnOff: 
		    			// on-centers
		    			// k
                        LGN_k[i] = K_onMagno[0];
                        LGN_k[i+nLGN] = -K_onMagno[0]; // off-surround
		    			// centers' tau, n 
		    			nR[i] = nR_onMagno[0];
		    			tauR[i] = tauR_onMagno[0];
		    			nR[i+nLGN] = nR_onMagno[0];
		    			tauR[i+nLGN] = tauR_onMagno[0];

		    		    nD[i] = nD_onMagno[0];
                        tauD[i] = tauD_onMagno[0];
		    		    nD[i+nLGN] = nD_onMagno[0];
                        tauD[i+nLGN] = tauD_onMagno[0];

                        // ratio
		    		    ratio[i] = ratio_onMagno[0];
		    		    ratio[i+nLGN] = ratio_onMagno[0];
		    			// delay
		    			delay[i] = delay_onMagno[0];
		    			delay[i+nLGN] = delay_onMagno[0];
		    	        covariant[i] = 1.0;  // magno have anti-correlates
                        break;
                    case InputType::OffOn:
                        // off-centers
		    			// k
                        LGN_k[i] = -K_offMagno[0];
                        LGN_k[i+nLGN] = K_offMagno[0]; // on-surround

		    			// centers' tau, n 
		    			nR[i] = nR_offMagno[0];
		    			tauR[i] = tauR_offMagno[0];
		    			nR[i+nLGN] = nR_offMagno[0];
		    			tauR[i+nLGN] = tauR_offMagno[0];

		    		    nD[i] = nD_offMagno[0];
                        tauD[i] = tauD_offMagno[0];
		    		    nD[i+nLGN] = nD_offMagno[0];
                        tauD[i+nLGN] = tauD_offMagno[0];

                        // ratio
		    		    ratio[i] = ratio_offMagno[0];
		    		    ratio[i+nLGN] = ratio_offMagno[0];
		    			// delay
		    			delay[i] = delay_offMagno[0];
		    			delay[i+nLGN] = delay_offMagno[0];
		    	        covariant[i] = 1.0;
                        break;
                    default:
		    			throw("There's no implementation of such RF for parvo LGN");
		    	}
		    	// c-s coneType, LGN_props.h
		    	switch (LGNtype[i]) {
		    		case InputType::MonLoff: case InputType::MoffLon:
		    			coneType[i] = 1;
		    			coneType[i+nLGN] = 0;
		    			break;
		    		case InputType::LonMoff: case InputType::LoffMon:
		    			coneType[i] = 0;
		    			coneType[i+nLGN] = 1;
		    			break;
                    case InputType::OnOff: case InputType::OffOn:
		    			coneType[i] = 3;
		    			coneType[i+nLGN] = 3;
		    			break;
		    		default: throw("There's no implementation of such RF for parvo LGN");
		    	}
		    	// non-linearity
		    	spont[i] =  spontPercent;
		    	//cout << "\r" << i << "/" << nLGN;
				if (static_cast<InputType_t>(LGNtype[i]) < 4) {
            		sharpness[i] = sharpness_dist[0];
            		c50[i] = c50_dist[0];
				} else {
            		sharpness[i] = sharpness_distM[0];
            		c50[i] = c50_distM[0];
				}
		    }
        }
        //============
        if (ignoreRetinogeniculateDelay) {
            Float minDelay = *min_element(delay.begin(), delay.end());
            auto substract = [minDelay](Float v) {
                return v-minDelay;
            };
            transform(delay.begin(), delay.end(), delay.begin(), substract);
            cout << "the first " << minDelay << "ms delay from retina to LGN is ignored\n";
        }
		
		fLGN.open(LGN_filename + output_suffix, fstream::out | fstream::binary);
		// append to LGN_polar and LGN_ecc positions
		// surround origin is changed
		if (!fLGN) {
			cout << "Cannot open or find " << LGN_filename <<" for LGN receptive field properties\n";
			return EXIT_FAILURE;
		}
		fLGN.write((char*)&nLGN, sizeof(Size));
		fLGN.write((char*)&LGNtype[0], nLGN*sizeof(InputType_t));
		fLGN.write((char*)&LGN_polar[0], 2*nLGN*sizeof(Float)); // in rad now
		fLGN.write((char*)&LGN_ecc[0], 2*nLGN*sizeof(Float));
		// new props
		fLGN.write((char*)&LGN_rw[0], 2*nLGN*sizeof(Float));
		fLGN.write((char*)&LGN_rh[0], 2*nLGN*sizeof(Float));
		fLGN.write((char*)&LGN_orient[0], 2*nLGN*sizeof(Float));

		fLGN.write((char*)&LGN_k[0], 2*nLGN*sizeof(Float));
		fLGN.write((char*)&ratio[0], 2*nLGN*sizeof(Float));
		fLGN.write((char*)&tauR[0], 2*nLGN*sizeof(Float));
		fLGN.write((char*)&tauD[0], 2*nLGN*sizeof(Float));
		fLGN.write((char*)&nR[0], 2*nLGN*sizeof(Float));
		fLGN.write((char*)&nD[0], 2*nLGN*sizeof(Float));
		fLGN.write((char*)&delay[0], 2*nLGN*sizeof(Float));

		fLGN.write((char*)&spont[0], nLGN*sizeof(Float));
		fLGN.write((char*)&c50[0], nLGN*sizeof(Float));
		fLGN.write((char*)&sharpness[0], nLGN*sizeof(Float));

		fLGN.write((char*)&coneType[0], 2*nLGN*sizeof(PosInt));
		fLGN.write((char*)&covariant[0], nLGN*sizeof(Float));
		fLGN.close();
	} else { // Use old setup
		fLGN.open(LGN_filename, fstream::in | fstream::binary);
		if (!fLGN) {
			cout << "Cannot open or find " << LGN_filename <<" for LGN receptive field properties\n";
			return EXIT_FAILURE;
		} else {
			cout << "reading LGN parameters\n";
		}
        Size tmp;
		fLGN.read(reinterpret_cast<char*>(&tmp), sizeof(Size));
		fLGN.read(reinterpret_cast<char*>(&LGNtype[0]), nLGN*sizeof(InputType_t));
		fLGN.read(reinterpret_cast<char*>(&LGN_polar[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&LGN_ecc[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&LGN_rw[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&LGN_rh[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&LGN_orient[0]), 2*nLGN*sizeof(Float));

		fLGN.read(reinterpret_cast<char*>(&LGN_k[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&ratio[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&tauR[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&tauD[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&nR[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&nD[0]), 2*nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&delay[0]), 2*nLGN*sizeof(Float));

		fLGN.read(reinterpret_cast<char*>(&spont[0]), nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&c50[0]), nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&sharpness[0]), nLGN*sizeof(Float));

		fLGN.read(reinterpret_cast<char*>(&coneType[0]), 2*nLGN*sizeof(PosInt));
		fLGN.read(reinterpret_cast<char*>(&covariant[0]), nLGN*sizeof(Float));
		fLGN.close();	
	}
        
	hSpatial_component hSpat(nLGN, 2, LGN_polar, LGN_rw, LGN_ecc, LGN_rh, LGN_orient, LGN_k);
	hTemporal_component hTemp(nLGN, 2, tauR, tauD, delay, ratio, nR, nD);
	hStatic_nonlinear hStat(nLGN, spont, c50, sharpness);

	hLGN_parameter hLGN(nLGN, 2, hSpat, hTemp, hStat, coneType, covariant);

	LGN_parameter dLGN(hLGN);
	usingGMem += hLGN.freeMem();

    InputType_t* dLGN_type;
    if (LGN_switch) {
		checkCudaErrors(cudaMalloc((void **) &dLGN_type, nLGN*sizeof(InputType_t)));
		checkCudaErrors(cudaMemcpy(dLGN_type, &(LGNtype[0]), nLGN*sizeof(InputType_t), cudaMemcpyHostToDevice));

    	usingGMem += nLGN*sizeof(InputType_t);
    }

	vector<int> sxyID(2*nLGN);
	ifstream fLGN_surfaceID;
	Size nsx, nsy;
	fLGN_surfaceID.open(LGN_surfaceID_filename, fstream::in | fstream::binary);
	if (!fLGN_surfaceID) {
		cout << "Cannot open or find " << LGN_surfaceID_filename <<" to read in LGN surface position.\n";
		return EXIT_FAILURE;
	} else {
		fLGN_surfaceID.read(reinterpret_cast<char*>(&nsx), sizeof(Size));
		fLGN_surfaceID.read(reinterpret_cast<char*>(&nsy), sizeof(Size));
		fLGN_surfaceID.read(reinterpret_cast<char*>(&sxyID[0]), 2*nLGN*sizeof(int));
		nsx++; // was max_xid not number of xid
		nsy++; // was max_yid not number of yid
		cout << "LGN_surface " << nsx << "x"<< nsy << "\n";
		int max_idx = *max_element(sxyID.begin(), sxyID.begin()+nLGN);
		int max_idy = *max_element(sxyID.begin()+nLGN, sxyID.end());
		if (max_idx >= nsx) {
			cout << "max_idx = " << max_idx << " < " << nsx << "\n"; 
			assert(max_idx < nsx);
		}
		if (max_idy >= nsy) {
			cout << "max_idy = " << max_idy << " < " << nsy << "\n"; 
			assert(max_idy < nsy);
		}
	}
	fLGN_surfaceID.close();

	// malloc for LGN
	size_t spikeGenSize = (2*sizeof(int) + 2*sizeof(Float) + sizeof(curandStateMRG32k3a)) * nLGN;

	size_t outputB4V1Size;
    size_t B4V1_hostSize;
    Float* outputB4V1;
    if (saveOutputB4V1) { 
        outputB4V1Size = (3+nRegion)*sizeof(Float) * nLGN;
        B4V1_hostSize = outputB4V1Size;
    } else {
        outputB4V1Size = 2*sizeof(Float) * nLGN;
        if (saveLGN_fr) {
            B4V1_hostSize = sizeof(Float) * nLGN;
        } else {
            B4V1_hostSize = 0;
        }
    }
    size_t preFFsize;
	size_t LGN_learnPreSize = nLearnTypeFF*nLGN*sizeof(Float);
    if (learnData_FF) {
         preFFsize = nLGN*sizeof(Float) + LGN_learnPreSize;
        if (B4V1_hostSize < preFFsize) {
            B4V1_hostSize = preFFsize;
        } 
    }
    if (saveOutputB4V1 || saveLGN_fr || learnData_FF) {
	    checkCudaErrors(cudaMallocHost((void**) &outputB4V1, B4V1_hostSize));
    }
	size_t B4V1Size = spikeGenSize + outputB4V1Size;


	char* gpu_B4V1;
	checkCudaErrors(cudaMalloc((void **)&gpu_B4V1, B4V1Size));

    usingGMem += B4V1Size;
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;

	int *d_sx = (int*) gpu_B4V1;
	int *d_sy =  d_sx + nLGN;
	Float* leftTimeRate = (Float*) (d_sy + nLGN);
	Float* lastNegLogRand = leftTimeRate + nLGN;
	curandStateMRG32k3a *randState = (curandStateMRG32k3a*) (lastNegLogRand + nLGN);

	Float* d_LGN_fr = (Float*) (randState + nLGN);
    Float* currentConvol = d_LGN_fr + nLGN;
    Float* luminance;
    Float* contrast;
    if (saveOutputB4V1) {
	    luminance = currentConvol + nLGN; 
	    contrast = luminance + nLGN; // contrast: nLGN * nRegion (center contrast and surround contrast are different because of separate cone input)
    }

	// initialize
	checkCudaErrors(cudaMemcpy(d_sx, &(sxyID[0]), sizeof(int)*2*nLGN, cudaMemcpyHostToDevice));

	// malloc LGN_surface
	cudaChannelFormatDesc surfaceDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* cuSurfArray;

	checkCudaErrors(cudaMalloc3DArray(&cuSurfArray, &surfaceDesc, make_cudaExtent(nsx, nsy, (1+3*nLearnTypeFF)), cudaArrayLayered|cudaArraySurfaceLoadStore));
	size_t LGNspSurfSize = nsx*nsy*(1+3*nLearnTypeFF)*sizeof(Float);
	usingGMem += LGNspSurfSize;
	cudaBindSurfaceToArray(LGNspikeSurface, cuSurfArray);
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;

	Size nLGN_block, nLGN_thread; // for LGN_nonlinear
	nLGN_thread = blockSize;
	nLGN_block = (nLGN + nLGN_thread - 1)/nLGN_thread;
    cout << "logRand_init<<<" << nLGN_block << ", " << nLGN_thread << ">>>" << "\n";
	logRand_init<<<nLGN_block, nLGN_thread>>>(lastNegLogRand, leftTimeRate, d_sx, d_sy, randState, seed, nLGN, nLearnTypeFF);
	seed++;
    #ifdef CHECK
	    getLastCudaError("logRand_init");
    #endif

	// Storage memory
	size_t maxConvolSize = nLGN;
	// SC and TW are computation intensive
	size_t parvo_TW_size = nRegion*nKernelSample*nParvo;
	size_t parvo_SC_size = 2*nRegion*nSample*nParvo;
	size_t parvo_SW_size = nSample*(nParvo>0); // normalized Gaussian has constant values at constant intervals, heterogeneity is reflected by Spatial Coordinates (SC_storage)
	size_t magno_TW_size = mKernelSample*nMagno;
	size_t magno_SC_size = 2*mSample*nMagno;
	size_t magno_SW_size = mSample*nMagno; // heterogeneity is also in the weights, not like parvo
	size_t gallerySize = (maxConvolSize + parvo_TW_size + parvo_SW_size + magno_TW_size + magno_SW_size)*sizeof(Float) + (magno_SC_size + parvo_SC_size)*sizeof(float);
	char* galleryOutput = new char[gallerySize]; 

	Float* gpu_LGN_gallery;
	checkCudaErrors(cudaMalloc((void **) &gpu_LGN_gallery, gallerySize));
	usingGMem += gallerySize;
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;

	Float* maxConvol = gpu_LGN_gallery;
	Float* parvo_TW_storage = maxConvol + maxConvolSize;
	Float* parvo_SW_storage = parvo_TW_storage + parvo_TW_size;
	float* parvo_SC_storage = (float*) (parvo_SW_storage + parvo_SW_size);
	Float* magno_TW_storage = (Float*) (parvo_SC_storage + parvo_SC_size);
	Float* magno_SW_storage = magno_TW_storage + magno_TW_size;
	float* magno_SC_storage = (float*) (magno_SW_storage + magno_SW_size);

	checkCudaErrors(cudaMemset(maxConvol, 0, nLGN*sizeof(Float)));

    checkCudaErrors(cudaDeviceSynchronize());
	cout << "LGN initialized\n";
	// finish LGN setup

	// V1 related memory
	ifstream fV1_pos;
	ifstream fV1_conMat, fV1_delayMat, fV1_vec, fNeighborBlock; 
	ifstream fConStats;

	Size nV1, nblock, neuronPerBlock, posDim;
	double *cpu_chunk_V1pos;
	double *V1_x;
	double *V1_y;
	double *V1_vx;
	double *V1_vy;
	double V1_x0, V1_xspan;
	double V1_y0, V1_yspan;
	double V1_vx0, V1_vxspan;
	double V1_vy0, V1_vyspan;
	fV1_pos.open(V1_pos_filename, fstream::in | fstream::binary);
	if (!fV1_pos) {
		cout << "Cannot open or find " << V1_pos_filename <<" to read V1 positions.\n";
		return EXIT_FAILURE;
	} else {
		fV1_pos.read(reinterpret_cast<char*>(&nblock), sizeof(Size));
		fV1_pos.read(reinterpret_cast<char*>(&neuronPerBlock), sizeof(Size));
		assert(neuronPerBlock == blockSize);
		fV1_pos.read(reinterpret_cast<char*>(&posDim), sizeof(Size));
		if (posDim > 2) {
			cout << "3-D not yet implemented\n";
			return EXIT_FAILURE;
		}
		fV1_pos.read(reinterpret_cast<char*>(&V1_x0), sizeof(double));
		fV1_pos.read(reinterpret_cast<char*>(&V1_xspan), sizeof(double));
		fV1_pos.read(reinterpret_cast<char*>(&V1_y0), sizeof(double));
		fV1_pos.read(reinterpret_cast<char*>(&V1_yspan), sizeof(double));
		nV1 = nblock * neuronPerBlock;
		cout << nV1 << " V1 neurons, x:[" << V1_x0 << ", " << V1_x0+V1_xspan << "], y:[" << V1_y0 << ", " << V1_y0 + V1_yspan << "]\n";
		if (!frameVisV1output) {
			cpu_chunk_V1pos = new double[nV1*2];
			fV1_pos.read(reinterpret_cast<char*>(cpu_chunk_V1pos), 2*nV1*sizeof(double));
		} else {
			cpu_chunk_V1pos = new double[nV1*4];
			fV1_pos.read(reinterpret_cast<char*>(cpu_chunk_V1pos), 2*nV1*sizeof(double));
			fV1_pos.read(reinterpret_cast<char*>(&V1_vx0), sizeof(double));
			fV1_pos.read(reinterpret_cast<char*>(&V1_vxspan), sizeof(double));
			fV1_pos.read(reinterpret_cast<char*>(&V1_vy0), sizeof(double));
			fV1_pos.read(reinterpret_cast<char*>(&V1_vyspan), sizeof(double));
		    cout << " V1 visual position, x:[" << V1_vx0 << ", " << V1_vx0+V1_vxspan << "], y:[" << V1_vy0 << ", " << V1_vy0 + V1_vyspan << "] in degree\n";
			fV1_pos.read(reinterpret_cast<char*>(cpu_chunk_V1pos + 2*nV1), 2*nV1*sizeof(double));
		}
		V1_x = cpu_chunk_V1pos;
		V1_y = V1_x + nV1;
        flattenBlock<double>(nblock, neuronPerBlock, cpu_chunk_V1pos);
		Float xMax, xMin;
		Float yMax, yMin;
		for (PosInt i = 0; i < nV1; i++) {
			if (i == 0) {
				xMax = V1_x[0];
				xMin = V1_x[0];
				yMax = V1_y[0];
				yMin = V1_y[0];
			}
			else {
				if (V1_x[i] > xMax) xMax = V1_x[i];
				if (V1_x[i] < xMin) xMin = V1_x[i];
				if (V1_y[i] > yMax) yMax = V1_y[i];
				if (V1_y[i] < yMin) yMin = V1_y[i];
			}
		}
        assert(xMin >= V1_x0);
        assert(xMax <= V1_x0+V1_xspan);
        assert(yMin >= V1_y0);
        assert(yMax <= V1_y0+V1_yspan);
		if (frameVisV1output) {
			V1_vx = V1_y + nV1;
			V1_vy = V1_vx + nV1;
			Float vxMax, vxMin;
			Float vyMax, vyMin;
			for (PosInt i = 0; i < nV1; i++) {
				if (i == 0) {
					vxMax = V1_vx[0];
					vxMin = V1_vx[0];
					vyMax = V1_vy[0];
					vyMin = V1_vy[0];
				}
				else {
					if (V1_vx[i] > vxMax) vxMax = V1_vx[i];
					if (V1_vx[i] < vxMin) vxMin = V1_vx[i];
					if (V1_vy[i] > vyMax) vyMax = V1_vy[i];
					if (V1_vy[i] < vyMin) vyMin = V1_vy[i];
				}
			}
            assert(vxMin >= V1_vx0);
            assert(vxMax <= V1_vx0+V1_vxspan);
            assert(vyMin >= V1_vy0);
            assert(vyMax <= V1_vy0+V1_vyspan);
		}
	}
	fV1_pos.close();

	vector<Float> featureValue;
	if (readFeature) {
		fV1_feature.open(V1_feature_filename, ios::in|ios::binary);
		if (!fV1_feature) {
			cout << "failed to open pos file:" << V1_feature_filename << "\n";
			return EXIT_FAILURE;
		}
		Size nFeature;
		fV1_feature.read(reinterpret_cast<char*>(&nFeature), sizeof(Size));
		featureValue.resize(nFeature*nV1); // [OD, OP, ..]
		vector<float> featureValue0(nFeature*nV1);
		fV1_feature.read(reinterpret_cast<char*>(&featureValue0[0]), sizeof(float)*nFeature*nV1);
		for (PosInt i=0; i<nV1*nFeature; i++) {
			featureValue[i] = static_cast<Float>(featureValue0[i]);
		}
		fV1_feature.close();
	} 
    
	// stats: frames, rawdata
	// cortical and LGN surface

    Size phyWidth = width*phyWidth_scale;
	Float V1_hwPhyRatio = V1_yspan/V1_xspan;
	Size phyHeight = ceil(V1_hwPhyRatio * phyWidth);
	if (phyHeight%2 == 1) phyHeight++;
	Size nPixel_phyV1 = phyWidth * phyHeight;

	// two visual field surface, left and right
    Size visWidth = width*visWidth_scale;
	Size nPixel_visV1, visHeight; // share with visLGN
	if (frameVisV1output || frameVisLGNoutput) {
		Float hwVisRatioV = LGN_yspan/LGN_xspan;
		visHeight = ceil(hwVisRatioV * visWidth);
		if (visHeight%2 == 1) visHeight++;
		// left + 4pixel gap + right = 1024
		nPixel_visV1 = 2*visWidth * visHeight;
	}
	Size nPixel_visLGN = nPixel_visV1;

	PosInt iFrameOutput = 0;
	// Allocate mem for framePosId nXXperPixel and outputFrame
	char *d_V1_phyFrame;
	Size *d_nV1perPhyPixel;
	PosInt *d_V1_phyFramePosId;
	Size maxV1perPixel;
	if (framePhyV1output) {
		// evaluate neuron id for each pixel in the frame by position
		vector<Int> pick(nV1,1); // dummy variable, picks for all neuron
		vector<vector<PosInt>> V1_phyFramePosId_v = getUnderlyingID<double>(&(V1_x[0]), &(V1_y[0]), &(pick[0]), 0, nV1, phyWidth, phyHeight, V1_x0, V1_xspan, V1_y0, V1_yspan, &maxV1perPixel, nV1); // height defined by yspan/xspan * width

        //DEBUG
            PosInt id_max = 0;
            for (PosInt i=0; i<nPixel_phyV1; i++) {
                if (V1_phyFramePosId_v[i].size() > 0) {
                    PosInt id = *max_element(V1_phyFramePosId_v[i].begin(), V1_phyFramePosId_v[i].end());
                    if (id > id_max) id_max = id;
                }
            }
            assert(id_max < nV1);
        //
		// determine size
		size_t V1_phyFrameSize = static_cast<size_t>(maxV1perPixel)*phyWidth*phyHeight * sizeof(PosInt);
		V1_phyFrameSize += nPixel_phyV1 * sizeof(Size);
		// allocate
		char* V1_phyFrame = new char[V1_phyFrameSize];
		Size *nV1perPhyPixel = (Size*) V1_phyFrame;
		PosInt* V1_phyFramePosId = (PosInt*) (nV1perPhyPixel + nPixel_phyV1);
		// assign
		for (PosInt i=0; i<nPixel_phyV1; i++) {
			nV1perPhyPixel[i] = V1_phyFramePosId_v[i].size();
			if (nV1perPhyPixel[i] > 0) {
                memcpy(V1_phyFramePosId + i*maxV1perPixel, &(V1_phyFramePosId_v[i][0]), sizeof(PosInt)*nV1perPhyPixel[i]);
                //DEBUG
                    PosInt id = *max_element(V1_phyFramePosId + i*maxV1perPixel, V1_phyFramePosId + i*maxV1perPixel + nV1perPhyPixel[i]);
                    assert(id < nV1);
                //
            }

		}
		// gpu allocate
		checkCudaErrors(cudaMalloc((void **) &d_V1_phyFrame, V1_phyFrameSize));
		usingGMem += V1_phyFrameSize;
		if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;
		// copy to gpu
		checkCudaErrors(cudaMemcpy(d_V1_phyFrame, V1_phyFrame, V1_phyFrameSize, cudaMemcpyHostToDevice));
		// place pointers
		d_nV1perPhyPixel = (Size*) d_V1_phyFrame;
		d_V1_phyFramePosId = (PosInt*) (d_nV1perPhyPixel + nPixel_phyV1);

		delete [] V1_phyFrame;
		iFrameOutput += 1;
		cout << "V1 frame output " << phyWidth << "x" << phyHeight << ", maximum " << maxV1perPixel << " V1 neurons per pixel.\n";
	}

	char* d_LGN_visFrame;
	PosInt *d_LGN_visFramePosId;
	Size *d_nLGNperPixel;
	Size maxLGNperPixel_I, maxLGNperPixel_C;
	if (frameVisLGNoutput) {
		Size nTmp = nLGN_C > nLGN_I	? nLGN_C: nLGN_I;
		// evaluate neuron id for each pixel in the frame by position
		vector<Int> pick(nTmp, true); // LGN index are well separated
        cout << "visWidth x visHeight " << visWidth << "x" << visHeight << "\n";
		vector<vector<PosInt>> LGN_visFramePosId_vI = getUnderlyingID<Float>(&(LGN_x[0]), &(LGN_y[0]), &(pick[0]), 0, nLGN_I, visWidth, visHeight, LGN_x0, LGN_xspan, LGN_y0, LGN_yspan, &maxLGNperPixel_I, nLGN_I);
        //DEBUG
            if (nLGN_I > 0) {
                PosInt id_maxI = 0;
                for (PosInt i=0; i<nPixel_visLGN/2; i++) {
                    if (LGN_visFramePosId_vI[i].size() > 0) {
                        PosInt id = *max_element(LGN_visFramePosId_vI[i].begin(), LGN_visFramePosId_vI[i].end());
                        if (id > id_maxI) id_maxI = id;
                    }
                }
                assert(id_maxI < nLGN_I);
            }
        //

		vector<vector<PosInt>> LGN_visFramePosId_vC = getUnderlyingID<Float>(&(LGN_x[nLGN_I]), &(LGN_y[nLGN_I]), &(pick[0]), nLGN_I, nLGN, visWidth, visHeight, LGN_x0, LGN_xspan, LGN_y0, LGN_yspan, &maxLGNperPixel_C, nLGN_C);

        //DEBUG
            if (nLGN_C > 0) {
                PosInt id_maxC = nLGN_I;
                for (PosInt i=0; i<nPixel_visLGN/2; i++) {
                    if (LGN_visFramePosId_vC[i].size() > 0) {
                        PosInt id = *max_element(LGN_visFramePosId_vC[i].begin(), LGN_visFramePosId_vC[i].end());
                        if (id > id_maxC) id_maxC = id;
                        if (id < nLGN_I) assert(false);
                    }
                }
                assert(id_maxC < nLGN);
            }
        //

		// determine size
		size_t LGN_visFrameSize = static_cast<size_t>(maxLGNperPixel_I + maxLGNperPixel_C)*visWidth*visHeight * sizeof(PosInt);
		LGN_visFrameSize += nPixel_visLGN * sizeof(Size);
		// allocate
		char* LGN_visFrame = new char[LGN_visFrameSize];
		Size* nLGNperPixel = (Size*) LGN_visFrame;
		PosInt* LGN_visFramePosId = (PosInt*) (nLGNperPixel + nPixel_visLGN);
		// assign
		PosInt offset = visWidth*visHeight;
		for (PosInt i=0; i<visWidth*visHeight; i++) {
			// Ipsi
			nLGNperPixel[i] = LGN_visFramePosId_vI[i].size();
			if (nLGNperPixel[i] > 0) {
                memcpy(LGN_visFramePosId + i*maxLGNperPixel_I, &(LGN_visFramePosId_vI[i][0]), sizeof(PosInt)*nLGNperPixel[i]);
                // DEBUG
                    PosInt id = *max_element(LGN_visFramePosId + i*maxLGNperPixel_I, LGN_visFramePosId + i*maxLGNperPixel_I + nLGNperPixel[i]);
                    assert(id < nLGN_I);
                //
            }
			// Contra
			nLGNperPixel[i+offset] = LGN_visFramePosId_vC[i].size();
			if (nLGNperPixel[i+offset] > 0) { 
                memcpy(LGN_visFramePosId + offset*maxLGNperPixel_I + i*maxLGNperPixel_C, &(LGN_visFramePosId_vC[i][0]), sizeof(PosInt)*nLGNperPixel[i+offset]);
                // DEBUG
                    PosInt id = *max_element(LGN_visFramePosId + offset*maxLGNperPixel_I + i*maxLGNperPixel_C, LGN_visFramePosId + offset*maxLGNperPixel_I + i*maxLGNperPixel_C + nLGNperPixel[i+offset]);
                    assert(id >= nLGN_I);
                    assert(id < nLGN);
                //
            }
		}
		// gpu allocate 
		checkCudaErrors(cudaMalloc((void **) &d_LGN_visFrame, LGN_visFrameSize));
		// copy to gpu
		checkCudaErrors(cudaMemcpy(d_LGN_visFrame, LGN_visFrame, LGN_visFrameSize, cudaMemcpyHostToDevice));
		// place pointers
		d_nLGNperPixel = (Size*) d_LGN_visFrame;
		d_LGN_visFramePosId = (PosInt*) (d_nLGNperPixel + nPixel_visLGN);

		delete [] LGN_visFrame;
		usingGMem += LGN_visFrameSize;
		if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;
		iFrameOutput += 4;
		cout << "LGN VF frame output " << visWidth << "x" << visHeight << ", maximum " << maxLGNperPixel_C << "(" << maxLGNperPixel_I << ")" << "Contra(Ipsi) LGN neurons per pixel.\n";
	}

	char* d_V1_visFrame;
	PosInt *d_V1_visFramePosId;
	Size *d_nV1perVisPixel;
	Size maxV1perPixel_I, maxV1perPixel_C;
	if (frameVisV1output) {
		// evaluate neuron id for each pixel in the frame by position
		Int* pick = new Int[nV1]; // only for OD pick
        Size nV1_I = 0;
        Size nV1_C = 0;
		if (readFeature) {
			for (PosInt i = 0; i<nV1; i++) {
                if (featureValue[i] < 0) {
				    pick[i] = 1;
                    nV1_I++;
                } else {
                    pick[i] = -1;
                    nV1_C++;
                }
			}
		} else {
			for (PosInt i = 0; i<nV1; i++) {
				pick[i] = 1;
			}
            nV1_I = nV1;
		}
		vector<vector<PosInt>> V1_visFramePosId_vI = getUnderlyingID<double>(&(V1_vx[0]), &(V1_vy[0]), pick, 0, nV1, visWidth, visHeight, V1_vx0, V1_vxspan, V1_vy0, V1_vyspan, &maxV1perPixel_I, nV1_I);
		for (PosInt i = 0; i<nV1; i++) {
			pick[i] = -pick[i];
		}

        //DEBUG
            PosInt id_max;
            if (nV1_I > 0) {
                id_max = 0;
                for (PosInt i=0; i<nPixel_visV1/2; i++) {
                    if (V1_visFramePosId_vI[i].size() > 0) {
                        PosInt id = *max_element(V1_visFramePosId_vI[i].begin(), V1_visFramePosId_vI[i].end());
                        if (id > id_max) id_max = id;
                    }
                }
                assert(id_max < nV1);
            }
        //

		vector<vector<PosInt>> V1_visFramePosId_vC = getUnderlyingID<double>(&(V1_vx[0]), &(V1_vy[0]), pick, 0, nV1, visWidth, visHeight, V1_vx0, V1_vxspan, V1_vy0, V1_vyspan, &maxV1perPixel_C, nV1_C);
		delete []pick;

        //DEBUG
            if (nV1_C > 0) {
                id_max = 0;
                for (PosInt i=0; i<nPixel_visV1/2; i++) {
                    if (V1_visFramePosId_vC[i].size() > 0) {
                        PosInt id = *max_element(V1_visFramePosId_vC[i].begin(), V1_visFramePosId_vC[i].end());
                        if (id > id_max) id_max = id;
                    }
                }
                assert(id_max < nV1);
            }
        //

		// detemine size
		size_t V1_visFrameSize = static_cast<size_t>(maxV1perPixel_I + maxV1perPixel_C)*visWidth*visHeight*sizeof(PosInt);
		V1_visFrameSize += nPixel_visV1 * sizeof(Size);
		// allocate
		char* V1_visFrame = new char[V1_visFrameSize];
		Size* nV1perVisPixel = (Size*) V1_visFrame;
		PosInt* V1_visFramePosId = (PosInt*) (nV1perVisPixel + nPixel_visV1);
		// assign
		PosInt offset = visWidth*visHeight;
		for (PosInt i=0; i<visWidth*visHeight; i++) {
			// Ipsi
			nV1perVisPixel[i] = V1_visFramePosId_vI[i].size();
			if (nV1perVisPixel[i] > 0) {
                memcpy(V1_visFramePosId + i * maxV1perPixel_I, &(V1_visFramePosId_vI[i][0]), sizeof(PosInt)*nV1perVisPixel[i]);
                //DEBUG
                    PosInt id = *max_element(V1_visFramePosId + i*maxV1perPixel_I, V1_visFramePosId + i*maxV1perPixel_I + nV1perVisPixel[i]);
                    assert(id < nV1);
                //
            }
			// Contra
			nV1perVisPixel[i+offset] = V1_visFramePosId_vC[i].size();
			if (nV1perVisPixel[i+offset] > 0) {
                memcpy(V1_visFramePosId + offset*maxV1perPixel_I + i*maxV1perPixel_C, &(V1_visFramePosId_vC[i][0]), sizeof(PosInt)*nV1perVisPixel[i+offset]);
                //DEBUG
                    PosInt id = *max_element(V1_visFramePosId + offset*maxV1perPixel_I + i*maxV1perPixel_C, V1_visFramePosId + offset*maxV1perPixel_I + i*maxV1perPixel_C + nV1perVisPixel[i+offset]);
                    assert(id < nV1);
                //
            }
		}
		// gpu allocate
		checkCudaErrors(cudaMalloc((void **) &d_V1_visFrame, V1_visFrameSize));
		checkCudaErrors(cudaMemcpy(d_V1_visFrame, V1_visFrame, V1_visFrameSize, cudaMemcpyHostToDevice));
		// place pointers
		d_nV1perVisPixel = (Size*) d_V1_visFrame;
		d_V1_visFramePosId = (PosInt*) (d_nV1perVisPixel + nPixel_visV1);

		delete []V1_visFrame;
		usingGMem += V1_visFrameSize;
		if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;
		iFrameOutput += 2;
		cout << "V1 VF frame output " << visWidth << "x" << visHeight << ", maximum " << maxV1perPixel_C << "(" << maxV1perPixel_I << ")" << "Contra(Ipsi) V1 neurons per pixel.\n";
	}

	Float *d_outputFrame;
	Float *d_V1SpPhyFrame;
	Float *d_V1SpVisFrame;
	Float *d_LGN_spVisFrame;
	size_t framesSize = 0;
    if (framePhyV1output) framesSize += nPixel_phyV1;
	if (frameVisV1output) framesSize += nPixel_visV1;
	if (frameVisLGNoutput) framesSize += nPixel_visLGN;

    Float *outputFrame = new Float[framesSize];
    if (framePhyV1output || frameVisV1output || frameVisLGNoutput) {
	    framesSize *= sizeof(Float);
	    checkCudaErrors(cudaMalloc((void **) &d_outputFrame, framesSize));
    }
    if (framePhyV1output) {
	    d_V1SpPhyFrame = d_outputFrame;
	    if (frameVisV1output) {
	    	d_V1SpVisFrame = d_V1SpPhyFrame + nPixel_phyV1;	
	    	if (frameVisLGNoutput) d_LGN_spVisFrame = d_V1SpVisFrame + nPixel_visV1;
	    } else {
	    	if (frameVisLGNoutput) d_LGN_spVisFrame = d_V1SpPhyFrame + nPixel_phyV1;	
	    }
    } else {
	    if (frameVisV1output) {
	    	d_V1SpVisFrame = d_outputFrame;
	    	if (frameVisLGNoutput) d_LGN_spVisFrame = d_V1SpVisFrame + nPixel_visV1;
	    } else {
	    	if (frameVisLGNoutput) d_LGN_spVisFrame = d_outputFrame;	
	    }
    }
	usingGMem += framesSize;
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;

    // far connection
	vector<Size> nVec(nV1);
	vector<vector<PosInt>> vecID(nV1);
	vector<vector<Float>> conVec(nV1);
	vector<vector<Float>> delayVec(nV1);
    size_t nFar = 0;
	fV1_vec.open(V1_vec_filename + conV1_suffix, fstream::in | fstream::binary);
	if (!fV1_vec) {
		cout << "Cannot open or find " << V1_vec_filename + conV1_suffix <<" to read V1 connection to farther neighbor.\n";
		return EXIT_FAILURE;
	} else {
		fV1_vec.read(reinterpret_cast<char*>(&nVec[0]), nV1*sizeof(Size));
		for (PosInt i=0; i<nV1; i++) {
			if (nVec[i] > 0) {
				vector<PosInt> tmp(nVec[i]);
				fV1_vec.read(reinterpret_cast<char*>(&tmp[0]), nVec[i]*sizeof(PosInt));
				vecID[i] = tmp;

				vector<float> ftmp(nVec[i]);
				fV1_vec.read(reinterpret_cast<char*>(&ftmp[0]), nVec[i]*sizeof(float));
				for (PosInt j=0; j<nVec[i]; j++) {
					conVec[i][j] = static_cast<Float>(ftmp[j]);
				}

				fV1_vec.read(reinterpret_cast<char*>(&ftmp[0]), nVec[i]*sizeof(float));
				for (PosInt j=0; j<nVec[i]; j++) {
					delayVec[i][j] = static_cast<Float>(ftmp[j]);
				}
                nFar += nVec[i];
			}
		}
	}
	fV1_vec.close();

    // closer neighbors
	// overlap multiple chunks of data transfer and computation to increase performance
    if (nChunk > nblock) {
        nChunk = nblock;
        cout << "nChunk is reduced to " << nblock << " (nblock)\n";
    }
	PosInt iSizeSplit = nblock % nChunk; // | maxChunkSize, i < iSizeSplit| remainChunkSize
	Size maxChunkSize = nblock/nChunk;
	Size remainChunkSize = maxChunkSize;
	if (iSizeSplit > 0) maxChunkSize++;
    cout << nChunk << " chunks in total, the first " << iSizeSplit << " chunks have " << maxChunkSize  << " blocks each, the others have " << remainChunkSize << " blocks \n";
	assert(maxChunkSize * iSizeSplit + remainChunkSize * (nChunk - iSizeSplit) == nblock);

	Size nearNeighborBlock;
	fV1_conMat.open(V1_conMat_filename + conV1_suffix, fstream::in | fstream::binary);
	if (!fV1_conMat) {
		cout << "Cannot open or find " << V1_conMat_filename + conV1_suffix <<" to read V1 cortical connection matrices.\n";
		return EXIT_FAILURE;
	} else {
		cout << "reading connectome from " << V1_conMat_filename << "\n";
		fV1_conMat.read(reinterpret_cast<char*>(&nearNeighborBlock), sizeof(Size));
	}
	fV1_delayMat.open(V1_delayMat_filename + conV1_suffix, fstream::in | fstream::binary);
	if (!fV1_delayMat) {
		cout << "Cannot open or find " << V1_delayMat_filename + conV1_suffix <<" to read V1 cortical distance matrices.\n";
		return EXIT_FAILURE;
	} else {
		Size tmp;
		fV1_delayMat.read(reinterpret_cast<char*>(&tmp), sizeof(Size));
		if (tmp != nearNeighborBlock) {
			cout << "conMat and delayMat does not match\n";
			return EXIT_FAILURE;
		}
	}

	// pinned pathway for conDelayMat concurrency
	if (matConcurrency < 2) {
		matConcurrency = 2;
		cout << "matConcurrency raised to 2, smaller values not implemented.\n";
	}
	if (matConcurrency > nChunk) {
        matConcurrency = nChunk;
		cout << "matConcurrency is reduced to " << nChunk << ", the same as nChunk\n";  
	}
	size_t nearBlockSize = static_cast<size_t>(nearNeighborBlock) * blockSize*blockSize;
	size_t sChunkMatSize = 2*maxChunkSize*nearBlockSize; // 2 for conMat and delayMat
	size_t rChunkMatSize = 2*remainChunkSize*nearBlockSize;
	cout << "single chunk of conDelayMat requires at most " << sChunkMatSize*sizeof(float)/1024.0/1024.0 << "Mb, smaller chunks require " << rChunkMatSize*sizeof(float)/1024.0/1024.0 << "Mb\n";
	size_t ccChunkMatSize; // # of (c)on(c)urrent chunks
	if (matConcurrency > iSizeSplit) {
		ccChunkMatSize = iSizeSplit * sChunkMatSize + (matConcurrency-iSizeSplit) * rChunkMatSize;
	} else {
		ccChunkMatSize = matConcurrency * sChunkMatSize;
	}
	int ccReduced = 0;
	while (usingGMem + ccChunkMatSize*sizeof(float) > deviceProps.totalGlobalMem) {
		if (matConcurrency > iSizeSplit) {
			ccChunkMatSize -= rChunkMatSize;
		} else {
			ccChunkMatSize -= sChunkMatSize;
		}
		matConcurrency--;
		ccReduced++;
	}
	if (ccReduced) {
		cout << "GPU does not have the required memory for requested for the size of concurrency " << matConcurrency + ccReduced << ", it's now reduced to " << matConcurrency << "\n";
	}
	cout << "matConcurrency of " << matConcurrency << " chunks requires " << ccChunkMatSize*sizeof(float)/1024.0/1024.0 << " Mb, total device gmem = " << deviceProps.totalGlobalMem/1024.0/1024.0 << "Mb\n";

	// pinned conDelayMat on Host
	float* p_conDelayMat;
	checkCudaErrors(cudaMallocHost((void**) &p_conDelayMat, ccChunkMatSize*sizeof(float)));

	// receiving end of pinned conDelayMat on Device
	float *d_mat; // d_delayMat;
	checkCudaErrors(cudaMalloc((void**)&d_mat, ccChunkMatSize*sizeof(float)));
	float **d_conDelayMat = new float*[matConcurrency];
	d_conDelayMat[0] = d_mat;
	for (PosInt i = 1; i<matConcurrency; i++) {
		size_t matChunkSize;
		if (i <= iSizeSplit) matChunkSize = sChunkMatSize; // this is the bug, <= is correct
		else matChunkSize = rChunkMatSize;
		d_conDelayMat[i] = d_conDelayMat[i-1] + matChunkSize; // may not be filled for iChunk > iSizeSplit
	}
	usingGMem += ccChunkMatSize*sizeof(float);
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;

	// reading the two matrices in a intertwined manner
	size_t matSize = nblock * nearBlockSize; 
	cout << "matSize = " << nblock << "x" << nearNeighborBlock << "x" << blockSize << "x" << blockSize << "x" << sizeof(float) << "=" << matSize * sizeof(float) / 1024.0 / 1024.0 << "Mb\n";
	float* conDelayMat0;
    if (matConcurrency < nChunk) {
        conDelayMat0 = new float[matSize*2];
    } else {
        assert(matSize*2 == ccChunkMatSize);
        conDelayMat0 = p_conDelayMat;
        cout << nChunk << " == " << matConcurrency << ", entire conMat and delayMat are pinned\n";
    }
	float** conDelayMat = new float*[nChunk];
	float** conMat = new float*[nChunk];
	float** delayMat = new float*[nChunk];
	size_t matOffset = 0;
	size_t matChunkSize = maxChunkSize*nearBlockSize;
	size_t chunkSize = matChunkSize;
	for (PosInt i=0; i<nChunk; i++) {
		if (i >= iSizeSplit) chunkSize = remainChunkSize*nearBlockSize;
		conDelayMat[i] = conDelayMat0 + matOffset;
		conMat[i] = conDelayMat[i];
		matOffset += chunkSize;
		delayMat[i] = conDelayMat0 + matOffset;
		matOffset += chunkSize;
	}
    assert(matOffset == matSize*2);
	assert(delayMat[nChunk-1] + chunkSize == conDelayMat0 + matSize*2);
    auto findDisConMax = [](float a[], float b[], size_t n) {
        Float max;
        max = static_cast<Float>(a[0]);
        for (PosIntL i=1; i<n; i++) {
            if (a[i] > max && b[i] > 0) {
                max =  static_cast<Float>(a[i]);
            }
        }
        return max;
    };
    auto scaleStrength =[sRatioV1] (float v) {
        return static_cast<float>(v*sRatioV1);
    };
	chunkSize = matChunkSize;
	for (PosInt i=0; i<nChunk; i++) {
		if (i >= iSizeSplit) chunkSize = remainChunkSize*nearBlockSize;
		if (fV1_conMat) fV1_conMat.read(reinterpret_cast<char*>(conMat[i]), chunkSize*sizeof(float));
		else assert(fV1_conMat);
        transform(conMat[i], conMat[i]+chunkSize, conMat[i], scaleStrength);
		if (fV1_delayMat) fV1_delayMat.read(reinterpret_cast<char*>(delayMat[i]), chunkSize*sizeof(float));
		else assert(fV1_delayMat);
	}
	fV1_conMat.close();
	fV1_delayMat.close();
    cout << "conMat and delayMat set\n";
    if (nChunk == matConcurrency) {
        checkCudaErrors(cudaMemcpy(d_mat, p_conDelayMat, ccChunkMatSize*sizeof(float), cudaMemcpyHostToDevice));
        // assert copy success
        //Float mean, dis;
        //size_t matOffset = 0;
        //for (PosInt i=0; i<nChunk; i++) {
        //    if (i>=iSizeSplit) chunkSize = remainChunkSize;
        //    else chunkSize = maxChunkSize;
        //    size_t mChunkSize = chunkSize*nearBlockSize;
        //    mean = accumulate(p_conDelayMat+matOffset, p_conDelayMat + matOffset + mChunkSize, 0.0)/(chunkSize*blockSize); 
        //    cout << "mean con: "  << mean << ", ";
        //    matOffset += mChunkSize;
        //    dis = accumulate(p_conDelayMat+matOffset, p_conDelayMat + matOffset + mChunkSize, 0.0)/(chunkSize*blockSize); 
        //    cout << "mean delay: "  << dis << "\n";
        //    matOffset += mChunkSize;
        //}
    }

	vector<Size> nNeighborBlock(nblock);
	vector<vector<PosInt>> neighborBlockId(nblock);
	PosInt *nBlockId = new PosInt[nearNeighborBlock*nblock];

    // neighboring blocks
	fNeighborBlock.open(neighborBlock_filename + conV1_suffix, fstream::in | fstream::binary);
	if (!fNeighborBlock) {
		cout << "Cannot open or find " << neighborBlock_filename + conV1_suffix <<" to read V1 neighbor block info.\n";
		return EXIT_FAILURE;
	} else {
		fNeighborBlock.read(reinterpret_cast<char*>(&nNeighborBlock[0]), nblock*sizeof(Size));
		for (PosInt i=0; i<nblock; i++) {
			if (nNeighborBlock[i] > 0) {
				vector<PosInt> tmp(nNeighborBlock[i]);
				fNeighborBlock.read(reinterpret_cast<char*>(&tmp[0]), nNeighborBlock[i]*sizeof(PosInt));
				neighborBlockId[i] = tmp;
				memcpy(&nBlockId[i*nearNeighborBlock], &tmp[0], nNeighborBlock[i]*sizeof(PosInt));
			}
            // assert copy success
            //cout << "block " << i << " has " << nNeighborBlock[i] << " neighbors:\n";
            //for (PosInt j=0; j<nNeighborBlock[i]; j++) {
            //    cout << nBlockId[i*nearNeighborBlock + j];
            //    if (j < nNeighborBlock[i] - 1) cout << ",";
            //}
            //cout << "\n";
		}
	}

	Size* d_neighborInfo;
	size_t neighborInfo_size = nblock*sizeof(Size) + nblock*nearNeighborBlock*sizeof(PosInt) + nV1*sizeof(Size);
	checkCudaErrors(cudaMalloc((void**)&d_neighborInfo, neighborInfo_size));
	Size* d_nNeighborBlock = d_neighborInfo;
	PosInt* d_neighborBlockId = (PosInt*) (d_nNeighborBlock + nblock);
	Size* d_nVec = (Size*) (d_neighborBlockId + nblock*nearNeighborBlock);

	checkCudaErrors(cudaMemcpy(d_nNeighborBlock, &(nNeighborBlock[0]), nblock*sizeof(Size), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_neighborBlockId, nBlockId, nblock*nearNeighborBlock*sizeof(PosInt), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_nVec, &(nVec[0]), nV1*sizeof(Size), cudaMemcpyHostToDevice));

	delete [] nBlockId;
	usingGMem += neighborInfo_size;
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;

    if (manual) {
        auto get_dis = [] (double a, double b) {
            Float dis = static_cast<Float>(square_root(a*a + b*b));
            return dis;
        };
        if (preList.size() != postList.size()) {
            cout << "the number of presynaptic neurons: " << preList.size() << " does not match with the number of postsynaptic neurons: " << postList.size() << " for the manual connections\n";
            return EXIT_FAILURE;
        } else {
            if (sList.size() == 1) {
                sList.assign(preList.size(), sList[0]);
            } else {
                if (preList.size() != postList.size()) {
                    cout << "the number of presynaptic neurons: " << preList.size() << " does not match with the number of connection strengths provided: " << sList.size() << " for the manual connections\n";
                }
            }
            nFar += sList.size();
            for (PosInt i = 0; i<postList.size(); i++) {
                PosInt bid = postList[i]/blockSize;
                PosInt pre_bid = preList[i]/blockSize;
                bool blocked = false;
                PosInt local_bid;
                for (PosInt ib=0; ib<nNeighborBlock[bid]; ib++) {
                    if (pre_bid == neighborBlockId[bid][ib]) {
                        blocked = true;
                        local_bid = ib;
                        break;
                    }
                }
                if (blocked) {
                    PosInt tid = postList[i]%blockSize;
                    PosInt iChunk;
                    if (bid >= iSizeSplit*maxChunkSize) {
                        iChunk = iSizeSplit + (bid - iSizeSplit*maxChunkSize)/remainChunkSize;
                        bid = bid - iSizeSplit*maxChunkSize - (iChunk-iSizeSplit)*remainChunkSize; // bid in the chunk
                    } else {
                        iChunk = bid/maxChunkSize;
                        bid = bid - iChunk*maxChunkSize;
                    }
                    // conMat: [nblock,nearNeighborBlock,blockDim.x,blockDim.x] last dim is the post-id: second-last pre-id
                    PosIntL mid = static_cast<PosIntL>(bid)*nearNeighborBlock*blockSize*blockSize + static_cast<PosIntL>(local_bid)*blockSize*blockSize + preList[i]*blockSize + postList[i];
                    float old_s = conMat[iChunk][mid];
                    conMat[iChunk][mid] = static_cast<float>(sList[i]);
                    cout << "changed the connection strength of post-id: " << postList[i] << " from pre-id: "  << preList[i] << " from " << old_s << " to " << sList[i] << ", at a distance of " << delayMat[iChunk][mid] << "mm\n";
                } else {
                    bool exist = false;
                    PosInt vid;
                    for (PosInt j=0; j<nVec[postList[i]]; j++) {
                        if (vecID[postList[i]][j] == preList[i]) {
                            exist = true;
                            vid = j;
                            break;
                        }
                    }
                    if (exist) {
                        Float old_s = conVec[postList[i]][vid];
                        conVec[postList[i]][vid] = sList[i];
                        cout << "changed the connection strength of post-id: " << postList[i] << " from pre-id: "  << preList[i] << " from " << old_s << " to " << sList[i] << ", at a distance of " << delayVec[postList[i]][vid] << "mm\n";
                    } else {
                        nVec[postList[i]] += 1;
                        vecID[postList[i]].push_back(preList[i]);
                        conVec[postList[i]].push_back(sList[i]);
                        Float distance = get_dis(V1_x[postList[i]]-V1_x[preList[i]], V1_y[postList[i]]-V1_y[preList[i]]);
                        delayVec[postList[i]].push_back(distance);
                        cout << "added connection to post-id: " << postList[i] << " from pre-id: "  << preList[i] << " with strength " << sList[i] << ", at a distance of " << distance << "mm\n";
                    }
                }
            }
        }
    }
	Float maxDistance = 0;
	for (PosInt i=0; i<nChunk; i++) {
        Float current_maxDistance = findDisConMax(delayMat[i], conMat[i], chunkSize);
	    if (maxDistance < current_maxDistance) maxDistance = current_maxDistance;
    }
	Size trainDepth = static_cast<Size>(ceil((maxDistance/speedOfThought)/dt)) + 1;
	cout << "spikeTrain retains spikes for " << trainDepth << " time steps for each neuron, calculated from a maximum connection distance " << maxDistance << " mm\n";
	delete []conMat;
	delete []delayMat;
    
    // a spikeTrain container that retains spikes before it reaches the post-synaptic neuron 
    // TODO: learning distant connections 
    vector<vector<vector<Float>>> fSpikeTrain(nV1, vector<vector<Float>>());
    vector<vector<Size>> fTrainDepth(nV1, vector<Size>());
    vector<vector<PosInt>> fCurrenSlot(nV1, vector<PosInt>());

    Float fMaxDistance = 0;
    for (PosInt i=0; i<nV1; i++) {
        vector<vector<Float>> preSpikeTrain(nVec[i], vector<Float>());
        fSpikeTrain[i] = preSpikeTrain;
        for (PosInt j=0; j<nVec[i]; j++) {
            fTrainDepth[i].push_back(static_cast<Size>(ceil((delayVec[i][j]/speedOfThought)/dt))+1);
            fCurrenSlot[i].push_back(0);
            vector<Float> iPreSpikeTrain(fTrainDepth[i][j], -1.0);
            fSpikeTrain[i][j] = iPreSpikeTrain;
        }
        if (nVec[i] > 0) {
            Float current_maxDistance = *max_element(delayVec[i].begin(), delayVec[i].end());
	        if (fMaxDistance < current_maxDistance) {
                fMaxDistance = current_maxDistance;
            }
        }
    }
    if (fMaxDistance > 0) {
        Size maxTrainDepth = static_cast<Size>(ceil((fMaxDistance/speedOfThought)/dt));
	    cout << "fSpikeTrain retains maximum " << fMaxDistance << " time steps for farther connections, calculated from a maximum distance of " << fMaxDistance << " mm\n";
    } else {
        assert(accumulate(nVec.begin(), nVec.end(), 0.0) == 0);
    }
    cout << "vector connections set\n";

	// pinned memory on CPU for heavy usages on
	// spikeTraie, voltage, gFF, gE, hE, gI, hI
	Float* pinnedMem;
	size_t trainSize = trainDepth*nV1;
	size_t ghSize = 2*nV1*(ngTypeE + ngTypeI)*sizeof(Float);
	size_t vSize = nV1*sizeof(Float);
	size_t wSize = vSize;
	size_t depSize = vSize;
    size_t ffSize = 2*nV1*ngTypeFF*sizeof(Float);
	size_t pinnedSize = depSize + trainSize*sizeof(Float) + vSize + ghSize + ffSize;
	if (iModel == 1) {
		pinnedSize += wSize;
	}
	checkCudaErrors(cudaMallocHost((void**) &pinnedMem, pinnedSize));
	Float *spikeTrain = pinnedMem;
	Float *depC;
	Float *w;
	Float *v;
	depC = spikeTrain + trainSize;
	if (iModel == 0) {
		v = depC + nV1;
	}
	if (iModel == 1) {
		w = depC + nV1;
		v = w + nV1;
	}
	Float *gFF = v + nV1;
	Float *hFF = gFF +  nV1*ngTypeFF;
	Float **gE = new Float*[nChunk];
	Float **gI = new Float*[nChunk];
	Float **hE = new Float*[nChunk];
	Float **hI = new Float*[nChunk];

	size_t eSize = maxChunkSize*blockSize*ngTypeE;
	size_t iSize = maxChunkSize*blockSize*ngTypeI;
	gE[0] = hFF + nV1*ngTypeFF;
	gI[0] = gE[0] + eSize;
	hE[0] = gI[0] + iSize;
	hI[0] = hE[0] + eSize;
	for (PosInt i = 1; i<nChunk; i++) {
		gE[i] = hI[i-1] + iSize; 
		if (i >= iSizeSplit) {
			eSize = remainChunkSize*blockSize*ngTypeE;
			iSize = remainChunkSize*blockSize*ngTypeI;
		}
		gI[i] = gE[i] + eSize;
		hE[i] = gI[i] + iSize;
		hI[i] = hE[i] + eSize;
	}
    assert(hI[nChunk-1] + iSize == pinnedMem + pinnedSize/sizeof(Float));

	// GPU arrays to receive g,h sum from conVec (pinned), FF not needed
	Float *d_gh;
	checkCudaErrors(cudaMalloc((void**)&d_gh, ghSize));
	Float **d_gEt = new Float*[nChunk];
	Float **d_gIt = new Float*[nChunk];
	Float **d_hEt = new Float*[nChunk];
	Float **d_hIt = new Float*[nChunk];
	eSize = maxChunkSize*blockSize*ngTypeE;
	iSize = maxChunkSize*blockSize*ngTypeI;
	d_gEt[0] = d_gh;
	d_gIt[0] = d_gEt[0] + eSize;
	d_hEt[0] = d_gIt[0] + iSize;
	d_hIt[0] = d_hEt[0] + eSize;
	for (PosInt i = 1; i<nChunk; i++) {
		d_gEt[i] = d_hIt[i-1] + iSize; 
		if (i >= iSizeSplit) {
			eSize = remainChunkSize*blockSize*ngTypeE;
			iSize = remainChunkSize*blockSize*ngTypeI;
		}
		d_gIt[i] = d_gEt[i] + eSize;
		d_hEt[i] = d_gIt[i] + iSize;
		d_hIt[i] = d_hEt[i] + eSize;
	}
    assert(d_hIt[nChunk-1] + iSize == d_gh + ghSize/sizeof(Float));
	usingGMem += ghSize;
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;

	// v, g (D2H), h (D only)
	Float *d_vgh;
	size_t totalSize = vSize + depSize + ghSize + ffSize;
	if (iModel == 1) {
		totalSize += wSize; 
	}
	checkCudaErrors(cudaMalloc((void**)&d_vgh, totalSize));
	usingGMem += totalSize;
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;

	Float *d_depC;
	Float *d_w;
	Float *d_v;
	d_depC = d_vgh;
	if (iModel == 0) {
		d_v = d_depC + nV1;
	}
	if (iModel == 1) {
		d_w = d_depC + nV1;
		d_v = d_w + nV1;
	} 
	Float *d_gFF = d_v + nV1;
	Float *d_hFF = d_gFF + nV1*ngTypeFF;
	Float **d_gE = new Float*[nChunk];
	Float **d_gI = new Float*[nChunk];
	Float **d_hE = new Float*[nChunk];
	Float **d_hI = new Float*[nChunk];
	eSize = maxChunkSize*blockSize*ngTypeE;
	iSize = maxChunkSize*blockSize*ngTypeI;
	d_gE[0] = d_hFF + nV1*ngTypeFF;
	d_gI[0] = d_gE[0] + eSize;
	d_hE[0] = d_gI[0] + iSize;
	d_hI[0] = d_hE[0] + eSize;
	for (PosInt i = 1; i<nChunk; i++) {
		d_gE[i] = d_hI[i-1] + iSize; 
		if (i >= iSizeSplit) {
			eSize = remainChunkSize*blockSize*ngTypeE;
			iSize = remainChunkSize*blockSize*ngTypeI;
		}
		d_gI[i] = d_gE[i] + eSize;
		d_hE[i] = d_gI[i] + iSize;
		d_hI[i] = d_hE[i] + eSize;
	}
    assert(d_hI[nChunk-1] + iSize == d_vgh + totalSize/sizeof(Float));

    // pass d_gE to kernels
    Float **dd_gE, **dd_gI, **dd_hE, **dd_hI;
    checkCudaErrors(cudaMalloc((void**)&dd_gE, nChunk*sizeof(Float*)));
    checkCudaErrors(cudaMemcpy(dd_gE, d_gE, nChunk*sizeof(Float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&dd_gI, nChunk*sizeof(Float*)));
    checkCudaErrors(cudaMemcpy(dd_gI, d_gI, nChunk*sizeof(Float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&dd_hE, nChunk*sizeof(Float*)));
    checkCudaErrors(cudaMemcpy(dd_hE, d_hE, nChunk*sizeof(Float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&dd_hI, nChunk*sizeof(Float*)));
    checkCudaErrors(cudaMemcpy(dd_hI, d_hI, nChunk*sizeof(Float*), cudaMemcpyHostToDevice));
    usingGMem += nChunk*sizeof(Float*)*4;
    cout << "conductance setup in chunks\n";

    curandStateMRG32k3a *rGenCond;
    curandStateMRG32k3a *rNoisy;
    Float *d_noisyDep;
    Float *d_synFailFF;
    Float *d_synFail;
    Float *d_synPerConFF;
    Float *d_synPerCon;
    Size *typeAcc;
    Float *d_pFF;
    Float *d_pE;
    Float *d_pI;

    Float *d_tonicDep;
    checkCudaErrors(cudaMalloc((void**)&d_tonicDep, nV1*sizeof(Float)));

    checkCudaErrors(cudaMalloc((void**)&rGenCond, nV1*sizeof(curandStateMRG32k3a)));
    checkCudaErrors(cudaMalloc((void**)&rNoisy, nV1*sizeof(curandStateMRG32k3a)));
    checkCudaErrors(cudaMalloc((void**)&d_noisyDep, nType*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void**)&d_synFailFF, nType*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void**)&d_synFail, nType*nType*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void**)&d_synPerConFF, nType*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void**)&d_synPerCon, nType*nType*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void**)&typeAcc, nType*sizeof(Size)));
    checkCudaErrors(cudaMalloc((void**)&d_pFF, nType*ngTypeFF*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void**)&d_pE,  nType*ngTypeE*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void**)&d_pI,  nType*ngTypeI*sizeof(Float)));
	checkCudaErrors(cudaMemcpy(d_noisyDep,  &(noisyDep[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_synFailFF,  &(synFailFF[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_synFail,  &(synFail[0]), nType*nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_synPerConFF, &(synPerConFF[0]), nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_synPerCon, &(synPerCon[0]), nType*nType*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(typeAcc, &(typeAccCount[0]), nType*sizeof(Size), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_pFF, &(pFF[0]), nType*ngTypeFF*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_pE,  &(pE[0]),  nType*ngTypeE*sizeof(Float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_pI,  &(pI[0]),  nType*ngTypeI*sizeof(Float), cudaMemcpyHostToDevice));
	usingGMem += 2*nV1*sizeof(curandStateMRG32k3a) + (ngTypeFF + ngTypeE + ngTypeI)*nType*sizeof(Float) + (2+nType)*nType * sizeof(Float) + (2+nType)*nType*sizeof(Size) + nV1*sizeof(Float);
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;
	cout << "synFail, receptor ratio set\n";

	vector<Float> iTonicDep(nV1, 0);
	fConStats.open(conStats_filename + conV1_suffix, fstream::in | fstream::binary);
	if (!fConStats) {
		cout << "Cannot open or find " << conStats_filename <<" to read V1 ExcRatios.\n";
		return EXIT_FAILURE;
	} else {
		Size discard;
		vector<float> ExcRatio(nV1, 0);

		fConStats.read(reinterpret_cast<char*>(&discard),sizeof(Size));
		assert(discard == nType);
    	fConStats.read(reinterpret_cast<char*>(&discard),sizeof(Size));
		assert(discard == nV1);
    	fConStats.read(reinterpret_cast<char*>(&ExcRatio[0]), nV1*sizeof(float));
		fConStats.close();
		Float maxExcRatio = *max_element(ExcRatio.begin(), ExcRatio.end());
		for (PosInt i=0; i<nblock; i++) {
			PosInt iType = 0;
			for (PosInt j=0; j<blockSize; j++) {
				PosInt id = i*blockSize + j;
				if (j == typeAccCount[iType]) {
					iType++;
				}
				iTonicDep[id] = tonicDep[iType]*ExcRatio[id]/maxExcRatio;
				assert(!std::isnan(ExcRatio[id]));
				assert(!std::isnan(iTonicDep[id]));
				assert(iTonicDep[id] >= 0);
				assert(iTonicDep[id] <= 1);
			}
		}
		checkCudaErrors(cudaMemcpy(d_tonicDep,  &(iTonicDep[0]), nV1*sizeof(Float), cudaMemcpyHostToDevice));
	}

    default_random_engine *h_rGenCond = new default_random_engine[nV1];
    for (PosInt i=0; i<nV1; i++) {
        h_rGenCond[i].seed(seed);
        seed++;
    }

	// for spikeTrain D2H (only output the current slot to file)
	Float *d_spikeTrain;
	checkCudaErrors(cudaMalloc((void**)&d_spikeTrain, (trainSize + nV1)*sizeof(Float)));
	Float *tBack = d_spikeTrain + trainSize;
	usingGMem += (trainSize + nV1)*sizeof(Float);
    checkCudaErrors(cudaMemset(d_spikeTrain, 0, (nV1+trainSize)*sizeof(Float)));

    Float *sp0 = new Float[nType*2];
    for (PosInt i=0; i<spE0.size(); i++) {
        sp0[i] = spE0[i];
    }
    for (PosInt i=0; i<spI0.size(); i++) {
        sp0[spE0.size() + i] = spI0[i];
    }

	if (print_log) {
    	cout << "sp0: ";
        for (PosInt i=0; i<nType*2; i++) {
			cout << sp0[i];
			if (i == nType*2 -1) cout << "\n";
			else cout << ", ";
        }
	}
    Float *d_sp0;
	checkCudaErrors(cudaMalloc((void**)&d_sp0, nType*2*sizeof(Float)));
	checkCudaErrors(cudaMemcpy(d_sp0, sp0,  nType*2*sizeof(Float), cudaMemcpyHostToDevice));
	usingGMem += nType*2*sizeof(Float);
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;
	
	// for spikeTrain D2H (only output the current slot to file)
	Float *d_ogh;
	Size ncond = nV1*(ngTypeE + ngTypeI);
	

	Size max_LGNperV1;
	Float* LGN_V1_s;
	read_LGN(LGN_V1_s_filename + conLGN_suffix, LGN_V1_s, max_LGNperV1, &(sRatioLGN[0]), &(typeAccCount[0]), nType, learnData_FF, nV1 <= 32); // assign LGN_V1_s and max_LGNperV1
	Float* sLGN;
	size_t sLGN_size = static_cast<size_t>(max_LGNperV1)*nV1*sizeof(Float);
	checkCudaErrors(cudaMalloc((void**)&sLGN, sLGN_size));
	usingGMem += sLGN_size;
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;

	checkCudaErrors(cudaMemcpy(sLGN, LGN_V1_s, sLGN_size, cudaMemcpyHostToDevice));
    if (!learnData_FF) { // free memory if not used later
	    delete []LGN_V1_s;
    }
    Float* d_LGN_sInfo;
    if (learnData_FF || getLGN_sp) {
        checkCudaErrors(cudaMalloc((void**)&d_LGN_sInfo, nLGN*sizeof(Float)));
		usingGMem += nLGN*sizeof(Float);
    }

    Float* LGN_sInfo;
    if (getLGN_sp) {
	    checkCudaErrors(cudaMallocHost((void**) &LGN_sInfo, nLGN*sizeof(Float)));
    }

	vector<vector<PosInt>> LGN_V1_ID = read_listOfList<PosInt>(LGN_V1_ID_filename + conLGN_suffix, false);
    Size avgLGNperV1 = 0;
    for (PosInt i=0; i<nV1; i++) {
        avgLGNperV1 += LGN_V1_ID[i].size();
    }
    avgLGNperV1 /= nV1;
	cout << "maximum LGN per V1: " << max_LGNperV1 << ", " << avgLGNperV1 << " on average\n";
	size_t surfacePosSize = static_cast<size_t>(2*max_LGNperV1)*nV1*sizeof(int) + nV1*sizeof(Size);
	char* surfacePos = new char[surfacePosSize];
	int* surface_xy = (int*) surfacePos;
	Size* nLGNperV1 = (Size*) (surface_xy + 2*max_LGNperV1*nV1);
	getLGN_V1_surface(sxyID, LGN_V1_ID, surface_xy, nLGNperV1, max_LGNperV1, nLGN);
    cout << "LGN->V1 surface constructed\n";

	// release memory from LGN_V1_ID and sxyID
	for (PosInt i=0; i<nV1; i++) {
		vector<PosInt>().swap(LGN_V1_ID[i]);
	}
	vector<vector<PosInt>>().swap(LGN_V1_ID);
	vector<int>().swap(sxyID);

	int* d_surfacePos;
	checkCudaErrors(cudaMalloc((void**)&d_surfacePos, surfacePosSize));
	int* LGN_idx = d_surfacePos;
	int* LGN_idy = LGN_idx + max_LGNperV1*nV1;
	Size* d_nLGNperV1 = (Size*) (LGN_idy + max_LGNperV1*nV1);
	usingGMem += surfacePosSize;
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;
	checkCudaErrors(cudaMemcpy(d_surfacePos, surfacePos, surfacePosSize, cudaMemcpyHostToDevice));

    
	// learning Vars
    Float *lVarFFpre;
    if (learnData_FF) {
		checkCudaErrors(cudaMalloc((void**)&lVarFFpre, LGN_learnPreSize));
    	usingGMem += LGN_learnPreSize;
	}
    // 
    size_t learnVarFFsize0 = (nE*nLearnTypeFF_E + nI*nLearnTypeFF_I)*2*nblock;
    size_t learnVarFFsize1 = nV1 + nE*nblock;
    size_t learnVarFFsize = learnVarFFsize0 + learnVarFFsize1;
	size_t LGN_learnPostSize = learnVarFFsize * sizeof(Float);
    Float* lVarFFpost;
    if (learnData_FF) checkCudaErrors(cudaMallocHost((void**)&lVarFFpost, learnVarFFsize*sizeof(Float)));
    size_t learnVarSize =  learnVarFFsize +
                           nE*nblock*(2+trainDepth)*2*nLearnTypeE + // vLTP_E, vLTD_E and vTripE
                           nE*nblock*2*nLearnTypeQ +  //vSTDP_QE
                           nI*nblock*2*trainDepth*nLearnTypeQ; // vSTDP_QI
    Float* learnVar;
    checkCudaErrors(cudaMalloc((void **) &learnVar, learnVarSize*sizeof(Float)));
    checkCudaErrors(cudaMemset(learnVar, 0, learnVarSize*sizeof(Float)));
    Float* vLTD_FF_E = learnVar;                              //    post, [nLearnTypeFF_E,        nblock, nE         ]
    Float* vTrip_FF_E = vLTD_FF_E + nLearnTypeFF_E*nblock*nE;  //   post, [nLearnTypeFF_E,        nblock, nE         ]
    Float* vLTD_FF_I = vTrip_FF_E + nLearnTypeFF_E*nblock*nE; //    post, [nLearnTypeFF_I,        nblock, nI         ]
    Float* vTrip_FF_I = vLTD_FF_I + nLearnTypeFF_I*nblock*nI; //    post, [nLearnTypeFF_I,        nblock, nI         ]
    Float* vAvgE = vTrip_FF_I + nLearnTypeFF_I*nblock*nI; //        post, [                       nblock, nE,       2]  2 is for decay and w/o decay
    Float* vAvgI = vAvgE + nblock*nE*2;                   //        post, [                       nblock, nI,        ]
    Float* vLTP_E = vAvgI + nblock*nI;                   //         pre,  [nLearnTypeE,    depth, nblock, nE,       2]
    Float* vLTD_E = vLTP_E + nLearnTypeE*trainDepth*nblock*nE*2; // post, [nLearnTypeE,           nblock, nE,       2]
    Float* vTripE = vLTD_E + nLearnTypeE*nblock*nE*2;      //       post, [nLearnTypeE,           nblock, nE,       2]
    Float* vSTDP_QE = vTripE + nLearnTypeE*nblock*nE*2;       //  E post, [nLearnTypeQ,           nblock, nE        2]
    Float* vSTDP_QI = vSTDP_QE + nLearnTypeQ*nblock*nE*2;     //   I pre, [nLearnTypeQ,    depth, nblock, nI,       2]

    usingGMem += learnVarSize*sizeof(Float);
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;
	cout << "implementing LGN_surface requires " << surfacePosSize/1024.0/1024.0 << " Mb\n";

	if (restore.empty()) {
    	default_random_engine rGen_initV1(seed);
    	seed++;
    	auto get_excRangeBound = [](Float lower, Float upper) {
			function<bool(Float)> bound = [lower, upper](Float value) {
				return (value >= upper || value <= lower);
			};
			return bound;
    	};
    	auto nonNegativeBound = get_incLowBound(0.0);
    	function<bool(Float)> noBound = [](Float value) {return false;};

		auto get_rand_from_norm = [](default_random_engine &rGen, Float mean, Float std, function<bool(Float)> &outOfBound) {
			function<float()> get_rand = [&rGen, mean, std, &outOfBound] () { // dont use ref capture on mean and std
		        static normal_distribution<Float> norm(0.0, 1.0);
				Float rand;
				Size count = 0;
				do {
					rand = norm(rGen)*std + mean;
					count++;
				} while (outOfBound(rand));
				return rand;
			};
			return get_rand;
		};

    	Float *h_w0 = w;
    	Float *h_v0 = v;
    	Float *h_gFF0 = gFF;

    	for (PosInt i=0; i<nType; i++) {
			if (iModel == 1) {
    	    	auto get_w0 = get_rand_from_norm(rGen_initV1, w0[i*2+0], w0[i*2+1], noBound);
    	    	for (PosInt iblock = 0; iblock < nblock; iblock++) {
				    generate(h_w0 + iblock*blockSize + typeAcc0[i], h_w0 + iblock*blockSize + typeAcc0[i+1], get_w0);
    	    	    assert(h_w0 + iblock*blockSize + typeAcc0[i+1] <= h_w0 + nV1);
    	    	}
			}
			if (use_v0) {
    			auto vBound = get_excRangeBound(vI, vT[i]);
    			auto get_v0 = get_rand_from_norm(rGen_initV1, v0[i*2+0], v0[i*2+1], vBound);
    			for (PosInt iblock = 0; iblock < nblock; iblock++) {
				    generate(h_v0 + iblock*blockSize + typeAcc0[i], h_v0 + iblock*blockSize + typeAcc0[i+1], get_v0);
    			    assert(h_v0 + iblock*blockSize + typeAcc0[i+1] <= h_v0 + nV1);
    			}
			} else {
    			for (PosInt iblock = 0; iblock < nblock; iblock++) {
					for (PosInt j=typeAcc0[i]; j<typeAcc0[i+1]; j++) {
						h_v0[iblock*blockSize + j] = vR[i] + (vT[i]-vR[i])*iTonicDep[iblock*blockSize + j];
					}
				}
			}
    	}

		Float v0Mean = accumulate(h_v0, h_v0 + nV1, 0.0)/nV1;
		Float v0Min = *min_element(h_v0, h_v0 + nV1);
		Float v0Max = *max_element(h_v0, h_v0 + nV1);
		cout << "v0 = ["  << v0Min << ", " << v0Mean << ", " << v0Max << "]\n";
    	for (PosInt i=0; i<nType; i++) {
    	    for (PosInt iblock = 0; iblock < nblock; iblock++) {
    	        for (PosInt j=0; j<blockSize; j++) {
    	            assert(h_v0[iblock*blockSize + j] < vThres[i]);
    	        }
			}
		}

		if (iModel == 1) {
			Float w0Mean = accumulate(h_w0, h_w0 + nV1, 0.0)/nV1;
			Float w0Min = *min_element(h_w0, h_w0 + nV1);
			Float w0Max = *max_element(h_w0, h_w0 + nV1);
			cout << "w0 = ["  << w0Min << ", " << w0Mean << ", " << w0Max << "]\n";
		}

    	for (PosInt i=0; i<nType; i++) {
    	    for (PosInt k = 0; k<ngTypeFF; k++) {
    	        auto get_gFF0 = get_rand_from_norm(rGen_initV1, gFF0[i*ngTypeFF*4+k*4+0], gFF0[i*ngTypeFF*4+k*4+0]*gFF0[i*ngTypeFF*4+k*4+1], nonNegativeBound);
    	        auto get_hFF0 = get_rand_from_norm(rGen_initV1, gFF0[i*ngTypeFF*4+k*4+2], gFF0[i*ngTypeFF*4+k*4+2]*gFF0[i*ngTypeFF*4+k*4+3], nonNegativeBound);
    	        for (PosInt iblock = 0; iblock < nblock; iblock++) {
    	            PosIntL ind = k*nV1 + iblock*blockSize;
		    	    generate(h_gFF0 + ind + typeAcc0[i], h_gFF0 + ind + typeAcc0[i+1], get_gFF0);
    	            ind += ngTypeFF*nV1;
		    	    generate(h_gFF0 + ind + typeAcc0[i], h_gFF0 + ind + typeAcc0[i+1], get_hFF0);
    	            for (PosInt j=typeAcc0[i]; j<typeAcc0[i+1]; j++) {
    	                PosIntL ind = iblock*blockSize + j;
    	                if (nLGNperV1[iblock*blockSize + j] == 0) {
    	                    ind += k*nV1;
    	                    h_gFF0[ind] = 0;
    	                    h_gFF0[ind+ngTypeFF*nV1] = 0;
    	                }
    	            }
    	        }
    	    }
    	}

		Float gFFMean = accumulate(h_gFF0, h_gFF0 + nV1, 0.0)/nV1;
		Float gFFMin = *min_element(h_gFF0, h_gFF0 + nV1);
		Float gFFMax = *max_element(h_gFF0, h_gFF0 + nV1);
		cout << "gFF0 = ["  << gFFMin << ", " << gFFMean << ", " << gFFMax << "]\n";

		if (iModel == 0) {
    		checkCudaErrors(cudaMemcpy(d_depC + nV1, v, vSize + ffSize, cudaMemcpyHostToDevice));
    		cout << "v, gFF...\n"; 
		} 
		if (iModel == 1) {
    		checkCudaErrors(cudaMemcpy(d_depC + nV1, w, wSize + vSize + ffSize, cudaMemcpyHostToDevice));
    		cout << "w, v, gFF...\n"; 
		}

		Float gEMean = 0;
		Float gIMean = 0;
    	for (PosInt i=0; i<nType; i++) {
    	    for (PosInt k = 0; k<ngTypeE; k++) {
    	        PosInt ind = i*ngTypeE*4+k*4;
    	        auto get_gE0 = get_rand_from_norm(rGen_initV1, gE0[ind+0], gE0[ind+0]*gE0[ind+1], nonNegativeBound);
    	        auto get_hE0 = get_rand_from_norm(rGen_initV1, gE0[ind+2], gE0[ind+2]*gE0[ind+3], nonNegativeBound);
    	        Size chunkSize = maxChunkSize;
		        for (PosInt q = 0; q<nChunk; q++) {
		        	if (q >= iSizeSplit) {
    	                chunkSize = remainChunkSize;
		        	}
    	            for (PosInt iblock = 0; iblock < chunkSize; iblock++) {
    	                PosInt offset = k*chunkSize*blockSize + iblock*blockSize;
		    	        generate(gE[q] + offset + typeAcc0[i], gE[q] + offset + typeAcc0[i+1], get_gE0);
		    	        generate(hE[q] + offset + typeAcc0[i], hE[q] + offset + typeAcc0[i+1], get_hE0);
						gEMean = accumulate(gE[q] + offset + typeAcc0[i], gE[q] + offset + typeAcc0[i+1], gEMean);
    	            }
		        }
    	    }
    	    for (PosInt k = 0; k<ngTypeI; k++) {
    	        PosInt ind = i*ngTypeI*4+k*4;
    	        auto get_gI0 = get_rand_from_norm(rGen_initV1, gI0[ind+0], gI0[ind+0]*gI0[ind+1], nonNegativeBound);
    	        auto get_hI0 = get_rand_from_norm(rGen_initV1, gI0[ind+2], gI0[ind+2]*gI0[ind+3], nonNegativeBound);
    	        Size chunkSize = maxChunkSize;
		        for (PosInt q = 0; q<nChunk; q++) {
		        	if (q >= iSizeSplit) {
    	                chunkSize = remainChunkSize;
		        	}
    	            for (PosInt iblock = 0; iblock < chunkSize; iblock++) {
    	                PosInt offset = k*chunkSize*blockSize + iblock*blockSize;
		    	        generate(gI[q] + offset + typeAcc0[i], gI[q] + offset + typeAcc0[i+1], get_gI0);
		    	        generate(hI[q] + offset + typeAcc0[i], hI[q] + offset + typeAcc0[i+1], get_hI0);
						gIMean = accumulate(gI[q] + offset + typeAcc0[i], gI[q] + offset + typeAcc0[i+1], gIMean);
    	            }
		        }
    	    }
    	}

    	checkCudaErrors(cudaMemcpy(d_gE[0], gE[0], ghSize, cudaMemcpyHostToDevice));
    	cout << "gE, gI...\n"; 

		cout << "mean(gE0) =  " << gEMean/nV1/ngTypeE << "\n";
		cout << "mean(gI0) =  " << gIMean/nV1/ngTypeI << "\n";

    	rand_spInit<<<nblock, blockSize>>>(tBack, d_spikeTrain, d_v, d_w, d_nLGNperV1, d_sp0, typeAcc, d_vR, d_tRef, d_tau_w, d_a, d_b, rGenCond, rNoisy, seed, nV1, nType, SCsplit, trainDepth, dt, iModel);
    	checkCudaErrors(cudaDeviceSynchronize());

		//debug
    		checkCudaErrors(cudaMemcpy(spikeTrain, d_spikeTrain, trainSize*sizeof(Float), cudaMemcpyDeviceToHost));
			for (PosInt i = 0; i<trainDepth; i++) {
				for (PosInt j=0; j<nV1; j++) {
					assert(!std::isnan(spikeTrain[i*nV1 + j]));
					if (spikeTrain[i*nV1 + j] < 1) {
						assert(spikeTrain[i*nV1 + j] == 0);
					}
				}
			}
			if (iModel == 0) {
				for (PosInt j=0; j<(ffSize + ghSize + vSize)/sizeof(Float); j++) {
					assert(!std::isnan(v[j]));
				}
			}
			if (iModel == 1) {
				for (PosInt j=0; j<(ffSize + ghSize + vSize + wSize)/sizeof(Float); j++) {
					assert(!std::isnan(w[j]));
				}
			}
		//
    	cout << "spiking... V1 initialized\n"; 
    	#ifdef CHECK
    	    getLastCudaError("spiking initialized");
    	#endif
    	seed++;
    	//checkCudaErrors(cudaMemset(d_spikeTrain, 0, nV1*trainDepth*sizeof(Float)));
    	delete []sp0;
		delete []surfacePos;
    	checkCudaErrors(cudaFree(d_sp0));
    	if (has_sp0) {
    	    #ifdef CHECK
		    	checkCudaErrors(cudaMemcpy(spikeTrain, d_spikeTrain, trainSize*sizeof(Float), cudaMemcpyDeviceToHost)); // to overlap with  recal_G, to be used in recal_Gvec
    	    #else
		        cudaMemcpy(spikeTrain, d_spikeTrain, trainSize*sizeof(Float), cudaMemcpyDeviceToHost); // to overlap with  recal_G, to be used in recal_Gvec
    	    #endif
			// debug
				for (PosInt i = 0; i<trainDepth; i++) {
					for (PosInt j=0; j<nV1; j++) {
						if (spikeTrain[i*nV1 + j] < 1) {
							assert(spikeTrain[i*nV1 + j] == 0);
						}
					}
				}
			//
    	}
	} else {
		cout << "initialization is ignored\n";
	}


	/* read V1_RF
	   fV1_RF.open(V1_RF_filename, fstream::in | fstream::binary);
	   if (!fV1_RF) {
	   cout << "Cannot open or find " << V1_RF_filename <<" to read in V1 properties.\n";
	   return EXIT_FAILURE;
	   } else {
	   fV1_RF.read(reinterpret_cast<char*>(&nV1), sizeof(Size));
	   cout << nV1 << " V1 neurons\n";
	   }
	   vector<Float> V1_x(nV1);
	   vector<Float> V1_y(nV1);
	   vector<Float> a(nV1);
	   vector<Float> baRatio(nV1);
	   vector<Float> sfreq(nV1);
	   vector<Float> theta(nV1);
	   vector<Float> phase(nV1);
	   vector<Float> amp(nV1);
	   vector<Float> sig(nV1);
	   vector<RFtype> V1Type(nV1);
	   vector<OutputType> RefType(nV1);
	   { // read V1 properties
	   fV1_RF.read(reinterpret_cast<char*>(&V1_x[0]), nV1 * sizeof(Float));
	   fV1_RF.read(reinterpret_cast<char*>(&V1_y[0]), nV1 * sizeof(Float));
	   fV1_RF.read(reinterpret_cast<char*>(&a[0]), nV1 * sizeof(Float));
	   fV1_RF.read(reinterpret_cast<char*>(&baRatio[0]), nV1 * sizeof(Float));
	   fV1_RF.read(reinterpret_cast<char*>(&sfreq[0]), nV1 * sizeof(Float));
	   fV1_RF.read(reinterpret_cast<char*>(&theta[0]), nV1 * sizeof(Float));
	   fV1_RF.read(reinterpret_cast<char*>(&phase[0]), nV1 * sizeof(Float));
	   fV1_RF.read(reinterpret_cast<char*>(&amp[0]), nV1 * sizeof(Float));
	   fV1_RF.read(reinterpret_cast<char*>(&sig[0]), nV1 * sizeof(Float));
	   fV1_RF.read(reinterpret_cast<char*>(&V1Type[0]), nV1 * sizeof(RFtype_t));
	   fV1_RF.read(reinterpret_cast<char*>(&RefType[0]), nV1 * sizeof(OutputType_t));
	   fV1_RF.close();
	   }
	 */

	fstream fSnapshot;
	PosInt it0 = 0;
	{ // output file tests
		if (!restore.empty() && !asInit) {
			fSnapshot.open(restore, fstream::in | fstream::binary);
			if (!fSnapshot) {
				cout << "cannot restore from " << restore << "\n";
				return EXIT_FAILURE;
			} else {
				fSnapshot.read(reinterpret_cast<char*>(&it0), sizeof(PosInt));
				fSnapshot.close();
			}
		}
		Size nt0 = nt + it0;
		if (saveLGN_fr) {
			if (!restore.empty() && !asInit) {
				fLGN_fr.open(LGN_fr_filename + output_suffix, fstream::out | fstream::in | fstream::binary | fstream::ate);
			} else {
				fLGN_fr.open(LGN_fr_filename + output_suffix, fstream::out | fstream::binary);
			}
			if (!fLGN_fr) {
				cout << "Cannot open or find " << LGN_fr_filename + output_suffix <<" for LGN firing rate output\n";
				return EXIT_FAILURE;
			} else {
				if (!restore.empty() && !asInit) {
					streampos eof = fLGN_fr.tellp();
					fLGN_fr.seekp(0);
					fLGN_fr.write((char*)&nt0, sizeof(Size));
					fLGN_fr.seekp(eof);
				} else {
					fLGN_fr.write((char*)&nt, sizeof(Size));
					fLGN_fr.write((char*)&nLGN, sizeof(Size));
				}
			}
		}

        if (rawData) {
			if (!restore.empty() && !asInit) {
		    	fRawData.open(rawData_filename + output_suffix, fstream::out | fstream::in | fstream::binary | fstream::ate);
			} else {
		    	fRawData.open(rawData_filename + output_suffix, fstream::out | fstream::binary);
			}
		    if (!fRawData) {
		    	cout << "Cannot open or find " << rawData_filename + output_suffix <<" for V1 simulation results.\n";
		    	return EXIT_FAILURE;
		    } else {
				if (!restore.empty() && !asInit) {
					streampos eof = fRawData.tellp();
					fRawData.seekp(sizeof(Float));
					fRawData.write((char*)&nt0, sizeof(Size));
					fRawData.seekp(eof);
				} else {
		    		fRawData.write((char*) &dt, sizeof(Float));
		    		fRawData.write((char*) &nt, sizeof(Size));
		    		fRawData.write((char*) &nV1, sizeof(Size));
                	fRawData.write((char*) &iModel, sizeof(int));
                	PosInt iHwrite = static_cast<PosInt>(hWrite);
                	fRawData.write((char*) &iHwrite, sizeof(PosInt));
		    		fRawData.write((char*) &ngTypeFF, sizeof(Size));
		    		fRawData.write((char*) &ngTypeE, sizeof(Size));
		    		fRawData.write((char*) &ngTypeI, sizeof(Size));
				}
		    }
        }
        if (learnData_FF) {
			if (!restore.empty() && !asInit) {
		    	fLearnData_FF.open(learnData_FF_filename + output_suffix, fstream::out | fstream::in | fstream::binary | fstream::ate);
			} else {
		    	fLearnData_FF.open(learnData_FF_filename + output_suffix, fstream::out | fstream::binary);
			}
		    if (!fLearnData_FF) {
		    	cout << "Cannot open or find " << learnData_FF_filename + output_suffix <<" for data related to LGN->V1 plasticity.\n";
		    	return EXIT_FAILURE;
            } else {
				if (!restore.empty() && !asInit) {
					streampos eof = fLearnData_FF.tellp();
					fLearnData_FF.seekp(sizeof(Float));
					fLearnData_FF.write((char*)&nt0, sizeof(Size));
					fLearnData_FF.seekp(eof);
				} else {
		    		fLearnData_FF.write((char*) &dt, sizeof(Float));
		    		fLearnData_FF.write((char*) &nt, sizeof(Size));
		    		fLearnData_FF.write((char*) &nLGN, sizeof(Size));
		    		fLearnData_FF.write((char*) &nE, sizeof(Size));
		    		fLearnData_FF.write((char*) &nI, sizeof(Size));
		    		fLearnData_FF.write((char*) &nblock, sizeof(Size));
                	PosInt iRawData = static_cast<PosInt>(rawData);
		    		fLearnData_FF.write((char*) &iRawData, sizeof(PosInt));
		    		fLearnData_FF.write((char*) &max_LGNperV1, sizeof(Size));
		    		fLearnData_FF.write((char*) &nLearnTypeFF_E, sizeof(Size));
		    		fLearnData_FF.write((char*) &nLearnTypeFF_I, sizeof(Size));
				}
                if (rawData) {
                    cout << "find V1 spike and gFF in " << rawData_filename + output_suffix << " for learnData_FF\n";
                }
                // loop in LGN spike, V1 spike, LGN->V1 strength
            }
			if (!restore.empty() && !asInit) {
            	f_sLGN.open(sLGN_filename + output_suffix, fstream::out | fstream::in | fstream::binary | fstream::ate);
			} else {
            	f_sLGN.open(sLGN_filename + output_suffix, fstream::out | fstream::binary);
			}
            if (!f_sLGN) {
		    	cout << "Cannot open or find " << sLGN_filename + output_suffix <<" to store the LGN->V1 connection strength over time.\n";
		    	return EXIT_FAILURE;
            } else {
				if (!restore.empty() && !asInit) {
					streampos eof = f_sLGN.tellp();
					f_sLGN.seekp(0);
					f_sLGN.write((char*)&nt0, sizeof(Size));
					f_sLGN.seekp(eof);
				} else {
		    		f_sLGN.write((char*) &nt, sizeof(Size));
		    		f_sLGN.write((char*) &nV1, sizeof(Size));
		    		f_sLGN.write((char*) &max_LGNperV1, sizeof(Size));
		    		f_sLGN.write((char*) &sRatioLGN[0], sizeof(Float));
		    		f_sLGN.write((char*) &nLearnTypeFF, sizeof(Size));
		    		f_sLGN.write((char*) &(gmaxLGN[0]), nLearnTypeFF*sizeof(Float));
				}
            }
        }
        
        if (getLGN_sp) {
			if (!restore.empty() && !asInit) {
            	fLGN_sp.open(LGN_sp_filename + output_suffix, fstream::out | fstream::in |fstream::binary | fstream::ate);
			} else {
            	fLGN_sp.open(LGN_sp_filename + output_suffix, fstream::out | fstream::binary);
			}
            if (!fLGN_sp) {
		    	cout << "Cannot open or find " << LGN_sp_filename + output_suffix <<" to store the LGN tsp.\n";
		    	return EXIT_FAILURE;
            } else {
				if (!restore.empty() && !asInit) {
					streampos eof = fLGN_sp.tellp();
					fLGN_sp.seekp(sizeof(Float));
					fLGN_sp.write((char*)&nt0, sizeof(Size));
					fLGN_sp.seekp(eof);
				} else {
		    		fLGN_sp.write((char*) &dt, sizeof(Float));
		    		fLGN_sp.write((char*) &nt, sizeof(Size));
		    		fLGN_sp.write((char*) &nLGN, sizeof(Size));
				}
            }
        }

		if (!restore.empty() && !asInit) {
			fOutputFrame.open(outputFrame_filename + output_suffix, fstream::out | fstream::in | fstream::binary | fstream::ate);
		} else {
			fOutputFrame.open(outputFrame_filename + output_suffix, fstream::out | fstream::binary);
		}
		if (!fOutputFrame) {
			cout << "Cannot open or find " << outputFrame_filename + output_suffix <<" for output V1 simulation results to frames.\n";
			return EXIT_FAILURE;
		} else {
			if (restore.empty() || asInit) {
				fOutputFrame.write((char*)&dt, sizeof(Float));
				fOutputFrame.write((char*)&ot, sizeof(Size));
				fOutputFrame.write((char*)&iFrameOutput, sizeof(Size));
				if (framePhyV1output) { //
				    fOutputFrame.write((char*)&phyWidth, sizeof(Size));
				    fOutputFrame.write((char*)&phyHeight, sizeof(Size));
            	}
				if (frameVisV1output) {
					fOutputFrame.write((char*)&visWidth, sizeof(Size));
					fOutputFrame.write((char*)&visHeight, sizeof(Size));
				}
				if (frameVisLGNoutput) {
					fOutputFrame.write((char*)&visWidth, sizeof(Size));
					fOutputFrame.write((char*)&visHeight, sizeof(Size));
				}
			}
		}

		if (saveLGN_gallery) {
			fLGN_gallery.open(LGN_gallery_filename + output_suffix, fstream::out | fstream::binary);
			if (!fLGN_gallery) {
				cout << "Cannot open or find " << LGN_gallery_filename + output_suffix <<" for storage check.\n";
				return EXIT_FAILURE;
			} else {
				fLGN_gallery.write((char*)&nParvo, sizeof(Size));
				fLGN_gallery.write((char*)&nRegion, sizeof(Size));
				fLGN_gallery.write((char*)&nKernelSample, sizeof(Size));
				fLGN_gallery.write((char*)&nSample, sizeof(Size));
				fLGN_gallery.write((char*)&nMagno, sizeof(Size));
				fLGN_gallery.write((char*)&mRegion, sizeof(Size));
				fLGN_gallery.write((char*)&mKernelSample, sizeof(Size));
				fLGN_gallery.write((char*)&mSample, sizeof(Size));
				fLGN_gallery.write((char*)&nParvo_I, sizeof(Size));
				fLGN_gallery.write((char*)&nParvo_C, sizeof(Size));
				fLGN_gallery.write((char*)&nMagno_I, sizeof(Size));
				fLGN_gallery.write((char*)&nMagno_C, sizeof(Size));
			}
		}

		if (saveOutputB4V1) {
			if (!restore.empty() && !asInit) {
				fOutputB4V1.open(outputB4V1_filename + output_suffix, fstream::out | fstream::in | fstream::binary | fstream::ate);
			} else {
				fOutputB4V1.open(outputB4V1_filename + output_suffix, fstream::out | fstream::binary);
			}
			if (!fOutputB4V1) {
				cout << "Cannot open or find " << outputB4V1_filename + output_suffix <<" to store ouput before V1.\n";
				return EXIT_FAILURE;
			} else {
				if (!restore.empty() && !asInit) {
					streampos eof = fOutputB4V1.tellp();
					fOutputB4V1.seekp(0);
					fOutputB4V1.write((char*)&nt0, sizeof(Size));
					fOutputB4V1.seekp(eof);
				} else {
					fOutputB4V1.write((char*)&nt, sizeof(Size));
					fOutputB4V1.write((char*)&dt, sizeof(Float));
					fOutputB4V1.write((char*)&nLGN, sizeof(Size));
				}
			}
		}
	}
	cout << "output file check, done\n";

	Size iKernelSampleT0;
	PosInt kernelSampleInterval = nRetrace/nKernelSample;
	if (kernelSampleInterval%2 == 0) {
		iKernelSampleT0 = kernelSampleInterval/2;
		cout << "sample parvo in intervals of " << kernelSampleInterval << " starting with " << iKernelSampleT0 << " in units of dt\n";
	} else {
		iKernelSampleT0 = 0;
		if (kernelSampleInterval > 1) {
			cout << "make parvo sample interval (" << kernelSampleInterval << ") even in the units of dt\n";
		}
	}
	// |--*--|--*--|--*--|, e.g., nKernelSample = 3->*
	Float kernelSampleDt = kernelSampleInterval*dt;
	Float kernelSampleT0 = iKernelSampleT0*dt;
	Size kernelSampleRate = stepRate/kernelSampleInterval; 
	if (sizeof(Float) == 4) {
		printf("parvo temporal kernel retraces %lf ms, samples %u points, sample rate = %u Hz\n", tau, nKernelSample, kernelSampleRate);
	} else {
		printf("parvo temporal kernel retraces %lf ms, samples %u points, sample rate = %u Hz\n", tau, nKernelSample, kernelSampleRate);
	}

    Size mKernelSampleT0;
    PosInt kermelSampleInterval = mRetrace/mKernelSample;
    if (kermelSampleInterval%2 == 0) {
		mKernelSampleT0 = kermelSampleInterval/2;
		cout << "sample magno in intervals of " << kermelSampleInterval << " starting with " << mKernelSampleT0 << " in units of dt\n";
	} else {
		mKernelSampleT0 = 0;
		if (kermelSampleInterval > 1) {
			cout << "make magno sample interval (" << kermelSampleInterval << ") even in the units of dt\n";
		}
	}
	// |--*--|--*--|--*--|, e.g., nKernelSample = 3->*
	Float kermelSampleDt = kermelSampleInterval*dt;
	Float kermelSampleT0 = mKernelSampleT0*dt;
	Size kermelSampleRate = stepRate/kermelSampleInterval; 
	if (sizeof(Float) == 4) {
		printf("magno temporal kernel retraces %f ms, samples %u points, sample rate = %u Hz\n", mau, mKernelSample, kermelSampleRate);
	} else {
		printf("magno temporal kernel retraces %lf ms, samples %u points, sample rate = %u Hz\n", mau, mKernelSample, kermelSampleRate);
	}



	// calculate phase difference between sampling point and next-frame point  
	bool moreDt = true;
	if (stepRate < frameRate) {
		moreDt = false;
		printf("frameRate > simulation rate!!");
	}
	// norm/denorm = the normalized advancing phase at each step, 
	// denorm is the mininum number of frames such that total frame length is a multiple of dt.
	PosInt norm;
	PosInt denorm = find_denorm(frameRate, stepRate , moreDt, norm);
	PosInt *exact_norm = new PosInt[denorm]; // to be normalized by denorm to form normalized phase
	PosInt current_norm = 0;
	cout << denorm << " exact phases in [0,1]: ";
	for (PosInt i=0; i<denorm; i++) {
		if (current_norm>denorm) {
			current_norm -= denorm;
		} 
		exact_norm[i] = current_norm;
		if (i<denorm - 1) {
			printf("%u/%u, ", current_norm, denorm);
		} else {
			printf("%u/%u\n", current_norm, denorm);
		}
		current_norm += norm;
	}
	assert(current_norm == denorm);

	PosInt *exact_it = new PosInt[denorm];
	for (PosInt i=0; i<denorm; i++) {
		exact_it[i] = (i*stepRate)/frameRate; // => quotient of "i*Tframe/Tdt"
		cout << exact_it[i] << " + " << exact_norm[i] << " / " << denorm << "\n";
	}
	// the first i frames' accumulated length in steps = exact_it[i] + exact_norm[i]/denorm
	assert((stepRate*denorm) % frameRate == 0);
	Size co_product = (stepRate*denorm)/frameRate; 
	// the number of non-zero minimum steps to meet frameLength * denorm with 0 phase
	Size ntPerFrame = co_product; // tPerFrame in the units of dt/denorm

    Size parvoFrame = (nRetrace*denorm + ntPerFrame-1)/ntPerFrame + 1;
    Size magnoFrame = (mRetrace*denorm + ntPerFrame-1)/ntPerFrame + 1;
	Size maxFrame = parvoFrame > magnoFrame? parvoFrame: magnoFrame;
    bool pLongerThanM = parvoFrame > magnoFrame;
    Size dFrame = magnoFrame > parvoFrame? magnoFrame - parvoFrame: parvoFrame - magnoFrame;
    cout << "difference in number of frames between parvo and magno |" << parvoFrame << " - " << magnoFrame << "| = " << dFrame << "\n";
	if (maxFrame > 2048) {
		cout << "a single layered texture object only allows 2048 layers < " << maxFrame << "\n";
		return EXIT_FAILURE;
	}
	// max frame need to be stored in texture for temporal convolution with the LGN kernel.
	//  |---|--------|--|
	cout << "temporal kernel retrace (tau): " << tau <<" ms, frame rate: " << frameRate << "Hz needs at most " <<  maxFrame << " frames\n";
	unsigned int nChannel = 3; // L, M, S
	Size texture_nElem = nPixelPerFrame*nChannel*maxFrame;
	cout << "=== texture memory required: " << maxFrame << "x" << width << "x" << height << " = " << texture_nElem*sizeof(float)/1024.0/1024.0 << "MB\n";

	// one cudaArray per channel
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *cuArr_L;
	cudaArray *cuArr_M;
	cudaArray *cuArr_S;

	checkCudaErrors(cudaMalloc3DArray(&cuArr_L, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));
	checkCudaErrors(cudaMalloc3DArray(&cuArr_M, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));
	checkCudaErrors(cudaMalloc3DArray(&cuArr_S, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));

	usingGMem += 3*width*height*maxFrame*sizeof(Float);
	if (checkGMemUsage(usingGMem, GMemAvail)) return EXIT_FAILURE;
	cout << "Using "<< usingGMem/1024.0/1024.0 << " Mb from a total of " << deviceProps.totalGlobalMem/1024.0/1024.0 << " Mb, remaining " << (deviceProps.totalGlobalMem - usingGMem)/1024.0/1024.0 << " Mb\n";

	// set params for layerd texture memory
	init_layer(L_retinaInput);
	init_layer(M_retinaInput);
	init_layer(S_retinaInput);

	// bind texture to cudaArrays
	checkCudaErrors(cudaBindTextureToArray(L_retinaInput, cuArr_L, channelDesc));
	checkCudaErrors(cudaBindTextureToArray(M_retinaInput, cuArr_M, channelDesc));
	checkCudaErrors(cudaBindTextureToArray(S_retinaInput, cuArr_S, channelDesc));

	float* LMS = new float[nPixelPerFrame*sizeof(float)*nChannel]; // memory head

	// LMS frame intensity array of [maxFrame] frames from (t-tau) -> t on hos 
	float* __restrict__ L;
	float* __restrict__ M;
	float* __restrict__ S;

	L = LMS;
	M = L + nPixelPerFrame;
	S = M + nPixelPerFrame;

	// initialize average to normalized mean luminance
	{// initialize texture to 0
		float* tLMS;
		float* __restrict__ tL;
		float* __restrict__ tM;
		float* __restrict__ tS;

		cudaStream_t initStream[3];
		for (PosInt i = 0; i < 3; i++) {
			checkCudaErrors(cudaStreamCreate(&initStream[i]));
		}

		checkCudaErrors(cudaMalloc((void **) &tLMS, texture_nElem*sizeof(float)));
		tL = tLMS;
		tM = tL + nPixelPerFrame*maxFrame;
		tS = tM + nPixelPerFrame*maxFrame;
		Size nGrid = (nPixelPerFrame*maxFrame + blockSize-1)/blockSize;
		cudaMemsetNonzero<float><<<nGrid, blockSize, 0, initStream[0]>>> (tL, nPixelPerFrame*maxFrame, init_L);
        #ifdef CHECK
		    getLastCudaError("memset failed");
        #endif
		cudaMemsetNonzero<float><<<nGrid, blockSize, 0, initStream[1]>>> (tM, nPixelPerFrame*maxFrame, init_M);
        #ifdef CHECK
		    getLastCudaError("memset failed");
        #endif
		cudaMemsetNonzero<float><<<nGrid, blockSize, 0, initStream[2]>>> (tS, nPixelPerFrame*maxFrame, init_S);
        #ifdef CHECK
		    getLastCudaError("memset failed");
        #endif

		prep_sample(0, width, height, tL, tM, tS, cuArr_L, cuArr_M, cuArr_S, maxFrame, cudaMemcpyDeviceToDevice); // implicit synchronized
		/* DEBUG
		dim3 fb(16,16,1);
		dim3 fg(16,16,maxFrame);
		testTexture<<<fg, fb>>> (init_L, init_M, init_S);
        #ifdef CHECK
		    getLastCudaError("texture read test failed");
        #endif
		 */
		checkCudaErrors(cudaFree(tLMS));
		for (PosInt i=0; i<3; i++) {
			checkCudaErrors(cudaStreamDestroy(initStream[i]));
		}
		cout << "all pixels in texture memory (frame buffers) initialized to " << init_L << ", " << init_M << ", " << init_S << " \n";
	}

	dim3 parvoBlock(nSpatialSample1D, nSpatialSample1D, 1);
	dim3 parvoGrid(nParvo, 2, 1);

	cout << "cuda memory all set.\n";

	cudaEvent_t storeReady;
	checkCudaErrors(cudaEventCreate(&storeReady));

	// store spatial and temporal weights determine the maximums of LGN kernel convolutions
    if (nParvo > 0) {
	    cout << "store_PM(0)<<<" << parvoGrid.x  << "x" << parvoGrid.y  << "x" << parvoGrid.z << ", " << parvoBlock.x  << "x" << parvoBlock.y  << "x" << parvoBlock.z << ">>>\n";
	    store_PM<<<parvoGrid, parvoBlock>>>(
	    		*dLGN.temporal,
	    		parvo_TW_storage,
	    		nKernelSample, kernelSampleDt, kernelSampleT0,
	    		*dLGN.spatial,
	    		parvo_SW_storage,
	    		parvo_SC_storage,
                maxConvol, // not filled
	    		0, nMagno_I, nParvo_I, nLGN,
                L_x0, L_y0, R_x0, R_y0,
	    		normViewDistance, nsig, 0, flat_retina);
        #ifdef CHECK
	        getLastCudaError("store failed");
        #endif
        #ifdef SYNC
            checkCudaErrors(cudaDeviceSynchronize());
        #endif

        // for max_convol
        Float maxRatio = 0;
        Float meanRatio = 0;
        Float varRatio = 0;
        for (Size i=0; i<nLGN; i++) {
            if (coneType[i] < 3) {
                Float dx = LGN_ecc[i]*cosine(LGN_polar[i]) - LGN_ecc[nLGN+i]*cosine(LGN_polar[nLGN+i]);
                Float dy = LGN_ecc[i]*sine(LGN_polar[i])   - LGN_ecc[nLGN+i]*sine(LGN_polar[nLGN+i]);
                Float r = square_root(dx*dx + dy*dy);
                Float wSpan = nsig * LGN_rw[nLGN+i] / SQRT2;
                Float hSpan = nsig * LGN_rh[nLGN+i] / SQRT2;
                Float R = wSpan>hSpan? wSpan:hSpan;
                wSpan = nsig * LGN_rw[i] / SQRT2;
                hSpan = nsig * LGN_rh[i] / SQRT2;
                r += wSpan>hSpan? wSpan:hSpan;
                R = r>R? r: R;
                Float ratio = sqrt(R*R/wSpan/hSpan);
                meanRatio += ratio;
                varRatio += ratio * ratio;
                if (ratio > maxRatio) {
                    maxRatio = ratio;
                }
            }
        }
        meanRatio /= nParvo;
        varRatio = varRatio/nParvo - meanRatio*meanRatio;
        cout << " mean S/C size ratio = " <<  meanRatio << " +- sqrt(" << varRatio << "), must be close to zero if negative, due to roundoff error.\n";
        cout << " max S/C size ratio = " <<  maxRatio << "\n";

        int carveout = 100;
        cudaFuncSetAttribute(parvo_maxConvol, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
        Size maxSample1D = static_cast<Size>(std::round(meanRatio)) * nSpatialSample1D;
        Size maxSample = maxSample1D * maxSample1D;
	    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
        size_t maxConvol_shared = maxSample*2*sizeof(Float);
        if (maxConvol_shared > deviceProps.sharedMemPerBlock-warpSize*sizeof(Float)) {
            maxSample1D = static_cast<Size>(floor(sqrt((deviceProps.sharedMemPerBlock-warpSize*sizeof(Float))/sizeof(Float)/2/nSample))) * nSpatialSample1D;
            maxSample = maxSample1D*maxSample1D;
            cout << "shared memory avail: " << deviceProps.sharedMemPerBlock-warpSize*sizeof(Float) << " < " << maxConvol_shared << ", constrained maxConvol's sample to " << maxSample << "\n";
            maxConvol_shared = maxSample*2*sizeof(Float);
        } else {
            cout << "determine maxConvol need nSample: " << maxSample << "\n";
        }
		assert(maxConvol_shared <= deviceProps.sharedMemPerBlock - warpSize*sizeof(Float));

	    parvo_maxConvol<<<nParvo, nSample, maxConvol_shared>>>(
                *dLGN.spatial,
                parvo_TW_storage,
                dLGN.covariant,
                maxConvol,
                maxSample1D, nParvo_I, nMagno_I, nLGN, nKernelSample, kernelSampleDt, nsig);
        #ifdef CHECK
	        getLastCudaError("get_maxConvol failed");
        #endif
        #ifdef SYNC
            checkCudaErrors(cudaDeviceSynchronize());
        #endif
    }
	dim3 magnoBlock(mSpatialSample1D, mSpatialSample1D, 1);
	dim3 magnoGrid(nMagno, 1, 1);
    if (nMagno > 0) {
	    cout << "store_PM(1)<<<" << magnoGrid.x  << "x" << magnoGrid.y  << "x" << magnoGrid.z << ", " << magnoBlock.x  << "x" << magnoBlock.y  << "x" << magnoBlock.z << ">>>\n";
	    store_PM<<<magnoGrid, magnoBlock>>>(
	    		*dLGN.temporal,
	    		magno_TW_storage,
	    		mKernelSample, kermelSampleDt, kermelSampleT0,
	    		*dLGN.spatial,
	    		magno_SW_storage,
	    		magno_SC_storage,
                maxConvol, // magno part wil be filled
	    		nParvo_I, nParvo_C, nMagno_I, nLGN,
                L_x0, L_y0, R_x0, R_y0,
	    		normViewDistance, nsig, 1, flat_retina);
        #ifdef CHECK
	        getLastCudaError("store failed");
        #endif
        #ifdef SYNC
            checkCudaErrors(cudaDeviceSynchronize());
        #endif
    }
    cudaEventRecord(storeReady, 0);
    cudaEventSynchronize(storeReady);
	cout << "convol parameters stored\n";
	if (saveLGN_gallery) {// storage check output
		checkCudaErrors(cudaMemcpy(galleryOutput, gpu_LGN_gallery, gallerySize, cudaMemcpyDeviceToHost));
		//Float* r_maxConvol = (Float *) galleryOutput;
		//Float maxMaxConvol = *max_element(r_maxConvol, r_maxConvol + nLGN);
		//Float meanMaxConvol = accumulate(r_maxConvol, r_maxConvol + nLGN, 0.0)/nLGN;
		//Float minMaxConvol = *min_element(r_maxConvol, r_maxConvol + nLGN);
		//cout << "maxConvol = [" << minMaxConvol << ", " << meanMaxConvol << ", " << maxMaxConvol << "]\n";

		fLGN_gallery.write((char*)galleryOutput, gallerySize);
		fLGN_gallery.close();
	}
	// calc LGN firing rate at the end of current dt
	PosInt currentFrame = 0; // current frame number from stimulus
	//PosInt oldFrame = currentFrame;
	// framePhase is fractionalized by denorm to fit frame duration with integer units.
	PosInt iFramePhaseHead = 0;
	PosInt parvoRemain = (nRetrace*denorm) % ntPerFrame;
	PosInt parvoComp = (ntPerFrame - parvoRemain) % ntPerFrame;
	PosInt iFramePhaseTail = parvoComp;

	PosInt mFramePhaseHead = 0;
	PosInt magnoRemain = (mRetrace*denorm) % ntPerFrame;
	PosInt magnoComp = (ntPerFrame - magnoRemain) % ntPerFrame;
	PosInt mFramePhaseTail = magnoComp;

	PosInt iFrameHead = currentFrame % maxFrame;
	//PosInt iPhase = 0;
	auto getTail = [](PosInt comp, PosInt phaseTail, PosInt phaseHead, PosInt head) {
		// point frametail to the tail of the LGN temporal convolution at t-tau
		PosInt tail;
		if (phaseTail < comp) {
			tail = head + 2;
			assert(phaseHead > phaseTail);
		} else {
			tail = head + 1;
			assert(phaseHead <= phaseTail);
		}
		return tail;
	};

	cudaEvent_t v_gFF_Ready, spReady, frameReady, spHostReady, LGN_ready, learnFF_event, magnoDone;
	checkCudaErrors(cudaEventCreate(&v_gFF_Ready));
	checkCudaErrors(cudaEventCreate(&spReady));
	checkCudaErrors(cudaEventCreate(&frameReady));
	checkCudaErrors(cudaEventCreate(&spHostReady));
	checkCudaErrors(cudaEventCreate(&LGN_ready));
	checkCudaErrors(cudaEventCreate(&magnoDone));
	checkCudaErrors(cudaEventCreate(&learnFF_event));

	cudaEvent_t *gReady1 = new cudaEvent_t[nChunk];
	cudaEvent_t *gReady2 = new cudaEvent_t[nChunk];
	cudaEvent_t *matReady = new cudaEvent_t[matConcurrency];
	for (PosInt i = 0; i < nChunk; i++) {
		checkCudaErrors(cudaEventCreate(&gReady1[i]));
		checkCudaErrors(cudaEventCreate(&gReady2[i]));
        if (i < matConcurrency) checkCudaErrors(cudaEventCreate(&matReady[i]));
	}
	cudaEvent_t *eTmp = new cudaEvent_t[nChunk];
	for (PosInt i = 0; i < nChunk; i++) {
		checkCudaErrors(cudaEventCreate(&eTmp[i]));
	}

	cudaStream_t mainStream, LGN_stream, magnoStream;
	checkCudaErrors(cudaStreamCreate(&mainStream));
	checkCudaErrors(cudaStreamCreate(&magnoStream));
	checkCudaErrors(cudaStreamCreate(&LGN_stream));

	cudaStream_t *stream = new cudaStream_t[nChunk];
	for (PosInt i = 0; i < nChunk; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}

	cudaStream_t ostream[3];
	for (PosInt i = 0; i < 3; i++) {
		checkCudaErrors(cudaStreamCreate(&ostream[i]));
	}

    //Size ont; 
    //if (framePhyV1output) {
    //    ont = nt/ot; // "not" cannot be a variable name!!
    //}
    bool spiked;
    bool farSpiked = false;
	PosInt currentTimeSlot;
	Float odt = ot*dt/1000.0;// get interval in sec
	if (!restore.empty() && !asInit) {
		cout << "simulation resumes from t = " << it0*dt << "...\n";
	} else {
		cout << "simulation starts: \n";
	}
    int varSlot = 0;
    InputActivation typeStatus;
    PosInt iStatus = 0;
	PosIntL oldTimeStamp = 0;
	PosInt oldFrameHead;
	//***************************
	// var sizes
	// 	LGN
	size_t LGN_convolSize = 2*nLGN*sizeof(Float) + nLGN*sizeof(curandStateMRG32k3a);
	size_t LGN_snapShotSize = LGN_convolSize;
	if (learning) {
		LGN_snapShotSize += LGN_learnPreSize;
		if (!learnData_FF) {
			LGN_snapShotSize += sLGN_size + LGN_learnPostSize;
		}
	}
	// 	V1
	size_t r_vghSize = ghSize + ffSize + vSize;
	if (iModel == 1) {
		r_vghSize += wSize;
	}
	size_t V1_snapShotSize = 2*nV1*sizeof(curandStateMRG32k3a) + nV1*sizeof(Float);
	// total
	size_t deviceMemorySize = LGN_snapShotSize + V1_snapShotSize;
	PosIntL timeStamp;
	string snapShot_fn;

	PosInt frameCycle = 0;
	if (!restore.empty()) {
		fSnapshot.open(restore, fstream::in | fstream::binary);
		if (!fSnapshot) {
			cout << "cannot restore from " << restore << "\n";
			return EXIT_FAILURE;
		} else {
			PosInt discard; 
			fSnapshot.read(reinterpret_cast<char*>(&discard), sizeof(PosInt));
			// indices
			if (asInit) {
				for (PosInt i=0; i<9; i++) {
					fSnapshot.read(reinterpret_cast<char*>(&discard), sizeof(PosInt));
				}
			} else {
				fSnapshot.read(reinterpret_cast<char*>(&frameCycle), sizeof(PosInt));
				fSnapshot.read(reinterpret_cast<char*>(&iFrameHead), sizeof(PosInt));
				fSnapshot.read(reinterpret_cast<char*>(&oldFrameHead), sizeof(PosInt));
				fSnapshot.read(reinterpret_cast<char*>(&currentFrame), sizeof(PosInt));
				fSnapshot.read(reinterpret_cast<char*>(&iFramePhaseHead), sizeof(PosInt));
				fSnapshot.read(reinterpret_cast<char*>(&iFramePhaseTail), sizeof(PosInt));
				fSnapshot.read(reinterpret_cast<char*>(&mFramePhaseHead), sizeof(PosInt));
				fSnapshot.read(reinterpret_cast<char*>(&mFramePhaseTail), sizeof(PosInt));
				fSnapshot.read(reinterpret_cast<char*>(&iStatus), sizeof(PosInt));
				cout << "it0 = " << it0 << "\n";
				cout << "frameCycle = " << frameCycle << "\n";
				cout << "iFrameHead = " << iFrameHead << "\n";
				cout << "oldFrameHead = " << oldFrameHead << "\n";
				cout << "currentFrame = " << currentFrame << "\n";
				cout << "iFramePhaseHead = " << iFramePhaseHead << "\n";
				cout << "iFramePhaseTail = " << iFramePhaseTail << "\n";
				cout << "mFramePhaseHead = " << mFramePhaseHead << "\n";
				cout << "mFramePhaseTail = " << mFramePhaseTail << "\n";
				cout << "iStatus = " << iStatus << "\n";
			}
			fSnapshot.read(reinterpret_cast<char*>(&currentTimeSlot), sizeof(PosInt));
			cout << "move currentTimeSlot to " << currentTimeSlot << "\n";
			
			// const sizes
			int r_iModel, r_learning;
			fSnapshot.read(reinterpret_cast<char*>(&r_iModel), sizeof(int));
			fSnapshot.read(reinterpret_cast<char*>(&r_learning), sizeof(int));
			Size r_nV1, r_nLGN, r_ngTypeFF, r_ngTypeE, r_ngTypeI, r_nType;
			fSnapshot.read(reinterpret_cast<char*>(&r_nV1), sizeof(Size));
			fSnapshot.read(reinterpret_cast<char*>(&r_nLGN), sizeof(Size));
			fSnapshot.read(reinterpret_cast<char*>(&r_ngTypeFF), sizeof(Size));
			fSnapshot.read(reinterpret_cast<char*>(&r_ngTypeE), sizeof(Size));
			fSnapshot.read(reinterpret_cast<char*>(&r_ngTypeI), sizeof(Size));
			fSnapshot.read(reinterpret_cast<char*>(&r_nType), sizeof(Size));
			Size r_trainDepth;
			fSnapshot.read(reinterpret_cast<char*>(&r_trainDepth), sizeof(Size));
			if (r_learning != learning || r_iModel != iModel || r_nV1 != nV1 || r_nLGN != nLGN || r_ngTypeFF != ngTypeFF || r_ngTypeE != ngTypeE || r_ngTypeI != ngTypeI) {
				cout << "patchV1 config is incompatible with the restore file config\n";
				return EXIT_FAILURE;
			}
			if (trainDepth != r_trainDepth) {
				cout << "trainDepth is inconsistent, due to change(s) in delayMat, maxDistance\n";
			}
			
			char* _restore = new char[deviceMemorySize];

			// read into host vars
			// 	LGN
			Float* r_leftTimeRate = (Float*) _restore; 
			Float* r_lastNegLogRand = r_leftTimeRate + nLGN;
			curandStateMRG32k3a* r_randState = (curandStateMRG32k3a*) (r_lastNegLogRand + nLGN);
			fSnapshot.read(reinterpret_cast<char*>(r_leftTimeRate), LGN_convolSize);
			checkCudaErrors(cudaMemcpy(leftTimeRate, r_leftTimeRate, LGN_convolSize, cudaMemcpyHostToDevice));
			Float* r_lVarFFpre;
			if (learning) {
				if (!learnData_FF) {
					LGN_V1_s = (Float*) (r_randState + nLGN); 
					lVarFFpost = LGN_V1_s + sLGN_size/sizeof(Float);
					r_lVarFFpre = lVarFFpost + learnVarFFsize;
				} else {
					r_lVarFFpre = (Float*) (r_randState + nLGN);
				}

				fSnapshot.read(reinterpret_cast<char*>(LGN_V1_s), sLGN_size);
				checkCudaErrors(cudaMemcpy(sLGN, LGN_V1_s, sLGN_size, cudaMemcpyHostToDevice));
				fSnapshot.read(reinterpret_cast<char*>(lVarFFpost), LGN_learnPostSize);
				checkCudaErrors(cudaMemcpy(learnVar, lVarFFpost, LGN_learnPostSize, cudaMemcpyHostToDevice));

				fSnapshot.read(reinterpret_cast<char*>(r_lVarFFpre), LGN_learnPreSize);
				checkCudaErrors(cudaMemcpy(lVarFFpre, r_lVarFFpre, LGN_learnPreSize, cudaMemcpyHostToDevice));
			}

			// 	V1
			curandStateMRG32k3a* r_rGenCond;
			if (learning) {
				r_rGenCond = (curandStateMRG32k3a *) (r_lVarFFpre + LGN_learnPreSize/sizeof(Float));
			} else {
				r_rGenCond = r_randState + nLGN;
			}
			fSnapshot.read(reinterpret_cast<char*>(r_rGenCond), nV1*sizeof(curandStateMRG32k3a));
			checkCudaErrors(cudaMemcpy(rGenCond, r_rGenCond, nV1*sizeof(curandStateMRG32k3a), cudaMemcpyHostToDevice));

			curandStateMRG32k3a* r_rNoisy =  r_rGenCond + nV1;
			fSnapshot.read(reinterpret_cast<char*>(r_rNoisy), nV1*sizeof(curandStateMRG32k3a));
			checkCudaErrors(cudaMemcpy(rNoisy, r_rNoisy, nV1*sizeof(curandStateMRG32k3a), cudaMemcpyHostToDevice));

			Float* r_tBack = (Float*) (r_rNoisy + nV1);
			fSnapshot.read(reinterpret_cast<char*>(r_tBack), nV1*sizeof(Float));
			checkCudaErrors(cudaMemcpy(tBack, r_tBack, nV1*sizeof(Float), cudaMemcpyHostToDevice));
			delete []_restore;
			// already pinned on host
			if (r_trainDepth == trainDepth) {
				fSnapshot.read(reinterpret_cast<char*>(spikeTrain), trainSize*sizeof(Float));
				checkCudaErrors(cudaMemcpy(d_spikeTrain, spikeTrain, trainSize*sizeof(Float), cudaMemcpyHostToDevice));
			}
			if (r_trainDepth < trainDepth) {
				cout << "the spikeTrain from snapshot will be padded with zeros to fill the new size \n";
				// r_trainDepth  = L + R
				//     L    c                    R
				// |--------|----------------|--------|
				Size L_trainSize = currentTimeSlot*nV1;
				if (L_trainSize > 0) {
					fSnapshot.read(reinterpret_cast<char*>(spikeTrain), L_trainSize*sizeof(Float));
				}

				Size s_trainSize = (trainDepth - r_trainDepth)*nV1;
				memset((char*)(spikeTrain+L_trainSize), 0.0, s_trainSize*sizeof(Float));

				Size R_trainSize = trainSize - s_trainSize-L_trainSize;
				if (R_trainSize > 0) {
					fSnapshot.read(reinterpret_cast<char*>(spikeTrain+L_trainSize+s_trainSize), R_trainSize*sizeof(Float));
				}

				checkCudaErrors(cudaMemcpy(d_spikeTrain, spikeTrain, trainSize*sizeof(Float), cudaMemcpyHostToDevice));
				assert(L_trainSize + s_trainSize + R_trainSize == trainDepth*nV1);
				assert(L_trainSize + R_trainSize == r_trainDepth*nV1);
			}
			if (r_trainDepth > trainDepth) {
				cout << "old spikeTrain will be cut off at currentTimeSlot-trainDepth to fit into the new size\n";
				if (currentTimeSlot >= trainDepth) {
					// L + trainDepth + R = r_trainDepth
					//  discard                          discard
					//   L            trainDepth       c   R
					// |--------|----------------------|--------|
					Float* discard;
					Size L_trainDepth = currentTimeSlot-trainDepth;
					size_t L_trainSize = L_trainDepth*nV1*sizeof(Float);
					if (L_trainSize > 0) {
						discard = new Float[L_trainSize];
						fSnapshot.read(reinterpret_cast<char*>(discard), L_trainSize);
						delete []discard;
					}

					fSnapshot.read(reinterpret_cast<char*>(spikeTrain), trainSize*sizeof(Float));
					checkCudaErrors(cudaMemcpy(d_spikeTrain, spikeTrain, trainSize*sizeof(Float), cudaMemcpyHostToDevice));

					Size R_trainSize = (r_trainDepth-L_trainDepth)*nV1-trainSize;
					if (R_trainSize > 0) {
						discard = new Float[R_trainSize];
						fSnapshot.read(reinterpret_cast<char*>(discard), R_trainSize*sizeof(Float));
						delete [] discard;
					}
					currentTimeSlot = trainDepth - 1;
				} else {
					// L + s_trainDepth + R = r_trainDepth
					// L + R = trainDepth
					//                  disccard
					//   L      c    s_trainDepth           R
					// |--------|----------------------|--------|

					Size L_trainDepth = currentTimeSlot;
					size_t L_trainSize = L_trainDepth*nV1*sizeof(Float);
					if (L_trainDepth > 0) {
						fSnapshot.read(reinterpret_cast<char*>(spikeTrain), L_trainSize);
					}

					Size s_trainSize = (r_trainDepth - trainDepth)*nV1;
					Float* discard = new Float[s_trainSize];
					fSnapshot.read(reinterpret_cast<char*>(discard), s_trainSize*sizeof(Float));
					delete []discard;

					size_t R_trainSize = ((r_trainDepth - L_trainDepth)*nV1 - s_trainSize)*sizeof(Float);
					fSnapshot.read(reinterpret_cast<char*>(spikeTrain+L_trainDepth*nV1), R_trainSize);

					assert(trainSize*sizeof(Float) == L_trainSize + R_trainSize);
					checkCudaErrors(cudaMemcpy(d_spikeTrain, spikeTrain, trainSize*sizeof(Float), cudaMemcpyHostToDevice));
				}
			}
			for (PosInt i = 0; i<trainDepth; i++) {
				for (PosInt j=0; j<nV1; j++) {
					assert(!std::isnan(spikeTrain[i*nV1 + j]));
					if (spikeTrain[i*nV1 + j] < 1) {
						assert(spikeTrain[i*nV1 + j] == 0);
					}
				}
			}
			if (iModel == 0) {
				fSnapshot.read(reinterpret_cast<char*>(v), r_vghSize);
				checkCudaErrors(cudaMemcpy(d_v, v, r_vghSize, cudaMemcpyHostToDevice));
				for (PosInt j=0; j<r_vghSize/sizeof(Float); j++) {
					assert(!std::isnan(v[j]));
				}
			} 
			if (iModel == 1) {
				fSnapshot.read(reinterpret_cast<char*>(w), r_vghSize);
				checkCudaErrors(cudaMemcpy(d_w, w, r_vghSize, cudaMemcpyHostToDevice));
				for (PosInt j=0; j<r_vghSize/sizeof(Float); j++) {
					assert(!std::isnan(w[j]));
				}
			}
			fSnapshot.close();
			cout << "succesfully restored simulation snapshot from " << restore << "\n";
		}
	} else {
    	if (has_sp0) {
    		cout << "presend spikes:\n";
			currentTimeSlot = trainDepth - 1;
    	    if (matConcurrency < nChunk) { // initially, staging all the chunks of the first matConcurrency
    	        //size_t initial_size;
    	        //if (matConcurrency > iSizeSplit) initial_size = iSizeSplit * maxChunkSize + (matConcurrency-iSizeSplit)*remainChunkSize;
    	        //else initial_size = matConcurrency*maxChunkSize;
    	        //auto start = chrono::high_resolution_clock::now();
    	        //memcpy((void*)(p_conDelayMat), (void*) conDelayMat[0], 2*initial_size*nearBlockSize*sizeof(Float));
    	        memcpy((void*)(p_conDelayMat), (void*) conDelayMat[0], ccChunkMatSize*sizeof(float));
    	        //auto end = chrono::high_resolution_clock::now();
    	        //double time_taken = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
    	        //cout << "memcpy of " << 2*initial_size*nearBlockSize*sizeof(Float)/1024.0/1024.0 << " Mb took " << time_taken/1e6 << " ms\n";
    	    } // else mats are already allocated to pinned memory
    	    
    	    Size chunkSize = maxChunkSize;
    	    PosInt block_offset = 0;
    	    size_t p_offset = 0;
    	    size_t p_total = 0;
    	    for (PosInt i = 0; i < nChunk; i++) {
    	        if (i >= iSizeSplit) chunkSize = remainChunkSize;
    	        size_t mChunkSize = chunkSize * nearBlockSize;
    	        if (i%matConcurrency == 0) {
    	            p_total += p_offset;
    	            p_offset = 0;
    	        }
    	        if (matConcurrency < nChunk) {
    	            // staging each chunk at a time
    	            if (i>=matConcurrency) {
    	                //cout << "#" << i << ":\n";
    	                //if (cudaSuccess == cudaEventQuery(eTmp[i-1])) cout << "b4memcpy last recal_G already finished\n";
    	                //else cout << "b4memcpy last recal_G still running\n";
    	    
    	                cudaEventSynchronize(matReady[i%matConcurrency]);
    	                //auto start = chrono::high_resolution_clock::now();
    	                memcpy((void*)(p_conDelayMat + p_offset), (void*) conDelayMat[i], 2*mChunkSize*sizeof(float));
    	                //auto end = chrono::high_resolution_clock::now();
    	                //double time_taken = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
    	                //cout << "memcpy of " << 2*mChunkSize*sizeof(Float)/1024.0/1024.0 << " Mb took " << time_taken/1e6 << " ms\n";
    	    
    	                //if (cudaSuccess == cudaEventQuery(eTmp[i-1])) cout << "after memcpy last recal_G already finished\n";
    	                //else cout << "after memcpy last recal_G still running\n";
    	            }
    	        }
    	        if (matConcurrency < nChunk || learnData_V1) {
    	        #ifdef CHECK
    	            checkCudaErrors(cudaMemcpyAsync(d_conDelayMat[i%matConcurrency], p_conDelayMat + p_offset, 2*mChunkSize*sizeof(float), cudaMemcpyHostToDevice, stream[i%matConcurrency]));
    	        #else
    	            cudaMemcpyAsync(d_conDelayMat[i%matConcurrency], p_conDelayMat + p_offset, 2*mChunkSize*sizeof(float), cudaMemcpyHostToDevice, stream[i%matConcurrency]);
    	        #endif
    	        #ifdef SYNC
    	            checkCudaErrors(cudaDeviceSynchronize());
    	        #endif
    	            cudaEventRecord(matReady[i%matConcurrency], stream[i%matConcurrency]);
    	        }
    	        /*
    	            if (i>0) {
    	                if (cudaSuccess == cudaEventQuery(eTmp[i-1])) cout << "after cudamemcpy last recal_G already finished\n";
    	                else cout << "after cudamemcpy last recal_G still running\n";
    	            }
    	        */
    	        float* d_conMat = d_conDelayMat[i%matConcurrency];
    	        float* d_delayMat = d_conMat + mChunkSize;
				if (noDelay) {
    	        	recal_G_mat<<< chunkSize, blockSize, 0, stream[i%matConcurrency]>>> (
    	        	        d_ogh,
							d_ogh + ncond,
    	        	        d_conMat, d_delayMat,
    	        	        d_nNeighborBlock+block_offset, d_neighborBlockId + block_offset*nearNeighborBlock,
    	        	        d_gE[i], d_gI[i], d_hE[i], d_hI[i],
    	        	        vAvgE, vLTP_E, vLTD_E, vTripE, vSTDP_QE, vSTDP_QI, d_pE, d_pI, typeAcc,
    	        	        rGenCond, d_synFail, d_synPerCon,
    	        	        dt, condE, condI, ngTypeE, ngTypeI,
    	        	        0, trainDepth,
    	        	        nearNeighborBlock, nE, nI, nV1, speedOfThought, learning, block_offset, nType, lE, lQ, i);
				else {
    	        	recal_G_mat<<< chunkSize, blockSize, 0, stream[i%matConcurrency]>>> (
    	        	        d_spikeTrain,
    	        	        d_conMat, d_delayMat,
    	        	        d_nNeighborBlock+block_offset, d_neighborBlockId + block_offset*nearNeighborBlock,
    	        	        d_gE[i], d_gI[i], d_hE[i], d_hI[i],
    	        	        vAvgE, vLTP_E, vLTD_E, vTripE, vSTDP_QE, vSTDP_QI, d_pE, d_pI, typeAcc,
    	        	        rGenCond, d_synFail, d_synPerCon,
    	        	        dt, condE, condI, ngTypeE, ngTypeI,
    	        	        0, trainDepth,
    	        	        nearNeighborBlock, nE, nI, nV1, speedOfThought, learning, block_offset, nType, lE, lQ, i);
				}
    	    
    	        //cudaEventRecord(eTmp[i], stream[i]);
    	    #ifdef CHECK
    	        getLastCudaError("recal_G failed");
    	    #endif
    	    #ifdef SYNC
    	        checkCudaErrors(cudaDeviceSynchronize());
    	    #endif
    	        block_offset += chunkSize;
    	        p_offset += 2*mChunkSize;
    	    }
    	    assert(block_offset == nblock);
    	    assert(p_total + p_offset == matSize*2);
    	    for (PosInt i = 0; i < matConcurrency; i++) {
    	        cudaEventRecord(gReady1[i], stream[i]);
    	    }
    	    
    	    Size nsp = 0;
    	    spiked = false;
    	    for (PosInt i=0; i<nV1; i++) {
    	        Float sp = spikeTrain[i];
    	        if (sp > 0) {
					assert(sp >= 1.0);
    	            spiked = true;
    	            nsp += static_cast<Size>(flooring(sp));
    	            //break;
    	        }
    	    }
    	    spiked = nsp>0;
    	    if (spiked) cout << "there's " << nsp << " spiking events before simulation start\n";
    	    else cout << "no near-neighbor spiking events in the time step\n";
    	    
    	    if (nFar) {
    	    	farSpiked = fill_fSpikeTrain(fSpikeTrain,  spikeTrain + nV1*currentTimeSlot, fCurrenSlot, vecID, nVec, nV1);
				if (farSpiked) {
    	        	for (PosInt i = 0; i < nChunk; i++) {
    	        	    if (i >= iSizeSplit) chunkSize = remainChunkSize;
    	        	    size_t gChunkSize = chunkSize*blockSize*(ngTypeE+ngTypeI)*sizeof(Float);
    	        	    size_t ghChunkSize = gChunkSize*2;
    	        	    // cpu accumulate conductances from far neighbors
    	        	    recal_G_vec(
    	        	            fSpikeTrain, fTrainDepth, fCurrenSlot,
    	        	            nVec, vecID, conVec, delayVec,
    	        	            gE[i], gI[i], hE[i], hI[i], &(pE[0]), &(pI[0]), &(typeAccCount[0]),
    	        	            h_rGenCond, &(synFail[0]), &(synPerCon[0]),
    	        	            dt, condE, condI, ngTypeE, ngTypeI,
    	        	            block_offset, nType,
    	        	            nE, nV1, speedOfThought, chunkSize);
    	        	    // g and h
    	        	#ifdef CHECK
    	        	    checkCudaErrors(cudaMemcpyAsync(d_gEt[i], gE[i], ghChunkSize, cudaMemcpyHostToDevice, stream[i])); // size in maxChunk
    	        	#else
    	        	    cudaMemcpyAsync(d_gEt[i], gE[i], ghChunkSize, cudaMemcpyHostToDevice, stream[i]); // size in maxChunk
    	        	#endif
    	        	    if (i >= matConcurrency) { // otherwise automatically queued in stream
    	        	        cudaStreamWaitEvent(stream[i], gReady1[i%matConcurrency], 0);
    	        	    }
    	        	    sum_G<<<chunkSize, blockSize, 0, stream[i]>>> (d_nVec + block_offset*blockSize, d_gEt[i], d_gE[i], d_gIt[i], d_gI[i], d_hEt[i], d_hE[i], d_hIt[i], d_hI[i], ngTypeE, ngTypeI);
    	        	    if (i < nChunk-1) {
    	        	        block_offset += chunkSize;
    	        	    }
    	        	}
				}
    	    }
    	    checkCudaErrors(cudaDeviceSynchronize());
			currentTimeSlot ++;
			currentTimeSlot = currentTimeSlot%trainDepth;
    	} else {
			currentTimeSlot = 0;
		}
	}
	if (iFrameHead > 0) { // if restore from file, forward to iFrameHead
		PosInt iSample = 0;
		for (PosInt i=0; i<iFrameHead + frameCycle*maxFrame; i++) {
			if (fStimulus) {
				fStimulus.read(reinterpret_cast<char*>(LMS), nChannel*nPixelPerFrame*sizeof(float));
				streampos current_fpos = fStimulus.tellg();
				if (current_fpos == eofStimulus) { // if at the end of input file loop back to the beginning of frame data
					fStimulus.seekg(sofStimulus);
                    if (print_log) {
					    cout << "next frame loops\n";
                    }
				}
			} else {
				cout << "stimulus format corrupted\n";
				return EXIT_FAILURE;
			}
			//cp to texture mem in device
			prep_sample(iSample, width, height, L, M, S, cuArr_L, cuArr_M, cuArr_S, 1, cudaMemcpyHostToDevice);
			iSample = (iSample+1) % maxFrame;
		}
		cout << "pushed input frame to last snapshot\n";
	}
    //***************************
	size_t pSharedSize = sizeof(Float)*2*nSample;
	if (pSharedSize < sizeof(Float)*2*parvoFrame) pSharedSize = sizeof(Float)*2*parvoFrame;
	for (unsigned int it = 0; it < nt; it++) {
		farSpiked = false;
		Float t = it*dt;
		// next frame comes between (t, t+dt), read and store frame to texture memory
		if ((it+it0+1)*denorm >= currentFrame*ntPerFrame) {
			// back insert frame into texture memory
			// TODO: realtime video stimulus control
			if (fStimulus) {
				fStimulus.read(reinterpret_cast<char*>(LMS), nChannel*nPixelPerFrame*sizeof(float));
				streampos current_fpos = fStimulus.tellg();
				if (current_fpos == eofStimulus) { // if at the end of input file loop back to the beginning of frame data
					fStimulus.seekg(sofStimulus);
                    if (print_log) {
					    cout << "next frame loops\n";
                    }
				}
			} else {
				cout << "stimulus format corrupted\n";
				return EXIT_FAILURE;
			}
			//cp to texture mem in device
			prep_sample(iFrameHead, width, height, L, M, S, cuArr_L, cuArr_M, cuArr_S, 1, cudaMemcpyHostToDevice);
            #ifdef SYNC
                checkCudaErrors(cudaDeviceSynchronize());
            #endif

			//cout << "eff t = " << (it+it0+1)*denorm << " >= " << currentFrame << " * " << ntPerFrame << " = " << currentFrame * ntPerFrame << "\n";
			//iPhase = (iPhase + 1) % denorm;
			currentFrame++;
			oldFrameHead = iFrameHead;
			if (iFrameHead+1 == maxFrame) {
				frameCycle++;
			}
			iFrameHead = (iFrameHead+1) % maxFrame;

            if (print_log) {
				if (sizeof(Float) == 4) {
			    	printf("\rsimulating@t = %lf -> %lf, frame %d#%d-%d, %.1f%%\n", t, t+dt, currentFrame/nFrame, currentFrame%nFrame, nFrame, 100*static_cast<float>(it+1)/nt);
				} else {
			    	printf("\rsimulating@t = %lf -> %lf, frame %d#%d-%d, %.1f%%\n", t, t+dt, currentFrame/nFrame, currentFrame%nFrame, nFrame, 100*static_cast<float>(it+1)/nt);
				}
            }
			//cout << "it = " << it << ", iFrameHead = " << iFrameHead << "\n";
		}
		// update frame for head and tail for convolution at t=(it + 1)*dt
		iFramePhaseTail = (iFramePhaseTail + denorm) % ntPerFrame;
		iFramePhaseHead = (iFramePhaseHead + denorm) % ntPerFrame;
		mFramePhaseTail = (mFramePhaseTail + denorm) % ntPerFrame;
		mFramePhaseHead = (mFramePhaseHead + denorm) % ntPerFrame;
		// point frametail to the tail of the LGN temporal convolution at t-tau
		PosInt iFrameTail = getTail(parvoComp, iFramePhaseTail, iFramePhaseHead, oldFrameHead + (1-pLongerThanM)*dFrame);
		PosInt mFrameTail = getTail(magnoComp, mFramePhaseTail, mFramePhaseHead, oldFrameHead + pLongerThanM*dFrame);
		//cout << "it = " << it << ", head at " << oldFrameHead << "\n";

		/* if it < nRetrace, padded zero-valued frames for t<0
		   -->|        |<-- framePhase
		   |--|------frame------|
		   |-----|-----|-----|-----|-----|
		   jt-2, jt-1, jt ...nRetrace... it
		 */// perform kernel convolution with built-in texture interpolation
        if (nParvo > 0) {
			if (it > 0) {
				cudaProfilerStart();
			}
		    parvoGrid.x = nParvo;
		    parvoGrid.y = 1;
		    LGN_convol_parvo<<<parvoGrid, parvoBlock, pSharedSize, mainStream>>>(
		    		luminance,
		    		parvo_SW_storage, parvo_SC_storage, parvo_TW_storage,
		    		currentConvol, contrast,
		    		dLGN.coneType, *dLGN.spatial,
		    		nParvo_I, nMagno_I, nLGN,
		    		normViewDistance,
		    		iFrameTail, maxFrame, ntPerFrame, iFramePhaseTail,
		    		Itau,
		    		iKernelSampleT0, kernelSampleInterval, nKernelSample,
		    		dt, denorm, saveOutputB4V1);
            #ifdef CHECK
		        getLastCudaError("LGN_convol_parvo failed");
            #endif
            #ifdef SYNC
                checkCudaErrors(cudaDeviceSynchronize());
            #endif
			if (it > 0) {
				cudaProfilerStop();
			}
        }
        if (nMagno > 0) {
            LGN_convol_magno<<<magnoGrid, magnoBlock, sizeof(Float)*mSample, magnoStream>>>(
                    luminance,
		    		magno_SW_storage, magno_SC_storage, magno_TW_storage,
		    		currentConvol, contrast,
		    		dLGN.coneType, *dLGN.spatial,
		    		nParvo_I, nMagno_I, nParvo_C,
		    		normViewDistance,
		    		mFrameTail, maxFrame, ntPerFrame, mFramePhaseTail,
		    		Itau,
		    		mKernelSampleT0, kermelSampleInterval, mKernelSample,
		    		dt, denorm, saveOutputB4V1);
            #ifdef CHECK
		        getLastCudaError("LGN_convol_magno failed");
            #endif
            #ifdef SYNC
                checkCudaErrors(cudaDeviceSynchronize());
            #endif
		    cudaEventRecord(magnoDone, magnoStream);
        }

		if (it > 0) { // seeking for overlap of data output with LGN input
            if (rawData) {
			    fRawData.write((char*) (spikeTrain + nV1*currentTimeSlot), nV1*sizeof(Float));
				// debug
					for (PosInt i = 0; i<trainDepth; i++) {
						for (PosInt j=0; j<nV1; j++) {
							if (spikeTrain[i*nV1 + j] < 1) {
								if (it > 1) {
									assert(spikeTrain[i*nV1 + j] == 0);
								} else {
									if (spikeTrain[i*nV1 + j] != 0) {
										cout << i << "-" << j << ": " << spikeTrain[i*nV1 + j] << "\n";
									}
								}
							}
						}
						//if (i==0) {
						//	if (spikeTrain[i*nV1 + 937] > 0) {
						//		cout << " spiked at" << spikeTrain[i*nV1 + 937] << "\n";
						//	}
						//}
					}
				//
				if (it%snapshotInterval != 0) { // else already synchronized
			    	cudaEventSynchronize(v_gFF_Ready);
				}
				if (iModel == 0) {
			    	fRawData.write((char*) depC, nV1*sizeof(Float)*(2+ngTypeFF*(1+hWrite)));
				}
				if (iModel == 1) {
			    	fRawData.write((char*) depC, nV1*sizeof(Float)*(3+ngTypeFF*(1+hWrite)));
				}
            }
            if (learnData_FF) {
                if (!rawData) {
			        fLearnData_FF.write((char*) (spikeTrain + nV1*currentTimeSlot), nV1*sizeof(Float));
					if (it%snapshotInterval != 0) { // else already synchronized
			        	cudaEventSynchronize(v_gFF_Ready);
					}
			        fLearnData_FF.write((char*) gFF, nV1*sizeof(Float));
                }
				if (it%snapshotInterval != 0) { // else already synchronized
		        	cudaEventSynchronize(learnFF_event);
				}
                /* DEBUG
                for (PosInt i=0; i<nV1; i++) {
                    if (i%blockSize < nE) {
                        PosInt j = (i/blockSize)*nE + i%blockSize;
                        if (spikeTrain[nV1*currentTimeSlot + i] > 0) {
                            cout << "lAvgE written to " << i << " eid: " << j << " equals " << lVarFFpost[learnVarFFsize0 + j*2]  << "\n";
                        }
                    }
                    Float sInfo = spikeTrain[nV1*currentTimeSlot + i];
                    if (sInfo > 0) {
                        Size nsp = flooring(sInfo);
                        cout << i << " stored " << nsp << " spike(s) at " << sInfo - nsp << "\n";
                        assert(sInfo == 0 || (sInfo >= 1 && sInfo < 2));
                    }
                }*/
			    fLearnData_FF.write((char*) LGN_V1_s, sLGN_size);
			    fLearnData_FF.write((char*) lVarFFpost, learnVarFFsize*sizeof(Float));
                f_sLGN.write((char*) LGN_V1_s, sLGN_size);
            }
			currentTimeSlot++;
            currentTimeSlot = currentTimeSlot%trainDepth;
			//cout << "-currentTimeSlot = " << currentTimeSlot << "\n";
		}
        if (nMagno > 0) {
            cudaStreamWaitEvent(mainStream, magnoDone, 0);
        }

		// generate LGN fr with logistic function
        if (LGN_switch) {
            if (it + it0 == LGN_switchIt[iStatus]) {
                typeStatus.assign(&(LGN_status[nInputType*iStatus]));
                cout << "status changed at tstep: " << LGN_switchIt[iStatus] << "\n";
                for (PosInt j=0; j<nInputType; j++) {
                    cout << typeStatus.actPercent[j];
                    if (j<nInputType-1) cout << ", ";
                    else  cout << "\n";
                }
                iStatus++;
            }
        }
		if (it > 0) {
			cudaProfilerStart();
		}
		LGN_nonlinear<<<nLGN_block, nLGN_thread, 0, mainStream>>>(nLGN, *dLGN.logistic, maxConvol, currentConvol, convolRatio, d_LGN_fr, d_LGN_sInfo, d_sx, d_sy, leftTimeRate, lastNegLogRand, randState, dLGN_type, typeStatus, lVarFFpre, varSlot, lFF_E_pre, lFF_I_pre, nLearnTypeFF, dt, learning, learnData_FF, LGN_switch, getLGN_sp);
		if (it > 0) {
			cudaProfilerStop();
		}
        #ifdef CHECK
		    getLastCudaError("LGN_nonlinear failed");
        #endif
        #ifdef SYNC
            checkCudaErrors(cudaDeviceSynchronize());
        #endif
		if (it > 0) { // seeking for overlap of data output with LGN input
			if (it%snapshotInterval != 0) { // else already synchronized
            	if (nFar) { 
            	    for (PosInt i=0; i<nChunk; i++) {
	        	        cudaEventSynchronize(gReady2[i]);
            	    }
            	} else {
	        	    cudaEventSynchronize(gReady2[0]);
            	}
			}
            if (rawData) {
		    	// write g to fRawData
		    	reshape_chunk_and_write(gE[0], fRawData, maxChunkSize, remainChunkSize, iSizeSplit, nChunk, ngTypeE, ngTypeI, nV1, hWrite);
		    }
        }
		if (saveLGN_fr || learnData_FF) {
		    cudaEventRecord(LGN_ready, mainStream);
		    cudaStreamWaitEvent(LGN_stream, LGN_ready, 0);
        }
        if (saveLGN_fr) {
		    if (saveOutputB4V1) { 
                #ifdef CHECK
			        checkCudaErrors(cudaMemcpyAsync(outputB4V1, d_LGN_fr, outputB4V1Size, cudaMemcpyDeviceToHost, LGN_stream));
                #else
			        cudaMemcpyAsync(outputB4V1, d_LGN_fr, outputB4V1Size, cudaMemcpyDeviceToHost, LGN_stream);
                #endif
		    } else { // use only the first row of outputB4V1
                #ifdef CHECK
				    checkCudaErrors(cudaMemcpyAsync(outputB4V1, d_LGN_fr, nLGN*sizeof(Float), cudaMemcpyDeviceToHost, LGN_stream));
                #else
				    cudaMemcpyAsync(outputB4V1, d_LGN_fr, nLGN*sizeof(Float), cudaMemcpyDeviceToHost, LGN_stream);
                #endif
			}
		    cudaEventRecord(LGN_ready, LGN_stream);
            #ifdef SYNC
                checkCudaErrors(cudaDeviceSynchronize());
            #endif
		}
        //cout << "LGN done\n";
		if (it == nt-1 & print_log) {
			if (sizeof(Float) == 4) {
				printf("\r@t = %f -> %f simulated, frame %d#%d-%d, %.1f%%\n", t, t+dt, currentFrame/nFrame, currentFrame%nFrame, nFrame, 100*static_cast<float>(it+1)/nt);
			} else {
				printf("\r@t = %lf -> %lf simulated, frame %d#%d-%d, %.1f%%\n", t, t+dt, currentFrame/nFrame, currentFrame%nFrame, nFrame, 100*static_cast<float>(it+1)/nt);
			}
			//oldFrame = currentFrame;
		}

		// TODO: block-wise compute_V
		// simulate V1 response
		//compute_V_collect_spike<max_ngTypeFF, max_ngTypeE, max_ngTypeI> <<<nblock, blockSize, 0, mainStream>>> (
		compute_V_collect_spike_learnFF<<<nblock, blockSize, 0, mainStream>>> (
				d_v, d_depC, d_w, d_gFF, d_hFF, dd_gE, dd_gI, dd_hE, dd_hI, // V1 neuron measurements
				d_nLGNperV1, sLGN, LGN_idx, LGN_idy, // LGN->V1 connections
				tBack, d_spikeTrain, // neuron spiking
                vLTD_FF_E, vTrip_FF_E, vLTD_FF_I, vTrip_FF_I, // FF excitatory learning vars
                vAvgE, vAvgI, // filtered spiking 
                vLTP_E, vLTD_E, vTripE, // E->E learning vars
                vSTDP_QE, vSTDP_QI, // I->E learning vars
                d_pFF, d_vR, d_vThres, d_gL, d_C, d_tRef, d_tonicDep, d_vT, d_deltaT, d_tau_w, d_a, d_b, typeAcc, 
                rGenCond, d_synFailFF, d_synPerConFF, rNoisy, d_noisyDep,
				currentTimeSlot, trainDepth, max_LGNperV1,
				ngTypeFF, ngTypeE, ngTypeI, condFF, condE, condI,
				dt, maxChunkSize, remainChunkSize, iSizeSplit, nChunk, nE, nI, nV1, learning, varSlot, nType,
                lFF_E_pre, lFF_I_pre, lFF_E_post, lFF_I_post, lE, lQ, iModel, noDelay); // learning const structs 
        #ifdef CHECK
		    getLastCudaError("compute_V_collect_spike failed");
        #endif
        #ifdef SYNC
            checkCudaErrors(cudaDeviceSynchronize());
        #endif
		cudaEventRecord(spReady, mainStream);
        varSlot = (varSlot+1)%2;
        if (saveLGN_fr) {
            cudaEventSynchronize(LGN_ready);
            if (saveOutputB4V1) {
			    fOutputB4V1.write((char*)outputB4V1, outputB4V1Size); // d_LGN_fr, currentConvol, luminance, contrast
            }
			fLGN_fr.write((char*)outputB4V1, nLGN*sizeof(Float)); 
        }
        if (learnData_FF || getLGN_sp) {
			Float* targetAddress;
        	if (getLGN_sp) targetAddress = LGN_sInfo;
			else targetAddress = outputB4V1;
            #ifdef CHECK
			    checkCudaErrors(cudaMemcpyAsync(targetAddress, d_LGN_sInfo, nLGN*sizeof(Float), cudaMemcpyDeviceToHost, LGN_stream));
            #else
			    cudaMemcpyAsync(targetAddress, d_LGN_sInfo, nLGN*sizeof(Float), cudaMemcpyDeviceToHost, LGN_stream);
            #endif
            #ifdef SYNC
                checkCudaErrors(cudaDeviceSynchronize());
            #endif
        }  

 		if (learnData_FF || (learning && (it+1)%snapshotInterval == 0)) {
            #ifdef CHECK
			    checkCudaErrors(cudaMemcpyAsync(outputB4V1+nLGN, lVarFFpre, LGN_learnPreSize, cudaMemcpyDeviceToHost, LGN_stream));
            #else
			    cudaMemcpyAsync(outputB4V1+nLGN, lVarFFpre, LGN_learnPreSize, cudaMemcpyDeviceToHost, LGN_stream);
			#endif
		    cudaEventRecord(LGN_ready, LGN_stream);
		}

        #ifdef CHECK
		    checkCudaErrors(cudaMemcpyAsync(spikeTrain + currentTimeSlot*nV1, d_spikeTrain + currentTimeSlot*nV1, nV1*sizeof(Float), cudaMemcpyDeviceToHost, mainStream)); // to overlap with  recal_G, to be used in recal_Gvec
        #else
		    cudaMemcpyAsync(spikeTrain + currentTimeSlot*nV1, d_spikeTrain + currentTimeSlot*nV1, nV1*sizeof(Float), cudaMemcpyDeviceToHost, mainStream); // to overlap with  recal_G, to be used in recal_Gvec
        #endif
        #ifdef SYNC
            checkCudaErrors(cudaDeviceSynchronize());
        #endif
		cudaEventRecord(spHostReady, mainStream);
        if (learnData_FF || (learning && (it+1)%snapshotInterval == 0)) {
            #ifdef CHECK
		        checkCudaErrors(cudaMemcpyAsync(LGN_V1_s, sLGN, sLGN_size, cudaMemcpyDeviceToHost, mainStream)); // to overlap with  recal_G, to be used in recal_Gvec
		        checkCudaErrors(cudaMemcpyAsync(lVarFFpost, learnVar, learnVarFFsize0*sizeof(Float), cudaMemcpyDeviceToHost, mainStream));
            #else
		        cudaMemcpyAsync(LGN_V1_s, sLGN, sLGN_size, cudaMemcpyDeviceToHost, mainStream);
		        cudaMemcpyAsync(lVarFFpost, learnVar, learnVarFFsize0*sizeof(Float), cudaMemcpyDeviceToHost, mainStream);
            #endif
            #ifdef SYNC
                checkCudaErrors(cudaDeviceSynchronize());
            #endif
            cudaEventSynchronize(LGN_ready);
			fLearnData_FF.write((char*) outputB4V1, preFFsize); // d_LGN_sInfo and lVarFFpre
        }
        if (getLGN_sp) {
            cudaEventSynchronize(LGN_ready);
			Float* targetAddress;
			if (learnData_FF) targetAddress = outputB4V1;
			else targetAddress = LGN_sInfo;
            fLGN_sp.write((char*) targetAddress, nLGN*sizeof(Float));
        }

        if (matConcurrency < nChunk) { // initially, staging all the chunks of the first matConcurrency
            //size_t initial_size;
            //if (matConcurrency > iSizeSplit) initial_size = iSizeSplit * maxChunkSize + (matConcurrency-iSizeSplit)*remainChunkSize;
            //else initial_size = matConcurrency*maxChunkSize;
            //auto start = chrono::high_resolution_clock::now();
		    //memcpy((void*)(p_conDelayMat), (void*) conDelayMat[0], 2*initial_size*nearBlockSize*sizeof(Float));
		    memcpy((void*)(p_conDelayMat), (void*) conDelayMat[0], ccChunkMatSize*sizeof(float));
            //auto end = chrono::high_resolution_clock::now();
            //double time_taken = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
            //cout << "memcpy of " << 2*initial_size*nearBlockSize*sizeof(Float)/1024.0/1024.0 << " Mb took " << time_taken/1e6 << " ms\n";
        } // else mats are already allocated to pinned memory

		Size chunkSize = maxChunkSize;
		PosInt block_offset = 0;
        size_t p_offset = 0;
        size_t p_total = 0;
		//if (it > 500) {
		//	cudaProfilerStart();
		//}
		for (PosInt i = 0; i < nChunk; i++) {
			if (i >= iSizeSplit) chunkSize = remainChunkSize;
			size_t mChunkSize = chunkSize * nearBlockSize;
            if (i%matConcurrency == 0) {
                p_total += p_offset;
                p_offset = 0;
            }
            if (matConcurrency < nChunk) {
                // staging each chunk at a time
                if (i>=matConcurrency) {
                    //cout << "#" << i << ":\n";
                    //if (cudaSuccess == cudaEventQuery(eTmp[i-1])) cout << "b4memcpy last recal_G already finished\n";
                    //else cout << "b4memcpy last recal_G still running\n";

                    cudaEventSynchronize(matReady[i%matConcurrency]);
                    //auto start = chrono::high_resolution_clock::now();
		            memcpy((void*)(p_conDelayMat + p_offset), (void*) conDelayMat[i], 2*mChunkSize*sizeof(float));
                    //auto end = chrono::high_resolution_clock::now();
                    //double time_taken = chrono::duration_cast<chrono::nanoseconds>(end-start).count();
                    //cout << "memcpy of " << 2*mChunkSize*sizeof(Float)/1024.0/1024.0 << " Mb took " << time_taken/1e6 << " ms\n";

                    //if (cudaSuccess == cudaEventQuery(eTmp[i-1])) cout << "after memcpy last recal_G already finished\n";
                    //else cout << "after memcpy last recal_G still running\n";
                }
            }
            if (matConcurrency < nChunk || learnData_V1) {
                #ifdef CHECK
			        checkCudaErrors(cudaMemcpyAsync(d_conDelayMat[i%matConcurrency], p_conDelayMat + p_offset, 2*mChunkSize*sizeof(float), cudaMemcpyHostToDevice, stream[i%matConcurrency]));
                #else
			        cudaMemcpyAsync(d_conDelayMat[i%matConcurrency], p_conDelayMat + p_offset, 2*mChunkSize*sizeof(float), cudaMemcpyHostToDevice, stream[i%matConcurrency]);
                #endif
                #ifdef SYNC
                    checkCudaErrors(cudaDeviceSynchronize());
                #endif
			    cudaEventRecord(matReady[i%matConcurrency], stream[i%matConcurrency]);
            }
            /*
            if (i>0) {
                if (cudaSuccess == cudaEventQuery(eTmp[i-1])) cout << "after cudamemcpy last recal_G already finished\n";
                else cout << "after cudamemcpy last recal_G still running\n";
            }*/
			float* d_conMat = d_conDelayMat[i%matConcurrency];
			float* d_delayMat = d_conMat + mChunkSize;
            if (i < matConcurrency) {
		        cudaStreamWaitEvent(stream[i%matConcurrency], spReady, 0);
            }
			//recal_G_mat<max_ngTypeE, max_ngTypeI><<< chunkSize, blockSize, 0, stream[i%matConcurrency]>>> (

			recal_G_mat<<< chunkSize, blockSize, 0, stream[i%matConcurrency]>>> (
					d_spikeTrain,
					d_conMat, d_delayMat,
					d_nNeighborBlock+block_offset, d_neighborBlockId + block_offset*nearNeighborBlock,
					d_gE[i], d_gI[i], d_hE[i], d_hI[i],
                    vAvgE, vLTP_E, vLTD_E, vTripE, vSTDP_QE, vSTDP_QI, d_pE, d_pI, typeAcc,
                    rGenCond, d_synFail, d_synPerCon,
					dt, condE, condI, ngTypeE, ngTypeI,
					currentTimeSlot, trainDepth,
					nearNeighborBlock, nE, nI, nV1, speedOfThought, learning, block_offset, nType, lE, lQ, i);

			//cudaEventRecord(eTmp[i], stream[i]);
            #ifdef CHECK
			    getLastCudaError("recal_G failed");
            #endif
            #ifdef SYNC
                checkCudaErrors(cudaDeviceSynchronize());
            #endif
			block_offset += chunkSize;
			p_offset += 2*mChunkSize;
		}
		//if (it > 500) {
		//	cudaProfilerStop();
		//}
        assert(block_offset == nblock);
        assert(p_total + p_offset == matSize*2);
		for (PosInt i = 0; i < matConcurrency; i++) {
			cudaEventRecord(gReady1[i], stream[i]);
        }
		if (iModel == 0) {
        	#ifdef CHECK
        		checkCudaErrors(cudaMemcpyAsync(depC, d_depC, nV1*sizeof(Float)*(2+ngTypeFF*(1+hWrite)), cudaMemcpyDeviceToHost, mainStream));
        	#else
        		cudaMemcpyAsync(depC, d_depC, nV1*sizeof(Float)*(2+ngTypeFF*(1+hWrite)), cudaMemcpyDeviceToHost, mainStream);
        	#endif
		}
		if (iModel == 1) {
        	#ifdef CHECK
        		checkCudaErrors(cudaMemcpyAsync(depC, d_depC, nV1*sizeof(Float)*(3+ngTypeFF*(1+hWrite)), cudaMemcpyDeviceToHost, mainStream));
        	#else
        		cudaMemcpyAsync(depC, d_depC, nV1*sizeof(Float)*(3+ngTypeFF*(1+hWrite)), cudaMemcpyDeviceToHost, mainStream);
        	#endif
		}
		cudaEventRecord(v_gFF_Ready, mainStream);
        #ifdef SYNC
            checkCudaErrors(cudaDeviceSynchronize());
        #endif

		block_offset = 0;
		chunkSize = maxChunkSize;
		cudaEventSynchronize(spHostReady);

        Size nsp = 0;
        spiked = false;
        for (PosInt i=0; i<nV1; i++) {
            Float sp = spikeTrain[currentTimeSlot*nV1 + i];
            if (sp > 0) {
                spiked = true;
                nsp += static_cast<Size>(flooring(sp));
                //break;
            }
        }
        spiked = nsp>0;
        if (print_log) {
            if (spiked) cout << "there's " << nsp << " spiking events during the current time step\n";
            else cout << "no near-neighbor spiking events in the time step\n";
        }

        if (nFar) {
        	farSpiked = fill_fSpikeTrain(fSpikeTrain,  spikeTrain + nV1*currentTimeSlot, fCurrenSlot, vecID, nVec,nV1);
			if (farSpiked) {
		    	for (PosInt i = 0; i < nChunk; i++) {
		    		if (i >= iSizeSplit) chunkSize = remainChunkSize;
		    		size_t gChunkSize = chunkSize*blockSize*(ngTypeE+ngTypeI)*sizeof(Float);
		    		size_t ghChunkSize = gChunkSize*2;
		    		// cpu accumulate conductances from far neighbors
		    		recal_G_vec(
		    				fSpikeTrain, fTrainDepth, fCurrenSlot,
		    				nVec, vecID, conVec, delayVec,
		    				gE[i], gI[i], hE[i], hI[i], &(pE[0]), &(pI[0]), &(typeAccCount[0]),
            	            h_rGenCond, &(synFail[0]), &(synPerCon[0]),
		    				dt, condE, condI, ngTypeE, ngTypeI,
		    				block_offset, nType,
		    				nE, nV1, speedOfThought, chunkSize);
		    		// g and h
            	    #ifdef CHECK
		    		    checkCudaErrors(cudaMemcpyAsync(d_gEt[i], gE[i], ghChunkSize, cudaMemcpyHostToDevice, stream[i])); // size in maxChunk
            	    #else
		    		    cudaMemcpyAsync(d_gEt[i], gE[i], ghChunkSize, cudaMemcpyHostToDevice, stream[i]); // size in maxChunk
            	    #endif
		    		if (i >= matConcurrency) { // otherwise automatically queued in stream
		    			cudaStreamWaitEvent(stream[i], gReady1[i%matConcurrency], 0);
		    		}
		    		//sum_G<max_ngTypeE, max_ngTypeI><<<chunkSize, blockSize, 0, stream[i]>>> (d_nVec + block_offset*blockSize, d_gEt[i], d_gE[i], d_gIt[i], d_gI[i], d_hEt[i], d_hE[i], d_hIt[i], d_hI[i], ngTypeE, ngTypeI);
		    		sum_G<<<chunkSize, blockSize, 0, stream[i]>>> (d_nVec + block_offset*blockSize, d_gEt[i], d_gE[i], d_gIt[i], d_gI[i], d_hEt[i], d_hE[i], d_hIt[i], d_hI[i], ngTypeE, ngTypeI);
		    		// 							  // char*
            	    #ifdef CHECK
            	        if (hWrite) checkCudaErrors(cudaMemcpyAsync(gE[i], d_gE[i], ghChunkSize, cudaMemcpyDeviceToHost, stream[i])); // size in chunk
            	        else checkCudaErrors(cudaMemcpyAsync(gE[i], d_gE[i], gChunkSize, cudaMemcpyDeviceToHost, stream[i])); // size in chunk
            	    #else
            	        if (hWrite) cudaMemcpyAsync(gE[i], d_gE[i], ghChunkSize, cudaMemcpyDeviceToHost, stream[i]); // size in chunk
            	        else cudaMemcpyAsync(gE[i], d_gE[i], gChunkSize, cudaMemcpyDeviceToHost, stream[i]); // size in chunk
            	    #endif
		    		if (i < nChunk-1) {
		    			block_offset += chunkSize;
		    		}
		    		cudaEventRecord(gReady2[i], stream[i]);
		    	}
			}
        }
		if (!farSpiked) {
            if (print_log) {
                cout << "no spikes from distant neighbor or no distant neighbor exists\n";
            }
		    for (PosInt i = 0; i < matConcurrency; i++) {
		    	cudaStreamWaitEvent(stream[0], gReady1[i], 0);
            }
            #ifdef CHECK
                if (hWrite) checkCudaErrors(cudaMemcpyAsync(gE[0], d_gE[0], ghSize, cudaMemcpyDeviceToHost, stream[0])); // size in chunk
                else checkCudaErrors(cudaMemcpyAsync(gE[0], d_gE[0], ghSize/2, cudaMemcpyDeviceToHost, stream[0])); // size in chunk
            #else
                if (hWrite) cudaMemcpyAsync(gE[0], d_gE[0], ghSize, cudaMemcpyDeviceToHost, stream[0]); // size in chunk
                else cudaMemcpyAsync(gE[0], d_gE[0], ghSize/2, cudaMemcpyDeviceToHost, stream[0]); // size in chunk
            #endif
		    cudaEventRecord(gReady2[0], stream[0]);
        }
        if (learnData_FF || (learning && (it+1)%snapshotInterval == 0)) {
		    for (PosInt i = 0; i < matConcurrency; i++) {
		        cudaStreamWaitEvent(mainStream, gReady1[i], 0);
            }
            #ifdef CHECK
		        checkCudaErrors(cudaMemcpyAsync(lVarFFpost+learnVarFFsize0, learnVar+learnVarFFsize0, learnVarFFsize1*sizeof(Float), cudaMemcpyDeviceToHost, mainStream));
            #else
		        cudaMemcpyAsync(lVarFFpost+learnVarFFsize0, learnVar+learnVarFFsize0, learnVarFFsize1*sizeof(Float), cudaMemcpyDeviceToHost, mainStream);
            #endif
		    cudaEventRecord(learnFF_event, mainStream);
        }
        #ifdef SYNC
            checkCudaErrors(cudaDeviceSynchronize());
        #endif
        if (framePhyV1output || frameVisV1output || frameVisLGNoutput) { // framed output
		    if (it%ot == 0) { // re-initialize
                #ifdef CHECK
		    	    checkCudaErrors(cudaMemsetAsync(d_outputFrame, 0, framesSize, ostream[0]));
                #else
		    	    cudaMemsetAsync(d_outputFrame, 0, framesSize, ostream[0]);
                #endif
                //cout << it << "%" << ot << " = " << it%ot << " reset phyV1\n";
		        cudaEventRecord(frameReady, ostream[0]);
		        if (frameVisV1output) {
		            cudaStreamWaitEvent(ostream[1], frameReady, 0);
                }
		        if (frameVisLGNoutput) {
		            cudaStreamWaitEvent(ostream[2], frameReady, 0);
                }
		    }
            if (spiked) {
                if (framePhyV1output) {
		            cudaStreamWaitEvent(ostream[0], spReady, 0);
		            pixelizeOutput<<<(nPixel_phyV1+blockSize-1)/blockSize, blockSize, 0, ostream[0]>>>(d_spikeTrain+currentTimeSlot*nV1, d_V1SpPhyFrame, d_V1_phyFramePosId, d_nV1perPhyPixel, maxV1perPixel, 0, nPixel_phyV1, nPixel_phyV1, nV1, odt);
                    #ifdef CHECK
		                getLastCudaError("pixelizeOutput phyV1 failed");
                    #endif
                }
                #ifdef SYNC
                    checkCudaErrors(cudaDeviceSynchronize());
                #endif

		        if (frameVisV1output) {
		            cudaStreamWaitEvent(ostream[1], spReady, 0);
		    	    pixelizeOutput<<<(nPixel_visV1+blockSize-1)/blockSize, blockSize, 0, ostream[1]>>>(d_spikeTrain+currentTimeSlot*nV1, d_V1SpVisFrame, d_V1_visFramePosId, d_nV1perVisPixel, maxV1perPixel_I, maxV1perPixel_C, nPixel_visV1/2, nPixel_visV1, nV1, odt);
                    #ifdef CHECK
		    	        getLastCudaError("pixelizeOutput visV1 failed");
                    #endif
                }
                #ifdef SYNC
                    checkCudaErrors(cudaDeviceSynchronize());
                #endif
		    }

		    if (frameVisLGNoutput) {
		        cudaStreamWaitEvent(ostream[2], spReady, 0);
		    	pixelizeOutput<<<(nPixel_visLGN+blockSize-1)/blockSize, blockSize, 0, ostream[2]>>>(d_LGN_fr, d_LGN_spVisFrame, d_LGN_visFramePosId, d_nLGNperPixel, maxLGNperPixel_I, maxLGNperPixel_C, nPixel_visLGN/2, nPixel_visLGN, nLGN, odt);
                #ifdef CHECK
		    	    getLastCudaError("pixelizeOutput visLGN failed");
                #endif
		    }

		    if ((it+1)%ot == 0) { // finished sum and output
                #ifdef CHECK
		    	    checkCudaErrors(cudaMemcpy(outputFrame, d_outputFrame, framesSize, cudaMemcpyDeviceToHost));
                #else
		    	    cudaMemcpy(outputFrame, d_outputFrame, framesSize, cudaMemcpyDeviceToHost);
                #endif
		    	fOutputFrame.write((char*)outputFrame, framesSize);
		    }
            #ifdef SYNC
                checkCudaErrors(cudaDeviceSynchronize());
            #endif
        }
        if (!print_log) {
			printf("\r%.1f%%", 100*static_cast<float>(it+1)/nt);
        }
		if ((it+1)%snapshotInterval == 0 || it == nt-1) {
			timeStamp = time(NULL);
			snapShot_fn = "snapShot_"+ to_string(timeStamp) + output_suffix;
			fSnapshot.open(snapShot_fn, fstream::out | fstream::binary);
			if (!fSnapshot) {
				cout << "cannot create " << snapShot_fn << " to store the snapshotInterval\n";
				return EXIT_FAILURE;
			} else {
				PosInt qt = it0 + it+1;
				fSnapshot.write((char*)&qt, sizeof(PosInt));
				fSnapshot.write((char*)&frameCycle, sizeof(PosInt));
				fSnapshot.write((char*)&iFrameHead, sizeof(PosInt));
				fSnapshot.write((char*)&oldFrameHead, sizeof(PosInt));
				fSnapshot.write((char*)&currentFrame, sizeof(PosInt));
				fSnapshot.write((char*)&iFramePhaseHead, sizeof(PosInt));
				fSnapshot.write((char*)&iFramePhaseTail, sizeof(PosInt));
				fSnapshot.write((char*)&mFramePhaseHead, sizeof(PosInt));
				fSnapshot.write((char*)&mFramePhaseTail, sizeof(PosInt));
				fSnapshot.write((char*)&iStatus, sizeof(PosInt));
				PosInt _currentTimeSlot = (currentTimeSlot + 1)%trainDepth;
				fSnapshot.write((char*)& _currentTimeSlot, sizeof(PosInt));
				//cout << "it0 = " << qt << "\n";
				//cout << "frameCycle = " << frameCycle << "\n";
				//cout << "iFrameHead = " << iFrameHead << "\n";
				//cout << "oldFrameHead = " << oldFrameHead << "\n";
				//cout << "currentFrame = " << currentFrame << "\n";
				//cout << "iFramePhaseHead = " << iFramePhaseHead << "\n";
				//cout << "iFramePhaseTail = " << iFramePhaseTail << "\n";
				//cout << "mFramePhaseHead = " << mFramePhaseHead << "\n";
				//cout << "mFramePhaseTail = " << mFramePhaseTail << "\n";
				//cout << "currentTimeSlot = " << _currentTimeSlot << "\n";
				//cout << "iStatus = " << iStatus << "\n";

				fSnapshot.write((char*)&iModel, sizeof(int));
				fSnapshot.write((char*)&learning, sizeof(int));

				fSnapshot.write((char*)&nV1, sizeof(Size));
				fSnapshot.write((char*)&nLGN, sizeof(Size));
				fSnapshot.write((char*)&ngTypeFF, sizeof(Size));
				fSnapshot.write((char*)&ngTypeE, sizeof(Size));
				fSnapshot.write((char*)&ngTypeI, sizeof(Size));
				fSnapshot.write((char*)&nType, sizeof(Size));
				fSnapshot.write((char*)&trainDepth, sizeof(Size));

				// write vars
				char* _sav = new char[deviceMemorySize];
				// LGN
				Float* r_leftTimeRate = (Float*) _sav;
				Float* r_lastNegLogRand = r_leftTimeRate + nLGN;
				curandStateMRG32k3a* r_randState = (curandStateMRG32k3a*) (r_lastNegLogRand + nLGN);
				checkCudaErrors(cudaMemcpy(r_leftTimeRate, leftTimeRate, LGN_convolSize, cudaMemcpyDeviceToHost));
				fSnapshot.write((char*)r_leftTimeRate, LGN_convolSize);
				Float* r_lVarFFpre;
				if (learning) {
					if (!learnData_FF) {
						LGN_V1_s = (Float*) (r_leftTimeRate + LGN_convolSize/sizeof(Float)); 
						lVarFFpost = LGN_V1_s + sLGN_size/sizeof(Float);
						r_lVarFFpre = lVarFFpost + learnVarFFsize * sizeof(Float);
					} else {
						r_lVarFFpre = (Float*) (r_leftTimeRate + LGN_convolSize/sizeof(Float));
					}

					// makesure ready
					cudaEventSynchronize(learnFF_event);
					checkCudaErrors(cudaMemcpy(LGN_V1_s, sLGN, sLGN_size, cudaMemcpyDeviceToHost));
					fSnapshot.write((char*)LGN_V1_s, sLGN_size);

					checkCudaErrors(cudaMemcpy(lVarFFpost, learnVar, learnVarFFsize*sizeof(Float), cudaMemcpyDeviceToHost));
					fSnapshot.write((char*)lVarFFpost, learnVarFFsize*sizeof(Float));

					checkCudaErrors(cudaMemcpy(r_lVarFFpre, lVarFFpre, LGN_learnPreSize, cudaMemcpyDeviceToHost));
					fSnapshot.write((char*)r_lVarFFpre, LGN_learnPreSize);
				}
				// V1
				curandStateMRG32k3a* r_rGenCond;
				if (learning) {
					r_rGenCond = (curandStateMRG32k3a *) (r_lVarFFpre + LGN_learnPreSize/sizeof(Float));
				} else {
					r_rGenCond = r_randState + nLGN;
				}
				checkCudaErrors(cudaMemcpy(r_rGenCond, rGenCond, nV1*sizeof(curandStateMRG32k3a), cudaMemcpyDeviceToHost));
				fSnapshot.write((char*)r_rGenCond, nV1*sizeof(curandStateMRG32k3a));

				curandStateMRG32k3a* r_rNoisy =  r_rGenCond + nV1;
				checkCudaErrors(cudaMemcpy(r_rNoisy, rNoisy, nV1*sizeof(curandStateMRG32k3a), cudaMemcpyDeviceToHost));
				fSnapshot.write((char*)r_rNoisy, nV1*sizeof(curandStateMRG32k3a));

				Float* r_tBack =  (Float*) (r_rNoisy + nV1);
				checkCudaErrors(cudaMemcpy(r_tBack, tBack, nV1*sizeof(Float), cudaMemcpyDeviceToHost));
				fSnapshot.write((char*)r_tBack, nV1*sizeof(Float));

				delete []_sav;
				// already pinned
				checkCudaErrors(cudaMemcpy(spikeTrain, d_spikeTrain, trainSize*sizeof(Float), cudaMemcpyDeviceToHost));
				fSnapshot.write((char*)spikeTrain, trainSize*sizeof(Float));
				// makesure ready
			    cudaEventSynchronize(v_gFF_Ready);
				if (nFar) { 
            	    for (PosInt i=0; i<nChunk; i++) {
	        	        cudaEventSynchronize(gReady2[i]);
            	    }
            	} else {
	        	    cudaEventSynchronize(gReady2[0]);
            	}

				for (PosInt i = 0; i<trainDepth; i++) {
					for (PosInt j=0; j<nV1; j++) {
						assert(!std::isnan(spikeTrain[i*nV1 + j]));
						if (spikeTrain[i*nV1 + j] < 1) {
							assert(spikeTrain[i*nV1 + j] == 0);
						}
					}
				}
				if (iModel == 0) {
					checkCudaErrors(cudaMemcpy(v, d_v, r_vghSize, cudaMemcpyDeviceToHost));
					fSnapshot.write((char*)v, r_vghSize);
					for (PosInt j=0; j<r_vghSize/sizeof(Float); j++) {
						assert(!std::isnan(v[j]));
					}
				} 
				if (iModel == 1) {
					checkCudaErrors(cudaMemcpy(w, d_w, r_vghSize, cudaMemcpyDeviceToHost));
					fSnapshot.write((char*)w, r_vghSize);
					for (PosInt j=0; j<r_vghSize/sizeof(Float); j++) {
						assert(!std::isnan(w[j]));
					}
				}
				fSnapshot.close();
				// remove old snapshotInterval
				if (oldTimeStamp != 0 && delPrevSnapshot) {
					string oldSnapshot_Fn = "snapShot_"+ to_string(oldTimeStamp) + output_suffix;
					if (remove(oldSnapshot_Fn.c_str()) != 0) cout << "failed to remove preivous snapshot\n";
					else cout << "... removed previous snapshot\n";
				}
				oldTimeStamp = timeStamp;
				cout << "saved a snapshot in " << snapShot_fn << "\n";
			}
		}
	}
    if (rawData) {
	    fRawData.write((char*) (spikeTrain + nV1*currentTimeSlot), nV1*sizeof(Float));
	    //cudaEventSynchronize(v_gFF_Ready); already by snapshot
		if (iModel == 0) {
	    	fRawData.write((char*) depC, nV1*sizeof(Float)*(2+ngTypeFF*(1+hWrite)));
		}
		if (iModel == 1) {
	    	fRawData.write((char*) depC, nV1*sizeof(Float)*(3+ngTypeFF*(1+hWrite)));
		}
	    // write g to fRawData
		/* already by snapshot
        if (nFar) { 
            for (PosInt i=0; i<nChunk; i++) {
	            cudaEventSynchronize(gReady2[i]);
            }
        } else {
	        cudaEventSynchronize(gReady2[0]);
        }*/
	    reshape_chunk_and_write(gE[0], fRawData, maxChunkSize, remainChunkSize, iSizeSplit, nChunk, ngTypeE, ngTypeI, nV1, hWrite);
    }
    if (learnData_FF) {
        if (!rawData) {
		    fLearnData_FF.write((char*) (spikeTrain + nV1*currentTimeSlot), nV1*sizeof(Float));
			//cudaEventSynchronize(v_gFF_Ready); already by snapshot
		    fLearnData_FF.write((char*) gFF, nV1*sizeof(Float));
        }
		//cudaEventSynchronize(learnFF_event); already by snapshot
		fLearnData_FF.write((char*) LGN_V1_s, sLGN_size);
		fLearnData_FF.write((char*) lVarFFpost, learnVarFFsize*sizeof(Float));
        f_sLGN.write((char*) LGN_V1_s, sLGN_size);
    }
	cout << "simulation for " << output_suffix0 << " done.\n";

	ofstream fPatchV1_cfg;
	fPatchV1_cfg.open(patchV1_cfg_filename + output_suffix, fstream::out | fstream::binary);
	if (!fPatchV1_cfg) {
		cout << "Cannot open or find " << patchV1_cfg_filename + output_suffix <<"\n";
		return EXIT_FAILURE;
	} else {

		unsigned int precision = sizeof(Float);
		fPatchV1_cfg.write((char*)&precision, sizeof(unsigned int));
		Float _vL = vL;	
		Float _vE = vE;	
		Float _vI = vI;	
		fPatchV1_cfg.write((char*) &_vL, sizeof(Float));
		fPatchV1_cfg.write((char*) &_vE, sizeof(Float));
		fPatchV1_cfg.write((char*) &_vI, sizeof(Float));
		fPatchV1_cfg.write((char*) &nType, sizeof(Size));
		fPatchV1_cfg.write((char*) &(vR[0]), nType*sizeof(Float));
		fPatchV1_cfg.write((char*) &(vThres[0]), nType*sizeof(Float));
		fPatchV1_cfg.write((char*) &(gL[0]), nType*sizeof(Float));
		fPatchV1_cfg.write((char*) &(vT[0]), nType*sizeof(Float));
		fPatchV1_cfg.write((char*) &(nTypeHierarchy[0]), 2*sizeof(Size));
		fPatchV1_cfg.write((char*) &(typeAccCount[0]), nType*sizeof(Size));
		fPatchV1_cfg.write((char*) &(sRatioLGN[0]), nType*sizeof(Float));
		fPatchV1_cfg.write((char*) &sRatioV1, sizeof(Float));
		fPatchV1_cfg.write((char*) &frRatioLGN, sizeof(Float));
		fPatchV1_cfg.write((char*) &convolRatio, sizeof(Float));

		fPatchV1_cfg.write((char*) &seed, sizeof(PosIntL));	
		fPatchV1_cfg.write((char*) &nLGN, sizeof(Size));	
		fPatchV1_cfg.write((char*) &nV1, sizeof(Size));	
		fPatchV1_cfg.write((char*) &nType, sizeof(Size));	
		fPatchV1_cfg.write((char*) &iModel, sizeof(int));	
		fPatchV1_cfg.write((char*) &learning, sizeof(int));	
		fPatchV1_cfg.write((char*) &ngTypeFF, sizeof(Size));	
		fPatchV1_cfg.write((char*) &ngTypeE, sizeof(Size));	
		fPatchV1_cfg.write((char*) &ngTypeI, sizeof(Size));	
		fPatchV1_cfg.write((char*) &(grFF[0]), ngTypeFF*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(gdFF[0]), ngTypeFF*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(grE[0]), ngTypeE*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(gdE[0]), ngTypeE*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(grI[0]), ngTypeI*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(gdI[0]), ngTypeI*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(pFF[0]), ngTypeFF*nType*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(pE[0]), ngTypeE*nType*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(pI[0]), ngTypeI*nType*sizeof(Float));	
		fPatchV1_cfg.write((char*) &frRatioLGN, sizeof(Float));	
		fPatchV1_cfg.write((char*) &convolRatio, sizeof(Float));	
		fPatchV1_cfg.write((char*) &(C[0]), nType*sizeof(Float));
		fPatchV1_cfg.write((char*) &(tRef[0]), nType*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(a[0]), nType*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(b[0]), nType*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(tau_w[0]), nType*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(deltaT[0]), nType*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(spE0[0]), nTypeHierarchy[0]*2*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(spI0[0]), nTypeHierarchy[1]*2*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(v0[0]), nType*2*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(w0[0]), nType*2*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(gFF0[0]), nType*2*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(gE0[0]), nType*2*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(gI0[0]), nType*2*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(synFail[0]), nType*nType*sizeof(Float));	
		fPatchV1_cfg.write((char*) &(synFailFF[0]), nType*sizeof(Float));	
	}

	{ // clean-up
		fStimulus.close();
        if (rawData) fRawData.close();
		if (saveLGN_fr) fLGN_fr.close();
		if (saveOutputB4V1) fOutputB4V1.close();
        if (learnData_FF) {
            fLearnData_FF.close();
            f_sLGN.close();
        }
        if (getLGN_sp) {
            fLGN_sp.close();
        }
		delete []galleryOutput;
		delete []d_gE;
		delete []d_gI;
		delete []d_hE;
		delete []d_hI;
		delete []d_gEt;
		delete []d_gIt;
		delete []d_hEt;
		delete []d_hIt;
		delete []gE;
		delete []gI;
		delete []hE;
		delete []hI;
		delete []conDelayMat;
		delete []d_conDelayMat;
		delete []exact_norm;
		delete []exact_it;
		delete []outputFrame;
        delete []h_rGenCond;
        if (matConcurrency < nChunk) {
            delete []conDelayMat0;
        }

		checkCudaErrors(cudaEventDestroy(v_gFF_Ready));
		checkCudaErrors(cudaEventDestroy(spReady));
		checkCudaErrors(cudaEventDestroy(frameReady));
		checkCudaErrors(cudaEventDestroy(spHostReady));
		checkCudaErrors(cudaEventDestroy(LGN_ready));
		checkCudaErrors(cudaEventDestroy(learnFF_event));
		for (PosInt i=0; i<nChunk; i++) {
			checkCudaErrors(cudaEventDestroy(gReady1[i]));
			checkCudaErrors(cudaEventDestroy(gReady2[i]));
            if (i < matConcurrency) checkCudaErrors(cudaEventDestroy(matReady[i]));
			checkCudaErrors(cudaEventDestroy(eTmp[i]));
		}
        delete []gReady1;
        delete []gReady2;
        delete []matReady;
        delete []eTmp;

		checkCudaErrors(cudaStreamDestroy(mainStream));
		checkCudaErrors(cudaStreamDestroy(LGN_stream));
		for (PosInt i=0; i<nChunk; i++) {
			checkCudaErrors(cudaStreamDestroy(stream[i]));
		}
		delete []stream;
		for (PosInt i=0; i<3; i++) {
			checkCudaErrors(cudaStreamDestroy(ostream[i]));
		}

        if (getLGN_sp) {
		    checkCudaErrors(cudaFreeHost(LGN_sInfo));
        }
		checkCudaErrors(cudaFreeHost(pinnedMem));
		checkCudaErrors(cudaFreeHost(p_conDelayMat));
        if (saveOutputB4V1 || saveLGN_fr || learnData_FF) {
		    checkCudaErrors(cudaFreeHost(outputB4V1));
        }
		dLGN.freeMem();

        if (LGN_switch) {
		    checkCudaErrors(cudaFree(dLGN_type));
        }
		checkCudaErrors(cudaFree(gpu_B4V1));
		checkCudaErrors(cudaFree(gpu_LGN_gallery));
		checkCudaErrors(cudaFree(d_mat));
		checkCudaErrors(cudaFree(d_vgh));
		checkCudaErrors(cudaFree(d_gh));
		checkCudaErrors(cudaFree(d_spikeTrain));
		checkCudaErrors(cudaFree(sLGN));
		checkCudaErrors(cudaFree(d_surfacePos));
		checkCudaErrors(cudaFree(d_neighborInfo));
        checkCudaErrors(cudaFree(dd_gE));
        checkCudaErrors(cudaFree(dd_gI));
        checkCudaErrors(cudaFree(dd_hE));
        checkCudaErrors(cudaFree(dd_hI));
        checkCudaErrors(cudaFree(rGenCond));
        checkCudaErrors(cudaFree(rNoisy));
        checkCudaErrors(cudaFree(d_noisyDep));
        checkCudaErrors(cudaFree(d_synFailFF));
        checkCudaErrors(cudaFree(d_synFail));
        checkCudaErrors(cudaFree(d_synPerConFF));
        checkCudaErrors(cudaFree(d_synPerCon));
        checkCudaErrors(cudaFree(typeAcc));
        checkCudaErrors(cudaFree(d_pFF));
        checkCudaErrors(cudaFree(d_pE));
        checkCudaErrors(cudaFree(d_pI));
        checkCudaErrors(cudaFree(d_vT));
        checkCudaErrors(cudaFree(d_vR));
        checkCudaErrors(cudaFree(d_gL));
        checkCudaErrors(cudaFree(d_C));
        checkCudaErrors(cudaFree(d_tRef));
        checkCudaErrors(cudaFree(d_tonicDep));
        checkCudaErrors(cudaFree(learnVar));
        if (learnData_FF) {
		    checkCudaErrors(cudaFreeHost(LGN_V1_s));
		    checkCudaErrors(cudaFreeHost(lVarFFpost));
		    checkCudaErrors(cudaFree(lVarFFpre));
		    checkCudaErrors(cudaFree(d_LGN_sInfo));
        }
        if (framePhyV1output) {
		    if (framePhyV1output) checkCudaErrors(cudaFree(d_V1_phyFrame));
		    if (frameVisV1output) checkCudaErrors(cudaFree(d_V1_visFrame));
		    if (frameVisLGNoutput) checkCudaErrors(cudaFree(d_LGN_visFrame));
		    checkCudaErrors(cudaFree(d_outputFrame));
        }
		checkCudaErrors(cudaFreeArray(cuArr_L));
		checkCudaErrors(cudaFreeArray(cuArr_M));
		checkCudaErrors(cudaFreeArray(cuArr_S));
		checkCudaErrors(cudaFreeArray(cuSurfArray));
		checkCudaErrors(cudaDeviceSynchronize());
		cout << "memory trace cleaned\n";
	}
	return EXIT_SUCCESS;
}
