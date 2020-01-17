#define _USE_MATH_DEFINES
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <tuple>
#include <vector>
#include "boost/program_options.hpp"
#include "LGN_props.cuh"
#include "LGN_props.h"
#include "discrete_input_convol.cuh"
#include "util/util.h"
#include "preprocess/RFtype.h"
#include "global.h"
#include "CUDA_MACRO.h"

// the retinal discrete x, y as cone receptors id
void init_layer(texture<float, cudaTextureType2DLayered> &layer) {
    layer.addressMode[0] = cudaAddressModeBorder;
    layer.addressMode[1] = cudaAddressModeBorder; 
    layer.filterMode = cudaFilterModeLinear;
    layer.normalized = true; //accessing coordinates are normalized
}

void prep_sample(unsigned int iSample, unsigned int width, unsigned int height, float* L, float* M, float* S, cudaArray *dL, cudaArray *dM, cudaArray *dS, unsigned int nSample, cudaMemcpyKind cpyKind) {
    // copy the three channels L, M, S of the #iSample frame to the cudaArrays dL, dM and dS
    cudaMemcpy3DParms params = {0};
    params.srcPos = make_cudaPos(0, 0, 0);
    params.dstPos = make_cudaPos(0, 0, iSample);
    params.extent = make_cudaExtent(width, height, nSample);
    params.kind = cpyKind;

    params.srcPtr = make_cudaPitchedPtr(L, width * sizeof(float), width, height);
    params.dstArray = dL;
    checkCudaErrors(cudaMemcpy3D(&params));

    params.srcPtr = make_cudaPitchedPtr(M, width * sizeof(float), width, height);
    params.dstArray = dM;
    checkCudaErrors(cudaMemcpy3D(&params));

    params.srcPtr = make_cudaPitchedPtr(S, width * sizeof(float), width, height);
    params.dstArray = dS;
    checkCudaErrors(cudaMemcpy3D(&params));
}

int main(int argc, char **argv) {
	namespace po = boost::program_options;
    using namespace std;
	using std::string;

    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
    printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);
    printf("total global memory: %f Mb.\n", deviceProps.totalGlobalMem/1024.0/1024.0);

#ifdef SINGLE_PRECISION
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
#else
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
    printf("maximum threads per MP: %d.\n", deviceProps.maxThreadsPerMultiProcessor);
    printf("shared memory per block: %zu bytes.\n", deviceProps.sharedMemPerBlock);
    printf("registers per block: %d.\n", deviceProps.regsPerBlock);
    cout << "\n";
    
    Float vFt;
	bool storeSpatial = true;
    Float dt; // in ms, better in fractions of binary 
	bool useNewLGN;
	bool testStorage;
	bool testLuminanceAd;
	bool checkConvol;
    SmallSize nSpatialSample1D; // spatial kernel sample size = nSpatialSample1D x nSpatialSample1D
    SmallSize nKernelSample;
    Float nsig = 3; // extent of spatial RF sampling in units of std
    Float tau; // required length of memory for LGN temporal kernel
    Float Itau; // in ms .. cone adaptation at 300ms https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003289
    PosInt frameRate; // Hz
    Size nt; // number of steps
	PosIntL seed;

	po::options_description generic_opt("Generic options");
	generic_opt.add_options()
		("seed,s", po::value<PosIntL>(&seed),"seed for trial")
		("cfg_file,c", po::value<string>()->default_value("patchV1.cfg"), "filename for configuration file")
		("help,h", "print usage");
	po::options_description top_opt("top-level configuration");
	// non-files
	top_opt.add_options()
		("vFt,v", po::value<Float>(&vFt)->default_value(1.0),"variable for test")
		("dt", po::value<Float>(&dt)->default_value(0.0625), "simulatoin time step in ms") 
		("nt", po::value<Size>(&nt)->default_value(8000), "total simulatoin time in units of time step")
		("nSpatialSample1D", po::value<SmallSize>(&nSpatialSample1D)->default_value(warpSize), "number of samples per x,y direction for a LGN spatial RF")
		("tau", po::value<Float>(&tau)->default_value(250.0), "the backward time interval that a LGN temporal RF should cover")
		("Itau", po::value<Float>(&Itau)->default_value(300.0), "the light intensity adaptation time-scale of a cone")
		("nKernelSample", po::value<Size>(&nKernelSample)->default_value(500), "number of samples per temporal kernel")
		("frameRate", po::value<PosInt>(&frameRate)->default_value(60), "frame rate of the input stimulus")
		("testStorage", po::value<bool>(&testStorage)->default_value(true), "check storage values, write data to disk, filename specified by storage_filename")
		("testLuminanceAd", po::value<bool>(&testLuminanceAd)->default_value(true), "check adapted luminance values, write data to disk, filename specified by luminanceAd_filename")
		("checkConvol", po::value<bool>(&checkConvol)->default_value(true), "check convolution values, write data to disk, filename specified by LGN_convol_filename")
		("useNewLGN", po::value<bool>(&useNewLGN)->default_value(true), "regenerate the a new ensemble of LGN parameters according to their distribution");

    // files
	string stimulus_filename, V1_filename, LGN_filename, LGN_V1_s_filename, LGN_V1_ID_filename; // inputs
    string LGN_fr_filename; // outputs
	string LGN_convol_filename, max_convol_filename, storage_filename, luminanceAd_filename;
	top_opt.add_options()
		("fStimulus", po::value<string>(&stimulus_filename)->default_value("stimulus.bin"),"file that stores LGN firing rates, array of size (nframes,width,height,3)")
		("fV1", po::value<string>(&V1_filename)->default_value("V1.bin"),"file that stores V1 neurons information")
		("fLGN", po::value<string>(&LGN_filename)->default_value("LGN(vpos).bin"),"file that stores LGN neurons information")
		("fLGN_V1_ID", po::value<string>(&LGN_V1_ID_filename)->default_value("LGN_V1_idList.bin"),"file stores LGN to V1 connections")
		("fLGN_V1_s", po::value<string>(&LGN_V1_s_filename)->default_value("LGN_V1_sList.bin"),"file stores LGN to V1 connection strengths")
		("fLGN_fr", po::value<string>(&LGN_fr_filename)->default_value("LGN_fr.bin"),"file stores LGN firing rates")
		("fLuminanceAd", po::value<string>(&luminanceAd_filename)->default_value("apdated_luminance.bin"),"file that stores adapted luminance values, mimicking cones' response") // TEST 
		("fLGN_convol", po::value<string>(&LGN_convol_filename)->default_value("LGN_convol.bin"),"file that stores LGN convolution values") // TEST 
		("fStorage", po::value<string>(&storage_filename)->default_value("storage.bin"),"file that stores spatial and temporal convolution parameters") // TEST 
		("fLGN_max", po::value<string>(&max_convol_filename)->default_value("max_convol.bin"),"file that stores LGN maximum values of convolution"); // TEST
	
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
	ifstream cfg_file{cfg_filename.c_str()};
	if (cfg_file) {
		po::store(po::parse_config_file(cfg_file, config_file_options), vm);
		cout << "Using configuration file: " << cfg_filename << "\n";
	} else {
		cout << "No configuration file is given, default values are used for non-specified parameters\n";
	}
	po::notify(vm);

    printf("simulating for %u steps, t = %f ms\n", nt, nt*dt);

    if (nSpatialSample1D > 32) {
		cout << "nSpatialSample1D has to be smaller than 32 (1024 threads per block).\n";
        return EXIT_FAILURE;
    }

    Float init_luminance = 2.0/6.0; //1.0/6.0;

    // from the retina facing out
	ifstream fStimulus; // inputs
    fstream fLGN, fV1; 
	ofstream fLGN_fr; // outputs
	ofstream fLGN_convol, fmax_convol, fStorage, fLuminanceAd;

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
        cout << "stimulus: " << nFrame << " frames of " << height << "x" << height << "\n";
    }
    nPixelPerFrame = width*height;
    vector<float> domain(2, 0); 
    fStimulus.read(reinterpret_cast<char*>(&domain[0]), 2 * sizeof(float));
    stimulus_buffer = domain[0];
    stimulus_range = domain[1];
    streampos sofStimulus = fStimulus.tellg();
    fStimulus.seekg(0, fStimulus.end);
    streampos eofStimulus = fStimulus.tellg();
    fStimulus.seekg(sofStimulus);
    // in each frame
    // stimulus start from 0 -> ecc
	/*			
	 			left-eye		right-eye
	     1	_______________ ______________
	 	 ^ |b-------------|b-------------| <- buffer(b)  
	     | |b|			  |b|			 |  
	     | |b|			  |b|			 | 
	     | |b|			  |b|			 | 
	     | |b|			  |b|			 | 
	     | |b|			  |b|			 | 
	     | |b|*		      |b|*		  	 |  2 x range
	     | |b|			  |b|			 | 
	     | |b|			  |b|			 | 
	     | |b|			  |b|			 | 
	     | |b|			  |b|			 | 
	 	 | |b|____________|b|____________|
	 	 0 |--------------|--------------| <- buffer(b)
	 	   0---------------------------->1
	 	   |b| <- range ->|b| <- range ->|				 
	*/

    // DON'T close file, still being read during simulation
    
    Float deg2rad = M_PI/180.0;
    Float rad2deg = 180.0/M_PI;
    Size nLGN_L, nLGN_R, nLGN;
	Float max_ecc, L_x0, L_y0, R_x0, R_y0, normViewDistance;
    fLGN.open(LGN_filename, fstream::in | fstream::binary);
    if (!fLGN) {
		cout << "Cannot open or find " << V1_filename <<" to read in LGN properties.\n";
        return EXIT_FAILURE;
    } else {
	    fLGN.read(reinterpret_cast<char*>(&nLGN_L), sizeof(Size));
	    fLGN.read(reinterpret_cast<char*>(&nLGN_R), sizeof(Size));
	    fLGN.read(reinterpret_cast<char*>(&max_ecc), sizeof(Float)); // in rad
    }
    nLGN = nLGN_L + nLGN_R;
	Float stimulus_extent = stimulus_range + stimulus_buffer;
    if (max_ecc >= stimulus_range) {
        printf("%f < %f\n", max_ecc, stimulus_range);
        assert(max_ecc < stimulus_range);
    }
	Float normEccMaxStimulus_extent = max_ecc/(2*stimulus_extent); // just the ecc at VF center, its surround can be much bigger, normalized for texture coordinates
    // normalized stimulus reading points for stimulus access
	L_x0 = stimulus_buffer/(2*stimulus_extent);
	L_y0 = 0.5;
	R_x0 = 0.5 + L_x0;
	R_y0 = 0.5;
    cout << nLGN << " LGN neurons, " << nLGN_L << " from left eye, " << nLGN_R << " from right eye, center positions are within the eccentricity of " << max_ecc << " deg, reaching normalized stimulus radius of " << normEccMaxStimulus_extent << "(" << max_ecc << ", " << stimulus_range << ")" << " and a buffer range of " << L_x0 << "(" << stimulus_buffer << ")" << ".\n";
    max_ecc = max_ecc * deg2rad;
	normViewDistance = normEccMaxStimulus_extent/tan(max_ecc);
    cout << "normalized view distance: " << normViewDistance << "\n";

    const SmallSize nType = 2;
    Size nSample = nSpatialSample1D * nSpatialSample1D;
    printf("=== temporal storage memory required: %dx%dx%d = %fMB\n", nKernelSample,nLGN,nType,nKernelSample*nLGN*nType*sizeof(Float)/1024.0/1024.0);
    printf("=== spatial storage memory required: %dx%dx%d = %fMB\n", nSample, nLGN, nType, nSample*nType*nLGN*sizeof(Float)/1024.0/1024.0);

    printf("=== texture coord storage memory required: %dx%dx%d = %fMB\n", nSample, nLGN, nType, 2*nSample*nType*nLGN*sizeof(float)/1024.0/1024.0);

    // vectors to initialize LGN
	// center surround, thus the x2
	vector<Float> LGN_polar(nLGN*2);
	vector<Float> LGN_ecc(nLGN*2);
	vector<InputType> LGNtype(nLGN);
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

    vector<Float> c50(nLGN*2);
    vector<Float> sharpness(nLGN*2);
    vector<Float> spont(nLGN*2);
	vector<PosInt> coneType(nLGN*2);
    vector<Float> covariant(nLGN);

	fLGN.read(reinterpret_cast<char*>(&LGNtype[0]), nLGN*sizeof(InputType_t));
	fLGN.read(reinterpret_cast<char*>(&LGN_polar[0]), nLGN*sizeof(Float));
    // polar is in rad
	fLGN.read(reinterpret_cast<char*>(&LGN_ecc[0]), nLGN*sizeof(Float));
    // ecc of center is in deg [0, nLGN), surround in rad [nLGN, 2*nLGN)
    for (Size i = 0; i <nLGN; i++) {
        assert(!isnan(LGN_ecc[i]));
    }
    auto transform_deg2rad = [deg2rad] (Float ecc) {return ecc*deg2rad;};
    transform(LGN_ecc.begin(), LGN_ecc.begin()+nLGN, LGN_ecc.begin(), transform_deg2rad);
    for (Size i = 0; i <nLGN; i++) {
        assert(!isnan(LGN_ecc[i]));
    }
	streampos eofLGN = fLGN.tellg();
	fLGN.seekg(0, fLGN.end);
	streampos true_eof = fLGN.tellg();
	if (eofLGN == true_eof || useNewLGN) { // Setup LGN here 
		cout << "initializing LGN spatial parameters...\n";
		fLGN.close();
    	// TODO: k, rx, ry, surround_x, surround_y or their distribution parameters readable from file
    	default_random_engine rGen_LGNsetup(seed);
    	seed++; // so that next random_engine use a different seed;
    	// lambda to return a function that generates random numbers from a given distribution
    	auto get_rand_from_gauss0 = [](default_random_engine &rGen, normal_distribution<Float> &dist, function<bool(Float)> &outOfBound) {
    	    function<float()> get_rand = [&rGen, &dist, &outOfBound] () {
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
    	auto get_excLowBound = [](Float thres) {
    	    function<bool(Float)> bound = [thres] (Float value) {
    	        return value <= thres;
    	    };
    	    return bound;
		};
    	auto positiveBound = get_excLowBound(0.0);
    	auto unityExcBound = get_excLowBound(1.0);
    	auto get_incLowBound = [](Float thres) {
    	    function<bool(Float)> bound = [thres] (Float value) {
    	        return value < thres;
    	    };
    	    return bound;
		};
    	auto unityIncBound = get_incLowBound(1.0);
    	auto nonNegativeBound = get_incLowBound(0.0);

    	// set test param for LGN subregion RF spatial kernel 
    	// Croner and Kaplan 1995
		// center
    	Float acuity = 0.03*deg2rad;
    	Float std = 0.01*deg2rad/1.349; // interquartile/1.349 = std 
    	normal_distribution<Float> norm(acuity, std);
    	generate(LGN_rw.begin(), LGN_rw.begin()+nLGN, get_rand_from_gauss0(rGen_LGNsetup, norm, positiveBound));

    	// surround
    	acuity = 0.18*deg2rad;
    	std = 0.07*deg2rad/1.349;
    	norm = normal_distribution<Float>(acuity,std);
    	generate(LGN_rw.begin()+nLGN, LGN_rw.end(), get_rand_from_gauss0(rGen_LGNsetup, norm, positiveBound));

    	// ry ~ rx circular sub RF
    	norm = normal_distribution<Float>(1.0, 1.0/30.0);
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
    	    Float intermediateEcc = LGN_ecc[i]+LGN_rh[i]*get_rand();
			Float eta = LGN_rw[i]*get_rand();
			orthPhiRotate3D_arc(LGN_polar[i], intermediateEcc, eta, LGN_polar[i+nLGN], LGN_ecc[i+nLGN]);
		}

    	// TODO: Group the same cone-specific types together for warp performance
    	// set test param for LGN subregion RF temporal kernel, also the K_c and K_s
    	// Benardete and Kaplan 1997 (source fitted by LGN_kernel.ipynb (mean)
		
        auto tspD_dist = [](Float n, Float tau) {
            Float tsp = (n-1)*tau;
            Float stride = n*tau/logrithm(n);
            Float mean = tsp + stride * 1.5;
            Float std = stride/6.0;
            return make_pair(mean, std);
        };
    	// ON-CENTER temporal mu and c.v.
    	                         // LGN_on = LGN_all * P_on/P_all
    	Float K_onC[2] = {29.13289963, 0.60*0.52/0.48}; //A spatially-integrated K
    	Float ratio_onC[2] = {0.56882181, 0.18*0.11/0.13}; //Hs

    	Float nR_onC[2] = {23.20564706, 0.28*0.18/0.23};
    	Float tauR_onC = 1.89874285; // controlled by tspR/(nR_onC-1)
        Float tspR_onC[2]; 
        tspR_onC[0] = (nR_onC[0]-1) * tauR_onC;
        tspR_onC[1] = 0.08*0.09/0.09; // NL*TauL, table2, table 5

    	Float tauD_onC[2] = {6.34054054, 0.52*0.45/0.59}; //tauS
        // nD'distribution will be controlled by tspD/tauD, with tspD from tspD_dist
    	//Float nD_onC[2] = {12.56210526, 0.28*0.18/0.23}; //NL

    	// OFF-CENTER temporal mu and c.v.
    	Float K_offC[2] = {22.71914498, 0.60*0.37/0.48}; // A
    	Float ratio_offC[2] = {0.60905210, 0.18*0.14/0.13}; //Hs

    	Float nR_offC[2] = {16.70023529, 0.28*0.20/0.23};
    	Float tauR_offC = 2.78125714;
		Float tspR_offC[2]; 	
        tspR_offC[0] = (nR_offC[0]-1) * tauR_offC;
        tspR_offC[1] = 0.08*0.09/0.09; // NL*TauL, table 5

    	Float tauD_offC[2] = {7.71891892, 0.52*0.62/0.59}; //tauS
        // nD'distribution will be controlled by tspD/tauD, with tspD from tspD_dist
    	//Float nD_offC[2] = {11.33052632, 0.28*0.20/0.23}; //NL

    	// delay from stimulus onset to LGN with mu and std.
    	Float delay_onC[2] = {13.45, sqrt(2.39*2.39+0.51*0.51)};
    	Float delay_offC[2] = {13.45, sqrt(2.39*2.39+0.61*0.61)};

    	// ON-SURROUND temporal mu and c.v.
    	Float K_onS[2] = {23.17378917,  1.45*1.53/1.64}; // A
    	Float ratio_onS[2] = {0.36090931, 0.57*0.74/0.64}; //Hs

    	Float nR_onS[2] = {50.29602888, 0.35*0.12/0.38};
    	Float tauR_onS = 0.90455076;
        Float tspR_onS[2]; 
        tspR_onS[0] = (nR_onS[0]-1) * tauR_onS;
        tspR_onS[1] = 0.09*0.10/0.08; // NL*TauL, table2, table 5

    	//Float nD_onS[2] = {22.54098361, 0.35*0.12/0.38}; //NL
    	Float tauD_onS[2] = {3.05084745, 0.90*1.08/0.94}; //tauS

    	// OFF-surround temporal mu and c.v.
    	Float K_offS[2] = {12.53276353, 1.45*1.84/1.64}; // A
    	Float ratio_offS[2] = {0.33004402, 0.57*0.59/0.64}; //Hs

    	Float nR_offS[2] = {26.23465704, 0.35*0.53/0.38};
    	Float tauR_offS = 1.83570595;
        Float tspR_offS[2]; 
        tspR_offS[0] = (nR_offS[0]-1) * tauR_offS;
        tspR_offS[1] = 0.09*0.06/0.08; // NL*TauL, table2, table 5

    	//Float nD_offS[2] = {8.29508197, 0.35*0.53/0.38}; //NL
    	Float tauD_offS[2] = {9.66101695, 0.90*0.71/0.94}; //tauS

    	// delay from stimulus onset to LGN with mu and std.
    	Float delay_onS[2] = {22.06611570, sqrt(1.48*1.48+0.51*0.51)};
    	Float delay_offS[2] = {26.68388430, sqrt(1.48*1.48+0.61*0.61)};
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
    	// tauR and nR anti-correlates for stable peak time
    	Float rho_Kc_Ks = 0.5;
    	Float rho_Kc_Ks_comp = sqrt(1.0-rho_Kc_Ks*rho_Kc_Ks);
    	Float rho_tau_n = -0.5;
    	Float rho_tau_n_comp = sqrt(1.0-rho_tau_n*rho_tau_n);
    	// reset norm to standard normal distribution
    	norm = normal_distribution<Float>(0.0, 1.0);

		/* unused, boundary data not available
    	auto get_bounds = [](Float floor, Float ceiling) {
    	    function<bool(Float)> outOfBound = [floor, ceiling] (Float value) {
    	        return (floor <= value && value <= ceiling);
    	    };
    	    return outOfBound;
		}; */
		cout << "initializing LGN temporal parameters...\n";
		Float pTspD[2];
		Float tspR, tspD;
    	for (unsigned int i=0; i<nLGN; i++) {
    	    // using median from table 2,3  (on,off)/all * table 5,6 with matching c.v.# Benardete and Kaplan 1997
    	    // fit with difference of exponentials in LGN_kernel.ipynb
    	    // cones' differences are not differentiated
    	    switch (LGNtype[i]) {
    	        // on-center, off-surround
    	        case InputType::MonLoff: case InputType::LonMoff:
    	            // k
    				tie(LGN_k[i], LGN_k[i+nLGN]) = get_rands_from_correlated_gauss(K_onC, K_offS, rho_Kc_Ks, rho_Kc_Ks_comp, rGen_LGNsetup, rGen_LGNsetup, positiveBound, nonNegativeBound);
					LGN_k[i+nLGN] *= -1; //off-surround !!!IMPORTANT sign change here

    	            // centers' tau, n 
    	            nR[i] = get_rand_from_gauss(nR_onC, rGen_LGNsetup, unityExcBound);
					tspR = get_rand_from_gauss(tspR_onC, rGen_LGNsetup, positiveBound);
					tauR[i] = tspR/(nR[i]-1);

        			tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i], tauR[i]);
					tauD[i] = get_rand_from_gauss(tauD_onC, rGen_LGNsetup, positiveBound);
					tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
    	            nD[i] = tspD/tauD[i];

    	            // surround' tau, n 
    	            nR[i+nLGN] = get_rand_from_gauss(nR_offS, rGen_LGNsetup, unityExcBound);
					tspR = get_rand_from_gauss(tspR_offS, rGen_LGNsetup, positiveBound);
					tauR[i+nLGN] = tspR/(nR[i+nLGN]-1);

        			tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i+nLGN], tauR[i+nLGN]);
					tauD[i+nLGN] = get_rand_from_gauss(tauD_offS, rGen_LGNsetup, positiveBound);
					tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
    	            nD[i+nLGN] = tspD/tauD[i+nLGN];

    	            // ratio
    	            ratio[i] = get_rand_from_gauss(ratio_onC, rGen_LGNsetup, nonNegativeBound);
    	            ratio[i+nLGN] = get_rand_from_gauss(ratio_offS, rGen_LGNsetup, nonNegativeBound);
    	            // delay
    	            delay[i] = get_rand_from_gauss(delay_onC, rGen_LGNsetup, nonNegativeBound);
    	            delay[i+nLGN] = get_rand_from_gauss(delay_offS, rGen_LGNsetup, nonNegativeBound);
    	            break;
    	        // off-centers
    	        case InputType::MoffLon: case InputType::LoffMon:
    	            // k
    				tie(LGN_k[i], LGN_k[i+nLGN]) = get_rands_from_correlated_gauss(K_offC, K_onS, rho_Kc_Ks, rho_Kc_Ks_comp, rGen_LGNsetup, rGen_LGNsetup, positiveBound, nonNegativeBound);
					LGN_k[i] *= -1; //off-center

    	            // centers' tau, n 
    	            nR[i] = get_rand_from_gauss(nR_offC, rGen_LGNsetup, unityExcBound);
					tspR = get_rand_from_gauss(tspR_offC, rGen_LGNsetup, positiveBound);
					tauR[i] = tspR/(nR[i]-1);

        			tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i], tauR[i]);
					tauD[i] = get_rand_from_gauss(tauD_offC, rGen_LGNsetup, positiveBound);
					tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
    	            nD[i] = tspD/tauD[i];

    	            // surround' tau, n 
    	            nR[i+nLGN] = get_rand_from_gauss(nR_onS, rGen_LGNsetup, unityExcBound);
					tspR = get_rand_from_gauss(tspR_onS, rGen_LGNsetup, positiveBound);
					tauR[i+nLGN] = tspR/(nR[i+nLGN]-1);

        			tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i+nLGN], tauR[i+nLGN]);
					tauD[i+nLGN] = get_rand_from_gauss(tauD_onS, rGen_LGNsetup, positiveBound);
					tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
    	            nD[i+nLGN] = tspD/tauD[i+nLGN];

    	            // ratio
    	            ratio[i] = get_rand_from_gauss(ratio_offC, rGen_LGNsetup, nonNegativeBound);
    	            ratio[i+nLGN] = get_rand_from_gauss(ratio_onS, rGen_LGNsetup, nonNegativeBound);
    	            // delay
    	            delay[i] = get_rand_from_gauss(delay_offC, rGen_LGNsetup, nonNegativeBound);
    	            delay[i+nLGN] = get_rand_from_gauss(delay_onS, rGen_LGNsetup, nonNegativeBound);
    	            break;
   		        default: throw("There's no implementation of such RF for parvo LGN");
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
   		        default: throw("There's no implementation of such RF for parvo LGN");
			}
    	    // non-linearity
    	    Float log_mean, log_std;
    	    std::tie(log_mean, log_std) = lognstats<Float>(0.1, 1);
    	    spont[i] = exp(log_mean + norm(rGen_LGNsetup) * log_std); // non-zero so no lambda for boundary check and stuff
    	    // NOTE: proportion between L and M, significantly overlap in cone response curve, not implemented to calculate max_convol
    	    covariant[i] = 0.53753461391295254; 
			//cout << "\rLGN #" << i;
    	}
		//cout << "\n";

        norm = normal_distribution<Float>(0.25, 0.1);
    	auto get_c50 = get_rand_from_gauss0(rGen_LGNsetup, norm, nonNegativeBound);
    	generate(c50.begin(), c50.end(), get_c50);

        norm = normal_distribution<Float>(10.0, 1.0);
    	auto get_sharpness = get_rand_from_gauss0(rGen_LGNsetup, norm, unityIncBound);
    	generate(sharpness.begin(), sharpness.end(), get_sharpness);

    	fLGN.open(LGN_filename, fstream::out | fstream::app | fstream::binary);
    	// append to LGN_polar and LGN_ecc positions
		fLGN.seekp(eofLGN);
		// surround origin is changed
    	fLGN.write((char*)&LGN_polar[nLGN], nLGN*sizeof(Float));
    	fLGN.write((char*)&LGN_ecc[nLGN], nLGN*sizeof(Float));
		// new props
    	fLGN.write((char*)&LGN_rw[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&LGN_rh[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&LGN_orient[0], 2*nLGN*sizeof(Float));

    	fLGN.write((char*)&LGN_k[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&ratio[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&LGN_k[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&ratio[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&tauR[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&tauD[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&nR[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&nD[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&delay[0], 2*nLGN*sizeof(Float));

    	fLGN.write((char*)&spont[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&c50[0], 2*nLGN*sizeof(Float));
    	fLGN.write((char*)&sharpness[0], 2*nLGN*sizeof(Float));

    	fLGN.write((char*)&coneType[0], 2*nLGN*sizeof(PosInt));
    	fLGN.write((char*)&covariant[0], nLGN*sizeof(Float));
    	fLGN.close();
	} else { // Use old setup
		cout << "reading LGN parameters\n";
		fLGN.seekg(eofLGN);
		fLGN.read(reinterpret_cast<char*>(&LGN_polar[nLGN]), nLGN*sizeof(Float));
		fLGN.read(reinterpret_cast<char*>(&LGN_ecc[nLGN]), nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&LGN_rw[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&LGN_rh[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&LGN_orient[0]), 2*nLGN*sizeof(Float));

    	fLGN.read(reinterpret_cast<char*>(&LGN_k[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&ratio[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&LGN_k[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&ratio[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&tauR[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&tauD[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&nR[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&nD[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&delay[0]), 2*nLGN*sizeof(Float));

    	fLGN.read(reinterpret_cast<char*>(&spont[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&c50[0]), 2*nLGN*sizeof(Float));
    	fLGN.read(reinterpret_cast<char*>(&sharpness[0]), 2*nLGN*sizeof(Float));

    	fLGN.read(reinterpret_cast<char*>(&coneType[0]), 2*nLGN*sizeof(PosInt));
    	fLGN.read(reinterpret_cast<char*>(&covariant[0]), nLGN*sizeof(Float));
		fLGN.close();	
	}

    hSpatial_component hSpat(nLGN, 2, LGN_polar, LGN_rw, LGN_ecc, LGN_rh, LGN_orient, LGN_k);
    hTemporal_component hTemp(nLGN, 2, tauR, tauD, delay, ratio, nR, nD);
    hStatic_nonlinear hStat(nLGN, spont, c50, sharpness);

    hLGN_parameter hLGN(nLGN, 2, hSpat, hTemp, hStat, coneType, covariant);

	LGN_parameter dLGN(hLGN);
    hLGN.freeMem();

	Size nLGN_block, nLGN_thread; // for LGN_nonlinear
	nLGN_block = (nLGN + blockSize - 1)/blockSize;
	nLGN_thread = blockSize;
	cout << "LGN initialized\n";
    // finish LGN setup

	vector<vector<Size>> LGN_V1_ID = read_listOfList<Size>(LGN_V1_ID_filename, false);
	vector<vector<Float>> LGN_V1_s = read_listOfList<Float>(LGN_V1_s_filename, false);

    Size nV1;

	// read nV1
    fV1.open(V1_filename, fstream::in | fstream::binary);
    if (!fV1) {
		cout << "Cannot open or find " << V1_filename <<" to read in V1 properties.\n";
		return EXIT_FAILURE;
    } else {
        fV1.read(reinterpret_cast<char*>(&nV1), sizeof(Size));
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
    	fV1.read(reinterpret_cast<char*>(&V1_x[0]), nV1 * sizeof(Float));
		fV1.read(reinterpret_cast<char*>(&V1_y[0]), nV1 * sizeof(Float));
    	fV1.read(reinterpret_cast<char*>(&a[0]), nV1 * sizeof(Float));
    	fV1.read(reinterpret_cast<char*>(&baRatio[0]), nV1 * sizeof(Float));
		fV1.read(reinterpret_cast<char*>(&sfreq[0]), nV1 * sizeof(Float));
		fV1.read(reinterpret_cast<char*>(&theta[0]), nV1 * sizeof(Float));
		fV1.read(reinterpret_cast<char*>(&phase[0]), nV1 * sizeof(Float));
		fV1.read(reinterpret_cast<char*>(&amp[0]), nV1 * sizeof(Float));
		fV1.read(reinterpret_cast<char*>(&sig[0]), nV1 * sizeof(Float));
		fV1.read(reinterpret_cast<char*>(&V1Type[0]), nV1 * sizeof(RFtype_t));
		fV1.read(reinterpret_cast<char*>(&RefType[0]), nV1 * sizeof(OutputType_t));
    	fV1.close();
	}

	{ // output file tests
    	fLGN_fr.open(LGN_fr_filename, fstream::out | fstream::binary);
    	if (!fLGN_fr) {
			cout << "Cannot open or find " << LGN_fr_filename <<" for LGN firing rate output\n";
			return EXIT_FAILURE;
    	} else {
            fLGN_fr.write((char*)&nt, sizeof(Size));
            fLGN_fr.write((char*)&nLGN, sizeof(Size));
        }

    	fmax_convol.open(max_convol_filename, fstream::out | fstream::binary);
    	if (!fmax_convol) {
			cout << "Cannot open or find " << max_convol_filename <<" for maximal LGN convolution.\n";
			return EXIT_FAILURE;
    	} else {
            fLGN_convol.write((char*)&nLGN, sizeof(Size));
        }

    	// test output 
		if (checkConvol) {
    		fLGN_convol.open(LGN_convol_filename, fstream::out | fstream::binary);
    		if (!fLGN_convol) {
				cout << "Cannot open or find " << LGN_convol_filename <<" for output LGN convolutions.\n";
				return EXIT_FAILURE;
    		} else {
                fLGN_convol.write((char*)&nt, sizeof(Size));
                fLGN_convol.write((char*)&nLGN, sizeof(Size));
            }
		}

		if (testStorage) {
    		fStorage.open(storage_filename, fstream::out | fstream::binary);
    		if (!fStorage) {
				cout << "Cannot open or find " << storage_filename <<" for storage check.\n";
				return EXIT_FAILURE;
    		}
		}
        if (testLuminanceAd) {
            fLuminanceAd.open(luminanceAd_filename, fstream::out | fstream::binary);
            if (!fLuminanceAd) {
	        	cout << "Cannot open or find " << storage_filename <<" for storage check.\n";
	        	return EXIT_FAILURE;
            } else {
                Size sizeTmp = 2;
                fLuminanceAd.write((char*)&nt, sizeof(Size));
                fLuminanceAd.write((char*)&sizeTmp, sizeof(Size));
                fLuminanceAd.write((char*)&nLGN, sizeof(Size));
            }
        }
		cout << "output file check, done\n";
	}

    bool simpleContrast = true; // TODO : implement this for comparison no cone adaptation if set to true
    PosInt stepRate = round(1000/dt);
    if (round(1000/dt) != 1000/dt) {
        cout << "stepRate = " << 1000/dt << "Hz, but it has to be a integer.\n";
        return EXIT_FAILURE;
    }
    if (frameRate > stepRate) {
        cout << "stepRate cannot be smaller than frameRate, 1000/dt.\n";
        return EXIT_FAILURE;
    }

    Size nRetrace = static_cast<Size>(round(tau/dt));
    if (round(tau/dt) != tau/dt) {
        cout << "tau should be divisible by dt.\n"; 
        return EXIT_FAILURE;
    }
    if (nRetrace % nKernelSample != 0) {
        cout << "tau in #dt should be divisible by nKernelSample.\n"; 
        return EXIT_FAILURE;
    }
    Float kernelSampleRate = static_cast<PosInt>(nKernelSample/tau*1000);

    PosInt kernelSampleInterval = nRetrace/nKernelSample;
    Float kernelSampleT0 = (kernelSampleInterval/2)*dt; 
    Float kernelSampleDt = kernelSampleInterval*dt;
    // |--*--|--*--|--*--|, e.g., nKernelSample = 3->*

    printf("temporal kernel retraces %f ms, samples %u points, sample rate = %f Hz\n", tau, nKernelSample, kernelSampleRate);

    PosInt maxFrame = static_cast<PosInt>(ceil(tau/1000 * frameRate)) + 1; 
	// max frame need to be stored in texture for temporal convolution with the LGN kernel.
    //  |---|--------|--|
    
    printf("temporal kernel retrace (tau): %f ms, frame rate: %d Hz needs at most %u frames\n", tau, frameRate, maxFrame);
	Float tPerFrame = 1000.0f / frameRate; //ms
    printf("~ %f plusminus 1 kernel sample per frame\n", tPerFrame/kernelSampleDt);
    printf("=== texture memory required: %dx%dx%d = %fMB\n", maxFrame,width,height, maxFrame*width*height*sizeof(float)/1024.0/1024.0);

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
    printf("%u exact phases in [0,1]: ", denorm);
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
    }
    // i frames' length in steps = exact_it[i] + exact_norm[i]/denorm
    assert((stepRate*denorm) % frameRate == 0);
    unsigned int co_product = (stepRate*denorm)/frameRate; 
    // the number of non-zero minimum steps to meet frameLength * denorm with 0 phase

    unsigned int nChannel = 3; // L, M, S
    // one cudaArray per channel
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArr_L;
    cudaArray *cuArr_M;
    cudaArray *cuArr_S;
    // allocate cudaArrays on the device
    checkCudaErrors(cudaMalloc3DArray(&cuArr_L, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_M, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_S, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));

    // set params for layerd texture memory
    init_layer(L_retinaConSig);
    init_layer(M_retinaConSig);
    init_layer(S_retinaConSig);

    // bind texture to cudaArrays
    checkCudaErrors(cudaBindTextureToArray(L_retinaConSig,  cuArr_L, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(M_retinaConSig,  cuArr_M, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(S_retinaConSig,  cuArr_S, channelDesc));

    float* LMS = new float[nPixelPerFrame*sizeof(float)*nChannel]; // memory head

    // LMS frame intensity array of [maxFrame] frames from (t-tau) -> t on hos 
    float* __restrict__ L;
    float* __restrict__ M;
    float* __restrict__ S;

    L = LMS;
    M = L + nPixelPerFrame;
    S = M + nPixelPerFrame;

    Float* arrayOfnLGN = new Float[nLGN]; // versatile host array of length nLGN to hold temporary data to be written to output
    Float* LGN_fr = new Float[nLGN]; // host array for LGN firing rate
    Float* __restrict__ dLGN_fr;
    Float* __restrict__ current_convol;
    Float* __restrict__ max_convol;
    checkCudaErrors(cudaMalloc((void **) &dLGN_fr, nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &current_convol, nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &max_convol, nLGN*sizeof(Float)));
    checkCudaErrors(cudaMemset(max_convol, 0, nLGN*sizeof(Float)));
	
	size_t maxStorageSize;
	if (2*nSample > nKernelSample) {
		maxStorageSize = 2*nType*nSample*nLGN;
	} else {
		maxStorageSize = nType*nKernelSample*nLGN;
	}
	Float* arrayOf_2nType_nSample_nLGN = new Float[maxStorageSize]; // host array for storages
    Float* decayIn;
    Float* lastF;
    Float* TW_storage;
    Float* SW_storage;
    float* SC_storage;
    checkCudaErrors(cudaMalloc((void **) &decayIn, nType*nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &lastF, nType*nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &TW_storage, nLGN*nType*nKernelSample*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &SW_storage, nLGN*nType*nSample*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &SC_storage, 2*nLGN*nType*nSample*sizeof(float)));

    checkCudaErrors(cudaMemset(decayIn, 0, nType*nLGN*sizeof(Float)));
    checkCudaErrors(cudaMemset(lastF, 0, nType*nLGN*sizeof(Float)));
    // initialize average to normalized mean luminance
    cudaStream_t s0, s1, s2;
    checkCudaErrors(cudaStreamCreate(&s0));
    checkCudaErrors(cudaStreamCreate(&s1));
    checkCudaErrors(cudaStreamCreate(&s2));

    dim3 initBlock(blockSize, 1, 1);
    dim3 initGrid((nPixelPerFrame + blockSize-1) / blockSize, 1, 1);

    {// initialize texture to 0
        float* tLMS;
        float* __restrict__ tL;
        float* __restrict__ tM;
        float* __restrict__ tS;
        checkCudaErrors(cudaMalloc((void **) &tLMS, nPixelPerFrame*sizeof(float)*3*maxFrame));
        tL = tLMS;
        tM = tL + nPixelPerFrame*maxFrame;
        tS = tM + nPixelPerFrame*maxFrame;
        checkCudaErrors(cudaMemset(tLMS, init_luminance, nPixelPerFrame*sizeof(float)*3*maxFrame));
        prep_sample(0, width, height, tL, tM, tS, cuArr_L, cuArr_M, cuArr_S, maxFrame, cudaMemcpyDeviceToDevice);
        checkCudaErrors(cudaFree(tLMS));
    }

    cudaEvent_t i0, i1, i2;

    checkCudaErrors(cudaEventCreate(&i0));
    checkCudaErrors(cudaEventCreate(&i1));
    checkCudaErrors(cudaEventCreate(&i2));

    dim3 convolBlock(nSpatialSample1D, nSpatialSample1D, 1);
    dim3 convolGrid(nLGN, 2, 1);
    
	cout << "cuda memory, set\n";

	cout << "store<<<" << convolGrid.x  << "x" << convolGrid.y  << "x" << convolGrid.z << ", " << convolBlock.x  << "x" << convolBlock.y  << "x" << convolBlock.z << ">>>\n";
    // store spatial and temporal weights determine the maximums of LGN kernel convolutions
    store<<<convolGrid, convolBlock>>>(
            max_convol,
            *dLGN.temporal,
            TW_storage,
            nKernelSample,
            kernelSampleDt,
            kernelSampleT0,
            *dLGN.spatial,
            SW_storage,
            SC_storage,
			nLGN_L,
		 	L_x0,
		 	L_y0,
		 	R_x0,
			R_y0,
			normViewDistance,
            nsig);
	getLastCudaError("store failed");
    checkCudaErrors(cudaMemcpy(arrayOfnLGN, max_convol, nLGN*sizeof(Float), cudaMemcpyDeviceToHost));
    fmax_convol.write((char*)arrayOfnLGN, nLGN*sizeof(Float));
    fmax_convol.close();
	if (testStorage) {// storage check output
        fStorage.write((char*)&nLGN, sizeof(Size));
        Size _nType = static_cast<Size>(nType);
        fStorage.write((char*)&_nType, sizeof(Size));
        fStorage.write((char*)&nKernelSample, sizeof(Size));
        fStorage.write((char*)&nSample, sizeof(Size));

    	checkCudaErrors(cudaMemcpy(arrayOf_2nType_nSample_nLGN, TW_storage, nLGN*nType*nKernelSample*sizeof(Float), cudaMemcpyDeviceToHost));
		fStorage.write((char*)arrayOf_2nType_nSample_nLGN, nType*nKernelSample*nLGN*sizeof(Float));

    	checkCudaErrors(cudaMemcpy(arrayOf_2nType_nSample_nLGN, SW_storage, nLGN*nType*nSample*sizeof(Float), cudaMemcpyDeviceToHost));
		fStorage.write((char*)arrayOf_2nType_nSample_nLGN, nLGN*nType*nSample*sizeof(Float));

    	checkCudaErrors(cudaMemcpy(arrayOf_2nType_nSample_nLGN, SC_storage, 2*nLGN*nType*nSample*sizeof(float), cudaMemcpyDeviceToHost));
		fStorage.write((char*)arrayOf_2nType_nSample_nLGN, 2*nLGN*nType*nSample*sizeof(float));

		fStorage.close();
	}
    // calc LGN firing rate at the end of current dt
    unsigned int currentFrame = 0; // current frame number from stimulus
    unsigned int iFrame = 0; //latest frame inserted into the dL dM dS,  initialization not necessary
    unsigned int iPhase = 0;
    Float framePhase = tPerFrame;
    PosInt oldFrame = currentFrame;
	cout << "convol weights stored\n";
	cout << "simulation starts: \n";
    for (unsigned int it = 0; it < nt; it++) {
        Float t = it*dt;
        // next frame comes between (t, t+dt), read and store frame to texture memory
        if (it+1 > (currentFrame/denorm)*co_product + exact_it[iPhase]) {
            iFrame = currentFrame % maxFrame;
			//cout << "processing frame #" << iFrame << "\n";
            // TODO: realtime video stimulus control
            if (fStimulus) {
                fStimulus.read(reinterpret_cast<char*>(LMS), nChannel*nPixelPerFrame*sizeof(float));
                //printf("frame #%i is read LMS\n", currentFrame);
                streampos current_fpos = fStimulus.tellg();
                if (current_fpos == eofStimulus) { // if at the end of input file loop back to the beginning of frame data
                    fStimulus.seekg(sofStimulus);
                    cout << "next frame loops\n";
                }
            } else {
                cout << "stimulus format corrupted\n";
                return EXIT_FAILURE;
            }
            //cp to texture mem in device
            prep_sample(iFrame, width, height, L, M, S, cuArr_L, cuArr_M, cuArr_S, 1, cudaMemcpyHostToDevice);
            /*{ // DEBUG printout
                printf("frame #%i prepared at t = %f, in (%f,%f) ~ %.3f%%.\n", currentFrame, currentFrame*tPerFrame, t, t+dt, exact_norm[iPhase]*100.0f/denorm);
            }*/
            currentFrame++;
            iPhase = (iPhase + 1) % denorm;
        }

        framePhase -= dt;
        if (framePhase < 0) framePhase += tPerFrame;
        framePhase = fmod(framePhase, tPerFrame);
		/* if it < nRetrace, padded zero-valued frames for t<0
            -->|        |<-- framePhase
            |--|------frame------|
            |-----|-----|-----|-----|-----|
            jt-2, jt-1, jt ...nRetrace... it
        */// perform kernel convolution with built-in texture interpolation
        convolGrid.x = nLGN;
        convolGrid.y = 2;
		//cout << "LGN_convol_c1s<<<" << convolGrid.x  << "x" << convolGrid.y  << "x" << convolGrid.z << ", " << convolBlock.x  << "x" << convolBlock.y  << "x" << convolBlock.z << ", " << sizeof(Float)*2*convolBlock.x*convolBlock.y << ">>>\n";
        LGN_convol_c1s<<<convolGrid, convolBlock, sizeof(Float)*nSample>>>(
        //LGN_convol_c1s<<<convolGrid, convolBlock>>>(
                decayIn,
                lastF,
                SW_storage,
                SC_storage,
                TW_storage,
                current_convol,
                dLGN.coneType,
                *dLGN.spatial,
                nsig,
				nLGN_L,
		 		L_x0,
		 		L_y0,
		 		R_x0,
				R_y0,
				normViewDistance,
                iFrame,
				maxFrame,
				tPerFrame,
                framePhase,
                Itau,
                kernelSampleDt,
                nKernelSample,
                dt);

		getLastCudaError("LGN_convol_c1s failed");
        checkCudaErrors(cudaMemcpy(arrayOfnLGN, current_convol, nLGN*sizeof(Float), cudaMemcpyDeviceToHost));
        fLGN_convol.write((char*)arrayOfnLGN, nLGN*sizeof(Float));
        if (testLuminanceAd) { // only test the center luminance
            checkCudaErrors(cudaMemcpy(arrayOfnLGN, decayIn, nLGN*sizeof(Float), cudaMemcpyDeviceToHost));
            fLuminanceAd.write((char*)arrayOfnLGN, nLGN*sizeof(Float));
            checkCudaErrors(cudaMemcpy(arrayOfnLGN, lastF, nLGN*sizeof(Float), cudaMemcpyDeviceToHost));
            fLuminanceAd.write((char*)arrayOfnLGN, nLGN*sizeof(Float));
        }

		// generate LGN fr with logistic function
        LGN_nonlinear<<<nLGN_block, nLGN_thread>>>(nLGN, *dLGN.logistic, max_convol, current_convol, dLGN_fr);
		getLastCudaError("LGN_nonlinear failed");
        checkCudaErrors(cudaMemcpy(LGN_fr, dLGN_fr, nLGN*sizeof(Float), cudaMemcpyDeviceToHost));
        fLGN_fr.write((char*)LGN_fr, nLGN*sizeof(Float));
        if (currentFrame > oldFrame || it == nt-1) {
	        printf("\r@t = %f -> %f simulated, frame %d#%d-%d, %.1f%%", t, t+dt, currentFrame/nFrame, currentFrame%nFrame, nFrame, 100*static_cast<float>(it+1)/nt);
            oldFrame = currentFrame;
        }
    }
    cout << "simulation done.\n";

    { // clean-up
        checkCudaErrors(cudaDeviceSynchronize());
        delete []LGN_fr;
        delete []arrayOfnLGN;
		delete []arrayOf_2nType_nSample_nLGN;
        delete []exact_norm;
        delete []exact_it;
        
	    fLGN_fr.close();
        fLGN_convol.close();
        fmax_convol.close();
        fStorage.close();
        fLuminanceAd.close();

        dLGN.freeMem();
	    checkCudaErrors(cudaStreamDestroy(s0));
        checkCudaErrors(cudaStreamDestroy(s1));
        checkCudaErrors(cudaStreamDestroy(s2));
        checkCudaErrors(cudaFree(dLGN_fr));
        checkCudaErrors(cudaFree(current_convol));
        checkCudaErrors(cudaFree(max_convol));
        checkCudaErrors(cudaFreeArray(cuArr_L));
        checkCudaErrors(cudaFreeArray(cuArr_M));
        checkCudaErrors(cudaFreeArray(cuArr_S));
        checkCudaErrors(cudaDeviceReset());
        cout << "memory trace cleaned\n";
    }
    return EXIT_SUCCESS;
}
