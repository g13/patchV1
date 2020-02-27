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
#include "stats.cuh"
#include "util/util.h"
#include "preprocess/RFtype.h"
#include "global.h"
#include "CUDA_MACRO.h"

// the retinal discrete x, y as cone receptors id
inline void init_layer(texture<float, cudaTextureType2DLayered> &layer) {
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
    // if a cudaArray is involved width is element not byte size
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
    
	bool storeSpatial = true;
    Float dt; // in ms, better in fractions of binary 
	bool useNewLGN;
	bool saveLGN_fr;
	bool saveLGN_gallery;
	bool saveOutputB4V1;
    Size nChunk;
	Size matConcurrency;
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
		("dt", po::value<Float>(&dt)->default_value(0.0625), "simulatoin time step in ms") 
		("nChunk,n", po::value<Size>(&nChunk)->default_value(10),"simulation in chunks, empricial")
		("matConcurrency,n", po::value<Size>(&matConcurrency)->default_value(2),"sum presynaptic inputs from connection matrices in parallel, depends on the availability of device memory")
		("speedOfThought", po::value<Float>(&speedOfThought)->default_value(1.0), "velocity of conduction, mm/ms") 
		("nt", po::value<Size>(&nt)->default_value(8000), "total simulatoin time in units of time step")
		("nSpatialSample1D", po::value<SmallSize>(&nSpatialSample1D)->default_value(warpSize), "number of samples per x,y direction for a LGN spatial RF")
		("tau", po::value<Float>(&tau)->default_value(250.0), "the backward time interval that a LGN temporal RF should cover")
		("Itau", po::value<Float>(&Itau)->default_value(300.0), "the light intensity adaptation time-scale of a cone")
		("nKernelSample", po::value<Size>(&nKernelSample)->default_value(500), "number of samples per temporal kernel")
		("frameRate", po::value<PosInt>(&frameRate)->default_value(50), "frame rate of the input stimulus")
		("saveLGN_fr", po::value<bool>(&saveLGN_fr)->default_value(true),"write LGN firing rates to disk, specify filename through LGN_fr_filename")
		("saveLGN_gallery", po::value<bool>(&saveLGN_gallery)->default_value(true), "check convolution kernels and maximum convolution values, write data to disk, specify filename through LGN_gallery_filename")
		("saveOutputB4V1", po::value<bool>(&saveOutputB4V1)->default_value(true), "check adapted luminance values, write data to disk, specify filename through outputB4V1_filename")
		("useNewLGN", po::value<bool>(&useNewLGN)->default_value(true), "regenerate the a new ensemble of LGN parameters according to their distribution");

    // files
	string stimulus_filename;
    string V1_RF_filename, V1_feature_filename;
    string V1_vec_filename, V1_delayMat_filename, V1_conMat_filename;
    string fLGN_surface_filename;
    string LGN_filename, LGN_vpos_filename, LGN_V1_s_filename, LGN_V1_ID_filename; // inputs
    string LGN_fr_filename; // outputs
	string LGN_convol_filename, LGN_gallery_filename, outputB4V1_filename;
	top_opt.add_options()
		("fStimulus", po::value<string>(&stimulus_filename)->default_value("stimulus.bin"),"file that stores LGN firing rates, array of size (nframes,width,height,3)")
		("fLGN_vpos", po::value<string>(&LGN_vpos_filename)->default_value("LGN_vpos.bin"),"file that stores LGN neurons information")
		("fLGN", po::value<string>(&LGN_filename)->default_value("LGN.bin"),"file that stores all the information of LGN neurons")
		("fLGN_fr", po::value<string>(&LGN_fr_filename)->default_value("LGN_fr.bin"),"file stores LGN firing rates")
		("fLGN_V1_ID", po::value<string>(&LGN_V1_ID_filename)->default_value("LGN_V1_idList.bin"),"file stores LGN to V1 connections")
		("fLGN_V1_s", po::value<string>(&LGN_V1_s_filename)->default_value("LGN_V1_sList.bin"),"file stores LGN to V1 connection strengths")
		("fLGN_surface", po::value<string>(&fLGN_surface_filename)->default_value("LGN_surface.bin"),"file stores LGN position on surface memory")
        ("fV1_RF", po::value<string>(&V1_RF_filename)->default_value("V1_RF.bin"), "file that stores V1 RF properties, (orientation info is in fV1_feature)")
        ("fV1_feature", po::value<string>(&V1_feature_filename)->default_value("V1_feature.bin"), "file to read spatially predetermined functional features of neurons")
        ("fV1_conMat", po::value<string>(&V1_conMat_filename)->default_value("V1_conMat.bin"), "file that stores V1 to V1 connection within the neighboring blocks")
        ("fV1_delayMat", po::value<string>(&V1_delayMat_filename)->default_value("V1_delayMat.bin"), "file that stores V1 to V1 transmission delay within the neighboring blocks")
        ("fV1_vec", po::value<string>(&V1_vec_filename)->default_value("V1_vec.bin"), "file that stores V1 to V1 connection ID, strength and transmission delay outside the neighboring blocks")
        ("fNeighborBlock", po::value<string>(&neighborBlock_filename)->default_value("V1_vec.bin"), "file that stores V1 to V1 connection ID, strength and transmission delay outside the neighboring blocks")
		("fOutputB4V1", po::value<string>(&outputB4V1_filename)->default_value("outputB4V1.bin"),"file that stores luminance values, contrasts, LGN convolution and their firing rates") // TEST 
		("fLGN_gallery", po::value<string>(&LGN_gallery_filename)->default_value("LGN_gallery_filename.bin"),"file that stores spatial and temporal convolution parameters") // TEST 
	
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

    // from the retina facing out
	ifstream fStimulus; // ext. inputs
    ifstream fV1_RF, fV1_feature; // V1 related
    ifstream fLGN_vpos; // LGN VF pos 
    fstream fLGN; // LGN properties
	ofstream fLGN_fr; // outputs
	ofstream fLGN_gallery, fOutputB4V1;
    ofstream fRawData;

    Float init_L, init_M, init_S;
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
    fLGN_vpos.open(LGN_vpos_filename, fstream::in | fstream::binary);
    if (!fLGN_vpos) {
		cout << "Cannot open or find " << LGN_vpos_filename <<" to read in LGN properties.\n";
        return EXIT_FAILURE;
    } else {
	    fLGN_vpos.read(reinterpret_cast<char*>(&nLGN_L), sizeof(Size));
	    fLGN_vpos.read(reinterpret_cast<char*>(&nLGN_R), sizeof(Size));
	    fLGN_vpos.read(reinterpret_cast<char*>(&max_ecc), sizeof(Float)); // in rad
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

    Size nType = 2;
    Size nSample = nSpatialSample1D * nSpatialSample1D;
    if (nKernelSample == 0) {
        cout << "kernel sampling points: " << nKernelSample << " must be positive integer.\n"; 
        return EXIT_FAILURE;
    }
	// TODO: reduce memory usage if no distribution
    printf("=== temporal storage memory required: %dx%dx%d = %fMB\n", nKernelSample,nLGN,nType,nKernelSample*nLGN*nType*sizeof(Float)/1024.0/1024.0);
    printf("=== spatial storage memory required: %dx%dx%d = %fMB\n", nSample, nLGN, nType, nSample*nType*nLGN*sizeof(Float)/1024.0/1024.0);
    printf("=== texture coord storage memory required: %dx%dx%d = %fMB\n", nSample, nLGN, nType, 2*nSample*nType*nLGN*sizeof(float)/1024.0/1024.0);

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
    vector<Float> sx(nLGN);
    vector<Float> sy(nLGN);
    Size nsx, nsy;
	fLGN_vpos.read(reinterpret_cast<char*>(&nsx), sizeof(Size));
	fLGN_vpos.read(reinterpret_cast<char*>(&nsy), sizeof(Size));
	fLGN_vpos.read(reinterpret_cast<char*>(&sx[0]), nLGN*sizeof(Float));
	fLGN_vpos.read(reinterpret_cast<char*>(&sy[0]), nLGN*sizeof(Float));

	fLGN_vpos.read(reinterpret_cast<char*>(&LGNtype[0]), nLGN*sizeof(InputType_t));
	if (useNewLGN) { // Setup LGN here 
		cout << "initializing LGN spatial parameters...\n";
    	// TODO: k, rx, ry, surround_x, surround_y or their distribution parameters readable from file

	    fLGN_vpos.read(reinterpret_cast<char*>(&LGN_polar[0]), nLGN*sizeof(Float));
        // polar is in rad
	    fLGN_vpos.read(reinterpret_cast<char*>(&LGN_ecc[0]), nLGN*sizeof(Float));
        // ecc of center is in deg [0, nLGN), surround in rad [nLGN, 2*nLGN)
        auto transform_deg2rad = [deg2rad] (Float ecc) {return ecc*deg2rad;};
        transform(LGN_ecc.begin(), LGN_ecc.begin()+nLGN, LGN_ecc.begin(), transform_deg2rad);
	    fLGN_vpos.close();


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
    	// Benardete and Kaplan 1997 (source fitted by LGN_kernel.ipynb (mean)
	    // empirical value
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
    	Float K_offC[2] = {22.71914498, 0.60*0.37/0.48}; // A [Hz per unit contrast (100% or 1%?)]
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
    	Float delay_onC[2] = {13.45, sqrt(2.39*2.39+0.51*0.51)};
    	Float delay_offC[2] = {13.45, sqrt(2.39*2.39+0.61*0.61)};

    	// ON-SURROUND temporal mu and c.v.
    	Float K_onS[2] = {23.17378917,  1.45*1.53/1.64}; // A
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
    	Float K_offS[2] = {12.53276353, 1.45*1.84/1.64}; // A
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
        auto get_nRonC = get_rand_from_clipped_gauss(nR_onC, nR_onClowBound, nR_onCupBound);
        auto get_nRoffC = get_rand_from_clipped_gauss(nR_offC, nR_offClowBound, nR_offCupBound);
        auto get_nRonS = get_rand_from_clipped_gauss(nR_onS, nR_onSlowBound, nR_onSupBound);
        auto get_nRoffS = get_rand_from_clipped_gauss(nR_offS, nR_offSlowBound, nR_offSupBound);
    	// tauR and nR anti-correlates for stable peak time
    	Float rho_Kc_Ks = 0.5;
    	Float rho_Kc_Ks_comp = sqrt(1.0-rho_Kc_Ks*rho_Kc_Ks);
    	// reset norm to standard normal distribution
    	norm = normal_distribution<Float>(0.0, 1.0);

		cout << "initializing LGN temporal parameters...\n";
		Float pTspD[2];
		Float tspR, tspD;
        // for spontaneous firing rate
    	Float log_mean, log_std;
        Float spontPercent = 0.3;
    	tie(log_mean, log_std) = lognstats<Float>(0.06, 0.01);
    	for (unsigned int i=0; i<nLGN; i++) {
    	    // using median from table 2,3  (on,off)/all * table 5,6 with matching c.v.# Benardete and Kaplan 1997
    	    // fit with difference of exponentials in LGN_kernel.ipynb
    	    // cones' differences are not differentiated
            if (LGNtype[i] == InputType::MonLoff || LGNtype[i] == InputType::LonMoff) {
    	        // on-center, off-surround
    	        // k
    			tie(LGN_k[i], LGN_k[i+nLGN]) = get_rands_from_correlated_gauss(K_onC, K_offS, rho_Kc_Ks, rho_Kc_Ks_comp, rGen_LGNsetup, rGen_LGNsetup, positiveBound, nonNegativeBound);
				LGN_k[i+nLGN] *= -1; //off-surround !!!IMPORTANT sign change here

    	        // centers' tau, n 
    	        nR[i] = get_nRonC(rGen_LGNsetup);
				tspR = get_rand_from_gauss(tspR_onC, rGen_LGNsetup, positiveBound);
				tauR[i] = tspR/(nR[i]-1);

        		tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i], tauR[i]);
                auto tauD_Cbound = get_excLowBound(tauR[i]);
				tauD[i] = get_rand_from_gauss(tauD_onC, rGen_LGNsetup, tauD_Cbound);
				tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
    	        nD[i] = tspD/tauD[i]+1;

    	        // surround' tau, n 
    	        nR[i+nLGN] = get_nRoffS(rGen_LGNsetup);
				tspR = get_rand_from_gauss(tspR_offS, rGen_LGNsetup, positiveBound);
				tauR[i+nLGN] = tspR/(nR[i+nLGN]-1);

        		tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i+nLGN], tauR[i+nLGN]);
                auto tauD_Sbound = get_excLowBound(tauR[i+nLGN]);
				tauD[i+nLGN] = get_rand_from_gauss(tauD_offS, rGen_LGNsetup, tauD_Sbound);
				tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
    	        nD[i+nLGN] = tspD/tauD[i+nLGN];

    	        // ratio
    	        ratio[i] = get_rand_from_gauss(ratio_onC, rGen_LGNsetup, nonNegativeBound);
    	        ratio[i+nLGN] = get_rand_from_gauss(ratio_offS, rGen_LGNsetup, nonNegativeBound);
    	        // delay
    	        delay[i] = get_rand_from_gauss(delay_onC, rGen_LGNsetup, nonNegativeBound);
    	        delay[i+nLGN] = get_rand_from_gauss(delay_offS, rGen_LGNsetup, nonNegativeBound);
            } else {
                if (LGNtype[i] == InputType::MoffLon || LGNtype[i] == InputType::LoffMon) {
    	            // off-centers
    	            // k
    		    	tie(LGN_k[i], LGN_k[i+nLGN]) = get_rands_from_correlated_gauss(K_offC, K_onS, rho_Kc_Ks, rho_Kc_Ks_comp, rGen_LGNsetup, rGen_LGNsetup, positiveBound, nonNegativeBound);
			    	LGN_k[i] *= -1; //off-center

    	            // centers' tau, n 
    	            nR[i] = get_nRoffC(rGen_LGNsetup);
			    	tspR = get_rand_from_gauss(tspR_offC, rGen_LGNsetup, positiveBound);
			    	tauR[i] = tspR/(nR[i]-1);

        	    	tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i], tauR[i]);
                    auto tauD_Cbound = get_excLowBound(tauR[i]);
			    	tauD[i] = get_rand_from_gauss(tauD_offC, rGen_LGNsetup, tauD_Cbound);
			    	tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
    	            nD[i] = tspD/tauD[i];

    	            // surround' tau, n 
    	            nR[i+nLGN] = get_nRonS(rGen_LGNsetup);
			    	tspR = get_rand_from_gauss(tspR_onS, rGen_LGNsetup, positiveBound);
			    	tauR[i+nLGN] = tspR/(nR[i+nLGN]-1);

        	    	tie(pTspD[0], pTspD[1]) = tspD_dist(nR[i+nLGN], tauR[i+nLGN]);
                    auto tauD_Sbound = get_excLowBound(tauR[i+nLGN]);
			    	tauD[i+nLGN] = get_rand_from_gauss(tauD_onS, rGen_LGNsetup, tauD_Sbound);
			    	tspD = get_rand_from_gauss(pTspD, rGen_LGNsetup, positiveBound);
    	            nD[i+nLGN] = tspD/tauD[i+nLGN];

    	            // ratio
    	            ratio[i] = get_rand_from_gauss(ratio_offC, rGen_LGNsetup, nonNegativeBound);
    	            ratio[i+nLGN] = get_rand_from_gauss(ratio_onS, rGen_LGNsetup, nonNegativeBound);
    	            // delay
    	            delay[i] = get_rand_from_gauss(delay_offC, rGen_LGNsetup, nonNegativeBound);
    	            delay[i+nLGN] = get_rand_from_gauss(delay_onS, rGen_LGNsetup, nonNegativeBound);
    	        } else {
   		            throw("There's no implementation of such RF for parvo LGN");
                }
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
            Float spontTmp = exp(log_mean + norm(rGen_LGNsetup) * log_std);
            while (spontTmp > spontPercent) {
                spontTmp = exp(log_mean + norm(rGen_LGNsetup) * log_std);
            }
    	    spont[i] =  spontTmp;
    	    // NOTE: proportion between L and M, significantly overlap in cone response curve, not implemented to calculate maxConvol
    	    covariant[i] = 0.53753461391295254; 
			//cout << "\rLGN #" << i;
    	}
		//cout << "\r";

        norm = normal_distribution<Float>(0.25, 0.03);
    	auto get_c50 = get_rand_from_gauss0(rGen_LGNsetup, norm, nonNegativeBound);
    	generate(c50.begin(), c50.end(), get_c50);
        for (Size j = 0; j<nLGN; j++) {
            assert(c50[j] < 1.0);
        }

        norm = normal_distribution<Float>(10.0, 1.0);
    	auto get_sharpness = get_rand_from_gauss0(rGen_LGNsetup, norm, unityIncBound);
    	generate(sharpness.begin(), sharpness.end(), get_sharpness);
        for (Size j = 0; j<nLGN; j++) {
            assert(sharpness[j] >= 1.0);
        }

    	fLGN.open(LGN_filename, fstream::out | fstream::binary);
    	// append to LGN_polar and LGN_ecc positions
		// surround origin is changed
        if (!fLGN) {
			cout << "Cannot open or find " << LGN_filename <<" for LGN receptive field properties\n";
			return EXIT_FAILURE;
        }
	    fLGN.write((char*)&LGNtype[0], nLGN*sizeof(InputType_t));
    	fLGN.write((char*)&LGN_polar[0], 2*nLGN*sizeof(Float));
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
    hLGN.freeMem();

    vector<PosInt> sxy(2*nLGN)
    ifstream fLGN_surface;
    fLGN_surface.open(fLGN_surface_filename, fstream::in | fstream::binary);
    if (!fLGN_surface) {
	    cout << "Cannot open or find " << fLGN_surface_filename <<" to read in LGN surface position.\n";
	    return EXIT_FAILURE;
    } else {
        fLGN_surface.read(reinterpret_cast<char*>(&sxy[0]), 2*nLGN*sizeof(PosInt));
    }
	size_t usingGMem = 0;

    // malloc for LGN
	size_t spikeGenSize = (2*sizeof(Float) + sizeof(curandStateMRG32k3a) + 2*sizeof(PosInt)) * nLGN;

    size_t outputB4V1Size = 4*sizeof(Float);
	size_t B4V1Size = spikeGenSize + outputB4V1Size;

	Float* outputB4V1 = new Float[outputB4V1Size];

	char* gpu_B4V1;
    checkCudaErrors(cudaMalloc((void **)&gpu_B4V1, B4V1Size));
	usingGMem += B4V1Size;

    PosInt *d_sx = (PosInt*) gpu_B4V1
    PosInt *d_sy =  d_sx + nLGN;
    Float* leftTimeRate = (Float*) d_sy + nLGN;
    Float* lastNegLogRand = leftTimeRate + nLGN;
    curandStateMRG32k3a *randState = (curandStateMRG32k3a*) lastNegLogRand + nLGN;
    Float* d_LGN_fr = (Float*) randState + nLGN;
    Float* currentConvol = d_LGN_fr + nLGN;
	// TODO: not necessary when not checking add option flag
    Float* luminance = currentConvol + nLGN; 
    Float* contrast = luminance + nLGN;


    // initialize
    checkCudaErrors(cudaMemcpy(d_sx, &(sxy[0]), sizeof(PosInt)*2*nLGN, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaMemset(maxConvol, 0, nLGN*sizeof(Float)));

    seed++;
	Size nLGN_block, nLGN_thread; // for LGN_nonlinear
	nLGN_block = (nLGN + blockSize - 1)/blockSize;
	nLGN_thread = blockSize;
    logRand_init<<<nLGN_block, nLGN_thread>>>(lastNegLogRand, leftTimeRate, randState, seed);
    getLastCudaError("logRand_init");

	// malloc LGN_surface
    cudaChannelFormatDesc surfaceDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuSurfArray;
    checkCudaErrors(cudaMallocArray(&cuSurfArray, &surfaceDesc, nsx, nsy, cudaArraySurfaceLoadStore));
    cudaBindSurfaceToArray(LGNspikeSurface, cuSurfArray);
	usingGMem += nsx*nsy*sizeof(Float);

	// Storage memory
	size_t maxConvolSize = nLGN;
	size_t TW_size = nType*nKernelSample*nLGN;
	size_t SW_size = nType*nSample*nLGN;
	size_t SC_size = 2*nType*nSample*nLGN;
	size_t gallerySize = (maxConvolSize + TW_size + SW_size)*sizeof(Float) + SC_size*sizeof(float);
	char* galleryOutput = new char[gallerySize]; 

	Float* gpu_LGN_gallery;
    checkCudaErrors(cudaMalloc((void **) &gpu_LGN_gallery, gallerySize));
	usingGMem += gallerySize;

    Float* maxConvol = gpu_LGN_gallery;
    Float* TW_storage = maxConvol + maxConvolSize;
    Float* SW_storage = TW_storage + TW_size;
    float* SC_storage = SW_storage + SW_size;
	

	cout << "LGN initialized\n";
    // finish LGN setup

    // V1 related memory
    ifstream fV1_pos;
    ifstream fV1_conMat, fV1_delayMat, fV1_vec; 

    Size nV1;
    Size nblock
    Size neuronPerBlock 
    Size posDim;
    double *cpu_chunk_V1pos;
    double *V1_x;
    double *V1_y;
    double *V1_vx;
    double *V1_vy;
    fV1_pos.open(V1_pos_filename, fstream::in | fstream::binary);
    if (!fV1_pos) {
		cout << "Cannot open or find " << V1_RF_filename <<" to read V1 positions.\n";
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
        fV1_pos.read(reinterpret_cast<char*>(&nV1), sizeof(Size));
        cout << nV1 << " V1 neurons\n";
        if (!getStatsFrameV) {
            cpu_chunk_V1pos = new Float[nV1*2];
            fV1_pos.read(reinterpret_cast<char*>(cpu_chunk_V1pos), 2*nV1*sizeof(double));
        } else {
            cpu_chunk_V1pos = new Float[nV1*4];
            fV1_pos.read(reinterpret_cast<char*>(cpu_chunk_V1pos), 4*nV1*sizeof(double));
        }
        V1_x = cpu_chunk_V1pos;
        V1_y = V1_x + nV1;
        if (getStatsFrameV) {
            V1_vx = V1_y + nV1;
            V1_vy = V1_vx + nV1;
        }
    }
    double xminV, xmaxV;
    double xmin, xmax;
    double yminV, ymaxV;
    double ymin, ymax;
    tie(xmin, xmax) = array_minmax<double>(V1_x, nV1);
    tie(ymin, ymax) = array_minmax<double>(V1_y, nV1);
    Float V1_hwRatio = static_cast<Float>((ymax-ymin)/(xmax-xmin));

    // stats: frames, rawdata
	// cortical surface
    Size ps_height = ceil(V1_hwRatio * ps_width);
    if (ps_height%2 == 1) ps_height++;
    Size pixel_stats = ps_width * ps_height;
	// visual field surface
    Size pixel_statsV, ps_widthV, ps_heightV;
	size_t posSize = 2*nV1*sizeof(double) + pixel_stats*sizeof(Float);
    if (getStatsFrameV) {
        tie(xminV, xmaxV) = array_minmax<double>(V1_vx, nV1);
        tie(yminV, ymaxV) = array_minmax<double>(V1_vy, nV1);
        Float V1_hwRatioV = static_cast<Float>((ymaxV-yminV)/(xmaxV-xminV));
    	ps_widthV = getStatsFrameV;
    	ps_heightV = ceil(V1_hwRatioV * ps_widthV);
    	// left + 1 + right
    	pixel_statsV = (2*ps_widthV+1) * ps_heightV;
		posSize += 2*nV1*sizeof(double) + pixel_statsV*sizeof(Float);
    }
	cout << "stats frame output requires " << posSize/1024.0/1024.0 << "Mb\n";

	// overlap multiple chunks of data transfer and computation to increase performance
    Size maxChunkSize;
    Size remainChunkSize = nblock % nChunk;
    if (remainChunkSize == 0) {
        maxChunkSize = nblock/nChunk;
        remainChunkSize = maxChunkSize;
    } else {
        maxChunkSize = (nblock-remainChunkSize)/(nChunk-1);
        assert((nblock-remainChunkSize)%(nChunk-1) == 0);
    }


    Size nearNeighborBlock;
    fV1_conMat.open(V1_conMat_filename, fstream::in | fstream::binary);
    if (!fV1_conMat) {
		cout << "Cannot open or find " << V1_conMat_filename <<" to read V1 cortical connection matrices.\n";
		return EXIT_FAILURE;
    } else {
        fV1_conMat.read(reinterpret_cast<char*>(&nearNeighborBlock), sizeof(Size)
    }
    fV1_delayMat.open(V1_delayMat_filename, fstream::in | fstream::binary);
    if (!fV1_delayMat) {
		cout << "Cannot open or find " << V1_delayMat_filename <<" to read V1 cortical distance matrices.\n";
		return EXIT_FAILURE;
    } else {
        Size tmp;
        fV1_delayMat.read(reinterpret_cast<char*>(&tmp), sizeof(Size)
        if (tmp != nearNeighborBlock) {
            cout << "conMat and delayMat does not match\n";
            return EXIT_FAILURE;
        }
    }

    // intertwine the two matrices for ovelapping gpu kernel later
    size_t blockChunkSize = nearNeighborBlock*blockSize*blockSize * sizeof(Float);
    size_t matSize = nblock * blockChunkSize; 
    Float* conDelayMat0 = new Float[matSize*2];
	Float** conDelayMat = new Float*[nChunk];
	Float** conMat = new Float*[nChunk];
	Float** delayMat = new Float*[nChunk];
    size_t matOffset = 0;
    size_t matChunkSize = maxChunkSize*blockChunkSize;
	Float maxDistance = 0;
    for (PosInt i=0; i<nChunk; i++) {
		size_t chunkSize = matChunkSize;
        if (i == nChunk-1) chunkSize = remainChunkSize*blockChunkSize;
		conDelayMat[i] = conDelayMat0 + matOffset;
		conMat[i] = conDelayMat[i];
        matOffset += chunkSize;
		delayMat[i] = conDelayMat0 + matOffset;
        if (i < nChunk - 1) {
            matOffset += chunkSize;
        }
	}

    for (PosInt i=0; i<nChunk; i++) {
		size_t chunkSize = matChunkSize;
        if (i == nChunk-1) chunkSize = remainChunkSize*blockChunkSize;
        fV1_conMat.read(reinterpret_cast<char*>(conMat[i]), chunkSize);
        fV1_delayMat.read(reinterpret_cast<char*>(delayMat[i]), chunkSize);
    	Float current_maxDistance = array_max(delayMat[i], chunkSize);
		if (maxDistance < current_maxDistance) maxDistance = current_maxDistance;
	}
    delete []*conMat;
    delete []*delayMat;
    Size trainDepth = ceil((maxDistance/speedOfThought)/dt);

    // single chunk of Pinned memory on CPU for heavy usages 
    // spikeTrain, voltage, gE, hE, gI, hI
	Float* pinnedMem;
	size_t trainSize = trainDepth*nV1;
	size_t ghSize = 2*nV1*(ngTypeE + ngTypeI)*sizeof(Float);
	size_t vSize = nV1*sizeof(Float);
	size_t pinnedSize = trainSize*sizeof(Float) + vSize + ghSize;
	checkCudaErrors(cudaMallocHost((void**) &pinnedMem, pinnedSize));
	Float *spikeTrain = pinnedMem;
    Float *v = spikeTrain + trainSize;
    Float **gE = new Float*[nChunk];
    Float **gI = new Float*[nChunk];
    Float **hE = new Float*[nChunk];
    Float **hI = new Float*[nChunk];

    size_t eSize = maxChunkSize*blockSize*ngTypeE;
    size_t iSize = maxChunkSize*blockSize*ngTypeI;
    gE[0] = v + nV1;
    gI[0] = gE[0] + eSize;
    hE[0] = gI[0] + iSize;
    hI[0] = hE[0] + eSize;
    for (PosInt i = 1; i<nChunk; i++) {
        gE[i] = hI[i-1] + iSize; 
        if (i == nChunk-1) {
            eSize == remainChunkSize*blockSize*ngTypeE;
            iSize == remainChunkSize*blockSize*ngTypeI;
        }
        gI[i] = gE[i] + eSize;
        hE[i] = gI[i] + iSize;
        hI[i] = hE[i] + eSize;
    }

	vghSize = vSize + ghSize;

	// GPU arrays to receive g,h sum from conVec (pinned)
    Float *d_gh;
    checkCudaErrors(cudaMalloc((void**)&d_gh, ghBothwaySize);
	usingGMem += ghBothwaySize;

    // v, g (D2H), h (D only)
    Float *d_vgh;
    checkCudaErrors(cudaMalloc((void**)&d_vgh, vghSize);
	usingGMem += vghSize;

    Float *d_v = d_vgh;
    Float **d_gE = new Float*[nChunk];
    Float **d_gI = new Float*[nChunk];
    Float **d_hE = new Float*[nChunk];
    Float **d_hI = new Float*[nChunk];
    eSize = maxChunkSize*blockSize*ngTypeE;
    iSize = maxChunkSize*blockSize*ngTypeI;
    d_gE[0] = d_v + nV1;
    d_gI[0] = d_gE[0] + eSize;
    d_hE[0] = d_gI[0] + iSize;
    d_hI[0] = d_hE[0] + eSize;
    for (PosInt i = 1; i<nChunk; i++) {
        d_gE[i] = d_hI[i-1] + iSize; 
        if (i == nChunk-1) {
            eSize == remainChunkSize*blockSize*ngTypeE;
            iSize == remainChunkSize*blockSize*ngTypeI;
        }
        d_gI[i] = d_gE[i] + eSize;
        d_hE[i] = d_gI[i] + iSize;
        d_hI[i] = d_hE[i] + eSize;
    }

    // for spikeTrain D2H (only output the current slot to file)
    trainSize *= sizeof(Float);
    Float *d_spikeTrain;
    checkCudaErrors(cudaMalloc((void**)&d_spikeTrain, trainSize);
	usingGMem += trainSize;

    init<<<nblock, blockSize>>>(d_spikeTrain, -1.0, nV1);
    cout << "spikeTrain retains spikes for " << trainDepth << "time steps\n";

    Size max_LGNperV1;
    Float* LGN_V1_s;
	read_listOfListToArray<Float>(LGN_V1_s_filename, LGN_V1_s, max_LGNperV1, false);
	PosInt* sLGN;
	size_t sLGN_size = max_LGNperV1*nV1*sizeof(PosInt);
    checkCudaErrors(cudaMalloc((void**)&sLGN, sLGN_size));
	usingGMem += sLGN_size;

    checkCudaErrors(cudaMemcpy(sLGN, LGN_V1_s, sLGN_size));
    delete []sLGN;

    vector<vector<PosInt>> LGN_V1_ID = read_listOfList<PosInt>(LGN_V1_ID_filename, false);
	size_t surfaceSize = max_LGNperV1*nV1*2;
    PosInt* LGN_V1_surface = new PosInt[surfaceSize];
	surfaceSize *= sizeof(PosInt);
    getLGN_V1_surface(sxy, LGN_V1_ID, LGN_V1_surface, max_LGNperV1, nLGN);
    // release memory from LGN_V1_ID and sxy
    LGN_V1_ID.swap(vector<PosInt>());
    sxy.swap(vector<PosInt>());

	PosInt* d_LGN_V1_surface;
    checkCudaErrors(cudaMalloc((void**)&d_LGN_V1_surface, surfaceSize));
	usingGMem += surfaceSize;

	PosInt* LGN_idx = d_LGN_V1_surface;
	PosInt* LGN_idy = LGN_idx + surfaceSize/2;
    checkCudaErrors(cudaMemcpy(d_LGN_V1_surface, LGN_V1_surface, surfaceSize, cudaMemcpyHostToDevice));

	delete []LGN_V1_surface;
	cout << "LGN_V1_surface requires " << surfaceSize/1024.0/1024.0 << " Mb\n";
		
	vector<Size> nVec(nV1);
	vector<vector<Size>> vecID(nV1);
	vector<vector<Size>> conVec(nV1);
    fV1_vec.open(V1_vec_filename, fstream::in | fstream::binary);
    if (!fV1_vec) {
		cout << "Cannot open or find " << V1_vec_filename <<" to read V1 connection to farther neighbor.\n";
		return EXIT_FAILURE;
    } else {
        fV1_vec.read(reinterpret_cast<char*>(&nVec[0]), nV1*sizeof(Size));
		for (PosInt i=0; i<nV1; i++) {
			if (nVec[i] > 0) {
				vector<PosInt> tmp(nVec[i]);
        		fV1_vec.read(reinterpret_cast<char*>(&tmp[0]), nVec[i]*sizeof(PosInt));
				vecID[i] = tmp;

				vector<Float> ftmp(nVec[i]);
        		fV1_vec.read(reinterpret_cast<char*>(&ftmp[0]), nVec[i]*sizeof(Float));
				conVec[i] = ftmp;

        		fV1_vec.read(reinterpret_cast<char*>(&ftmp[0]), nVec[i]*sizeof(Float));
				delayVec[i] = ftmp;
			}
		}
	}

	vector<Size> nNeighborBlock(nblock);
	vector<vector<PosInt>> neighborBlockId(nblock);
    fNeighborBlock.open(neighborBlock_filename, fstream::in | fstream::binary);
	if (!fNeighborBlock) {
		cout << "Cannot open or find " << V1_vec_filename <<" to read V1 neighbor block info.\n";
		return EXIT_FAILURE;
	} else {
        fNeighborBlock.read(reinterpret_cast<char*>(&nNeighborBlock[0]), nblock*sizeof(Size));
		for (PosInt i=0; i<nblock; i++) {
			if (nNeighborBlock[i] > 0) {
				vector<PosInt> tmp(nNeighborBlock[i]);
        		fNeighborBlock.read(reinterpret_cast<char*>(&tmp[0]), nNeighborBlock[i]*sizeof(PosInt));
				neighborBlockId[i] = tmp;
			}
		}
	}

    if (!fV1_vec) {
		cout << "Cannot open or find " << V1_vec_filename <<" to read V1 connection to farther neighbor.\n";
		return EXIT_FAILURE;
    } else {
        fV1_vec.read(reinterpret_cast<char*>(&nVec[i]), nV1*sizeof(Size));
		for (PosInt i=0; i<nV1; i++) {
			if (nVec[i] > 0) {
				vecID[i].assign(nVec[i], 0);
        		fV1_vec.read(reinterpret_cast<char*>(&vecID[i]), nVec[i]*sizeof(PosInt));
				conVec[i].assign(nVec[i], 0);
        		fV1_vec.read(reinterpret_cast<char*>(&conVec[i]), nVec[i]*sizeof(Float));
				delayVec[i].assign(nVec[i], 0);
        		fV1_vec.read(reinterpret_cast<char*>(&delayVec[i]), nVec[i]*sizeof(Float));
			}
		}
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
    /* read V1_feature
        fV1_feature.open(V1_feature_filename, ios::in|ios::binary);
	    if (!fV1_feature) {
	    	cout << "failed to open pos file:" << V1_feature_filename << "\n";
	    	return EXIT_FAILURE;
	    }
	    Size nFeature;
        fV1_feature.read(reinterpret_cast<char*>(&nFeature), sizeof(Size));
	    vector<Float> featureValue(nFeature*nV1); // [OD, OP, ..]
        fV1_feature.read(reinterpret_cast<char*>(&featureValue[0]), sizeof(Float)*nFeature*nV1);
	    fV1_feature.close();
    */

	{ // output file tests
		if (saveLGN_fr) {
    		fLGN_fr.open(LGN_fr_filename, fstream::out | fstream::binary);
    		if (!fLGN_fr) {
				cout << "Cannot open or find " << LGN_fr_filename <<" for LGN firing rate output\n";
				return EXIT_FAILURE;
    		} else {
        	    fLGN_fr.write((char*)&nt, sizeof(Size));
        	    fLGN_fr.write((char*)&nLGN, sizeof(Size));
        	}
		}

        fRawData.open(rawData_filename, fstream::out | fstream::binary);
        if (!fRawData) {
			cout << "Cannot open or find " << fRawData_filename <<" for V1 simulation results.\n";
			return EXIT_FAILURE;
        } else {
            fRawData.write((char*) &nt, sizeof(Size));

        }

		if (saveLGN_gallery) {
    		fLGN_gallery.open(LGN_gallery_filename, fstream::out | fstream::binary);
    		if (!fLGN_gallery) {
				cout << "Cannot open or find " << LGN_gallery_filename <<" for storage check.\n";
				return EXIT_FAILURE;
    		} else {
        		fLGN_gallery.write((char*)&nLGN, sizeof(Size));
        		fLGN_gallery.write((char*)&nType, sizeof(Size));
        		fLGN_gallery.write((char*)&nKernelSample, sizeof(Size));
        		fLGN_gallery.write((char*)&nSample, sizeof(Size));
			}
		}
        if (saveOutputB4V1) {
            fOutputB4V1.open(outputB4V1_filename, fstream::out | fstream::binary);
            if (!fOutputB4V1) {
	        	cout << "Cannot open or find " << outputB4V1_filename <<" to store ouput before V1.\n";
	        	return EXIT_FAILURE;
            } else {
                fOutputB4V1.write((char*)&nt, sizeof(Size));
                fOutputB4V1.write((char*)&dt, sizeof(Float));
                fOutputB4V1.write((char*)&nLGN, sizeof(Size));
            }
        }
		cout << "output file check, done\n";
	}

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
        cout << "tau should be divisible by dt.\n"; 
        return EXIT_FAILURE;
    }
    if (nRetrace % nKernelSample != 0) {
        cout << "tau in #dt should be divisible by nKernelSample.\n"; 
        return EXIT_FAILURE;
    }

    Size iKernelSampleT0;
    PosInt kernelSampleInterval = nRetrace/nKernelSample;
    if (kernelSampleInterval%2 == 0) {
        iKernelSampleT0 = kernelSampleInterval/2;
		cout << "sample in intervals of " << kernelSampleInterval << " starting with " << iKernelSampleT0 << " in units of dt\n";
    } else {
        iKernelSampleT0 = 0;
        if (kernelSampleInterval > 1) {
            cout << "make sample interval (" << kernelSampleInterval << ") even in the units of dt\n";
        }
    }
    // |--*--|--*--|--*--|, e.g., nKernelSample = 3->*
    Float kernelSampleDt = kernelSampleInterval*dt;
    Float kernelSampleT0 = iKernelSampleT0*dt;
    Size kernelSampleRate = stepRate/kernelSampleInterval; 
    printf("temporal kernel retraces %f ms, samples %u points, sample rate = %u Hz\n", tau, nKernelSample, kernelSampleRate);

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
        cout << exact_it[i] << " + " << exact_norm[i] << " / " << denorm << "\n";
    }
    // the first i frames' accumulated length in steps = exact_it[i] + exact_norm[i]/denorm
    assert((stepRate*denorm) % frameRate == 0);
    Size co_product = (stepRate*denorm)/frameRate; 
    // the number of non-zero minimum steps to meet frameLength * denorm with 0 phase
    Size ntPerFrame = co_product; // tPerFrame in the units of dt/denorm

    Size maxFrame = (nRetrace*denorm + ntPerFrame-1)/ntPerFrame + 1; 
    if (maxFrame > 2048) {
        cout << "a single layered texture object only allows 2048 layers < " << maxFrame << "\n";
        return EXIT_FAILURE;
    }
	// max frame need to be stored in texture for temporal convolution with the LGN kernel.
    //  |---|--------|--|
    printf("temporal kernel retrace (tau): %f ms, frame rate: %d Hz needs at most %u frames\n", tau, frameRate, maxFrame);
    unsigned int nChannel = 3; // L, M, S
    Size texture_nElem = nPixelPerFrame*nChannel*maxFrame;
    printf("=== texture memory required: %dx%dx%d = %fMB\n", maxFrame,width,height, texture_nElem*sizeof(float)/1024.0/1024.0);

    // one cudaArray per channel
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArr_L;
    cudaArray *cuArr_M;
    cudaArray *cuArr_S;

    checkCudaErrors(cudaMalloc3DArray(&cuArr_L, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_M, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));
    checkCudaErrors(cudaMalloc3DArray(&cuArr_S, &channelDesc, make_cudaExtent(width, height, maxFrame), cudaArrayLayered));

	usingGMem += 3*width*height*maxFrame*sizeof(Float);

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

	// pinned pathway for conDelayMat
	if (matConcurrency > nChunk) {
		cout << "matConcurrency is reduced to " << nChunk << "  == nChunk\n";  
	}
	size_t sChunkMatSize = 2*maxChunkSize*blockChunkSize;
	size_t rChunkMatSize = 2*remainChunkSize*blockChunkSize;
	cout << "single chunk of conDelayMat requires " << sChunkMatSize/1024.0/1024.0 << ", the last chunk requires " << rChunkMatSize << " Mb\n";
	size_t ccChunkMatSize;
	if (matConcurrency == nChunk) { // if total concurrency, match the final chunk size
		ccChunkMatSize = (matConcurrency-1) * sChunkMatSize + rChunkMatSize;
	} else {
		ccChunkMatSize = matConcurrency * sChunkMatSize;
	}
	int ccReduced = 0;
	while (usingGMem + ccChunkMatSize > deviceProps.totalGlobalMem) {
		matConcurrency--;
		if (matConcurrency == nChunk) {
			ccChunkMatSize -= rChunkMatSize;
		} else {
			ccChunkMatSize -= sChunkMatSize;
		}
		ccReduced++;
	}
	if (ccReduced) {
		cout << "GPU does not have the required memory for requested matConcurrency " << matConcurrency + ccReduced << ", it's now reduced to " << matConcurrency << "\n";
	}

	cout << "matConcurrency of " << matConcurrency << " chunks requires " << ccChunkMatSize/1024.0/1024.0 << " Mb, total device gmem = " << deviceProps.totalGlobalMem/1024.0/1024.0 << "Mb\n";

	// pinned conDelayMat on Host
	Float *p_conDelayMat;
	checkCudaErrors(cudaMallocHost((void**) &p_conDelayMat, ccChunkMatSize));

	// receiving end of pinned conDelayMat on Device
    Float *d_mat; // d_delayMat;
    checkCudaErrors(cudaMalloc((void**)&d_mat, ccChunkMatSize));
	usingGMem += ccChunkMatSize;
	Float **d_conDelayMat = new Float[matConcurrency];
	d_conDelayMat[0] = d_mat;
	for (PosInt i = 1; i<matConcurrency; i++) {
		d_conDelayMat[i] = d_conDelayMat[i-1] + sChunkMatSize;
	}

	cout << "Using "<< usingGMem/1024.0/1024.0 << " Mb from a total of " << deviceProps.totalGlobalMem/1024.0/1024.0 << " Mb, remaining " << (deviceProps.totalGlobalMem - usingGMem)/1024.0/1024.0 << " Mb\n";

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
        cudaMemsetNonzero<<<nGrid, blockSize, 0, initStream[0]>>> (tL, nPixelPerFrame*maxFrame, init_L);
	    getLastCudaError("memset failed");
        cudaMemsetNonzero<<<nGrid, blockSize, 0, initStream[1]>>> (tM, nPixelPerFrame*maxFrame, init_M);
	    getLastCudaError("memset failed");
        cudaMemsetNonzero<<<nGrid, blockSize, 0, initStream[2]>>> (tS, nPixelPerFrame*maxFrame, init_S);
	    getLastCudaError("memset failed");

        prep_sample(0, width, height, tL, tM, tS, cuArr_L, cuArr_M, cuArr_S, maxFrame, cudaMemcpyDeviceToDevice); // implicit synchronized
		/* DEBUG
        	dim3 fb(16,16,1);
        	dim3 fg(16,16,maxFrame);
        	testTexture<<<fg, fb>>> (init_L, init_M, init_S);
	    	getLastCudaError("texture read test failed");
		*/
        checkCudaErrors(cudaFree(tLMS));
    	for (PosInt i=0; i<3; i++) {
		    checkCudaErrors(cudaStreamDestroy(initStream[i]));
    	}
        cout << "all pixels in texture memory (frame buffers) initialized to " << init_L << ", " << init_M << ", " << init_S << " \n";
    }

    dim3 convolBlock(nSpatialSample1D, nSpatialSample1D, 1);
    dim3 convolGrid(nLGN, 2, 1);
    
	cout << "cuda memory, set\n";

	cout << "store<<<" << convolGrid.x  << "x" << convolGrid.y  << "x" << convolGrid.z << ", " << convolBlock.x  << "x" << convolBlock.y  << "x" << convolBlock.z << ">>>\n";
    // store spatial and temporal weights determine the maximums of LGN kernel convolutions
    store<<<convolGrid, convolBlock>>>(
            maxConvol,
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
	cout << "convol parameters stored\n";

	if (saveLGN_gallery) {// storage check output
    	checkCudaErrors(cudaMemcpy(galleryOutput, gpu_LGN_gallery, gallerySize, cudaMemcpyDeviceToHost));

    	fLGN_gallery.write((char*)galleryOutput, gallerySize);
    	fLGN_gallery.close();
	    cout << "convol parameters written to disk.\n";
	}
    // calc LGN firing rate at the end of current dt
    PosInt currentFrame = 0; // current frame number from stimulus
    PosInt oldFrame = currentFrame;
    // framePhase is fractionalized by denorm to fit frame duration with integer units.
    PosInt iFramePhaseHead = 0;
    PosInt remain = (nRetrace*denorm) % ntPerFrame;
    PosInt comp = (ntPerFrame - remain) % ntPerFrame;
    PosInt iFramePhaseTail = comp;

    PosInt iFrameHead = currentFrame % maxFrame;
    //PosInt iPhase = 0;
    auto getTail = [comp](PosInt phaseTail, PosInt phaseHead, PosInt head) {
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

    cudaEvent_t V1Ready, spReady, gReady;
	checkCudaErrors(cudaEventCreate(&V1Ready));
	checkCudaErrors(cudaEventCreate(&spReady));
	checkCudaErrors(cudaEventCreate(&gReady));
	//cudaEvent_t *gReady = new cudaEvent_t[nChunk];
    //for (PosInt i = 0; i < nChunk; i++) {
	//	checkCudaErrors(cudaEventCreate(&gReady[i]));
    //}

    cudaStream_t mainStream;
    checkCudaErrors(cudaStreamCreate(&mainStream));

    cudaStream_t *stream = new cudaStream_t[nChunk];
    for (PosInt i = 0; i < nChunk; i++) {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    PosInt currentTimeSlot = 0;
	cout << "simulation starts: \n";
    for (unsigned int it = 0; it < nt; it++) {
        Float t = it*dt;
        PosInt oldFrameHead;
        // next frame comes between (t, t+dt), read and store frame to texture memory
        if ((it+1)*denorm >= currentFrame*ntPerFrame) {
            // back insert frame into texture memory
            // TODO: realtime video stimulus control
            if (fStimulus) {
                fStimulus.read(reinterpret_cast<char*>(LMS), nChannel*nPixelPerFrame*sizeof(float));
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
            prep_sample(iFrameHead, width, height, L, M, S, cuArr_L, cuArr_M, cuArr_S, 1, cudaMemcpyHostToDevice);

            //iPhase = (iPhase + 1) % denorm;
            currentFrame++;
            oldFrameHead = iFrameHead;
            iFrameHead = (iFrameHead+1) % maxFrame;

	        printf("\rsimulating@t = %f -> %f, frame %d#%d-%d, %.1f%%\n", t, t+dt, currentFrame/nFrame, currentFrame%nFrame, nFrame, 100*static_cast<float>(it+1)/nt);
        }
        // update frame for head and tail for convolution at t=(it + 1)*dt
        iFramePhaseTail = (iFramePhaseTail + denorm) % ntPerFrame;
        //if (iFramePhaseTail == 0) iFramePhaseTail = ntPerFrame;
        iFramePhaseHead = (iFramePhaseHead + denorm) % ntPerFrame;
        // point frametail to the tail of the LGN temporal convolution at t-tau
        PosInt iFrameTail = getTail(iFramePhaseTail, iFramePhaseHead, oldFrameHead);
        cout << "it = " << it << ", head at " << oldFrameHead << "\n";

		/* if it < nRetrace, padded zero-valued frames for t<0
            -->|        |<-- framePhase
            |--|------frame------|
            |-----|-----|-----|-----|-----|
            jt-2, jt-1, jt ...nRetrace... it
        */// perform kernel convolution with built-in texture interpolation
        convolGrid.x = nLGN;
        convolGrid.y = 1;
        LGN_convol_c1s<<<convolGrid, convolBlock, sizeof(Float)*nSample, mainStream>>>(
                luminance,
                SW_storage, SC_storage, TW_storage,
                currentConvol, contrast,
                dLGN.coneType, *dLGN.spatial,
				nLGN_L,
				normViewDistance,
                iFrameTail, maxFrame, ntPerFrame, iFramePhaseTail,
                Itau,
                iKernelSampleT0, kernelSampleInterval, nKernelSample,
                dt, denorm);

		getLastCudaError("LGN_convol_c1s failed");
		if (it > 0) { // seeking for overlap of data output with LGN input
    		fRawData.write((char*) (spikeTrain + nV1*currentTimeSlot), nV1*sizeof(Float));
    		fRawData.write((char*) v, nV1*sizeof(Float));
        	currentTimeSlot++;
		}

		// generate LGN fr with logistic function
        LGN_nonlinear<<<nLGN_block, nLGN_thread, 0, mainStream>>>(nLGN, *dLGN.logistic, maxConvol, currentConvol, d_LGN_fr, d_sx, d_sy, leftTimeRate, lastNegLogRand, randState);
		if (it > 0) { // seeking for overlap of data output with LGN input
			cudaEventSynchronize(gReady);
			// write g to fRawData
    		reshape_chunk_and_write(gE[0], fRawData, maxChunkSize, remainChunkSize, nChunk, ngTypeE, ngTypeI, nV1);
		}

		getLastCudaError("LGN_nonlinear failed");
        if (saveOutputB4V1) { // TODO: use Async pinned memory
        	checkCudaErrors(cudaMemcpy(outputB4V1, d_LGN_fr, outputB4V1Size, cudaMemcpyDeviceToHost));

        	fOutputB4V1.write((char*)outputB4V1, outputB4V1Size); // d_LGN_fr, currentConvol, luminance, contrast
        }
		if (saveLGNfr) {// use only the first row of outputB4V1
			if (!saveOutputB4V1) { // else already copied
        		checkCudaErrors(cudaMemcpy(outputB4V1, d_LGN_fr, nLGN*sizeof(Float), cudaMemcpyDeviceToHost));
			}
        	fLGN_fr.write((char*)outputB4V1, nLGN*sizeof(Float)); 
		}

        if (it == nt-1) {
	        printf("\r@t = %f -> %f simulated, frame %d#%d-%d, %.1f%%", t, t+dt, currentFrame/nFrame, currentFrame%nFrame, nFrame, 100*static_cast<float>(it+1)/nt);
            oldFrame = currentFrame;
        }

        // TODO: block-wise compute_V
        // simulate V1 response
        compute_V_collect_spike <<<nblock, blockSize, 0, mainStream>>> (
                d_v, d_gE, d_gI, d_hE, d_hI,
                d_spikeTrain, tBack,
                sLGN, LGN_idx, LGN_idy,
                stateE, stateI,
                currentTimeSlot, trainDepth,
                max_nLGN,
                ngTypeE, ngTypeI, condE, condI,
                dt, nV1, mE, seed);
		getLastCudaError("compute_V_collect_spike failed");

        checkCudaErrors(cudaMemcpyAsync(spikeTrain, d_spikeTrain, trainSize, cudaMemcpyDeviceToHost, mainStream)); // to overlap with  recal_G, to be used in recal_Gvec
        cudaEventRecord(spReady, mainStream);

        size_t chunkSize = maxChunkSize;
        PosInt block_offset = 0;
		size_t mat_offset = 0;
        for (PosInt i = 0; i < nChunk; ib++) {
            if (i == nChunk-1) chunkSize = remainChunkSize;
            size_t mChunkSize = chunkSize*blockChunkSize;
			size_t p_offset = mat_offset%ccChunkMatSize;
			memcpy((void*) p_conDelayMat + p_offset, (void*) conDelayMat[i], 2*mChunkSize);
        	checkCudaErrors(cudaMemcpyAsync(d_conDelayMat[i%matConcurrency], p_conDelayMat + p_offset, 2*mChunkSize, cudaMemcpyHostToDevice, stream[i%matConcurrency]));

			Float* d_conMat = d_conDelayMat[i%matConcurrency];
            Float* d_delayMat = d_conMat + mChunkSize;
			if (i == 0) {
				cudaEventSynchronize(vReady);
			}
            recal_G<<< chunkSize, blockSize, 0, stream[i%matConcurrency]>>> (
                    d_spikeTrain,
                    d_conMat[i%matConcurrency], d_delayMat[i%matConcurrency],
                    d_gE[i], d_gI[i], d_hE[i], d_hI[i],
                    nNeighborBlock, neighborBlockId + block_offset*nearNeighborBlock,
                    condE, condI, ngTypeE, ngTypeI,
                    currentTimeSlot, trainDepth,
                    nearNeighborBlock, nV1, mE, speedOfThought);
		    getLastCudaError("recal_G failed");
            if (i < nChunk - 1) {
                block_offset += chunkSize;
				mat_offset += 2*mChunkSize;
            }
        }

        checkCudaErrors(cudaMemcpyAsync(v, d_v, nV1*sizeof(Float), cudaMemcpyDeviceToHost, mainStream)); // to overlap with  recal_G, to be used in recal_Gvec

        block_offset = 0;
        chunkSize = maxChunkSize;
		cudaEventSynchronize(spReady);
		size_t pinned_H2D = 0
		size_t pinned_D2H = ghSize;
        for (PosInt i = 0; i < nChunk; i++) {
            if (i == nChunk-1) chunkSize = remainChunkSize;
			size_t gChunkSize = chunkSize*blockSize*(ngTypeE+ngTypeI)*sizeof(Float);
			size_t ghChunkSize = gChunkSize*2;
            // cpu accumulate conductances from far neighbors
            recal_G_vec(
                    spikeTrain,
                    nVec, vecID, conVec, delayVec,
                    gE[i], gI[i], hE[i], hI[i],
                    condE, condI, ngTypeE, ngTypeI,
                    block_offset, currentTimeSlot, trainDepth,
                    nearNeighborBlock, nV1, mE, speedOfThought, chunkSize, maxChunkSize);
            // g and h
            checkCudaErrors(cudaMemcpyAsync(d_gEt[i], gE[i], ghChunkSize, cudaMemcpyHostToDevice, stream[i])); // size in maxChunk

            sum_G_output_G<<<chunkSize, blockSize, 0, stream[i]>>> (nVec + block_offset*blockSize, d_gEt[i], d_gE[i], d_gIt[i], d_gI[i], d_hEt[i], d_hE[i], d_hIt[i], d_hI[i], block_offset);

            // g only
			// 							  // char*
            checkCudaErrors(cudaMemcpyAsync(gE[i], d_gE[i], gChunkSize, cudaMemcpyDeviceToHost, stream[i])); // size in chunk
			if (i < nChunk-1) {
            	block_offset += chunkSize;
            	pinned_H2D += ghChunkSize;
            	pinned_D2H += gChunkSize;
			} else {
				cudaEventRecord(gReady, stream[i]);
			}
		}
    }
    fRawData.write((char*) (spikeTrain + nV1*currentTimeSlot), nV1*sizeof(Float));
    fRawData.write((char*) v, nV1*sizeof(Float));
	// write g to fRawData
	cudaEventSynchronize(gReady);
    reshape_chunk_and_write(gE, fRawData, maxChunkSize, remainChunkSize, nChunk, ngTypeE, ngTypeI, nV1);
    cout << "simulation done.\n";

    { // clean-up
        delete []outputB4V1;
		delete []galleryOutput;
        delete []V1_intra;
        delete []*d_gE;
        delete []*d_gI;
        delete []*d_hE;
        delete []*d_hI;
        delete []*gE;
        delete []*gI;
        delete []*hE;
        delete []*hI;
        delete []*conDelayMat;
        delete []*d_conDelayMat;
        delete []exact_norm;
        delete []exact_it;
        
        fRawData.close();
	    fLGN_fr.close();
        fLGN_gallery.close();
        fOutputB4V1.close();

    	checkCudaErrors(cudaStreamDestroy(mainStream));
    	checkCudaErrors(cudaEventDestroy(vReady));
    	checkCudaErrors(cudaEventDestroy(spReady));
        for (PosInt i=0; i<nChunk; i++) {
	        checkCudaErrors(cudaStreamDestroy(stream[i]));
        }
        delete []stream;
        checkCudaErrors(cudaFreeHost(pinnedMem));
        checkCudaErrors(cudaFreeHost(p_conDelayMat));
        dLGN.freeMem();
        checkCudaErrors(cudaFree(gpu_B4V1));
        checkCudaErrors(cudaFree(gpu_LGN_gallery));
        checkCudaErrors(cudaFree(d_mat));
        checkCudaErrors(cudaFree(d_vgh));
        checkCudaErrors(cudaFree(d_gh));
        checkCudaErrors(cudaFree(d_spikeTrain));
        checkCudaErrors(cudaFree(sLGN));
        checkCudaErrors(cudaFree(d_LGN_V1_surface));
        checkCudaErrors(cudaFreeArray(cuArr_L));
        checkCudaErrors(cudaFreeArray(cuArr_M));
        checkCudaErrors(cudaFreeArray(cuArr_S));
        checkCudaErrors(cudaFreeArray(cuSurfArray));
        checkCudaErrors(cudaDeviceReset());
        checkCudaErrors(cudaDeviceSynchronize());
        cout << "memory trace cleaned\n";
    }
    return EXIT_SUCCESS;
}
