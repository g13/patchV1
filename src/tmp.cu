#include <fstream>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include "discrete_input_convol.h"
#include "global.h"
#include "boost/program_options.hpp"

// the retinal discrete x, y as cone receptors id
void init_layer(texture<float, cudaTextureType2DLayered> &layer) {
    layer.addressMode[0] = cudaAddressModeBorder;
    layer.addressMode[1] = cudaAddressModeBorder; 
    layer.filterMode = cudaFilterModeLinear;
    layer.normalized = true; //accessing coordinates are normalized
}

void prep_sample(unsigned int iSample, unsigned int width, unsigned int height, float* L, float* M, float* S, cudaArray *dL, cudaArray *dM, cudaArray *dS, unsigned int nSample) {
    // copy the three channels L, M, S of the #iSample frame to the cudaArrays dL, dM and dS
    cudaMemcpy3DParms params = {0};
    params.srcPos = make_cudaPos(0, 0, 0);
    params.dstPos = make_cudaPos(0, 0, iSample);
    params.extent = make_cudaExtent(width, height, nSample);
    params.kind = cudaMemcpyHostToDevice;

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
    Float dt; // in ms, better in fractions of binary 
    Float ecc;
    Size width;
    Size height;
	Size nLGN_x;
	Size nLGN_y;
    SmallSize nSpatialSample1D; // spatial kernel sample size = nSpatialSample1D x nSpatialSample1D
    SmallSize nKernelSample; // spatial kernel sample size = nSpatialSample1D x nSpatialSample1D
    Float nsig = 3; // extent of spatial RF sampling in units of std
    Float tau = 256.0f; // required length of memory for LGN temporal kernel
    Float Itau = 300.0f; // in ms .. cone adaptation at 300ms https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003289
    PosInt frameRate; // Hz
    Size nt; // number of steps
    PosInt nKernelSample; // kernel sampling

	po::options_description generic_opt("Generic options");
	generic.add_options()
		("cfg_file,c", po::value<string>()->default_value("patchV1.cfg"), "filename for configuration file")
		("help,h", "print usage");
	po::options_description top_opt("top-level configuration");
	top_opt.add_options()
		("seed,s", po::value<PosInt>(&seed),"seed for trial")
		("dt", po::value<Float>(&dt)->default_value(0.125), "simulatoin time step") 
		("nt", po::value<Size>(&nt)->default_value(8000), "total simulatoin time in units of time step") // TODO: determine by stimulus
		("nSpatialSample1D", po::value<SmallSize>(&nSpatialSample1D)->default_value(warpSize), "number of samples per x,y direction for a LGN spatial RF")
		("tau", po::value<Float>(&tau)->default_value(256.0), "the backward time interval that a LGN temporal RF should cover")
		("Itau", po::value<Float>(&Itau)->default_value(150.0), "the light intensity adaptation time-scale of a cone")
		("nKernelSample", po::value<Size>(&nKernelSample)->default_value(256.0), "number of samples per x,y direction for LGN spatial RF")
		("frameRate", po::value<Float>(&frameRate)->default_value(60), "frame rate of the input stimulus");

    // files
	string stimulus_filename, V1_filename, LGN_filename, LGN_V1_s_filename, LGN_V1_ID_filename; // inputs
    string LGN_fr_filename; // outputs
	string LGN_convol_filename, max_convol_filename;
	top_opt.add_options()
		("fStimulus", po::value<string>(&stimulus_filename)->default_value("stimulus.bin"),"file that stores LGN firing rates, array of size (nframes,width,height,3)")
		("fV1", po::value<string>(&V1_filename)->default_value("V1.bin"),"file that stores V1 neurons information")
		("fLGN", po::value<string>(&LGN_filename)->default_value("LGN.bin"),"file that stores LGN neurons information")
		("fLGN_V1_ID", po::value<string>(&LGN_V1_ID_filename)->default_value("LGN_V1_idList.bin"),"file stores LGN to V1 connections")
		("fLGN_V1_s", po::value<string>(&LGN_V1_s_filename)->default_value("LGN_V1_sList.bin"),"file stores LGN to V1 connection strengths")
		("fLGN_fr", po::value<string>(&LGN_fr_filename)->default_value("LGN_fr.bin"),"file stores LGN firing rates")
		("fLGN_convol", po::value<string>(&LGN_convol_filename)->default_value("LGN_convol.bin"),"file that stores LGN convolution values") // TEST 
		("fLGN_max", po::value<string>(&max_convol_filename)->default_value("max_convol.bin"),"file that stores LGN maximum values of convolution"); // TEST
	
	po::options_description cmdline_options;
	cmdline_options.add(generic_opt).add(top_opt);

	po::options_description config_file;
	config_file_options.add(generic_opt).add(top_opt);

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
	if (vm.count("help")) {
		cout << cmdline_options << "\n";
		return EXIT_SUCCESS;
	}
	strin cfg_filename = vm["cfg_file"].as<string>();
	ifstream cfg_file{cfg_filename.c_str()};
	if (cfg_file) {
		po::store(po::parse_config_file(cfg_file, cfg_file_options), vm);
		cout << "Using configuration file: " << cfg_filename << "\n";
	} else {
		cout << "No configuration file is given, default values are used for non-specified paraeters\n";
	}
	po::notify(vm);

    printf("simulating for %u steps, t = %f ms\n", nt, nt*dt);

    if (nSpatialSample1D > 32) {
        cout << "nSpatialSample1D has to be smaller than 32 (1024 threads per block).\n"
        return EXIT_FAILURE;
    }

    //TODO: collect CUDA device properties to determine grid and block sizes
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, 0));
    printf("CUDA device [%s] has %d Multi-Processors ", deviceProps.name, deviceProps.multiProcessorCount);
    printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);

#ifdef SINGLE_PRECISION
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
#else
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
    printf("maximum threads per MP: %d.", deviceProps.maxThreadsPerMultiProcessor);


    Float init_luminance = 2.0/6.0; //1.0/6.0;

    // from the retina facing out
    const Float toRad = M_PI/180.0f;
	ifstream fStimulus, fLGN, fV1, fLGN; // inputs
    ofstream fLGN_fr // outputs
	ofstream fLGN_convol, fmax_convol;

    fStimulus.open(stimulus_filename, fstream::in | fstream::binary);
    Size nPixelPerFrame;
    Float deg0, ldeg; // VF in x-dimension: (deg0, deg0+ldeg); y-dimension: (-ldeg, ldeg)
    if (!fStimulus) {
        cout << "cannot open " << stimulus_filename << "\n";
        return EXIT_FAILURE;
    } else {
        vector<int> stimulus_dimensions(3, 0); 
        fStimulus.read(reinterpret_cast<char*>(&stimulus_dimensions[0]), 4 * sizeof(int));
        nFrame = stimulus_dimensions[0];
        height = stimulus_dimensions[1];
        width = stimulus_dimensions[2];
        vector<float> domain(2, 0); 
        fStimulus.read(reinterpret_cast<char*>(&domain[0]), 2 * sizeof(float));
        deg0 = domain[0];
        ldeg = domain[1];
        nPixelPerFrame = width*height;
        if (height != width) {
            // TODO:
            cout << "width != height, not implemented\n";
            return EXIT_FAILURE;
        }
    }
    
    Size nV1;
    Float midL = 0 - deg0;
    Float midR = ldeg - deg0;
	vector<Float> V1_x, V1_y;
    fV1.open(V1_filename, fstream::in | fstream::binary);
    if (!fV1) {
        fV1.read(reinterpret_cast<char*>(&nV1), sizeof(Size));
        fV1.read(reinterpret_cast<char*>(&V1_x[0]), nV1 * sizeof(Float));
	    fV1.read(reinterpret_cast<char*>(&V1_y[0]), nV1 * sizeof(Float));
        fV1.read(reinterpret_cast<char*>(&a[0]), nV1 * sizeof(Float));
        fV1.read(reinterpret_cast<char*>(&baRatio[0], nV1 * sizeof(Float));
	    fV1.read(reinterpret_cast<char*>(&sfreq[0], nV1 * sizeof(Float));
	    fV1.read(reinterpret_cast<char*>(&theta[0], nV1 * sizeof(Float));
	    fV1.read(reinterpret_cast<char*>(&phase[0], nV1 * sizeof(Float));
	    fV1.read(reinterpret_cast<char*>(&amp[0], nV1 * sizeof(Float));
	    fV1.read(reinterpret_cast<char*>(&sig[0], nV1 * sizeof(Float));
        cout << nV1 << " V1 neurons\n";
    }

    Size nLGN;
	vector<Float> LGN_x, LGN_y;
	vector<Int> OnOff;
    fLGN.open(LGN_filename, fstream::in | fstream::binary);
    if (!fLGN) {
	    fLGN_vpos.read(reinterpret_cast<char*>(&nLGN), sizeof(Int));
	    fLGN_vpos.read(reinterpret_cast<char*>(&LGN_x[0]), nLGN*sizeof(Float));
	    fLGN_vpos.read(reinterpret_cast<char*>(&LGN_y[0]), nLGN*sizeof(Float));
	    fLGN_vpos.read(reinterpret_cast<char*>(&on_off[0]), nLGN*sizeof(Int));
        cout << nLGN << " LGN neurons\n";
	    fLGN_vpos.close();
    }

	vector<vector<Int>> LGN_V1_ID = read_listOfList<Int>(LGN_V1_ID_filename, false);
	vector<vector<Int>> LGN_V1_s = read_listOfList<Float>(LGN_V1_s_filename, false);

    // output file
    fLGN_fr.open(LGN_fr_filename, fstream::out | fstream::binary);
    fLGN_convol.open(LGN_convol_filename, fstream::out | fstream::binary);
    fmax_convol.open(max_convol_filename, fstream::out | fstream::binary);

    bool simpleContrast = true; // no cone adaptation if set to true
    PosInt stepRate = round(1000/dt);
    if (round(1000/dt) != 1000/dt) {
        cout << "stepRate = 1000/dt has to be a integer.\n";
        return EXIT_FAILURE;
    }
    if (frameRate > stepRate) {
        cout << "stepRate cannot be smaller than frameRate, 1000/dt.\n";
        return EXIT_FAILURE;
    }

    const SmallSize nType = 2;
    Size nSample = nSpatialSample1D * nSpatialSample1D;
    height = width;


    // setup LGN here
    hLGN_parameter;

    // set test param for LGN subregion RF kernel 
    /*
        Group the same cone-specific types together for warp performance
		Check ./patchV1/LGN_kernel.ipynb
    */
    for (unsigned int i=0; i<nLGN; i++) {
		// subject to function of eccentricity
		hLGN.center.x[i] = 0.5f/width + (i%nLGN_x) * (1.0f-1.0f/width)/(nLGN_x-1);
		hLGN.center.y[i] = 0.5f/height + (i/nLGN_y) * (1.0f-1.0f/height)/(nLGN_y-1);
        hLGN.center.rx[i] = 0.015;		
		hLGN.center.ry[i] = 0.015;

		// subject to some distribution
		hLGN.center.k[i] = 16.5 * (i % 4 < 2 ? 1 : -1); // on - off center
        hLGN.center.tauR[i] = 5.2/1.55;
        hLGN.center.tauD[i] = 17/1.55;
        hLGN.center.nR[i] = 10;
        hLGN.center.nD[i] = 6;
		hLGN.center.delay[i] = 3.5+2.95;
        hLGN.center.ratio[i] = 25.9/16.5;

		// subject to function of eccentricity
		hLGN.surround.x[i] = hLGN.center.x[i];
		hLGN.surround.y[i] = hLGN.center.y[i];
        hLGN.surround.rx[i] = 0.03;  	// function of eccentricity
		hLGN.surround.ry[i] = 0.03;

		// subject to some distribution
        hLGN.surround.k[i] = -copy(6.4, hLGN.center.k[i]);
        hLGN.surround.tauR[i] = 8;
        hLGN.surround.tauD[i] = 14;
        hLGN.surround.nR[i] = 13;
        hLGN.surround.nD[i] = 6;
		hLGN.surround.delay[i] = 3.5+1.21+15;
		hLGN.surround.ratio[i] = 17.6/6.4;

        hLGN.covariant[i] = 0.53753461391295254; //between L and M, significantly overlap in cone response curve
        hLGN.centerType[i] = i%2;
        hLGN.surroundType[i] = (i+1)%2;

        hLGN.logistic.spont[i] = 0.0;
        hLGN.logistic.c50[i] = 0.5;
        hLGN.logistic.sharpness[i] = 1.0;
    }

    LGN_parameter dLGN(nLGN, hLGN);
    // finish LGN setup

    // set params for layerd texture memory
    init_layer(L_retinaConSig);
    init_layer(M_retinaConSig);
    init_layer(S_retinaConSig);

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
    Float kernelSampleDt = sampleInterval*dt;

    printf("temporal kernel retraces %f ms, samples %u points, sample rate = %u Hz\n", tau, nKernelSample, kernelSampleRate);

    PosInt maxFrame = static_cast<PosInt>(ceil(tau/1000 * frameRate) + 1; // max frame need to be stored in texture for temporal convolution with the LGN kernel.
    //  |---|--------|--|
    
    printf("temporal kernel retrace (tau) = %f ms needs at most %u frames\n", tau, maxFrame);
	_float tPerFrame = 1000.0f / frameRate; //ms
    printf("~ %f plusminus 1 kernel sample per frame\n", tPerFrame/kernelSampleDt);

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
        printf("i == %f", (exact_it[i] + (Float)exact_norm[i]/denorm)*dt*frameRate);
    }
    // i frames' length in steps = exact_it[i] + exact_norm[i]/denorm
    assert((stepRate*denorm) % frameRate == 0);
    unsigned int co_product = (stepRate*denorm)/frameRate; 
    // the number of minimum steps to meet frameLength * denorm with 0 phase

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

    // bind texture to cudaArrays
    checkCudaErrors(cudaBindTextureToArray(L_retinaConSig,  cuArr_L, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(M_retinaConSig,  cuArr_M, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(S_retinaConSig,  cuArr_S, channelDesc));

    float* LMS; // memory head

    // LMS frame intensity array of [maxFrame] frames from (t-tau) -> t on device
    float* __restrict__ L;
    float* __restrict__ M;
    float* __restrict__ S;

    // allocate the memory for a video of maxFrame with channels, and the memory for the contrast signal extracted from the video by sampling
    checkCudaErrors(cudaMalloc((void **) &LMS, nPixelPerFrame*sizeof(float)*3));

    L = LMS;
    M = L + nPixelPerFrame;
    S = M + nPixelPerFrame;

    Float* LGN_fr = new Float[nLGN];
    Float* __restrict__ d_LGNfr;
    Float* __restrict__ max_convol;
    checkCudaErrors(cudaMalloc((void **) &d_LGNfr, nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &max_convol, nLGN*sizeof(Float)));

    Float* decayIn;
    Float* lastF;
    Float* TW_storage;
    Float* SW_storage;
    Float* SC_storage;
    Float* dxdy_storage;
    checkCudaErrors(cudaMalloc((void **) &decayIn, nType*nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &lastF, nType*nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &TW_storage, nType*nKernelSample*nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &SW_storage, nType*nSample*nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &SC_storage, 2*nType*nSample*nLGN*sizeof(Float)));
    checkCudaErrors(cudaMalloc((void **) &dxdy_storage, 2*nType*nSample*nLGN*sizeof(Float)));

    checkCudaErrors(cudaMemset(decayIn, 0, nType*nLGN*sizeof(Float)));
    checkCudaErrors(cudaMemset(lastF, 0, nType*nLGN*sizeof(Float)));
    // initialize average to normalized mean luminnace
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
        prep_sample(0, width, height, tL, tM, tS, cuArr_L, cuArr_M, cuArr_S, maxFrame);
        checkCudaErrors(cudaFree(tLMS));
    }

    cudaEvent_t i0, i1, i2;

    checkCudaErrors(cudaEventCreate(&i0));
    checkCudaErrors(cudaEventCreate(&i1));
    checkCudaErrors(cudaEventCreate(&i2));

    dim3 convolBlock(nSpatialSample1D, nSpatialSample1D, 1);
    dim3 convolGrid(nLGN, 2, 1);

    // store spatial and temporal weights determine the maximums of LGN kernel convolutions
    store<<<convolGrid, convolBlock>>>(
            max_convol,
            d_LGN.temporal,
            TW_storage,
            nKernelSample,
            kernelSampleDt,
            kernelSampleT0,
            d_LGN.spatial,
            SW_storage,
            SC_storage,
            dxdy_storage,
            nsig,
            storeSpatial);
    checkCudaErrors(cudaMemcpy(LGNfr, max_convol, nLGN*sizeof(Float), cudaMemcpyDeviceToHost));
    fmax_convol.write((char*)LGNfr, nLGN*sizeof(Float));
    // calc LGN firing rate at the end of current dt
    unsigned int currentFrame = 0; // current frame number from stimulus
    unsigned int iFrame = 0; //latest frame inserted into the dL dM dS,  initialization not necessary
    unsigned int iPhase = 0;
    unsigned int iPhase_old = 0; // initialization not necessary
    Float framePhase;

    for (unsigned int it = 0; it < nt; it++) {
        Float t = it*dt;
        // next frame comes between (t, t+dt), read and store frame to texture memory
        if (it+1 > (currentFrame/denorm)*co_product + exact_it[iPhase]) {
            iFrame = currentFrame % maxFrame;
            if (fStimulus.eof()) { // if at the end of input file loop back to the beginning
                fStimulus.clear()
                fStimulus.seekg(0, ios::beg);
            }
            fStimulus.read(reinterpret_cast<char*>(LMS, 3*nPixelPerFrame*sizeof(float));
                        
            //cp to texture mem in device
            prep_sample(iFrame, width, height, L, M, S, cuArr_L, cuArr_M, cuArr_S, 1);
            printf("frame #%i prepared at t = %f, in (%f,%f) ~ %.3f%%.\n", currentFrame, currentFrame*tPerFrame, t, t+dt, exact_norm[iPhase]*100.0f/denorm);
            currentFrame++;
            iPhase = (iPhase + 1) % denorm;
        }
        if (it > nRetrace) {
            framePhase += dt;
            framePhase = fmod(framePhase, tPerFrame);
            //    -->|        |<-- framePhase
            //    |--|------frame------|
            //    |-----|-----|-----|-----|-----|
            //    jt-2, jt-1, jt ...nRetrace... it
        } else {
            framePhase = framePhase0;
        }
        // perform kernel convolution with built-in texture interpolation
        convolGrid.y = 1;
        LGN_convol_c1s<<<convolGrid, convolBlock, sizeof(Float)*2*convolBlock.x*convolBlock.y>>>(
                decayIn,
                lastF,
                SW_storage,
                SC_storage,
                dxdy_storage,
                TW_storage,
                d_LGNfr,
                d_LGN.coneType,
                d_LGN.spatial,
                nsig,
                iFrame,
                framePhase,
                Itau,
                kernelSampleDt,
                nKernelSample,
                dt,
                storeSpatial);

		getLastCudaError("LGN_convol_c1s failed");
        checkCudaErrors(cudaMemcpy(LGNfr, d_LGNfr, nLGN*sizeof(Float), cudaMemcpyDeviceToHost));
        fLGN_convol.write((char*)LGNfr, nLGN*sizeof(Float));

		// generate LGN fr with logistic function
        LGN_nonlinear<<<nLGN_x, nLGN_y>>>(d_LGNfr, d_LGN.logistic, max_convol);
		getLastCudaError("LGN_nonlinear failed");
        checkCudaErrors(cudaMemcpy(LGNfr, d_LGNfr, nLGN*sizeof(Float), cudaMemcpyDeviceToHost));
        fLGN_fr.write((char*)LGN_fr, nLGN*sizeof(Float));
    }

    checkCudaErrors(cudaDeviceSynchronize());
    delete []LGN_fr;
    delete []exact_norm;
    delete []exact_it;
    
    fLGN_fr.close();
    fLGN_convol.close();
    fmax_convol.close();

    dLGN.freeMem();
    hLGN.freeMem();
	checkCudaErrors(cudaStreamDestroy(s0));
    checkCudaErrors(cudaStreamDestroy(s1));
    checkCudaErrors(cudaStreamDestroy(s2));
    checkCudaErrors(cudaFree(LMS));
    checkCudaErrors(cudaFree(d_LGNfr));
    checkCudaErrors(cudaFree(max_convol));
    checkCudaErrors(cudaFreeArray(cuArr_L));
    checkCudaErrors(cudaFreeArray(cuArr_M));
    checkCudaErrors(cudaFreeArray(cuArr_S));
    checkCudaErrors(cudaFreeArray(cuArr_frame));
    cudaDeviceReset();
    return 0;
}
