#include "patch.h"

int main(int argc, char *argv[])
{
    std::ofstream v_file, spike_file, gE_file, gI_file;
    float time;
    //cudaEventCreateWithFlags(&gReady, cudaEventDisableTiming);
    curandStateMRG32k3a *state, *randState;
    unsigned long long seed;
    //seed = 183765712;
    seed = std::time(0);
    int device;
    int b1,b2;
	int ms = 1;
    b1 = 160;
    b2 = 128;
    bool printStep = false;
    bool moreSharedMemThanBlocks = true;
    double flatRate = 80.0f; // kHz
    double t = 2.5f;
	unsigned int mt = 1;
    unsigned int nstep = 200;
    double ffsE = 1e-3;
    double s0 = 0e-2*ffsE;
    double ffsI = 5e-2; // no effect here for 1st layer ffw inputs
    int iFlatRate = -1;
	char tmp[101];
    /* Overwrite parameters */
    for (int i = 0; i<argc; i++) {
        printf(argv[i]);
        printf(" ");
    }
    printf("\n");
    if (argc == 2) {
        sscanf(argv[argc-1],"%u",&seed); 
    }
    if (argc == 3) {
        sscanf(argv[argc-1],"%d",&b2); 
        sscanf(argv[argc-2],"%d",&b1); 
    }
    if (argc == 4) {
        sscanf(argv[argc-1],"%d",&b2); 
        sscanf(argv[argc-2],"%d",&b1); 
        sscanf(argv[argc-3],"%d",&nstep);
    }
    if (argc == 5) {
        sscanf(argv[argc-1],"%u",&seed);
        sscanf(argv[argc-2],"%d",&b2);
        sscanf(argv[argc-3],"%d",&b1);
        sscanf(argv[argc-4],"%d",&nstep);
    }
	if (argc == 6) {
		sscanf(argv[argc-1], "%d", &iFlatRate);
		sscanf(argv[argc-2], "%u", &seed);
		sscanf(argv[argc-3], "%d", &b2);
		sscanf(argv[argc-4], "%d", &b1);
		sscanf(argv[argc-5], "%d", &nstep);
	}
	if (argc == 8) {
		sscanf(argv[argc-1], "%100s", tmp);
		sscanf(argv[argc-2], "%d", &mt);
		sscanf(argv[argc-3], "%d", &iFlatRate);
		sscanf(argv[argc-4], "%u", &seed);
		sscanf(argv[argc-5], "%d", &b2);
		sscanf(argv[argc-6], "%d", &b1);
		sscanf(argv[argc-7], "%d", &nstep);
	}
	std::string theme = tmp;
	std::cout << "theme = " << theme << "\n";
    if (!theme.empty()) {
        theme = '-'+theme;
    }
    printf("%i x %i, %i steps, seed = %u\n", b1, b2, nstep, seed);
	unsigned int networkSize = b1*b2;
	double s = s0/(networkSize);
	assert(s == 0);
    if (networkSize/10.0 != float(networkSize/10)) {
        printf("To have higher computation occupancy make a factor of 10 in networkSize\n");
    }
	int warpSize = 32;
	int init_b2 = warpSize;
	int init_b1 = networkSize / init_b2;
    double eiRatio = 3.0f/4.0f;
    int b1E = b1;
    int b2E = b2*eiRatio;
    int b1I = b1;
    int b2I = b2*(1-eiRatio);
    unsigned int nE = networkSize*eiRatio;
    unsigned int nI = networkSize-nE;
	t = mt * t;
    double dt = t/float(nstep); // ms
    if (iFlatRate < 0) {
        flatRate = flatRate*1000.0; // Hz
    } else {
        flatRate = iFlatRate*1000.0;
    }
    /* to be extended */
    bool presetInit = false;
    unsigned int ngTypeE = 2;
    unsigned int ngTypeI = 1;
    double riseTimeE[2] = {1.0f, 5.0f}; // ms
    double riseTimeI[1] = {1.0f};
    double decayTimeE[2] = {3.0f, 80.0f};
    double decayTimeI[1] = {5.0f};

    ConductanceShape condE(riseTimeE, decayTimeE, ngTypeE);
    ConductanceShape condI(riseTimeI, decayTimeI, ngTypeI);
    printf("designated input rate = %3.1fHz\n", flatRate);
	printf("dt = %f ms\n", dt);
    printf("nE = %i, nI = %i\n", nE, networkSize-nE);
    printf("t = %f x %i = %f\n", dt, nstep, t);
	double fInput = flatRate / 1000.0f * dt;
	int _nInput = ceil(fInput);
    int nskip = 1;
	if (fInput < 1.0) {
        nskip = round(_nInput/fInput);
    }
	if (networkSize / float(warpSize) != float(networkSize / warpSize)) {
		printf("please make networkSize multiples of %i to run on GPU\n", warpSize);
		return EXIT_FAILURE;
	}
    struct cudaDeviceProp properties;  
    double *v, *gE, *gI, *preMat; 
    int *eventRate, *d_eventRate;
    double *d_v1, *d_v2, *dv, *d_gE, *d_gI, *d_hE, *d_hI, *d_fE, *d_fI, *d_preMat, *d_inputRate;
    double *d_a0, *d_b0, *d_a1, *d_b1, *v_current, *v_old, *d_v_hlf;
    double *gactVec, *hactVec;
    double *leftTimeRate, *lastNegLogRand;
    double *spikeTrain, *d_spikeTrain, *tBack;
    bool *not_matched;
    bool *d_not_matched;

    while (init_b2 < 256 && init_b1 > 1) {
        init_b2 = init_b2*2;
        init_b1 = init_b1/2;
    }
    printf("init size %i, %i\n", init_b1, init_b2);

	/* check for double precision support */
	CUDA_CALL(cudaGetDevice(&device));
	CUDA_CALL(cudaGetDeviceProperties(&properties, device));
	if (!(properties.major >= 2 || (properties.major == 1 && properties.minor >= 3))) {
		printf(" double precision not supported\n");
		return EXIT_FAILURE;
	}
    /* inits that used by both cpu and gpu */
    CUDA_CALL(cudaMalloc((void **)&randState, networkSize * sizeof(curandStateMRG32k3a)));

    preMat = new double[networkSize * networkSize];
    CUDA_CALL(cudaMalloc((void **)&d_preMat, networkSize * networkSize * sizeof(double)));

    CUDA_CALL(cudaMallocHost((void**)&v, networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_v1,           networkSize * sizeof(double)));

    randInit<<<init_b1,init_b2>>>(d_preMat, d_v1, randState, s, networkSize, seed);
    CUDA_CHECK();
    CUDA_CALL(cudaMemcpy(preMat, d_preMat, networkSize*networkSize*sizeof(double),cudaMemcpyDeviceToHost));
	double sum = 0;
	for (unsigned int i = 0; i < networkSize*networkSize; i++) {
		sum += preMat[i];
	}
	printf("sum of preMat = %f\n", sum);
    CUDA_CALL(cudaMemcpy(v, d_v1, networkSize*sizeof(double),cudaMemcpyDeviceToHost));
    printf("storage size of preMat %.1fMb\n", float(networkSize*networkSize*sizeof(double))/1024.0/1024.0);

	#ifdef TEST_WITH_MANUAL_FFINPUT
		printf("for testing purpose, feedforward input is set to %i per %fms\n", _nInput, dt*nskip);
        printf("realized input rate = %f kHz\n", _nInput/(dt*nskip));
		printf("for manual testing, please change the inputTime manually in source and recompile\n");
        cpu_version(networkSize, _nInput, nskip, nstep, dt, nE, preMat, v, ffsE, ffsI, theme);
    #endif

    unsigned int nbatch, batchEnd, batchStep;
    unsigned int ngType;
    if (ngTypeE > ngTypeI) {
        ngType = ngTypeE;
    } else {
        ngType = ngTypeI;
    }
    // v, gE, gI, spikeTrain
    unsigned int hostMemToDiskPerStep = ceil(networkSize * (sizeof(double) + ngTypeE*sizeof(double) + ngTypeI*sizeof(double) + sizeof(int) )/(1024*1024));
    //batchStep = floor(HALF_MEMORY_OCCUPANCY/hostMemToDiskPerStep);
    batchStep = 1;
    if (batchStep < 10) {
        printf("consider increase HALF_MEMORY_OCCUPANCY, batch step = %i\n", batchStep);
    } else {
        if (batchStep == 0) {
            printf("increase HALF_MEMORY_OCCUPANCY, memory to write on disk per step: %i Mb", hostMemToDiskPerStep);
            return EXIT_FAILURE;
        }
    }
    nbatch = nstep/batchStep; 
    batchEnd = nstep - batchStep*nbatch;
    int alt = 1;
    cudaEvent_t iStart, iStop;
    cudaEventCreate(&iStart);
    cudaEventCreate(&iStop);
    CUDA_CALL(cudaEventRecord(iStart, 0));
    /* Allocate space for results on host */
    //pinned memory
    CUDA_CALL(cudaMallocHost((void**)&gE,          networkSize * ngTypeE * sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&gI,          networkSize * ngTypeI *sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&spikeTrain,  networkSize * sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&eventRate,   networkSize * sizeof(int) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&not_matched, networkSize * sizeof(bool)));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&d_v2,           networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&dv,             networkSize * sizeof(double)));
	CUDA_CALL(cudaMalloc((void **)&d_v_hlf,        networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_gE,           networkSize * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_gI,           networkSize * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_hE,           networkSize * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_hI,           networkSize * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_fE,           networkSize * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_fI,           networkSize * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_a0,            networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_b0,            networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_a1,            networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_b1,            networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_inputRate,    networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_eventRate,    networkSize * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&d_spikeTrain,   networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&tBack,          networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gactVec,        networkSize * ngType * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hactVec,        networkSize * ngType * sizeof(double)));
    /* Allocate space for rng on device */
    CUDA_CALL(cudaMalloc((void **)&state,          networkSize * sizeof(curandStateMRG32k3a)));
    CUDA_CALL(cudaMalloc((void **)&leftTimeRate,   networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&lastNegLogRand, networkSize * sizeof(double)));
    /* Allocate space for partial reduce results on device */
    /* Allocate variables that allow write-write conflict for global-OR operation on device*/
    CUDA_CALL(cudaMalloc((void **)&d_not_matched, networkSize * sizeof(bool)));
    
    int maxTPB = properties.maxThreadsPerBlock/ms;
    int EmaxTPB, ImaxTPB;
    int mE, mI; 
    if (maxTPB < nE) {
        EmaxTPB = maxTPB;
        mE = (nE+EmaxTPB-1)/EmaxTPB;
        EmaxTPB = nE/mE;
    } else {
        mE = 1;
        EmaxTPB = nE;
    }
    while (EmaxTPB*mE != nE && EmaxTPB > EmaxTPB/2) {
        mE = mE + 1;
        EmaxTPB = nE/mE;
    }

    if (maxTPB < nI) {
        ImaxTPB = maxTPB;
        mI = (nI+ImaxTPB-1)/ImaxTPB;
        ImaxTPB = nI/mI;
    } else {
        mI = 1;
        ImaxTPB = nI;
    }
    while (ImaxTPB*mI != nI && ImaxTPB > ImaxTPB/2) {
        mI = mI + 1;
        ImaxTPB = nI/mI;
    }

    dim3 rgE_b1, rgI_b1;
    int EnTPB = networkSize/(networkSize/EmaxTPB);
    int InTPB = networkSize/(networkSize/ImaxTPB);
    if (EnTPB > maxTPB) {
        EnTPB = maxTPB;
    }
    if (InTPB > maxTPB) {
        InTPB = maxTPB;
    }
    dim3 rgE_b2(EnTPB,1);
    dim3 rgI_b2(InTPB,1);
    printf("mE = %i, mI = %i\n", mE, mI);
    //dim3 rgE_b2(EmaxTPB,1);
    //dim3 rgI_b2(ImaxTPB,1);
    int msE = 1; // multiple shared actVec load per thread
    int msI = 1;
    int s_actVec_lE; // length of shared actVec
    int s_actVec_lI;
    unsigned int rgE_shared;
    unsigned int rgI_shared;

    s_actVec_lE = EmaxTPB;
    rgE_shared = 2*ngTypeE*s_actVec_lE*sizeof(double);
    if (rgE_shared > properties.sharedMemPerBlock) {
        printf("E: The size of the requested shared memory %iKb by recal_G is not available\n", float(rgE_shared)/1024);
        return EXIT_FAILURE;
    } else {
        if (moreSharedMemThanBlocks) {
            while (rgE_shared*2  < properties.sharedMemPerBlock && mE/float(msE*2) == float(mE/(msE*2))) {
                msE = msE * 2;
                rgE_shared = rgE_shared * 2;
            }
        }
    }
    s_actVec_lE = msE*s_actVec_lE; // number of actVec each chunk dump into shared mem, msE multiples of maxTPB
    rgE_b1.x = nE/s_actVec_lE; // chunks of maxTPB neurons
    rgE_b1.y = networkSize/EnTPB; // total number of presynaptic neurons divided by the the shared actVec
    printf("E: recal_G<<<(%i,%i,%i)x(%i,%i,%i), %iKb>>>, msE = %i\n", rgE_b1.x, rgE_b1.y, rgE_b1.z, rgE_b2.x, rgE_b2.y, rgE_b2.z, rgE_shared/1024, msE);

    s_actVec_lI = ImaxTPB;
    rgI_shared = 2*ngTypeI*s_actVec_lI*sizeof(double);
    if (rgI_shared > properties.sharedMemPerBlock) {
        printf("I: The size of the requested shared memory %iKb by recal_G is not available\n", float(rgI_shared)/1024);
        return EXIT_FAILURE;
    } else {
        if (moreSharedMemThanBlocks) {
            while (rgI_shared*2  < properties.sharedMemPerBlock && mI/float(msI*2) == float(mI/(msI*2))) {
                msI = msI * 2;
                rgI_shared = rgI_shared * 2;
            }
        }
    }
    s_actVec_lI = msI*s_actVec_lI;
    rgI_b1.x = nI/s_actVec_lI;
    rgI_b1.y = networkSize/InTPB;
    printf("I: recal_G<<<(%i,%i,%i)x(%i,%i,%i), %iKb>>>, msI = %i\n", rgI_b1.x, rgI_b1.y, rgI_b1.z, rgI_b2.x, rgI_b2.y, rgI_b2.z, rgI_shared/1024, msI);

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    int rE_b2, rI_b2;
    double *gE_b1x, *gI_b1x, *hE_b1x, *hI_b1x;
    if (rgE_b1.x >= 32) {
        int e = 5;
        while (rgE_b1.x > 1<<e) e++;
        rE_b2 = 1<<e;
        printf("blockdims for reduction of %i per thread : %i x %i \n", rgE_b1.x, networkSize, rE_b2);
        CUDA_CALL(cudaMalloc((void **)&gE_b1x,  networkSize * rE_b2 * ngTypeE * sizeof(double)));
        CUDA_CALL(cudaMalloc((void **)&hE_b1x,  networkSize * rE_b2 * ngTypeE * sizeof(double)));
    }
    if (rgI_b1.x >= 32) {
        int e = 5;
        while (rgI_b1.x > 1<<e) e++;
        rI_b2 = 1<<e;
        printf("blockdims for reduction of %i per thread : %i x %i \n", rgI_b1.x, networkSize, rE_b2);
        CUDA_CALL(cudaMalloc((void **)&gI_b1x,  networkSize * rI_b2 * ngTypeI * sizeof(double)));
        CUDA_CALL(cudaMalloc((void **)&hI_b1x,  networkSize * rI_b2 * ngTypeI * sizeof(double)));
    }




    /* Create CUDA events */
    cudaEvent_t start, stop, spikeCorrected, gReadyE, gReadyI, vReady, initialSpikesObtained, spikeRateReady, eventRateReady;
    cudaEvent_t kStart, kStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&gReadyE);
    cudaEventCreate(&gReadyI);
    cudaEventCreate(&vReady);
    cudaEventCreate(&spikeCorrected);
    cudaEventCreate(&initialSpikesObtained);
    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);
    cudaEventCreate(&spikeRateReady);
    cudaEventCreate(&eventRateReady);
    /* Initialize device arrays */
    // CUDA streams for init
    cudaStream_t i1, i2, i3, i4, i5, i6, i7;
    CUDA_CALL(cudaStreamCreate(&i1));
    CUDA_CALL(cudaStreamCreate(&i2));
    CUDA_CALL(cudaStreamCreate(&i3));
    CUDA_CALL(cudaStreamCreate(&i4));
    CUDA_CALL(cudaStreamCreate(&i5));
    CUDA_CALL(cudaStreamCreate(&i6));
    CUDA_CALL(cudaStreamCreate(&i7));
    if (!presetInit) {
        for (unsigned int i=0; i<networkSize; i++) {
            gE[i] = 0.0f;
            gI[i] = 0.0f;
            spikeTrain[i] = dt;
        }
        // init rand generation for poisson
        logRand_init<<<init_b1,init_b2,0,i1>>>(lastNegLogRand, state, seed);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i2>>>(d_inputRate, flatRate/1000.0f);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i4>>>(d_v2, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i5>>>(leftTimeRate, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i6>>>(tBack, -1.0f); 
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeE,init_b2,0,i7>>>(d_fE, ffsE);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i1>>>(d_fI, ffsI);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeE,init_b2,0,i2>>>(d_gE, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i3>>>(d_gI, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeE,init_b2,0,i4>>>(d_hE, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i5>>>(d_hI, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngType,init_b2,0,i6>>>(gactVec, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngType,init_b2,0,i7>>>(hactVec, 0.0f);
        CUDA_CHECK();
        //CUDA_CALL(cudaEventRecord(kStart, 0));
        //CUDA_CALL(cudaEventRecord(kStop, 0));
        //CUDA_CALL(cudaEventSynchronize(kStop));
        //CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
        //printf("logRand_init<<<%ix%i>>> cost %.1fms\n", init_b1*init_b1*init_b2, init_b2, time);
    }
    CUDA_CALL(cudaStreamDestroy(i1));
    CUDA_CALL(cudaStreamDestroy(i2));
    CUDA_CALL(cudaStreamDestroy(i3));
    CUDA_CALL(cudaStreamDestroy(i4));
    CUDA_CALL(cudaStreamDestroy(i5));
    CUDA_CALL(cudaStreamDestroy(i6));
    CUDA_CALL(cudaStreamDestroy(i7));
    CUDA_CALL(cudaEventRecord(iStop, 0));
    CUDA_CALL(cudaEventSynchronize(iStop));
    CUDA_CALL(cudaEventElapsedTime(&time, iStart, iStop));
    printf("initialization cost %fms\n", time);

    /* Create CUDA streams */
    cudaStream_t s1, s2, s3, s4, s5;
    CUDA_CALL(cudaStreamCreate(&s1));
    CUDA_CALL(cudaStreamCreate(&s2));
    CUDA_CALL(cudaStreamCreate(&s3));
    CUDA_CALL(cudaStreamCreate(&s4));
    CUDA_CALL(cudaStreamCreate(&s5));
    unsigned int shared_mem = 0;
    v_file.open("v_ictorious" + theme + ".bin", std::ios::out|std::ios::binary);
    spike_file.open("s_uspicious" + theme + ".bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_nerous" + theme + ".bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_berish" + theme + ".bin", std::ios::out|std::ios::binary);
    CUDA_CALL(cudaEventRecord(start, 0));
    double events = 0.0f;
    int spikes = 0;
    unsigned int ibatch = 0;
    unsigned int bStep = 0;
    unsigned int batchOffset = 0;
    unsigned int copySize = batchStep;
    unsigned int n = networkSize*copySize;
    CUDA_CALL(cudaEventRecord(gReadyE, 0));
    CUDA_CALL(cudaEventRecord(gReadyI, 0));
    CUDA_CALL(cudaEventRecord(vReady, 0));
    
    //for (int ibatch=0; i<nbatch; ibatch++) {
    //    if(ibatch == nbatch-1) {
    //        copySize = batchEnd;
    //    }
        bool it = true;
        double timeV = 0.0f;
        double timeG = 0.0f;
        double timeS = 0.0f;
        double timeP = 0.0f;
        unsigned int nmatch = 0;
        int iskip = 1;
        int nInput;
        for (int i=0; i<nstep; i++) {
            if (iskip % nskip == 0) {
                nInput = _nInput;
            } else {
                nInput = 0;
            }
            unsigned int offset;
            //offset = networkSize*(batchOffset + i);
            offset = 0;
            /* Write voltage to file */
            CUDA_CALL(cudaEventSynchronize(vReady));
            v_file.write((char*)v, networkSize * sizeof(double));
            /* Write conductance of last step to disk */
            CUDA_CALL(cudaEventSynchronize(gReadyE));
            gE_file.write((char*)&(gE[n*ngTypeE*batchOffset]),     n*ngTypeE*sizeof(double));
            CUDA_CALL(cudaEventSynchronize(gReadyI));
            gI_file.write((char*)&(gI[n*ngTypeI*batchOffset]),     n*ngTypeI*sizeof(double));
            /* Compute voltage (acquire initial spikes) */
            if (i%2 == 0) {
                v_current = d_v2;
                v_old = d_v1;
            } else {
                v_current = d_v1;
                v_old = d_v2;
            }
            #ifdef KERNEL_PERFORMANCE
                CUDA_CALL(cudaEventRecord(kStart, 0));
            #endif
            compute_dV <<<b1, b2, shared_mem, s1>>> (v_old, dv, d_gE, d_gI, d_hE, d_hI, d_a0, d_b0, d_a1, d_b1, d_preMat, d_inputRate, d_eventRate, d_spikeTrain, tBack, gactVec, hactVec, d_fE, d_fI, leftTimeRate, lastNegLogRand, d_v_hlf, state, ngTypeE, ngTypeI, ngType, condE, condI, dt, networkSize, nE, seed, nInput, it);
            CUDA_CHECK();
            CUDA_CALL(cudaEventRecord(initialSpikesObtained, s1));
            CUDA_CALL(cudaEventRecord(kStop, 0));
            CUDA_CALL(cudaEventSynchronize(kStop));
            CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
            timeV += time;
            if (printStep) {
                printf("A single step of compute_V cost %fms\n", time);
            }
            // Copy feedforward eventRate to file
            CUDA_CALL(cudaStreamWaitEvent(s1, initialSpikesObtained, 0));
            CUDA_CALL(cudaMemcpyAsync(eventRate, d_eventRate, networkSize * sizeof(int), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaEventRecord(eventRateReady, s1));
            /* Spike correction */
            bool no_match = true;
            unsigned int imatch = 0;
            CUDA_CALL(cudaEventRecord(kStart, 0));
            while (no_match && imatch < networkSize) {
                //printf("correct_spike <<< %i x %i >>>\n", b1, b2);
                correct_spike <<<b1, b2, 0, s1>>> (d_not_matched, d_spikeTrain, d_v_hlf, v_old, dv, d_a0, d_b0, d_a1, d_b1, v_current, d_preMat, tBack, ngTypeE, ngTypeI, condE, condI, dt, nE, networkSize);
                CUDA_CHECK();
	            //printf("correct_spike <<< %i x %i >>> finished\n", b1, b2);
                CUDA_CALL(cudaMemcpyAsync(not_matched, d_not_matched, networkSize * sizeof(bool), cudaMemcpyDeviceToHost,s1));
                CUDA_CALL(cudaEventRecord(spikeCorrected, s1));
                no_match = false;
                CUDA_CALL(cudaEventSynchronize(spikeCorrected)); 
                //printf("synchronized\n");
				//CUDA_CALL(cudaDeviceSynchronize());
                for (int iw = 0; iw < warpSize; iw ++) {
					//printf("iw = %i\n", iw);
                    bool iw_no_match = false;
                    for (unsigned int j=0; j<networkSize/warpSize; j++) {
                        unsigned int wid = j*warpSize + iw;
						//printf("%i = %i*%i + %i\n", wid, j, warpSize, iw);
                        if (wid >= networkSize) {
                            break;
                        }
                        if (not_matched[j*warpSize + iw]) {
                            iw_no_match = true;
                            break;
                        }
                    }
                    if (iw_no_match) {
                        no_match = true;
                        break;
                    }
                }
                imatch++;
            }
            nmatch += imatch;
            //printf("%i matching iterations\n", imatch);
            #ifdef KERNEL_PERFORMANCE
                CUDA_CALL(cudaEventRecord(kStop, 0));
                CUDA_CALL(cudaEventSynchronize(kStop));
                CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
                if (printStep) {
                    printf("spike correction of step %u cost %fms\n", i, time);
                }
                timeS += time;
            #endif
            // Finalize current step
            #ifdef KERNEL_PERFORMANCE
                CUDA_CALL(cudaEventRecord(kStart, 0));
            #endif
            // Copy voltage and spikeTrain of the current step to host (first E then I)
            CUDA_CALL(cudaMemcpyAsync(spikeTrain, d_spikeTrain, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaEventRecord(spikeRateReady, s1));
            CUDA_CALL(cudaMemcpyAsync(v, v_current, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaEventRecord(vReady, s1));
            // Get presynaptic conductance ready
            //printf("prepare_cond <<< %i x %i >>> %i, %i, %f \n", b1E, b2E, b1, b2, eiRatio);
            prepare_cond <<<b1E, b2E, 0, s2>>> (tBack, d_spikeTrain, gactVec, hactVec, condE, dt, ngTypeE, 0, networkSize);
            CUDA_CHECK();
            CUDA_CALL(cudaEventRecord(gReadyE, s2));
            prepare_cond <<<b1I, b2I, 0, s3>>> (tBack, d_spikeTrain, gactVec, hactVec, condI, dt, ngTypeE, nE, networkSize);
            CUDA_CHECK();
            CUDA_CALL(cudaEventRecord(gReadyI, s3));
            #ifdef KERNEL_PERFORMANCE
                CUDA_CALL(cudaEventRecord(kStop, 0));
                CUDA_CALL(cudaEventSynchronize(kStop));
                CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
                if (printStep) {
                    printf("establish the current step cost %fms\n", time);
                }
                timeP += time;
            #endif

            /* Recalibrate conductance to postsynaptic neurons, for the next step*/
            // recal E
            #ifdef KERNEL_PERFORMANCE
                CUDA_CALL(cudaEventRecord(kStart, 0));
            #endif
            CUDA_CALL(cudaStreamWaitEvent(s2, gReadyI, 0));
            recal_G <<<rgE_b1,rgE_b2,rgE_shared,s2>>> (d_gE, d_hE, d_preMat,
                                                     gactVec, hactVec,
                                                     gE_b1x, hE_b1x,
                                                     networkSize, 0, ngTypeE, s_actVec_lE, msE);
            CUDA_CHECK();
            // recal I
            CUDA_CALL(cudaStreamWaitEvent(s3, gReadyE, 0));
            recal_G<<<rgI_b1,rgI_b2,rgI_shared,s3>>>(d_gI, d_hI, d_preMat,
                                                     gactVec, hactVec,
                                                     gI_b1x, hI_b1x,
                                                     networkSize, nE, ngTypeI, s_actVec_lI, msI);
            CUDA_CHECK();
            if (rgE_b1.x >= 32) {
                //  reduce sum
                reduce_G<<<networkSize, rE_b2, sizeof(double)*2*rE_b2, s2>>>(d_gE, d_hE, gE_b1x, hE_b1x, ngTypeE, rgE_b1.x);
                CUDA_CHECK();
            }
            // copy exc conductance to host
            CUDA_CALL(cudaMemcpyAsync(gE, d_gE, networkSize * ngTypeE * sizeof(double), cudaMemcpyDeviceToHost, s2));
            if (rgI_b1.x >= 32) {
                reduce_G<<<networkSize, rI_b2, sizeof(double)*2*rI_b2, s3>>>(d_gI, d_hI, gI_b1x, hI_b1x, ngTypeI, rgI_b1.x);
                CUDA_CHECK();
            }
            // copy inh conductance to host
            CUDA_CALL(cudaMemcpyAsync(gI, d_gI, networkSize * ngTypeI * sizeof(double), cudaMemcpyDeviceToHost, s3));
            #ifdef KERNEL_PERFORMANCE
                CUDA_CALL(cudaEventRecord(kStop, 0));
                CUDA_CALL(cudaEventSynchronize(kStop));
                CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
                if (printStep) {
                    printf("A single step of recal_G cost %fms\n", time);
                }
                timeG += time;
            #endif
            CUDA_CALL(cudaEventRecord(gReadyE, s2));
            CUDA_CALL(cudaEventRecord(gReadyI, s3));
            //printf("\r total: %3.1f, batch: %3.1f", 100.0f*float(ibatch+1)/nbatch, float(i)/copySize);
            CUDA_CALL(cudaEventSynchronize(eventRateReady));
            CUDA_CALL(cudaEventSynchronize(spikeRateReady));
            /* Copy spikeTrain to host */
            spike_file.write((char*)spikeTrain,  n*sizeof(double));
            printf("\r stepping: %3.1f%%", 100.0f*float(i+1)/nstep);
            //printf("stepping: %3.1f%%\n", 100.0f*float(i+1)/nstep);
            fflush(stdout);
            it = false;
            double _events = 0.0f;
            int _spikes = 0;
            for (int j=0; j<networkSize; j++) {
                _events += eventRate[j];
                if (spikeTrain[j] < dt) {
                    _spikes++;
                    //printf("s[%i] = %f, ", j, spikeTrain[j]);
                }
            }
            //printf("\n");
            events += _events;
            spikes += _spikes;
            if (printStep) {
                printf("instant input rate = %fkHz, dt = %f, networkSize = %i\n", _events/(dt*networkSize), dt, networkSize);
                printf("instant firing rate = %fHz\n", _spikes/(dt*networkSize)*1000.0);
            }
            if (batchOffset == 0) {
                batchOffset = 0;
                //batchOffset = batchStep;
            } else {
                batchOffset = 0;
            }
            iskip++;
        }
        /* WHen hit HALF_MEMORY_OCCUPANCY, write half of the array to disk, the other half left to receive from device */
        // Alternating
        // switch batchOffset
    //}

    v_file.write((char*)v, networkSize * sizeof(double));
    //spike_file.write((char*)spikeTrain, networkSize * sizeof(int));
    gE_file.write((char*)gE, networkSize * ngTypeE * sizeof(double));
    gI_file.write((char*)gI, networkSize * ngTypeI * sizeof(double));
    printf("\n");

    printf("flatRate = %fkHz, realized mean input rate = %fkHz\n", flatRate/1000.0, float(events)/(dt*nstep*networkSize));
    printf("mean firing rate = %fHz\n", float(spikes)/(dt*nstep*networkSize)*1000.0);

    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&time, start, stop));
    printf("CUDA takes %fms, runtime/realtime ratio ms %fms\n", time, time/(dt*nstep));
    printf("compute_V takes %fms, ratio ms %fms\n", timeV, timeV/(dt*nstep));
    printf("recal_G takes %fms, ratio ms %fms\n", timeG, timeG/(dt*nstep));
    printf("correct_spike takes %fms, ratio ms %fms\n", timeS, timeS/(dt*nstep));
    printf("matching %i times, %f per step\n", nmatch, float(nmatch)/nstep);
    printf("prepare_cond and establish_v takes %fms, ratio ms %fms\n", timeP, timeP/(dt*nstep));

    /* Cleanup */
    printf("Cleaning up:\n");
    CUDA_CALL(cudaStreamDestroy(s1));
    CUDA_CALL(cudaStreamDestroy(s2));
    CUDA_CALL(cudaStreamDestroy(s3));
    printf("    CUDA streams destroyed\n");
    if (v_file.is_open()) v_file.close();
    if (spike_file.is_open()) spike_file.close();
    if (gE_file.is_open()) gE_file.close();
    if (gI_file.is_open()) gI_file.close();
    printf("    Output files closed\n");
    
    CUDA_CALL(cudaFree(state));
    CUDA_CALL(cudaFree(d_v1));
    CUDA_CALL(cudaFree(d_v2));
    CUDA_CALL(cudaFree(d_gE));
    CUDA_CALL(cudaFree(d_gI));
    CUDA_CALL(cudaFree(d_hE));
    CUDA_CALL(cudaFree(d_hI));
    CUDA_CALL(cudaFree(d_fE));
    CUDA_CALL(cudaFree(d_fI));
    CUDA_CALL(cudaFree(dv));
    CUDA_CALL(cudaFree(gactVec));
    CUDA_CALL(cudaFree(hactVec));
    CUDA_CALL(cudaFree(d_preMat));
    CUDA_CALL(cudaFree(d_a0));
    CUDA_CALL(cudaFree(d_b0));
    CUDA_CALL(cudaFree(d_a1));
    CUDA_CALL(cudaFree(d_b1));
    if (rgE_b1.x >= 32) {
        CUDA_CALL(cudaFree(gE_b1x));
        CUDA_CALL(cudaFree(hE_b1x));
    }
    if (rgI_b1.x >= 32) {
        CUDA_CALL(cudaFree(gI_b1x));
        CUDA_CALL(cudaFree(hI_b1x));
    }
    CUDA_CALL(cudaFree(leftTimeRate));
    CUDA_CALL(cudaFree(lastNegLogRand));
    CUDA_CALL(cudaFree(d_inputRate));
    CUDA_CALL(cudaFree(d_eventRate));
    CUDA_CALL(cudaFree(d_spikeTrain));
	CUDA_CALL(cudaFree(d_not_matched));
    CUDA_CALL(cudaFree(tBack));
    printf("    Device memory freed\n");
    CUDA_CALL(cudaFreeHost(v));
    CUDA_CALL(cudaFreeHost(gE));
    CUDA_CALL(cudaFreeHost(gI));
    CUDA_CALL(cudaFreeHost(eventRate));
    CUDA_CALL(cudaFreeHost(spikeTrain));
	CUDA_CALL(cudaFreeHost(not_matched));
    delete []preMat;
    printf("    Host memory freed\n");
    return EXIT_SUCCESS;
}
