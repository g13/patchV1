#include "test.h"

int main(int argc, char *argv[])
{
    std::ofstream p_file, v_file, spike_file, nSpike_file, gE_file, gI_file;
    float time;
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
    double flatRate = 100.0f; // Hz
    double t = 1.0f;
	unsigned int mt = 1;
    unsigned int nstep = 200;
    double ffsE = 1e-3;
    double s0 = 1e-2*ffsE;
    double ffsI = 5e-2;
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
    if (iFlatRate > 0) {
        flatRate = iFlatRate*1.0f;
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
	double dInput = 1000.0f/flatRate;
    if (dt/dInput > MAX_FFINPUT_PER_DT) {
        printf("increase MAX_FFINPUT_PER_DT, or decrease input rate.\n");
        return EXIT_FAILURE;
    }
	if (networkSize / float(warpSize) != float(networkSize / warpSize)) {
		printf("please make networkSize multiples of %i to run on GPU\n", warpSize);
		return EXIT_FAILURE;
	}
    struct cudaDeviceProp properties;  
    double *v, *gE, *gI, *preMat, *firstInput; 
    int *eventRate, *d_eventRate;
    double *d_v, *d_gE, *d_gI, *d_hE, *d_hI, *d_fE, *d_fI, *d_preMat, *d_inputRate;
    double *d_a, *d_b;
    double *gactVec, *hactVec;
    double *leftTimeRate, *lastNegLogRand;
    double *spikeTrain, *d_spikeTrain, *tBack;
    unsigned int *nSpike, *d_nSpike;

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
    firstInput = new double[networkSize];
    CUDA_CALL(cudaMalloc((void **)&leftTimeRate,   networkSize * sizeof(double)));

    CUDA_CALL(cudaMalloc((void **)&randState, networkSize * sizeof(curandStateMRG32k3a)));

    preMat = new double[networkSize * networkSize];
    CUDA_CALL(cudaMalloc((void **)&d_preMat, networkSize * networkSize * sizeof(double)));

    CUDA_CALL(cudaMallocHost((void**)&v, networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_v,           networkSize * sizeof(double)));

    randInit<<<init_b1,init_b2>>>(d_preMat, d_v, leftTimeRate, randState, s, networkSize, seed, dInput);
    CUDA_CHECK();
    CUDA_CALL(cudaMemcpy(preMat, d_preMat, networkSize*networkSize*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(firstInput, leftTimeRate, networkSize*sizeof(double),cudaMemcpyDeviceToHost));
	double sum = 0;
	for (unsigned int i = 0; i < networkSize*networkSize; i++) {
		sum += preMat[i];
	}
	printf("sum of preMat = %f\n", sum);
    CUDA_CALL(cudaMemcpy(v, d_v, networkSize*sizeof(double),cudaMemcpyDeviceToHost));
    printf("storage size of preMat %.1fMb\n", float(networkSize*networkSize*sizeof(double))/1024.0/1024.0);

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
    CUDA_CALL(cudaMallocHost((void**)&nSpike,      networkSize * sizeof(unsigned int) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&eventRate,   networkSize * sizeof(int) * batchStep * alt));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&d_gE,           networkSize * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_gI,           networkSize * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_hE,           networkSize * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_hI,           networkSize * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_fE,           networkSize * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_fI,           networkSize * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_a,            networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_b,            networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_inputRate,    networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_eventRate,    networkSize * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&d_spikeTrain,   networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_nSpike,       networkSize * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void **)&tBack,          networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gactVec,        networkSize * ngType * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hactVec,        networkSize * ngType * sizeof(double)));
    /* Allocate space for rng on device */
    CUDA_CALL(cudaMalloc((void **)&state,          networkSize * sizeof(curandStateMRG32k3a)));
    CUDA_CALL(cudaMalloc((void **)&lastNegLogRand, networkSize * sizeof(double)));
    /* Allocate space for partial reduce results on device */
    
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
        printf("E: The size of the requested shared memory %fKb by recal_G is not available\n", float(rgE_shared)/1024);
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
        printf("I: The size of the requested shared memory %fKb by recal_G is not available\n", float(rgI_shared)/1024);
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
        printf("blockdims for reduction of %i per thread : %i x %i \n", rgI_b1.x, networkSize, rI_b2);
        CUDA_CALL(cudaMalloc((void **)&gI_b1x,  networkSize * rI_b2 * ngTypeI * sizeof(double)));
        CUDA_CALL(cudaMalloc((void **)&hI_b1x,  networkSize * rI_b2 * ngTypeI * sizeof(double)));
    }




    /* Create CUDA events */
    cudaEvent_t start, stop, gReadyE, gReadyI, vReady, vComputed, spikeRateReady, eventRateReady;
    cudaEvent_t kStart, kStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&gReadyE);
    cudaEventCreate(&gReadyI);
    cudaEventCreate(&vReady);
    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);
    cudaEventCreate(&vComputed);
    cudaEventCreate(&spikeRateReady);
    cudaEventCreate(&eventRateReady);
    /* Initialize device arrays */
    // CUDA streams for init
    cudaStream_t i1, i2, i3;
    CUDA_CALL(cudaStreamCreate(&i1));
    CUDA_CALL(cudaStreamCreate(&i2));
    CUDA_CALL(cudaStreamCreate(&i3));
    if (!presetInit) {
        for (unsigned int i=0; i<networkSize; i++) {
            for (unsigned int ig=0; ig<ngTypeE; ig++) {
                gE[ig*networkSize + i] = 0.0f;
            }
            for (unsigned int ig=0; ig<ngTypeI; ig++) {
                gI[ig*networkSize + i] = 0.0f;
            }
        }
        // init rand generation for poisson
        logRand_init<<<init_b1,init_b2,0,i1>>>(lastNegLogRand, state, seed);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i2>>>(d_inputRate, flatRate/1000.0f);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i3>>>(tBack, -1.0f); 
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeE,init_b2,0,i2>>>(d_fE, ffsE);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i3>>>(d_fI, ffsI);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeE,init_b2,0,i2>>>(d_gE, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i3>>>(d_gI, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeE,init_b2,0,i2>>>(d_hE, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i3>>>(d_hI, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngType,init_b2,0,i2>>>(gactVec, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngType,init_b2,0,i3>>>(hactVec, 0.0f);
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
    CUDA_CALL(cudaEventRecord(iStop, 0));
    CUDA_CALL(cudaEventSynchronize(iStop));
    CUDA_CALL(cudaEventElapsedTime(&time, iStart, iStop));
    printf("initialization cost %fms\n", time);

    /* Create CUDA streams */
    cudaStream_t s1, s2, s3;
    CUDA_CALL(cudaStreamCreate(&s1));
    CUDA_CALL(cudaStreamCreate(&s2));
    CUDA_CALL(cudaStreamCreate(&s3));
    unsigned int shared_mem = 0;
    p_file.open("p_ushy" + theme + ".bin", std::ios::out|std::ios::binary);
    v_file.open("v_ictorious" + theme + ".bin", std::ios::out|std::ios::binary);
    spike_file.open("s_uspicious" + theme + ".bin", std::ios::out|std::ios::binary);
    nSpike_file.open("n_arcotic" + theme + ".bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_nerous" + theme + ".bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_berish" + theme + ".bin", std::ios::out|std::ios::binary);

    p_file.write((char*)&nE, sizeof(unsigned int));
    p_file.write((char*)&nI, sizeof(unsigned int));
    p_file.write((char*)&ngTypeE, sizeof(unsigned int));
    p_file.write((char*)&ngTypeI, sizeof(unsigned int));
    double dtmp = vL;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = vT;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = vE;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = vI;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = gL_E;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = gL_I;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = tRef_E;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = tRef_I;
    p_file.write((char*)&tmp, sizeof(double));
    p_file.write((char*)&nstep, sizeof(unsigned int));
    p_file.write((char*)&dt, sizeof(double));
    p_file.write((char*)&flatRate, sizeof(double));


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
        double timeV = 0.0f;
        double timeG = 0.0f;
        for (int i=0; i<nstep; i++) {
            unsigned int offset;
            //offset = networkSize*(batchOffset + i);
            offset = 0;
            /* Write voltage of last step to disk */
            CUDA_CALL(cudaEventSynchronize(vReady));
            v_file.write((char*)v, networkSize * sizeof(double));
            /* Write conductance of last step to disk */
            CUDA_CALL(cudaEventSynchronize(gReadyE));
            gE_file.write((char*)gE, networkSize*ngTypeE*sizeof(double));
            CUDA_CALL(cudaEventSynchronize(gReadyI));
            gI_file.write((char*)gI, networkSize*ngTypeI*sizeof(double));
            /* Compute voltage */
            #ifdef KERNEL_PERFORMANCE
                CUDA_CALL(cudaEventRecord(kStart, 0));
            #endif
            compute_V<<<b1, b2, shared_mem, s1>>>(d_v, d_gE, d_gI, d_hE, d_hI, d_a, d_b, d_preMat, d_inputRate, d_eventRate, d_spikeTrain, d_nSpike, tBack, gactVec, hactVec, d_fE, d_fI, leftTimeRate, lastNegLogRand, state, ngTypeE, ngTypeI, ngType, condE, condI, dt, networkSize, nE, seed, dInput);
            CUDA_CHECK();
            CUDA_CALL(cudaEventRecord(vComputed, s1));
            #ifdef KERNEL_PERFORMANCE
                CUDA_CALL(cudaEventRecord(kStop, 0));
                CUDA_CALL(cudaEventSynchronize(kStop));
                CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
                timeV += time;
            #endif
            if (printStep) {
                printf("A single step of compute_V cost %fms\n", time);
            }
            /* Copy voltage to host */
            CUDA_CALL(cudaMemcpyAsync(v, d_v, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaEventRecord(vReady, s1));
            /* Copy spikeTrain to host */
            CUDA_CALL(cudaMemcpyAsync(spikeTrain, d_spikeTrain, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaMemcpyAsync(nSpike, d_nSpike, networkSize * sizeof(unsigned int), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaEventRecord(spikeRateReady, s1));
            /* Copy input events to host */
            CUDA_CALL(cudaMemcpyAsync(eventRate, d_eventRate, networkSize * sizeof(int), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaEventRecord(eventRateReady, s1));

            /* Recalibrate conductance to postsynaptic neurons, for the next step*/
            CUDA_CALL(cudaEventSynchronize(vComputed));
            #ifdef KERNEL_PERFORMANCE
                CUDA_CALL(cudaEventRecord(kStart, 0));
            #endif
            // recal E
            recal_G<<<rgE_b1,rgE_b2,rgE_shared,s2>>>(d_gE, d_hE, d_preMat,
                                                     gactVec, hactVec,
                                                     gE_b1x, hE_b1x,
                                                     networkSize, 0, ngTypeE, s_actVec_lE, msE);
            CUDA_CHECK();
            // recal I
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
            CUDA_CALL(cudaEventSynchronize(eventRateReady));
            CUDA_CALL(cudaEventSynchronize(spikeRateReady));
            /* Write spikeTrain of current step to disk */
            spike_file.write((char*)spikeTrain,  n*sizeof(double));
            nSpike_file.write((char*)nSpike,      n*sizeof(unsigned int));
            printf("\r stepping: %3.1f%%", 100.0f*float(i+1)/nstep);
            //printf("stepping: %3.1f%%, t = %f \n", 100.0f*float(i+1)/nstep, (i+1)*dt);
            double _events = 0.0f;
            unsigned int _spikes = 0;
            for (int j=0; j<networkSize; j++) {
                _events += eventRate[j];
                _spikes += nSpike[j];
            }
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
        }
        /* WHen hit HALF_MEMORY_OCCUPANCY, write half of the array to disk, the other half left to receive from device */
        // Alternating
        // switch batchOffset
    //}

    v_file.write((char*)v, networkSize * sizeof(double));
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
    int nTimer = 2;
    p_file.write((char*)&nTimer, sizeof(int));
    p_file.write((char*)&timeV, sizeof(double));
    p_file.write((char*)&timeG, sizeof(double));

    /* Cleanup */
    printf("Cleaning up:\n");
    CUDA_CALL(cudaStreamDestroy(s1));
    CUDA_CALL(cudaStreamDestroy(s2));
    CUDA_CALL(cudaStreamDestroy(s3));
    printf("    CUDA streams destroyed\n");
    if (p_file.is_open()) p_file.close();
    if (v_file.is_open()) v_file.close();
    if (spike_file.is_open()) spike_file.close();
    if (nSpike_file.is_open()) nSpike_file.close();
    if (gE_file.is_open()) gE_file.close();
    if (gI_file.is_open()) gI_file.close();
    printf("    Output files closed\n");
    
    CUDA_CALL(cudaFree(state));
    CUDA_CALL(cudaFree(d_v));
    CUDA_CALL(cudaFree(d_gE));
    CUDA_CALL(cudaFree(d_gI));
    CUDA_CALL(cudaFree(d_hE));
    CUDA_CALL(cudaFree(d_hI));
    CUDA_CALL(cudaFree(d_fE));
    CUDA_CALL(cudaFree(d_fI));
    CUDA_CALL(cudaFree(gactVec));
    CUDA_CALL(cudaFree(hactVec));
    CUDA_CALL(cudaFree(d_preMat));
    CUDA_CALL(cudaFree(d_a));
    CUDA_CALL(cudaFree(d_b));
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
    CUDA_CALL(cudaFree(d_nSpike));
    CUDA_CALL(cudaFree(tBack));
    printf("    Device memory freed\n");
    CUDA_CALL(cudaFreeHost(v));
    CUDA_CALL(cudaFreeHost(gE));
    CUDA_CALL(cudaFreeHost(gI));
    CUDA_CALL(cudaFreeHost(eventRate));
    CUDA_CALL(cudaFreeHost(spikeTrain));
    CUDA_CALL(cudaFreeHost(nSpike));
    delete []preMat;
    delete []firstInput;
    printf("    Host memory freed\n");
    return EXIT_SUCCESS;
}
