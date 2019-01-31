#include "patch.h"

int main(int argc, char *argv[])
{
	#pragma float_control( except, on )
    #ifndef SKIP_IO
        std::ofstream p_file, v_file, spike_file, nSpike_file, gE_file, gI_file;
    #endif
    float time;
    curandStateMRG32k3a *stateE, *randState;
    curandStateMRG32k3a *stateI;
    unsigned long long seed;
    //seed = 183765712;
    seed = std::time(0);
    int device;
    int b1,b2;
    b1 = 160;
    b2 = 128;
    bool printStep = false;
    bool moreSharedMemThanBlocks = true;
    double flatRateE = 100.0f; // Hz
    double flatRateI0 = 4;
    double t = 1.0f;
    unsigned int nstep = 200;
    double EffsE = 1e-1;
    double IffsE0 = 0.0;
    double sEE0 = 0.0;
    double sIE0 = 0.0;
    double sEI0 = 0.0;
    double sII0 = 0.0;
    double EffsI0 = 0.7;
    double IffsI0 = 0.7;
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
		sscanf(argv[argc-1], "%lf", &flatRateE);
		sscanf(argv[argc-2], "%u", &seed);
		sscanf(argv[argc-3], "%d", &b2);
		sscanf(argv[argc-4], "%d", &b1);
		sscanf(argv[argc-5], "%d", &nstep);
	}
	if (argc == 15) {
		sscanf(argv[argc-1], "%100s", tmp);
		sscanf(argv[argc-2], "%lf", &t);
		sscanf(argv[argc-3], "%lf", &flatRateI0);
		sscanf(argv[argc-4], "%lf", &flatRateE);
		sscanf(argv[argc-5], "%u", &seed);
		sscanf(argv[argc-6], "%d", &b2);
		sscanf(argv[argc-7], "%d", &b1);
		sscanf(argv[argc-8], "%lf", &sII0);
		sscanf(argv[argc-9], "%lf", &sEI0);
		sscanf(argv[argc-10], "%lf", &sIE0);
		sscanf(argv[argc-11], "%lf", &sEE0);
		sscanf(argv[argc-12], "%lf", &IffsE0);
		sscanf(argv[argc-13], "%lf", &EffsE);
		sscanf(argv[argc-14], "%d", &nstep);
	}

	std::string theme = tmp;
	std::cout << "theme = " << theme << "\n";
    if (!theme.empty()) {
        theme = '-'+theme;
    }
    printf("%i x %i, %i steps, seed = %u\n", b1, b2, nstep, seed);
	unsigned int networkSize = b1*b2;
    if (networkSize/10.0 != float(networkSize/10)) {
        printf("To have higher computation occupancy make a factor of 10 in networkSize\n");
    }
	int init_b2 = warpSize;
	int init_b1 = networkSize / init_b2;
    double eiRatio = 3.0f/4.0f;
    unsigned int nE = networkSize*eiRatio;
    unsigned int nI = networkSize-nE;
    double sEE = sEE0*EffsE/nE;
    double sIE = sIE0*EffsE/nE;
    double sEI = sEI0*EffsE/nI;
    double sII = sII0*EffsE/nI;
    double IffsE = IffsE0*EffsE;
    printf("EffsE = %e, IffsE = %e\n", EffsE, IffsE);
    double EffsI = EffsI0*EffsE;
    double IffsI = IffsI0*EffsE;
    printf("EffsI = %e, IffsI = %e\n", EffsI, IffsI);
    printf("sEE = %e\n", sEE);
    printf("sIE = %e\n", sIE);
    printf("sEI = %e\n", sEI);
    printf("sII = %e\n", sII);
    double dt0 = t/static_cast<double>(nstep); // ms
    printf("dt0 = %.16e\n",dt0);
    double dt=1;
    if (dt0 > 1) {
        double next_dt = static_cast<int>(dt) << 1;
        while (next_dt < dt0) {
            next_dt = static_cast<int>(next_dt) << 1;
            dt = static_cast<int>(dt) << 1;
        }
        if (next_dt-dt0 < dt0-dt) {
            dt = next_dt;
        }
    } else {
        double next_dt = dt/2;
        while (next_dt > dt0) {
            next_dt /= 2;
            dt /= 2;
        }
        if (dt0-next_dt < dt-dt0) {
            dt = next_dt;
        }
    }
    t = dt * nstep;
    /* to be extended */
    bool presetInit = false;
    double riseTimeE[2] = {1.0f, 5.0f}; // ms
    double riseTimeI[1] = {1.0f};
    double decayTimeE[2] = {3.0f, 80.0f};
    double decayTimeI[1] = {5.0f};

    ConductanceShape condE(riseTimeE, decayTimeE, ngTypeE);
    ConductanceShape condI(riseTimeI, decayTimeI, ngTypeI);
    double flatRateI = flatRateI0 * flatRateE;
    printf("designated input rateE = %3.1fHz\n", flatRateE);
    printf("designated input rateI = %3.1fHz\n", flatRateI);
	printf("dt = %f ms\n", dt);
    printf("nE = %i, nI = %i\n", nE, networkSize-nE);
    printf("t = %.16e x %i = %.16e\n", dt, nstep, t);
	double dInputE = 1000.0f/flatRateE;
	double dInputI = 1000.0f/flatRateI;
    if (dt/dInputE > MAX_FFINPUT_PER_DT) {
        printf("increase MAX_FFINPUT_PER_DT, or decrease input rate E.\n");
        return EXIT_FAILURE;
    }
    if (dt/dInputI > MAX_FFINPUT_PER_DT) {
        printf("increase MAX_FFINPUT_PER_DT, or decrease input rate I.\n");
        return EXIT_FAILURE;
    }
	if (networkSize / float(warpSize) != float(networkSize / warpSize)) {
		printf("please make networkSize multiples of %i to run on GPU\n", warpSize);
		return EXIT_FAILURE;
	}
    struct cudaDeviceProp properties;  
    double *v, *gE, *gI, *preMat; 
    int *eventRateE, *d_eventRateE;
    int *eventRateI, *d_eventRateI;
    double *d_v, *d_gE, *d_gI, *d_hE, *d_hI, *d_fE, *d_fI, *d_preMat, *d_inputRateE, *d_inputRateI;
	#if SCHEME == 2
		double *d_dVs;
	#endif
    double *leftTimeRateE, *lastNegLogRandE;
    double *leftTimeRateI, *lastNegLogRandI;
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
    CUDA_CALL(cudaMalloc((void **)&leftTimeRateE,   networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&leftTimeRateI,   networkSize * sizeof(double)));
    /* Allocate space for rng on device */
    CUDA_CALL(cudaMalloc((void **)&stateE,          networkSize * sizeof(curandStateMRG32k3a)));
    CUDA_CALL(cudaMalloc((void **)&stateI,          networkSize * sizeof(curandStateMRG32k3a)));
    CUDA_CALL(cudaMalloc((void **)&lastNegLogRandE, networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&lastNegLogRandI, networkSize * sizeof(double)));

    CUDA_CALL(cudaMalloc((void **)&randState, networkSize * sizeof(curandStateMRG32k3a)));

    preMat = new double[networkSize * networkSize];
    CUDA_CALL(cudaMalloc((void **)&d_preMat, networkSize * networkSize * sizeof(double)));

    CUDA_CALL(cudaMallocHost((void**)&v, networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_v,           networkSize * sizeof(double)));

    logRand_init<<<init_b1,init_b2>>>(lastNegLogRandE, stateE, seed, leftTimeRateE, dInputE, 0);
    CUDA_CHECK();
    logRand_init<<<init_b1,init_b2>>>(lastNegLogRandI, stateI, seed-networkSize, leftTimeRateI, dInputI, networkSize);
    CUDA_CHECK();

    randInit<<<init_b1,init_b2>>>(d_preMat, d_v, randState, sEE, sIE, sEI, sII, networkSize, nE, seed);
    CUDA_CHECK();
    CUDA_CALL(cudaMemcpy(preMat, d_preMat, networkSize*networkSize*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(v, d_v, networkSize*sizeof(double),cudaMemcpyDeviceToHost));
    printf("storage size of preMat %.1fMb\n", float(networkSize*networkSize*sizeof(double))/1024.0/1024.0);

    unsigned int nbatch, batchEnd, batchStep;
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
    CUDA_CALL(cudaMallocHost((void**)&gE,          networkSize * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMallocHost((void**)&gI,          networkSize * ngTypeI *sizeof(double)));
    CUDA_CALL(cudaMallocHost((void**)&spikeTrain,  networkSize * sizeof(double)));
    CUDA_CALL(cudaMallocHost((void**)&nSpike,      networkSize * sizeof(unsigned int)));
    CUDA_CALL(cudaMallocHost((void**)&eventRateE,   networkSize * sizeof(int)));
    CUDA_CALL(cudaMallocHost((void**)&eventRateI,   networkSize * sizeof(int)));

    /* Allocate space for results on device */
	CUDA_CALL(cudaMalloc((void **)&d_dVs,		 networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_gE,         networkSize * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_gI,         networkSize * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_hE,         networkSize * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_hI,         networkSize * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_fE,         networkSize * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_fI,         networkSize * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_inputRateE, networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_inputRateI, networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_eventRateE, networkSize * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&d_eventRateI, networkSize * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&d_spikeTrain, networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_nSpike,     networkSize * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void **)&tBack,        networkSize * sizeof(double)));
    
    /* Create CUDA events */
    cudaEvent_t start, stop, gReadyE, gReadyI, vReady, vComputed, spikeRateReady, nSpikeReady, eventRateEReady, eventRateIReady;
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
	cudaEventCreate(&nSpikeReady);
#ifndef FULL_SPEED
    cudaEventCreate(&eventRateEReady);
	cudaEventCreate(&eventRateIReady);
#endif
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
        init<double><<<init_b1,init_b2,0,i1>>>(d_inputRateE, flatRateE/1000.0f);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i2>>>(d_inputRateI, flatRateI/1000.0f);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i3>>>(tBack, -1.0f); 
        CUDA_CHECK();
		init<double><<<init_b1, init_b2, 0, i1>>>(d_dVs, 0.0f);
		CUDA_CHECK();
        f_init<<<init_b1,init_b2,0,i2>>>(d_fE, networkSize, nE, ngTypeE, EffsE, IffsE);
        CUDA_CHECK();
        f_init<<<init_b1,init_b2,0,i3>>>(d_fI, networkSize, nE, ngTypeI, EffsI, IffsI);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeE,init_b2,0,i2>>>(d_gE, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i3>>>(d_gI, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeE,init_b2,0,i2>>>(d_hE, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i3>>>(d_hI, 0.0f);
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
    cudaStream_t s1, s2, s3, s4, s5, s6, s7;
    CUDA_CALL(cudaStreamCreate(&s1));
    CUDA_CALL(cudaStreamCreate(&s2));
    CUDA_CALL(cudaStreamCreate(&s3));
    CUDA_CALL(cudaStreamCreate(&s4));
    CUDA_CALL(cudaStreamCreate(&s5));
    CUDA_CALL(cudaStreamCreate(&s6));
    CUDA_CALL(cudaStreamCreate(&s7));
    #ifndef SKIP_IO
        #ifdef SPIKE_CORRECTION
            p_file.open("p_ushy" + theme + "_ssc.bin", std::ios::out|std::ios::binary);
            v_file.open("v_ictorious" + theme + "_ssc.bin", std::ios::out|std::ios::binary);
            spike_file.open("s_uspicious" + theme + "_ssc.bin", std::ios::out|std::ios::binary);
            nSpike_file.open("n_arcotic" + theme + "_ssc.bin", std::ios::out|std::ios::binary);
            gE_file.open("gE_nerous" + theme + "_ssc.bin", std::ios::out|std::ios::binary);
            gI_file.open("gI_berish" + theme + "_ssc.bin", std::ios::out|std::ios::binary);
        #else
            p_file.open("p_ushy" + theme + ".bin", std::ios::out|std::ios::binary);
            v_file.open("v_ictorious" + theme + ".bin", std::ios::out|std::ios::binary);
            spike_file.open("s_uspicious" + theme + ".bin", std::ios::out|std::ios::binary);
            nSpike_file.open("n_arcotic" + theme + ".bin", std::ios::out|std::ios::binary);
            gE_file.open("gE_nerous" + theme + ".bin", std::ios::out|std::ios::binary);
            gI_file.open("gI_berish" + theme + ".bin", std::ios::out|std::ios::binary);
        #endif
    
        p_file.write((char*)&nE, sizeof(unsigned int));
        p_file.write((char*)&nI, sizeof(unsigned int));
    	unsigned int u_ngTypeE = ngTypeE;
    	unsigned int u_ngTypeI = ngTypeI;
        p_file.write((char*)&u_ngTypeE, sizeof(unsigned int));
        p_file.write((char*)&u_ngTypeI, sizeof(unsigned int));
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
        p_file.write((char*)&flatRateE, sizeof(double));
    #endif

    CUDA_CALL(cudaEventRecord(start, 0));
    double eventsE = 0.0f;
    double eventsI = 0.0f;
    unsigned int spikesE = 0;
    unsigned int spikesI = 0;
    unsigned int ibatch = 0;
    unsigned int bStep = 0;
    unsigned int batchOffset = 0;
    unsigned int copySize = batchStep;
    unsigned int n = networkSize*copySize;
    CUDA_CALL(cudaEventRecord(gReadyE, 0));
    CUDA_CALL(cudaEventRecord(gReadyI, 0));
    CUDA_CALL(cudaEventRecord(vReady, 0));
    
    double timeV = 0.0f;
    double timeIO = 0.0f;
    double exc_input_ratio = 0.0f;
    double gEavgE = 0.0f;
    double gIavgE = 0.0f;
    double gEavgI = 0.0f;
    double gIavgI = 0.0f;
    for (int i=0; i<nstep; i++) {
        unsigned int offset;
        offset = 0;
        /* Write voltage of last step to disk */
        CUDA_CALL(cudaEventSynchronize(vReady));
        #ifndef SKIP_IO
            v_file.write((char*)v, networkSize * sizeof(double));
        #endif
        /* Write conductance of last step to disk */
        CUDA_CALL(cudaEventSynchronize(gReadyE));
        #ifndef SKIP_IO
            gE_file.write((char*)gE, networkSize*ngTypeE*sizeof(double));
        #endif
        CUDA_CALL(cudaEventSynchronize(gReadyI));
        #ifndef SKIP_IO
            gI_file.write((char*)gI, networkSize*ngTypeI*sizeof(double));
        #endif
        #ifndef FULL_SPEED
            for (unsigned int j=0; j<networkSize; j++) {
                if (j<nE) {
                    for (unsigned int ig=0; ig<ngTypeE; ig++) {
                        gEavgE += gE[ig*networkSize + j];
                    }
                    for (unsigned int ig=0; ig<ngTypeI; ig++) {
                        gIavgE += gI[ig*networkSize + j];
                    }
                } else {
                    for (unsigned int ig=0; ig<ngTypeE; ig++) {
                        gEavgI += gE[ig*networkSize + j];
                    }
                    for (unsigned int ig=0; ig<ngTypeI; ig++) {
                        gIavgI += gI[ig*networkSize + j];
                    }
                }
            }
        #endif
        /* Compute voltage */
        #ifdef KERNEL_PERFORMANCE
            CUDA_CALL(cudaEventRecord(kStart, 0));
        #endif
        dim3 grid3(1);
        dim3 block3(1024);
        #ifdef SPIKE_CORRECTION
            unsigned int shared_mem = 1024*sizeof(double)+2*1024*sizeof(unsigned int);
			#if SCHEME < 2
				compute_V<<<grid3, block3, shared_mem, s1>>>(d_v, d_gE, d_gI, d_hE, d_hI, d_preMat, d_inputRateE, d_inputRateI, d_eventRateE, d_eventRateI, d_spikeTrain, d_nSpike, tBack, d_fE, d_fI, leftTimeRateE, leftTimeRateI, lastNegLogRandE, lastNegLogRandI, stateE, stateI, condE, condI, dt, networkSize, nE, seed, dInputE, dInputI, i*dt);
			#else
				compute_V<<<grid3, block3, shared_mem, s1>>>(d_v, d_dVs, d_gE, d_gI, d_hE, d_hI, d_preMat, d_inputRateE, d_inputRateI, d_eventRateE, d_eventRateI, d_spikeTrain, d_nSpike, tBack, d_fE, d_fI, leftTimeRateE, leftTimeRateI, lastNegLogRandE, lastNegLogRandI, stateE, stateI, condE, condI, dt, networkSize, nE, seed, dInputE, dInputI, i*dt);
			#endif
        #else
            unsigned int shared_mem = 1024*sizeof(double)+1024*sizeof(double);
			#if SCHEME < 2
                compute_V_without_ssc<<<grid3, block3, shared_mem, s1>>>(d_v, d_gE, d_gI, d_hE, d_hI, d_preMat, d_inputRateE, d_inputRateI, d_eventRateE, d_eventRateI, d_spikeTrain, d_nSpike, tBack, d_fE, d_fI, leftTimeRateE, leftTimeRateI, lastNegLogRandE, lastNegLogRandI, stateE, stateI, condE, condI, dt, networkSize, nE, seed, dInputE, dInputI, i*dt);
            #else
                compute_V_without_ssc<<<grid3, block3, shared_mem, s1>>>(d_v, d_dVs, d_gE, d_gI, d_hE, d_hI, d_preMat, d_inputRateE, d_inputRateI, d_eventRateE, d_eventRateI, d_spikeTrain, d_nSpike, tBack, d_fE, d_fI, leftTimeRateE, leftTimeRateI, lastNegLogRandE, lastNegLogRandI, stateE, stateI, condE, condI, dt, networkSize, nE, seed, dInputE, dInputI, i*dt);
            #endif
        #endif
        CUDA_CHECK();
        CUDA_CALL(cudaEventRecord(vComputed, s1));
        #ifdef KERNEL_PERFORMANCE
            CUDA_CALL(cudaEventRecord(kStop, 0));
            CUDA_CALL(cudaEventSynchronize(kStop));
            CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
            timeV += time;
        #endif
        #ifndef FULL_SPEED
		    if (printStep) {
		    	printf("A single step of compute_V cost %fms\n", time);
		    }
        #endif
		#ifdef KERNEL_PERFORMANCE
			CUDA_CALL(cudaEventRecord(kStart, 0));
		#endif
		CUDA_CALL(cudaEventSynchronize(vComputed));
        /* Copy spikeTrain to host */
        CUDA_CALL(cudaMemcpyAsync(spikeTrain, d_spikeTrain, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s1));
		CUDA_CALL(cudaEventRecord(spikeRateReady, s1));
		CUDA_CALL(cudaMemcpyAsync(nSpike, d_nSpike, networkSize * sizeof(unsigned int), cudaMemcpyDeviceToHost, s2));
		CUDA_CALL(cudaEventRecord(nSpikeReady, s2));
        /* Copy input events to host */
        #ifndef FULL_SPEED
            CUDA_CALL(cudaMemcpyAsync(eventRateE, d_eventRateE, networkSize * sizeof(int), cudaMemcpyDeviceToHost, s3));
		    CUDA_CALL(cudaEventRecord(eventRateEReady, s3));
            CUDA_CALL(cudaMemcpyAsync(eventRateI, d_eventRateI, networkSize * sizeof(int), cudaMemcpyDeviceToHost, s4));
		    CUDA_CALL(cudaEventRecord(eventRateIReady, s4));
        #endif
		/* Copy voltage to host */
		CUDA_CALL(cudaMemcpyAsync(v, d_v, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s5));
		CUDA_CALL(cudaEventRecord(vReady, s5));
        // copy exc conductance to host
        CUDA_CALL(cudaMemcpyAsync(gE, d_gE, networkSize * ngTypeE * sizeof(double), cudaMemcpyDeviceToHost, s6));
		CUDA_CALL(cudaEventRecord(gReadyE, s6));
        // copy inh conductance to host
        CUDA_CALL(cudaMemcpyAsync(gI, d_gI, networkSize * ngTypeI * sizeof(double), cudaMemcpyDeviceToHost, s7));
		CUDA_CALL(cudaEventRecord(gReadyI, s7));

        CUDA_CALL(cudaEventSynchronize(spikeRateReady));
        /* Write spikeTrain of current step to disk */
        #ifndef SKIP_IO
            spike_file.write((char*)spikeTrain,  networkSize*sizeof(double));
        #endif
		CUDA_CALL(cudaEventSynchronize(nSpikeReady));
        #ifndef SKIP_IO
            nSpike_file.write((char*)nSpike,     networkSize*sizeof(unsigned int));
        #endif

        #ifndef FULL_SPEED
            CUDA_CALL(cudaEventSynchronize(eventRateEReady));
		    CUDA_CALL(cudaEventSynchronize(eventRateIReady));
            #ifndef DEBUG
                printf("\r stepping: %3.1f%%", 100.0f*float(i+1)/nstep);
            #else
                printf("stepping: %3.1f%%, t = %f \n", 100.0f*float(i+1)/nstep, (i+1)*dt);
            #endif
            double _eventsE = 0.0f;
            double _eventsI = 0.0f;
            unsigned int _spikes = 0;
            double sEi = 0.0f;
            for (int j=0; j<networkSize; j++) {
                _eventsE += eventRateE[j];
                _eventsI += eventRateI[j];
                _spikes += nSpike[j];
                if (j<nE) {
                    spikesE += nSpike[j];
                } else {
                    spikesI += nSpike[j];
                }
                for (unsigned int k=0; k<nE; k++) {
                     sEi += preMat[k*networkSize + j] * nSpike[k];
                }
            }
            eventsE += _eventsE;
            eventsI += _eventsI;
            exc_input_ratio += sEi/networkSize;
            if (printStep) {
                printf("instant exc input rate = %fkHz, dt = %f, networkSize = %i\n", _eventsE/(dt*networkSize), dt, networkSize);
                printf("instant inh input rate = %fkHz, dt = %f, networkSize = %i\n", _eventsI/(dt*networkSize), dt, networkSize);
                printf("instant firing rate = %fHz\n", _spikes/(dt*networkSize)*1000.0);
            }
        #endif

		#ifdef KERNEL_PERFORMANCE
			CUDA_CALL(cudaEventRecord(kStop, 0));
			CUDA_CALL(cudaEventSynchronize(kStop));
			CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
			if (printStep) {
				printf("Memcpy and Disk IO cost %fms\n", time);
			}
			timeIO += time;
		#endif
    }
    /* WHen hit HALF_MEMORY_OCCUPANCY, write half of the array to disk, the other half left to receive from device */

    #ifndef SKIP_IO
        v_file.write((char*)v, networkSize * sizeof(double));
        gE_file.write((char*)gE, networkSize * ngTypeE * sizeof(double));
        gI_file.write((char*)gI, networkSize * ngTypeI * sizeof(double));
    #endif
    printf("\n");
    #ifndef FULL_SPEED
        printf("flatRateE = %fHz, realized mean input rate = %fHz\n", flatRateE, 1000.0*float(eventsE)/(dt*nstep*networkSize));
        printf("flatRateI = %fHz, realized mean input rate = %fHz\n", flatRateI, 1000.0*float(eventsI)/(dt*nstep*networkSize));
        printf("exc firing rate = %eHz\n", float(spikesE)/(dt*nstep*nE)*1000.0);
        printf("inh firing rate = %eHz\n", float(spikesI)/(dt*nstep*nI)*1000.0);
    #endif

    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&time, start, stop));
    printf("CUDA takes %fms, runtime/realtime ratio ms %fms\n", time, time/(dt*nstep));
    printf("compute_V takes %fms, ratio ms %fms\n", timeV, timeV/(dt*nstep));
    printf("IO takes %fms, ratio ms %fms\n", timeIO, timeIO/(dt*nstep));
    #ifndef FULL_SPEED
        printf("input ratio recurrent:feedforward = %f\n", exc_input_ratio/((EffsE*nE+IffsE*nI)/networkSize*dt*nstep/dInputE));
        printf("           exc,        inh\n");
        printf("avg gE = %e, %e\n", gEavgE/nstep/nE, gEavgI/nstep/nI);
        printf("avg gI = %e, %e\n", gIavgE/nstep/nE, gIavgI/nstep/nI);
    #endif
    int nTimer = 1;
    #ifndef SKIP_IO
        p_file.write((char*)&nTimer, sizeof(int));
        p_file.write((char*)&timeV, sizeof(double));
        //p_file.write((char*)&timeIO, sizeof(double));
    #endif

    /* Cleanup */
    printf("Cleaning up:\n");
    CUDA_CALL(cudaStreamDestroy(s1));
    CUDA_CALL(cudaStreamDestroy(s2));
    CUDA_CALL(cudaStreamDestroy(s3));
    CUDA_CALL(cudaStreamDestroy(s4));
    CUDA_CALL(cudaStreamDestroy(s5));
    CUDA_CALL(cudaStreamDestroy(s6));
    CUDA_CALL(cudaStreamDestroy(s7));
    printf("    CUDA streams destroyed\n");
    
    #ifndef SKIP_IO
        if (p_file.is_open()) p_file.close();
        if (v_file.is_open()) v_file.close();
        if (spike_file.is_open()) spike_file.close();
        if (nSpike_file.is_open()) nSpike_file.close();
        if (gE_file.is_open()) gE_file.close();
        if (gI_file.is_open()) gI_file.close();
    #endif
    printf("    Output files closed\n");
    
    CUDA_CALL(cudaFree(stateE));
    CUDA_CALL(cudaFree(stateI));
    CUDA_CALL(cudaFree(d_v));
	CUDA_CALL(cudaFree(d_dVs));
    CUDA_CALL(cudaFree(d_gE));
    CUDA_CALL(cudaFree(d_gI));
    CUDA_CALL(cudaFree(d_hE));
    CUDA_CALL(cudaFree(d_hI));
    CUDA_CALL(cudaFree(d_fE));
    CUDA_CALL(cudaFree(d_fI));
    CUDA_CALL(cudaFree(d_preMat));
    CUDA_CALL(cudaFree(leftTimeRateE));
    CUDA_CALL(cudaFree(leftTimeRateI));
    CUDA_CALL(cudaFree(lastNegLogRandE));
    CUDA_CALL(cudaFree(lastNegLogRandI));
    CUDA_CALL(cudaFree(d_inputRateE));
    CUDA_CALL(cudaFree(d_inputRateI));
    CUDA_CALL(cudaFree(d_eventRateE));
    CUDA_CALL(cudaFree(d_eventRateI));
    CUDA_CALL(cudaFree(d_spikeTrain));
    CUDA_CALL(cudaFree(d_nSpike));
    CUDA_CALL(cudaFree(tBack));
    printf("    Device memory freed\n");
    CUDA_CALL(cudaFreeHost(v));
    CUDA_CALL(cudaFreeHost(gE));
    CUDA_CALL(cudaFreeHost(gI));
    CUDA_CALL(cudaFreeHost(eventRateE));
    CUDA_CALL(cudaFreeHost(eventRateI));
    CUDA_CALL(cudaFreeHost(spikeTrain));
    CUDA_CALL(cudaFreeHost(nSpike));
    delete []preMat;
    printf("    Host memory freed\n");
    return EXIT_SUCCESS;
}
