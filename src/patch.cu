#include "patch.h"

int main(int argc, char *argv[])
{
    std::ofstream v_file, spike_file, gE_file, gI_file;
    float time;
    //cudaEventCreateWithFlags(&gReady, cudaEventDisableTiming);
    curandStateMRG32k3a *state;
    unsigned long long seed;
    //seed = 183765712;
    seed = std::time(0);
    int device;
    int b1,b2;
    b1 = 160;
    b2 = 128;
    bool printStep = false;
    unsigned int nstep = 200;
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
    printf("%i x %i, %i steps, seed = %u\n", b1, b2, nstep, seed);
	unsigned int networkSize = b1*b2;
    int warpSize = 32;
    if (networkSize/float(warpSize) != float(networkSize/warpSize)) {
        printf("please make networkSize multiples of %i\n", warpSize);
        return EXIT_FAILURE;
    }
    if (networkSize/10.0 != float(networkSize/10)) {
        printf("To have higher computation occupancy make a factor of 10 in networkSize\n");
    }
	int init_b2 = warpSize;
	int init_b1 = networkSize / init_b2;
    double eiRatio = 3.0f/4.0f;
    int b1E = b1*eiRatio;
    int b2E = b2*eiRatio;
    int b1I = b1*(1-eiRatio);
    int b2I = b2*(1-eiRatio);
    unsigned int nE = networkSize*eiRatio;
    unsigned int nI = networkSize-nE;
    double t = 25.0f;
    double dt = t/float(nstep); // ms
    double flatRate = 10000.0f; // Hz
    //double flatRate = 0.0f; // Hz
    double ffsE = 1e-1;
    double s = 1.0*ffsE/(networkSize);
    double ffsI = 1e-1;
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
    printf("nE = %i, nI = %i\n", nE, networkSize-nE);
    printf("t = %f x %i = %f\n", dt, nstep, t);
    cpu_version(networkSize, flatRate/1000.0, nstep, dt, nE, s, seed, ffsE, ffsI);
    struct cudaDeviceProp properties;  
    double *v, *gE, *gI, *preMat; 
    int *eventRate, *d_eventRate;
    double *d_v1, *d_v2, *d_gE, *d_gI, *d_hE, *d_hI, *d_fE, *d_fI, *d_preMat, *d_inputRate;
    double *d_a, *d_b, *v_current, *v_old, *d_v_hlf;
    double *gactVec, *hactVec;
    double *leftTimeRate, *lastNegLogRand;
    double *spikeTrain, *d_spikeTrain, *tBack;
    bool *matched;
    bool *d_matched;

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
    /* Allocate space for results on host */
    //pinned memory
    CUDA_CALL(cudaMallocHost((void**)&v,          networkSize * sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&gE,         networkSize * ngTypeE * sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&gI,         networkSize * ngTypeI *sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&spikeTrain, networkSize * sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&eventRate,  networkSize * sizeof(int) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void **)&matched, networkSize * sizeof(bool)));
    preMat = (double *)calloc(networkSize, sizeof(double));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&d_v1,           networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_v2,           networkSize * sizeof(double)));
	CUDA_CALL(cudaMalloc((void **)&d_v_hlf,        networkSize * sizeof(double)));
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
    CUDA_CALL(cudaMalloc((void **)&tBack,          networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gactVec,        networkSize * ngType * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hactVec,        networkSize * ngType * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_preMat,       networkSize * networkSize * sizeof(double)));
    /* Allocate space for rng on device */
    CUDA_CALL(cudaMalloc((void **)&state,          networkSize * sizeof(curandStateMRG32k3a)));
    CUDA_CALL(cudaMalloc((void **)&leftTimeRate,   networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&lastNegLogRand, networkSize * sizeof(double)));
    /* Allocate space for partial reduce results on device */
    /* Allocate variables that allow write-write conflict for global-OR operation on device*/
    CUDA_CALL(cudaMalloc((void **)&d_matched, networkSize * sizeof(bool)));
    
    int EmaxTPB, ImaxTPB;
    int mE, mI; 
    if (properties.maxThreadsPerBlock < nE) {
        EmaxTPB = properties.maxThreadsPerBlock;
        mE = (nE+EmaxTPB-1)/EmaxTPB;
        EmaxTPB = nE/mE;
    } else {
        mE = 1;
        EmaxTPB = nE;
    }
    while (EmaxTPB*mE != nE && EmaxTPB > EmaxTPB/2) {
        mE = mE + 1;
        EmaxTPB = nI/mE;
    }

    if (properties.maxThreadsPerBlock < nI) {
        ImaxTPB = properties.maxThreadsPerBlock;
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
    dim3 rgE_b2(EmaxTPB,1);
    dim3 rgI_b2(ImaxTPB,1);
    int msE = 1; // multiple shared actVec load per thread
    int msI = 1;
    int s_actVec_lE;
    int s_actVec_lI;
    unsigned int rgE_shared;
    unsigned int rgI_shared;

    s_actVec_lE = EmaxTPB;
    rgE_shared = 2*ngTypeE*s_actVec_lE*sizeof(double);
    if (rgE_shared > properties.sharedMemPerBlock) {
        printf("E: The size of the requested shared memory %iKb by recal_G is not available\n", rgE_shared/1024);
        return EXIT_FAILURE;
    } else {
        while (rgE_shared*2  < properties.sharedMemPerBlock && mE/float(msE*2) == float(mE/(msE*2))) {
            msE = msE * 2;
            rgE_shared = rgE_shared * 2;
        }
    }
    s_actVec_lE = msE*s_actVec_lE;
    rgE_b1.x = mE/msE;
    rgE_b1.y = mE;
    printf("E: recal_G<<<(%i,%i,%i)x(%i,%i,%i), %iKb>>>\n", rgE_b1.x, rgE_b1.y, rgE_b1.z, rgE_b2.x, rgE_b2.y, rgE_b2.z, rgE_shared/1024);

    s_actVec_lI = ImaxTPB;
    rgI_shared = 2*ngTypeI*s_actVec_lI*sizeof(double);
    if (rgI_shared > properties.sharedMemPerBlock) {
        printf("I: The size of the requested shared memory %iKb by recal_G is not available\n", rgI_shared/1024);
        return EXIT_FAILURE;
    } else {
        while (rgI_shared*2  < properties.sharedMemPerBlock && mI/float(msI*2) == float(mI/(msI*2))) {
            msI = msI * 2;
            rgI_shared = rgI_shared * 2;
        }
    }
    s_actVec_lI = msI*s_actVec_lI;
    rgI_b1.x = mI/msI;
    rgI_b1.y = mI;
    printf("I: recal_G<<<(%i,%i,%i)x(%i,%i,%i), %iKb>>>\n", rgI_b1.x, rgI_b1.y, rgI_b1.z, rgI_b2.x, rgI_b2.y, rgI_b2.z, rgI_shared/1024);

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    int rE_b2, rI_b2;
    double *gE_b1y, *gI_b1y, *hE_b1y, *hI_b1y;
    if (rgE_b1.x >= 32) {
        int e = 5;
        while (rgE_b1.x > 1<<e) e++;
        rE_b2 = 1<<e;
        printf("blockdims for reduction of %i per thread : %i x %i \n", rgE_b1.x, networkSize, rE_b2);
        CUDA_CALL(cudaMalloc((void **)&gE_b1y,  networkSize * rE_b2 * ngTypeE * sizeof(double)));
        CUDA_CALL(cudaMalloc((void **)&hE_b1y,  networkSize * rE_b2 * ngTypeE * sizeof(double)));
    }
    if (rgI_b1.x >= 32) {
        int e = 5;
        while (rgI_b1.x > 1<<e) e++;
        rI_b2 = 1<<e;
        printf("blockdims for reduction of %i per thread : %i x %i \n", rgI_b1.x, networkSize, rE_b2);
        CUDA_CALL(cudaMalloc((void **)&gI_b1y,  networkSize * rI_b2 * ngTypeI * sizeof(double)));
        CUDA_CALL(cudaMalloc((void **)&hI_b1y,  networkSize * rI_b2 * ngTypeI * sizeof(double)));
    }




    /* Create CUDA events */
    cudaEvent_t start, stop, gReady, spikeCorrected, conductanceReady, initialSpikesObtained;
    cudaEvent_t kStart, kStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&gReady);
    cudaEventCreate(&spikeCorrected);
    cudaEventCreate(&conductanceReady);
    cudaEventCreate(&initialSpikesObtained);
    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);
    /* Initialize device arrays */
    // CUDA streams for init
    cudaStream_t i1, i2, i3;
    CUDA_CALL(cudaStreamCreate(&i1));
    CUDA_CALL(cudaStreamCreate(&i2));
    CUDA_CALL(cudaStreamCreate(&i3));
    if (presetInit) {
    } else {
        for (unsigned int i=0; i<networkSize; i++) {
            v[i] = 0.0f;
            gE[i] = 0.0f;
            gI[i] = 0.0f;
            spikeTrain[i] = dt;
        }
        // init rand generation for poisson
        logRand_init<<<init_b1,init_b2,0,i1>>>(lastNegLogRand, state, seed);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i2>>>(d_inputRate, flatRate/1000.0f);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i3>>>(d_v1, 0.0f);
        CUDA_CHECK();
        //init<double><<<init_b1,init_b2,0,i3>>>(d_v2, 0.0f);
        //CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i2>>>(leftTimeRate, 0.0f);
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
        printf("storage size of preMat %.1fMb\n", float(networkSize*networkSize*sizeof(double))/1024.0/1024.0);
        init<<<init_b1*init_b1*init_b2,init_b2,0,i2>>>(d_preMat, s);
        CUDA_CHECK();
        //CUDA_CALL(cudaEventRecord(kStop, 0));
        //CUDA_CALL(cudaEventSynchronize(kStop));
        //CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
        //printf("logRand_init<<<%ix%i>>> cost %.1fms\n", init_b1*init_b1*init_b2, init_b2, time);
    }
    CUDA_CALL(cudaStreamDestroy(i1));
    CUDA_CALL(cudaStreamDestroy(i2));
    CUDA_CALL(cudaStreamDestroy(i3));

    /* Create CUDA streams */
    cudaStream_t s1, s2, s3;
    CUDA_CALL(cudaStreamCreate(&s1));
    CUDA_CALL(cudaStreamCreate(&s2));
    CUDA_CALL(cudaStreamCreate(&s3));
    unsigned int shared_mem = 0;
    v_file.open("v_ictorious.bin", std::ios::out|std::ios::binary);
    spike_file.open("s_uspicious.bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_nerous.bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_berish.bin", std::ios::out|std::ios::binary);
    CUDA_CALL(cudaEventRecord(start, 0));
    double events = 0.0f;
    int spikes = 0;
    unsigned int ibatch = 0;
    unsigned int bStep = 0;
    unsigned int batchOffset = 0;
    unsigned int copySize = batchStep;
    unsigned int n = networkSize*copySize;
    
    //for (int ibatch=0; i<nbatch; ibatch++) {
    //    if(ibatch == nbatch-1) {
    //        copySize = batchEnd;
    //    }
        bool it = true;
        double time1 = 0.0f;
        double time2 = 0.0f;
        for (int i=0; i<nstep; i++) {
            unsigned int offset; 
            //offset = networkSize*(batchOffset + i);
            offset = 0;
            CUDA_CALL(cudaStreamWaitEvent(s1, gReady, 0));
            /* Compute voltage (acquire initial spikes) */
            CUDA_CALL(cudaEventRecord(kStart, 0));
            if (i%2 == 0) {
                v_current = d_v2;
                v_old = d_v1;
            } else {
                v_current = d_v1;
                v_old = d_v2;
            }
            compute_V <<<b1, b2, shared_mem, s1>>> (v_current, d_gE, d_gI, d_hE, d_hI, d_a, d_b, d_preMat, d_inputRate, d_eventRate, d_spikeTrain, tBack, gactVec, hactVec, d_fE, d_fI, leftTimeRate, lastNegLogRand, d_v_hlf, state, ngTypeE, ngTypeI, ngType, condE, condI, dt, networkSize, nE, seed, it);
            CUDA_CHECK();
            CUDA_CALL(cudaEventRecord(kStop, 0));
            CUDA_CALL(cudaEventSynchronize(kStop));
            CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
            time1 += time;
            CUDA_CALL(cudaEventRecord(initialSpikesObtained, s1));
            if (printStep) {
                printf("A single step of compute_V cost %fms\n", time);
            }
            /* Spike correction */
            bool matching = true;
            unsigned int imatch = 0;
            CUDA_CALL(cudaEventRecord(kStart, 0));
            while (matching || imatch < networkSize) {
                CUDA_CALL(cudaStreamWaitEvent(s1, initialSpikesObtained, 0));
                correct_spike <<<b1, b2, 0, s1>>> (d_matched, d_spikeTrain, d_v_hlf, v_old, d_preMat, ngTypeE, ngTypeI, condE, condI, dt, nE, networkSize);
                CUDA_CHECK();
                CUDA_CALL(cudaMemcpyAsync(&matched, d_matched, networkSize*sizeof(bool), cudaMemcpyDeviceToHost, s1));
                matching = false;
                for (int iw = 0; iw < warpSize; iw ++) {
                    bool iwMatch = true;
                    for (unsigned int j=0; j<networkSize/warpSize; j++) {
                        if (matched[j*warpSize + iw]) {
                            iwMatch = false;
                            break;
                        }
                    }
                    if (!iwMatch) {
                        matching = true;
                        break;
                    }
                }
                imatch++;
            }
            printf("%i matching iterations\n", imatch);
            it = false;
            /* Write voltage of last step to disk */
            v_file.write((char*)&(v[n*batchOffset]),               n*sizeof(double));
            /* Write spikeTrain of last step to disk */
            spike_file.write((char*)&(spikeTrain[n*batchOffset]),  n*sizeof(double));
            /* Copy voltage to host */
            CUDA_CALL(cudaMemcpyAsync(&(v[offset]), v_current, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaMemcpyAsync(eventRate, d_eventRate, networkSize * sizeof(int), cudaMemcpyDeviceToHost, s1));
            /* Copy spikeTrain to host */
            CUDA_CALL(cudaMemcpyAsync(&(spikeTrain[offset]), d_spikeTrain, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s1));

            CUDA_CALL(cudaStreamWaitEvent(s2, spikeCorrected, 0));
            /* Recalibrate conductance */
            // recal E
            CUDA_CALL(cudaEventRecord(kStart, 0));
            prepare_cond <<<b1E, b2E, 0, s2>>> (tBack, spikeTrain, gactVec, hactVec, condE, dt, ngTypeE, 0, networkSize);
            recal_G <<<rgE_b1,rgE_b2,rgE_shared,s2>>> (d_gE, d_hE, d_preMat,
                                                     gactVec, hactVec,
                                                     gE_b1y, hE_b1y,
                                                     networkSize, 0, ngTypeE, s_actVec_lE, msE);
            CUDA_CHECK();
            // recal I
            CUDA_CALL(cudaStreamWaitEvent(s3, spikeCorrected, 0));
            prepare_cond <<<b1I, b2I, 0, s2>>> (tBack, spikeTrain, gactVec, hactVec, condI, dt, ngTypeE, nE, networkSize);
            recal_G<<<rgI_b1,rgI_b2,rgI_shared,s3>>>(d_gI, d_hI, d_preMat,
                                                     gactVec, hactVec,
                                                     gI_b1y, hI_b1y,
                                                     networkSize, nE, ngTypeI, s_actVec_lI, msI);
            CUDA_CHECK();
            if (rgE_b1.x >= 32) {
                //  reduce sum
                reduce_G<<<networkSize, rE_b2, sizeof(double)*2*rE_b2, s2>>>(d_gE, d_hE, gE_b1y, hE_b1y, ngTypeE, rgE_b1.x);
                CUDA_CHECK();
            }
            if (rgI_b1.x >= 32) {
                reduce_G<<<networkSize, rI_b2, sizeof(double)*2*rI_b2, s3>>>(d_gI, d_hI, gI_b1y, hI_b1y, ngTypeI, rgI_b1.x);
                CUDA_CHECK();
            }
            CUDA_CALL(cudaEventRecord(kStop, 0));
            CUDA_CALL(cudaEventSynchronize(kStop));
            CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
            if (printStep) {
                printf("A single step of recal_G cost %fms\n", time);
            }
            time2 += time;
            /* Write conductance of last step to disk */
            gE_file.write((char*)&(gE[n*ngTypeE*batchOffset]),     n*ngTypeE*sizeof(double));
            gI_file.write((char*)&(gI[n*ngTypeE*batchOffset]),     n*ngTypeI*sizeof(double));
            /* Copy conductance to host */
            CUDA_CALL(cudaMemcpyAsync(&(gE[offset*ngTypeE]), d_gE, networkSize * ngTypeE * sizeof(double), cudaMemcpyDeviceToHost, s2));
            CUDA_CALL(cudaMemcpyAsync(&(gI[offset*ngTypeI]), d_gI, networkSize * ngTypeI * sizeof(double), cudaMemcpyDeviceToHost, s2));
            CUDA_CALL(cudaEventRecord(gReady, s2));
            //printf("\r total: %3.1f, batch: %3.1f", 100.0f*float(ibatch+1)/nbatch, float(i)/copySize);
            printf("\r stepping: %3.1f%%", 100.0f*float(i+1)/nstep);
            fflush(stdout);
            double _events = 0.0f;
            int _spikes = 0;
            for (int j=0; j<networkSize; j++) {
                _events += eventRate[j];
                if (spikeTrain[j] < dt) {
                    _spikes++;
                }
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
    spike_file.write((char*)spikeTrain, networkSize * sizeof(int));
    gE_file.write((char*)gE, networkSize * ngTypeE * sizeof(double));
    gI_file.write((char*)gI, networkSize * ngTypeI * sizeof(double));
    printf("\n");

    printf("flatRate = %fkHz, realized mean input rate = %fkHz\n", flatRate/1000.0, float(events)/(dt*nstep*networkSize));
    printf("mean firing rate = %fHz\n", float(spikes)/(dt*nstep*networkSize)*1000.0);

    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&time, start, stop));
    printf("CUDA takes %fms, runtime/realtime ratio ms %fms\n", time, time/(dt*nstep));
    printf("compute_V takes %fms, ratio ms %fms\n", time1, time1/(dt*nstep));
    printf("recal_G takes %fms, ratio ms %fms\n", time2, time2/(dt*nstep));

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
    CUDA_CALL(cudaFree(gactVec));
    CUDA_CALL(cudaFree(hactVec));
    CUDA_CALL(cudaFree(d_preMat));
    CUDA_CALL(cudaFree(d_a));
    CUDA_CALL(cudaFree(d_b));
    if (rgE_b1.x >= 32) {
        CUDA_CALL(cudaFree(gE_b1y));
        CUDA_CALL(cudaFree(hE_b1y));
    }
    if (rgI_b1.x >= 32) {
        CUDA_CALL(cudaFree(gI_b1y));
        CUDA_CALL(cudaFree(hI_b1y));
    }
    CUDA_CALL(cudaFree(leftTimeRate));
    CUDA_CALL(cudaFree(lastNegLogRand));
    CUDA_CALL(cudaFree(d_inputRate));
    CUDA_CALL(cudaFree(d_eventRate));
    CUDA_CALL(cudaFree(d_spikeTrain));
    CUDA_CALL(cudaFree(tBack));
    printf("    Device memory freed\n");
    CUDA_CALL(cudaFreeHost(v));
    CUDA_CALL(cudaFreeHost(gE));
    CUDA_CALL(cudaFreeHost(gI));
    CUDA_CALL(cudaFreeHost(eventRate));
    CUDA_CALL(cudaFreeHost(spikeTrain));
    free(preMat);
    printf("    Host memory freed\n");
    return EXIT_SUCCESS;
}