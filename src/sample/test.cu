#include "test.h"

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
    b1 = 128;
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
        sscanf(argv[argc-1],"%d",&seed); 
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
        sscanf(argv[argc-1],"%d",&seed);
        sscanf(argv[argc-2],"%d",&b2);
        sscanf(argv[argc-3],"%d",&b1);
        sscanf(argv[argc-4],"%d",&nstep);
    }
    printf("%i x %i, %i steps, seed = %i\n", b1, b2, nstep, seed);
	unsigned int networkSize = b1*b2;
    int warpSize = 32;
	int init_b2 = warpSize;
	int init_b1 = ceil(networkSize / init_b2);
    unsigned int nE = networkSize*3/4;
    double t = 0.25f;
    double dt = t/float(nstep); // ms
    double flatRate = 10000.0f; // Hz
    //double flatRate = 0.0f; // Hz
    double ffsE = 1e-2;
    double s = 1.0*ffsE/(networkSize);
    double ffsI = 1e-2;
    printf("designated rate = %3.1fHz\n", flatRate);
    printf("nE = %i, nI = %i\n", nE, networkSize-nE);
    cpu_version(networkSize, flatRate/1000.0, nstep, dt, nE, s, seed, ffsE, ffsI);
    struct cudaDeviceProp properties;  
    double *v, *gE, *gI, *preMat; 
    int *eventRate, *d_eventRate;
    double *d_v, *d_gE, *d_gI, *d_hE, *d_hI, *d_fE, *d_fI, *d_preMat, *d_inputRate;
    double *d_a, *d_b;
    double *gactVecE, *hactVecE;
    double *gactVecI, *hactVecI;
    double *leftTimeRate, *lastNegLogRand;
    double *spikeTrain, *d_spikeTrain, *tBack;
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

    while (init_b2 < 256) {
        init_b2 = init_b2*2;
        init_b1 = init_b1/2;
    }
	if (float(networkSize / init_b2) != init_b1) {
		printf("make networkSize multiples of %i", init_b2);
		return EXIT_FAILURE;
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
    preMat = (double *)calloc(networkSize, sizeof(double));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&d_v,            networkSize * sizeof(double)));
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
    CUDA_CALL(cudaMalloc((void **)&gactVecE,       networkSize * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hactVecE,       networkSize * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gactVecI,       networkSize * ngTypeI * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hactVecI,       networkSize * ngTypeI *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_preMat,       networkSize * networkSize * sizeof(double)));
    /* Allocate space for rng on device */
    CUDA_CALL(cudaMalloc((void **)&state,          networkSize * sizeof(curandStateMRG32k3a)));
    CUDA_CALL(cudaMalloc((void **)&leftTimeRate,   networkSize * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&lastNegLogRand, networkSize * sizeof(double)));
    /* Allocate space for partial reduce results on device */

    unsigned int nReduceThreads = networkSize/2;
    unsigned int rn_b1, rn_b2; 
    if (nReduceThreads > 1048576) { //pow(2,20))
        printf("reduce size exceeded 2^20, need extra layer of reduction");
        return EXIT_FAILURE;
    }
    if (b2<32) {
        rn_b1 = b1/4;    
        rn_b2 = b2*2;
    } else {
        if (b2==32) {
            rn_b1 = b1/2;
            rn_b2 = b2;
        } else {
            if (b1 > 64) {
                rn_b1 = b1/2;
                rn_b2 = b2;
            } else {
                rn_b1 = b1;
                rn_b2 = b2/2;
            }
        }
    }
    printf("blockdims for 2-layer reduction: %i x %i and 1 x %i \n", rn_b1, rn_b2, rn_b1/2);

    double *gEproduct_b1, *gIproduct_b1, *hEproduct_b1, *hIproduct_b1;
    CUDA_CALL(cudaMalloc((void **)&gEproduct_b1,  networkSize * rn_b1 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gIproduct_b1,  networkSize * rn_b1 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hEproduct_b1,  networkSize * rn_b1 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hIproduct_b1,  networkSize * rn_b1 * ngTypeI * sizeof(double)));



    /* Create CUDA events */
    cudaEvent_t start, stop, gReady, spikeCorrected, initialSpikesObtained;
    cudaEvent_t kStart, kStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&gReady);
    cudaEventCreate(&spikeCorrected);
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
            spikeTrain[i] = -1.0f;
        }
        // init rand generation for poisson
        logRand_init<<<init_b1,init_b2,0,i1>>>(lastNegLogRand, state, seed);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i2>>>(d_inputRate, flatRate/1000.0f);
        CUDA_CHECK();
        init<double><<<init_b1,init_b2,0,i3>>>(d_v, 0.0f);
        CUDA_CHECK();
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
        init<double><<<init_b1*ngTypeE,init_b2,0,i2>>>(gactVecE, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i3>>>(gactVecI, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeE,init_b2,0,i2>>>(hactVecE, 0.0f);
        CUDA_CHECK();
        init<double><<<init_b1*ngTypeI,init_b2,0,i3>>>(hactVecI, 0.0f);
        CUDA_CHECK();
        //printf("<<<(%i,%i)x(%i,%i)>>>\n", init_b1,init_b1,init_b2, init_b2);
        printf("storage size of preMat %.1fMb\n", float(networkSize*networkSize*sizeof(double))/1024.0/1024.0);
        printf("preMat size = %ix%i = %i\n",init_b1*init_b1*init_b2,init_b2,networkSize*networkSize);
        //init<<<init_b1*init_b1*init_b2,init_b2,0,i2>>>(d_preMat, s, 1);
        dim3 init2D_b1(init_b1*init_b2,init_b1);
        dim3 init2D_b2(init_b2,1);
        init2D<double><<<init2D_b1,init2D_b2,0,i2>>>(d_preMat, s, 1);
        CUDA_CHECK();
        //CUDA_CALL(cudaEventRecord(kStart, 0));
        //int log_b1 = 2;
        //int log_b2 = init_b2*init_b1/2;
        //while (log_b2 > 256) {
        //    log_b2 = log_b2/2;
        //    log_b1 = log_b1*2;
        //}
        //logRand_init<<<log_b1,log_b2,0,i1>>>(lastNegLogRand, state, seed);
        //CUDA_CHECK();
        //CUDA_CALL(cudaEventRecord(kStop, 0));
        //CUDA_CALL(cudaEventSynchronize(kStop));
        //CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
        //printf("logRand_init<<<%ix%i>>> cost %.1fms\n", log_b1, log_b2, time);
        //printf("logRand_init<<<%ix%i>>> cost %.1fms\n", init_b1, init_b2, time);
    }
    CUDA_CALL(cudaStreamDestroy(i1));
    CUDA_CALL(cudaStreamDestroy(i2));
    CUDA_CALL(cudaStreamDestroy(i3));

    /* Create CUDA streams */
    cudaStream_t s1, s2;
    CUDA_CALL(cudaStreamCreate(&s1));
    CUDA_CALL(cudaStreamCreate(&s2));
    unsigned int shared_mem = 48;
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
            compute_V<<<b1, b2, shared_mem, s1>>>(d_v, d_gE, d_gI, d_hE, d_hI, d_a, d_b, d_preMat, d_inputRate, d_eventRate, d_spikeTrain, tBack, gactVecE, hactVecE, gactVecI, hactVecI, d_fE, d_fI, leftTimeRate, lastNegLogRand, state, ngTypeE, ngTypeI, condE, condI, dt, networkSize, nE, seed, it);
            CUDA_CHECK();
            CUDA_CALL(cudaEventRecord(kStop, 0));
            CUDA_CALL(cudaEventSynchronize(kStop));
            CUDA_CALL(cudaEventElapsedTime(&time, kStart, kStop));
            if (printStep) {
                printf("A single step of compute_V cost %fms\n", time);
            }
            time1 += time;
            it = false;
            //CUDA_CALL(cudaEventRecord(initialSpikesObtained, s1));
            /* Spike correction */
            CUDA_CALL(cudaEventRecord(spikeCorrected, s1));
            /* Write voltage of last step to disk */
            v_file.write((char*)&(v[n*batchOffset]),               n*sizeof(double));
            /* Write spikeTrain of last step to disk */
            spike_file.write((char*)&(spikeTrain[n*batchOffset]),  n*sizeof(double));
            /* Copy voltage to host */
            CUDA_CALL(cudaMemcpyAsync(&(v[offset]), d_v, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaMemcpyAsync(eventRate, d_eventRate, networkSize * sizeof(int), cudaMemcpyDeviceToHost, s1));
            /* Copy spikeTrain to host */
            CUDA_CALL(cudaMemcpyAsync(&(spikeTrain[offset]), d_spikeTrain, networkSize * sizeof(double), cudaMemcpyDeviceToHost, s1));

            CUDA_CALL(cudaStreamWaitEvent(s2, spikeCorrected, 0));
            /* Recalibrate conductance */
            CUDA_CALL(cudaEventRecord(kStart, 0));
            flat_recal_G<<<b1,b2,shared_mem,s2>>>(d_gE, d_gI, d_hE, d_hI, d_preMat, gactVecE, hactVecE, gactVecI, hactVecI, gEproduct_b1, hEproduct_b1, gIproduct_b1, hIproduct_b1, networkSize, ngTypeE, ngTypeI, rn_b1, rn_b2);
            //recal_G<<<b1,b2,shared_mem,s2>>>(d_gE, d_gI, d_hE, d_hI, d_preMat, gactVecE, hactVecE, gactVecI, hactVecI, gEproduct_b1, hEproduct_b1, gIproduct_b1, hIproduct_b1, networkSize, ngTypeE, ngTypeI, rn_b1, rn_b2);
            //naive_recal_G<<<b1,b2,shared_mem,s2>>>(d_gE, d_gI, d_hE, d_hI, d_preMat, gactVecE, hactVecE, gactVecI, hactVecI, gEproduct_b1, hEproduct_b1, gIproduct_b1, hIproduct_b1, networkSize, ngTypeE, ngTypeI, rn_b1, rn_b2);
            CUDA_CHECK();
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
                if (spikeTrain[j] > 0.0f) {
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
    printf("    CUDA streams destroyed\n");
    if (v_file.is_open()) v_file.close();
    if (spike_file.is_open()) spike_file.close();
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
    CUDA_CALL(cudaFree(gactVecE));
    CUDA_CALL(cudaFree(gactVecI));
    CUDA_CALL(cudaFree(hactVecE));
    CUDA_CALL(cudaFree(hactVecI));
    CUDA_CALL(cudaFree(d_preMat));
    CUDA_CALL(cudaFree(d_a));
    CUDA_CALL(cudaFree(d_b));
    CUDA_CALL(cudaFree(gEproduct_b1));
    CUDA_CALL(cudaFree(gIproduct_b1));
    CUDA_CALL(cudaFree(hEproduct_b1));
    CUDA_CALL(cudaFree(hIproduct_b1));
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
