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
    b1 = 16;
    b2 = 16;
    unsigned int nstep = 320;
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
    unsigned int h_nE = b1*b2*3/4;
    unsigned int t = 20;
    double dt = float(t)/float(nstep); // ms
    double flatRate = 500.0f; // Hz
    double ffsE = 1e-2;
    double s = 1.0*ffsE/(b1*b2);
    double ffsI = 1e-2;
    cpu_version(b1*b2, flatRate/1000.0, nstep, dt, h_nE, s, seed, ffsE, ffsI);
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

    /* populate __constant__ variables */
    double h_vE = 14.0f/3.0f; // dimensionaless (non-dimensionalized)
    double h_vI = -2.0f/3.0f;
    double h_vL = 0.0f, h_vT = 1.0f;
    double h_gL_E = 0.05f, h_gL_I = 0.1f; // kHz
    double h_tRef_E = 0.5f, h_tRef_I = 0.25f; // ms
    CUDA_CALL(cudaMemcpyToSymbol(vE,     &h_vE, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(vI,     &h_vI, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(vL,     &h_vL, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(gL_E,   &h_gL_E, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(gL_I,   &h_gL_I, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(tRef_E, &h_tRef_E, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(tRef_I, &h_tRef_I, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(nE,     &h_nE, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    // potentially variable
    CUDA_CALL(cudaMemcpyToSymbol(vT,     &h_vT, sizeof(double), 0, cudaMemcpyHostToDevice));

    /* check for double precision support */
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&properties,device));
    if (!( properties.major >= 2 || (properties.major == 1 && properties.minor >= 3))) {
        printf(" double precision not supported\n");
        return EXIT_FAILURE;
    }

    unsigned int nbatch, batchEnd, batchStep;
    // v, gE, gI, spikeTrain
    double hostMemToDiskPerStep = b1 * b2 * (sizeof(double) + ngTypeE*sizeof(double) + ngTypeI*sizeof(double) + sizeof(int) )/(1024*1024);
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
    CUDA_CALL(cudaMallocHost((void**)&v,          b1 * b2 * sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&gE,         b1 * b2 * ngTypeE * sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&gI,         b1 * b2 * ngTypeI *sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&spikeTrain, b1 * b2 * sizeof(double) * batchStep * alt));
    CUDA_CALL(cudaMallocHost((void**)&eventRate,  b1 * b2 * sizeof(int) * batchStep * alt));
    preMat = (double *)calloc(b1 * b2, sizeof(double));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&d_v,            b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_gE,           b1 * b2 * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_gI,           b1 * b2 * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_hE,           b1 * b2 * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_hI,           b1 * b2 * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_fE,           b1 * b2 * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_fI,           b1 * b2 * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_a,            b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_b,            b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_inputRate,    b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_eventRate,    b1 * b2 * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&d_spikeTrain,   b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&tBack,          b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gactVecE,       b1 * b2 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hactVecE,       b1 * b2 * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gactVecI,       b1 * b2 * ngTypeI * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hactVecI,       b1 * b2 * ngTypeI *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_preMat,       b1 * b2 * b1 * b2 * sizeof(double)));
    /* Allocate space for rng on device */
    CUDA_CALL(cudaMalloc((void **)&state,          b1 * b2 * sizeof(curandStateMRG32k3a)));
    CUDA_CALL(cudaMalloc((void **)&leftTimeRate,   b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&lastNegLogRand, b1 * b2 * sizeof(double)));
    /* Allocate space for partial reduce results on device */
    unsigned int sizeToReduce = b1*b2;
    unsigned int nReduceThreads = sizeToReduce/2;
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
    CUDA_CALL(cudaMalloc((void **)&gEproduct_b1,  b1 * b2 * rn_b1 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gIproduct_b1,  b1 * b2 * rn_b1 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hEproduct_b1,  b1 * b2 * rn_b1 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hIproduct_b1,  b1 * b2 * rn_b1 * ngTypeI * sizeof(double)));



    /* Initialize device arrays */
    // CUDA streams for init
    cudaStream_t i1, i2, i3;
    CUDA_CALL(cudaStreamCreate(&i1));
    CUDA_CALL(cudaStreamCreate(&i2));
    CUDA_CALL(cudaStreamCreate(&i3));
    if (presetInit) {
    } else {
        for (unsigned int i=0; i<b1*b2; i++) {
            v[i] = 0.0f;
            gE[i] = 0.0f;
            gI[i] = 0.0f;
            spikeTrain[i] = -1.0f;
        }
        // init rand generation for poisson
        logRand_init<<<b1,b2,0,i1>>>(lastNegLogRand, state, seed);
        init<double><<<b1,b2,0,i2>>>(d_inputRate, flatRate/1000.0f);
        init<double><<<b1,b2,0,i3>>>(d_v, 0.0f);
        init<double><<<b1,b2,0,i1>>>(leftTimeRate, 0.0f);
        init<double><<<b1,b2*ngTypeE,0,i2>>>(d_fE, ffsE);
        init<double><<<b1,b2*ngTypeI,0,i3>>>(d_fI, ffsI);
        init<double><<<b1,b2*ngTypeE,0,i1>>>(d_gE, 0.0f);
        init<double><<<b1,b2*ngTypeI,0,i2>>>(d_gI, 0.0f);
        init<double><<<b1,b2*ngTypeE,0,i3>>>(d_hE, 0.0f);
        init<double><<<b1,b2*ngTypeI,0,i1>>>(d_hI, 0.0f);
        init<double><<<b1,b2*ngTypeE,0,i2>>>(gactVecE, 0.0f);
        init<double><<<b1,b2*ngTypeI,0,i3>>>(gactVecI, 0.0f);
        init<double><<<b1,b2*ngTypeE,0,i1>>>(hactVecE, 0.0f);
        init<double><<<b1,b2*ngTypeI,0,i2>>>(hactVecI, 0.0f);
        init<double><<<b1*b1*b2,b2,0,i3>>>(d_preMat, s);
    }
    CUDA_CALL(cudaStreamDestroy(i1));
    CUDA_CALL(cudaStreamDestroy(i2));
    CUDA_CALL(cudaStreamDestroy(i3));

    /* Create CUDA events */
    cudaEvent_t start, stop, gReady, spikeCorrected, initialSpikesObtained;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&gReady);
    cudaEventCreate(&spikeCorrected);
    cudaEventCreate(&initialSpikesObtained);
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
    int events = 0;
    int spikes = 0;
    unsigned int ibatch = 0;
    unsigned int bStep = 0;
    unsigned int batchOffset = 0;
    unsigned int copySize = batchStep;
    unsigned int n = b1*b2*copySize;
    //for (int ibatch=0; i<nbatch; ibatch++) {
    //    if(ibatch == nbatch-1) {
    //        copySize = batchEnd;
    //    }
        for (int i=0; i<nstep; i++) {
            unsigned int offset; 
            //offset = b1*b2*(batchOffset + i);
            offset = 0;
            CUDA_CALL(cudaStreamWaitEvent(s1, gReady, 0));
            CUDA_CALL(cudaDeviceSynchronize());
            /* Compute voltage (acquire initial spikes) */
            compute_V<<<b1, b2, shared_mem, s1>>>(d_v, d_gE, d_gI, d_hE, d_hI, d_a, d_b, d_preMat, d_inputRate, d_eventRate, d_spikeTrain, tBack, gactVecE, hactVecE, gactVecI, hactVecI, d_fE, d_fI, leftTimeRate, lastNegLogRand, state, ngTypeE, ngTypeI, condE, condI, dt, b1*b2, seed);
            CUDA_CHECK();
            //CUDA_CALL(cudaEventRecord(initialSpikesObtained, s1));
            /* Spike correction */
            CUDA_CALL(cudaEventRecord(spikeCorrected, s1));
            /* Write voltage of last step to disk */
            v_file.write((char*)&(v[n*batchOffset]),               n*sizeof(double));
            /* Write spikeTrain of last step to disk */
            spike_file.write((char*)&(spikeTrain[n*batchOffset]),  n*sizeof(double));
            /* Copy voltage to host */
            CUDA_CALL(cudaMemcpyAsync(&(v[offset]), d_v, b1 * b2 * sizeof(double), cudaMemcpyDeviceToHost, s1));
            CUDA_CALL(cudaMemcpyAsync(eventRate, d_eventRate, b1 * b2 * sizeof(int), cudaMemcpyDeviceToHost, s1));
            /* Copy spikeTrain to host */
            CUDA_CALL(cudaMemcpyAsync(&(spikeTrain[offset]), d_spikeTrain, b1 * b2 * sizeof(double), cudaMemcpyDeviceToHost, s1));

            CUDA_CALL(cudaStreamWaitEvent(s2, spikeCorrected, 0));
            /* Recalibrate conductance */
            CUDA_CALL(cudaDeviceSynchronize());
            recal_G<<<b1,b2,shared_mem,s2>>>(d_gE, d_gI, d_hE, d_hI, d_preMat, gactVecE, hactVecE, gactVecI, hactVecI, gEproduct_b1, hEproduct_b1, gIproduct_b1, hIproduct_b1, b1*b2, ngTypeE, ngTypeI, rn_b1, rn_b2);
            CUDA_CHECK();
            /* Write conductance of last step to disk */
            gE_file.write((char*)&(gE[n*ngTypeE*batchOffset]),     n*ngTypeE*sizeof(double));
            gI_file.write((char*)&(gI[n*ngTypeE*batchOffset]),     n*ngTypeI*sizeof(double));
            /* Copy conductance to host */
            CUDA_CALL(cudaMemcpyAsync(&(gE[offset*ngTypeE]), d_gE, b1 * b2 * ngTypeE * sizeof(double), cudaMemcpyDeviceToHost, s2));
            CUDA_CALL(cudaMemcpyAsync(&(gI[offset*ngTypeI]), d_gI, b1 * b2 * ngTypeI * sizeof(double), cudaMemcpyDeviceToHost, s2));
            CUDA_CALL(cudaEventRecord(gReady, s2));
            //printf("\r total: %3.1f, batch: %3.1f", 100.0f*float(ibatch+1)/nbatch, float(i)/copySize);
            printf("\r stepping: %3.1f%%", 100.0f*float(i+1)/nstep);
            fflush(stdout);
            double _events = 0.0f;
            int _spikes = 0;
            for (int j=0; j<b1*b2; j++) {
                _events += eventRate[j];
                if (spikeTrain[j] > 0.0f) {
                    _spikes++;
                }
            }
            events += _events;
            spikes += _spikes;
            //printf("instant input rate = %fkHz\n", _events/(b1*b2));
            //printf("instant firing rate = %fHz\n", _fr/(dt*b1*b2)*1000.0);
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

    v_file.write((char*)v, b1 * b2 * sizeof(double));
    spike_file.write((char*)spikeTrain, b1 * b2 * sizeof(int));
    gE_file.write((char*)gE, b1 * b2 * ngTypeE * sizeof(double));
    gI_file.write((char*)gI, b1 * b2 * ngTypeI * sizeof(double));
    printf("\n");
    
    printf("flatRate = %fkHz, realized mean input rate = %fkHz\n", flatRate/1000.0, float(events)/(dt*nstep*b1*b2));
    printf("mean firing rate = %fHz\n", float(spikes)/(dt*nstep*b1*b2)*1000.0);

    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&time, start, stop));
    printf("Runtime = %.1fms, runtime per neuronal ms %.1fms\n", time, time/(dt*nstep));
    //cpu_version(b1*b2, condE, condI, flatRate*dt/1000.0, nstep, dt, ngTypeE, ngTypeI, h_nE, s);

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
    //cpu_version(b1*b2, condE, condI, flatRate*dt/1000.0, nstep, dt, ngTypeE, ngTypeI, h_nE, s);
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
