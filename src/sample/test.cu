#include "test.h"

int main(int argc, char *argv[])
{
    std::ofstream v_file, gE_file, gI_file;
    float time;
    //cudaEventCreateWithFlags(&gReady, cudaEventDisableTiming);
    curandStateMRG32k3a *state;
    unsigned long long seed = 1234;
    int device;
    int b1,b2;
    b1 = 64;
    b2 = 16;
    unsigned int h_nE = b1*b2*3/4;
    unsigned int nstep = 10;
    double dt = 0.125f; // ms
    struct cudaDeviceProp properties;  
    double *v, *gE, *gI, *preMat, *inputRate; 
    double *d_v, *d_gE, *d_gI, *d_hE, *d_hI, *d_fE, *d_fI, *d_preMat, *d_inputRate;
    double *d_a, *d_b;
    double *gactVecE, *hactVecE;
    double *gactVecI, *hactVecI;
    double *lastInfo;
    double *actVec;
    double flatRate = 10000.0f; // Hz
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
    /* Overwrite parameters */
    if (argc >= 2)  {
        sscanf(argv[argc-1],"%d",&b2); 
        sscanf(argv[argc-2],"%d",&b1); 
        sscanf(argv[argc-3],"%d",&nstep); 
        sscanf(argv[argc-4],"%f",&flatRate); 
    }
    printf("%i x %i, %i steps, inputRate=%f \n", b1, b2, nstep, flatRate);

    /* populate __constant__ variables */
    double h_vE = 14.0f/3.0f; // dimensionaless (non-dimensionalized)
    double h_vI = -2.0f/3.0f;
    double h_vL = 0.0f, h_vT = 1.0f;
    double h_gL_E = 0.05f, h_gL_I = 0.1f; // kHz
    double h_tRef_E = 2.0f, h_tRef_I = 1.0f; // ms
    CUDA_CALL(cudaMemcpyToSymbol(vE,     &h_vE, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(vI,     &h_vI, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(vL,     &h_vL, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(gL_E,   &h_gL_E, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(gL_I,   &h_gL_I, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(tRef_E, &h_tRef_E, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(tRef_I, &h_tRef_I, sizeof(double), 0, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(nE,     &h_nE, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    // potentially variable
    CUDA_CALL(cudaMemcpyToSymbol(vT,     &h_vT, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

    /* check for double precision support */
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&properties,device));
    if (!( properties.major >= 2 || (properties.major == 1 && properties.minor >= 3))) {
        printf(" double precision not supported\n");
        return EXIT_FAILURE;
    }

    /* Allocate space for results on host */
    //pinned memory
    CUDA_CALL(cudaMallocHost((void**)&v,         b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMallocHost((void**)&gE,        b1 * b2 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMallocHost((void**)&gI,        b1 * b2 * ngTypeI *sizeof(double)));
    CUDA_CALL(cudaMallocHost((void**)&inputRate, b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMallocHost((void**)&actVec,    b1 * b2 * sizeof(double)));
    preMat = (double *)calloc(b1 * b2, sizeof(double));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&d_v,         b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_gE,        b1 * b2 * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_gI,        b1 * b2 * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_hE,        b1 * b2 * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_hI,        b1 * b2 * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_fE,        b1 * b2 * ngTypeE *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_fI,        b1 * b2 * ngTypeI * sizeof(double))); 
    CUDA_CALL(cudaMalloc((void **)&d_a,         b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_b,         b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_inputRate, b1 * b2 * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gactVecE,    b1 * b2 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hactVecE,    b1 * b2 * ngTypeI *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gactVecI,    b1 * b2 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hactVecI,    b1 * b2 * ngTypeI *sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&d_preMat,    b1 * b2 * b1 * b2 * sizeof(double)));
    /* Allocate space for rng on device */
    CUDA_CALL(cudaMalloc((void **)&state,       b1 * b2 * sizeof(curandStateMRG32k3a)));
    CUDA_CALL(cudaMalloc((void **)&lastInfo,    b1 * b2 * sizeof(double)));
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
    CUDA_CALL(cudaMalloc((void **)&gEproduct_b1,  b1 * b2 * b1 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&gIproduct_b1,  b1 * b2 * b1 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hEproduct_b1,  b1 * b2 * b1 * ngTypeE * sizeof(double)));
    CUDA_CALL(cudaMalloc((void **)&hIproduct_b1,  b1 * b2 * b1 * ngTypeI * sizeof(double)));



    /* Initialize device arrays */
    // CUDA streams for init
    cudaStream_t i1, i2, i3;
    CUDA_CALL(cudaStreamCreate(&i1));
    CUDA_CALL(cudaStreamCreate(&i2));
    CUDA_CALL(cudaStreamCreate(&i3));
    if (presetInit) {
    } else {
        init<double><<<b1,b2,0,i1>>>(d_inputRate, flatRate*dt/1000.0);
        init<double><<<b1,b2*ngTypeE,0,i2>>>(d_fE, 1e-1);
        init<double><<<b1,b2*ngTypeI,0,i3>>>(d_fI, 1e-1);
        CUDA_CHECK();
        CUDA_CALL(cudaMemset(d_v, 0, b1 * b2 * sizeof(double)));
        CUDA_CALL(cudaMemset(d_gE, 0, b1 * b2 * ngTypeE * sizeof(double)));
        CUDA_CALL(cudaMemset(d_gI, 0, b1 * b2 * ngTypeI * sizeof(double)));
        CUDA_CALL(cudaMemset(d_hE, 0, b1 * b2 * ngTypeE * sizeof(double)));
        CUDA_CALL(cudaMemset(d_hI, 0, b1 * b2 * ngTypeI * sizeof(double)));
        CUDA_CALL(cudaMemset(d_preMat, 1, b1 * b2 * b1 * b2 * sizeof(double)));
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
    bool init = true;
    v_file.open("v_ictorious.bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_nerous.bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_berish.bin", std::ios::out|std::ios::binary);
    CUDA_CALL(cudaEventRecord(start, 0));
    double sum = 0.0f;
    double fr = 0.0f;
    for (int i=0; i<nstep; i++) {
        printf("==== i = %i started ===== \n", i);
        CUDA_CALL(cudaStreamWaitEvent(s1, gReady, 0));
        /* Compute voltage (acquire initial spikes) */
        compute_V<<<b1, b2, shared_mem, s1>>>(d_v, d_gE, d_gI, d_hE, d_hI, d_a, d_b, d_preMat, d_inputRate, gactVecE, hactVecE, gactVecI, hactVecI, d_fE, d_fI, lastInfo, state, nstep, ngTypeE, ngTypeI, condE, condI, dt, b1*b2, seed, init);
        CUDA_CHECK();
        //CUDA_CALL(cudaEventRecord(initialSpikesObtained, s1));
        /* Spike correction */
        CUDA_CALL(cudaEventRecord(spikeCorrected, s1));
        /* Write voltage of last step to disk */
        v_file.write((char*)v, b1 * b2 * sizeof(double));
        /* Copy voltage to host */
        CUDA_CALL(cudaMemcpyAsync(v, d_v, b1 * b2 * sizeof(double), cudaMemcpyDeviceToHost, s1));
        CUDA_CALL(cudaMemcpyAsync(inputRate, d_inputRate, b1 * b2 * sizeof(double), cudaMemcpyDeviceToHost, s1));
        CUDA_CALL(cudaMemcpyAsync(actVec, gactVecE, b1 * b2 * sizeof(double), cudaMemcpyDeviceToHost, s1));

        CUDA_CALL(cudaStreamWaitEvent(s2, spikeCorrected, 0));
        /* Recalibrate conductance */
        recal_G<<<b1,b2,shared_mem,s2>>>(d_a, d_b, d_gE, d_gI, d_hE, d_hI,d_preMat, gactVecE, hactVecE, gactVecI, hactVecI, gEproduct_b1, hEproduct_b1, gIproduct_b1, hIproduct_b1, b1*b2, ngTypeE, ngTypeI, rn_b1, rn_b2);
        CUDA_CHECK();
        /* Write conductance of last step to disk */
        gE_file.write((char*)gE, b1 * b2 * ngTypeE * sizeof(double));
        gI_file.write((char*)gI, b1 * b2 * ngTypeI * sizeof(double));
        /* Copy conductance to host */
        CUDA_CALL(cudaMemcpyAsync(gE, d_gE, b1 * b2 * ngTypeE * sizeof(double), cudaMemcpyDeviceToHost, s2));
        CUDA_CALL(cudaMemcpyAsync(gI, d_gI, b1 * b2 * ngTypeI * sizeof(double), cudaMemcpyDeviceToHost, s2));
        CUDA_CALL(cudaEventRecord(gReady, s2));
        init = false;
        printf("==== i = %i ended ===== \n", i);
        double _sum = 0.0f;
        double _fr = 0.0f;
        for (int j=0; j<b1*b2; j++) {
            _sum += inputRate[j];
            if (actVec[j] > 0.0f) {
                _fr += 1.0;
            }
        }
        sum += _sum;
        fr += _fr;
        printf("instant input rate = %fkHz\n", _sum/(b1*b2));
        printf("instant firing rate = %fHz\n", _fr/(dt*b1*b2)*1000.0);

    }

    v_file.write((char*)v, b1 * b2 * sizeof(double));
    gE_file.write((char*)gE, b1 * b2 * ngTypeE * sizeof(double));
    gI_file.write((char*)gI, b1 * b2 * ngTypeI * sizeof(double));
    
    printf("flatRate = %fkHz, realized mean input rate = %fkHz\n", flatRate*dt/1000.0, sum/(nstep*b1*b2));
    printf("mean firing rate = %fHz\n", fr/(dt*nstep*b1*b2)*1000.0);

    CUDA_CALL(cudaEventRecord(stop, 0));
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&time, start, stop));
    printf("Runtime per neuronal time: %f\n", time/(dt*nstep));

    /* Cleanup */
    printf("Cleaning up:\n");
    CUDA_CALL(cudaStreamDestroy(s1));
    CUDA_CALL(cudaStreamDestroy(s2));
    printf("    CUDA streams destroyed\n");
    if (v_file.is_open()) v_file.close();
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
    CUDA_CALL(cudaFree(lastInfo));
    printf("    Device memory freed\n");
    CUDA_CALL(cudaFreeHost(v));
    CUDA_CALL(cudaFreeHost(gE));
    CUDA_CALL(cudaFreeHost(gI));
    CUDA_CALL(cudaFreeHost(actVec));
    CUDA_CALL(cudaFreeHost(inputRate));
    free(preMat);
    printf("    Host memory freed\n");
    return EXIT_SUCCESS;
}
