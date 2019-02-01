#include "patch.h"

int main(int argc, char *argv[])
{
    float time;
    curandStateMRG32k3a *stateE, *randState;
    curandStateMRG32k3a *stateI;
    unsigned long long seed;
    //seed = 183765712;
    seed = std::time(0);
    int device;
    int b1,b2;
    b1 = 32;
    b2 = 32;
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
    struct cudaDeviceProp properties;  
    double *v, *gE, *gI, *preMat, *firstInputE, *firstInputI; 
    int *eventRateE, *d_eventRateE;
    int *eventRateI, *d_eventRateI;
    double *d_v1, *d_preMat;
    double *leftTimeRateE, *lastNegLogRandE;
    double *leftTimeRateI, *lastNegLogRandI;
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
    firstInputE = new double[networkSize];
    firstInputI = new double[networkSize];
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
    CUDA_CALL(cudaMalloc((void **)&d_v1,           networkSize * sizeof(double)));

    logRand_init<<<init_b1,init_b2>>>(lastNegLogRandE, stateE, seed, leftTimeRateE, dInputE, 0);
    CUDA_CHECK();
    logRand_init<<<init_b1,init_b2>>>(lastNegLogRandI, stateI, seed-networkSize, leftTimeRateI, dInputI, networkSize);
    CUDA_CHECK();

    randInit<<<init_b1,init_b2>>>(d_preMat, d_v1, randState, sEE, sIE, sEI, sII, networkSize, nE, seed);
    CUDA_CHECK();
    CUDA_CALL(cudaMemcpy(preMat, d_preMat, networkSize*networkSize*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(firstInputE, leftTimeRateE, networkSize*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(firstInputI, leftTimeRateI, networkSize*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(v, d_v1, networkSize*sizeof(double),cudaMemcpyDeviceToHost));
    printf("storage size of preMat %.1fMb\n", float(networkSize*networkSize*sizeof(double))/1024.0/1024.0);

    cpu_version(networkSize, dInputE, dInputI, nstep, dt, nE, preMat, v, firstInputE, firstInputI, seed, EffsE, IffsE, EffsI, IffsI, theme, flatRateE, flatRateI, condE, condI);

    printf("Cleaning up:\n");
    CUDA_CALL(cudaFree(stateE));
    CUDA_CALL(cudaFree(stateI));
    CUDA_CALL(cudaFree(d_v1));
    CUDA_CALL(cudaFree(d_preMat));
    CUDA_CALL(cudaFree(leftTimeRateE));
    CUDA_CALL(cudaFree(leftTimeRateI));
    CUDA_CALL(cudaFree(lastNegLogRandE));
    CUDA_CALL(cudaFree(lastNegLogRandI));
    printf("    Device memory freed\n");
    CUDA_CALL(cudaFreeHost(v));
    delete []preMat;
    delete []firstInputE;
    delete []firstInputI;
    printf("    Host memory freed\n");
    return EXIT_SUCCESS;
}
