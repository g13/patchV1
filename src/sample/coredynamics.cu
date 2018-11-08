#include "coredynamics.cuh"

__forceinline__  __device__ double get_a(double gE, double gI, double gL) {
    return gE + gI + gL;
}

__forceinline__  __device__ double get_b(double gE, double gI, double gL) {
    return gE * vE + gI * vI + gL * vL;
}

template <typename T>
__device__ __forceinline__ void warpReduce(volatile T* data, int id, int halfLen) {
    #pragma unroll
    for (unsigned int i=halfLen; i>0; i>>=1) { 
        data[id] += data[id + i];
    }
}

template <typename T>
__device__ __forceinline__ void reduce2(volatile T *g, volatile T *h, int id, int halfLen) {
    if (halfLen >= 32) {
        #pragma unroll 
        for (unsigned int i=halfLen; i>32; i>>=1) { 
            if (id < i) {
                // keep data stored sequentially
                g[id] += g[id + i];
                h[id] += h[id + i];
            }
            __syncthreads();
        }
        // warp size no need to __syncthreads
        if (id < 32) {
            warpReduce(g, id, halfLen);
            warpReduce(h, id, halfLen);
        }
    } else {
        if (id < halfLen) {
            warpReduce(g, id, halfLen);
            warpReduce(h, id, halfLen);
        }
    }
}

template <typename T>
__global__ void partial_dot1d(T* x, T* y1, T* y2, T* g, T* h, int size) {
    extern __shared__ T product[];
    T *product_g = product;
    T *product_h = &(product_g[size]) ;
    unsigned int blockLen = blockDim.x;
    unsigned int blockLen_half = blockDim.x/2;
    // thread index
    unsigned int block_id = threadIdx.x;
    unsigned int grid_id = blockIdx.x;
    unsigned int global_id = grid_id*(2*blockLen) + block_id;
    // elmenent-wise product to shared memory
    product_g[block_id] = x[global_id]*y1[global_id] + x[global_id + blockLen] * y1[global_id+blockLen];
    product_h[block_id] = x[global_id]*y2[global_id] + x[global_id + blockLen] * y2[global_id+blockLen];
    __syncthreads();
    // reduction within block

    //if (block_id == 0 ) {
    //    printf("blockLen/2 %i, dataSize %i \n",  blockLen_half, size);
    //}
    reduce2<T>(product_g, product_h, block_id, blockLen_half);

    if (block_id == 0) {
        g[grid_id] = product_g[0];
        h[grid_id] = product_h[0];
    }
}

template <typename T>
__global__ void final_reduce(T* pg, T* ph, T* g, T* h, int size) {
    extern __shared__ T partials[];
    T* partial_g = partials;
    T* partial_h = &(partial_g[size]);
    unsigned int blockLen = blockDim.x;
    unsigned int blockLen_half = blockDim.x/2;
    // thread index
    unsigned int block_id = threadIdx.x;
    unsigned int global_id = blockIdx.x*(2*blockLen) + block_id;
    // elmenent-wise product to shared memory
    partial_g[block_id] = pg[global_id] + pg[global_id + blockLen];
    partial_h[block_id] = ph[global_id] + ph[global_id + blockLen];
    __syncthreads();
    // reduction within block
    reduce2<T>(partial_g, partial_h, block_id, blockLen_half);

    if (block_id == 0) {
        (*g) += partial_g[0];
        (*h) += partial_h[0];
        //printf("r1 = %f, r2 = %f \n", partial_g[0], partial_h[0]);
    }
}

__global__ void recal_G(double* __restrict__ gE,
                        double* __restrict__ gI,
                        double* __restrict__ hE,
                        double* __restrict__ hI,
                        double* __restrict__ preMat,
                        double* __restrict__ gactVecE,
                        double* __restrict__ hactVecE,
                        double* __restrict__ gactVecI,
                        double* __restrict__ hactVecI,
                        double* __restrict__ gEproduct_b1,
                        double* __restrict__ hEproduct_b1,
                        double* __restrict__ gIproduct_b1,
                        double* __restrict__ hIproduct_b1,
                        unsigned int networkSize, unsigned int ngTypeE, unsigned int ngTypeI, unsigned int b1, unsigned int b2
                        ) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int ig=0; ig<ngTypeE; ig++) {
        unsigned int aid = networkSize*ig;
        unsigned int bid = b1*ngTypeE*id + b1*ig;
        unsigned int gid = aid+id;
        partial_dot1d<double><<<b1, b2, b2*2*sizeof(double)>>>(&(preMat[id]), &(gactVecE[aid]), &(hactVecE[aid]), &(gEproduct_b1[bid]), &(hEproduct_b1[bid]),b2);
        //d_CUDA_CHECK();
        final_reduce<double><<<1, b1/2, b1*sizeof(double)>>>(&(gEproduct_b1[bid]), &(hEproduct_b1[bid]), &(gE[gid]), &(hE[gid]), b1/2);
        //d_CUDA_CHECK();
    }
    //printf("id-%i: %f -> %f\n", id, bgE, gE[id]);
    #pragma unroll
    for (int ig=0; ig<ngTypeI; ig++) {
        unsigned int aid = networkSize*ig;
        unsigned int bid = b1*ngTypeI*id + b1*ig;
        unsigned int gid = aid+id;
        partial_dot1d<double><<<b1, b2, b2*2*sizeof(double)>>>(&(preMat[id]), &(gactVecI[aid]), &(hactVecI[aid]), &(gIproduct_b1[bid]), &(hIproduct_b1[bid]), b2);
        //d_CUDA_CHECK();
        final_reduce<double><<<1, b1/2, b1*sizeof(double)>>>(&(gIproduct_b1[bid]), &(hIproduct_b1[bid]), &(gI[gid]), &(hI[gid]), b1/2);
        //d_CUDA_CHECK();
    }
}

__global__ void logRand_init(double *logRand, curandStateMRG32k3a *state, unsigned long long seed) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed, id, 0, &localState);
    logRand[id] = -log(curand_uniform_double(&localState));
	cuPrintf("%f\n", logRand[id]);
    //logRand[id] = 1.0f;
    state[id] = localState;
}

__device__ int set_input_time(double inputTime[],
                              double dt,
                              double rate,
                              double *leftTimeRate,
                              double *lastNegLogRand,
                              curandStateMRG32k3a* __restrict__ state) {
    int i = 0;
    double tau, dTau, negLogRand;
    tau = (*lastNegLogRand - (*leftTimeRate))/rate;
    if (tau > dt) {
        *leftTimeRate += (dt * rate);
        return i;
    } else do {
        inputTime[i] = tau;
        negLogRand = -log(curand_uniform_double(state));
        dTau = negLogRand/rate;
        tau += dTau;
        i++;
        if (i == MAX_FFINPUT_PER_DT) {
            printf("exceeding max input per dt %i\n", MAX_FFINPUT_PER_DT);
            //printf("inputTime: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", inputTime[0], inputTime[1], inputTime[2], inputTime[3], inputTime[4], inputTime[5], inputTime[6], inputTime[7], inputTime[8], inputTime[9]);
            break;
        }
    } while (tau <= dt);
    *lastNegLogRand = negLogRand;
    *leftTimeRate = (dt - tau + dTau) * rate;
    return i;
}

__host__ __device__ void evolve_g(ConductanceShape &cond,
                                  double* __restrict__ g, 
                                  double* __restrict__ h, 
                                  double* __restrict__ f,
                                  double inputTime[],
                                  unsigned int nInput, double dt, unsigned int ig
                                  ) {

    cond.decay_conductance(g, h, dt, ig); 
    for (int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, *f, dt-inputTime[i], ig);
    }
}

 __device__  double step(Func_RK2* lif, double dt, double tRef) {
    lif->tsp = -1.0f;
    if (lif->tBack <= 0.0f) {
        // not in refractory period
        lif->runge_kutta_2(dt);
        if (lif->v > vT) {
            // crossed threshold
            lif->tsp = lif->compute_spike_time(dt); 
            lif->tBack = lif->tsp + tRef;
            //printf("neuron #%i fired initially\n", id);
        }
    } 
    // return from refractory period
    if (lif->tBack > 0.0f && lif->tBack < dt) {
        lif->compute_pseudo_v0(dt);
        lif->runge_kutta_2(dt);
        lif->tBack = -1.0f;
    } 
    // during refractory period
    if (lif->tBack > dt) {
        lif->reset_v(); 
        lif->tBack -= dt;
    }
    return lif->tsp;
}

__global__ void compute_V(double* __restrict__ v,
                          double* __restrict__ gE,
                          double* __restrict__ gI,
                          double* __restrict__ hE,
                          double* __restrict__ hI,
                          double* __restrict__ a,
                          double* __restrict__ b,
                          double* __restrict__ preMat,
                          double* __restrict__ inputRate,
                          int* __restrict__ eventRate,
                          double* __restrict__ spikeTrain,
                          double* __restrict__ tBack,
                          double* __restrict__ gactVecE,
                          double* __restrict__ hactVecE,
                          double* __restrict__ gactVecI,
                          double* __restrict__ hactVecI,
                          double* __restrict__ fE,
                          double* __restrict__ fI,
                          double* __restrict__ leftTimeRate,
                          double* __restrict__ lastNegLogRand,
                          curandStateMRG32k3a* __restrict__ state,
                          unsigned int ngTypeE, unsigned int ngTypeI, ConductanceShape condE, ConductanceShape condI, double dt, unsigned int networkSize, unsigned int nE, unsigned long long seed) {

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    double gL, tRef;
    if (id < nE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    LIF lif(v[id], tBack[id]);
    /* set a0 b0 for the first step */
    double gI_t;
    double gE_t;
    // init cond E 
    gE_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeE; ig++) {
        gE_t += gE[networkSize*ig + id];
    }
    //  cond I 
    gI_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeI; ig++) {
        gI_t += gI[networkSize*ig + id];
    }
    lif.set_p0(gE_t, gI_t, gL);

    /* Get feedforward input */
    // consider use shared memory for dynamic allocation
    double inputTime[MAX_FFINPUT_PER_DT];
    curandStateMRG32k3a localState = state[id];
    int nInput;
    //if (init) {
    //    nInput = 1;
    //    inputTime[0] = dt*0.9f;
    //    lastNegLogRand[id] = 1.0f;
    //    leftTimeRate[id] = 0.0f;
    //} else {
        nInput = set_input_time(inputTime, dt, inputRate[id], &(leftTimeRate[id]), &(lastNegLogRand[id]), &localState);
    //}
    // return a realization of Poisson input rate
    eventRate[id] = nInput;
    // update rng state 
    state[id] = localState;
    /* evolve g to t+dt with ff input only */
    unsigned int gid;
    gE_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeE; ig++) {
        gid = networkSize*ig + id;
        double g_i = gE[gid];
        double h_i = hE[gid];
        double f_i = fE[gid];
        evolve_g(condE, &g_i, &h_i, &f_i, inputTime, nInput, dt, ig);
        gE_t += g_i;
        gE[gid] = g_i;
        hE[gid] = h_i;
        // for learning
        //fE[gid] = f_i;
    }
    //printf("id %i, exc cond ready.\n",id);
    gI_t = 0.0f;
    /* no feed-forward inhibitory input (setting nInput = 0) */
    #pragma unroll
    for (int ig=0; ig<ngTypeI; ig++) {
        gid = networkSize*ig + id;
        double g_i = gI[gid];
        double h_i = hI[gid];
        double f_i = fI[gid];
        evolve_g(condI, &g_i, &h_i, &f_i, inputTime, 0, dt, ig);
        gI_t += g_i;
        gI[gid] = g_i;
        hI[gid] = h_i;
        // for learning
        //fI[gid] = f_i;
    }
    lif.set_p1(gE_t, gI_t, gL);
    // rk2 step

    spikeTrain[id] = step(&lif, dt, tRef);
	v[id] = lif.v;
    tBack[id] = lif.tBack;

    //setup acting vectors
    double g_end, h_end;
    int spiked = 0;
    if (id < nE) {
        #pragma unroll
        for (int ig=0; ig<ngTypeE; ig++) {
            g_end = 0.0;
            h_end = 0.0;
            if (spikeTrain[id]>0.0f) {
                condE.compute_single_input_conductance(&g_end, &h_end, 1.0f, dt-lif.tsp, ig);
                spiked = 1;
            }
            gid = networkSize*ig+id;
            gactVecE[gid] = spiked*g_end;
            hactVecE[gid] = spiked*h_end;
        }
    } else {
        #pragma unroll
        for (int ig=0; ig<ngTypeI; ig++) {
            g_end = 0.0;
            h_end = 0.0;
            if (spikeTrain[id]>0.0f) {
                condI.compute_single_input_conductance(&g_end, &h_end, 1.0f, dt-lif.tsp, ig);
                spiked = 1;
            }
            gid = networkSize*ig+id;
            gactVecI[gid] = spiked*g_end;
            hactVecI[gid] = spiked*h_end;
        }
    }
    //printf("id-%i, gend %f, hend %f, spiked %i \n",id, g_end, h_end, spiked);
    //if (id == 0) {
    //    printf("fml\n");
    //}
}

__device__ void Func_RK2::runge_kutta_2(double dt) {
    double fk0 = eval0(v0);
    double fk1 = eval1(v0 + dt*fk0);
    v = v0 + dt*(fk0+fk1)/2.0f;
}


__device__ double LIF:: compute_spike_time(double dt) {
    return (vT-v0)/(v-v0)*dt;
}

__device__ void LIF:: compute_pseudo_v0(double dt) {
    v0 = (vL-tBack*(b0 + b1 - a1*b0*dt)/2.0f)/(1.0f+tBack*(-a0 - a1 + a1*a0*dt)/2.0f);
    runge_kutta_2(dt);
}


__device__ void LIF::set_p0(double gE, double gI, double gL) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

__device__ void LIF::set_p1(double gE, double gI, double gL) {
    a1 = get_a(gE, gI, gL);
    b1 = get_b(gE, gI, gL); 
}

inline  __host__ __device__ double eval_LIF(double a, double b, double v) {
    return -a * v + b;
}

__device__ double LIF:: eval0(double _v) {
    return eval_LIF(a0,b0,_v);
}
__device__ double LIF:: eval1(double _v) {
    return eval_LIF(a1,b1,_v);
}

__device__ void LIF:: reset_v() {
    v = vL;
}
