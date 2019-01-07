#include "coredynamics.h"

__global__ void recal_G(double* __restrict__ g,
                        double* __restrict__ h,
                        double* __restrict__ preMat,
                        double* __restrict__ gactVec,
                        double* __restrict__ hactVec,
                        double* __restrict__ g_b1x,
                        double* __restrict__ h_b1x,
                        unsigned int n, unsigned int offset, unsigned int ngType, unsigned int ns, int m) 
{
    // 2D blockGrid
    // -> D-1 pieces of actVec 
    // -> D-2 pieces of post-synaptic neurons 
    // 1D threadBlock
    extern __shared__ double actVec[];
    double *gaV = actVec;
    double *haV = &(actVec[ngType*ns]);
    unsigned int id = blockDim.x*blockIdx.y + threadIdx.x;
    unsigned int ss = ns/m;
    #pragma unroll
    for (int ig=0; ig<ngType; ig++) {
        #pragma unroll
        for (int i=0; i<m; i++) {
            // av = double[ngType,#(ns),ns]
            // actVec = double[ngType,n]
            if (threadIdx.x < ss) {
                unsigned int sid = ig*ns + (i*ss + threadIdx.x);
                unsigned int gid = (ig*n + offset + ns*blockIdx.x) + (i*ss + threadIdx.x);
                gaV[sid] = gactVec[gid];
                haV[sid] = hactVec[gid];
            }
        }
    }
    __syncthreads();
    for (int ig=0; ig<ngType; ig++) {
        double g_t = 0.0f;
        double h_t = 0.0f;
        for (int i = 0; i<ns; i++) {
            unsigned sid = ig*ns + i;
            if (gaV[sid] > 0) {
                unsigned pid = (offset + blockIdx.x*ns + i)*n + id;
                double s = preMat[pid];
                g_t += gaV[sid] * s;
                h_t += haV[sid] * s;
            }
        }
        if (gridDim.x < 32) {
            if (g_t > 0) {
                unsigned int gid = ig*n + id;
                atomicAdd(&(g[gid]), g_t);
                atomicAdd(&(h[gid]), h_t);
            }
        } else {
            // b1x = double[ngType, n/ns(gridDim.x), n]
            unsigned int b1xid = ig*n*gridDim.x + blockIdx.x*n + id;
            g_b1x[b1xid] = g_t;
            h_b1x[b1xid] = h_t;
        }
    }
}

__global__ void reduce_G(double* __restrict__ g,
                         double* __restrict__ h,
                         double* __restrict__ g_b1x, 
                         double* __restrict__ h_b1x,
                         unsigned int ngType, int n) 
{ 
    // b1x = double[ngType, n/ns(gridDim.x), n]
    // n x #(ns)
    extern __shared__ double blk[];
    double* g_blk = blk;
    double* h_blk = &(blk[blockDim.x]);
    for (int ig=0; ig<ngType; ig++) {
        unsigned int gid = ig*blockDim.x*gridDim.x + threadIdx.x*gridDim.x + blockIdx.x;
        if (gid < n) {
            // can do coalesce read optimization here (transpose in shared mem)
            g_blk[threadIdx.x] = g_b1x[gid];
            h_blk[threadIdx.x] = g_b1x[gid];
        } else {
            g_blk[threadIdx.x] = 0.0f;
            h_blk[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int i=blockDim.x/2; i>=32; i>>=1) {
            if (threadIdx.x < i) {
                g_blk[threadIdx.x] += g_blk[threadIdx.x + i];
                h_blk[threadIdx.x] += h_blk[threadIdx.x + i];
            }
            __syncthreads();
        }
        if (threadIdx.x < 32) {
            double g_warp = g_blk[threadIdx.x];
            double h_warp = h_blk[threadIdx.x];
            for (int offset = 16; offset > 0; offset /= 2) {
                g_warp += __shfl_down_sync(FULL_MASK, g_warp, offset);  
                h_warp += __shfl_down_sync(FULL_MASK, h_warp, offset);  
            }
            if (threadIdx.x == 0) {
                unsigned int id = ig*gridDim.x + blockIdx.x;
                g[id] += g_warp;
                h[id] += g_warp;
            }
        }
    }
}

__global__ void logRand_init(double *logRand, curandStateMRG32k3a *state, unsigned long long seed, double *lTR, double dInput) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed+id, 0, 0, &localState);
    logRand[id] = -log(curand_uniform_double(&localState));
    state[id] = localState;

    // lTR works as firstInputTime
    #ifdef TEST_WITH_MANUAL_FFINPUT
        lTR[id] = curand_uniform_double(&localState)*dInput;
    #endif
}

__global__ void randInit(double* __restrict__ preMat, 
						 double* __restrict__ v, 
						 curandStateMRG32k3a* __restrict__ state,
double sEE, double sIE, double sEI, double sII, unsigned int networkSize, unsigned int nE, unsigned long long seed) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed+id, 0, 0, &localState);
    v[id] = vL + curand_uniform_double(&localState) * (vT-vL);
    double mean, std, ratio;
    if (id < nE) {
        mean = log(sEE/sqrt(1.0f+1.0f/sEE));
        std = sqrt(log(1.0f+1.0f/sEE));
        ratio = 0.0;
        for (unsigned int i=0; i<nE; i++) {
            double x = curand_log_normal_double(&localState, mean, std);
            preMat[i*networkSize + id] = x;
            ratio += x;
        }
        if (sEE > 0) {
            ratio = sEE * nE / ratio;
            for (unsigned int i=0; i<nE; i++) {
                preMat[i*networkSize + id] = preMat[i*networkSize + id]*ratio;
            }
        } else {
            for (unsigned int i=0; i<nE; i++) {
                preMat[i*networkSize + id] = 0.0f;
            }
        }
        //mean = log(sEI/sqrt(1.0f+1.0f/sEI));
        //std = sqrt(log(1.0f+1.0f/sEI));
        mean = sEI;
        std = sEI*0.125;
        ratio = 0.0;
        for (unsigned int i=nE; i<networkSize; i++) {
            //double x = curand_log_normal_double(&localState, mean, std);
            double x = curand_normal_double(&localState)*std+mean;
            if (x<0) x = 0;
            preMat[i*networkSize + id] = x;
            ratio += x;
        }
        if (sEI > 0){
            ratio = sEI * (networkSize-nE) / ratio;
            for (unsigned int i=nE; i<networkSize; i++) {
                preMat[i*networkSize + id] = preMat[i*networkSize + id]*ratio;
            }
        } else {
            for (unsigned int i=nE; i<networkSize; i++) {
                preMat[i*networkSize + id] = 0.0f;
            }
        }
    } else {
        //mean = log(sIE/sqrt(1.0f+1.0f/sIE));
        //std = sqrt(log(1.0f+1.0f/sIE));
        mean = sIE;
        std = sIE*0.125;
        ratio = 0.0;
        for (unsigned int i=0; i<nE; i++) {
            //double x = curand_log_normal_double(&localState, mean, std);
            double x = curand_normal_double(&localState)*std+mean;
            if (x<0) x = 0;
            preMat[i*networkSize + id] = x;
            ratio += x;
        }
        if (sIE > 0) {
            ratio = sIE * nE / ratio;
            for (unsigned int i=0; i<nE; i++) {
                preMat[i*networkSize + id] = preMat[i*networkSize + id]*ratio;
            }
        } else {
            for (unsigned int i=0; i<nE; i++) {
                preMat[i*networkSize + id] = 0.0f;
            }
        }
        //mean = log(sII/sqrt(1.0f+1.0f/sII));
        //std = sqrt(log(1.0f+1.0f/sII));
        mean = sII;
        std = sII*0.125;
        ratio = 0.0;
        for (unsigned int i=nE; i<networkSize; i++) {
            //double x = curand_log_normal_double(&localState, mean, std);
            double x = curand_normal_double(&localState)*std+mean;
            if (x<0) x = 0;
            preMat[i*networkSize + id] = x;
            ratio += x;
        }
        if (sII > 0){
            ratio = sII * (networkSize-nE) / ratio;
            for (unsigned int i=nE; i<networkSize; i++) {
                preMat[i*networkSize + id] = preMat[i*networkSize + id]*ratio;
            }
        } else {
            for (unsigned int i=nE; i<networkSize; i++) {
                preMat[i*networkSize + id] = 0.0f;
            }
        }
    }
}

__global__ void f_init(double* __restrict__ f, unsigned networkSize, unsigned int nE, unsigned int ngType, double Ef, double If) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nE) {
        for (unsigned int ig=0; ig<ngType; ig++) {
            f[ig*networkSize + id] = Ef;
        }
    } else {
        for (unsigned int ig=0; ig<ngType; ig++) {
            f[ig*networkSize + id] = If;
        }
    }
}

__device__ int set_input_time(double inputTime[],
                              double dt,
                              double rate,
                              double *leftTimeRate,
                              double *lastNegLogRand,
                              curandStateMRG32k3a* __restrict__ state)
{
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
            //printf("rate = %f, lastNegLogRand = %f, leftTimeRate = %f \n", rate, *lastNegLogRand, *leftTimeRate);
            //printf("inputTime[0]: %f, inputTime[1]: %f\n", inputTime[0], inputTime[1]);
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
                                  int nInput, double dt, unsigned int ig)
{
    cond.decay_conductance(g, h, dt, ig); 
    for (int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, *f, dt-inputTime[i], ig);
    }
}

__device__  double step(LIF* lif, double dt, double tRef, unsigned int id, double gE, double gI, double tsp[]) {
    lif->tsp = dt;
    lif->spikeCount = 0;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            lif->recompute_v0(dt);
        }
        lif->implicit_rk2(dt);
        while (lif->v > vT && lif->tBack < dt) {
            // crossed threshold
            lif->compute_spike_time(dt); 
            tsp[lif->spikeCount] = lif->tsp;
            lif->spikeCount++;
            if (lif->spikeCount == MAX_SPIKE_PER_DT) {
                printf("increase MAX_SPIKE_PER_DT or decrease dt\n");
                assert(lif->spikeCount < MAX_SPIKE_PER_DT);
            }
            lif->tBack = lif->tsp + tRef;
            if (lif->tBack < dt) {
                // refractory period ended during dt
                lif->recompute(dt);
            }
        }
    } 
    if (lif->tBack >= dt) {
        // during refractory period
        lif->reset_v();
    }
    lif->tBack -= dt;
#ifdef DEBUG
    if (lif->spikeCount > 1) {
        printf("#%i spiked %i in one time step %f, refractory period = %f ms, only the last tsp is recorded\n", id, lif->spikeCount, dt, tRef);
    }
#endif
    if (lif->v < vI) {
#ifdef DEBUG
		printf("#%i implicit rk2 is A-Stable! something is off gE1 = %f, gI1 = %f, v = %f, v0 = %f, a0 = %f, b0 = %f, a1 = %f, b1 = %f\n", id, gE, gI, lif->v, lif->v0, lif->a0, lif->b0, lif->a1, lif->b1);
#endif
        lif->v = vI;
    }   
    return lif->tsp;
}

__device__ void LIF::implicit_rk2(double dt) {
    v = impl_rk2(dt, a0, b0, a1, b1, v0);
}

__device__ void LIF::compute_spike_time(double dt, double t0) {
    tsp = comp_spike_time(v, v0, dt, t0);
}

__device__ void LIF::recompute(double dt, double t0) {
    double rB = dt/(tBack-t0) - 1; 
    double denorm = 2 + a1*dt;
    double A = (2 - a0*dt)/denorm;
    double B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
    v = A*v0 + B;
}

__device__ void LIF::recompute_v(double dt, double t0) {
    double rB = dt/(tBack-t0) - 1; 
    double denorm = 2 + a1*dt;
    double A = (2 - a0*dt)/denorm;
    double B = (b0 + b1)*dt/denorm;
    v = recomp_v(A, B, rB);
}

__device__ void LIF::recompute_v0(double dt, double t0) {
    double rB = dt/(tBack-t0) - 1; 
    double denorm = 2 + a1*dt;
    double A = (2 - a0*dt)/denorm;
    double B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
}

__device__ void LIF::set_p0(double gE, double gI, double gL) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

__device__ void LIF::set_p1(double gE, double gI, double gL) {
    a1 = get_a(gE, gI, gL);
    b1 = get_b(gE, gI, gL); 
}

__device__ void LIF::reset_v() {
    v = vL;
}

__global__ void compute_V(double* __restrict__ v,
                          double* __restrict__ gE,
                          double* __restrict__ gI,
                          double* __restrict__ hE,
                          double* __restrict__ hI,
                          double* __restrict__ a,
                          double* __restrict__ b,
                          double* __restrict__ preMat,
                          double* __restrict__ inputRateE,
                          double* __restrict__ inputRateI,
                          int* __restrict__ eventRateE,
                          int* __restrict__ eventRateI,
                          double* __restrict__ spikeTrain,
                          unsigned int* __restrict__ nSpike,
                          double* __restrict__ tBack,
                          double* __restrict__ gactVec,
                          double* __restrict__ hactVec,
                          double* __restrict__ fE,
                          double* __restrict__ fI,
                          double* __restrict__ leftTimeRateE,
                          double* __restrict__ leftTimeRateI,
                          double* __restrict__ lastNegLogRandE,
                          double* __restrict__ lastNegLogRandI,
                          curandStateMRG32k3a* __restrict__ stateE,
                          curandStateMRG32k3a* __restrict__ stateI,
                          unsigned int ngTypeE, unsigned int ngTypeI, unsigned int ngType, ConductanceShape condE, ConductanceShape condI, double dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, double dInputE, double dInputI)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    LIF lif(v[id], tBack[id]);
    double gL, tRef;
    if (id < nE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    /* set a0 b0 for the first step */
    double gI_t;
    double gE_t;
    // init cond E 
    gE_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
        gE_t += gE[networkSize*ig + id];
    }
    //  cond I 
    gI_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
        gI_t += gI[networkSize*ig + id];
    }
    lif.set_p0(gE_t, gI_t, gL);
    /* Get feedforward input */
    // consider use shared memory for dynamic allocation
    double inputTimeE[MAX_FFINPUT_PER_DT];
    double inputTimeI[MAX_FFINPUT_PER_DT];
    curandStateMRG32k3a localStateE = stateE[id];
    curandStateMRG32k3a localStateI = stateI[id];
    int nInputE, nInputI;
    #ifdef TEST_WITH_MANUAL_FFINPUT
        nInputE = 0;
        if (leftTimeRateE[id] < dt) {
            inputTimeE[nInputE] = leftTimeRateE[id];
            nInputE++;
            double tmp = leftTimeRateE[id] + dInputE;
            while (tmp < dt){
                inputTimeE[nInputE] = tmp;
                nInputE++;
                tmp += dInputE;
            }
            leftTimeRateE[id] = tmp - dt;
        } else {
            leftTimeRateE[id] -= dt;
        }

        nInputI = 0;
        if (leftTimeRateI[id] < dt) {
            inputTimeI[nInputI] = leftTimeRateI[id];
            nInputI++;
            double tmp = leftTimeRateI[id] + dInputI;
            while (tmp < dt){
                inputTimeI[nInputI] = tmp;
                nInputI++;
                tmp += dInputI;
            }
            leftTimeRateI[id] = tmp - dt;
        } else {
            leftTimeRateI[id] -= dt;
        }
    #else
        nInputE = set_input_time(inputTimeE, dt, inputRateE[id], &(leftTimeRateE[id]), &(lastNegLogRandE[id]), &localStateE);
        nInputI = set_input_time(inputTimeI, dt, inputRateI[id], &(leftTimeRateI[id]), &(lastNegLogRandI[id]), &localStateI);
    #endif
    //__syncwarp();
    // return a realization of Poisson input rate
    eventRateE[id] = nInputE;
    eventRateI[id] = nInputI;
    // update rng state 
    stateE[id] = localStateE;
    stateI[id] = localStateI;
    /* evolve g to t+dt with ff input only */
    unsigned int gid;
    gE_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeE; ig++) {
        gid = networkSize*ig + id;
        double g_i = gE[gid];
        double h_i = hE[gid];
        double f_i = fE[gid];
        evolve_g(condE, &g_i, &h_i, &f_i, inputTimeE, nInputE, dt, ig);
        //__syncwarp();
        gE_t += g_i;
        gE[gid] = g_i;
        hE[gid] = h_i;
        // for learning
        //fE[gid] = f_i;
    }
    gI_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeI; ig++) {
        gid = networkSize*ig + id;
        double g_i = gI[gid];
        double h_i = hI[gid];
        double f_i = fI[gid];
        evolve_g(condI, &g_i, &h_i, &f_i, inputTimeI, nInputI, dt, ig);
        //__syncwarp();
        gI_t += g_i;
        gI[gid] = g_i;
        hI[gid] = h_i;
        // for learning
        //fI[gid] = f_i;
    }
    lif.set_p1(gE_t, gI_t, gL);
    // rk2 step
    double tsp[MAX_SPIKE_PER_DT];
    spikeTrain[id] = step(&lif, dt, tRef, /*the last 2 args are for deugging*/ id, gE_t, gI_t, tsp);
    nSpike[id] = lif.spikeCount;
    if (lif.v < vI) {
#ifdef DEBUG
		printf("#%i something is off gE = %f, gI = %f, v = %f\n", id, gE_t, gI_t, lif.v);
#endif
        lif.v = vI;
    }   
	v[id] = lif.v;
    tBack[id] = lif.tBack;

    //setup acting vectors
    double g_end, h_end;
    if (lif.spikeCount > 0) {
        int ngType;
        ConductanceShape *cond; 
        if (id < nE) {
            ngType = ngTypeE;
            cond = &condE;
        } else {
            ngType = ngTypeI;
            cond = &condI;
        }
        #pragma unroll
        for (int ig=0; ig<ngType; ig++) {
            gid = networkSize*ig+id;
            gactVec[gid] = 0.0f;
            hactVec[gid] = 0.0f;
            for (int i=0; i<lif.spikeCount; i++) {
                g_end = 0.0f;
                h_end = 0.0f;
                cond->compute_single_input_conductance(&g_end, &h_end, 1.0f, dt-tsp[i], ig);
                gactVec[gid] += g_end;
                hactVec[gid] += h_end;
            }
        }
    } else {
        for (int ig=0; ig<ngType; ig++) {
            gid = networkSize*ig + id;
            gactVec[gid] = 0.0f;
            hactVec[gid] = 0.0f;
        }
    }
}
