#include "coredynamics.h"

__global__ void recal_G(double* __restrict__ g,
                        double* __restrict__ h,
                        double* __restrict__ preMat,
                        double* __restrict__ gactVec,
                        double* __restrict__ hactVec,
                        double* __restrict__ g_b1y,
                        double* __restrict__ h_b1y,
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
    #pragma unroll
    for (int ig=0; ig<ngType; ig++) {
        #pragma unroll
        for (int i=0; i<m; i++) {
            // av = double[ngType,#(ns),ns]
            // actVec = double[ngType,n]
            if (threadIdx.x < ns) {
                unsigned int sid = ig*ns + (i*blockDim.x + threadIdx.x);
                unsigned int gid = (ig*n + offset + ns*blockIdx.x) + (i*blockDim.x + threadIdx.x);
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
            unsigned pid = (offset + blockIdx.x*ns + i)*n + id;
            g_t += gaV[sid] * preMat[pid];
            h_t += haV[sid] * preMat[pid];
        }
        if (gridDim.x < 32) {
            unsigned int gid = ig*n + id;
            atomicAdd(&(g[gid]), g_t);
            atomicAdd(&(h[gid]), h_t);
        } else {
            // b1y = double[ngType, m, n]
            unsigned int b1yid = ig*n*gridDim.x + n*blockIdx.x + id;
            g_b1y[b1yid] = g_t;
            h_b1y[b1yid] = h_t;
        }
    }
}

__global__ void reduce_G(double* __restrict__ g,
                         double* __restrict__ h,
                         double* __restrict__ g_b1y, 
                         double* __restrict__ h_b1y,
                         unsigned int ngType, int n) 
{ 
    // n x #(ns)
    extern __shared__ double blk[];
    double* g_blk = blk;
    double* h_blk = &(blk[blockDim.x]);
    for (int ig=0; ig<ngType; ig++) {
        unsigned int gid = ig*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
        if (gid < n) {
            g_blk[threadIdx.x] = g_b1y[gid];
            h_blk[threadIdx.x] = g_b1y[gid];
        } else {
            g_blk[threadIdx.x] = 0.0f;
            h_blk[threadIdx.x] = 0.0f;
        }
        for (int i=blockDim.x/2; i>=32; i>>=1) {
             if (threadIdx.x < i) {
                 g_blk[threadIdx.x] += g_blk[threadIdx.x + i];
                 h_blk[threadIdx.x] += h_blk[threadIdx.x + i];
             }
             __syncthreads();
         }

        double g_warp = g_blk[threadIdx.x];
        double h_warp = h_blk[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            g_warp += __shfl_down_sync(FULL_MASK, g_warp, offset);  
            h_warp += __shfl_down_sync(FULL_MASK, h_warp, offset);  
        }
        if (threadIdx.x == 0) {
            unsigned int id = ig*gridDim.x + blockIdx.x;
            g[id] = g_warp;
            h[id] = g_warp;
        }
    }
}

__global__ void logRand_init(double *logRand, curandStateMRG32k3a *state, unsigned long long seed) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed+id, 0, 0, &localState);
    logRand[id] = -log(curand_uniform_double(&localState));
    //printf("logRand0 = %f\n", logRand[id]);
    //logRand[id] = 1.0f;
    state[id] = localState;
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
                                  unsigned int nInput, double dt, unsigned int ig)
{
    cond.decay_conductance(g, h, dt, ig); 
    for (int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, *f, dt-inputTime[i], ig);
    }
}

__device__  double step(Func_RK2* lif, double dt, double tRef, unsigned int id, double gE_t) {
    lif->tsp = -1.0f;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            lif->compute_pseudo_v0(dt);
            lif->tBack = -1.0f;
        }
        __syncthreads();
        lif->runge_kutta_2(dt);
        while (lif->v > vT && lif->tBack < 0.0f) {
            // crossed threshold

            if (lif->v > vE) {
                printf("#%i exc conductance is too high %f\n", id, gE_t);
            }
            
            lif->tsp = lif->compute_spike_time(dt); 
            lif->tBack = lif->tsp + tRef;
            //printf("neuron #%i fired initially\n", id);
            //assert(lif->tBack > 0);
            if (lif->tBack < dt) {
                // refractory period ended during dt
                lif->compute_v(dt);
                lif->tBack = -1.0f;
                if (lif->v > vT) {
                    printf("multiple spike in one time step, only the last spike is counted, refractory period = %f ms, dt = %f\n", tRef, dt);
                    //assert(lif->v <= vT);
                }
            }
        }
    } 
    if (lif->tBack >= dt) {
        // during refractory period
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
                          double* __restrict__ gactVec,
                          double* __restrict__ hactVec,
                          double* __restrict__ fE,
                          double* __restrict__ fI,
                          double* __restrict__ leftTimeRate,
                          double* __restrict__ lastNegLogRand,
                          curandStateMRG32k3a* __restrict__ state,
                          unsigned int ngTypeE, unsigned int ngTypeI, unsigned int ngType, ConductanceShape condE, ConductanceShape condI, double dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, int nInput, bool it)
{
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
    #ifdef TEST_WITH_MANUAL_FFINPUT
        #pragma unroll
        for (int iInput = 0; iInput < nInput; iInput++) {
            inputTime[iInput] = (iInput + double(id)/networkSize)*dt/nInput;
        }
        // not used if not RAND
        lastNegLogRand[id] = 1.0f;
        leftTimeRate[id] = 0.0f;
    #else
        nInput = set_input_time(inputTime, dt, inputRate[id], &(leftTimeRate[id]), &(lastNegLogRand[id]), &localState);
    #endif
    //__syncwarp();
    //if (it) {
    //    printf("nInput = %i\n", nInput);
    //}
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
        //__syncwarp();
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
        //__syncwarp();
        gI_t += g_i;
        gI[gid] = g_i;
        hI[gid] = h_i;
        // for learning
        //fI[gid] = f_i;
    }
    lif.set_p1(gE_t, gI_t, gL);
    // rk2 step

    spikeTrain[id] = step(&lif, dt, tRef, /*the last 2 args are for deugging*/ id, gE_t);
	v[id] = lif.v;
    tBack[id] = lif.tBack;

    //setup acting vectors
    double g_end, h_end;
    if (spikeTrain[id]>0.0f) {
        if (id < nE) {
            #pragma unroll
            for (int ig=0; ig<ngTypeE; ig++) {
                g_end = 0.0f;
                h_end = 0.0f;
                condE.compute_single_input_conductance(&g_end, &h_end, 1.0f, dt-lif.tsp, ig);
                gid = networkSize*ig+id;
                gactVec[gid] = g_end;
                hactVec[gid] = h_end;
            }
        } else {
            #pragma unroll
            for (int ig=0; ig<ngTypeI; ig++) {
                g_end = 0.0f;
                h_end = 0.0f;
                condI.compute_single_input_conductance(&g_end, &h_end, 1.0f, dt-lif.tsp, ig);
                gid = networkSize*ig+id;
                gactVec[gid] = g_end;
                hactVec[gid] = h_end;
            }
        }
    } else {
        for (int ig=0; ig<ngType; ig++) {
            gid = networkSize*ig+id;
            gactVec[gid] = 0.0f;
            hactVec[gid] = 0.0f;
        }
    }
}

__device__ void Func_RK2::runge_kutta_2(double dt) {
    double fk0 = eval0(v0);
    double fk1 = eval1(v0 + dt*fk0);
    v = v0 + dt*(fk0+fk1)/2.0f;
}

__device__ double LIF::compute_spike_time(double dt) {
    return (vT-v0)/(v-v0)*dt;
}

__device__ void LIF::compute_v(double dt) {
    v = compute_v1(dt, a0, b0, a1, b1, vL, tBack);
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

__device__ double LIF::eval0(double _v) {
    return eval_LIF(a0,b0,_v);
}

__device__ double LIF::eval1(double _v) {
    return eval_LIF(a1,b1,_v);
}

__device__ void LIF::reset_v() {
    v = vL;
}
