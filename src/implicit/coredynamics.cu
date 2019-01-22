#include "coredynamics.h"

__device__ void warp_min(double* array, unsigned int* id) {
    double value = array[threadIdx.x];
    double index = id[threadIdx.x];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        double compare = __shfl_down_sync(FULL_MASK, value, offset);
        unsigned int comp_id = __shfl_down_sync(FULL_MASK, index, offset);
        if (value > compare) {
            value = compare;
            index = comp_id;
        }
    }
    if (threadIdx.x % warpSize == 0) {
        unsigned int head = threadIdx.x/warpSize;
        array[head] = value;
        id[head] = index;
    }
}

__device__ void find_min(double* array, unsigned int* id) { 
    warp_min(array, id);
    __syncthreads();
    if (threadIdx.x < warpSize) {
        warp_min(array, id);
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

__device__  double dab(LIF &lif, double t0, double t1, double _dt) {
    double dt = t1 - t0;
    lif.tsp = _dt;
    lif.correctMe = true;
    // not in refractory period
    if (lif.tBack < t1) {
        // return from refractory period
        if (lif.tBack > t0) {
            lif.recompute_v0(dt, t0);
        }
        lif.implicit_rk2(dt);
        if (lif.v > vT) {
            // crossed threshold
            lif.compute_spike_time(dt, t0); 
        }
    } else {
        lif.reset_v();
        if (lif.tBack >= _dt) {
            lif.correctMe = false;
        }
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
                          ConductanceShape condE, ConductanceShape condI, double dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, double dInputE, double dInputI)
{
    extern __shared__ double tempSpike[1024];
    extern __shared__ unsigned int spid[1024];
    unsigned int id = threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    LIF lif(v[id], tBack[id]);
    lif.spikeCount = 0;
    double gL, tRef;
    if (id < nE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    // for committing conductance
    double gE_local[ngTypeE];
    double hE_local[ngTypeE];
    double gI_local[ngTypeI];
    double hI_local[ngTypeI];
    // for not yet committed conductance
    double gE_retrace[ngTypeE];
    double hE_retrace[ngTypeE];
    double gI_retrace[ngTypeI];
    double hI_retrace[ngTypeI];

    double gE_t, gI_t;
    // init cond E 
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
        gE_local[ig] = gE[networkSize*ig + id];
        hE_local[ig] = hE[networkSize*ig + id];
        gE_retrace[ig] = gE_local[ig];
        hE_retrace[ig] = hE_local[ig];
    }
    //  cond I 
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
        gI_local[ig] = gI[networkSize*ig + id];
        hI_local[ig] = hI[networkSize*ig + id];
        gI_retrace[ig] = gI_local[ig];
        hI_retrace[ig] = hI_local[ig];
    }
    /* Get feedforward input */
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
    unsigned int gid;
    // set conductances
    prep_cond(lif, condE, gE_retrace, hE_retrace, inputTimeE, nInputE, condI, gI_retrace, hI_retrace, inputTimeI, nInputI, gL, new_dt); 
    // initial spike guess
    tempSpike[id] = dab(lif, 0, dt, dt);

    // spike-spike correction
    __syncthreads();
    double t0 = 0.0;
    find_min(tempSpike, spid);
    double min = tempSpike[0];
    unsigned int imin_old = spid[0];
    unsigned int imin;
    int iInputE = 0, iInputI = 0;
    int jInputE, jInputI;
    while (min < dt) {
        // t0 ------- min ---t_hlf
        if (lif.correctMe) {
            double t_hlf = min;
            double new_dt = t_hlf - t0;
            // prep inputTime
            jInputE = iInputE;
            if (jInputE < MAX_FFINPUT_PER_DT) {
                while (inputTimeE[jInputE] < t_hlf) {
                    jInputE++;
                    if (jInputE == MAX_FFINPUT_PER_DT) break;
                }
            }
            jInputI = iInputI;
            if (jInputI < MAX_FFINPUT_PER_DT) {
                while (inputTimeI[jInputI] < t_hlf) {
                    jInputI++;
                    if (jInputI == MAX_FFINPUT_PER_DT) break;
                }
            }
        }
        __syncwarp();
        // prep retracable conductance
        if (lif.correctMe) {
            #pragma unroll
            for (int ig=0; ig<ngTypE; ig++) {
                gE_retrace[ig] = gE_local[ig];
                hE_retrace[ig] = hE_local[ig];
            }
            #pragma unroll
            for (int ig=0; ig<ngTypI; ig++) {
                gI_retrace[ig] = gI_local[ig];
                hI_retrace[ig] = hI_local[ig];
            }
            prep_cond(lif, condE, gE_retrace, hE_retrace, &inputTimeE[iInputE], jInputE-iInputE, condI, gI_retrace, hI_retrace, &inputTimeI[iInputI], jInputI-iInputI, gL, new_dt); 
            // get putative tsp 
            tempSpike[id] = dab(lif, t0, t_hlf, dt);
        }
        __syncthreads()
        find_min(tempSpike, spid);
        min = tempSpike[0];
        imin = spid[0];
        if (imin != imin_old) {
            imin_old = imin;
            continue;
        }
        // commit the spike
        if (id == imin) {
            spikeTrain[id] = lif.tsp;
            lif.spikeCount++;
            lif.tBack = lif.tsp + tRef;
            if (lif.tBack > dt) {
                lif.reset_v();
                lif.correctMe = false;
            }
        }
        // t_hlf ------------- dt
        new_dt = dt - t_hlf;
        if (lif.correctMe) {
            if (imin < nE) {
                #pragma unroll
                for (unsigned ig=0; ig++; ig<ngTypeE) {
                    condE.compute_single_input_conductance(gE_retrace[ig], hE_retrace[ig], preMat[imin*n + id], t_hlf-min, ig);
                }
            } else {
                #pragma unroll
                for (unsigned ig=0; ig++; ig<ngTypeI) {
                    condI.compute_single_input_conductance(gI_retrace[ig], hI_retrace[ig], preMat[imin*n + id], t_hlf-min, ig);
                }
            }

            // commit for next t0
            #pragma unroll
            for (unsigned ig=0; ig++; ig<ngTypeE) {
                gE_local[ig] = gE_retrace[ig];
                hE_local[ig] = hE_retrace[ig];
            }
            #pragma unroll
            for (unsigned ig=0; ig++; ig<ngTypeE) {
                gI_local[ig] = gI_retrace[ig];
                hI_local[ig] = hI_retrace[ig];
            }
            iInputE = jInputE;
            iInputI = jInputI;

            prep_cond(lif, condE, gE_retrace, hE_retrace, &inputTimeE[iInputE], nInputE-iInputE, condI, gI_retrace, hI_retrace, &inputTimeI[iInputI], nInputI-iInputI, gL, new_dt); 
            tempSpike[id] = dab(lif, t_hlf, dt, dt);
            // next spike
        }
        __syncthreads()
        t0 = min;
        find_min(tempSpike, spid);
        min = tempSpike[0];
        imin_old = spid[0];
    }

    nSpike[id] = lif.spikeCount;
	v[id] = lif.v;
    tBack[id] = lif.tBack;
}

__device__ void prep_cond(LIF &lif, ConductanceShape &condE, double gE[], double hE[], double inputTimeE[], int nInputE,
ConductanceShape &condI, double gI[], double hI[], double inputTimeI[], int nInputI, double gL, double dt) {
    double gE_t = 0.0f;    
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
        gE_t += gE[ig];
    }
    double gI_t = 0.0f;    
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
        gI_t += gI[ig];
    }
    /* set a0 b0 for the first step */
    lif.set_p0(gE_t, gI_t, gL);

    /* evolve g to t+dt with determined inputs only */
    gE_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeE; ig++) {
        evolve_g(condE, &gE[ig], &hE[ig], &fE, &inputTimeE[iInputE], nInputE, dt, ig);
        gE_t += gE[ig];
    }
    gI_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeI; ig++) {
        evolve_g(condI, &gI[ig], &hI[ig], &fI, &inputTimeI[iInputI], nInputI, dt, ig);
        gI_t += gI[ig];
    }
    lif.set_p1(gE_t, gI_t, gL);
}
