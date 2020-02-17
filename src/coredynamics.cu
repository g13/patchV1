#include "coredynamics.h"

void logRand_init(Float *logRand,
                  Float *lTR,
                  curandStateMRG32k3a *state,
                  PosIntL seed)
{
    Size id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed+id+offset, 0, 0, &localState);
    Float rand = uniform(&localState);
    logRand[id] = -log(uniform(&localState));
    state[id] = localState;
    lTR[id] = logRand[id]*rand;
}

__global__ 
void preMatRandInit(Float* __restrict__ preMat, 
					Float* __restrict__ v, 
					curandStateMRG32k3a* __restrict__ state,
                    Float sEE, Float sIE, Float sEI, Float sII,
                    Size networkSize, Size nE, BigSize seed)
{
    Size id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed+id, 0, 0, &localState);
    v[id] = vL + uniform(&localState) * (vT-vL) * 0.8;
    Float mean, std, ratio;
    if (id < nE) {
        mean = log(sEE/sqrt(1.0f+1.0f/sEE));
        std = sqrt(log(1.0f+1.0f/sEE));
        ratio = 0.0;
        for (Size i=0; i<nE; i++) {
            Float x = log_normal(&localState, mean, std);
            preMat[i*networkSize + id] = x;
            ratio += x;
        }
        if (sEE > 0) {
            ratio = sEE * nE / ratio;
            for (Size i=0; i<nE; i++) {
                preMat[i*networkSize + id] = preMat[i*networkSize + id]*ratio;
            }
        } else {
            for (Size i=0; i<nE; i++) {
                preMat[i*networkSize + id] = 0.0f;
            }
        }
        //mean = log(sEI/sqrt(1.0f+1.0f/sEI));
        //std = sqrt(log(1.0f+1.0f/sEI));
        mean = sEI;
        std = sEI*0.125;
        ratio = 0.0;
        for (Size i=nE; i<networkSize; i++) {
            //Float x = log_normal(&localState, mean, std);
            Float x = normal(&localState)*std+mean;
            if (x<0) x = 0;
            preMat[i*networkSize + id] = x;
            ratio += x;
        }
        if (sEI > 0){
            ratio = sEI * (networkSize-nE) / ratio;
            for (Size i=nE; i<networkSize; i++) {
                preMat[i*networkSize + id] = preMat[i*networkSize + id]*ratio;
            }
        } else {
            for (Size i=nE; i<networkSize; i++) {
                preMat[i*networkSize + id] = 0.0f;
            }
        }
    } else {
        //mean = log(sIE/sqrt(1.0f+1.0f/sIE));
        //std = sqrt(log(1.0f+1.0f/sIE));
        mean = sIE;
        std = sIE*0.125;
        ratio = 0.0;
        for (Size i=0; i<nE; i++) {
            //Float x = log_normal(&localState, mean, std);
            Float x = normal(&localState)*std+mean;
            if (x<0) x = 0;
            preMat[i*networkSize + id] = x;
            ratio += x;
        }
        if (sIE > 0) {
            ratio = sIE * nE / ratio;
            for (Size i=0; i<nE; i++) {
                preMat[i*networkSize + id] = preMat[i*networkSize + id]*ratio;
            }
        } else {
            for (Size i=0; i<nE; i++) {
                preMat[i*networkSize + id] = 0.0f;
            }
        }
        //mean = log(sII/sqrt(1.0f+1.0f/sII));
        //std = sqrt(log(1.0f+1.0f/sII));
        mean = sII;
        std = sII*0.125;
        ratio = 0.0;
        for (Size i=nE; i<networkSize; i++) {
            //Float x = log_normal(&localState, mean, std);
            Float x = normal(&localState)*std+mean;
            if (x<0) x = 0;
            preMat[i*networkSize + id] = x;
            ratio += x;
        }
        if (sII > 0){
            ratio = sII * (networkSize-nE) / ratio;
            for (Size i=nE; i<networkSize; i++) {
                preMat[i*networkSize + id] = preMat[i*networkSize + id]*ratio;
            }
        } else {
            for (Size i=nE; i<networkSize; i++) {
                preMat[i*networkSize + id] = 0.0f;
            }
        }
    }
}

__global__ 
void f_init(Float* __restrict__ f,
            Size networkSize, Size nE, Size ngType,
            Float Ef, Float If)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nE) {
        for (Size ig=0; ig<ngType; ig++) {
            f[ig*networkSize + id] = Ef;
        }
    } else {
        for (Size ig=0; ig<ngType; ig++) {
            f[ig*networkSize + id] = If;
        }
    }
}

__device__
Float manual_ffinput(Float inputTime[], Float lTR, Float dInput, Int &nInput, Float dt) {
    nInput = 0;
    if (lTR < dt) {
        inputTime[nInput] = lTR;
        nInput++;
        Float tmp = lTR + dInput;
        while (tmp < dt){
            inputTime[nInput] = tmp;
            nInput++;
            tmp += dInput;
        }
        lTR = tmp - dt;
    } else {
        lTR -= dt;
    }
    return lTR;
}
    
__device__
Int set_test_input_time(Float inputTime[],
                        Float dt,
                        Float rate,
                        Float &tau,
                        curandStateMRG32k3a &state)
{
    Int i = 0;
    if (tau > dt) {
        tau -= dt;
        return i;
    } else do {
        inputTime[i] = tau;
        tau -= log(uniform(&state))/rate;
        i++;
        if (i == MAX_FFINPUT_PER_DT) {
            printf("exceeding max input per dt %i\n", MAX_FFINPUT_PER_DT);
            break;
        }
    } while (tau <= dt);
    tau -= dt;
    return i;
}

__device__ 
Int set_input_time(Float inputTime[],
                   Float dt,
                   Float rate,
                   Float &leftTimeRate,
                   Float &lastNegLogRand,
                   curandStateMRG32k3a* __restrict__ state)
{
    Int i = 0;
    Float tau, dTau, negLogRand;
    tau = (lastNegLogRand - leftTimeRate)/rate;
    if (tau > dt) {
        leftTimeRate += (dt * rate);
        return i;
    } else do {
        inputTime[i] = tau;
        negLogRand = -log(uniform(state));
        dTau = negLogRand/rate;
        tau += dTau;
        i++;
        if (i == MAX_FFINPUT_PER_DT) {
            printf("exceeding max input per dt %i\n", MAX_FFINPUT_PER_DT);
            break;
        }
    } while (tau <= dt);
    lastNegLogRand = negLogRand;
    leftTimeRate = (dt - tau + dTau) * rate;
    return i;
}

__device__ void evolve_g(ConductanceShape &cond,
                                  Float &g, 
                                  Float &h, 
                                  Float f,
                                  Float inputTime[],
                                  Int nInput, Float dt, Float dt0, Size ig)
{
    cond.decay_conductance(g, h, dt, ig);
    for (Int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, f, dt0-inputTime[i], ig);
    }
}

__device__ 
Float prep_cond(ConductanceShape &cond, Float g[], Float h[], Float f[], Float inputTime[], Int nInput, Size ngType, Float new_dt, Float dt) {
    // p0 should already be ready.
	Float g_total = 0.0f;
    #pragma unroll
	for (Int ig=0; ig<ngType; ig++) {
		evolve_g(cond, g[ig], h[ig], f[ig], inputTime, nInput, new_dt, dt, ig);
		g_total += g[ig];
	}
    return g_total;
}

__device__ 
void modify_g(ConductanceShape &cond, Float &g0, Float &h0, Float &g1, Float &h1, Float strength, Float dtsp, Float tsp, Float dt, Size i) {
    if (dtsp == 0) {
        h0 += strength;
    } else {
        cond.compute_single_input_conductance(g0, h0, strength, dtsp, i);
    }
    cond.compute_single_input_conductance(g1, h1, strength, dt-tsp, i);
}

__device__
void set_p(LIF* lif, Float gE0[], Float gI0[], Float gE1[], Float gI1[], Float gL) {
	Float gE_t = 0.0f;
#pragma unroll
	for (Size ig = 0; ig < ngTypeE; ig++) {
		gE_t += gE0[ig];
	}
	Float gI_t = 0.0f;
#pragma unroll
	for (Size ig = 0; ig < ngTypeI; ig++) {
		gI_t += gI0[ig];
	}
	lif->set_p0(gE_t, gI_t, gL);

	gE_t = 0.0f;
#pragma unroll
	for (Size ig = 0; ig < ngTypeE; ig++) {
		gE_t += gE1[ig];
	}
	gI_t = 0.0f;
#pragma unroll
	for (Size ig = 0; ig < ngTypeI; ig++) {
		gI_t += gI1[ig];
	}
	lif->set_p1(gE_t, gI_t, gL);
}

__device__
void one(LIF* lif, Float dt, Float tRef, Size id, Float gE, Float gI) {
    lif->tsp = dt;
    lif->spikeCount = 0;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            lif->recompute_v0(dt);
            lif->tBack = -1.0;
        }
        lif->compute_v(dt);
        while (lif->v > vT) {
            // crossed threshold
            lif->compute_spike_time(dt);
            lif->spikeCount++;
            lif->tBack = lif->tsp + tRef;
            if (lif->tBack < dt) {
                lif->recompute(dt);
            } else {
                break;
            }
        }
    }
    __syncwarp();
    if (lif->tBack >= dt) {
        lif->reset_v();
        lif->tBack -= dt;
    }
}

__device__
void initial(LIF* lif, Float dt) {
    lif->tsp = dt;
    lif->correctMe = true;
    lif->spikeCount = 0;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0) {
            lif->recompute_v0(dt);
        }
        lif->compute_v(dt);
        if (lif->v > vT) {
            // crossed threshold
            lif->compute_spike_time(dt); 
        }
    } else {
        lif->reset_v();
        lif->correctMe = false;
    }
}

__device__
void step(LIF* lif, Float t0, Float t1, Float tRef) {
    // not in refractory period
    if (lif->tBack < t1) {
        Float dt = t1 - t0;
        // return from refractory period
        if (lif->tBack > t0) {
            lif->recompute_v0(dt, t0);
        }
        lif->compute_v(dt);
        while (lif->v > vT) {
            // crossed threshold
            lif->compute_spike_time(dt, t0); 
            lif->spikeCount++;
            lif->tBack = lif->tsp + tRef;
            if (lif->tBack < t1) {
                lif->recompute(dt, t0);
            } else {
                break;
            }
        }
    }
    if (lif->v < vI) {
        lif->v = vI;
    }
}

__device__
void dab(LIF* lif, Float t0, Float _dt) {
    Float dt = _dt - t0;
    // return from refractory period
    //#ifdef DEBUG
    assert(lif->tBack < _dt);
	//#endif
    if (lif->tBack > t0) {
        lif->recompute_v0(dt, t0);
    }
    lif->compute_v(dt);
    if (lif->v > vT) {
        // crossed threshold
        lif->compute_spike_time(dt, t0); 
    }
}

__device__
void recal_G(LIF* lif, Float shared[], Size ngType, Size n0, Size n1, ConductanceShape &cond, Float gl[], Float hl[], Float preMat[], Float dt, Size id, Size networkSize) {
    Size n = n1 - n0;
    Float *h = (Float*) shared;
    Float *g = (Float*)(shared + n);
    #pragma unroll
    for (Size ig=0; ig<ngType; ig++) {
        if (n0 <= id && id < n1) {
            Float gact = 0.0f;
            Float hact = 0.0f;
            if (lif->spikeCount > 0) {
                cond.compute_single_input_conductance(gact, hact, lif->spikeCount, dt-lif->tsp, ig);
            }
            __syncwarp();
            g[id-n0] = gact; 
            h[id-n0] = hact; 
        } 
        __syncthreads();
        // optimze mem bandwidth
        Size warp_id = id/warpSize;
        for (Size i=warp_id; i<n+warp_id; i++) {
            Size ibank = i%n;
            Float strength = preMat[ibank*networkSize + id];
            gl[ig] += g[ibank] * strength;
            hl[ig] += h[ibank] * strength;
        }
    }
}

__global__ 
void compute_V_without_ssc(Float* __restrict__ v,
                           Float* __restrict__ gE,
                           Float* __restrict__ gI,
                           Float* __restrict__ hE,
                           Float* __restrict__ hI,
                           Float* __restrict__ preMat,
                           Float* __restrict__ inputRateE,
                           Float* __restrict__ inputRateI,
                           Int* __restrict__ eventRateE,
                           Int* __restrict__ eventRateI,
                           Float* __restrict__ spikeTrain,
                           Size* __restrict__ nSpike,
                           Float* __restrict__ tBack,
                           Float* __restrict__ fE,
                           Float* __restrict__ fI,
                           Float* __restrict__ leftTimeRateE,
                           Float* __restrict__ leftTimeRateI,
                           Float* __restrict__ lastNegLogRandE,
                           Float* __restrict__ lastNegLogRandI,
                           curandStateMRG32k3a* __restrict__ stateE,
                           curandStateMRG32k3a* __restrict__ stateI,
                           ConductanceShape condE, ConductanceShape condI, Float dt, Size networkSize, Size nE, BigSize seed, Float dInputE, Float dInputI, Float t)
{
    extern __shared__ Float shared[];
    Float *spike = shared;
    Size *nsp = (Size*)(shared + blockSize);

    Size id = threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    #if SCHEME == 0
        rk2 lif(v[id], tBack[id]);
    #endif

    #if SCHEME == 1
        impl_rk2 lif(v[id], tBack[id]);
	#endif

	Float gL, tRef;
    if (id < nE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    Size gid;

    Float gE_local[ngTypeE];
    Float hE_local[ngTypeE];
    Float gI_local[ngTypeI];
    Float hI_local[ngTypeI];

    // init cond E 
    Float fE_local[ngTypeE];
    Float gE_t = 0.0f;
    #pragma unroll
    for (Size ig=0; ig<ngTypeE; ig++) {
        gid = networkSize*ig + id;
        gE_local[ig] = gE[gid];
        hE_local[ig] = hE[gid];
        fE_local[ig] = fE[gid];
        gE_t += gE_local[ig];
    }
    Float lTRE = leftTimeRateE[id];
    //  cond I 
    Float fI_local[ngTypeI];
    Float gI_t = 0.0f;
    #pragma unroll
    for (Size ig=0; ig<ngTypeI; ig++) {
        gid = networkSize*ig + id;
        gI_local[ig] = gI[gid];
        hI_local[ig] = hI[gid];
        fI_local[ig] = fI[gid];
        gI_t += gI_local[ig];
    }
    Float lTRI = leftTimeRateI[id];
    lif.set_p0(gE_t, gI_t, gL);
    // Get feedforward input
    // consider use shared memory for dynamic allocation
    Float inputTimeE[MAX_FFINPUT_PER_DT];
    Float inputTimeI[MAX_FFINPUT_PER_DT];
    Int nInputE=0, nInputI=0;
    #ifdef TEST_WITH_MANUAL_FFINPUT
        lTRE = manual_ffinput(inputTimeE, lTRE, dInputE, nInputE, dt);
        lTRI = manual_ffinput(inputTimeI, lTRI, dInputI, nInputI, dt);
    #else
        curandStateMRG32k3a localStateE;
        curandStateMRG32k3a localStateI;
        Float irE = inputRateE[id];
        Float irI = inputRateI[id];
        #ifdef TEST_CONVERGENCE_NO_ROUNDING_ERR
            if (irE > 0) {
                localStateE = stateE[id];
                nInputE = set_test_input_time(inputTimeE, dt, irE, lTRE, localStateE);
		        stateE[id] = localStateE;
            }
            if (irI > 0) {
                localStateI = stateI[id];
		        nInputI = set_test_input_time(inputTimeI, dt, irI, lTRI, localStateI);
		        stateI[id] = localStateI;
            }
        #else
            if (irE > 0) {
                localStateE = stateE[id];
                nInputE = set_input_time(inputTimeE, dt, irE, lTRE, lastNegLogRandE[id], localStateE);
		        stateE[id] = localStateE;
            }
            if (irI > 0) {
                localStateI = stateI[id];
		        nInputI = set_input_time(inputTimeI, dt, irI, lTRI, lastNegLogRandI[id], localStateI);
		        stateI[id] = localStateI;
            }
        #endif
    #endif
    leftTimeRateE[id] = lTRE;
    leftTimeRateI[id] = lTRI;
    //__syncwarp();
    // return a realization of Poisson input rate
    #ifndef FULL_SPEED
        eventRateE[id] = nInputE;
        eventRateI[id] = nInputI;
    #endif
    // evolve g to t+dt with ff input only
    gE_t = prep_cond(condE, gE_local, hE_local, fE_local, inputTimeE, nInputE, ngTypeE, dt, dt); 
    gI_t = prep_cond(condI, gI_local, hI_local, fI_local, inputTimeI, nInputI, ngTypeI, dt, dt); 
    lif.set_p1(gE_t, gI_t, gL);
    // rk2 step
    one(&lif, dt, tRef, id, gE_t, gI_t);
	assert(lif.v <= vT);
    assert(lif.tsp > 0);
    assert(lif.tsp <=dt);
    __syncwarp();
    // write data to global
    spikeTrain[id] = lif.tsp;
    nSpike[id] = lif.spikeCount;
	v[id] = lif.v;
    tBack[id] = lif.tBack;
    spike[id] = lif.tsp;

    //* neat but not faster **
    // recalibrate conductance from cortical spikes using shared mem
    // E
    //recal_G(&lif, shared, ngTypeE, 0, nE, condE, gE_local, hE_local, preMat, dt, id, networkSize);
    // I
    //recal_G(&lif, shared, ngTypeI, nE, networkSize, condI, gI_local, hI_local, &preMat[nE*networkSize], dt, id, networkSize);

    block_reduce<Size>(nsp, lif.spikeCount);
    Size total_spike = nsp[0];
    __syncthreads();
    if (total_spike > 0) {
        nsp[id] = lif.spikeCount;
        __syncthreads();
        #pragma unroll
        for (Size i=0; i<blockSize; i++) {
            Float spikeCount = nsp[i];
            if (spikeCount > 0) {
                Float strength = preMat[i*networkSize + id] * spikeCount;
                if (i < nE) {
                    #pragma unroll
                    for (Size ig=0; ig<ngTypeE; ig++) {
                        condE.compute_single_input_conductance(gE_local[ig], hE_local[ig], strength, dt-spike[i], ig);
                    }
                } else {
                    #pragma unroll
                    for (Size ig=0; ig<ngTypeI; ig++) {
                        condI.compute_single_input_conductance(gI_local[ig], hI_local[ig], strength, dt-spike[i], ig);
                    }
                }
            }
        }
    }

    // update conductance to global memory
    #pragma unroll
    for (Size ig=0; ig<ngTypeE; ig++) {
        gE[id] = gE_local[ig];
        hE[id] = hE_local[ig];
    }
    #pragma unroll
    for (Size ig=0; ig<ngTypeI; ig++) {
        gI[id] = gI_local[ig];
        hI[id] = hI_local[ig];
    }
}

#endif
