#include "coredynamics.h"
#include "CONST.h"
#include "cuda_util.h"
#include "rk2.h"

#ifdef SINGLE_PRECISION
	//using func0 = _float(*)(curandStateMRG32k3a_t*);
	//__device__ func0 uniform = &curand_uniform;
	//__device__ func0 normal = &curand_normal;
	//using func1 = _float(*)(curandStateMRG32k3a_t*, _float, _float);
	//__device__ func1 log_normal = &curand_log_normal;
    #define log_normal curand_log_normal
    #define normal curand_normal
    #define uniform curand_uniform
#else
	//using func0 = _float(*)(curandStateMRG32k3a_t*);
	//__device__ func0 uniform = &curand_uniform_double;
	//__device__ func0 normal = &curand_normal_double;
	//using func1 = _float(*)(curandStateMRG32k3a_t*, _float, _float);
	//__device__ func1 log_normal = &curand_log_normal_double;
    #define log_normal curand_log_normal_double
    #define normal curand_normal_double
    #define uniform curand_uniform_double
#endif

__global__ void logRand_init(_float *logRand, curandStateMRG32k3a *state, unsigned long long seed, _float *lTR, _float dInput, unsigned int offset) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed+id+offset, 0, 0, &localState);
    logRand[id] = -log(uniform(&localState));
    state[id] = localState;

    #ifdef TEST_WITH_MANUAL_FFINPUT
        lTR[id] = uniform(&localState)*dInput;
    #else
        lTR[id] = logRand[id]*uniform(&localState)*dInput;
    #endif
}

__global__ void preMatRandInit(_float* __restrict__ preMat, 
						       _float* __restrict__ v, 
						       curandStateMRG32k3a* __restrict__ state,
_float sEE, _float sIE, _float sEI, _float sII, unsigned int networkSize, unsigned int nE, unsigned long long seed) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed+id, 0, 0, &localState);
    v[id] = vL + uniform(&localState) * (vT-vL) * 0.8;
    _float mean, std, ratio;
    if (id < nE) {
        mean = log(sEE/sqrt(1.0f+1.0f/sEE));
        std = sqrt(log(1.0f+1.0f/sEE));
        ratio = 0.0;
        for (unsigned int i=0; i<nE; i++) {
            _float x = log_normal(&localState, mean, std);
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
            //_float x = log_normal(&localState, mean, std);
            _float x = normal(&localState)*std+mean;
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
            //_float x = log_normal(&localState, mean, std);
            _float x = normal(&localState)*std+mean;
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
            //_float x = log_normal(&localState, mean, std);
            _float x = normal(&localState)*std+mean;
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

__global__ void f_init(_float* __restrict__ f, unsigned networkSize, unsigned int nE, unsigned int ngType, _float Ef, _float If) {
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

__device__  _float manual_ffinput(_float inputTime[], _float lTR, _float dInput, int &nInput, _float dt) {
        nInput = 0;
        if (lTR < dt) {
            inputTime[nInput] = lTR;
            nInput++;
            _float tmp = lTR + dInput;
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
    
__device__ int set_test_input_time(_float inputTime[],
                              _float dt,
                              _float rate,
                              _float &tau,
                              curandStateMRG32k3a &state)
{
    int i = 0;
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

__device__ int set_input_time(_float inputTime[],
                              _float dt,
                              _float rate,
                              _float &leftTimeRate,
                              _float &lastNegLogRand,
                              curandStateMRG32k3a* __restrict__ state)
{
    int i = 0;
    _float tau, dTau, negLogRand;
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
                                  _float &g, 
                                  _float &h, 
                                  _float f,
                                  _float inputTime[],
                                  int nInput, _float dt, _float dt0, unsigned int ig)
{
    cond.decay_conductance(g, h, dt, ig);
    for (int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, f, dt0-inputTime[i], ig);
    }
}

__device__ _float prep_cond(ConductanceShape &cond, _float g[], _float h[], _float f[], _float inputTime[], int nInput, unsigned int ngType, _float new_dt, _float dt) {
    // p0 should already be ready.
	_float g_total = 0.0f;
    #pragma unroll
	for (int ig=0; ig<ngType; ig++) {
		evolve_g(cond, g[ig], h[ig], f[ig], inputTime, nInput, new_dt, dt, ig);
		g_total += g[ig];
	}
    return g_total;
}

__device__ void modify_g(ConductanceShape &cond, _float &g0, _float &h0, _float &g1, _float &h1, _float strength, _float dtsp, _float tsp, _float dt, unsigned int i) {
    if (dtsp == 0) {
        h0 += strength;
    } else {
        cond.compute_single_input_conductance(g0, h0, strength, dtsp, i);
    }
    cond.compute_single_input_conductance(g1, h1, strength, dt-tsp, i);
}

__device__ void set_p(LIF* lif, _float gE0[], _float gI0[], _float gE1[], _float gI1[], _float gL) {
	_float gE_t = 0.0f;
#pragma unroll
	for (unsigned int ig = 0; ig < ngTypeE; ig++) {
		gE_t += gE0[ig];
	}
	_float gI_t = 0.0f;
#pragma unroll
	for (unsigned int ig = 0; ig < ngTypeI; ig++) {
		gI_t += gI0[ig];
	}
	lif->set_p0(gE_t, gI_t, gL);

	gE_t = 0.0f;
#pragma unroll
	for (unsigned int ig = 0; ig < ngTypeE; ig++) {
		gE_t += gE1[ig];
	}
	gI_t = 0.0f;
#pragma unroll
	for (unsigned int ig = 0; ig < ngTypeI; ig++) {
		gI_t += gI1[ig];
	}
	lif->set_p1(gE_t, gI_t, gL);
}

__device__  void one(LIF* lif, _float dt, _float tRef, unsigned int id, _float gE, _float gI) {
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

__device__  void initial(LIF* lif, _float dt) {
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

__device__  void step(LIF* lif, _float t0, _float t1, _float tRef) {
    // not in refractory period
    if (lif->tBack < t1) {
        _float dt = t1 - t0;
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

__device__ void dab(LIF* lif, _float t0, _float _dt) {
    _float dt = _dt - t0;
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

__device__ void recal_G(LIF* lif, _float shared[], unsigned int ngType, unsigned int n0, unsigned int n1, ConductanceShape &cond, _float gl[], _float hl[], _float preMat[], _float dt, unsigned int id, unsigned int networkSize) {
    unsigned int n = n1 - n0;
    _float *h = (_float*) shared;
    _float *g = (_float*)(shared + n);
    #pragma unroll
    for (unsigned int ig=0; ig<ngType; ig++) {
        if (n0 <= id && id < n1) {
            _float gact = 0.0f;
            _float hact = 0.0f;
            if (lif->spikeCount > 0) {
                cond.compute_single_input_conductance(gact, hact, lif->spikeCount, dt-lif->tsp, ig);
            }
            __syncwarp();
            g[id-n0] = gact; 
            h[id-n0] = hact; 
        } 
        __syncthreads();
        // optimze mem bandwidth
        unsigned int warp_id = id/warpSize;
        for (unsigned int i=warp_id; i<n+warp_id; i++) {
            unsigned int ibank = i%n;
            _float strength = preMat[ibank*networkSize + id];
            gl[ig] += g[ibank] * strength;
            hl[ig] += h[ibank] * strength;
        }
    }
}

#if SCHEME < 2
    #ifdef SPIKE_CORRECTION

__global__ void
compute_V(_float* __restrict__ v,
          _float* __restrict__ gE,
          _float* __restrict__ gI,
          _float* __restrict__ hE,
          _float* __restrict__ hI,
          _float* __restrict__ preMat,
          _float* __restrict__ inputRateE,
          _float* __restrict__ inputRateI,
          int* __restrict__ eventRateE,
          int* __restrict__ eventRateI,
          _float* __restrict__ spikeTrain,
          unsigned int* __restrict__ nSpike,
          _float* __restrict__ tBack,
          _float* __restrict__ fE,
          _float* __restrict__ fI,
          _float* __restrict__ leftTimeRateE,
          _float* __restrict__ leftTimeRateI,
          _float* __restrict__ lastNegLogRandE,
          _float* __restrict__ lastNegLogRandI,
          curandStateMRG32k3a* __restrict__ stateE,
          curandStateMRG32k3a* __restrict__ stateI,
          ConductanceShape condE, ConductanceShape condI, _float dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, _float dInputE, _float dInputI, _float t)
{
    __shared__ _float tempSpike[blockSize];
    __shared__ unsigned int spid[warpSize];
    __shared__ unsigned int spikeCount[blockSize];
    unsigned int id = threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    #if SCHEME == 0
        rk2 lif(v[id], tBack[id]);
    #endif

    #if SCHEME == 1
        impl_rk2 lif(v[id], tBack[id]);
    #endif
    _float gL, tRef;
    if (id < nE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    // for committing conductance
    _float gE_local[ngTypeE];
    _float hE_local[ngTypeE];
    _float gI_local[ngTypeI];
    _float hI_local[ngTypeI];
    // for not yet committed conductance
    _float gE_retrace[ngTypeE];
    _float hE_retrace[ngTypeE];
    _float gI_retrace[ngTypeI];
    _float hI_retrace[ngTypeI];
    // init cond E 
	_float fE_local[ngTypeE];
	_float gE_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
		unsigned int gid = networkSize * ig + id;
        gE_local[ig] = gE[gid];
        hE_local[ig] = hE[gid];
		fE_local[ig] = fE[gid];
        gE_retrace[ig] = gE_local[ig];
        hE_retrace[ig] = hE_local[ig];
		gE_t += gE_local[ig];
    }
    _float lTRE  = leftTimeRateE[id];
    //  cond I 
	_float fI_local[ngTypeI];
	_float gI_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
		unsigned int gid = networkSize * ig + id;
		gI_local[ig] = gI[gid];
        hI_local[ig] = hI[gid];
		fI_local[ig] = fI[gid];
        gI_retrace[ig] = gI_local[ig];
        hI_retrace[ig] = hI_local[ig];
		gI_t += gI_local[ig];
    }
    _float lTRI = leftTimeRateI[id];
	lif.set_p0(gE_t, gI_t, gL);
    /* Get feedforward input */
    _float inputTimeE[MAX_FFINPUT_PER_DT];
    _float inputTimeI[MAX_FFINPUT_PER_DT];
    int nInputE=0, nInputI=0;
    #ifdef TEST_WITH_MANUAL_FFINPUT
        lTRE = manual_ffinput(inputTimeE, lTRE, dInputE, nInputE, dt);
        lTRI = manual_ffinput(inputTimeI, lTRI, dInputI, nInputI, dt);
    #else
        curandStateMRG32k3a localStateE;
        curandStateMRG32k3a localStateI;
        _float irE = inputRateE[id];
        _float irI = inputRateI[id];

        #ifdef TEST_CONVERGENCE_NO_ROUNDING_ERR
            if (irE > 0) {
                localStateE = stateE[id];
                nInputE = set_test_input_time(inputTimeE, dt, irE, lTRE,localStateE);
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
    // return a realization of Poisson input rate
	#ifndef FULL_SPEED
		eventRateE[id] = nInputE;
		eventRateI[id] = nInputI;
	#endif
    // set conductances
    gE_t = prep_cond(condE, gE_local, hE_local, fE_local, inputTimeE, nInputE, ngTypeE, dt, dt); 
    gI_t = prep_cond(condI, gI_local, hI_local, fI_local, inputTimeI, nInputI, ngTypeI, dt, dt); 
	lif.set_p1(gE_t, gI_t, gL);
    spikeTrain[id] = dt;
    // initial spike guess
    #ifdef DEBUG
        _float old_v0 = lif.v0;
        _float old_tBack = lif.tBack;
    #endif
    initial(&lif, dt);
    _float v0 = lif.v0;
    #ifdef DEBUG
    	if (lif.tsp < dt) {
    		printf("first %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp);	
    	}
        if (lif.tsp > dt) {
    		printf("efirst %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp);	
        }
    #endif
	assert(lif.tsp <= dt);
	assert(lif.tsp > 0);
    // spike-spike correction
	//__syncthreads();
	find_min(tempSpike, lif.tsp, spid);
	_float t0 = 0.0;
    _float t_hlf = tempSpike[0];
    unsigned int imin = spid[0];
    #ifdef DEBUG
    	if (id == 0) {
    		if (t_hlf == dt) {
    			printf("first_ no spike\n");
    		} else {
    			printf("first_ %u: %e < %e ?\n", imin, t_hlf, dt);
    		}
    	}
    #endif
	__syncthreads();
    int iInputE = 0, iInputI = 0;
    while (t_hlf < dt) {
        // t0 ------- min ---t_hlf
        /************ This may be optimized to be per warp decision **************/
        lif.tsp = dt;
        //unsigned int MASK = __ballot_sync(FULL_MASK, lif.correctMe);
        if (lif.correctMe) {
            // prep inputTime
            int jInputE = iInputE;
            if (jInputE < nInputE) {
                while (inputTimeE[jInputE] < t_hlf) {
                    jInputE++;
                    if (jInputE == nInputE) break;
                }
            }
            int jInputI = iInputI;
            if (jInputI < nInputI) {
                while (inputTimeI[jInputI] < t_hlf) {
                    jInputI++;
                    if (jInputI == nInputI) break;
                }
            }
            // prep retracable conductance
            _float new_dt = t_hlf-t0;
            gE_t = prep_cond(condE, gE_retrace, hE_retrace, fE_local, &inputTimeE[iInputE], jInputE-iInputE, ngTypeE, new_dt, t_hlf);
            gI_t = prep_cond(condI, gI_retrace, hI_retrace, fI_local, &inputTimeI[iInputI], jInputI-iInputI, ngTypeI, new_dt, t_hlf); 
	        lif.set_p1(gE_t, gI_t, gL);
            // commit for next ext. inputs.
            iInputE = jInputE;
            iInputI = jInputI;
            // get tsp decided
            #ifdef DEBUG
			    _float old_v0 = lif.v0;
			    _float old_tBack = lif.tBack;
            #endif
            unsigned int old_count = lif.spikeCount;
            lif.v0 = v0;
            step(&lif, t0, t_hlf, tRef);
            if (id == imin && lif.tsp == dt) {
                lif.spikeCount++;
                lif.tsp = t_hlf;
                lif.tBack = t_hlf + tRef;
            }
            if (lif.tsp < dt) {
                //lif.tsp = t_hlf;
                spikeTrain[id] = lif.tsp;
                spikeCount[id] = lif.spikeCount - old_count;
                //lif.tBack = lif.tsp + tRef;
                if (lif.tBack >= dt) {
                    lif.reset_v();
                    lif.correctMe = false;
                }
                #ifdef DEBUG
                        printf("t0: %e, t_hlf: %e\n", t0, t_hlf);
		                printf("hlf %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %ex%i\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp, spikeCount[id]);
                #endif
		        assert(lif.tsp <= t_hlf+EPS);
		        assert(lif.tsp > t0-EPS);
            }
            if (lif.v > vT) {
                assert(lif.tBack < dt && lif.tBack >= t_hlf-EPS);
            }
        }
        __syncwarp();
        tempSpike[id] = lif.tsp;
        __syncthreads();
        #ifdef DEBUG
            unsigned int counter = 0;
        #endif
        // commit the spikes
        #pragma unroll
        for (unsigned int i=0; i<blockSize; i++) {
            _float tsp = tempSpike[i];
            if (tsp < dt) {
                _float strength = preMat[i*networkSize + id] * spikeCount[i];
                _float dtsp = t_hlf-tsp;
                #ifdef DEBUG
                    if (id==0) {
                        counter++;
                        printf("%u: %e\n", i, tsp);
                    }
                #endif
                if (i < nE) {
                    #pragma unroll
			    	for (unsigned int ig=0; ig<ngTypeE; ig++) {
                        modify_g(condE, gE_retrace[ig], hE_retrace[ig], gE_local[ig], hE_local[ig], strength, dtsp, tsp, dt, ig);
                    }
			    } else {
			    	#pragma unroll
			    	for (unsigned int ig=0; ig<ngTypeI; ig++) {
                        modify_g(condI, gI_retrace[ig], hI_retrace[ig], gI_local[ig], hI_local[ig], strength, dtsp, tsp, dt, ig);
			    	}
                }
            }
        }
        #ifdef DEBUG
            if (id==0) {
                printf("%u spikes\n", counter);
            }
        #endif
		__syncthreads();
        // t_hlf ------------- dt

        lif.tsp = dt;

		if (lif.correctMe) {
            set_p(&lif, gE_retrace, gI_retrace, gE_local, gI_local, gL);
			v0 = lif.v;
            lif.v0 = lif.v;
            #ifdef DEBUG
		        _float old_v0 = lif.v0;
		        _float old_tBack = lif.tBack;
            #endif
            //get putative tsp
            dab(&lif, t_hlf, dt);
            #ifdef DEBUG
			    if (lif.tsp < dt) {
		            printf("end %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e > %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp, t_hlf); 
                } 
                if (lif.tsp > dt) {
		            printf("eend %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e > %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp, t_hlf);
                }
            #endif
            assert(lif.tsp <= dt);
			assert(lif.tsp > t_hlf-EPS);
        }
		// next spike
        find_min(tempSpike, lif.tsp, spid);
		t0 = t_hlf;
        t_hlf = tempSpike[0];
        imin = spid[0];
		__syncthreads();
        #ifdef DEBUG
        	if (id == 0) {
        		if (t_hlf == dt) {
        			printf("end_ no spike\n");
        		} else {
        			printf("end_  %u: %e < %e ?\n", imin, t_hlf, dt);
        		}
        	}
        #endif
    }
    #ifdef DEBUG
        if (lif.v > vT) {
	        printf( "after_ %u-%i: v = %e->%e, tBack = %e, tsp = %e==%e, t_hlf = %e\n", id, lif.correctMe, lif.v0, lif.v, lif.tBack, lif.tsp, tempSpike[id], t_hlf);
	    }
    #endif
	assert(lif.v < vT);
    //__syncwarp();
    // commit conductance to global mem
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
	    unsigned int gid = networkSize * ig + id;
        gE[gid] = gE_local[ig];
        hE[gid] = hE_local[ig];
    }
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
	    unsigned int gid = networkSize * ig + id;
        gI[gid] = gI_local[ig];
        hI[gid] = hI_local[ig];
    }
    nSpike[id] = lif.spikeCount;
	v[id] = lif.v;
    if (lif.tBack > 0) {
        tBack[id] = lif.tBack - dt;
    }
}

    #else

__global__ void
compute_V_without_ssc(_float* __restrict__ v,
                      _float* __restrict__ gE,
                      _float* __restrict__ gI,
                      _float* __restrict__ hE,
                      _float* __restrict__ hI,
                      _float* __restrict__ preMat,
                      _float* __restrict__ inputRateE,
                      _float* __restrict__ inputRateI,
                      int* __restrict__ eventRateE,
                      int* __restrict__ eventRateI,
                      _float* __restrict__ spikeTrain,
                      unsigned int* __restrict__ nSpike,
                      _float* __restrict__ tBack,
                      _float* __restrict__ fE,
                      _float* __restrict__ fI,
                      _float* __restrict__ leftTimeRateE,
                      _float* __restrict__ leftTimeRateI,
                      _float* __restrict__ lastNegLogRandE,
                      _float* __restrict__ lastNegLogRandI,
                      curandStateMRG32k3a* __restrict__ stateE,
                      curandStateMRG32k3a* __restrict__ stateI,
                      ConductanceShape condE, ConductanceShape condI, _float dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, _float dInputE, _float dInputI, _float t)
{
    extern __shared__ _float shared[];
    _float *spike = shared;
    unsigned int *nsp = (unsigned int*)(shared + blockSize);

    unsigned int id = threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    #if SCHEME == 0
        rk2 lif(v[id], tBack[id]);
    #endif

    #if SCHEME == 1
        impl_rk2 lif(v[id], tBack[id]);
	#endif

	_float gL, tRef;
    if (id < nE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    unsigned int gid;

    _float gE_local[ngTypeE];
    _float hE_local[ngTypeE];
    _float gI_local[ngTypeI];
    _float hI_local[ngTypeI];

    // init cond E 
    _float fE_local[ngTypeE];
    _float gE_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
        gid = networkSize*ig + id;
        gE_local[ig] = gE[gid];
        hE_local[ig] = hE[gid];
        fE_local[ig] = fE[gid];
        gE_t += gE_local[ig];
    }
    _float lTRE = leftTimeRateE[id];
    //  cond I 
    _float fI_local[ngTypeI];
    _float gI_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
        gid = networkSize*ig + id;
        gI_local[ig] = gI[gid];
        hI_local[ig] = hI[gid];
        fI_local[ig] = fI[gid];
        gI_t += gI_local[ig];
    }
    _float lTRI = leftTimeRateI[id];
    lif.set_p0(gE_t, gI_t, gL);
    /* Get feedforward input */
    // consider use shared memory for dynamic allocation
    _float inputTimeE[MAX_FFINPUT_PER_DT];
    _float inputTimeI[MAX_FFINPUT_PER_DT];
    int nInputE=0, nInputI=0;
    #ifdef TEST_WITH_MANUAL_FFINPUT
        lTRE = manual_ffinput(inputTimeE, lTRE, dInputE, nInputE, dt);
        lTRI = manual_ffinput(inputTimeI, lTRI, dInputI, nInputI, dt);
    #else
        curandStateMRG32k3a localStateE;
        curandStateMRG32k3a localStateI;
        _float irE = inputRateE[id];
        _float irI = inputRateI[id];
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
    /* evolve g to t+dt with ff input only */
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

    /** neat but not faster **/ 
    // recalibrate conductance from cortical spikes using shared mem
    // E
    //recal_G(&lif, shared, ngTypeE, 0, nE, condE, gE_local, hE_local, preMat, dt, id, networkSize);
    // I
    //recal_G(&lif, shared, ngTypeI, nE, networkSize, condI, gI_local, hI_local, &preMat[nE*networkSize], dt, id, networkSize);

    block_reduce(nsp, lif.spikeCount);
    unsigned int total_spike = nsp[0];
    __syncthreads();
    if (total_spike > 0) {
        nsp[id] = lif.spikeCount;
        __syncthreads();
        #pragma unroll
        for (unsigned int i=0; i<blockSize; i++) {
            _float spikeCount = nsp[i];
            if (spikeCount > 0) {
                _float strength = preMat[i*networkSize + id] * spikeCount;
                if (i < nE) {
                    #pragma unroll
                    for (unsigned int ig=0; ig<ngTypeE; ig++) {
                        condE.compute_single_input_conductance(gE_local[ig], hE_local[ig], strength, dt-spike[i], ig);
                    }
                } else {
                    #pragma unroll
                    for (unsigned int ig=0; ig<ngTypeI; ig++) {
                        condI.compute_single_input_conductance(gI_local[ig], hI_local[ig], strength, dt-spike[i], ig);
                    }
                }
            }
        }
    }

    // update conductance to global memory
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
        gE[id] = gE_local[ig];
        hE[id] = hE_local[ig];
    }
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
        gI[id] = gI_local[ig];
        hI[id] = hI_local[ig];
    }
}

    #endif
#else

__device__ void evolve_g_diff_and_int(ConductanceShape &cond,
                                      _float &ig,
                                      _float &g, 
                                      _float &h, 
                                      _float &dg,
                                      _float f,
                                      _float inputTime[],
                                      int nInput, _float dt, _float dt0, unsigned int igType) {
    cond.diff_and_int_decay(ig, g, h, dg, dt, igType);
    for (int i=0; i<nInput; i++) {
        cond.diff_and_int_cond(ig, g, h, dg, f, dt0-inputTime[i], igType);
    }
}

__device__ _float prep_cond_diff_and_int(ConductanceShape &cond, _float &ig_total, _float &dg_total, _float g[], _float h[], _float f[], _float inputTime[], int nInput, unsigned int ngType, _float new_dt, _float dt) {
	_float g_total = 0.0f;
    dg_total = 0.0f;
    #pragma unroll
	for (int i=0; i<ngType; i++) {
		_float ig, dg;
		evolve_g_diff_and_int(cond, ig, g[i], h[i], dg, f[i], inputTime, nInput, new_dt, dt, i);
		ig_total += ig;
		g_total += g[i];
        dg_total += dg;
	}
    return g_total;
}

__device__ void modify_g_diff_and_int(ConductanceShape &cond, _float &g0, _float &h0, _float &ig0, _float &dg0, _float &g1, _float &h1, _float &ig1, _float &dg1, _float strength, _float dtsp, _float tsp, _float dt, unsigned int i) {
	_float old_ig0 = ig0;
	_float old_ig1 = ig1;
	if (dtsp == 0) {
        h0 += strength;
    } else {
        cond.diff_and_int_cond(ig0, g0, h0, dg0, strength, dtsp, i);
    }
    cond.diff_and_int_cond(ig1, g1, h1, dg1, strength, dt-tsp, i);
	assert(strength >= 0);
	assert(ig0 >= old_ig0);
	assert(ig1 >= old_ig1);
	if (ig1 - old_ig1 < ig0 - old_ig0) {
		printf("delta ig0 = %e, ig1 = %e\n", ig1 - old_ig1, ig0 - old_ig0);
		assert(ig1 - old_ig1 >= ig0 - old_ig0);
	}
}

__global__ void
int_V(_float* __restrict__ v,
	  _float* __restrict__ dVs,
      _float* __restrict__ gE,
      _float* __restrict__ gI,
      _float* __restrict__ hE,
      _float* __restrict__ hI,
      _float* __restrict__ preMat,
      _float* __restrict__ inputRateE,
      _float* __restrict__ inputRateI,
      int* __restrict__ eventRateE,
      int* __restrict__ eventRateI,
      _float* __restrict__ spikeTrain,
      unsigned int* __restrict__ nSpike,
      _float* __restrict__ tBack,
      _float* __restrict__ fE,
      _float* __restrict__ fI,
      _float* __restrict__ leftTimeRateE,
      _float* __restrict__ leftTimeRateI,
      _float* __restrict__ lastNegLogRandE,
      _float* __restrict__ lastNegLogRandI,
      curandStateMRG32k3a* __restrict__ stateE,
      curandStateMRG32k3a* __restrict__ stateI,
      ConductanceShape condE, ConductanceShape condI, _float dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, _float dInputE, _float dInputI, _float t)
{
    __shared__ _float tempSpike[blockSize];
    __shared__ unsigned int spid[warpSize];
    __shared__ unsigned int spikeCount[blockSize];
    unsigned int id = threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
	rangan_int lif(v[id], tBack[id], dVs[id]);
    _float gL, tRef;
    if (id < nE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    // for committing conductance
    _float gE_local[ngTypeE];
    _float hE_local[ngTypeE];
    _float gI_local[ngTypeI];
    _float hI_local[ngTypeI];
    // for not yet committed conductance
    _float gE_retrace[ngTypeE];
    _float hE_retrace[ngTypeE];
    _float gI_retrace[ngTypeI];
    _float hI_retrace[ngTypeI];

    _float dgE_local;
    _float dgI_local;
	_float G_local;
    _float dgI_retrace;
    _float dgE_retrace;
    _float G_retrace;

    // init cond E 
	_float fE_local[ngTypeE];
	_float gE_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
		unsigned int gid = networkSize * ig + id;
        gE_local[ig] = gE[gid];
        hE_local[ig] = hE[gid];
		fE_local[ig] = fE[gid];
        gE_retrace[ig] = gE_local[ig];
        hE_retrace[ig] = hE_local[ig];
		gE_t += gE_local[ig];
    }
    _float lTRE  = leftTimeRateE[id];
    //  cond I 
	_float fI_local[ngTypeI];
	_float gI_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
		unsigned int gid = networkSize * ig + id;
		gI_local[ig] = gI[gid];
        hI_local[ig] = hI[gid];
		fI_local[ig] = fI[gid];
        gI_retrace[ig] = gI_local[ig];
        hI_retrace[ig] = hI_local[ig];
		gI_t += gI_local[ig];
    }
    _float lTRI = leftTimeRateI[id];
	lif.set_p0(gE_t, gI_t, gL);
    /* Get feedforward input */
    _float inputTimeE[MAX_FFINPUT_PER_DT];
    _float inputTimeI[MAX_FFINPUT_PER_DT];
    int nInputE=0, nInputI=0;
    #ifdef TEST_WITH_MANUAL_FFINPUT
        lTRE = manual_ffinput(inputTimeE, lTRE, dInputE, nInputE, dt);
        lTRI = manual_ffinput(inputTimeI, lTRI, dInputI, nInputI, dt);
    #else
        curandStateMRG32k3a localStateE;
        curandStateMRG32k3a localStateI;
        _float irE = inputRateE[id];
        _float irI = inputRateI[id];

        #ifdef TEST_CONVERGENCE_NO_ROUNDING_ERR
            if (irE > 0) {
                localStateE = stateE[id];
                nInputE = set_test_input_time(inputTimeE, dt, irE, lTRE,localStateE);
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
    // return a realization of Poisson input rate
	#ifndef FULL_SPEED
		eventRateE[id] = nInputE;
		eventRateI[id] = nInputI;
	#endif
    // set conductances
    G_local = 0.0f;
    gE_t = prep_cond_diff_and_int(condE, G_local, dgE_local, gE_local, hE_local, fE_local, inputTimeE, nInputE, ngTypeE, dt, dt);
    gI_t = prep_cond_diff_and_int(condI, G_local, dgI_local, gI_local, hI_local, fI_local, inputTimeI, nInputI, ngTypeI, dt, dt);
	lif.set_p1(gE_t, gI_t, gL);
    lif.set_dVs1(dgE_local, dgI_local);
    lif.set_G(G_local, gL, dt);
	assert(G_local >= 0);
    spikeTrain[id] = dt;
    // initial spike guess
    #ifdef DEBUG
        _float old_v0 = lif.v0;
        _float old_tBack = lif.tBack;
    #endif
    initial(&lif, dt);
    _float v0 = lif.v0;
    #ifdef DEBUG
    	if (lif.tsp < dt) {
    		printf("first %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp);	
    	}
        if (lif.tsp > dt) {
    		printf("efirst %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp);	
        }
    #endif
	assert(lif.tsp <= dt);
	assert(lif.tsp > 0);
    // spike-spike correction
	//__syncthreads();
	find_min(tempSpike, lif.tsp, spid);
	_float t0 = 0.0;
    _float t_hlf = tempSpike[0];
    unsigned int imin = spid[0];
    #ifdef DEBUG
    	if (id == 0) {
    		if (t_hlf == dt) {
    			printf("first_ no spike\n");
    		} else {
    			printf("first_ %u: %e < %e ?\n", imin, t_hlf, dt);
    		}
    	}
    #endif
	__syncthreads();
    int iInputE = 0, iInputI = 0;
    while (t_hlf < dt) {
        // t0 ------- min ---t_hlf
        /************ This may be optimized to be per warp decision **************/
        lif.tsp = dt;
        //unsigned int MASK = __ballot_sync(FULL_MASK, lif.correctMe);
        if (lif.correctMe) {
            // prep inputTime
            int jInputE = iInputE;
            if (jInputE < nInputE) {
                while (inputTimeE[jInputE] < t_hlf) {
                    jInputE++;
                    if (jInputE == nInputE) break;
                }
            }
            int jInputI = iInputI;
            if (jInputI < nInputI) {
                while (inputTimeI[jInputI] < t_hlf) {
                    jInputI++;
                    if (jInputI == nInputI) break;
                }
            }
            // prep retracable conductance
            _float new_dt = t_hlf-t0;
            G_retrace = 0.0f;
            gE_t = prep_cond_diff_and_int(condE, G_retrace, dgE_retrace, gE_retrace,  hE_retrace, fE_local, &inputTimeE[iInputE], jInputE-iInputE, ngTypeE, new_dt, t_hlf);
            gI_t = prep_cond_diff_and_int(condI, G_retrace, dgI_retrace, gI_retrace,  hI_retrace, fI_local, &inputTimeI[iInputI], jInputI-iInputI, ngTypeI, new_dt, t_hlf);
			lif.set_p1(gE_t, gI_t, gL);
			lif.set_dVs1(dgE_retrace, dgI_retrace);
			lif.set_G(G_retrace, gL, new_dt);
			assert(G_retrace >= 0);
			assert(G_retrace <= G_local);
            // commit for next ext. inputs.
            iInputE = jInputE;
            iInputI = jInputI;
            // get tsp decided
            #ifdef DEBUG
			    _float old_v0 = lif.v0;
			    _float old_tBack = lif.tBack;
            #endif
            unsigned int old_count = lif.spikeCount;
            lif.v0 = v0;
            step(&lif, t0, t_hlf, tRef);
            if (id == imin && lif.tsp == dt) {
                lif.spikeCount++;
                lif.tsp = t_hlf;
                lif.tBack = t_hlf + tRef;
            }
            if (lif.tsp < dt) {
                //lif.tsp = t_hlf;
                spikeTrain[id] = lif.tsp;
                spikeCount[id] = lif.spikeCount - old_count;
                //lif.tBack = lif.tsp + tRef;
                if (lif.tBack >= dt) {
                    lif.reset_v();
                    lif.correctMe = false;
                }
                #ifdef DEBUG
                        printf("t0: %e, t_hlf: %e\n", t0, t_hlf);
		                printf("hlf %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %ex%i\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp, spikeCount[id]);
                #endif
		        assert(lif.tsp <= t_hlf+EPS);
		        assert(lif.tsp > t0-EPS);
            }
            if (lif.v > vT) {
                assert(lif.tBack < dt && lif.tBack >= t_hlf-EPS);
            }
        }
        __syncwarp();
        tempSpike[id] = lif.tsp;
        __syncthreads();
        #ifdef DEBUG
            unsigned int counter = 0;
        #endif
        // commit the spikes
        #pragma unroll
        for (unsigned int i=0; i<blockSize; i++) {
            _float tsp = tempSpike[i];
            if (tsp < dt) {
                _float strength = preMat[i*networkSize + id] * spikeCount[i];
                _float dtsp = t_hlf-tsp;
                #ifdef DEBUG
                    if (id==0) {
                        counter++;
                        printf("%u: %e\n", i, tsp);
                    }
                #endif
                if (i < nE) {
                    #pragma unroll
			    	for (unsigned int ig=0; ig<ngTypeE; ig++) {
						modify_g_diff_and_int(condE, gE_retrace[ig], hE_retrace[ig], G_retrace, dgE_retrace, gE_local[ig], hE_local[ig], G_local, dgE_local , strength, dtsp, tsp, dt, ig);
                    }
			    } else {
			    	#pragma unroll
			    	for (unsigned int ig=0; ig<ngTypeI; ig++) {
                        modify_g_diff_and_int(condI, gI_retrace[ig], hI_retrace[ig], G_retrace, dgI_retrace, gI_local[ig], hI_local[ig], G_local, dgI_local, strength, dtsp, tsp, dt, ig);
			    	}
                }
            }
        }
        #ifdef DEBUG
            if (id==0) {
                printf("%u spikes\n", counter);
            }
        #endif
		__syncthreads();
        // t_hlf ------------- dt

        lif.tsp = dt;
        set_p(&lif, gE_retrace, gI_retrace, gE_local, gI_local, gL);
		lif.set_dVs1(dgE_local, dgI_local);
		if (lif.correctMe) {
			lif.set_dVs0(dgE_retrace, dgI_retrace);
			assert(G_local >= G_retrace);
			G_local = G_local-G_retrace;
			lif.set_G(G_local, gL, dt-t_hlf);
			v0 = lif.v;
            lif.v0 = lif.v;
            #ifdef DEBUG
		        _float old_v0 = lif.v0;
		        _float old_tBack = lif.tBack;
            #endif
            //get putative tsp
            dab(&lif, t_hlf, dt);
            #ifdef DEBUG
			    if (lif.tsp < dt) {
		            printf("end %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e > %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp, t_hlf); 
                } 
                if (lif.tsp > dt) {
		            printf("eend %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e > %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp, t_hlf);
                }
            #endif
            assert(lif.tsp <= dt);
			assert(lif.tsp > t_hlf-EPS);
        }
		// next spike
        find_min(tempSpike, lif.tsp, spid);
		t0 = t_hlf;
        t_hlf = tempSpike[0];
        imin = spid[0];
		__syncthreads();
        #ifdef DEBUG
        	if (id == 0) {
        		if (t_hlf == dt) {
        			printf("end_ no spike\n");
        		} else {
        			printf("end_  %u: %e < %e ?\n", imin, t_hlf, dt);
        		}
        	}
        #endif
    }
    #ifdef DEBUG
        if (lif.v > vT) {
	        printf( "after_ %u-%i: v = %e->%e, tBack = %e, tsp = %e==%e, t_hlf = %e\n", id, lif.correctMe, lif.v0, lif.v, lif.tBack, lif.tsp, tempSpike[id], t_hlf);
	    }
    #endif
	assert(lif.v < vT);
    //__syncwarp();
    // commit conductance to global mem
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
	    unsigned int gid = networkSize * ig + id;
        gE[gid] = gE_local[ig];
        hE[gid] = hE_local[ig];
    }
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
	    unsigned int gid = networkSize * ig + id;
        gI[gid] = gI_local[ig];
        hI[gid] = hI_local[ig];
    }
    nSpike[id] = lif.spikeCount;
	v[id] = lif.v;
    if (lif.tBack > 0) {
        tBack[id] = lif.tBack - dt;
    }
	dVs[id] = lif.dVs1;
}

#endif
