#include "coredynamics.h"

__device__ void warp0_min(double* array, unsigned int* id) {
    double value = array[threadIdx.x];
    double index = id[threadIdx.x];
    __syncwarp();
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        double compare = __shfl_down_sync(FULL_MASK, value, offset);
        unsigned int comp_id = __shfl_down_sync(FULL_MASK, index, offset);
        if (value > compare) {
            value = compare;
            index = comp_id;
        }
        __syncwarp();
    }
    if (threadIdx.x == 0) {
        array[0] = value;
        id[0] = index;
    }
}
__device__ void warps_min(double* array, double data, unsigned int* id) {
	double index = threadIdx.x;
    __syncwarp();
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        double comp_data = __shfl_down_sync(FULL_MASK, data, offset);
        unsigned int comp_id = __shfl_down_sync(FULL_MASK, index, offset);
        if (data > comp_data) {
            data = comp_data;
            index = comp_id;
        }
        __syncwarp();
    }
    __syncthreads();
    if (threadIdx.x % warpSize == 0) {
        unsigned int head = threadIdx.x/warpSize;
        array[head] = data;
        id[head] = index;
    }
}

__device__ void find_min(double* array, double data, unsigned int* id) { 
	warps_min(array, data, id);
    __syncthreads();
    if (threadIdx.x < warpSize) {
        warp0_min(array, id);
    }
    __syncthreads();
}

__device__ void warps_reduce(unsigned int* array) {
    unsigned int data = array[threadIdx.x];
    __syncwarp();
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(FULL_MASK, data, offset);
    }
    __syncthreads();
    if (threadIdx.x % warpSize == 0) {
        array[threadIdx.x/warpSize] = data;
    }
}

__device__ void warp0_reduce(unsigned int* array) {
    unsigned int data = array[threadIdx.x];
    __syncwarp();
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        data += __shfl_down_sync(FULL_MASK, data, offset);
    }
    if (threadIdx.x == 0) {
        array[0] = data;
    }
}

__device__ void block_reduce(unsigned int* array) {
	warps_reduce(array);
    __syncthreads();
    if (threadIdx.x < warpSize) {
        warp0_reduce(array);
    }
    __syncthreads();
}

__global__ void logRand_init(double *logRand, curandStateMRG32k3a *state, unsigned long long seed, double *lTR, double dInput, unsigned int offset) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed+id+offset, 0, 0, &localState);
    logRand[id] = -log(curand_uniform_double(&localState));
    state[id] = localState;

    #ifdef TEST_WITH_MANUAL_FFINPUT
        lTR[id] = curand_uniform_double(&localState)*dInput;
    #else
        lTR[id] = curand_uniform_double(&localState);
    #endif
}

__global__ void randInit(double* __restrict__ preMat, 
						 double* __restrict__ v, 
						 curandStateMRG32k3a* __restrict__ state,
double sEE, double sIE, double sEI, double sII, unsigned int networkSize, unsigned int nE, unsigned long long seed) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = state[id];
    curand_init(seed+id, 0, 0, &localState);
    v[id] = vL + curand_uniform_double(&localState) * (vT-vL) * 0.5;
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

__device__ void prep_cond(LIF* lif, ConductanceShape &condE, double gE[], double hE[], double fE[], double inputTimeE[], int nInputE,
	ConductanceShape &condI, double gI[], double hI[], double fI[], double inputTimeI[], int nInputI, double gL, double dt) {
    // p0 should already be ready.
	double gE_t = 0.0f;
    #pragma unroll
	for (int ig=0; ig<ngTypeE; ig++) {
		evolve_g(condE, &gE[ig], &hE[ig], &fE[ig], inputTimeE, nInputE, dt, ig);
		gE_t += gE[ig];
	}
	double gI_t = 0.0f;
    #pragma unroll
	for (int ig=0; ig<ngTypeI; ig++) {
		evolve_g(condI, &gI[ig], &hI[ig], &fI[ig], inputTimeI, nInputI, dt, ig);
		gI_t += gI[ig];
	}
	lif->set_p1(gE_t, gI_t, gL);
}

__device__ void set_p(LIF* lif, double gE0[], double gI0[], double gE1[], double gI1[], double gL) {
	double gE_t = 0.0f;
    #pragma unroll
	for (unsigned int ig=0; ig<ngTypeE; ig++) {
		gE_t += gE0[ig];
	}
	double gI_t = 0.0f;
    #pragma unroll
	for (unsigned int ig=0; ig<ngTypeI; ig++) {
		gI_t += gI0[ig];
	}
	lif->set_p0(gE_t, gI_t, gL);

	gE_t = 0.0f;
    #pragma unroll
	for (unsigned int ig=0; ig<ngTypeE; ig++) {
		gE_t += gE1[ig];
	}
	gI_t = 0.0f;
    #pragma unroll
	for (unsigned int ig=0; ig<ngTypeI; ig++) {
		gI_t += gI1[ig];
	}
	lif->set_p1(gE_t, gI_t, gL);
}

__device__  void one(LIF* lif, double dt, double tRef, unsigned int id, double gE, double gI) {
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

__device__  void initial(LIF* lif, double dt) {
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

__device__  void step(LIF* lif, double t0, double t1, double tRef) {
    // not in refractory period
    if (lif->tBack < t1) {
        double dt = t1 - t0;
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

__device__  void dab(LIF* lif, double t0, double _dt) {
    double dt = _dt - t0;
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

__global__ void 
__launch_bounds__(blockSize, 1)
compute_V(double* __restrict__ v,
          double* __restrict__ gE,
          double* __restrict__ gI,
          double* __restrict__ hE,
          double* __restrict__ hI,
          double* __restrict__ preMat,
          double* __restrict__ inputRateE,
          double* __restrict__ inputRateI,
          int* __restrict__ eventRateE,
          int* __restrict__ eventRateI,
          double* __restrict__ spikeTrain,
          unsigned int* __restrict__ nSpike,
          double* __restrict__ tBack,
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
    __shared__ double tempSpike[blockSize];
    __shared__ unsigned int spid[warpSize];
    __shared__ unsigned int spikeCount[blockSize];
    unsigned int id = threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    #ifdef IMPLICIT
        impl_rk2 lif(v[id], tBack[id]);
    #else
        rk2 lif(v[id], tBack[id]);
    #endif
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

    // init cond E 
	double fE_local[ngTypeE];
	double gE_t = 0.0f;
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
    //  cond I 
	double fI_local[ngTypeI];
	double gI_t = 0.0f;
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
	lif.set_p0(gE_t, gI_t, gL);
    /* Get feedforward input */
    double inputTimeE[MAX_FFINPUT_PER_DT];
    double inputTimeI[MAX_FFINPUT_PER_DT];
    int nInputE=0, nInputI=0;
    #ifdef TEST_WITH_MANUAL_FFINPUT
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
        curandStateMRG32k3a localStateE;
        curandStateMRG32k3a localStateI;
        double irE = inputRateE[id];
        double irI = inputRateI[id];
        if (irE > 0) {
            localStateE = stateE[id];
            nInputE = set_input_time(inputTimeE, dt, irE, &(leftTimeRateE[id]), &(lastNegLogRandE[id]), &localStateE);
		    stateE[id] = localStateE;
        }
        if (irI > 0) {
            localStateI = stateI[id];
		    nInputI = set_input_time(inputTimeI, dt, irI, &(leftTimeRateI[id]), &(lastNegLogRandI[id]), &localStateI);
		    stateI[id] = localStateI;
        }
    #endif
    // return a realization of Poisson input rate
	#ifndef FULL_SPEED
		eventRateE[id] = nInputE;
		eventRateI[id] = nInputI;
	#endif
    // set conductances
    prep_cond(&lif, condE, gE_local, hE_local, fE_local, inputTimeE, nInputE, condI, gI_local, hI_local, fI_local, inputTimeI, nInputI, gL, dt); 
    spikeTrain[id] = dt;
    // initial spike guess
    #ifdef DEBUG
        double old_v0 = lif.v0;
        double old_tBack = lif.tBack;
    #endif
    initial(&lif, dt);
    #ifdef DEBUG
    	if (lif.tsp < dt) {
    		printf("first %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp);	
    	}
    #endif
	assert(lif.tsp > 0);
    // spike-spike correction
	//__syncthreads();
	find_min(tempSpike, lif.tsp, spid);
	double t0 = 0.0;
    double t_hlf = tempSpike[0];
    unsigned int imin = spid[0];
	__syncthreads();
    #ifdef DEBUG
    	if (id == 0) {
    		if (t_hlf == dt) {
    			printf("first_ no spike\n");
    		} else {
    			printf("first_ %u: %e < %e ?\n", imin, t_hlf, dt);
    		}
    	}
    #endif
    int iInputE = 0, iInputI = 0;
    int jInputE, jInputI;
    while (t_hlf < dt) {
        // t0 ------- min ---t_hlf
        lif.tsp = dt;
        /************ This may be optimized to be per warp decision **************/
        //unsigned int MASK = __ballot_sync(FULL_MASK, lif.correctMe);
        if (lif.correctMe) {
            double new_dt = t_hlf - t0;
            // prep inputTime
            jInputE = iInputE;
            if (jInputE < nInputE) {
                while (inputTimeE[jInputE] < t_hlf) {
                    jInputE++;
                    if (jInputE == nInputE) break;
                }
            }
            jInputI = iInputI;
            if (jInputI < nInputI) {
                while (inputTimeI[jInputI] < t_hlf) {
                    jInputI++;
                    if (jInputI == nInputI) break;
                }
            }
            // prep retracable conductance
            prep_cond(&lif, condE, gE_retrace, hE_retrace, fE_local, &inputTimeE[iInputE], jInputE-iInputE, condI, gI_retrace, hI_retrace, hI_local, &inputTimeI[iInputI], jInputI-iInputI, gL, new_dt); 
            // commit for next ext. inputs.
            iInputE = jInputE;
            iInputI = jInputI;
            // get tsp decided
            #ifdef DEBUG
			    double old_v0 = lif.v0;
			    double old_tBack = lif.tBack;
            #endif
            unsigned int old_count = lif.spikeCount;
            step(&lif, t0, t_hlf, tRef);
            if (id == imin && lif.tsp == dt) {
                lif.spikeCount++;
                lif.tsp = t_hlf;
                lif.tBack = t_hlf + tRef;
            }
            if (lif.tsp < dt) {
                spikeTrain[id] = lif.tsp;
                spikeCount[id] = lif.spikeCount - old_count;
                //lif.tsp = t_hlf;
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
        // commit the spikes
        #ifdef DEBUG
            int counter = 0;
        #endif
        #pragma unroll
        for (unsigned int i=0; i<blockSize; i++) {
            double tsp = tempSpike[i];
            if (tsp < dt) {
                //tsp = t_hlf;
                double strength = preMat[i*networkSize + id] * spikeCount[i];
                double dtsp = t_hlf-tsp;
                #ifdef DEBUG
                    counter++;
                    if (id==0) {
			        	printf("%u: %e\n", i, tsp);
                    }
                #endif
                if (i < nE) {
                    #pragma unroll
			    	for (unsigned int ig=0; ig<ngTypeE; ig++) {
                        if (dtsp == 0) {
                            hE_retrace[ig] += strength;
                        } else {
                            condE.compute_single_input_conductance(&gE_retrace[ig], &hE_retrace[ig], strength, dtsp, ig);
                        }
                        condE.compute_single_input_conductance(&gE_local[ig], &hE_local[ig], strength, dt-tsp, ig);
                    }
			    } else {
			    	#pragma unroll
			    	for (unsigned int ig=0; ig<ngTypeI; ig++) {
                        if (dtsp == 0) {
                            hI_retrace[ig] += strength;
                        } else {
			    		    condI.compute_single_input_conductance(&gI_retrace[ig], &hI_retrace[ig], strength, dtsp, ig);
                        }
			    		condI.compute_single_input_conductance(&gI_local[ig], &hI_local[ig], strength, dt-tsp, ig);
			    	}
                }
            }
        }
		__syncthreads();
        #ifdef DEBUG
            if (id == 0) {
                printf("t0-t_hlf: %i spikes\n", counter);
            }
        #endif
        // t_hlf ------------- dt

        lif.tsp = dt;
        if (lif.correctMe) {
            set_p(&lif, gE_retrace, gI_retrace, gE_local, gI_local, gL);                                        
            lif.v0 = lif.v;
            //get putative tsp
            #ifdef DEBUG
			    double old_v0 = lif.v0;
			    double old_tBack = lif.tBack;
            #endif
            dab(&lif, t_hlf, dt);
            #ifdef DEBUG
			    if (lif.tsp < dt) {
		            printf("end %u: v0 = %e, v = %e->%e, tBack %e->%e tsp %e > %e\n", id, old_v0, lif.v0, lif.v, old_tBack, lif.tBack, lif.tsp, t_hlf); 
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

__global__ void 
__launch_bounds__(blockSize, 1)
compute_V_without_ssc(double* __restrict__ v,
                      double* __restrict__ gE,
                      double* __restrict__ gI,
                      double* __restrict__ hE,
                      double* __restrict__ hI,
                      double* __restrict__ preMat,
                      double* __restrict__ inputRateE,
                      double* __restrict__ inputRateI,
                      int* __restrict__ eventRateE,
                      int* __restrict__ eventRateI,
                      double* __restrict__ spikeTrain,
                      unsigned int* __restrict__ nSpike,
                      double* __restrict__ tBack,
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
    __shared__ double spike[blockSize];
    __shared__ unsigned int nsp[blockSize];
    unsigned int id = threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    #ifdef IMPL
        impl_rk2 lif(v[id], tBack[id]);
    #else
        rk2 lif(v[id], tBack[id]);
    #endif
    double gL, tRef;
    if (id < nE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    double gE_local[ngTypeE];
    double hE_local[ngTypeE];
    double gI_local[ngTypeI];
    double hI_local[ngTypeI];
    /* set a0 b0 for the first step */
    double gI_t;
    double gE_t;
    // init cond E 
    unsigned int gid;
    gE_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeE; ig++) {
        gid = networkSize*ig + id;
        gE_local[ig] = gE[gid];
        hE_local[ig] = hE[gid];
        gE_t += gE_local[ig];
    }
    //  cond I 
    gI_t = 0.0f;
    #pragma unroll
    for (unsigned int ig=0; ig<ngTypeI; ig++) {
        gid = networkSize*ig + id;
        gI_local[ig] = gI[gid];
        hI_local[ig] = hI[gid];
        gI_t += gI_local[ig];
    }
    lif.set_p0(gE_t, gI_t, gL);
    /* Get feedforward input */
    // consider use shared memory for dynamic allocation
    double inputTimeE[MAX_FFINPUT_PER_DT];
    double inputTimeI[MAX_FFINPUT_PER_DT];
    int nInputE=0, nInputI=0;
    #ifdef TEST_WITH_MANUAL_FFINPUT
        if (leftTimeRateE[id] < dt) {
            inputTimeE[nInputE] = leftTimeRateE[id];
            nInputE++;
            double tmp = leftTimeRateE[id] + dInputE;
            while (tmp < dt) {
                inputTimeE[nInputE] = tmp;
                nInputE++;
                tmp += dInputE;
            }
            leftTimeRateE[id] = tmp - dt;
        } else {
            leftTimeRateE[id] -= dt;
        }

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
        curandStateMRG32k3a localStateE;
        curandStateMRG32k3a localStateI;
        double irE = inputRateE[id];
        double irI = inputRateI[id];
        if (irE > 0) {
            localStateE = stateE[id];
            nInputE = set_input_time(inputTimeE, dt, irE, &(leftTimeRateE[id]), &(lastNegLogRandE[id]), &localStateE);
		    stateE[id] = localStateE;
        }
        if (irI > 0) {
            localStateI = stateI[id];
		    nInputI = set_input_time(inputTimeI, dt, irI, &(leftTimeRateI[id]), &(lastNegLogRandI[id]), &localStateI);
		    stateI[id] = localStateI;
        }
    #endif
    //__syncwarp();
    // return a realization of Poisson input rate
    #ifndef FULL_SPEED
        eventRateE[id] = nInputE;
        eventRateI[id] = nInputI;
    #endif
    /* evolve g to t+dt with ff input only */
    gE_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeE; ig++) {
        gid = networkSize*ig + id;
        evolve_g(condE, &gE_local[ig], &hE_local[ig], &fE[gid], inputTimeE, nInputE, dt, ig);
        gE_t += gE_local[ig];
    }
    gI_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeI; ig++) {
        gid = networkSize*ig + id;
        evolve_g(condI, &gI_local[ig], &hI_local[ig], &fI[gid], inputTimeI, nInputI, dt, ig);
        gI_t += gI_local[ig];
    }
    lif.set_p1(gE_t, gI_t, gL);

    // implicit rk2 step
    one(&lif, dt, tRef, id, gE_t, gI_t);
    __syncthreads();
	assert(lif.v <= vT);
    assert(lif.tsp > 0);
    __syncwarp();

    // write data to global
    spikeTrain[id] = lif.tsp;
    nSpike[id] = lif.spikeCount;
    tBack[id] = lif.tBack;
	v[id] = lif.v;
    spike[id] = lif.tsp;
    nsp[id] = lif.spikeCount;
    __syncthreads();

    // recalibrate conductance from cortical spikes
    #pragma unroll
    for (unsigned int i=0; i<blockSize; i++) {
        double spikeCount = nsp[i];
        if (spikeCount > 0) {
            double strength = preMat[i*networkSize + id] * spikeCount;
            if (i < nE) {
                #pragma unroll
                for (unsigned int ig=0; ig<ngTypeE; ig++) {
                    condE.compute_single_input_conductance(&gE_local[ig], &hE_local[ig], strength, dt-spike[i], ig);
                }
            } else {
                #pragma unroll
                for (unsigned int ig=0; ig<ngTypeI; ig++) {
                    condI.compute_single_input_conductance(&gI_local[ig], &hI_local[ig], strength, dt-spike[i], ig);
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
