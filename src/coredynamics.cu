#include "coredynamics.cuh"
//TODO: gap junction and learning in cortex, synaptic depression
//TODO: synaptic failure, noise
extern surface<void, cudaSurfaceType2DLayered> LGNspikeSurface;

__launch_bounds__(1024,2)
__global__ 
void rand_spInit(Float* __restrict__ tBack,
                 Float* __restrict__ spikeTrain,
                 Float* __restrict__ v,
                 Float* __restrict__ w,
                 Size* __restrict__ nLGNperV1,
                 Float* __restrict__ sp0,
                 Size* __restrict__ typeAcc,
                 Float* __restrict__ vR,
                 Float* __restrict__ gL,
                 Float* __restrict__ tRef_type,
                 Float* __restrict__ tau_w,
                 Float* __restrict__ a,
                 Float* __restrict__ b,
                 curandStateMRG32k3a* __restrict__ rGenCond,
                 curandStateMRG32k3a* __restrict__ rNoisy,
                 PosIntL seed, Size networkSize, Size nType, Size SCsplit, Size trainDepth, Float dt, bool iModel) 
{
    PosIntL id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < networkSize) {
        curandStateMRG32k3a localState = rGenCond[id];
        curandStateMRG32k3a state = rNoisy[id];
        Size iLGN = nLGNperV1[id];
        Size type;
        for (PosInt i=0; i<nType; i++) {
            if (id%blockSize < typeAcc[i]) {
                type = i;
                break;
            }
        }

        curand_init(seed + id, 0, 0, &localState);
        curand_init(seed + networkSize + id, 0, 0, &state);
        Float rand = uniform(&localState);
        Float chance;
        Float ref = 0.0;
        if (iLGN > SCsplit) {
            chance = sp0[type*2 + 0]; 
        } else {
            chance = sp0[type*2 + 1]; 
        }

        Float tRef = tRef_type[type];
        if (chance > 0) {
            if (rand < chance) {
                Float tsp = uniform(&localState);
                spikeTrain[0*networkSize + id] = 1.0 + tsp;
				Float tb = tRef - (1-tsp)*dt;
				if (tb > 0) {
                	tBack[id] = tb;
				}
				Float v0 = vR[type];
                v[id] = v0;
				if (iModel == 1) {
					Float A = a[type]*(v0-vL) * tau_w[type];
					Float w0 = w[id] + b[type];
					w[id] = (w0 - A) * exponential(-dt*(1-tsp)/tau_w[type]) + A;
				}
                ref = (1-tsp)*dt + tRef - dt;
            }
        }
        rNoisy[id] = state;
        for (PosInt i=trainDepth-1; i>0; i--) {
            if (ref < dt) {
                if (ref < 0) ref = 0;
                if (uniform(&localState) < chance*(dt-ref)/dt) {
                    Float tsp = uniform(&localState)*(dt-ref)/dt;
                    spikeTrain[i*networkSize + id] = 1.0  + tsp;
                    ref = tRef + (1-tsp)*dt;
                    assert(tsp >= 0);
                } 
            }
            ref -= dt;
        }
        rGenCond[id] = localState;
    }
}

__launch_bounds__(1024,2)
__global__
void logRand_init(Float* __restrict__ logRand,
                  Float* __restrict__ lTR,
                  int* __restrict__ LGN_idx,
                  int* __restrict__ LGN_idy,
                  curandStateMRG32k3a *state,
                  PosIntL seed, Size n, Size nFF)
{
    Size id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		curandStateMRG32k3a localState = state[id];
		curand_init(seed + id, 0, 0, &localState);
		Float rand = uniform(&localState);
		logRand[id] = -log(uniform(&localState));
		state[id] = localState;
		lTR[id] = 0;
        int x = LGN_idx[id];
        int y = LGN_idy[id];
        Float value = 0; // this is needed, otherwise surf2DLayeredwrite will raise runtime error
        surf2DLayeredwrite(value, LGNspikeSurface, 4*x, y, 0);
        #pragma unroll sum_nLearnTypeFF
        for (int i=0; i<nFF; i++) {
            surf2DLayeredwrite(value, LGNspikeSurface, 4*x, y, 1+3*i+0);
            surf2DLayeredwrite(value, LGNspikeSurface, 4*x, y, 1+3*i+1);
            surf2DLayeredwrite(value, LGNspikeSurface, 4*x, y, 1+3*i+2);
        }
	}
}


//TODO: distant connection learning
void recal_G_vec(
        std::vector<std::vector<std::vector<Float>>> &spikeTrain, std::vector<std::vector<Size>> &trainDepth, std::vector<std::vector<PosInt>> &currentTimeSlot,
        std::vector<Size> &nVec,  std::vector<std::vector<PosInt>> &vecID, std::vector<std::vector<Float>> &conVec, std::vector<std::vector<Float>> &delayVec,
        Float gE[], Float gI[], Float hE[], Float hI[], Float pE[], Float pI[], Size typeAcc[],
        std::default_random_engine *rGenCond, Float synFail[],
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt block_offset, Size nType, Size nE, Size nV1, Float speedOfThought, Size chunkSize) 
{
    Float ipE[max_ngTypeE];
    Float ipI[max_ngTypeI];
    Float local_gE[max_ngTypeE];
    Float local_hE[max_ngTypeE];
    Float local_gI[max_ngTypeI];
    Float local_hI[max_ngTypeI];
    PosInt i0 = block_offset*blockSize;
    std::normal_distribution<Float> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<Float> uniform_dist(0.0, 1.0);
    for (PosInt i=0; i<chunkSize*blockSize; i++) {
        // initialize
        PosInt itype;
        #pragma unroll max_nType
        for (PosInt j=0; j<nType; j++) {
            if (i%blockSize < typeAcc[j]) {
                itype = j;
                break;
            }
        }
        if (nVec[i] == 0) continue;
        #pragma unroll max_ngTypeE
        for (PosInt ig=0; ig<ngTypeE; ig++) {
            local_gE[ig] = 0.0f;
            local_hE[ig] = 0.0f;
            ipE[ig] = pE[itype*ngTypeE + ig];
        }
        #pragma unroll max_ngTypeI
        for (PosInt ig=0; ig<ngTypeI; ig++) {
            local_gI[ig] = 0.0f;
            local_hI[ig] = 0.0f;
            ipI[ig] = pI[itype*ngTypeI + ig];
        }
        #pragma unroll 4
        for (PosInt j = 0; j < nVec[i0+i]; j++) {
            PosInt ipre = vecID[i0+i][j];
            PosInt tid = ipre%blockSize; 
            Float strength = conVec[i0+i][j];
            Float time2post = delayVec[i0+i][j]/speedOfThought;
            Float *local_g;
            Float *local_h;
            Float *ip;
			PosInt jtype;
    		#pragma unroll (max_nType)
    		for (PosInt k=0; k<nType; k++) {
    		    if (tid < typeAcc[k]) {
    		        jtype = k;
    		        break;
    		    }
    		}
            Float p = synFail[jtype*nType + itype];
            Size ngType;
            ConductanceShape *cond;
            // TODO direct output to g and h (local memory vs register)
            if (tid < nE) {
                local_g = local_gE;
                local_h = local_hE;
                ngType = ngTypeE;
                cond = &condE;
                ip = ipE;
            } else {
                local_g = local_gI;
                local_h = local_hI;
                ngType = ngTypeI;
                cond = &condI;
                ip = ipI;
            }
            PosInt it2post = static_cast<PosInt>(ceiling(time2post/dt));
            time2post = it2post*dt - time2post;
            assert(time2post>=0);
            assert(time2post<dt);
            PosInt k0 = currentTimeSlot[i0+i][j] - it2post + trainDepth[i0+i][j];
            currentTimeSlot[i0+i][j] = (currentTimeSlot[i0+i][j]+1)%trainDepth[i0+i][j];
            #pragma unroll 2
            for (PosInt k = 0; k < 2; k++) {
                Float dtsp = spikeTrain[i0+i][j][k0+k];
                Size nsp = flooring(dtsp);
                if (nsp > 0) {
                    dtsp = (dtsp - nsp + k)*dt - time2post;
                    if (dtsp < dt && dtsp >= 0) {
						Float f = strength;
						if (p> 0) {
							f *= (1-p);
							Float rand = normal_dist(rGenCond[i0 + i]);
							f += square_root(p*f) * rand;
						}
						if (f > 0) {
							dtsp = dt - dtsp;
            				#pragma unroll max_ngType
            				for (PosInt ig=0; ig<ngType; ig++) {
                            	cond->compute_single_input_conductance(local_g[ig], local_h[ig], nsp*f*ip[ig], dtsp, ig);
							}
                        }
                    } 
                }
            }
        }
        // output
        #pragma unroll max_ngTypeE
        for (PosInt ig=0; ig<ngTypeE; ig++) {
            PosInt gid = ig*chunkSize*blockSize + i;
            gE[gid] = local_gE[ig];
            hE[gid] = local_hE[ig];
        }
        #pragma unroll max_ngTypeI
        for (PosInt ig=0; ig<ngTypeI; ig++) {
            PosInt gid = ig*chunkSize*blockSize + i;
            gI[gid] = local_gI[ig];
            hI[gid] = local_hI[ig];
        }
    }
}

//template<int ntimesFF, int ntimesE, int ntimesI>
__launch_bounds__(1024,1)
__global__ 
void compute_V_collect_spike_learnFF(
        Float* __restrict__ v,
        Float* __restrict__ dep,
        Float* __restrict__ w,
        Float* __restrict__ gFF, // not in chunks
        Float* __restrict__ hFF,
        Float** __restrict__ gE, // in chunks
        Float** __restrict__ gI,
        Float** __restrict__ hE,
        Float** __restrict__ hI,
        Size* __restrict__ nLGN,
        Float* __restrict__ sLGN,
        int* __restrict__ LGN_idx,
        int* __restrict__ LGN_idy,
        Float* __restrict__ tBack,
        Float* __restrict__ spikeTrain, //         [                depth, nblock, blockSize  ]
        Float* __restrict__ vLTD_FF_E, //    post, [nLearnTypeFF_E,        nblock, nE         ]
        Float* __restrict__ vTrip_FF_E, //   post, [nLearnTypeFF_E,        nblock, nE         ]
        Float* __restrict__ vLTD_FF_I, //    post, [nLearnTypeFF_I,        nblock, nI         ]
        Float* __restrict__ vTrip_FF_I, //   post, [nLearnTypeFF_I,        nblock, nI         ]
        Float* __restrict__ vAvgE, //        post, [                       nblock, nE,       2]
        Float* __restrict__ vAvgI, //        post, [                       nblock, nI         ]
        Float* __restrict__ vLTP_E, //        pre, [nLearnTypeE,    depth, nblock, nE,       2]
        Float* __restrict__ vLTD_E, //       post, [nLearnTypeE,           nblock, nE,       2]
        Float* __restrict__ vTripE, //       post, [nLearnTypeE,           nblock, nE,       2]
        Float* __restrict__ vSTDP_QE,  //  E post, [nLearnTypeQ,           nblock, nE        2]
        Float* __restrict__ vSTDP_QI,  //   I pre, [nLearnTypeQ,    depth, nblock, nI,       2]
        Float* __restrict__ pFF,
        Float* __restrict__ vR,
        Float* __restrict__ vThres,
        Float* __restrict__ gL,
        Float* __restrict__ tRef,
        Float* __restrict__ tonicDep,
        Float* __restrict__ vT,
        Float* __restrict__ deltaT,
        Float* __restrict__ tau_w,
        Float* __restrict__ a,
        Float* __restrict__ b,
        Size* __restrict__ typeAcc,
        curandStateMRG32k3a* __restrict__ rGenCond,
        Float* __restrict__ synFailFF,
        curandStateMRG32k3a* __restrict__ rNoisy,
        Float* __restrict__ noisyDep,
        PosInt currentTimeSlot, Size trainDepth, Size max_nLGN, Size ngTypeFF, Size ngTypeE, Size ngTypeI, ConductanceShape condFF, ConductanceShape condE, ConductanceShape condI, Float dt, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1, int learning, int varSlot, Size nType,
        LearnVarShapeFF_E_pre  learnE_pre,  LearnVarShapeFF_I_pre  learnI_pre, 
        LearnVarShapeFF_E_post learnE_post, LearnVarShapeFF_I_post learnI_post, 
        LearnVarShapeE learnE, LearnVarShapeQ learnQ, int iModel)
{
	//assert(blockDim.x == blockSize);
    PosInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    PosInt iChunk;
    Size chunkSize;
    PosInt cid;
    if (blockIdx.x >= iSizeSplit*maxChunkSize) {
        iChunk = iSizeSplit + (blockIdx.x-iSizeSplit*maxChunkSize)/remainChunkSize;
        chunkSize = remainChunkSize*blockDim.x;
        cid = tid - (iSizeSplit*maxChunkSize + (iChunk-iSizeSplit)*remainChunkSize)*blockDim.x;
    } else {
        iChunk = blockIdx.x/maxChunkSize;
        chunkSize = maxChunkSize*blockDim.x;
        cid = tid - iChunk*maxChunkSize*blockDim.x;
    }

    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    // TODO: load individual gl, tref
    PosInt itype;
    #pragma unroll max_nType
    for (PosInt j=0; j<nType; j++) {
        if (threadIdx.x < typeAcc[j]) {
            itype = j;
            break;
        }
    }
    LIF* model;
    if (iModel == 0) {
        model = new LIF(v[tid], tBack[tid], vR[itype], vThres[itype], gL[itype], tRef[itype], tonicDep[tid]);
    } else {
        model = new AdEx(w[tid], tau_w[itype], a[itype], b[itype], v[tid], tBack[tid], vR[itype], vThres[itype], gL[itype], tRef[itype], vT[itype], deltaT[itype], tonicDep[tid]);
	}

    /* set a0 b0 and a1 b1 */
    // cond FF
    //#pragma unroll (MAX_NGTYPE_FF)
    Size m = nLGN[tid];
    //Size nsp_FFt = 0;
    curandStateMRG32k3a localState = rGenCond[tid];
    curandStateMRG32k3a state = rNoisy[tid];
	
	Float ge[max_ngTypeE];
	Float he[max_ngTypeE];
	Float gi[max_ngTypeI];
	Float hi[max_ngTypeI];

	Float gE_t1 = 0.0;
	Float gI_t1 = 0.0;
	if (model->spikeCount == 0) {
		//	cond E, decay only
		//		g1, initialize g0	
    	#pragma unroll (max_ngTypeE) 
    	for (PosInt ig=0; ig<ngTypeE; ig++) {
    	    PosInt gid = chunkSize*ig + cid;
    	    Float g = gE[iChunk][gid];
    	    Float h = hE[iChunk][gid];
			ge[ig] = g;
			he[ig] = h;

    	    condE.decay_conductance(g, h, dt, ig); 
    	    gE[iChunk][gid] = g;
    	    hE[iChunk][gid] = h;
			gE_t1 += g;
    	}
    	//	cond I, decay only
		//		g1, initialize g0	
    	#pragma unroll (max_ngTypeI)
    	for (PosInt ig=0; ig<ngTypeI; ig++) {
    	    PosInt gid = chunkSize*ig + cid;
    	    Float g = gI[iChunk][gid];
    	    Float h = hI[iChunk][gid];
			gi[ig] = g;
			hi[ig] = h;

    	    condI.decay_conductance(g, h, dt, ig); 
    		gI[iChunk][gid] = g;
    		hI[iChunk][gid] = h;
			gI_t1 += g;
    	}
	}


	Float g0[max_ngTypeFF];
	Float h0[max_ngTypeFF];
	Float g1[max_ngTypeFF];
	Float h1[max_ngTypeFF];
    #pragma unroll (max_ngTypeFF) //(ntimesFF)
    for (PosInt ig=0; ig<ngTypeFF; ig++) {
        PosInt gid = nV1*ig + tid; // not in chunks
        g1[ig] = gFF[gid]; // g1, all input and decay
        h1[ig] = hFF[gid];
        g0[ig] = g1[ig]; // only before tBack
        h0[ig] = h1[ig];

		//	decay
    	condFF.decay_conductance(g1[ig], h1[ig], dt, ig); //  decayed from new_t0 to tBack
	}
	/* debug for snapshot
		if (tid == 0) {
			printf("V1: rand0 = %f, rand1 = %f, gFF = %f, tBack = %f, v = %f, , gE = %f, gI =%f\n", uniform(&state), uniform(&localState), g1[0] + g1[1], model->tBack, model->v0, gE_t1, gI_t1);
		}
		__syncthreads();
	*/

	Float p = synFailFF[itype];

	bool backingUpFromRef = model->tBack < dt && model->tBack > 0;
	Float new_t0 = 0;
	Float sInfo = 0;
	Float *f = new Float[m];
	Float noise = normal(&state)*noisyDep[itype];
	dep[tid] = model->depC*(1+noise/square_root(dt));
	do {
		Float dtBack;
		if (backingUpFromRef) {
			dtBack = model->tBack - new_t0;
		}
		//	condFF
		//		decay part
    	#pragma unroll (max_ngTypeFF) //(ntimesFF)
    	for (PosInt ig=0; ig<ngTypeFF; ig++) {
			if (backingUpFromRef) { // collect conductance at tBack
    		    condFF.decay_conductance(g0[ig], h0[ig], dtBack, ig); //  decayed from new_t0 to tBack
			}
		}
		//		input part
    	#pragma unroll (4)
    	for (PosInt i = 0; i<m; i++) {
    	    PosInt lid = tid*max_nLGN + i;
			if (model->spikeCount == 0) {
				f[i] = sLGN[lid]*(1-p);
			}
    	    int x = LGN_idx[lid];
    	    int y = LGN_idy[lid];
    	    Float sInfo;
    	    surf2DLayeredread(&sInfo, LGNspikeSurface, 4*x, y, 0);
    	    Float nsp = flooring(sInfo); // integer part: #spikes
    	    Float tsp = (sInfo - nsp)*dt; // decimal part: normalized mean tsp
			
			if (nsp > 0) {
				if (tsp >= new_t0) {
					if (model->spikeCount == 0 && p > 0) {
						Float rand = normal(&localState);
						f[i] += square_root(p * f[i])*rand;
					}
					if (f[i] > 0) {
						Float ddt;
						if (backingUpFromRef) {
							ddt = model->tBack - tsp;
						}
    					#pragma unroll (max_ngTypeFF) //(ntimesFF)
    					for (PosInt ig=0; ig<ngTypeFF; ig++) {
    			    		Float str = f[i] * pFF[itype*ngTypeFF + ig];
							if (backingUpFromRef) {
								if (ddt > 0) { // before tBack
    			    				condFF.compute_single_input_conductance(g0[ig], h0[ig], str*nsp, ddt, ig);
    			    			}
							}
							if (model->spikeCount == 0) { // all inputs
    			    			condFF.compute_single_input_conductance(g1[ig], h1[ig], str*nsp, dt*(1-tsp), ig);
							}
						}
					}
				}
			}
    	}
		if (model->spikeCount == 0) {
    		rGenCond[tid] = localState;
		}
		//	collect
		Float gE_t0 = 0.0;
    	#pragma unroll (max_ngTypeFF) //(ntimesFF)
    	for (PosInt ig=0; ig<ngTypeFF; ig++) {
        	PosInt gid = nV1*ig + tid;
    	    gE_t0 += g0[ig];
			if (model->spikeCount == 0) {
    	    	gE_t1 += g1[ig];
    			gFF[gid] = g1[ig];
    			hFF[gid] = h1[ig];
			}
    	}
    	
		//	condE, g0
    	#pragma unroll (max_ngTypeE)
    	for (PosInt ig=0; ig<ngTypeE; ig++) {
			if (backingUpFromRef) {
    	    	condE.decay_conductance(ge[ig], he[ig], dtBack, ig); 
			}
    	    gE_t0 += ge[ig];
		}

		//	condI, g0
    	Float gI_t0 = 0.0;
    	#pragma unroll (max_ngTypeI)
    	for (PosInt ig=0; ig<ngTypeI; ig++) {
			if (backingUpFromRef) {
    	    	condI.decay_conductance(gi[ig], hi[ig], dtBack, ig); 
			}
    	    gI_t0 += gi[ig];
		}

    	// stepping
		model->tsp = 0;
		if (model->tBack < dt) {
    		model->set_p0(gE_t0, gI_t0);
			if (model->spikeCount == 0) {
    			model->set_p1(gE_t1, gI_t1);
			}

			Float new_dt = dt - model->tBack;
			if (backingUpFromRef) { //	stepping other variable before tBack
				model->rk2_vFixedBefore(dtBack);
			} 
			model->rk2(new_dt, noise);

			// check spiking
    	    if (model->v > model->vThres) { // forbids firing exactly at the end of the timestep, 
    	        // crossed threshold
				new_t0 = model->tBack;
    	        model->compute_spike_time(new_dt, new_t0); 
                if (model->tsp < 0) {
                    printf("%u: v:%f -> %f; new_dt = %f, new_t0 = %f\n", tid, model->v0, model->v, new_dt, new_t0);
                    assert(model->tsp >= 0);
                }
    	        sInfo += model->tsp;
    	        model->spikeCount++;
    	        model->tBack = model->tsp + model->tRef;
				backingUpFromRef = model->tBack < dt;
				if (backingUpFromRef) {
					model->reset0();
				}
    	    } else {
				if (model->tBack > 0) model->tBack = 0;
				backingUpFromRef = false;
			}
		} 
		if (model->tBack >= dt) { // tRef till end
			model->reset1();
			model->rk2_vFixedAfter(dt-model->tsp);
			model->tBack -= dt;
		}
    	/* evolve g to t+dt with ff input only */

		// debug
		if (isnan(model->v)) {
			printf("v[%u] = nan, v0 = %f, tBack = %f\n", tid, model->v0, model->tBack);
			assert(!isnan(model->v0));
		}
	
	} while (backingUpFromRef);
	delete []f;
	rNoisy[tid] = state;

    if (model->spikeCount > 0) {
		sInfo /= model->spikeCount*dt; //decimal part: tsp (normalize by dt)
		//model->tBack -= dt;
	}
    assert(model->tBack >= 0);
	//if (model->tBack < 0) model->tBack = 0;
    sInfo += model->spikeCount; // integer part: nsp
    spikeTrain[nV1*currentTimeSlot + tid] = sInfo;
    assert(sInfo >= 0);

	/* debug
    	if (isnan(sInfo) || tid == 16737) {
    	    Size nsp = flooring(sInfo);
    	    printf("%u(%u): spiked at sInfo: %f, %u + %f, gFF[0] = %f, gFF[1] = %f, gE[0] = %f, gE[1] = %f, gE_t = %f, gI_t = %f\n", tid, cid, sInfo, nsp, sInfo - nsp, gFF[tid], gFF[tid+nV1], gE[iChunk][cid], gE[iChunk][cid + chunkSize], gE_t1, gI_t1);
    	    assert(!isnan(sInfo));
    	}
    	if (sInfo > 0 && (threadIdx.x == 0 || threadIdx.x == 768)) {
    	if (sInfo > 0 && (gI_t0 > 0 || threadIdx.x >= nE)) {
    	    Size nsp = flooring(sInfo);
    	    printf("%u(%u): spiked at sInfo: %u + %f, gF = %e(%u), gE = %e, gI = %e\n", tid, cid, nsp, sInfo - nsp, gFF[tid], m, gE[iChunk][cid], gI_t0);
    	}
	*/
	v[tid] = model->v;
	if (iModel == 1) {
		Float** var = new Float*[1];
		var[0] = w+tid;
		model->update(var);
		delete []var;
	}
    tBack[tid] = model->tBack;
    delete []model;

    if (learning) {
        Float nsp = flooring(sInfo);
        Float tsp = sInfo>0? sInfo - nsp: 1;
        // will compute ff learning, first row at start of time step, second row at tsp
        Float lFF[2*2*max_nLearnTypeFF]; // row 0: start, row 1: sp
        Float lAvg[2];
        // only temporary store
        Float lE[3*max_nLearnTypeE];
        Float lQ[max_nLearnTypeQ];
        // read ff (post) lVar
        PosInt eid = nE*blockIdx.x+threadIdx.x;
        if (learning < 4) { // read regardless of cortical spike 
            if (threadIdx.x < nE) {
                #pragma unroll max_nLearnTypeFF_E
                for (PosInt i=0; i<learnE_post.n; i++) {
                    lFF[2*i+0] =  vLTD_FF_E[nE*gridDim.x*i + eid];
                    lFF[2*i+1] = vTrip_FF_E[nE*gridDim.x*i + eid];
                }
                lAvg[0] = vAvgE[eid*2];
            } else {
                if (learnI_post.n) {
                    PosInt iid = nI*blockIdx.x+threadIdx.x-nE;
                    #pragma unroll max_nLearnTypeFF_I
                    for (PosInt i=0; i<learnI_post.n; i++) {
                        lFF[2*i+0] =  vLTD_FF_I[nI*gridDim.x*i + iid];
                        lFF[2*i+1] = vTrip_FF_I[nI*gridDim.x*i + iid];
                    }
                    lAvg[0] = vAvgI[iid];
                }
            }
        }
        if (nsp > 0) {
            if (learning !=3) { // E and Q are active, read cortical lVar and AvgE if previouly not read
                if (threadIdx.x < nE) {
                    // E
                    #pragma unroll max_nLearnTypeE
                    for (PosInt i=0; i<learnE.n; i++) {
                        lE[3*i+0] = vLTP_E[(nE*gridDim.x*trainDepth*i + nE*gridDim.x*currentTimeSlot + eid)*2];
                        lE[3*i+1] = vLTD_E[(nE*gridDim.x*i + eid)*2];
                        lE[3*i+2] = vTripE[(nE*gridDim.x*i + eid)*2];
                    }
                    // Q_E
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) {
                        lQ[i] = vSTDP_QE[(nE*gridDim.x*i + eid)*2];
                    }
                    if (learning == 4) { // otherwise already read
                        lAvg[0] = vAvgE[eid*2];
                    }
                } else {
                    // Q_I
                    PosInt iid = nI*(gridDim.x*currentTimeSlot + blockIdx.x) + threadIdx.x-nE;
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) {
                        lQ[i] = vSTDP_QI[(nI*gridDim.x*trainDepth*i + iid)*2];
                    }
                }
            }
            if (learning < 4) { // compute ff post vars' decay till tsp
                if (threadIdx.x < nE) {
                    #pragma unroll max_nLearnTypeFF_E
                    for (PosInt i=0; i<learnE_post.n; i++) {
                        lFF[2*max_nLearnTypeFF + 2*i+0] = lFF[2*i+0];
                        lFF[2*max_nLearnTypeFF + 2*i+1] = lFF[2*i+1];
                    }
                    #pragma unroll max_nLearnTypeFF_E
                    for (PosInt i=0; i<learnE_post.n; i++) {
                        decay(lFF[2*max_nLearnTypeFF + 2*i+0], learnE_post.tau[2*i+0], tsp);
                        decay(lFF[2*max_nLearnTypeFF + 2*i+1], learnE_post.tau[2*i+1], tsp);
                    }
                } else {
                    if (learnI_post.n) {
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt i=0; i<learnI_post.n; i++) {
                            lFF[2*max_nLearnTypeFF + 2*i+0] = lFF[2*i+0];
                            lFF[2*max_nLearnTypeFF + 2*i+1] = lFF[2*i+1];
                        }
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt i=0; i<learnI_post.n; i++) {
                            decay(lFF[2*max_nLearnTypeFF + 2*i+0], learnI_post.tau[2*i+0], tsp);
                            decay(lFF[2*max_nLearnTypeFF + 2*i+1], learnI_post.tau[2*i+1], tsp);
                        }
                        lAvg[1] = lAvg[0];
                        decay(lAvg[1], learnI_post.tau[2*learnI_post.n], tsp);
                    }
                }
            }
            if (threadIdx.x < nE) { // compute AvgE
                lAvg[1] = lAvg[0];
                decay(lAvg[1], learnE_post.tau[2*learnE_post.n], tsp);
            }
            if (learning !=3) { // compute and store lVars of E, Q and AvgE
                // compute
                if (threadIdx.x < nE) {
                    #pragma unroll max_nLearnTypeE
                    for (PosInt i=0; i<learnE.n; i++) {
                        decay(lE[3*i+0], learnE.tau[3*i+0], tsp);
                        decay(lE[3*i+1], learnE.tau[3*i+1], tsp);
                        decay(lE[3*i+2], learnE.tau[3*i+2], tsp);
                    }
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) {
                        decay(lQ[i], learnQ.tau[2*i+0], tsp); // Q_E
                    }
                } else {
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) {
                        decay(lQ[i], learnQ.tau[2*i+1], tsp); // Q_I
                    }
                }
                // store
                if (threadIdx.x < nE) {
                    #pragma unroll max_nLearnTypeE
                    for (PosInt i=0; i<learnE.n; i++) {
                         vLTP_E[(nE*gridDim.x*trainDepth*i + nE*gridDim.x*currentTimeSlot + eid)*2 + 1] = lE[3*i+0];
                         vLTD_E[(nE*gridDim.x*i + eid)*2 + 1] = lE[3*i+1];
                         vTripE[(nE*gridDim.x*i + eid)*2 + 1] = lE[3*i+2];
                    }
                    vAvgE[2*eid+1] = lAvg[1];
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) { // store to the second slot of the array
                        vSTDP_QE[(nE*gridDim.x*i + eid)*2 + 1] =  lQ[i];
                    }
                } else {
                    PosInt iid = nI*(gridDim.x*currentTimeSlot + blockIdx.x) + threadIdx.x-nE;
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) { // store to the second slot of the array
                        vSTDP_QI[(nI*gridDim.x*trainDepth*i + iid)*2 + 1] =  lQ[i];
                    }
                }
            }
        }
        // learn LGN connection and update LGN lVars
        if (learning < 4 && (threadIdx.x < nE || learnI_pre.n)) { 
            // learn
            for (PosInt i = 0; i<m; i++) {
                PosInt lid = tid*max_nLGN + i;
                Float f = sLGN[lid];
                int x = LGN_idx[lid];
                int y = LGN_idy[lid];
                Float sInfo_FF;
                surf2DLayeredread(&sInfo_FF, LGNspikeSurface, 4*x, y, 0);
                Float nsp_FF = flooring(sInfo_FF);
                Float tsp_FF = sInfo_FF > 0? sInfo_FF - nsp_FF: 1;
                if (nsp_FF > 0) { // LTD, regarless of post spike
                    PosInt cPick;
                    Float delta_t;
                    if (tsp_FF < tsp) {
                        cPick = 0; // from start
                        delta_t = tsp_FF;
                    } else {
                        cPick = 1; // from tsp
                        delta_t = tsp_FF-tsp;
                    }
                    delta_t *= dt;
                    if (threadIdx.x < nE) {
                        #pragma unroll max_nLearnTypeFF_E
                        for (PosInt j=0; j<learnE_pre.n; j++) {
                            Float A_LTD = learnE_post.A_ratio[j] * learnE_pre.tauLTP[j] * lAvg[cPick] * lAvg[cPick]/learnE_post.targetFR;
                            //Float A_LTD = learnFF_E.A_LTP[j]; TODO: alternative homeostatic design
                            /*debug
							if (tid == 0 && i == 0) {
                                printf("%u-%u, A_LTD: %e = %e*%e*%e^2/%e\n", tid, i, A_LTD, learnE_post.A_ratio[j], learnE_pre.tauLTP[j], lAvg[cPick], learnE_post.targetFR);
								printf("%u-%u, old_f: %e\n", tid, i, f);
                            }*/
                            f -= if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t) * A_LTD;
                            /*debug
							if (tid == 0 && i == 0) {
								printf("%u-%u, new_f: %e\n", tid, i, f);
								Float df = if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t) * A_LTD;
								printf("%u-%u, df %e = %e*%e\n", tid, i, df, if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t), A_LTD);
							}*/
                        }
                    } else {
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt j=0; j<learnI_pre.n; j++) {
                            Float A_LTD = learnI_post.A_ratio[j] * learnI_pre.tauLTP[j] * lAvg[cPick] * lAvg[cPick]/learnE_post.targetFR;
                            f -= if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnI_post.tau[2*j+0], delta_t) * A_LTD;
                        }
                    }
                } 
                if (nsp > 0) { // LTP, regardless of pre spike
                    PosInt fPick;
                    Float delta_t;
                    if (tsp_FF < tsp) {
                        fPick = 2;
                        delta_t = tsp-tsp_FF;
                    } else {
                        fPick = varSlot;
                        delta_t = tsp;
                    }
                    delta_t *= dt;
                    if (threadIdx.x < nE) {
                        #pragma unroll max_nLearnTypeFF_E
                        for (PosInt j=0; j<learnE_pre.n; j++) {
                            Float lFF_pre;
                            surf2DLayeredread(&lFF_pre, LGNspikeSurface, 4*x, y, 1+3*j+fPick);
                            /*debug
                            if (tid == 0 && i == 0) {
                                printf("%u-%u, LTP, old_f = %e, lFF_pre = %e\n", tid, i, f, lFF_pre);
                            }*/
                            f += if_decay(lFF_pre, learnE_pre.tauLTP[j], delta_t) * lFF[max_nLearnTypeFF*2 + 2*j+1] * learnE_post.A_LTP[j];
                            /*debug
                            if (tid == 0 && i == 0) {
                                printf("%u-%u, new_f:%e += %e*%e*%e\n", tid, i, f, if_decay(lFF_pre, learnE_pre.tauLTP[j], delta_t), lFF[max_nLearnTypeFF*2 + 2*j+1], learnE_post.A_LTP[j]);
                            }*/
                        }
                    } else {
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt j=0; j<learnI_pre.n; j++) {
                            Float lFF_pre;
                            surf2DLayeredread(&lFF_pre, LGNspikeSurface, 4*x, y, 1+3*j+fPick);
                            f += if_decay(lFF_pre, learnI_pre.tauLTP[j], delta_t) * lFF[max_nLearnTypeFF*2 + 2*j+1] * learnI_post.A_LTP[j];
                        }
                    }
                }
                if (threadIdx.x < nE) {
                   if (f < learnE_post.gmin) {
                        f = learnE_post.gmin;
                   }
                   if (f > learnE_post.gmax) {
                        f = learnE_post.gmax;
                   }
                } else {
                   if (f < learnI_post.gmin) {
                        f = learnI_post.gmin;
                   }
                   if (f > learnI_post.gmax) {
                        f = learnI_post.gmax;
                   }
                }
                sLGN[lid] = f;
            }
            // update FF vars; lAvg(E) to be updated after cortical learning if nLearnTypeE > 0
            Float delta_t = 1;
            PosInt cPick = nsp > 0? 1: 0;
            if (nsp > 0) { 
                delta_t -= tsp;
            }
            delta_t *= dt;
            if (threadIdx.x < nE) {
                #pragma unroll max_nLearnTypeFF_E
                for (PosInt i=0; i<learnE_post.n; i++) {
                    lFF[cPick*2*max_nLearnTypeFF + 2*i+0] += nsp; // LTD_E
                    lFF[cPick*2*max_nLearnTypeFF + 2*i+1] += nsp; // TripE
                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+0], learnE_post.tau[2*i+0], delta_t);
                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+1], learnE_post.tau[2*i+1], delta_t);
                }
                if (learning == 3) { // no E, only FF_E, otherwise to be used again and update in recal_G
                    lAvg[cPick] += nsp;
                    decay(lAvg[cPick], learnE_post.tau[2*learnE_post.n], delta_t);
                }
            } else {
                #pragma unroll max_nLearnTypeFF_I
                for (PosInt i=0; i<learnI_post.n; i++) {
                    lFF[cPick*2*max_nLearnTypeFF + 2*i+0] += nsp; // LTD_I
                    lFF[cPick*2*max_nLearnTypeFF + 2*i+1] += nsp; // TripI
                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+0], learnI_post.tau[2*i+0], delta_t);
                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+1], learnI_post.tau[2*i+1], delta_t);
                }
                lAvg[cPick] += nsp;
                decay(lAvg[cPick], learnI_post.tau[2*learnI_post.n], delta_t);
            }
            // store LGN lVars 
            if (threadIdx.x < nE) {
                PosInt eid = nE*blockIdx.x+threadIdx.x;
                #pragma unroll max_nLearnTypeFF_E
                #pragma unroll max_nLearnTypeFF_E
                for (PosInt i=0; i<learnE_post.n; i++) {
                    vLTD_FF_E[nE*gridDim.x*i + eid]  = lFF[cPick*2*max_nLearnTypeFF + 2*i+0];
                    vTrip_FF_E[nE*gridDim.x*i + eid] = lFF[cPick*2*max_nLearnTypeFF + 2*i+1];
                }
                if (learning == 3) { // no E, only FF_E
                    vAvgE[eid*2] = lAvg[cPick]; 
                }
            } else {
                PosInt iid = nI*blockIdx.x+threadIdx.x-nE;
                #pragma unroll max_nLearnTypeFF_I
                for (PosInt i=0; i<learnI_post.n; i++) {
                    vLTD_FF_I[nI*gridDim.x*i + iid]  = lFF[cPick*2*max_nLearnTypeFF + 2*i+0];
                    vTrip_FF_I[nI*gridDim.x*i + iid] = lFF[cPick*2*max_nLearnTypeFF + 2*i+1];
                }
                vAvgI[iid] = lAvg[cPick];
            }
        }
    }
}

//template<int ntimesE, int ntimesI>
__launch_bounds__(1024, 2)
__global__  // <<< nblock[partial], blockSize >>>
void recal_G_mat(
        Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        Float* __restrict__ conMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        Float* __restrict__ delayMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        Size* __restrict__ nNeighborBlock,
        PosInt* __restrict__ neighborBlockId,
        Float* __restrict__ gE, // [ngTypeE, nV1]
        Float* __restrict__ gI, // [ngTypeI, nV1] 
        Float* __restrict__ hE,
        Float* __restrict__ hI,
        Float* __restrict__ vAvgE, //        post, [                       nblock, nE,       2]
        Float* __restrict__ vLTP_E, //        pre, [nLearnTypeE,    depth, nblock, nE,       2]
        Float* __restrict__ vLTD_E, //       post, [nLearnTypeE,           nblock, nE,       2]
        Float* __restrict__ vTripE, //       post, [nLearnTypeE,           nblock, nE,       2]
        Float* __restrict__ vSTDP_QE,  //  E post, [nLearnTypeQ,           nblock, nE        2]
        Float* __restrict__ vSTDP_QI,  //   I pre, [nLearnTypeQ,    depth, nblock, nI,       2]
        Float* __restrict__ pE,
        Float* __restrict__ pI,
        Size* __restrict__ typeAcc,
        curandStateMRG32k3a* __restrict__ rGenCond,
        Float* __restrict__ synFail,
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt currentTimeSlot, Size trainDepth, Size nearNeighborBlock, Size nE, Size nI, Size nV1, Float speedOfThought, int learning, PosInt block_offset, Size nType,
        LearnVarShapeE lE, LearnVarShapeQ lQ, PosInt iChunk)
{
    // each thread is the post neuron that collects its presynaptic input conductances
    // initialize
    PosInt itype;
    #pragma unroll (max_nType)
    for (PosInt i=0; i<nType; i++) {
        if (threadIdx.x < typeAcc[i]) {
            itype = i;
            break;
        }
    }

    Float ipE[max_ngTypeE];
    Float local_gE[max_ngTypeE];
    Float local_hE[max_ngTypeE];
    #pragma unroll (max_ngTypeE)
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        local_gE[ig] = 0.0f;
        local_hE[ig] = 0.0f;
        ipE[ig] = pE[itype*ngTypeE + ig];
    }
    Float ipI[max_ngTypeI];
    Float local_gI[max_ngTypeI];
    Float local_hI[max_ngTypeI];
    //#pragma unroll (ntimesI)
    #pragma unroll (max_ngTypeI)
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        local_gI[ig] = 0.0f;
        local_hI[ig] = 0.0f;
        ipI[ig] = pI[itype*ngTypeI + ig];
    }
    // TODO: cortical learning
    //Float trip_post[2*max_nLearnTypeE];
    //Float LTD_post[2*max_nLearnTypeE];
    PosInt ipost = (block_offset+blockIdx.x)*blockSize + threadIdx.x;
    curandStateMRG32k3a localState = rGenCond[ipost];
    Float post_sInfo = spikeTrain[nV1*currentTimeSlot + ipost];
    Float postNsp = flooring(post_sInfo);
    Float postTsp = postNsp>0? post_sInfo - postNsp: 1;
    Float lAvgE;
    if (learning != 3) {
        if (threadIdx.x < nE) {
            PosInt cPick = postNsp>0? 1:0;
            PosInt eid = (block_offset+blockIdx.x)*nE + threadIdx.x;
            lAvgE = vAvgE[2*eid+cPick];
        }
    }

    __syncthreads();
    #pragma unroll (4)
    for (PosInt ib = 0; ib < nNeighborBlock[blockIdx.x]; ib++) {
		PosInt local_bid = blockIdx.x*nearNeighborBlock + ib;
		PosInt bid = neighborBlockId[local_bid];
        // check for old spikes
        #pragma unroll
        for (PosInt i=0; i<blockSize; i++) {
			PosInt ipre = bid*blockSize + i;
            // access each presynaptic neurons in stride
            // conMat: [nblock,nearNeighborBlock,blockDim.x,blockDim.x] last dim is the post-id: second-last pre-id
            PosIntL mid = static_cast<PosIntL>((local_bid*blockSize + i)*blockSize + threadIdx.x);
            Float strength = conMat[mid];
			PosInt jtype;	
    		#pragma unroll (max_nType)
    		for (PosInt j=0; j<nType; j++) {
    		    if (i < typeAcc[j]) {
    		        jtype = j;
    		        break;
    		    }
    		}
            if (strength != 0) {
                //Float LTP_pre[max_nLearnTypeE];
                //Float Q_pre[max_nLearnTypeQ];
            	Float p = synFail[jtype*nType + itype];
                Float time2post = delayMat[mid]/speedOfThought;
                Float *local_g;
                Float *local_h;
                Float *ip;
                Size ngType;
                ConductanceShape *cond;
                if (i < nE) {
                    local_g = local_gE;
                    local_h = local_hE;
                    ngType = ngTypeE;
                    cond = &condE;
                    ip = ipE;
                } else {
                    local_g = local_gI;
                    local_h = local_hI;
                    ngType = ngTypeI;
                    cond = &condI;
                    ip = ipI;
                }
                PosInt it2post = static_cast<PosInt>(ceiling(time2post/dt));
                time2post = it2post*dt - time2post;
				/* debug
                	if (time2post < 0) {
                	    printf("time2post = distance/speed = %1.3e/%1.3e = %1.3e, it2post*dt, %1.3e*%1.3e = %1.3e\n", delayMat[mid], speedOfThought, delayMat[mid]/speedOfThought, it2post, dt, it2post*dt);
                	    assert(time2post>=0);
                		assert(time2post<dt);
                	}
				*/
                PosInt j0 = currentTimeSlot - it2post + trainDepth;
                //|<-   it2post               ->|
                //|j0                           |currentTimeSlot
                //|--*--o---|o-*------|---------|---------| thus 2
                //   | tsp  tsp|               
                // ->|         |<- distance adjusted dt
                // ->| distance/speedOfThought  |<-
                //|  |<- time2post
                #pragma unroll 2
                for (PosInt j=0; j<2; j++) { 
                	PosInt isp = nV1*((j0 + j)%trainDepth) + ipre;
                	Float dtsp = spikeTrain[isp];
                	Size nsp = flooring(dtsp);
					if (nsp > 0)  {
                	    dtsp = (dtsp - nsp + j)*dt - time2post;
                		if (dtsp < dt && dtsp >= 0) {
							Float f = strength; 
							if (p > 0) {
								f *= (1-p);
								Float rand = normal(&localState);
								f += square_root(p*f) * rand;
							}
							if (f > 0) {
								dtsp = dt - dtsp;
                				#pragma unroll (max_ngType)
                				for (PosInt ig=0; ig<ngType; ig++) {
                				    cond->compute_single_input_conductance(local_g[ig], local_h[ig], nsp*f*ip[ig], dtsp, ig);
                				}
							}
						}
					}
                }
            }
            __syncwarp(); // may not be needed
        }
    }
    rGenCond[ipost] = localState;
    if (learning != 3) { // update learning variables
        if (threadIdx.x < nE) {
            PosInt eid = (block_offset+blockIdx.x)*nE + threadIdx.x;
            Float delta_t = dt;
            if (postNsp > 0) {
                delta_t = dt*(1 - postTsp);
            }
            lAvgE += postNsp;
            decay(lAvgE, lE.tau[3*lE.n], delta_t);
            vAvgE[eid*2] = lAvgE;
            /* DEBUG
            if (postNsp > 0) {
                printf("lAvgE:%e of %u, eid:%u is updated\n", lAvgE, ipost, eid);
            }*/
            /*
            #pragma unroll (max_nLearnTypeE)
            for (PosInt i=0; i<lE.n; i++) {
            }*/
        }
    }

    PosInt id = blockIdx.x*blockSize + threadIdx.x;
    //#pragma unroll (ntimesE)
    #pragma unroll (max_ngTypeE)
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        PosInt gid = ig*gridDim.x*blockDim.x + id;
        gE[gid] += local_gE[ig];
        hE[gid] += local_hE[ig];
    }
    //#pragma unroll (ntimesI)
    #pragma unroll (max_ngTypeI)
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        PosInt gid = ig*gridDim.x*blockDim.x + id;
        gI[gid] += local_gI[ig];
        hI[gid] += local_hI[ig];
    }
}

//template<int ntimesE, int ntimesI>
__launch_bounds__(1024, 2)
__global__
void sum_G(
        Size* __restrict__ nVec,
        Float* __restrict__ gEt,
        Float* __restrict__ gE,
        Float* __restrict__ gIt,
        Float* __restrict__ gI,
        Float* __restrict__ hEt,
        Float* __restrict__ hE,
        Float* __restrict__ hIt,
        Float* __restrict__ hI,
        Size ngTypeE, Size ngTypeI)
{
    PosInt id = blockIdx.x*blockDim.x + threadIdx.x;
    if (nVec[id] > 0) {
        //#pragma unroll (ntimesE)
        #pragma unroll (max_ngTypeE)
        for (PosInt ig=0; ig<ngTypeE; ig++) {
            PosInt gid = ig*gridDim.x*blockDim.x + id;
            gE[gid] += gEt[gid];
            hE[gid] += hEt[gid];
        }
        //#pragma unroll (ntimesI) 
        #pragma unroll (max_ngTypeI) 
        for (PosInt ig=0; ig<ngTypeI; ig++) {
            PosInt gid = ig*gridDim.x*blockDim.x + id;
            gI[gid] += gIt[gid];
            hI[gid] += hIt[gid];
        }
    }
    #ifdef DEBUG
        if (id == 1||id == 2) {
            printf("#%u gE[0] = %f, gE[1] = %f, gI = %f\n", id, gE[id], gE[gridDim.x*blockDim.x + id], gI[id]);
        }
    #endif
}

////template<int ntimesFF, int ntimesE, int ntimesI>
//__launch_bounds__(1024,2)
//__global__ 
//void compute_V_collect_spike_learnFF0(
//        Float* __restrict__ v,
//        Float* __restrict__ gFF, // not in chunks
//        Float* __restrict__ hFF,
//        Float** __restrict__ gE, // in chunks
//        Float** __restrict__ gI,
//        Float** __restrict__ hE,
//        Float** __restrict__ hI,
//        Size* __restrict__ nLGN,
//        Float* __restrict__ sLGN,
//        int* __restrict__ LGN_idx,
//        int* __restrict__ LGN_idy,
//        Float* __restrict__ tBack,
//        Float* __restrict__ spikeTrain, //         [                depth, nblock, blockSize  ]
//        Float* __restrict__ vLTD_FF_E, //    post, [nLearnTypeFF_E,        nblock, nE         ]
//        Float* __restrict__ vTrip_FF_E, //   post, [nLearnTypeFF_E,        nblock, nE         ]
//        Float* __restrict__ vLTD_FF_I, //    post, [nLearnTypeFF_I,        nblock, nI         ]
//        Float* __restrict__ vTrip_FF_I, //   post, [nLearnTypeFF_I,        nblock, nI         ]
//        Float* __restrict__ vAvgE, //        post, [                       nblock, nE,       2]
//        Float* __restrict__ vAvgI, //        post, [                       nblock, nI         ]
//        Float* __restrict__ vLTP_E, //        pre, [nLearnTypeE,    depth, nblock, nE,       2]
//        Float* __restrict__ vLTD_E, //       post, [nLearnTypeE,           nblock, nE,       2]
//        Float* __restrict__ vTripE, //       post, [nLearnTypeE,           nblock, nE,       2]
//        Float* __restrict__ vSTDP_QE,  //  E post, [nLearnTypeQ,           nblock, nE        2]
//        Float* __restrict__ vSTDP_QI,  //   I pre, [nLearnTypeQ,    depth, nblock, nI,       2]
//        Float* __restrict__ pFF,
//        Float* __restrict__ vR_type,
//        Float* __restrict__ vT_type,
//        Float* __restrict__ vThres_type,
//        Float* __restrict__ gL_type,
//        Float* __restrict__ tRef_type,
//        Size* __restrict__ typeAcc,
//        curandStateMRG32k3a* __restrict__ rGenCond,
//        Float* __restrict__ noisyCondFF,
//        Float* __restrict__ synFailFF,
//        PosInt currentTimeSlot, Size trainDepth, Size max_nLGN, Size ngTypeFF, Size ngTypeE, Size ngTypeI, ConductanceShape condFF, ConductanceShape condE, ConductanceShape condI, Float dt, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1, int learning, int varSlot, Size nType,
//        LearnVarShapeFF_E_pre  learnE_pre,  LearnVarShapeFF_I_pre  learnI_pre, 
//        LearnVarShapeFF_E_post learnE_post, LearnVarShapeFF_I_post learnI_post, 
//        LearnVarShapeE learnE, LearnVarShapeQ learnQ, int iModel)
//{
//	//assert(blockDim.x == blockSize);
//    PosInt tid = blockIdx.x * blockDim.x + threadIdx.x;
//    PosInt iChunk;
//    Size chunkSize;
//    PosInt cid;
//    if (blockIdx.x >= iSizeSplit*maxChunkSize) {
//        iChunk = iSizeSplit + (blockIdx.x-iSizeSplit*maxChunkSize)/remainChunkSize;
//        chunkSize = remainChunkSize*blockDim.x;
//        cid = tid - (iSizeSplit*maxChunkSize + (iChunk-iSizeSplit)*remainChunkSize)*blockDim.x;
//    } else {
//        iChunk = blockIdx.x/maxChunkSize;
//        chunkSize = maxChunkSize*blockDim.x;
//        cid = tid - iChunk*maxChunkSize*blockDim.x;
//    }
//
//    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
//    // TODO: load individual gl, tref
//    PosInt itype;
//    #pragma unroll max_nType
//    for (PosInt j=0; j<nType; j++) {
//        if (threadIdx.x < typeAcc[j]) {
//            itype = j;
//            break;
//        }
//    }
//    IF* singleNeuron;
//    if (iModel == 0) {
//        singleNeuron = new LIF(v[tid], tBack[tid], vR[itype], vT[itype], tRef[itype], gL[itype]);
//    }
//    /* set a0 b0 and a1 b1 */
//    // cond FF
//    //#pragma unroll (MAX_NGTYPE_FF)
//    Size m = nLGN[tid];
//    //Size nsp_FFt = 0;
//    curandStateMRG32k3a localState = rGenCond[tid];
//    Float gE_t0 = 0.0;
//	Float gE_t1 = 0.0;
//    #pragma unroll (max_ngTypeFF) //(ntimesFF)
//    for (PosInt ig=0; ig<ngTypeFF; ig++) {
//        PosInt gid = nV1*ig + tid; // not in chunks
//        Float g = gFF[gid];
//        Float h = hFF[gid];
//        //if (tid == 16737) {
//        //    printf("%u-%u: g:%e h:%e\n", tid, ig, g, h);
//        //}
//        gE_t0 += g;
//        // conductance of the end of the last time step
//        condFF.decay_conductance(g, h, dt, ig); //  decayed to the end of the current step
//        // Get LGN input
//
//        #pragma unroll (4)
//        for (PosInt i = 0; i<m; i++) {
//            PosInt lid = tid*max_nLGN + i;
//            Float f = sLGN[lid];
//            int x = LGN_idx[lid];
//            int y = LGN_idy[lid];
//            Float sInfo;
//            surf2DLayeredread(&sInfo, LGNspikeSurface, 4*x, y, 0);
//            Float nsp = flooring(sInfo); // integer part: #spikes
//            Float tsp = sInfo - nsp; // decimal part: normalized mean tsp
//            Float str = f * pFF[itype*ngTypeFF + ig];
//            Float g0 = 0.0;
//            if (nsp > 0 && uniform(&localState) > synFailFF[ig]) {
//                condFF.compute_single_input_conductance(g0, h, str*nsp, dt*(1-tsp), ig);
//            }
//            if (noisyCondFF[ig] > 0) {
//                Float rand = normal(&localState);
//                Float noise = noisyCondFF[ig]*str*pFF[itype*ngTypeFF + ig]*rand;
//                g0 += noise;
//                if (g0<0) g0 = 0;
//            //if (abs(noise) > str || tid == 16737) {
//            //    printf("%u-%u: noise:%e = %e * %e * %f, %e\n", tid, lid, noise, noisyCondFF[ig], str, pFF[itype*ngTypeFF+ig], rand);
//            //    assert(abs(noise) < str);
//            //}
//            //if (tid == 16737) {
//            //    printf("str:%e * noisyCondFF:%e * rand:%e = %e\n", str, noisyCondFF[ig], rand, str*noisyCondFF[ig] * rand);
//            //}
//            }
//            g += g0;
//        }
//        gE_t1 += g;
//        gFF[gid] = g;
//        hFF[gid] = h;
//    }
//    rGenCond[tid] = localState;
//    // cond E 
//    //#pragma unroll (MAX_NGTYPE_E)
//    #pragma unroll (max_ngTypeE) 
//    for (PosInt ig=0; ig<ngTypeE; ig++) {
//        PosInt gid = chunkSize*ig + cid;
//        Float g = gE[iChunk][gid];
//        Float h = hE[iChunk][gid];
//        gE_t0 += g;
//        condE.decay_conductance(g, h, dt, ig); 
//        gE_t1 += g;
//        gE[iChunk][gid] = g;
//        hE[iChunk][gid] = h;
//    }
//    // cond I 
//    Float gI_t0 = 0.0;
//	Float gI_t1 = 0.0;
//    //#pragma unroll (MAX_NGTYPE_I)
//    //#pragma unroll (ntimesI)
//    #pragma unroll (max_ngTypeI)
//    for (PosInt ig=0; ig<ngTypeI; ig++) {
//        PosInt gid = chunkSize*ig + cid;
//        Float g = gI[iChunk][gid];
//        Float h = hI[iChunk][gid];
//        gI_t0 += g;
//        condI.decay_conductance(g, h, dt, ig); 
//        gI_t1 += g;
//        gI[iChunk][gid] = g;
//        hI[iChunk][gid] = h;
//    }
//    singleNeuron->set_p0(gE_t0, gI_t0);
//    singleNeuron->set_p1(gE_t1, gI_t1);
//    /* evolve g to t+dt with ff input only */
//    // step
//    Float sInfo = step(singleNeuron, dt, /*the last 3 args are for deugging*/ tid, gE_t1, gI_t1);
//    spikeTrain[nV1*currentTimeSlot + tid] = sInfo;
//
//    //if (isnan(sInfo) || tid == 16737) {
//    //    Size nsp = flooring(sInfo);
//    //    printf("%u(%u): spiked at sInfo: %f, %u + %f, gFF[0] = %f, gFF[1] = %f, gE[0] = %f, gE[1] = %f, gE_t = %f, gI_t = %f\n", tid, cid, sInfo, nsp, sInfo - nsp, gFF[tid], gFF[tid+nV1], gE[iChunk][cid], gE[iChunk][cid + chunkSize], gE_t1, gI_t1);
//    //    assert(!isnan(sInfo));
//    //}
//    /*DEBUG
//    //if (sInfo > 0 && (threadIdx.x == 0 || threadIdx.x == 768)) {
//    if (sInfo > 0 && (gI_t0 > 0 || threadIdx.x >= nE)) {
//        Size nsp = flooring(sInfo);
//        printf("%u(%u): spiked at sInfo: %u + %f, gF = %e(%u), gE = %e, gI = %e\n", tid, cid, nsp, sInfo - nsp, gFF[tid], m, gE[iChunk][cid], gI_t0);
//    }*/
//	v[tid] = singleNeuron->v;
//    tBack[tid] = singleNeuron->tBack;
//    delete []singleNeuron;
//    if (learning) {
//        Float nsp = flooring(sInfo);
//        Float tsp = sInfo>0? sInfo - nsp: 1;
//        // will compute ff learning, first row at start of time step, second row at tsp
//        Float lFF[2*2*max_nLearnTypeFF]; // row 0: start, row 1: sp
//        Float lAvg[2];
//        // only temporary store
//        Float lE[3*max_nLearnTypeE];
//        Float lQ[max_nLearnTypeQ];
//        // read ff (post) lVar
//        PosInt eid = nE*blockIdx.x+threadIdx.x;
//        if (learning < 4) { // read regardless of cortical spike 
//            if (threadIdx.x < nE) {
//                #pragma unroll max_nLearnTypeFF_E
//                for (PosInt i=0; i<learnE_post.n; i++) {
//                    lFF[2*i+0] =  vLTD_FF_E[nE*gridDim.x*i + eid];
//                    lFF[2*i+1] = vTrip_FF_E[nE*gridDim.x*i + eid];
//                }
//                lAvg[0] = vAvgE[eid*2];
//            } else {
//                if (learnI_post.n) {
//                    PosInt iid = nI*blockIdx.x+threadIdx.x-nE;
//                    #pragma unroll max_nLearnTypeFF_I
//                    for (PosInt i=0; i<learnI_post.n; i++) {
//                        lFF[2*i+0] =  vLTD_FF_I[nI*gridDim.x*i + iid];
//                        lFF[2*i+1] = vTrip_FF_I[nI*gridDim.x*i + iid];
//                    }
//                    lAvg[0] = vAvgI[iid];
//                }
//            }
//        }
//        if (nsp > 0) {
//            if (learning !=3) { // E and Q are active, read cortical lVar and AvgE if previouly not read
//                if (threadIdx.x < nE) {
//                    // E
//                    #pragma unroll max_nLearnTypeE
//                    for (PosInt i=0; i<learnE.n; i++) {
//                        lE[3*i+0] = vLTP_E[(nE*gridDim.x*trainDepth*i + nE*gridDim.x*currentTimeSlot + eid)*2];
//                        lE[3*i+1] = vLTD_E[(nE*gridDim.x*i + eid)*2];
//                        lE[3*i+2] = vTripE[(nE*gridDim.x*i + eid)*2];
//                    }
//                    // Q_E
//                    #pragma unroll max_nLearnTypeQ
//                    for (PosInt i=0; i<learnQ.n; i++) {
//                        lQ[i] = vSTDP_QE[(nE*gridDim.x*i + eid)*2];
//                    }
//                    if (learning == 4) { // otherwise already read
//                        lAvg[0] = vAvgE[eid*2];
//                    }
//                } else {
//                    // Q_I
//                    PosInt iid = nI*(gridDim.x*currentTimeSlot + blockIdx.x) + threadIdx.x-nE;
//                    #pragma unroll max_nLearnTypeQ
//                    for (PosInt i=0; i<learnQ.n; i++) {
//                        lQ[i] = vSTDP_QI[(nI*gridDim.x*trainDepth*i + iid)*2];
//                    }
//                }
//            }
//            if (learning < 4) { // compute ff post vars' decay till tsp
//                if (threadIdx.x < nE) {
//                    #pragma unroll max_nLearnTypeFF_E
//                    for (PosInt i=0; i<learnE_post.n; i++) {
//                        lFF[2*max_nLearnTypeFF + 2*i+0] = lFF[2*i+0];
//                        lFF[2*max_nLearnTypeFF + 2*i+1] = lFF[2*i+1];
//                    }
//                    #pragma unroll max_nLearnTypeFF_E
//                    for (PosInt i=0; i<learnE_post.n; i++) {
//                        decay(lFF[2*max_nLearnTypeFF + 2*i+0], learnE_post.tau[2*i+0], tsp);
//                        decay(lFF[2*max_nLearnTypeFF + 2*i+1], learnE_post.tau[2*i+1], tsp);
//                    }
//                } else {
//                    if (learnI_post.n) {
//                        #pragma unroll max_nLearnTypeFF_I
//                        for (PosInt i=0; i<learnI_post.n; i++) {
//                            lFF[2*max_nLearnTypeFF + 2*i+0] = lFF[2*i+0];
//                            lFF[2*max_nLearnTypeFF + 2*i+1] = lFF[2*i+1];
//                        }
//                        #pragma unroll max_nLearnTypeFF_I
//                        for (PosInt i=0; i<learnI_post.n; i++) {
//                            decay(lFF[2*max_nLearnTypeFF + 2*i+0], learnI_post.tau[2*i+0], tsp);
//                            decay(lFF[2*max_nLearnTypeFF + 2*i+1], learnI_post.tau[2*i+1], tsp);
//                        }
//                        lAvg[1] = lAvg[0];
//                        decay(lAvg[1], learnI_post.tau[2*learnI_post.n], tsp);
//                    }
//                }
//            }
//            if (threadIdx.x < nE) { // compute AvgE
//                lAvg[1] = lAvg[0];
//                decay(lAvg[1], learnE_post.tau[2*learnE_post.n], tsp);
//            }
//            if (learning !=3) { // compute and store lVars of E, Q and AvgE
//                // compute
//                if (threadIdx.x < nE) {
//                    #pragma unroll max_nLearnTypeE
//                    for (PosInt i=0; i<learnE.n; i++) {
//                        decay(lE[3*i+0], learnE.tau[3*i+0], tsp);
//                        decay(lE[3*i+1], learnE.tau[3*i+1], tsp);
//                        decay(lE[3*i+2], learnE.tau[3*i+2], tsp);
//                    }
//                    #pragma unroll max_nLearnTypeQ
//                    for (PosInt i=0; i<learnQ.n; i++) {
//                        decay(lQ[i], learnQ.tau[2*i+0], tsp); // Q_E
//                    }
//                } else {
//                    #pragma unroll max_nLearnTypeQ
//                    for (PosInt i=0; i<learnQ.n; i++) {
//                        decay(lQ[i], learnQ.tau[2*i+1], tsp); // Q_I
//                    }
//                }
//                // store
//                if (threadIdx.x < nE) {
//                    #pragma unroll max_nLearnTypeE
//                    for (PosInt i=0; i<learnE.n; i++) {
//                         vLTP_E[(nE*gridDim.x*trainDepth*i + nE*gridDim.x*currentTimeSlot + eid)*2 + 1] = lE[3*i+0];
//                         vLTD_E[(nE*gridDim.x*i + eid)*2 + 1] = lE[3*i+1];
//                         vTripE[(nE*gridDim.x*i + eid)*2 + 1] = lE[3*i+2];
//                    }
//                    vAvgE[2*eid+1] = lAvg[1];
//                    #pragma unroll max_nLearnTypeQ
//                    for (PosInt i=0; i<learnQ.n; i++) { // store to the second slot of the array
//                        vSTDP_QE[(nE*gridDim.x*i + eid)*2 + 1] =  lQ[i];
//                    }
//                } else {
//                    PosInt iid = nI*(gridDim.x*currentTimeSlot + blockIdx.x) + threadIdx.x-nE;
//                    #pragma unroll max_nLearnTypeQ
//                    for (PosInt i=0; i<learnQ.n; i++) { // store to the second slot of the array
//                        vSTDP_QI[(nI*gridDim.x*trainDepth*i + iid)*2 + 1] =  lQ[i];
//                    }
//                }
//            }
//        }
//        // learn LGN connection and update LGN lVars
//        if (learning < 4 && (threadIdx.x < nE || learnI_pre.n)) { 
//            // learn
//            for (PosInt i = 0; i<m; i++) {
//                PosInt lid = tid*max_nLGN + i;
//                Float f = sLGN[lid];
//                int x = LGN_idx[lid];
//                int y = LGN_idy[lid];
//                Float sInfo_FF;
//                surf2DLayeredread(&sInfo_FF, LGNspikeSurface, 4*x, y, 0);
//                Float nsp_FF = flooring(sInfo_FF);
//                Float tsp_FF = sInfo_FF > 0? sInfo_FF - nsp_FF: 1;
//                if (nsp_FF > 0) { // LTD, regarless of post spike
//                    PosInt cPick;
//                    Float delta_t;
//                    if (tsp_FF < tsp) {
//                        cPick = 0; // from start
//                        delta_t = tsp_FF;
//                    } else {
//                        cPick = 1; // from tsp
//                        delta_t = tsp_FF-tsp;
//                    }
//                    delta_t *= dt;
//                    if (threadIdx.x < nE) {
//                        #pragma unroll max_nLearnTypeFF_E
//                        for (PosInt j=0; j<learnE_pre.n; j++) {
//                            Float A_LTD = learnE_post.A_ratio[j] * learnE_pre.tauLTP[j] * lAvg[cPick] * lAvg[cPick]/learnE_post.targetFR;
//                            //Float A_LTD = learnFF_E.A_LTP[j]; TODO: alternative homeostatic design
//                            /*debug
//							if (tid == 0 && i == 0) {
//                                printf("%u-%u, A_LTD: %e = %e*%e*%e^2/%e\n", tid, i, A_LTD, learnE_post.A_ratio[j], learnE_pre.tauLTP[j], lAvg[cPick], learnE_post.targetFR);
//								printf("%u-%u, old_f: %e\n", tid, i, f);
//                            }*/
//                            f -= if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t) * A_LTD;
//                            /*debug
//							if (tid == 0 && i == 0) {
//								printf("%u-%u, new_f: %e\n", tid, i, f);
//								Float df = if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t) * A_LTD;
//								printf("%u-%u, df %e = %e*%e\n", tid, i, df, if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t), A_LTD);
//							}*/
//                        }
//                    } else {
//                        #pragma unroll max_nLearnTypeFF_I
//                        for (PosInt j=0; j<learnI_pre.n; j++) {
//                            Float A_LTD = learnI_post.A_ratio[j] * learnI_pre.tauLTP[j] * lAvg[cPick] * lAvg[cPick]/learnE_post.targetFR;
//                            f -= if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnI_post.tau[2*j+0], delta_t) * A_LTD;
//                        }
//                    }
//                } 
//                if (nsp > 0) { // LTP, regardless of pre spike
//                    PosInt fPick;
//                    Float delta_t;
//                    if (tsp_FF < tsp) {
//                        fPick = 2;
//                        delta_t = tsp-tsp_FF;
//                    } else {
//                        fPick = varSlot;
//                        delta_t = tsp;
//                    }
//                    delta_t *= dt;
//                    if (threadIdx.x < nE) {
//                        #pragma unroll max_nLearnTypeFF_E
//                        for (PosInt j=0; j<learnE_pre.n; j++) {
//                            Float lFF_pre;
//                            surf2DLayeredread(&lFF_pre, LGNspikeSurface, 4*x, y, 1+3*j+fPick);
//                            /*debug
//                            if (tid == 0 && i == 0) {
//                                printf("%u-%u, LTP, old_f = %e, lFF_pre = %e\n", tid, i, f, lFF_pre);
//                            }*/
//                            f += if_decay(lFF_pre, learnE_pre.tauLTP[j], delta_t) * lFF[max_nLearnTypeFF*2 + 2*j+1] * learnE_post.A_LTP[j];
//                            /*debug
//                            if (tid == 0 && i == 0) {
//                                printf("%u-%u, new_f:%e += %e*%e*%e\n", tid, i, f, if_decay(lFF_pre, learnE_pre.tauLTP[j], delta_t), lFF[max_nLearnTypeFF*2 + 2*j+1], learnE_post.A_LTP[j]);
//                            }*/
//                        }
//                    } else {
//                        #pragma unroll max_nLearnTypeFF_I
//                        for (PosInt j=0; j<learnI_pre.n; j++) {
//                            Float lFF_pre;
//                            surf2DLayeredread(&lFF_pre, LGNspikeSurface, 4*x, y, 1+3*j+fPick);
//                            f += if_decay(lFF_pre, learnI_pre.tauLTP[j], delta_t) * lFF[max_nLearnTypeFF*2 + 2*j+1] * learnI_post.A_LTP[j];
//                        }
//                    }
//                }
//                if (threadIdx.x < nE) {
//                   if (f < learnE_post.gmin) {
//                        f = learnE_post.gmin;
//                   }
//                   if (f > learnE_post.gmax) {
//                        f = learnE_post.gmax;
//                   }
//                } else {
//                   if (f < learnI_post.gmin) {
//                        f = learnI_post.gmin;
//                   }
//                   if (f > learnI_post.gmax) {
//                        f = learnI_post.gmax;
//                   }
//                }
//                sLGN[lid] = f;
//            }
//            // update FF vars; lAvg(E) to be updated after cortical learning if nLearnTypeE > 0
//            Float delta_t = 1;
//            PosInt cPick = nsp > 0? 1: 0;
//            if (nsp > 0) { 
//                delta_t -= tsp;
//            }
//            delta_t *= dt;
//            if (threadIdx.x < nE) {
//                #pragma unroll max_nLearnTypeFF_E
//                for (PosInt i=0; i<learnE_post.n; i++) {
//                    lFF[cPick*2*max_nLearnTypeFF + 2*i+0] += nsp; // LTD_E
//                    lFF[cPick*2*max_nLearnTypeFF + 2*i+1] += nsp; // TripE
//                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+0], learnE_post.tau[2*i+0], delta_t);
//                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+1], learnE_post.tau[2*i+1], delta_t);
//                }
//                if (learning == 3) { // no E, only FF_E, otherwise to be used again and update in recal_G
//                    lAvg[cPick] += nsp;
//                    decay(lAvg[cPick], learnE_post.tau[2*learnE_post.n], delta_t);
//                }
//            } else {
//                #pragma unroll max_nLearnTypeFF_I
//                for (PosInt i=0; i<learnI_post.n; i++) {
//                    lFF[cPick*2*max_nLearnTypeFF + 2*i+0] += nsp; // LTD_I
//                    lFF[cPick*2*max_nLearnTypeFF + 2*i+1] += nsp; // TripI
//                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+0], learnI_post.tau[2*i+0], delta_t);
//                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+1], learnI_post.tau[2*i+1], delta_t);
//                }
//                lAvg[cPick] += nsp;
//                decay(lAvg[cPick], learnI_post.tau[2*learnI_post.n], delta_t);
//            }
//            // store LGN lVars 
//            if (threadIdx.x < nE) {
//                PosInt eid = nE*blockIdx.x+threadIdx.x;
//                #pragma unroll max_nLearnTypeFF_E
//                #pragma unroll max_nLearnTypeFF_E
//                for (PosInt i=0; i<learnE_post.n; i++) {
//                    vLTD_FF_E[nE*gridDim.x*i + eid]  = lFF[cPick*2*max_nLearnTypeFF + 2*i+0];
//                    vTrip_FF_E[nE*gridDim.x*i + eid] = lFF[cPick*2*max_nLearnTypeFF + 2*i+1];
//                }
//                if (learning == 3) { // no E, only FF_E
//                    vAvgE[eid*2] = lAvg[cPick]; 
//                }
//            } else {
//                PosInt iid = nI*blockIdx.x+threadIdx.x-nE;
//                #pragma unroll max_nLearnTypeFF_I
//                for (PosInt i=0; i<learnI_post.n; i++) {
//                    vLTD_FF_I[nI*gridDim.x*i + iid]  = lFF[cPick*2*max_nLearnTypeFF + 2*i+0];
//                    vTrip_FF_I[nI*gridDim.x*i + iid] = lFF[cPick*2*max_nLearnTypeFF + 2*i+1];
//                }
//                vAvgI[iid] = lAvg[cPick];
//            }
//        }
//    }
//}
//
//__device__
//__forceinline__
//void LIF::recompute(Float dt, Float t0) {
//    Float rB = dt/(tBack-t0) - 1; 
//    Float denorm = 2 + a1*dt;
//    Float A = (2 - a0*dt)/denorm;
//    Float B = (b0 + b1)*dt/denorm;
//    v0 = recomp_v0(A, B, rB);
//    v = A*v0 + B;
//}
//
//__device__ 
//__forceinline__
//void LIF::recompute_v(Float dt, Float t0) {
//    Float rB = dt/(tBack-t0) - 1; 
//    Float denorm = 2 + a1*dt;
//    Float A = (2 - a0*dt)/denorm;
//    Float B = (b0 + b1)*dt/denorm;
//    v = recomp_v(A, B, rB);
//}
//
//__device__ 
//__forceinline__
//void LIF::recompute_v0(Float dt, Float t0) {
//    Float rB = dt/(tBack-t0) - 1; 
//    Float denorm = 2 + a1*dt;
//    Float A = (2 - a0*dt)/denorm;
//    Float B = (b0 + b1)*dt/denorm;
//    v0 = recomp_v0(A, B, rB);
//}
//__device__
//__forceinline__
//Float step(LIF* model, Float dt, PosInt id, Float gE, Float gI) {
//    model->spikeCount = 0;
//    Float sInfo = 0.0;
//    // not in refractory period
//    if (model->tBack < dt) {
//        // return from refractory period
//        if (model->tBack > 0.0f) {
//            model->recompute_v0(dt);
//            #ifdef DEBUG
//                if (id == 0 || id == 768) {
//                    printf("backed\n");
//                }
//            #endif
//        }
//        model->rk2(dt);
//        while (model->v > model->vThres && model->tBack < dt) { // forbids firing exactly at the end of the timestep, 
//            // crossed threshold
//            model->compute_spike_time(dt); 
//            sInfo += model->tsp;
//            model->spikeCount++;
//            model->tBack = model->tsp + model->tRef;
//            #ifdef DEBUG
//                if (id == 0 || id == 768) {
//                    printf("#%u spiked at %f, to come back at %f\n", id, model->tsp, model->tBack);
//                }
//            #endif
//            if (model->tBack < dt) {
//                // refractory period ended during dt
//                model->recompute(dt);
//            }
//        }
//    }
//    if (model->tBack >= dt) {
//        // during refractory period
//        model->reset1();
//    }
//    model->tBack -= dt;
//    #ifdef DEBUG
//        if (model->v < vI) {
//    		printf("#%i implicit rk2 is A-Stable! something is off gE1 = %f, gI1 = %f, v = %f, v0 = %f, a0 = %f, b0 = %f, a1 = %f, b1 = %f\n", id, gE, gI, model->v, model->v0, model->a0, model->b0, model->a1, model->b1);
//        }   
//    #endif
//    if (model->spikeCount > 0) sInfo /= model->spikeCount*dt; //decimal part: tsp (normalize by dt)
//    sInfo += model->spikeCount; // integer part: nsp
//    #ifdef DEBUG
//        if ((sInfo > 0 && sInfo < 1) || model->spikeCount >= 2) {
//            printf("sInfo = %.3f, gE = %.3e, gI = %.3e, spikeCount = %u\n", sInfo, gE, gI, model->spikeCount);
//            assert(sInfo == 0 || (sInfo >= 1 && sInfo < 2 && model->spikeCount < 2));
//        }
//    #endif
//    __syncwarp();
//    return sInfo;
//}
