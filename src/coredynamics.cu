#include "coredynamics.cuh"
//TODO: gap junction and learning in cortex, synaptic depression
//TODO: synaptic failure, noise

__launch_bounds__(1024,2)
__global__ 
void rand_spInit(Float* __restrict__ tBack,
                 Float* __restrict__ spikeTrain,
				 PosInt* __restrict__ ipre, // [depth, nblock, nTypeHierarchy]
        		 Size* __restrict__ npre, // [depth, nblock, nTypeHierarchy]
                 Float* __restrict__ output_g,
                 Float* __restrict__ output_h,
                 Float* __restrict__ v,
                 Float* __restrict__ w,
                 Size* __restrict__ nLGNperV1,
                 Float* __restrict__ sp0,
                 Size* __restrict__ typeAcc,
                 Float* __restrict__ vR,
                 Float* __restrict__ tRef_type,
                 Float* __restrict__ tau_w,
                 Float* __restrict__ a,
                 Float* __restrict__ b,
                 curandStateMRG32k3a* __restrict__ rGenCond,
                 curandStateMRG32k3a* __restrict__ rNoisy,
                 PosIntL seed, Size networkSize, Size nType, Size SCsplit, Size trainDepth, Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, Size nE, Size nI, int noDelay, bool iModel) 
{
    __shared__ PosInt counter[2];
    if (threadIdx.x < 2) {
        counter[threadIdx.x] = 0;
    }
    __syncthreads();
    PosIntL id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = rGenCond[id];
    curandStateMRG32k3a state = rNoisy[id];
    Size iLGN = nLGNperV1[id];
    Size type;
    for (PosInt i=0; i<nType; i++) {
        if (id%blockDim.x < typeAcc[i]) {
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
	Float sInfo = v[id];
	PosInt sid;
	//PosInt oid;
    if (chance > 0) {
        if (rand < chance) {
            Float tsp = uniform(&localState);
            sInfo = 1.0 + tsp;
			Float tb = tRef - (1-tsp)*dt;
			if (tb > 0) {
            	tBack[id] = tb;
			}
			Float v0 = vR[type];
            v[id] = v0;
			if (iModel == 1) {
				Float A = a[type]*(v0 - vL) * tau_w[type];
				Float w0 = w[id] + b[type];
				w[id] = (w0 - A) * exponential(-dt*(1-tsp)/tau_w[type]) + A;
			}
            ref = (1-tsp)*dt + tRef - dt;
			if (threadIdx.x < nE) {
				sid = atomicAdd_block(&counter[0],1);
			} else {
				sid = atomicAdd_block(&counter[1],1);
			}
		}
	}
    __syncthreads();
	if (noDelay) {
		if (sInfo >= 1) {
			Float tsp = (sInfo-1)*dt;
			Size block_ngType = (ngTypeE*nE + ngTypeI*nI)*blockIdx.x;
			PosInt i = block_ngType;
			Size npreE = counter[0];
    		if (threadIdx.x < nE) {
				i += sid;
    	    	#pragma unroll (max_ngTypeE)
    	    	for (PosInt ig=0; ig<ngTypeE; ig++) {
    	    		Float local_g = 0;
    	    		Float local_h = 0;
    	    		condE.compute_single_input_conductance(local_g, local_h, 1, dt-tsp, ig);
					output_g[ig*npreE + i] = local_g;
					output_h[ig*npreE + i] = local_h;
    	    	}
    		} else {
				Size npreI = counter[1];
				i += npreE*ngTypeE + sid;
    	    	#pragma unroll (max_ngTypeI)
    	    	for (PosInt ig=0; ig<ngTypeI; ig++) {
    	    		Float local_g = 0;
    	    		Float local_h = 0;
    	    		condI.compute_single_input_conductance(local_g, local_h, 1, dt-tsp, ig);
					output_g[ig*npreI + i] = local_g;
					output_h[ig*npreI + i] = local_h;
    	    	}
    	    	sid += npreE; // go after exc
    		}
			ipre[blockIdx.x*blockDim.x + sid] = threadIdx.x;
		}
		if (threadIdx.x < 2) {
    	    npre[threadIdx.x*gridDim.x + blockIdx.x] = counter[threadIdx.x];
    	}
	}
	spikeTrain[0*networkSize + id] = sInfo;
    rNoisy[id] = state;
    for (PosInt i=trainDepth-1; i>0; i--) {
        if (ref < dt) {
            if (ref < 0) ref = 0;
            if (uniform(&localState) < chance*(dt-ref)/dt) {
                Float tsp = uniform(&localState)*(dt-ref)/dt;
                spikeTrain[i*networkSize + id] = 1.0  + tsp;
                ref = tRef + (1-tsp)*dt;
                assert(tsp >= 0);
            } else {
        		spikeTrain[i*networkSize + id] = v[id];
			}
        } else {
        	spikeTrain[i*networkSize + id] = vR[type];
		}
        ref -= dt;
    }
    rGenCond[id] = localState;
	assert(v[id] < 0); 
	//if (blockIdx.x == 0 && threadIdx.x == 960 ) {
	//	if (noDelay) {
	//		printf("#%u: v0 = %.3f, w0  = %.3f, sInfo = %.3f, og = %.3f, oh = %.3f\n", id, v[id], w[id], spikeTrain[id], output_g[oid], output_h[oid]);
	//	} else {
	//		printf("#%u: v0 = %.3f, w0  = %.3f, sInfo = %.3f\n", id, v[id], w[id], spikeTrain[id]);
	//	}
	//}
}

__launch_bounds__(1024,1)
__global__
void logRand_init(Float* __restrict__ logRand,
                  Float* __restrict__ lTR,
                  int* __restrict__ LGN_idx,
                  int* __restrict__ LGN_idy,
                  curandStateMRG32k3a *state,
                  cudaSurfaceObject_t LGNspikeSurface,
                  PosIntL seed, Size n, Size nFF)
{
    Size id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) {
		curandStateMRG32k3a localState = state[id];
		curand_init(seed + id, 0, 0, &localState);
		Float rand = uniform(&localState);
		logRand[id] = -logarithm(uniform(&localState));
		state[id] = localState;
		lTR[id] = 0.0;
        int x = LGN_idx[id];
        int y = LGN_idy[id];
        float value = 0; // this is needed, otherwise surf2DLayeredwrite will raise runtime error
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
        std::vector<std::vector<std::vector<Float>>> &spikeTrain, std::vector<std::vector<Size>> &trainDepth, std::vector<std::vector<PosInt>> &currentTimeSlot, Float og[], Float oh[],
        std::vector<Size> &nVec,  std::vector<std::vector<PosInt>> &vecID, std::vector<std::vector<Float>> &conVec, std::vector<std::vector<Float>> &delayVec,
        Float gE[], Float gI[], Float hE[], Float hI[], Float pE[], Float pI[], Size typeAcc[],
        std::default_random_engine *rGenCond, Float synFail[], Float synPerCon[],
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt block_offset, Size nType, Size nE, Size nI, Size nV1, Float speedOfThought, Size chunkSize, bool noFarDelay, PosInt it, Size neuronPerBlock) 
{
    Float ipE[max_ngTypeE];
    Float ipI[max_ngTypeI];
    Float local_gE[max_ngTypeE];
    Float local_hE[max_ngTypeE];
    Float local_gI[max_ngTypeI];
    Float local_hI[max_ngTypeI];
    PosInt i0 = block_offset*neuronPerBlock;
    std::normal_distribution<Float> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<Float> uniform_dist(0.0, 1.0);
    for (PosInt i=0; i<chunkSize*neuronPerBlock; i++) {
        // initialize
        PosInt itype;
        for (PosInt j=0; j<nType; j++) {
            if (i%neuronPerBlock< typeAcc[j]) {
                itype = j;
                break;
            }
        }
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
		if (noFarDelay) {
        	if (nVec[i0 + i] == 0) {
        		for (PosInt j = 0; j < nVec[i0+i]; j++) {
        		    PosInt ipre = vecID[i0+i][j];
        		    PosInt tid = ipre%neuronPerBlock; 
        		    Float strength = conVec[i0+i][j];
        		    Float *local_g;
        		    Float *local_h;
        		    Float *ip;
					PosInt jtype;
    				#pragma unroll
    				for (PosInt k=0; k<nType; k++) {
    				    if (tid < typeAcc[k]) {
    				        jtype = k;
    				        break;
    				    }
    				}
        		    Float p = synFail[jtype*nType + itype];
        		    Float nSyn = synPerCon[jtype*nType + itype];
        		    Size ngType;
        		    // TODO direct output to g and h (local memory vs register)
        		    if (tid < nE) {
        		        local_g = local_gE;
        		        local_h = local_hE;
        		        ngType = ngTypeE;
        		        ip = ipE;
        		    } else {
        		        local_g = local_gI;
        		        local_h = local_hI;
        		        ngType = ngTypeI;
        		        ip = ipI;
        		    }
					if (p > 0) {
						Float normed_std = square_root(p*(1-p)/nSyn);
						Float rand = normal_dist(rGenCond[i0 + i]);
						strength *= 1-p + normed_std*rand;
					}
					if (strength > 0) {
						Size bid = ipre/neuronPerBlock;
						Size block_ngType = (ngTypeE*nE + ngTypeI*nI)*bid;
						PosInt id;
        				if (tid < nE) {
        				    ngType = ngTypeE;
							id = block_ngType + tid*ngTypeE;
        				} else {
        				    ngType = ngTypeI;
							id = block_ngType + nE*ngTypeE + (tid-nE)*ngTypeI;
        				}
            	    	for (PosInt ig=0; ig<ngType; ig++) {
							Float g = og[id + ig];
							Float h = oh[id + ig];
							Float str = strength*ip[ig];
							local_g[ig] += str*g;
							local_h[ig] += str*h;
            	    	}
        		    }
        		}
			}
		} else {
        	#pragma unroll 4
        	for (PosInt j = 0; j < nVec[i0+i]; j++) {
        	    PosInt ipre = vecID[i0+i][j];
        	    PosInt tid = ipre%neuronPerBlock; 
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
        	    Float nSyn = synPerCon[jtype*nType + itype];
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
        	    int k0 = currentTimeSlot[i0+i][j] - it2post; 
				if (k0 < 0) k0 += trainDepth[i0+i][j];
        	    currentTimeSlot[i0+i][j] = (currentTimeSlot[i0+i][j]+1)%trainDepth[i0+i][j];
        	    #pragma unroll 2
        	    for (PosInt k = 0; k < 2; k++) {
        	        Float dtsp = spikeTrain[i0+i][j][(k0+k)%trainDepth[i0+i][j]];
					//if (i == 8*1024 + 180) printf("sInfo = %f, k0=%d, k = %u, size = %u, depth = %u\n", dtsp, k0, k, spikeTrain[i0+i][j].size(), trainDepth[i0+i][j]);
        	        if (dtsp >= 1.0) {
						Size nsp = flooring(dtsp);
        	            dtsp = (dtsp - nsp + k)*dt - time2post;
        	            if (dtsp < dt && dtsp >= 0) {
							Float f = strength;
							if (p > 0) {
								Float normed_std = square_root(p*(1-p)/nSyn);
								Float rand = normal_dist(rGenCond[i0 + i]);
								f *= 1-p + normed_std*rand;
							}
							if (f > 0) {
								dtsp = dt - dtsp;
        	    				#pragma unroll max_ngType
        	    				for (PosInt ig=0; ig<ngType; ig++) {
        	                    	cond->compute_single_input_conductance(local_g[ig], local_h[ig], nsp*f*ip[ig], dtsp, ig);
									//if (i == 8*1024+180) {
									//	printf("	it:%u, far ipre: %u, g=%f, str=%f, h=%f, nsp=%u, dtsp=%f\n", it, ipre, local_g[ig], f*ip[ig], local_h[ig], nsp, dtsp);
									//}
								}
        	                }
        	            } 
        	        }
        	    }
        	}
		}
        // output
        #pragma unroll max_ngTypeE
        for (PosInt ig=0; ig<ngTypeE; ig++) {
            PosInt gid = ig*chunkSize*neuronPerBlock + i;
			//if (i0+i == 8*1024+180) {
			//	printf("it:%u, far gE0: %f\n", it, gE[gid]);
			//}
            gE[gid] = local_gE[ig];
            hE[gid] = local_hE[ig];
			//if (i0+i == 8*1024+180) {
			//	printf("far gE1: %f, gid = %u\n", gE[gid], gid);
			//}
			assert(!isnan(local_gE[ig])); 
			assert(!isnan(local_hE[ig])); 
        }
        #pragma unroll max_ngTypeI
        for (PosInt ig=0; ig<ngTypeI; ig++) {
            PosInt gid = ig*chunkSize*neuronPerBlock + i;
            gI[gid] = local_gI[ig];
            hI[gid] = local_hI[ig];
			assert(!isnan(local_gI[ig])); 
			assert(!isnan(local_hI[ig])); 
        }
    }
}

void recal_Gap_vec(
        std::vector<std::vector<std::vector<Float>>> &gapTrain, std::vector<std::vector<Size>> &gapDepth, std::vector<std::vector<PosInt>> &gap_currentTimeSlot,
        std::vector<Size> &nGapVec, std::vector<std::vector<PosInt>> &gapVecID, std::vector<std::vector<Float>> &gapVec, std::vector<std::vector<Float>> &gapDelayVec,
		std::vector<Float> &vThres, Float gap[], Size typeAcc[],
        Float dt, PosInt block_offset, Size nType, Size nTypeE, Size nI, Float speedOfThought, Size chunkSize, bool noFarDelay, Size neuronPerBlock)
{
    PosInt i0 = block_offset*nI;
    for (PosInt i=0; i<chunkSize*nI; i++) {
		if (noFarDelay) {
			for (PosInt j=0; j<nGapVec[i0+i]; j++) {
    	    	Float gap_strength = static_cast<Float>(gapVec[i0+i][j]);
				PosInt ipre = gapVecID[i0+i][j];
				PosInt jtype;
    			//#pragma unroll (max_nType)
    			for (PosInt k=nTypeE; k<nType; k++) {
    			    if (ipre%neuronPerBlock < typeAcc[k]) {
    			        jtype = k;
    			        break;
    			    }
    			}
				if (gap_strength != 0) {
    	    	    Float v_pre = gapTrain[i0+i][j][0];
					v_pre = v_pre > 0? vThres[jtype]: v_pre;
					gap[i] += gap_strength * v_pre;
				}
			}
		} else {
			for (PosInt j=0; j<nGapVec[i0+i]; j++) {
            	Float gap_strength = static_cast<Float>(gapVec[i0+i][j]);
				PosInt ipre = gapVecID[i0+i][j];
				PosInt jtype;
    			for (PosInt k=nTypeE; k<nType; k++) {
    			    if (ipre%neuronPerBlock < typeAcc[k]) {
    			        jtype = k;
    			        break;
    			    }
    			}
            	Float time2post = static_cast<Float>(gapDelayVec[i0+i][j])/speedOfThought;
                PosInt it2post = static_cast<PosInt>(ceiling(time2post/dt));
                time2post = it2post*dt - time2post;
                int j0 = gap_currentTimeSlot[i0+i][j] - it2post;
				if (j0 < 0) j0 += gapDepth[i0+i][j];

        		gap_currentTimeSlot[i0+i][j] = (gap_currentTimeSlot[i0+i][j]+1)%gapDepth[i0+i][j];
                Float v_pre = gapTrain[i0+i][j][j0];
				v_pre = v_pre > 0? vThres[jtype]: v_pre;
				gap[i] += gap_strength * v_pre;
			}
		}
	}
}

//template<int ntimesFF, int ntimesE, int ntimesI>
__global__ 
void compute_V_collect_spike_learnFF(
        Float* __restrict__ v,
        Float* __restrict__ dep,
        Float* __restrict__ w,
        Float* __restrict__ gapS,
        Float* __restrict__ gFF, // not in chunks
        Float* __restrict__ hFF,
        Float** __restrict__ gE, // in chunks
        Float** __restrict__ gI,
        Float** __restrict__ hE,
        Float** __restrict__ hI,
        Float** __restrict__ gap,
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
        Float* __restrict__ C,
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
        Float* __restrict__ synPerConFF,
        curandStateMRG32k3a* __restrict__ rNoisy,
        Float* __restrict__ noisyDep,
        Float* __restrict__ last_noise,
        Float* __restrict__ output_g,
        Float* __restrict__ output_h,
        Float* __restrict__ totalFF,
        Float* __restrict__ totalFF_inf,
        Float tau_noise, PosInt currentTimeSlot, Size trainDepth, Size max_nLGN, Size ngTypeFF, Size ngTypeE, Size ngTypeI, ConductanceShape condFF, ConductanceShape condE, ConductanceShape condI, Float dt, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1, int learning, int varSlot, Size nType,
		cudaSurfaceObject_t LGNspikeSurface,
        LearnVarShapeFF_E_pre  learnE_pre,  LearnVarShapeFF_I_pre  learnI_pre, 
        LearnVarShapeFF_E_post learnE_post, LearnVarShapeFF_I_post learnI_post, 
        LearnVarShapeE learnE, LearnVarShapeQ learnQ, Float exp_homeo, int iModel, int noDelay, int applyHomeo, bool symmetricHomeo, bool InhGap, bool rebound)
{
    // get different ids in chunks
    PosInt tid = blockIdx.x * blockDim.x + threadIdx.x;
	PosInt gap_tid;
	if (threadIdx.x >= nE) {
    	gap_tid = blockIdx.x * nI + threadIdx.x-nE;
	}
    PosInt iChunk;
    Size chunkSize;
    PosInt cid;
    PosInt gap_id;
    if (blockIdx.x >= iSizeSplit*maxChunkSize) {
        iChunk = iSizeSplit + (blockIdx.x-iSizeSplit*maxChunkSize)/remainChunkSize;
        chunkSize = remainChunkSize*blockDim.x;
        cid = tid - (iSizeSplit*maxChunkSize + (iChunk-iSizeSplit)*remainChunkSize)*blockDim.x;
		if (threadIdx.x >= nE) {
			gap_id = gap_tid - (iSizeSplit*maxChunkSize + (iChunk-iSizeSplit)*remainChunkSize)*nI;
		}
    } else {
        iChunk = blockIdx.x/maxChunkSize;
        chunkSize = maxChunkSize*blockDim.x;
        cid = tid - iChunk*maxChunkSize*blockDim.x;
		if (threadIdx.x >= nE) {
			gap_id = gap_tid - iChunk*maxChunkSize*nI;
		}
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

	Float local_gapS = (threadIdx.x >= nE && InhGap) ? gapS[gap_tid]: 0;
    AdEx model(w[tid], tau_w[itype], a[itype], b[itype], v[tid], tBack[tid], vR[itype], vThres[itype], gL[itype], C[itype], tRef[itype], vT[itype], deltaT[itype], local_gapS, tonicDep[tid]);

	Float noise;
	if (tau_noise > 0) {
		noise = last_noise[tid];
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
		//assert(g >= 0);
		//assert(h >= 0);
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
		//assert(g >= 0);
		//assert(h >= 0);
		gI_t1 += g;
    }

	Float g0[max_ngTypeFF];
	Float h0[max_ngTypeFF];
	Float g1[max_ngTypeFF];
	Float h1[max_ngTypeFF];
    Float p_FF[max_ngTypeFF]; // receptor type proportion 
    #pragma unroll (max_ngTypeFF) //(ntimesFF)
    for (PosInt ig=0; ig<ngTypeFF; ig++) {
        PosInt gid = nV1*ig + tid; // not in chunks
        g1[ig] = gFF[gid]; // g1, all input and decay
        h1[ig] = hFF[gid];
        p_FF[ig] = pFF[itype*ngTypeFF + ig];
        g0[ig] = g1[ig]; // only before tBack
        h0[ig] = h1[ig];

		//	decay
    	condFF.decay_conductance(g1[ig], h1[ig], dt, ig); //  decayed from new_t0 to tBack
	}
	/* debug for snapshot
		if (tid == 8*1024+180) {
			printf("V1: rand0 = %f, rand1 = %f, gFF = %f, tBack = %f, v = %f, , gE = %f, gI =%f\n", uniform(&state), uniform(&localState), g1[0] + g1[1], model.tBack, model.v0, gE_t1, gI_t1);
		}
		__syncthreads();
	*/

	Float p = synFailFF[itype];
	Float nSyn = synPerConFF[itype];

	if (tau_noise == 0) {
		noise = square_root(2*noisyDep[itype]*abs(model.depC)*dt);
		if (noise > 0) noise *= normal(&state);
	} else {
		Float exp_noise = exponential(-dt/tau_noise);
		noise = noise*(exp_noise-1) + square_root(noisyDep[itype]*abs(model.depC)/tau_noise*(1-exp_noise*exp_noise)) * normal(&state);
	}

	bool backingUpFromRef = model.tBack < dt && model.tBack > 0;
	Float new_t0 = 0;
	Float sInfo = 0;
	Float *f = new Float[m];
	Size count = 0;

	dep[tid] = model.depC + noise;
	last_noise[tid] = noise;

	//if (tid == 960) {
	//	Float local_gap = threadIdx.x >= nE ? gap[iChunk][gap_id]: 0;
	//	printf("before #%u: v = %.3f, tBack = %.3f, vT = %.3f, deltaT = %.3f, a0 = %.3f, a1 = %.3f, b0 = %.3f, b1 = %.3f, gapS = %.3f, gap = %.3f, w=%.3f, w0=%.3f, depC = %.3f, dep = %.3f\n", tid, model.v0, model.tBack, model.vT, model.deltaT, model.a0, model.a1, model.b0, model.b1, local_gapS, local_gap, model.w, model.w0, model.depC);
	//	assert(v[tid] == model.v0);
	//	assert(v[tid] < vThres[itype]);
	//}
	//__syncthreads();

	do {
		Float dtBack;
		if (backingUpFromRef) {
			dtBack = model.tBack - new_t0;
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
            PosInt lid = i*nV1 + tid; //transposed
			if (model.spikeCount == 0) {
				f[i] = sLGN[lid];
			}

    	    int x = LGN_idx[lid];
    	    int y = LGN_idy[lid];
    	    float sInfo_FF;
    	    surf2DLayeredread(&sInfo_FF, LGNspikeSurface, 4*x, y, 0);
    	    Size nsp_FF = static_cast<Size>(flooring(sInfo_FF)); // integer part: #spikes
    	    Float tsp_FF = (sInfo_FF - nsp_FF)*dt; // decimal part: normalized mean tsp
			
			if (nsp_FF > 0) {
				if (tsp_FF >= new_t0) {
					if (model.spikeCount == 0 && p > 0) {
						Float normed_std = square_root(p*(1-p)/nSyn);
						Float rand = normal(&localState);
						f[i] *= 1-p + normed_std*rand;
					}
					if (f[i] > 0) {
						Float ddt;
						if (backingUpFromRef) {
							ddt = model.tBack - tsp_FF;
						}
    					#pragma unroll (max_ngTypeFF) //(ntimesFF)
    					for (PosInt ig=0; ig<ngTypeFF; ig++) {
    			    		Float str = f[i] * p_FF[ig];
							if (backingUpFromRef) {
								if (ddt > 0) { // before tBack
    			    				condFF.compute_single_input_conductance(g0[ig], h0[ig], str*nsp_FF, ddt, ig);
    			    			}
							}
							if (model.spikeCount == 0) { // all inputs
    			    			condFF.compute_single_input_conductance(g1[ig], h1[ig], str*nsp_FF, dt-tsp_FF, ig);
							}
						}
					}
				}
			}
    	}
		if (model.spikeCount == 0) {
    		rGenCond[tid] = localState;
		}
		//	collect
		Float gE_t0 = 0.0;
    	#pragma unroll (max_ngTypeFF) //(ntimesFF)
    	for (PosInt ig=0; ig<ngTypeFF; ig++) {
        	PosInt gid = nV1*ig + tid;
    	    gE_t0 += g0[ig];
			if (model.spikeCount == 0) {
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
		model.tsp = 0;
		if (model.tBack < dt) {
			if (threadIdx.x >= nE && InhGap) {
				assert(!isnan(gap[iChunk][gap_id]));
    			model.set_p0(gE_t0, gI_t0, gap[iChunk][gap_id]);

			} else {
    			model.set_p0(gE_t0, gI_t0, 0);
			}
			if (model.spikeCount == 0) {
				if (threadIdx.x >= nE && InhGap) {
    				model.set_p1(gE_t1, gI_t1, gap[iChunk][gap_id]);
				} else {
    				model.set_p1(gE_t1, gI_t1, 0);
				}
			}

			Float new_dt = dt - model.tBack;
			if (backingUpFromRef) { //	stepping other variable before tBack
				model.rk2_vFixedBefore(dtBack);
			} 
			model.rk2(new_dt, noise);


			// check spiking
    	    if (model.v > model.vThres) { // forbids firing exactly at the end of the timestep, 
    	        // crossed threshold
				new_t0 = model.tBack;
    	        model.compute_spike_time(new_dt, new_t0);
				// debug
				 	//if (tid == 1928) {
                	//    printf("%u: v:%f -> %f; new_dt = %f, new_t0 = %f\n", tid, model.v0, model.v, new_dt, new_t0);
					//}
                	if (model.tsp < 0 || isnan(model.tsp) || count > 0) {
						Float expCurr = model.gL*model.deltaT*exponential((model.v0-model.vT)/model.deltaT);
                	    printf("#%u(%u,%u): v:%.1f -> %.1f; a:%.3f(%.3f,%.3f) -> %.3f(%.3f,%.3f), b:%.3f -> %.3f, w:%.3f -> %.3f, exp:%.3f, deltav = %.1f, new_t0 = %.3f, tsp: %.3f, new_dt = %.3f\n", count, tid/blockDim.x, tid%blockDim.x, model.v0, model.v, model.a0, gE_t0, gI_t0, model.a1, gE_t1, gI_t1, model.b0, model.b1, model.w0, model.w, expCurr, (-model.a0*model.v0+model.b0-model.w0 + expCurr)/model.C*new_dt, new_t0, model.tsp, new_dt);
                	    assert(model.tsp >= 0);
                	    assert(!isnan(model.tsp));
                	}
				//
    	        sInfo += model.tsp;
    	        model.spikeCount++;
    	        model.tBack = model.tsp + model.tRef;
				//if (tid == 1928) {
				//	printf("tBack = %f = %f + %f\n", tBack, model.tsp, model.tRef);
				//}
				backingUpFromRef = model.tBack < dt;
				if (backingUpFromRef) {
					model.reset0();
				}
				//if (tid == 1928) {
				//	printf("v[%u] = %f, tBack = %f, vT = %f, deltaT = %f, b0 = %f, b1 = %f, w=%f, w0=%f, depC = %f, dep = %f\n", tid, model.v, model.tBack, model.vT, model.deltaT, model.b0, model.b1, model.w, model.w0, model.depC, dep[tid]);
				//}
    	    } else {
				if (model.tBack > 0) model.tBack = 0;
				backingUpFromRef = false;
			}
		} 
		if (model.tBack >= dt) { // tRef till end
			model.reset1();
			model.rk2_vFixedAfter(dt-model.tsp);
			model.tBack -= dt;
		}
    	/* evolve g to t+dt with ff input only */

		// debug
			if ((isnan(model.v) || model.tBack < 0 || model.v > vThres[itype])) {
				Float local_gap = (threadIdx.x >= nE && InhGap) ? gap[iChunk][gap_id]: 0;
				printf("dead v[%u] = %f, tBack = %f, vT = %f, deltaT = %f, a0 = %f, a1 = %f, b0 = %f, b1 = %f, gapS = %f, gap = %f, w=%f, w0=%f, depC = %f, dep = %f\n", tid, model.v, model.tBack, model.vT, model.deltaT, model.a0, model.a1, model.b0, model.b1, local_gapS, local_gap, model.w, model.w0, model.depC, dep[tid]);
				assert(!isnan(model.v0));
    			assert(model.tBack >= 0);
    			assert(model.v <= vThres[itype]);
			}
		//
		count++;
	} while (backingUpFromRef);
	delete []f;
	rNoisy[tid] = state;

    if (model.spikeCount > 0) {
		sInfo /= model.spikeCount*dt; //decimal part: tsp (normalize by dt)
        sInfo += model.spikeCount; // integer part: nsp
	} else {
        sInfo = model.v;
	} 
    spikeTrain[nV1*currentTimeSlot + tid] = sInfo;
	/*
	if (tid == 8*1024 + 180 || tid == 26959) {
		if (sInfo > 0) {
			printf("%u spiked, sInfo = %f\n", tid, sInfo);
		} else {
			printf("%u sub-thres, sInfo = %f\n", tid, model.v);
		}
	}
	if (tid == 8*1024 + 180) {
		Float local_gap = threadIdx.x >= nE ? gap[iChunk][gap_id]: 0;
		printf("after #%u: v = %.3f, tBack = %.3f, vT = %.3f, deltaT = %.3f, a0 = %.3f, a1 = %.3f, b0 = %.3f, b1 = %.3f, gapS = %.3f, gap = %.3f, w=%.3f, w0=%.3f, depC = %.3f, dep = %.3f\n", tid, model.v, model.tBack, model.vT, model.deltaT, model.a0, model.a1, model.b0, model.b1, local_gapS, local_gap, model.w, model.w0, model.depC, dep[tid]);
		assert(model.v <= vThres[itype]);
	} 
	*/

	v[tid] = model.v;
	if (iModel == 1) {
		w[tid] = model.w;
	}
    tBack[tid] = model.tBack;

    if (learning && learning < 4) {
		Float nsp, tsp;
		if (sInfo > 0) {
			nsp = flooring(sInfo);
			tsp = (sInfo - nsp)*dt;
		} else {
			nsp = 0;
			tsp = dt;
		}
        // will compute ff learning, first row at start of time step, second row at tsp
        Float lFF[2*2*max_nLearnTypeFF]; // row 0: start, row 1: sp
        Float lAvg[2];
        // only temporary store
        Float lE[3*max_nLearnTypeE];
        Float lQ[max_nLearnTypeQ];
        // read ff (post) lVar
        PosInt eid = nE*blockIdx.x+threadIdx.x;
        // read lVars regardless of cortical spike 
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

        if (threadIdx.x < nE || learnI_pre.n) { 
            Float local_totalFF0;
            Float local_totalFF_inf;
            Float new_totalFF0; 
            Float homeostatic_change;
			Float delta_f;
			if (applyHomeo) {
				local_totalFF0 = totalFF[tid];
            	local_totalFF_inf = totalFF_inf[tid];
                Float d_totalF = local_totalFF0-local_totalFF_inf;
                if (d_totalF > 0 || symmetricHomeo) {
            	    new_totalFF0 = d_totalF*exp_homeo + local_totalFF_inf;
				    switch (applyHomeo) {	
				    	case 1: 
                            homeostatic_change = new_totalFF0/local_totalFF0;
				    		break;
				    	case 2:
				    		homeostatic_change = (new_totalFF0 - local_totalFF0)/m;
				    		break;
				    }
                } else {
            	    new_totalFF0 = local_totalFF0;
				    switch (applyHomeo) {	
				    	case 1: 
                            homeostatic_change = 1;
				    		break;
				    	case 2:
				    		homeostatic_change = 0;
				    		break;
				    }
                }
				delta_f = 0.0;
				/*
				if (tid == 743) {
					printf("exp_homeo = %f, f = %f->%f, finf = %f, ratio = %f\n", exp_homeo, local_totalFF0, new_totalFF0, local_totalFF_inf, homeostatic_change);
				}*/
			}
            // learn LGN connection and update LGN lVars
            // learn
            for (PosInt i = 0; i<m; i++) {
                PosInt lid = i*nV1 + tid; //transposed
                Float f = sLGN[lid];
                // pruning process not revertible
                if (f == 0 && rebound) {
                    continue;
                }
				switch (applyHomeo) {
					case 1:
						f *= homeostatic_change;
						break;
					case 2:
						f += homeostatic_change;
						break;
				}
                int x = LGN_idx[lid];
                int y = LGN_idy[lid];
                float sInfo_FF;
                surf2DLayeredread(&sInfo_FF, LGNspikeSurface, 4*x, y, 0);
                Size nsp_FF = static_cast<Size>(flooring(sInfo_FF));
                Float tsp_FF = (sInfo_FF > 0? sInfo_FF - nsp_FF: 1)*dt;
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
                    if (threadIdx.x < nE) {
                        #pragma unroll max_nLearnTypeFF_E
                        for (PosInt j=0; j<learnE_pre.n; j++) {
                            Float A_LTD = learnE_post.A_ratio[j];
							if (learnE_post.targetFR > 0) {
								A_LTD *= learnE_pre.tauLTP[j] * lAvg[cPick] * lAvg[cPick];
							}
                            /*debug
							if (lFF[cPick*max_nLearnTypeFF*2 + 2*j+0] > 0 && i == 0) {
                                printf("%u-%u, A_LTD: %e = %e*%e*%e^2/%e\n", tid, i, A_LTD, learnE_post.A_ratio[j], learnE_pre.tauLTP[j], lAvg[cPick], learnE_post.targetFR);
								printf("%u-%u, old_f: %e, lFF = %e\n", tid, i, f, lFF[cPick*max_nLearnTypeFF*2 + 2*j+0]);
                            }*/
                            Float df = if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t) * A_LTD;
                            f -= df;
							if (applyHomeo) delta_f -= df;
                            /*debug
							if (lFF[cPick*max_nLearnTypeFF*2 + 2*j+0] > 0 && i == 0) {
								printf("%u-%u, new_f: %e\n", tid, i, f);
								Float df = if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t) * A_LTD;
								printf("%u-%u, df %e = %e*%e\n", tid, i, df, if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t), A_LTD);
							}*/
                        }
                    } else {
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt j=0; j<learnI_pre.n; j++) {
							Float A_LTD = learnI_post.A_ratio[j];
							if (learnI_post.targetFR > 0) {
								A_LTD *= learnI_pre.tauLTP[j] * lAvg[cPick] * lAvg[cPick];
							}
                            Float df = if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnI_post.tau[2*j+0], delta_t) * A_LTD;
                            f -= df;
                            if (applyHomeo) delta_f -= df;
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
                    if (threadIdx.x < nE) {
                        #pragma unroll max_nLearnTypeFF_E
                        for (PosInt j=0; j<learnE_pre.n; j++) {
                            float lFF_pref;
                            surf2DLayeredread(&lFF_pref, LGNspikeSurface, 4*x, y, 1+3*j+fPick);
							Float lFF_pre = static_cast<Float>(lFF_pref);
                            /*debug
                            if (lFF_pre > 0 && lFF[max_nLearnTypeFF*2 + 2*j+1] > 0 && i == 0) {
                                printf("%u-%u, LTP, old_f = %e, lFF_pre = %e\n", tid, i, f, lFF_pre);
                            }*/
                            Float df = if_decay(lFF_pre, learnE_pre.tauLTP[j], delta_t) * lFF[max_nLearnTypeFF*2 + 2*j+1] * learnE_post.A_LTP[j];
                            if (applyHomeo) delta_f += df;
                            f += df;
                            /*debug
                            if (lFF_pre > 0 && lFF[max_nLearnTypeFF*2 + 2*j+1] > 0 && i == 0) {
                                printf("%u-%u, new_f:%e += %e*%e*%e\n", tid, i, f, if_decay(lFF_pre, learnE_pre.tauLTP[j], delta_t), lFF[max_nLearnTypeFF*2 + 2*j+1], learnE_post.A_LTP[j]);
                            }*/
                        }
                    } else {
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt j=0; j<learnI_pre.n; j++) {
                            float lFF_pref;
                            surf2DLayeredread(&lFF_pref, LGNspikeSurface, 4*x, y, 1+3*j+fPick);
							Float lFF_pre = static_cast<Float>(lFF_pref);
                            Float df = if_decay(lFF_pre, learnI_pre.tauLTP[j], delta_t) * lFF[max_nLearnTypeFF*2 + 2*j+1] * learnI_post.A_LTP[j];
							if (applyHomeo) delta_f += df;
                            f += df;
                        }
                    }
                }
                if (threadIdx.x < nE) {
                   	if (f < learnE_post.gmin) {
                        if (applyHomeo) delta_f += learnE_post.gmin-f;
                   	    f = learnE_post.gmin;
                   	}
                   	if (f > learnE_post.gmax) {
                        if (applyHomeo) delta_f -= f-learnE_post.gmax;
                   	    f = learnE_post.gmax;
                   	}
                } else {
                   	if (f < learnI_post.gmin) {
                        if (applyHomeo) delta_f += learnI_post.gmin-f;
                   	    f = learnI_post.gmin;
                   	}
                   	if (f > learnI_post.gmax) {
                        if (applyHomeo) delta_f -= f-learnI_post.gmax;
                   	    f = learnI_post.gmax;
                   	}
                }
				/*
				if (applyHomeo && lid == 743*max_nLGN + 30) {
					printf("%f->%f\n", sLGN[lid], f);
				}*/
                sLGN[lid] = f;
            }
            if (applyHomeo) totalFF[tid] = new_totalFF0 + delta_f;

            // update FF vars; lAvg(E) to be updated after cortical learning if nLearnTypeE > 0
            Float delta_t = dt;
            PosInt cPick = nsp > 0? 1: 0;
            if (nsp > 0) { 
                delta_t -= tsp;
            }
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

	if (noDelay && sInfo > 0) {
        ConductanceShape *cond;
        Size ngType; 
        Float nsp = flooring(sInfo);
		Size block_ngType = (ngTypeE*nE + ngTypeI*nI)*blockIdx.x;
		PosInt i = block_ngType;
        if (threadIdx.x < nE) {
            ngType = ngTypeE;
        	cond = &condE;
			i += threadIdx.x*ngTypeE;
        } else {
            ngType = ngTypeI;
            cond = &condI;
			i += nE*ngTypeE + (threadIdx.x-nE)*ngTypeI;
        }
        Float tsp = (sInfo - nsp)*dt;
        #pragma unroll (max_ngType)
        for (PosInt ig=0; ig<ngType; ig++) {
        	Float local_g = 0;
        	Float local_h = 0;
        	cond->compute_single_input_conductance(local_g, local_h, nsp, dt-tsp, ig);
			output_g[i + ig] = local_g; // no need to initialize, as spikeTrain array work as a pointer to updated output vectors only
			output_h[i + ig] = local_h; 
			assert(output_g[i + ig] >= 0);
			assert(output_h[i + ig] >= 0);
        }
	}
}

__launch_bounds__(1024, 1)
__global__ 
void compute_V_collect_spike_learnFF_fast(
        Float* __restrict__ v,
        Float* __restrict__ dep,
        Float* __restrict__ w,
        Float* __restrict__ gapS,
        Float* __restrict__ gFF, // not in chunks
        Float* __restrict__ hFF,
        Float** __restrict__ gE, // in chunks
        Float** __restrict__ gI,
        Float** __restrict__ hE,
        Float** __restrict__ hI,
        Float** __restrict__ gap,
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
        Float* __restrict__ C,
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
        Float* __restrict__ synPerConFF,
        curandStateMRG32k3a* __restrict__ rNoisy,
        Float* __restrict__ noisyDep,
        Float* __restrict__ last_noise,
        PosInt* __restrict__ ipre,
        Size* __restrict__ npre,
        Float* __restrict__ output_g,
        Float* __restrict__ output_h,
        Float* __restrict__ totalFF,
        Float* __restrict__ totalFF_inf,
        Float tau_noise, PosInt currentTimeSlot, Size trainDepth, Size max_nLGN, Size ngTypeFF, Size ngTypeE, Size ngTypeI, ConductanceShape condFF, ConductanceShape condE, ConductanceShape condI, Float dt, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1, int learning, int varSlot, Size nType,
		cudaSurfaceObject_t LGNspikeSurface,
        LearnVarShapeFF_E_pre  learnE_pre,  LearnVarShapeFF_I_pre  learnI_pre, 
        LearnVarShapeFF_E_post learnE_post, LearnVarShapeFF_I_post learnI_post, 
        LearnVarShapeE learnE, LearnVarShapeQ learnQ, Float exp_homeo, int iModel, int noDelay, int applyHomeo, bool symmetricHomeo, bool InhGap, bool rebound)
{
    __shared__ PosInt counter[2];
    if (threadIdx.x < 2) {
        counter[threadIdx.x] = 0;
    }
    __syncthreads();
    // get different ids in chunks
    PosInt tid = blockIdx.x * blockDim.x + threadIdx.x;
	PosInt gap_tid;
	if (threadIdx.x >= nE) {
    	gap_tid = blockIdx.x * nI + threadIdx.x-nE;
	}
    PosInt iChunk;
    Size chunkSize;
    PosInt cid;
    PosInt gap_id;
    if (blockIdx.x >= iSizeSplit*maxChunkSize) {
        iChunk = iSizeSplit + (blockIdx.x-iSizeSplit*maxChunkSize)/remainChunkSize;
        chunkSize = remainChunkSize*blockDim.x;
        cid = tid - (iSizeSplit*maxChunkSize + (iChunk-iSizeSplit)*remainChunkSize)*blockDim.x;
		if (threadIdx.x >= nE && InhGap) {
			gap_id = gap_tid - (iSizeSplit*maxChunkSize + (iChunk-iSizeSplit)*remainChunkSize)*nI;
		}
    } else {
        iChunk = blockIdx.x/maxChunkSize;
        chunkSize = maxChunkSize*blockDim.x;
        cid = tid - iChunk*maxChunkSize*blockDim.x;
		if (threadIdx.x >= nE && InhGap) {
			gap_id = gap_tid - iChunk*maxChunkSize*nI;
		}
    }

	bool pInfo = false;
	//if (blockIdx.x == 33 && threadIdx.x == 678) {
	if (false) {
		pInfo = true;
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

	Float local_gapS = threadIdx.x >= nE? gapS[gap_tid]: 0;
    AdEx model(w[tid], tau_w[itype], a[itype], b[itype], v[tid], tBack[tid], vR[itype], vThres[itype], gL[itype], C[itype], tRef[itype], vT[itype], deltaT[itype], local_gapS, tonicDep[tid]);

	Float noise;
	if (tau_noise > 0) {
		noise = last_noise[tid];
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
		//assert(g >= 0);
		//assert(h >= 0);
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
		//assert(g >= 0);
		//assert(h >= 0);
		gI_t1 += g;
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
    	condFF.decay_conductance(g1[ig], h1[ig], dt, ig); //  decay
	}
	/* debug for snapshot
		if (tid == 8*1024+180) {
			printf("V1: rand0 = %f, rand1 = %f, gFF = %f, tBack = %f, v = %f, , gE = %f, gI =%f\n", uniform(&state), uniform(&localState), g1[0] + g1[1], model.tBack, model.v0, gE_t1, gI_t1);
		}
		__syncthreads();
	*/

	Float p = synFailFF[itype];
	Float nSyn = synPerConFF[itype];
	Float _noisyDep = noisyDep[itype]*(vT[itype] - vR[itype]);

	if (tau_noise == 0) {
		noise = square_root(2*_noisyDep*dt);
		if (noise > 0) noise *= normal(&state);
	} else {
		Float exp_noise = exponential(-dt/tau_noise);
		noise = noise*(exp_noise-1) + square_root(_noisyDep/tau_noise*(1-exp_noise*exp_noise)) * normal(&state);
	}

	bool backingUpFromRef = model.tBack < dt && model.tBack > 0;

	dep[tid] = model.depC + noise;
	last_noise[tid] = noise;

	/*
	if (tid == 960) {
		Float local_gap = threadIdx.x >= nE ? gap[iChunk][gap_id]: 0;
		printf("before #%u: v = %.3f, tBack = %.3f, vT = %.3f, deltaT = %.3f, a0 = %.3f, a1 = %.3f, b0 = %.3f, b1 = %.3f, gapS = %.3f, gap = %.3f, w=%.3f, w0=%.3f, depC = %.3f, dep = %.3f\n", tid, model.v0, model.tBack, model.vT, model.deltaT, model.a0, model.a1, model.b0, model.b1, local_gapS, local_gap, model.w, model.w0, model.depC);
		assert(v[tid] == model.v0);
		assert(v[tid] < vThres[itype]);
	}
	__syncthreads();
    */

	//	condFF
	//		decay part
    #pragma unroll (max_ngTypeFF) //(ntimesFF)
    for (PosInt ig=0; ig<ngTypeFF; ig++) {
		if (backingUpFromRef) { // collect conductance at tBack
    	    condFF.decay_conductance(g0[ig], h0[ig], model.tBack, ig); //  decayed to tBack
		}
	}
	//		input part
    #pragma unroll (4)
    for (PosInt i = 0; i<m; i++) {
        PosInt lid = i*nV1 + tid; //transposed
		Float f = sLGN[lid];

        int x = LGN_idx[lid];
        int y = LGN_idy[lid];
        float sInfo_FF;
        surf2DLayeredread(&sInfo_FF, LGNspikeSurface, 4*x, y, 0);
        Size nsp_FF = static_cast<Size>(flooring(sInfo_FF)); // integer part: #spikes
		
		if (nsp_FF > 0) {
            Float tsp_FF = (sInfo_FF - nsp_FF)*dt; // decimal part: normalized mean tsp
			if (p > 0) {
				Float normed_std = square_root(p*(1-p)/nSyn);
				Float rand = normal(&localState);
				f *= 1-p + normed_std*rand;
			}
			if (f > 0) {
				if (pInfo) {
					printf("\n#V1:%i-%i received LGN %i, (%i,%i) input at %f of size %.1f\n", blockIdx.x, threadIdx.x, i, x, y, tsp_FF, f);
				}
				Float ddt;
				if (backingUpFromRef) {
					ddt = model.tBack - tsp_FF;
				}
    			#pragma unroll (max_ngTypeFF) //(ntimesFF)
    			for (PosInt ig=0; ig<ngTypeFF; ig++) {
    				Float str = f * pFF[itype*ngTypeFF + ig];
					if (backingUpFromRef && ddt > 0) {
    					condFF.compute_single_input_conductance(g0[ig], h0[ig], str*nsp_FF, ddt, ig);
					}
    				condFF.compute_single_input_conductance(g1[ig], h1[ig], str*nsp_FF, dt-tsp_FF, ig);
				}
			}
		}
    }
    rGenCond[tid] = localState;
	//	collect
	Float gE_t0 = 0.0;
    #pragma unroll (max_ngTypeFF) //(ntimesFF)
    for (PosInt ig=0; ig<ngTypeFF; ig++) {
    	PosInt gid = nV1*ig + tid;
        gE_t0 += g0[ig];
        gE_t1 += g1[ig];
    	gFF[gid] = g1[ig];
    	hFF[gid] = h1[ig];
    }
    
	//	condE, g0
    #pragma unroll (max_ngTypeE)
    for (PosInt ig=0; ig<ngTypeE; ig++) {
		if (backingUpFromRef) {
        	condE.decay_conductance(ge[ig], he[ig], model.tBack, ig); 
		}
        gE_t0 += ge[ig];
	}

	//	condI, g0
    Float gI_t0 = 0.0;
    #pragma unroll (max_ngTypeI)
    for (PosInt ig=0; ig<ngTypeI; ig++) {
		if (backingUpFromRef) {
        	condI.decay_conductance(gi[ig], hi[ig], model.tBack, ig); 
		}
        gI_t0 += gi[ig];
	}

    // stepping
	model.tsp = 0;
	if (model.tBack < dt) {
		if (threadIdx.x >= nE && InhGap) {
			//assert(!isnan(gap[iChunk][gap_id]));
    		model.set_p0(gE_t0, gI_t0, gap[iChunk][gap_id]);
    		model.set_p1(gE_t1, gI_t1, gap[iChunk][gap_id]);
		} else {
    		model.set_p0(gE_t0, gI_t0, 0);
    		model.set_p1(gE_t1, gI_t1, 0);
		}

		Float new_dt = dt - model.tBack;
		if (backingUpFromRef) { //	stepping other variable before tBack
			model.rk2_vFixedBefore(model.tBack);
		} 
		model.rk2(new_dt, noise);


		// check spiking
        if (model.v > model.vThres) { // forbids firing exactly at the end of the timestep, 
            // crossed threshold
            model.compute_spike_time(new_dt, model.tBack);
			// debug
			 	//if (tid == 1928) {
            	//    printf("%u: v:%f -> %f; new_dt = %f, t0 = %f\n", tid, model.v0, model.v, new_dt, model.tBack);
				//}
            	if (model.tsp < 0 || isnan(model.tsp)) {
					Float expCurr = model.gL*model.deltaT*exponential((model.v0-model.vT)/model.deltaT);
            	    printf("(%u,%u): v:%.1f -> %.1f; a:%.3f(%.3f,%.3f) -> %.3f(%.3f,%.3f), b:%.3f -> %.3f, w:%.3f -> %.3f, exp:%.3f, deltav = %.1f, t0 = %.3f, tsp: %.3f, new_dt = %.3f\n", tid/blockDim.x, tid%blockDim.x, model.v0, model.v, model.a0, gE_t0, gI_t0, model.a1, gE_t1, gI_t1, model.b0, model.b1, model.w0, model.w, expCurr, (-model.a0*model.v0+model.b0-model.w0 + expCurr)/model.C*new_dt, model.tBack, model.tsp, new_dt);
            	    assert(model.tsp >= 0);
            	    assert(!isnan(model.tsp));
            	}
			//
            model.spikeCount = 1;
            model.tBack = model.tsp + model.tRef;
			//if (tid == 1928) {
			//	printf("tBack = %f = %f + %f\n", tBack, model.tsp, model.tRef);
			//}
			assert(model.tBack >= dt);
			//if (tid == 1928) {
			//	printf("v[%u] = %f, tBack = %f, vT = %f, deltaT = %f, b0 = %f, b1 = %f, w=%f, w0=%f, depC = %f, dep = %f\n", tid, model.v, model.tBack, model.vT, model.deltaT, model.b0, model.b1, model.w, model.w0, model.depC, dep[tid]);
			//}
        } else {
			if (model.tBack > 0) model.tBack = 0;
		}
	} 
	if (model.tBack >= dt) { // tRef till end
		model.reset1();
		model.rk2_vFixedAfter(dt-model.tsp);
		model.tBack -= dt;
	}
    /* evolve g to t+dt with ff input only */

	/* debug
		if ((isnan(model.v) || model.tBack < 0 || model.v > vThres[itype])) {
			Float local_gap = threadIdx.x >= nE ? gap[iChunk][gap_id]: 0;
			printf("dead v[%u] = %f, tBack = %f, vT = %f, deltaT = %f, a0 = %f, a1 = %f, b0 = %f, b1 = %f, gapS = %f, gap = %f, w=%f, w0=%f, depC = %f, dep = %f\n", tid, model.v, model.tBack, model.vT, model.deltaT, model.a0, model.a1, model.b0, model.b1, local_gapS, local_gap, model.w, model.w0, model.depC, dep[tid]);
			assert(!isnan(model.v0));
    		assert(model.tBack >= 0);
    		assert(model.v <= vThres[itype]);
		}
	*/
	rNoisy[tid] = state;

    Float sInfo = 0.0;
    if (model.spikeCount > 0) {
        sInfo = 1.0 + model.tsp/dt;
	} else {
        sInfo = model.v;
    }
    spikeTrain[nV1*currentTimeSlot + tid] = sInfo;
	/*
	    if (tid == 8*1024 + 180 || tid == 26959) {
	    	if (sInfo > 0) {
	    		printf("%u spiked, sInfo = %f\n", tid, sInfo);
	    	} else {
	    		printf("%u sub-thres, sInfo = %f\n", tid, model.v);
	    	}
	    }
	    if (tid == 8*1024 + 180) {
	    	Float local_gap = threadIdx.x >= nE ? gap[iChunk][gap_id]: 0;
	    	printf("after #%u: v = %.3f, tBack = %.3f, vT = %.3f, deltaT = %.3f, a0 = %.3f, a1 = %.3f, b0 = %.3f, b1 = %.3f, gapS = %.3f, gap = %.3f, w=%.3f, w0=%.3f, depC = %.3f, dep = %.3f\n", tid, model.v, model.tBack, model.vT, model.deltaT, model.a0, model.a1, model.b0, model.b1, local_gapS, local_gap, model.w, model.w0, model.depC, dep[tid]);
	    	assert(model.v <= vThres[itype]);
	    } 
	*/

	v[tid] = model.v;
	if (iModel == 1) {
		w[tid] = model.w;
	}
    tBack[tid] = model.tBack;

    if (learning && learning < 4) {
		Float nsp, tsp;
		if (sInfo > 0) {
			nsp = flooring(sInfo);
			tsp = (sInfo - nsp)*dt;
		} else {
			nsp = 0;
			tsp = dt;
		}
        // will compute ff learning, first row at start of time step, second row at tsp
        Float lFF[2*2*max_nLearnTypeFF]; // row 0: start, row 1: sp
        Float lAvg[2];
        // only temporary store
        Float lE[3*max_nLearnTypeE];
        Float lQ[max_nLearnTypeQ];
        // read ff (post) lVar
        PosInt eid = nE*blockIdx.x+threadIdx.x;
        // read lVars regardless of cortical spike 
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

        if (threadIdx.x < nE || learnI_pre.n) { 
            Float local_totalFF0;
            Float local_totalFF_inf;
            Float new_totalFF0; 
            Float homeostatic_change;
			Float delta_f;
			if (applyHomeo) {
				local_totalFF0 = totalFF[tid];
            	local_totalFF_inf = totalFF_inf[tid];
                Float d_totalF = local_totalFF0-local_totalFF_inf;
                if (d_totalF > 0 || symmetricHomeo) {
            	    new_totalFF0 = d_totalF*exp_homeo + local_totalFF_inf;
				    switch (applyHomeo) {	
				    	case 1: 
                            homeostatic_change = new_totalFF0/local_totalFF0;
				    		break;
				    	case 2:
				    		homeostatic_change = (new_totalFF0 - local_totalFF0)/m;
				    		break;
				    }
                } else {
            	    new_totalFF0 = local_totalFF0;
				    switch (applyHomeo) {	
				    	case 1: 
                            homeostatic_change = 1;
				    		break;
				    	case 2:
				    		homeostatic_change = 0;
				    		break;
				    }
                }

				delta_f = 0.0;
				/*
				if (tid == 743) {
					printf("exp_homeo = %f, f = %f->%f, finf = %f, ratio = %f\n", exp_homeo, local_totalFF0, new_totalFF0, local_totalFF_inf, homeostatic_change);
				}*/
			}
            // learn LGN connection and update LGN lVars
            // learn
            for (PosInt i = 0; i<m; i++) {
                PosInt lid = i*nV1 + tid; //transposed
                Float f = sLGN[lid];
                // pruning process not revertible
                if (f == 0 && !rebound) {
                    continue;
                }
				switch (applyHomeo) {
					case 1:
						f *= homeostatic_change;
						break;
					case 2:
						f += homeostatic_change;
						break;
				}
                int x = LGN_idx[lid];
                int y = LGN_idy[lid];
                float sInfo_FF;
                surf2DLayeredread(&sInfo_FF, LGNspikeSurface, 4*x, y, 0);
                Size nsp_FF = static_cast<Size>(flooring(sInfo_FF));
                Float tsp_FF = (sInfo_FF > 0? sInfo_FF - nsp_FF: 1)*dt;
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
                    if (threadIdx.x < nE) {
                        #pragma unroll max_nLearnTypeFF_E
                        for (PosInt j=0; j<learnE_pre.n; j++) {
                            Float A_LTD = learnE_post.A_ratio[j];
							if (learnE_post.targetFR > 0) {
								A_LTD *= learnE_pre.tauLTP[j] * lAvg[cPick] * lAvg[cPick];
							}
                            /*debug
							if (lFF[cPick*max_nLearnTypeFF*2 + 2*j+0] > 0 && i == 0) {
                                printf("%u-%u, A_LTD: %e = %e*%e*%e^2/%e\n", tid, i, A_LTD, learnE_post.A_ratio[j], learnE_pre.tauLTP[j], lAvg[cPick], learnE_post.targetFR);
								printf("%u-%u, old_f: %e, lFF = %e\n", tid, i, f, lFF[cPick*max_nLearnTypeFF*2 + 2*j+0]);
                            }*/
                            Float df = if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t) * A_LTD;
                            f -= df;
							if (applyHomeo) delta_f -= df;
                            /*debug
							if (lFF[cPick*max_nLearnTypeFF*2 + 2*j+0] > 0 && i == 0) {
								printf("%u-%u, new_f: %e\n", tid, i, f);
								Float df = if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t) * A_LTD;
								printf("%u-%u, df %e = %e*%e\n", tid, i, df, if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnE_post.tau[2*j+0], delta_t), A_LTD);
							}*/
                        }
                    } else {
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt j=0; j<learnI_pre.n; j++) {
							Float A_LTD = learnI_post.A_ratio[j];
							if (learnI_post.targetFR > 0) {
								A_LTD *= learnI_pre.tauLTP[j] * lAvg[cPick] * lAvg[cPick];
							}
                            Float df = if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*j+0], learnI_post.tau[2*j+0], delta_t) * A_LTD;
                            f -= df;
                            if (applyHomeo) delta_f -= df;
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
                    if (threadIdx.x < nE) {
                        #pragma unroll max_nLearnTypeFF_E
                        for (PosInt j=0; j<learnE_pre.n; j++) {
                            float lFF_pref;
                            surf2DLayeredread(&lFF_pref, LGNspikeSurface, 4*x, y, 1+3*j+fPick);
							Float lFF_pre = static_cast<Float>(lFF_pref);
                            /*debug
                            if (lFF_pre > 0 && lFF[max_nLearnTypeFF*2 + 2*j+1] > 0 && i == 0) {
                                printf("%u-%u, LTP, old_f = %e, lFF_pre = %e\n", tid, i, f, lFF_pre);
                            }*/
                            Float df = if_decay(lFF_pre, learnE_pre.tauLTP[j], delta_t) * lFF[max_nLearnTypeFF*2 + 2*j+1] * learnE_post.A_LTP[j];
                            if (applyHomeo) delta_f += df;
                            f += df;
                            /*debug
                            if (lFF_pre > 0 && lFF[max_nLearnTypeFF*2 + 2*j+1] > 0 && i == 0) {
                                printf("%u-%u, new_f:%e += %e*%e*%e\n", tid, i, f, if_decay(lFF_pre, learnE_pre.tauLTP[j], delta_t), lFF[max_nLearnTypeFF*2 + 2*j+1], learnE_post.A_LTP[j]);
                            }*/
                        }
                    } else {
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt j=0; j<learnI_pre.n; j++) {
                            float lFF_pref;
                            surf2DLayeredread(&lFF_pref, LGNspikeSurface, 4*x, y, 1+3*j+fPick);
							Float lFF_pre = static_cast<Float>(lFF_pref);
                            Float df = if_decay(lFF_pre, learnI_pre.tauLTP[j], delta_t) * lFF[max_nLearnTypeFF*2 + 2*j+1] * learnI_post.A_LTP[j];
							if (applyHomeo) delta_f += df;
                            f += df;
                        }
                    }
                }
                if (threadIdx.x < nE) {
                   	if (f < learnE_post.gmin) {
                        if (applyHomeo) delta_f += learnE_post.gmin-f;
                   	    f = learnE_post.gmin;
                   	}
                   	if (f > learnE_post.gmax) {
                        if (applyHomeo) delta_f -= f-learnE_post.gmax;
                   	    f = learnE_post.gmax;
                   	}
                } else {
                   	if (f < learnI_post.gmin) {
                        if (applyHomeo) delta_f += learnI_post.gmin-f;
                   	    f = learnI_post.gmin;
                   	}
                   	if (f > learnI_post.gmax) {
                        if (applyHomeo) delta_f -= f-learnI_post.gmax;
                   	    f = learnI_post.gmax;
                   	}
                }
				/*
				if (applyHomeo && lid == 743*max_nLGN + 30) {
					printf("%f->%f\n", sLGN[lid], f);
				}*/
                sLGN[lid] = f;
            }
            if (applyHomeo) totalFF[tid] = new_totalFF0 + delta_f;

            // update FF vars; lAvg(E) to be updated after cortical learning if nLearnTypeE > 0
            Float delta_t = dt;
            PosInt cPick = nsp > 0? 1: 0;
            if (nsp > 0) { 
                delta_t -= tsp;
            }
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
    
    // count and id spikes
    bool exist = sInfo >= 1;
    PosInt type, id;
    if (threadIdx.x < nE) {
        type = 0;
    } else {
        type = 1;
    }
    if (exist) {
        id = atomicAdd_block(&counter[type], 1);
    }
    __syncthreads();
    if (exist) { // fill output_g/h
        Float nsp = flooring(sInfo);
		Size block_ngType = (ngTypeE*nE + ngTypeI*nI)*blockIdx.x;
		PosInt i = block_ngType;
        Float tsp = (sInfo - nsp)*dt;
        Size npreE = counter[0];
        if (threadIdx.x < nE) {
            assert(id < npreE);
			i += id;
            #pragma unroll (max_ngTypeE)
            for (PosInt ig=0; ig<ngTypeE; ig++) {
            	Float local_g = 0;
            	Float local_h = 0;
            	condE.compute_single_input_conductance(local_g, local_h, nsp, dt-tsp, ig);
		    	output_g[ig*npreE + i] = local_g;
		    	output_h[ig*npreE + i] = local_h;
            }
        } else {
            Size npreI = counter[1];
			i += npreE*ngTypeE + id;
            #pragma unroll (max_ngTypeI)
            for (PosInt ig=0; ig<ngTypeI; ig++) {
            	Float local_g = 0;
            	Float local_h = 0;
            	condI.compute_single_input_conductance(local_g, local_h, nsp, dt-tsp, ig);
		    	output_g[ig*npreI + i] = local_g;
		    	output_h[ig*npreI + i] = local_h;
            }
            assert(id < npreI);
            id += npreE; // go after exc
        }
        ipre[blockIdx.x*blockDim.x + id] = threadIdx.x;
	}
    if (threadIdx.x < 2) {
        npre[threadIdx.x*gridDim.x + blockIdx.x] = counter[threadIdx.x];
    }
}

//template<int ntimesE, int ntimesI>
__launch_bounds__(1024, 1)
__global__  // <<< nblock[partial], blockSize >>>
void recal_G_mat(
        Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        float* __restrict__ conMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        float* __restrict__ delayMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        float* __restrict__ gapMat, // [nblock, nearNeighborBlock, nI, nI]
        Size* __restrict__ nNearNeighborBlock,
        PosInt* __restrict__ neighborBlockId,
        Float* __restrict__ gE, // [ngTypeE, nV1]
        Float* __restrict__ gI, // [ngTypeI, nV1] 
        Float* __restrict__ hE,
        Float* __restrict__ hI,
        Float* __restrict__ gap, // gap
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
        Float* __restrict__ synPerCon,
		Float* __restrict__ vThres,
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt currentTimeSlot, Size trainDepth, Size nearNeighborBlock, Size nE, Size nI, Size nV1, Float speedOfThought, int learning, PosInt block_offset, Size nType, Size nTypeE, Size nTypeI,
        LearnVarShapeE lE, LearnVarShapeQ lQ, PosInt iChunk, bool InhGap)
{
    // each thread is the post neuron that collects its presynaptic input conductances
    // initialize
	__shared__ Size tA[max_nType];
	if (threadIdx.x == 0) {
    	for (PosInt i=0; i<nType; i++) {
    	    tA[i] = typeAcc[i];
    	}
	}
	__syncthreads();
    PosInt itype;
    #pragma unroll (max_nType)
    for (PosInt i=0; i<nType; i++) {
        if (threadIdx.x < tA[i]) {
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
	Float gap_s = 0;
    // TODO: cortical learning
    //Float trip_post[2*max_nLearnTypeE];
    //Float LTD_post[2*max_nLearnTypeE];
    PosInt ipost = (block_offset+blockIdx.x)*blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = rGenCond[ipost];
    Float post_sInfo = spikeTrain[nV1*currentTimeSlot + ipost];
    Float postNsp = flooring(post_sInfo);
    Float postTsp = post_sInfo>0? post_sInfo - postNsp: 1;
    Float lAvgE;
    if (learning != 3) {
        if (threadIdx.x < nE) {
            PosInt cPick = postNsp>0? 1:0;
            PosInt eid = (block_offset+blockIdx.x)*nE + threadIdx.x;
            lAvgE = vAvgE[2*eid+cPick];
        }
    }

    //__syncthreads();
    #pragma unroll (4)
    for (PosInt ib = 0; ib < nNearNeighborBlock[blockIdx.x]; ib++) {
		PosInt local_bid = blockIdx.x*nearNeighborBlock + ib;
		PosInt bid = neighborBlockId[local_bid];
        // check for old spikes
        #pragma unroll
        for (PosInt i=0; i<blockDim.x; i++) {
			PosInt ipre = bid*blockDim.x + i;
            // access each presynaptic neurons in stride
            // conMat: [nblock,nearNeighborBlock,blockDim.x,blockDim.x] last dim is the post-id: second-last pre-id
            PosIntL mid = static_cast<PosIntL>((local_bid*blockDim.x + i)*blockDim.x + threadIdx.x);
            Float strength = static_cast<Float>(conMat[mid]);
            if (strength != 0) {
				PosInt jtype;
    			#pragma unroll (max_nType)
    			for (PosInt j=0; j<nType; j++) {
    			    if (i < tA[j]) {
    			        jtype = j;
    			        break;
    			    }
    			}
                //Float LTP_pre[max_nLearnTypeE];
                //Float Q_pre[max_nLearnTypeQ];
            	Float p = synFail[jtype*nType + itype];
				Float nSyn = synPerCon[jtype*nType + itype];
                Float time2post = static_cast<Float>(delayMat[mid])/speedOfThought;
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
                int j0 = currentTimeSlot - it2post;
				if (j0 < 0) j0 += trainDepth;
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
					if (dtsp >= 1.0)  {
						Size nsp = flooring(dtsp);
                	    dtsp = (dtsp - nsp + j)*dt - time2post;
                		if (dtsp < dt && dtsp >= 0) {
							Float f = strength; 
							if (p > 0) {
								Float normed_std = square_root(p*(1-p)/nSyn);
								Float rand = normal(&localState);
								f *= 1-p + normed_std*rand;
							}
							if (f > 0) {
								dtsp = dt - dtsp;
                				#pragma unroll (max_ngType)
                				for (PosInt ig=0; ig<ngType; ig++) {
                				    cond->compute_single_input_conductance(local_g[ig], local_h[ig], nsp*f*ip[ig], dtsp, ig);
									assert(local_g[ig] >= 0);
									assert(local_h[ig] >= 0);
                				}
							}
						}
					}
                }
            }
            __syncwarp(); // may not be needed
			if (threadIdx.x >= nE && i >= nE) {
            	PosIntL gid = static_cast<PosIntL>((local_bid*nI + (i-nE))*nI + threadIdx.x-nE);
            	Float gap_strength = static_cast<Float>(gapMat[gid]);
				if (gap_strength != 0) {
					PosInt jtype;
    				//#pragma unroll (max_nType)
    				for (PosInt j=nTypeE; j<nType; j++) {
    				    if (i < tA[j]) {
    				        jtype = j;
    				        break;
    				    }
    				}
            		PosIntL mid = static_cast<PosIntL>((local_bid*blockDim.x + i)*blockDim.x + threadIdx.x);
                	Float time2post = static_cast<Float>(delayMat[mid])/speedOfThought;
            	    PosInt it2post = static_cast<PosInt>(ceiling(time2post/dt));
            	    time2post = it2post*dt - time2post;
                	int j0 = currentTimeSlot - it2post;
					if (j0 < 0) j0 += trainDepth;
            	    Float v_pre = spikeTrain[nV1*(j0%trainDepth) + ipre];
					v_pre = v_pre > 0? vThres[jtype]: v_pre;
					gap_s += gap_strength * v_pre;
				}
			}
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
            	}
            	#pragma unroll (max_nLearnTypeE)
            	for (PosInt i=0; i<lE.n; i++) {
            	}
			*/
        }
    }

    PosInt id = blockIdx.x*blockDim.x + threadIdx.x;
    //#pragma unroll (ntimesE)
    #pragma unroll (max_ngTypeE)
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        PosInt gid = ig*gridDim.x*blockDim.x + id;
        gE[gid] += local_gE[ig];
        hE[gid] += local_hE[ig];
		assert(gE[gid] >= 0);
		assert(hE[gid] >= 0);
    }
    //#pragma unroll (ntimesI)
    #pragma unroll (max_ngTypeI)
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        PosInt gid = ig*gridDim.x*blockDim.x + id;
        gI[gid] += local_gI[ig];
        hI[gid] += local_hI[ig];
		assert(gI[gid] >= 0);
		assert(hI[gid] >= 0);
    }
	if (threadIdx.x >= nE && InhGap) {
    	PosInt gap_ipost = blockIdx.x*nI + threadIdx.x-nE;
		gap[gap_ipost] = gap_s;
		assert(gap_s >= 0);
	}
}

//template<int ntimesE, int ntimesI>
__launch_bounds__(1024, 2)
__global__
void sum_G(
        Size* __restrict__ nVec, // block_offset accounted for
        Float* __restrict__ gEt,
        Float* __restrict__ gE,
        Float* __restrict__ gIt,
        Float* __restrict__ gI,
        Float* __restrict__ hEt,
        Float* __restrict__ hE,
        Float* __restrict__ hIt,
        Float* __restrict__ hI,
        Size ngTypeE, Size ngTypeI, PosInt it)
{
    PosInt id = blockIdx.x*blockDim.x + threadIdx.x;
    if (nVec[id] > 0) {
        //#pragma unroll (ntimesE)
        #pragma unroll (max_ngTypeE)
        for (PosInt ig=0; ig<ngTypeE; ig++) {
            PosInt gid = ig*gridDim.x*blockDim.x + id;
			//if (id == 8*1024+180) {
			//	printf("it:%u, sum gE0: %f\n", it, gE[gid]);
			//}
            gE[gid] += gEt[gid];
            hE[gid] += hEt[gid];
			//if (id == 8*1024+180) {
			//	printf("sum gE1: %f, gid = %u\n", gE[gid], gid);
			//}
        }
        //#pragma unroll (ntimesI) 
        #pragma unroll (max_ngTypeI) 
        for (PosInt ig=0; ig<ngTypeI; ig++) {
            PosInt gid = ig*gridDim.x*blockDim.x + id;
            gI[gid] += gIt[gid];
            hI[gid] += hIt[gid];
        }
    }
}

__launch_bounds__(1024, 2)
__global__
void sum_Gap(
        Size* __restrict__ nGapVec, // block_offset accounted for
        Float* __restrict__ gapt,
        Float* __restrict__ gap)
{
    PosInt id = blockIdx.x*blockDim.x + threadIdx.x;
    if (nGapVec[id] > 0) {
        gap[id] += gapt[id];
	}
}

__launch_bounds__(1024, 1)
__global__  // <<< nblock[partial], blockSize >>>
void recal_G_mat_nd( // no distance involved for close-range connections, i.e., no delay in spike transport
		Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        Float* __restrict__ output_g, // [depth, nblock, blockSize]
        Float* __restrict__ output_h, // [depth, nblock, blockSize]
        float* __restrict__ conMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        float* __restrict__ gapMat, // [nblock, nearNeighborBlock, nI, nI]
        Size* __restrict__ nNearNeighborBlock, // block_offset accounted for
        PosInt* __restrict__ neighborBlockId, // block_offset accounted for
        Float* __restrict__ gE, // [ngTypeE, nV1]
        Float* __restrict__ gI, // [ngTypeI, nV1] 
        Float* __restrict__ hE,
        Float* __restrict__ hI,
        Float* __restrict__ gap, // gap
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
        Float* __restrict__ synPerCon,
		Float* __restrict__ vThres,
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, Size nearNeighborBlock, Size nE, Size nI, Size nV1, int learning, PosInt block_offset, Size nType, Size nTypeE, Size nTypeI,
        LearnVarShapeE lE, LearnVarShapeQ lQ, PosInt iChunk, PosInt it, bool InhGap)
{
    // each thread is the post neuron that collects its presynaptic input conductances
    // initialize
	__shared__ Size tA[max_nType];
	if (threadIdx.x == 0) {
    	for (PosInt i=0; i<nType; i++) {
    	    tA[i] = typeAcc[i];
    	}
	}
	__syncthreads();
    PosInt itype;
    #pragma unroll (max_nType)
    for (PosInt i=0; i<nType; i++) {
        if (threadIdx.x < tA[i]) {
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
	Float gap_s = 0.0;
	//Size nGap = 0;
    // TODO: cortical learning
    //Float trip_post[2*max_nLearnTypeE];
    //Float LTD_post[2*max_nLearnTypeE];
    PosInt ipost = (block_offset+blockIdx.x)*blockDim.x + threadIdx.x;
    curandStateMRG32k3a localState = rGenCond[ipost];
    Float post_sInfo = spikeTrain[ipost];
    Float postNsp = flooring(post_sInfo);
    Float postTsp = post_sInfo>0? post_sInfo - postNsp: 1;
    Float lAvgE;
    if (learning != 3) {
        if (threadIdx.x < nE) {
            PosInt cPick = postNsp>0? 1:0;
            PosInt eid = (block_offset+blockIdx.x)*nE + threadIdx.x;
            lAvgE = vAvgE[2*eid+cPick];
        }
    }

    //__syncthreads();
    #pragma unroll (4)
    for (PosInt ib = 0; ib < nNearNeighborBlock[blockIdx.x]; ib++) {
		PosInt local_bid = blockIdx.x*nearNeighborBlock + ib;
		PosInt bid = neighborBlockId[local_bid];
        // check for old spikes
        #pragma unroll
        for (PosInt i=0; i<blockDim.x; i++) {
			PosInt ipre = bid*blockDim.x + i;
            // access each presynaptic neurons in stride
            // conMat: [nblock,nearNeighborBlock,blockDim.x,blockDim.x] last dim is the post-id: second-last pre-id
            PosIntL mid = static_cast<PosIntL>((local_bid*blockDim.x + i)*blockDim.x + threadIdx.x);
            Float strength = static_cast<Float>(conMat[mid]);
            if (strength != 0  && spikeTrain[ipre] > 0) {
				PosInt jtype;	
    			#pragma unroll (max_nType)
    			for (PosInt j=0; j<nType; j++) {
    			    if (i < tA[j]) {
    			        jtype = j;
    			        break;
    			    }
    			}
                //Float LTP_pre[max_nLearnTypeE];
                //Float Q_pre[max_nLearnTypeQ];
            	Float p = synFail[jtype*nType + itype];
				Float nSyn = synPerCon[jtype*nType + itype];
                Float *local_g;
                Float *local_h;
                Float *ip;
                Size ngType;
                if (i < nE) {
                    local_g = local_gE;
                    local_h = local_hE;
                    ngType = ngTypeE;
                    ip = ipE;
                } else {
                    local_g = local_gI;
                    local_h = local_hI;
                    ngType = ngTypeI;
                    ip = ipI;
                }
				if (p > 0) {
					Float normed_std = square_root(p*(1-p)/nSyn);
					Float rand = normal(&localState);
					strength *= 1-p + normed_std*rand;
				}
				if (strength > 0) {
					Size block_ngType = (ngTypeE*nE + ngTypeI*nI)*bid;
					PosInt id;
        			if (i < nE) {
        			    ngType = ngTypeE;
						id = block_ngType + i*ngTypeE;
        			} else {
        			    ngType = ngTypeI;
						id = block_ngType + nE*ngTypeE + (i-nE)*ngTypeI;
        			}
                	for (PosInt ig=0; ig<ngType; ig++) {
						Float g = output_g[id + ig];
						Float h = output_h[id + ig];
						assert(g >= 0);
						assert(h >= 0);
						Float str = strength*ip[ig];
						local_g[ig] += str*g;
						local_h[ig] += str*h;
						assert(local_g[ig] >= 0);
						assert(local_h[ig] >= 0);
						//if (ipost == 8*1024+180) {
						//	printf("	it:%u, ipre: %u-%u, g=%f, str=%f, h=%f\n", it, bid, i, g, str, h);
						//}
                	}
				}
            }
            __syncwarp(); // may not be needed
			if (threadIdx.x >= nE && i >= nE) {
            	PosIntL gid = static_cast<PosIntL>((local_bid*nI + (i-nE))*nI + threadIdx.x-nE);
            	Float gap_strength = static_cast<Float>(gapMat[gid]);
				if (gap_strength > 0) {
					PosInt jtype;
    				//#pragma unroll (max_nType)
    				for (PosInt j=nTypeE; j<nType; j++) {
    				    if (i < tA[j]) {
    				        jtype = j;
    				        break;
    				    }
    				}
            	    Float v_pre = spikeTrain[ipre];
					v_pre = v_pre > 0? vThres[jtype]: v_pre;
					gap_s += gap_strength * v_pre;
					assert(!isnan(gap_s));
					//if (ipost == 960) {
					//	nGap++;
					//}
				}
			}
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
            	}
            	#pragma unroll (max_nLearnTypeE)
            	for (PosInt i=0; i<lE.n; i++) {
            	}
			*/
        }
    }

    PosInt id = blockIdx.x*blockDim.x + threadIdx.x;
    //#pragma unroll (ntimesE)
    #pragma unroll (max_ngTypeE)
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        PosInt gid = ig*gridDim.x*blockDim.x + id;
		assert(gE[gid] >= 0);
		assert(hE[gid] >= 0);
		//if (ipost == 8*1024+180) {
		//	printf("it:%u, gE0 -> %f\n", it, gE[gid]);
		//}
        gE[gid] += local_gE[ig];
        hE[gid] += local_hE[ig];
		//if (ipost == 8*1024+180) {
		//	printf("gE1 -> %f, gid = %u\n", gE[gid], gid);
		//}
		assert(gE[gid] >= 0);
		assert(hE[gid] >= 0);
    }
    //#pragma unroll (ntimesI)
    #pragma unroll (max_ngTypeI)
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        PosInt gid = ig*gridDim.x*blockDim.x + id;
		assert(gI[gid] >= 0);
		assert(hI[gid] >= 0);
        gI[gid] += local_gI[ig];
        hI[gid] += local_hI[ig];
		assert(gI[gid] >= 0);
		assert(hI[gid] >= 0);
    }
	if (threadIdx.x >= nE && InhGap) {
    	PosInt gap_ipost = blockIdx.x*nI + threadIdx.x-nE;
		gap[gap_ipost] = gap_s;
		assert(!isnan(gap_s));
		//if (ipost == 960) {
		//	assert(gap_ipost == 960 - 896);
		//	printf("nGap[960] = %u, gapS = %.3f\n", nGap, gap_s);
		//}
	}
}

__launch_bounds__(1024, 1)
__global__  // <<< nblock[partial], blockSize >>>
void recal_G_mat_nd_fast( // no distance involved for close-range connections, i.e., no delay in spike transport
        Float* __restrict__ spikeTrain, // [depth, nblock, nTypeHierarchy]
        PosInt* __restrict__ ipre, // [depth, nblock, nTypeHierarchy]
        Size* __restrict__ npre, // [depth, nblock, nTypeHierarchy]
        Float* __restrict__ output_g, // sparse [depth, nblock, blockSize]
        Float* __restrict__ output_h, // sparse [depth, nblock, blockSize]
        float* __restrict__ conMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        float* __restrict__ gapMat, // [nblock, nearNeighborBlock, nI, nI]
        Size* __restrict__ nNearNeighborBlock, // block_offset accounted for
        PosInt* __restrict__ neighborBlockId, // block_offset accounted for
        Float* __restrict__ gE, // [ngTypeE, nV1]
        Float* __restrict__ gI, // [ngTypeI, nV1] 
        Float* __restrict__ hE,
        Float* __restrict__ hI,
        Float* __restrict__ gap, // gap
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
        Float* __restrict__ synPerCon,
		Float* __restrict__ vThres,
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, Size nearNeighborBlock, Size nE, Size nI, Size nV1, int learning, PosInt block_offset, Size nType, Size nTypeE, Size nTypeI, Size nblock,
        LearnVarShapeE lE, LearnVarShapeQ lQ, PosInt iChunk, PosInt it, bool InhGap)
{
    // each thread is the post neuron that collects its presynaptic input conductances
    // initialize
	__shared__ Size tA[max_nType];
	__shared__ Float p[max_nType*max_nType];
	__shared__ Float nSyn[max_nType*max_nType];
	extern __shared__ char shared[];
    Size nb = nNearNeighborBlock[blockIdx.x];
    PosInt ipost = (block_offset+blockIdx.x)*blockDim.x + threadIdx.x;

    assert(nType*nType < blockDim.x);
	if (threadIdx.x < nType * nType) {
	    if (threadIdx.x < nType) {
    	    tA[threadIdx.x] = typeAcc[threadIdx.x];
        }
        p[threadIdx.x] = synFail[threadIdx.x];
        nSyn[threadIdx.x] = synPerCon[threadIdx.x];
	}
   
    Size *shared_npre = (Size*) shared;
    PosInt *shared_ipre = (PosInt*) (shared_npre + nb*2);
    Float *pre_sInfo = (Float*) shared_ipre; // reuse for gap
    Float *og = pre_sInfo + blockDim.x;
    Float *oh = og + blockDim.x; // TODO: not counting ngTypeE and ngTypeI, hope for 1/2 spikes at most
    assert(nb <= blockDim.x);
    if (threadIdx.x < nb) {
		PosInt local_bid = blockIdx.x*nearNeighborBlock + threadIdx.x;
		PosInt bid = neighborBlockId[local_bid];
        shared_npre[threadIdx.x] = npre[bid];
        shared_npre[nb + threadIdx.x] = npre[nblock + bid];
    }
    __syncthreads();

     PosInt itype;
    //#pragma unroll (max_nType)
    for (PosInt i=0; i<nType; i++) {
        if (threadIdx.x < tA[i]) {
            itype = i;
            break;
        }
    }
    curandStateMRG32k3a localState = rGenCond[ipost];

    Float ipE[max_ngTypeE]; // by being shared, can it accelerate the code?
    Float local_gE[max_ngTypeE];
    Float local_hE[max_ngTypeE];
    //#pragma unroll (max_ngTypeE)
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        local_gE[ig] = 0.0f;
        local_hE[ig] = 0.0f;
        if (itype*ngTypeE + ig >= nType*ngTypeE) {
            printf("ig=%u, nType=%u, ngTypeE=%u, itype =%u\n", ig, nType, ngTypeE, itype);
            assert(itype*ngTypeE + ig < nType*ngTypeE);
        }
        ipE[ig] = pE[itype*ngTypeE + ig];
    }
    Float ipI[max_ngTypeI];
    Float local_gI[max_ngTypeI];
    Float local_hI[max_ngTypeI];
    //#pragma unroll (max_ngTypeI)
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        local_gI[ig] = 0.0f;
        local_hI[ig] = 0.0f;
        ipI[ig] = pI[itype*ngTypeI + ig];
    }
	Float gap_s = 0.0;
    PosInt block_ngType = ngTypeE*nE + ngTypeI*nI;
    // TODO: cortical learning
    //Float trip_post[2*max_nLearnTypeE];
    //Float LTD_post[2*max_nLearnTypeE];
    for (PosInt ib = 0; ib < nb; ib++) {
		PosInt local_bid, bid;
        Size npreE = shared_npre[ib];
        Size npreI = shared_npre[nb+ib];
		local_bid = blockIdx.x*nearNeighborBlock + ib;
        if (npreE > 0 || npreI > 0) {
		    bid = neighborBlockId[local_bid];
        }
        if (npreE > 0) {// exc neighbors in the block fired
            if (ib > 0) __syncthreads();// sync shared memory save
            // branching is only loop-dependent, no thread-lock
            if (threadIdx.x < npreE) {
                PosInt id = block_ngType*bid + threadIdx.x;
                shared_ipre[threadIdx.x] = ipre[bid*blockDim.x + threadIdx.x];
                if (shared_ipre[threadIdx.x] >= nE) {
                    printf("ipre = %u, npreE = %u, bid = %u\n", shared_ipre[threadIdx.x], npreE, bid);
                    assert(shared_ipre[threadIdx.x] < nE);
                }
                //#pragma unroll (max_ngTypeE)
                for (PosInt ig=0; ig<ngTypeE; ig++) {
                    og[ig*npreE + threadIdx.x] = output_g[ig*npreE + id];
                    oh[ig*npreE + threadIdx.x] = output_h[ig*npreE + id];
                }
            }
            __syncthreads(); // sync dynamic shared memory save
            //#pragma unroll
            for (PosInt i=0; i<npreE; i++) {
                PosIntL mid = static_cast<PosIntL>((local_bid*blockDim.x + shared_ipre[i])*blockDim.x + threadIdx.x);
                Float strength = static_cast<Float>(conMat[mid]);
                if (strength > 0) {
		    	    PosInt jtype;	
    	    	    //#pragma unroll (max_nType)
    	    	    for (PosInt j=0; j<nTypeE; j++) {
    	    	        if (i < tA[j]) {
    	    	            jtype = j;
    	    	            break;
    	    	        }
    	    	    }
                    PosInt idx = jtype*nType + itype;
                    Float local_p = p[idx];
		    		Float normed_std = square_root(local_p*(1-local_p)/nSyn[idx]);
		    		Float rand = normal(&localState);
		    		strength *= 1-local_p + normed_std*rand;
		    	    if (strength > 0) {
                        //#pragma unroll (max_ngTypeE)
                	    for (PosInt ig=0; ig<ngTypeE; ig++) {
		    		    	Float str = strength*ipE[ig];
		    		    	local_gE[ig] += str*og[ig*npreE + i];
		    		    	local_hE[ig] += str*oh[ig*npreE + i];
							assert(local_gE[ig] >= 0);
							assert(local_hE[ig] >= 0);
                	    }
                    }
                }
            }
        }
        if (npreI > 0) {// inh neighbors in the block fired
            __syncthreads(); // sync dynamic shared memory save
            if (threadIdx.x < npreI) {
			    PosInt id = block_ngType*bid + npreE*ngTypeE + threadIdx.x;
                shared_ipre[threadIdx.x] = ipre[bid*blockDim.x + npreE + threadIdx.x];
                if (shared_ipre[threadIdx.x] < nE || shared_ipre[threadIdx.x] >= blockDim.x) {
                    printf("ipre = %u, npreI = %u, bid = %u\n", shared_ipre[threadIdx.x], npreI, bid);
                    assert(shared_ipre[threadIdx.x] >= nE && shared_ipre[threadIdx.x] < blockDim.x);
                }
                //#pragma unroll (max_ngTypeI)
                for (PosInt ig=0; ig<ngTypeI; ig++) {
                    og[ig*npreI + threadIdx.x] = output_g[ig*npreI + id];
                    oh[ig*npreI + threadIdx.x] = output_h[ig*npreI + id];
                }
            }
            __syncthreads(); // sync dynamic shared memory save
            //#pragma unroll
            for (PosInt i=0; i<npreI; i++) {
                PosIntL mid = static_cast<PosIntL>((local_bid*blockDim.x + shared_ipre[i])*blockDim.x + threadIdx.x);
                Float strength = static_cast<Float>(conMat[mid]);
                if (strength > 0) {
		    	    PosInt jtype;	
    	    	    //#pragma unroll (max_nTypeE)
    	    	    for (PosInt j=nTypeE; j<nType; j++) {
    	    	        if (i < tA[j]) {
    	    	            jtype = j;
    	    	            break;
    	    	        }
    	    	    }
                    PosInt idx = jtype*nType + itype;
                    Float local_p = p[idx];
		    		Float normed_std = square_root(local_p*(1-local_p)/nSyn[idx]);
		    		Float rand = normal(&localState);
		    		strength *= 1-local_p + normed_std*rand;
		    	    if (strength > 0) {
                        //#pragma unroll (max_ngTypeE)
                	    for (PosInt ig=0; ig<ngTypeI; ig++) {
		    		    	Float str = strength*ipI[ig];
		    		    	local_gI[ig] += str*og[ig*npreI + i];
		    		    	local_hI[ig] += str*oh[ig*npreI + i];
							assert(local_gI[ig] >= 0);
							assert(local_hI[ig] >= 0);
                	    }
                    }
                }
            }
        }
        if (InhGap) {
            __syncthreads(); // sync dynamic shared memory save
            if (threadIdx.x < nI) {
                pre_sInfo[threadIdx.x] = spikeTrain[bid*blockDim.x + nE + threadIdx.x];
            }
            __syncthreads(); // sync dynamic shared memory save
            for (PosInt i=0; i<nI; i++) {
                PosIntL gid = static_cast<PosIntL>((local_bid*nI + (i-nE))*nI + threadIdx.x-nE);
                Float strength = static_cast<Float>(gapMat[gid]);
			    if (strength > 0) {
                    Float v_pre = pre_sInfo[i];
			    	PosInt jtype;
    		    	//#pragma unroll (max_nType)
    		    	for (PosInt j=nTypeE; j<nType; j++) {
    		    	    if (i < tA[j]) {
    		    	        jtype = j;
    		    	        break;
    		    	    }
    		    	}
			    	v_pre = v_pre > 0? vThres[jtype]: v_pre;
			    	gap_s += strength * v_pre;
			    }
            }
        }
    }
    rGenCond[ipost] = localState;

    Float post_sInfo = spikeTrain[ipost];
    Float postNsp = flooring(post_sInfo);
    Float postTsp = post_sInfo>0? post_sInfo - postNsp: 1;
    Float lAvgE;
    if (learning != 3) {
        if (threadIdx.x < nE) {
            PosInt cPick = postNsp>0? 1:0;
            PosInt eid = (block_offset+blockIdx.x)*nE + threadIdx.x;
            lAvgE = vAvgE[2*eid+cPick];
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
            	}
            	#pragma unroll (max_nLearnTypeE)
            	for (PosInt i=0; i<lE.n; i++) {
            	}
			*/
        }
    }

    PosInt id = blockIdx.x*blockDim.x + threadIdx.x;
    //#pragma unroll (max_ngTypeE)
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        PosInt gid = ig*gridDim.x*blockDim.x + id;
		assert(gE[gid] >= 0);
		assert(hE[gid] >= 0);
		//if (ipost == 8*1024+180) {
		//	printf("it:%u, gE0 -> %f\n", it, gE[gid]);
		//}
        gE[gid] += local_gE[ig];
        hE[gid] += local_hE[ig];
		//if (ipost == 8*1024+180) {
		//	printf("gE1 -> %f, gid = %u\n", gE[gid], gid);
		//}
		assert(gE[gid] >= 0);
		assert(hE[gid] >= 0);
    }
    //#pragma unroll (max_ngTypeI)
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        PosInt gid = ig*gridDim.x*blockDim.x + id;
        gI[gid] += local_gI[ig];
        hI[gid] += local_hI[ig];
    }
	if (threadIdx.x >= nE && InhGap) {
    	PosInt gap_ipost = blockIdx.x*nI + threadIdx.x-nE;
		gap[gap_ipost] = gap_s;
		assert(!isnan(gap_s));
	}
}
