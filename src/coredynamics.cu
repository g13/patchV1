#include "coredynamics.cuh"
extern surface<void, cudaSurfaceType2DLayered> LGNspikeSurface;

__launch_bounds__(1024,2)
__global__
void logRand_init(Float* __restrict__ logRand,
                  Float* __restrict__ lTR,
                  PosInt* __restrict__ LGN_idx,
                  PosInt* __restrict__ LGN_idy,
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
		lTR[id] = logRand[id] * rand;
        int x = static_cast<int>(LGN_idx[id]);
        int y = static_cast<int>(LGN_idy[id]);
        Float value = -1.0; // this is needed, otherwise surf2DLayeredwrite will raise runtime error
        surf2DLayeredwrite(value, LGNspikeSurface, 4*x, y, 0);
        #pragma unroll max_ngTypeFF
        for (int i=0; i<nFF; i++) {
            surf2DLayeredwrite(value, LGNspikeSurface, 4*x, y, 1+3*i+0);
            surf2DLayeredwrite(value, LGNspikeSurface, 4*x, y, 1+3*i+1);
            surf2DLayeredwrite(value, LGNspikeSurface, 4*x, y, 1+3*i+2);
        }
	}
}

__device__
__forceinline__
void evolve_gLGN(ConductanceShape &cond, Float &g, Float &h, Float sInfo, Float f, Float dt, PosInt ig) {
    Float nsp = flooring(sInfo); // integer part: #spikes - 1
    Float tsp = sInfo - nsp; // decimal part: normalized mean tsp
    cond.compute_single_input_conductance(g, h, f*nsp, dt*(1-tsp), ig);
}

__device__
__forceinline__
Float step(LIF* lif, Float dt, Float tRef, PosInt id, Float gE, Float gI) {
    lif->spikeCount = 0;
    Float sInfo = 0.0;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            lif->recompute_v0(dt);
            #ifdef DEBUG
                if (id == 0 || id == 768) {
                    printf("backed\n");
                }
            #endif
        }
        lif->implicit_rk2(dt);
        while (lif->v > vT && lif->tBack < dt) { // forbids firing exactly at the end of the timestep, 
            // crossed threshold
            lif->compute_spike_time(dt); 
            sInfo += lif->tsp;
            lif->spikeCount++;
            lif->tBack = lif->tsp + tRef;
            #ifdef DEBUG
                if (id == 0 || id == 768) {
                    printf("#%u spiked at %f, to come back at %f\n", id, lif->tsp, lif->tBack);
                }
            #endif
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
        if (id == 2) {
            printf("#%u v = %f, gI = %f\n", id, lif->v, gI);
        }
        if (lif->v < vI) {
    		printf("#%i implicit rk2 is A-Stable! something is off gE1 = %f, gI1 = %f, v = %f, v0 = %f, a0 = %f, b0 = %f, a1 = %f, b1 = %f\n", id, gE, gI, lif->v, lif->v0, lif->a0, lif->b0, lif->a1, lif->b1);
        }   
    #endif
    if (lif->spikeCount > 0) sInfo /= lif->spikeCount*dt; //decimal part: tsp (normalize by dt)
    sInfo += lif->spikeCount; // integer part: nsp
    __syncwarp();
    return sInfo;
}

__device__
__forceinline__
void LIF::implicit_rk2(Float dt) {
    v = impl_rk2(dt, a0, b0, a1, b1, v0);
}

__device__
__forceinline__
void LIF::compute_spike_time(Float dt, Float t0) {
    tsp = comp_spike_time(v, v0, dt, t0);
}

__device__
__forceinline__
void LIF::recompute(Float dt, Float t0) {
    Float rB = dt/(tBack-t0) - 1; 
    Float denorm = 2 + a1*dt;
    Float A = (2 - a0*dt)/denorm;
    Float B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
    v = A*v0 + B;
}

__device__ 
__forceinline__
void LIF::recompute_v(Float dt, Float t0) {
    Float rB = dt/(tBack-t0) - 1; 
    Float denorm = 2 + a1*dt;
    Float A = (2 - a0*dt)/denorm;
    Float B = (b0 + b1)*dt/denorm;
    v = recomp_v(A, B, rB);
}

__device__ 
__forceinline__
void LIF::recompute_v0(Float dt, Float t0) {
    Float rB = dt/(tBack-t0) - 1; 
    Float denorm = 2 + a1*dt;
    Float A = (2 - a0*dt)/denorm;
    Float B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
}

__device__ 
__forceinline__
void LIF::set_p0(Float gE, Float gI, Float gL) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

__device__ 
__forceinline__
void LIF::set_p1(Float gE, Float gI, Float gL) {
    a1 = get_a(gE, gI, gL);
    b1 = get_b(gE, gI, gL); 
}

__device__ 
__forceinline__
void LIF::reset_v() {
    v = vL;
}

void recal_G_vec(
        std::vector<std::vector<std::vector<Float>>> &spikeTrain, std::vector<std::vector<Size>> &trainDepth, std::vector<std::vector<PosInt>> &currentTimeSlot,
        std::vector<Size> &nVec,  std::vector<std::vector<PosInt>> &vecID, std::vector<std::vector<Float>> &conVec, std::vector<std::vector<Float>> &delayVec,
        Float gE[], Float gI[], Float hE[], Float hI[],
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt block_offset, Size nE, Size nV1, Float speedOfThought, Size chunkSize) 
{
    Float local_gE[max_ngTypeE];
    Float local_hE[max_ngTypeE];
    Float local_gI[max_ngTypeI];
    Float local_hI[max_ngTypeI];
    PosInt i0 = block_offset*blockSize;
    for (PosInt i=0; i<chunkSize*blockSize; i++) {
        // initialize
        if (nVec[i] == 0) continue;
        #pragma unroll max_ngTypeE
        for (PosInt ig=0; ig<ngTypeE; ig++) {
            local_gE[ig] = 0.0f;
            local_hE[ig] = 0.0f;
        }
        #pragma unroll max_ngTypeI
        for (PosInt ig=0; ig<ngTypeI; ig++) {
            local_gI[ig] = 0.0f;
            local_hI[ig] = 0.0f;
        }
        #pragma unroll 4
        for (PosInt j = 0; j < nVec[i0+i]; j++) {
            PosInt ipre = vecID[i0+i][j];
            PosInt tid = ipre%blockSize;

            Float strength = conVec[i0+i][j];
            Float time2post = delayVec[i0+i][j]/speedOfThought;
            Float *local_g;
            Float *local_h;
            Size ngType;
            ConductanceShape *cond;
            // TODO direct output to g and h
            if (tid < nE) {
                local_g = local_gE;
                local_h = local_hE;
                ngType = ngTypeE;
                cond = &condE;
            } else {
                local_g = local_gI;
                local_h = local_hI;
                ngType = ngTypeI;
                cond = &condI;
            }
            PosInt it2post = static_cast<PosInt>(ceiling(time2post/dt));
            time2post = it2post*dt - time2post;
            assert(time2post>=0);
            assert(time2post<dt);
            PosInt k0 = currentTimeSlot[i0+i][j] - it2post + trainDepth[i0+i][j];
            currentTimeSlot[i0+i][j] = (currentTimeSlot[i0+i][j]+1)%trainDepth[i0+i][j];
            #pragma unroll 2
            for (PosInt k = 0; k < 2; k++) {
                Float sInfo = spikeTrain[i0+i][j][k0+k];
                if (sInfo >= 0) {
                    Float nsp = flooring(sInfo);
                    Float tsp = (sInfo - nsp + k)*dt - time2post;
                    if (tsp < dt && tsp >= 0){
                        #pragma unroll max_ngType
                        for (PosInt ig=0; ig<ngType; ig++) {
                            cond->compute_single_input_conductance(local_g[ig], local_h[ig], strength*nsp, dt-tsp, ig);
                        }
                    }
                }
                //__syncwarp(); // may not be needed
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
__launch_bounds__(1024,2)
__global__ 
void compute_V_collect_spike_learnFF(
        Float* __restrict__ v,
        Float* __restrict__ gFF, // not in chunks
        Float* __restrict__ hFF,
        Float** __restrict__ gE, // in chunks
        Float** __restrict__ gI,
        Float** __restrict__ hE,
        Float** __restrict__ hI,
        Float* __restrict__ tBack,
        Float* __restrict__ spikeTrain, //         [                depth, nblock, blockSize  ]
        Float* __restrict__ vLTD_FF_E, //    post, [nLearnTypeFF_E,        nblock, nE         ]
        Float* __restrict__ vTrip_FF_E, //   post, [nLearnTypeFF_E,        nblock, nE         ]
        Float* __restrict__ vLTD_FF_I, //    post, [nLearnTypeFF_I,        nblock, nI         ]
        Float* __restrict__ vTrip_FF_I, //   post, [nLearnTypeFF_I,        nblock, nI         ]
        Float* __restrict__ vAvgE, //        post, [                       nblock, nE,       2]
        Float* __restrict__ vAvgI, //        post, [                       nblock, nI,       2]
        Float* __restrict__ vLTP_E, //        pre, [nLearnTypeE,    depth, nblock, nE,       2]
        Float* __restrict__ vLTD_E, //       post, [nLearnTypeE,           nblock, nE,       2]
        Float* __restrict__ vTripE, //       post, [nLearnTypeE,           nblock, nE,       2]
        Float* __restrict__ vSTDP_QE,  //  E post, [nLearnTypeQ,           nblock, nE        2]
        Float* __restrict__ vSTDP_QI,  //   I pre, [nLearnTypeQ,    depth, nblock, nI,       2]
        Size* __restrict__ nLGN,
        Float* __restrict__ sLGN,
        PosInt* __restrict__ LGN_idx,
        PosInt* __restrict__ LGN_idy,
        PosInt currentTimeSlot, Size trainDepth, Size max_nLGN, Size ngTypeFF, Size ngTypeE, Size ngTypeI, ConductanceShape condFF, ConductanceShape condE, ConductanceShape condI, Float dt, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1, PosIntL seed,
        LearnVarShapeFF_E_pre  learnE_pre,  LearnVarShapeFF_I_pre  learnI_pre, 
        LearnVarShapeFF_E_post learnE_post, LearnVarShapeFF_I_post learnI_post, 
        LearnVarShapeE learnE, LearnVarShapeQ learnQ)
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
    LIF lif(v[tid], tBack[tid]);
    Float gL, tRef;
    if (threadIdx.x < nE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    /* set a0 b0 and a1 b1 */
    Float gE_t0 = 0.0;
	Float gE_t1 = 0.0;
    // cond FF
    //#pragma unroll (MAX_NGTYPE_FF)
    #pragma unroll (max_ngTypeFF) //(ntimesFF)
    for (PosInt ig=0; ig<ngTypeFF; ig++) {
        PosInt gid = nV1*ig + tid; // not in chunks
        Float g = gFF[gid];
        Float h = hFF[gid];
        gE_t0 += g;
        // conductance of the end of the last time step
        condFF.decay_conductance(g, h, dt, ig); //  decayed to the end of the current step
        // Get LGN input
    	Size m = nLGN[tid];

        #pragma unroll (4)
        for (PosInt i = 0; i<m; i++) {
            PosInt lid = tid*max_nLGN + i;
            Float f = sLGN[lid];
            int x = static_cast<int>(LGN_idx[lid]);
            int y = static_cast<int>(LGN_idy[lid]);
            Float sInfo;
            surf2DLayeredread(&sInfo, LGNspikeSurface, 4*x, y, 0);
            if (sInfo >= 0.0) {
                evolve_gLGN(condFF, g, h, sInfo, f, dt, ig);
            }
        }
        gE_t1 += g;
        gFF[gid] = g;
        hFF[gid] = h;
    }
    // cond E 
    //#pragma unroll (MAX_NGTYPE_E)
    #pragma unroll (max_ngTypeE) 
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        PosInt gid = chunkSize*ig + cid;
        Float g = gE[iChunk][gid];
        Float h = hE[iChunk][gid];
        gE_t0 += g;
        condE.decay_conductance(g, h, dt, ig); 
        gE_t1 += g;
        gE[iChunk][gid] = g;
        hE[iChunk][gid] = h;
    }
    // cond I 
    Float gI_t0 = 0.0;
	Float gI_t1 = 0.0;
    //#pragma unroll (MAX_NGTYPE_I)
    //#pragma unroll (ntimesI)
    #pragma unroll (max_ngTypeI)
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        PosInt gid = chunkSize*ig + cid;
        Float g = gI[iChunk][gid];
        Float h = hI[iChunk][gid];
        gI_t0 += g;
        condI.decay_conductance(g, h, dt, ig); 
        gI_t1 += g;
        gI[iChunk][gid] = g;
        hI[iChunk][gid] = h;
    }
    lif.set_p0(gE_t0, gI_t0, gL);
    lif.set_p1(gE_t1, gI_t1, gL);
    /* evolve g to t+dt with ff input only */
    // step
    Float sInfo = step(&lif, dt, tRef, /*the last 3 args are for deugging*/ tid, gE_t1, gI_t1);
    spikeTrain[nV1*currentTimeSlot + tid];
	v[tid] = lif.v;
    tBack[tid] = lif.tBack;
    if (learning)
        Float nsp = flooring(sInfo);
        Float tsp = sInfo>0? sInfo - nsp: 1;
        // will compute ff learning, first row at start of time step, second row at tsp
        Float lFF[2*2*max_nLearnTypeFF]; // row 0: start, row 1: sp
        Float lAvg[2];
        // only store
        Float lE[3*max_nLearnTypeE];
        Float lQ[max_nLearnTypeQ];
        if (nsp > 0) {
            // read ff lVar
            PosInt eid = nE*blockIdx.x+threadIdx.x;
            if (learning < 4) { 
                if (threadIdx.x < nE) {
                    #pragma unroll max_nLearnTypeFF_E
                    for (PosInt i=0; i<learnE_post.n; i++) {
                        lFF[2*i+0] =  vLTD_FF_E[nE*nblock*i + eid];
                        lFF[2*i+1] = vTrip_FF_E[nE*nblock*i + eid];
                    }
                } else {
                    if (learnI_post.n) {
                        PosInt iid = nI*blockIdx.x+threadIdx.x-nE;
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt i=0; i<learnI_post.n; i++) {
                            lFF[2*i+0] =  vLTD_FF_I[nI*nblock*i + iid];
                            lFF[2*i+1] = vTrip_FF_I[nI*nblock*i + iid];
                        }
                        lAvg[0] = vAvgI[iid];
                    }
                }
            }
            // read cortical lVar
            if (learning !=3) { // E and Q
                // E
                if (threadIdx.x < nE) {
                    #pragma unroll max_nLearnTypeE
                    for (PosInt i=0; i<learnE.n; i++) {
                        lE[3*i+0] = vLTP_E[(nE*nblock*trainDepth*i + nE*nblock*currentTimeSlot + eid)*2];
                        lE[3*i+1] = vLTD_E[(nE*nblock*i + eid)*2];
                        lE[3*i+2] = vTripE[(nE*nblock*i + eid)*2];
                    }
                    // Q_E
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) {
                        lQ[i] = vSTDP_QE[(nE*nblock*i + eid)*2];
                    }
                } else {
                    // Q_I
                    PosInt iid = nI*(nblock*currentTimeSlot + blockIdx.x) + threadIdx.x-nE;
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) {
                        lQ[i] = vSTDP_QI[(nI*nblock*trainDepth*i + iid)*2];
                    }
                }
            }
            // read sp average
            if (threadIdx.x < nE) {
                lAvg[0] = vAvgE[eid*2];
            }
            // compute ff post vars' decay till tsp
            if (learning < 4) {
                if (threadIdx.x < nE) {
                    #pragma unroll max_nLearnTypeFF_E
                    for (PosInt i=0; i<learnE_post.n; i++) {
                        lFF[2*max_nLearnTypeFF + 2*i+0] = lFF[2*i+0];
                        lFF[2*max_nLearnTypeFF + 2*i+1] = lFF[2*i+1];
                    }
                    #pragma unroll max_nLearnTypeFF_E
                    for (PosInt i=0; i<learnE_post.n; i++) {
                        decay(lFF[2*max_nLearnTypeFF + 2*i+0], learnFF_E.tau[2*i+0], tsp);
                        decay(lFF[2*max_nLearnTypeFF + 2*i+1], learnFF_E.tau[2*i+1], tsp);
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
                            decay(lFF[2*max_nLearnTypeFF + 2*i+0], learnFF_I.tau[2*i+0], tsp);
                            decay(lFF[2*max_nLearnTypeFF + 2*i+1], learnFF_I.tau[2*i+1], tsp);
                        }
                        lAvg[1] = lAvg[0];
                        decay(lAvg[1], learnFF_I.tau[2*learnFF_I.n], tsp);
                    }
                }
            }
            if (threadIdx.x < nE) {
                lAvg[1] = lAvg[0];
                decay(lAvg[1], learnE_post.tau[2*learnE_post.n], tsp);
            }
            // compute and store E and Q
            if (learning !=3) { 
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
                         vLTP_E[(nE*nblock*trainDepth*i + nE*nblock*currentTimeSlot + eid)*2 + 1] = lE[3*i+0];
                         vLTD_E[(nE*nblock*i + eid)*2 + 1] = lE[3*i+1];
                         vTripE[(nE*nblock*i + eid)*2 + 1] = lE[3*i+2];
                    }
                    vAvgE[2*eid+1] = lAvg[1];
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) { // store to the second slot of the array
                        vSTDP_QE[(nE*nblock*i + eid)*2 + 1] =  lQ[i];
                    }
                } else {
                    PosInt iid = nI*(nblock*currentTimeSlot + blockIdx.x) + threadIdx.x-nE;
                    #pragma unroll max_nLearnTypeQ
                    for (PosInt i=0; i<learnQ.n; i++) { // store to the second slot of the array
                        vSTDP_QI[(nI*nblock*trainDepth*i + iid)*2 + 1] =  lQ[i];
                    }
                }
            }
        }
        // learn LGN connection and update LGN lVars
        if (learning < 4) { 
            // learn
            for (PosInt i = 0; i<m; i++) {
                PosInt lid = tid*max_nLGN + i;
                Float f = sLGN[lid];
                int x = static_cast<int>(LGN_idx[lid]);
                int y = static_cast<int>(LGN_idy[lid]);
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
                        for (PosInt i=0; i<learnE_pre.n; i++) {
                            Float A_LTD = learnFF_E.A_ratio[i] * learnFF_E_pre.tau[i] * lAvg[cPick] * lAvg[cPick]/ 8000.0;
                            //Float A_LTD = learnFF_E.A_LTP[i]; TODO: alternative homeostatic design
                            f -= if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*i+0], learnE_pre.tau[2*i+0], delta_t) * A_LTD * nsp_FF;
                        }
                    } else {
                        #pragma unroll max_nLearnTypeFF_I
                        for (PosInt i=0; i<learnI_pre.n; i++) {
                            Float A_LTD = learnI_post.A_ratio[i] * learnI_pre.tau[i] * lAvg[cPick] * lAvg[cPick]/ 8000.0;
                            f -= if_decay(lFF[cPick*max_nLearnTypeFF*2 + 2*i+0], learnI_pre.tau[2*i+0], delta_t) * A_LTD * nsp_FF;
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
                        for (PosInt i=0; i<learnE_pre.n; i++) {
                            Float lFF_pre;
                            surf2DLayeredread(&lFF_pre, 4*x, y, 1+3*i+fPick);
                            f += if_decay(lFF_pre, learnE_pre.tau[i], delta_t) * lFF[max_nLearnTypeFF*2 + 2*i+1] * learnE_post.A_LTP[i] * nsp;
                        }
                    } else {
                        if (learnI_pre.n) {
                            #pragma unroll max_nLearnTypeFF_I
                            for (PosInt i=0; i<learnI_pre.n; i++) {
                                Float lFF_pre;
                                surf2DLayeredread(&lFF_pre, 4*x, y, 1+3*i+fPick);
                                f += if_decay(lFF_pre, learnI_pre.tau[i], delta_t) * lFF[max_nLearnTypeFF*2 + 2*i+1] * learnI_post.A_LTP[i] * nsp;
                            }
                        }
                    }
                }
                sLGN[lid] = f;
            }
            // update FF vars, lAvg to be updated after cortical learning
            Float delta_t = 1;
            PosInt cPick = nsp > 0? 1: 0;
            if (nsp > 0) { 
                delta_t -= tsp;
            }
            delta *= dt;
            if (threadIdx.x < nE) {
                PosInt eid = nE*blockIdx.x+threadIdx.x;
                #pragma unroll max_nLearnTypeFF_E
                for (PosInt i=0; i<nLearnTypeFF_E; i++) {
                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+0], learnFF_E.tau[2*i+0], delta_t);
                    decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+1], learnFF_E.tau[2*i+1], delta_t);
                }
            } else {
                if (learnI_post.n) {
                    PosInt iid = nI*blockIdx.x+threadIdx.x-nE;
                    #pragma unroll max_nLearnTypeFF_I
                    for (PosInt i=0; i<nLearnTypeFF_I; i++) {
                        decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+0], learnFF_I.tau[2*i+0], delta_t);
                        decay(lFF[cPick*2*max_nLearnTypeFF + 2*i+1], learnFF_I.tau[2*i+1], delta_t);
                    }
                }
            }
            // store LGN lVars 
            if (threadIdx.x < nE) {
                PosInt eid = nE*blockIdx.x+threadIdx.x;
                #pragma unroll max_nLearnTypeFF_E
                #pragma unroll max_nLearnTypeFF_E
                for (PosInt i=0; i<learnE_post.n; i++) {
                    vLTD_FF_E[nE*nblock*i + eid]  = lFF[cPick*2*max_nLearnTypeFF + 2*i+0];
                    vTrip_FF_E[nE*nblock*i + eid] = lFF[cPick*2*max_nLearnTypeFF + 2*i+1];
                }
            } else {
                if (learnI_post.n) {
                    PosInt iid = nI*blockIdx.x+threadIdx.x-nE;
                    #pragma unroll max_nLearnTypeFF_I
                    for (PosInt i=0; i<learnI_post.n; i++) {
                        vLTD_FF_I[nI*nblock*i + iid]  = lFF[cPick*2*max_nLearnTypeFF + 2*i+0];
                        vTrip_FF_I[nI*nblock*i + iid] = lFF[cPick*2*max_nLearnTypeFF + 2*i+1];
                    }
                }
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
        //Float* __restrict__ PreVar_V1,
        //Float* __restrict__ PostVar,
        //Float* __restrict__ PripVar,
        //Float* __restrict__ FrVar,
        //Float* __restrict__ qVar,
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt currentTimeSlot, Size trainDepth, Size nearNeighborBlock, Size nE, Size nV1, Float speedOfThought) 
{
    // each thread is the post neuron that collects its presynaptic input conductances
    // initialize
    Float local_gE[MAX_NGTYPE_E];
    Float local_hE[MAX_NGTYPE_E];
    #pragma unroll (max_ngTypeE)
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        local_gE[ig] = 0.0f;
        local_hE[ig] = 0.0f;
    }
    Float local_gI[MAX_NGTYPE_I];
    Float local_hI[MAX_NGTYPE_I];
    //#pragma unroll (ntimesI)
    #pragma unroll (max_ngTypeI)
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        local_gI[ig] = 0.0f;
        local_hI[ig] = 0.0f;
    }
    // TODO: cortical learning
    //Float trip_post[2*max_nLearnTypeE];
    //Float LTD_post[2*max_nLearnTypeE];
    //PosInt ipost = spikeTrain[nV1*pid + currentTimeSlot] > 0? 1:0;
    
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
            if (strength != 0) {
                Float LTP_pre[max_nLearnTypeE];
                Float Q_pre[max_nLearnTypeQ];
                Float time2post = delayMat[mid]/speedOfThought;
                Float *local_g;
                Float *local_h;
                Size ngType;
                ConductanceShape *cond;
                if (i < nE) {
                    local_g = local_gE;
                    local_h = local_hE;
                    ngType = ngTypeE;
                    cond = &condE;
                } else {
                    local_g = local_gI;
                    local_h = local_hI;
                    ngType = ngTypeI;
                    cond = &condI;
                }
                PosInt it2post = static_cast<PosInt>(ceiling(time2post/dt));
                time2post = it2post*dt - time2post;
                assert(time2post>=0);
                assert(time2post<dt);
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
                    // from older to newer
                    PosInt isp = nV1*((j0 + j)%trainDepth) + ipre;
                    Float sInfo = spikeTrain[isp];
                    if (sInfo >= 0) { // could fire at the instant t = t_i
                        Float nsp = flooring(sInfo);
                        Float tsp = (sInfo - nsp + j)*dt - time2post;
			    	    // DEBUG
                        PosInt id = neighborBlockId[blockIdx.x*nearNeighborBlock]*blockDim.x + threadIdx.x;
			    	    if (ipre == 0 && id == 1) {
			    	    	printf("0: %f -> 1 %f, from %u + %u -> %u, %u, %1.7e\n", sInfo, tsp, j0, j, currentTimeSlot, it2post, time2post/dt);
                            for (PosInt k = 0; k<trainDepth; k++) {
                                printf("%f,", spikeTrain[nV1*k+ipre]);
                            }
                            printf("\n");
			    	    }//
                        if (tsp < dt && tsp >=0) {
                            for (PosInt ig=0; ig<ngType; ig++) {
                                cond->compute_single_input_conductance(local_g[ig], local_h[ig], strength*nsp, dt-tsp, ig);
                            }
                        }
                    }
                    //__syncwarp(); // may not be needed
                }
            }
            __syncwarp(); // may not be needed
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
