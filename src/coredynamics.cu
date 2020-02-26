#include "coredynamics.h"
extern surface<void, cudaSurfaceType2D> LGNspikeSurface;

__global__
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

__device__
__forceinline__
void evolve_gLGN(ConductanceShape &cond, Float &g, Float &h, Float sInfo, Float dt, PosInt ig) {
    Float nsp = flooring(sInfo); // integer part: #spikes - 1
    Float tsp = sInfo - nsp; // decimal part: normalized mean tsp
    nsp += 1;
    cond.compute_single_input_conductance(g, h, sLGN*nsp, dt*(1-tsp), ig);
}

__device__
__forceinline__
Float step(LIF* lif, Float dt, Float tRef, Float *tsp, PosInt id, Float gE, Float gI) {
    lif->spikeCount = 0;
    Float sInfo = 0.0;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            lif->recompute_v0(dt);
        }
        lif->implicit_rk2(dt);
        while (lif->v > vT && lif->tBack < dt) { // forbids firing exactly at the end of the timestep, 
            // crossed threshold
            lif->compute_spike_time(dt); 
            sInfo += lif->tsp;
            lif->spikeCount++;
            lif->tBack = lif->tsp + tRef;
            if (lif->tBack < dt) {
                // refractory period ended during dt
                lif->recompute(dt);
            }
        }
    }
    if (lif->spikeCount > 0) {
        sInfo = sInfo / (lif->spikeCount * dt);
    }
    // access only the first element [id*depth + currentTimeSlot]
    tsp[0] = sInfo + lif->spikeCount-1;
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

__launch_bounds(1024,2)
__global__ 
void compute_V_collect_spike(
        Float* __restrict__ v,
        Float* __restrict__ gE,
        Float* __restrict__ gI,
        Float* __restrict__ hE,
        Float* __restrict__ hI,
        Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        Float* __restrict__ tBack,
        Float* __restrict__ sLGN,
        Float* __restrict__ LGN_idx,
        Float* __restrict__ LGN_idy,
        PosInt* __restrict__ blockVready,
        curandStateMRG32k3a* __restrict__ stateE,
        curandStateMRG32k3a* __restrict__ stateI,
        PosInt currentTimeSlot, Size trainDepth, Size max_nLGN, Size ngTypeE, Size ngTypeI, Size ngType, ConductanceShape condE, ConductanceShape condI, Float dt, Size networkSize, Size mE, PosIntL seed)
{
    PosInt id = blockIdx.x * blockDim.x + threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    // TODO: load individual gl, tref
    LIF lif(v[id], tBack[id]);
    Float gL, tRef;
    if (threadIdx.x < mE) {
        tRef = tRef_E;
        gL = gL_E;
    } else {
        tRef = tRef_I;
        gL = gL_I;
    }
    /* set a0 b0 for the first step */
    Float gI_t;
    Float gE_t;
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
    /* evolve g to t+dt with ff input only */
    unsigned int gid;
    gE_t = 0.0f;
    // m nLGN
    // n networkSize
    Size m = nLGN[id];
    #pragma unroll
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        gid = networkSize*ig + id;
        Float g = gE[gid];
        Float h = hE[gid];
        // conductance of the end of the last time step
        condE.decay_conductance(g, h, dt, ig); //  decayed to the end of the current step
        // Get LGN input
        for (Size i = 0; i<m; i++) {
            PosInt lid = id*max_nLGN + i;
            Float sLGN = LGN_V1_s[lid];
            PosInt x = LGN_V1_surface[lid];
            PosInt y = LGN_V1_surface[id*max_nLGN + lid];
            Float sInfo;
            surf2Dread(&sInfo, LGNspikeSurface, 4*x, y);
            if (sInfo >= 0.0) {
                evolve_gLGN(g, h, sInfo, dt, ig);
            }
        }
        gE_t += g;
        gE[gid] = g;
        hE[gid] = h;
    }
    gI_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeI; ig++) {
        gid = networkSize*ig + id;
        Float g_i = gI[gid];
        Float h_i = hI[gid];
        condI.decay_conductance(g, h, dt, ig); 
        gI_t += g;
        gI[gid] = g;
        hI[gid] = h;
    }
    lif.set_p1(gE_t, gI_t, gL);
    // step
    step(&lif, dt, tRef, spikeTrain+id*trainDepth+currentTimeSlot /*the last 2 args are for deugging*/ id, gE_t, gI_t);
    if (lif.v < vI) {
#ifdef DEBUG
		printf("#%i something is off gE = %f, gI = %f, v = %f\n", id, gE_t, gI_t, lif.v);
#endif
        lif.v = vI;
    }   
	v[id] = lif.v;
    tBack[id] = lif.tBack;
}

__launch_bounds__(1024, 2)
__global__  // <<< nblock[partial], blockSize >>>
void recal_G(
        Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        Float* __restrict__ preMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        Float* __restrict__ delayMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        Size* __restrict__ nNeighborBlock,
        Float* __restrict__ gE, // [ngTypeE, networkSize]
        Float* __restrict__ gI, // [ngTypeI, networkSize] 
        Float* __restrict__ hE,
        Float* __restrict__ hI,
        PosInt* __restrict__ blockGready,
        ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt block_offset, PosInt currentTimeSlot, Size trainDepth, Size nearNeighborBlock, Size networkSize, Size mE, Float speedOfThought) 
{
    // each thread is the post neuron that collects its presynaptic input conductances
    // initialize
    Float *local_gE = new Float[ngTypeE];
    Float *local_hE = new Float[ngTypeE];
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        local_gE[ig] = 0.0f;
        local_hE[ig] = 0.0f;
    }
    Float *local_gI = new Float[ngTypeI];
    Float *local_hI = new Float[ngTypeI];
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        local_gI[ig] = 0.0f;
        local_hI[ig] = 0.0f;
    }

    for (PosInt ib = 0; ib < nNeighborBlock[blockIdx.x]; ib++) {
        // check for old spikes
        for (PosInt i=0; i<blockSize; i++) {
            // access each presynaptic neurons in stride
            PosIntL mid = ((blockIdx.x*nearNeighborBlock + ib)*blockSize + i)*blockSize + threadIdx.x;
            Float strength = preMat[mid];
            Float distance = delayMat[mid];
            Float *local_g;
            Size ngType;
            ConductanceShape *cond;
            if (i > mE) {
                local_g = local_gI;
                ngType = ngTypeI;
                cond = &condE;
            } else {
                local_g = local_gE;
                ngType = ngTypeE;
                cond = &condI;
            }
            for (PosInt j=1; j<trainDepth+1; j++) { // older to newer
                PosInt isp = ipre*trainDepth + (currentTimeSlot + j) % trainDepth;
                Float sInfo = spikeTrain[isp];
                if (sInfo >= 0) {
                    Float nsp = flooring(sInfo);
                    Float tsp = (sInfo - nsp + (trainDepth-j))*dt - distance/speedOfThought;
                    if (tsp >= dt) continue; // this spike has passed
                    else if (tsp < 0) break; // break at the first spike that did not arrive
                    nsp += 1;
                    for (PosInt ig=0; ig<ngType; ig++) {
                        cond->compute_single_input_conductance(local_g, local_h, strength*nsp, dt*(1-tsp), ig);
                    }
                }
                //__syncwarp(); // may not be needed
            }
        }
    }

    PosInt id = blockIdx.x*blockSize + threadIdx.x;
    for (PosInt ig=0; ig<ngTypeE; ig++) {
        PosInt gid = ig*gridDim.x*blockSize + id;
        gE[gid] = local_gE[ig];
        hE[gid] = local_hE[ig];
    }
    delete [] local_gE;
    delete [] local_gI;
    for (PosInt ig=0; ig<ngTypeI; ig++) {
        PosInt gid = ig*gridDim.x*blockSize + id;
        gI[gid] = local_gI[ig];
        hI[gid] = local_hI[ig];
    }
    delete [] local_hE;
    delete [] local_hI;
}

void recal_Gvec(
        Float spikeTrain[],
        vector<Size> &nVec,  vector<PosInt> &vecID, vector<Float> &conVec, vector<Float> &delayVec,
        Float gE[], Float gI[], Float hE[], Float hI[],
        ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt block_offset, PosInt currentTimeSlot, Size trainDepth, Size nearNeighborBlock, Size networkSize, Size mE, Float speedOfThought, Size chunkSize, Size maxChunkSize) 
{
    Float *local_gE = new Float[ngTypeE];
    Float *local_hE = new Float[ngTypeE];
    Float *local_gI = new Float[ngTypeI];
    Float *local_hI = new Float[ngTypeI];
    PosInt i0 = block_offset*blockSize;
    for (PosInt i=0; i<chunkSize*blockSize; i++) {
        // initialize
        for (PosInt ig=0; ig<ngTypeE; ig++) {
            local_gE[ig] = 0.0f;
            local_hE[ig] = 0.0f;
        }
        for (PosInt ig=0; ig<ngTypeI; ig++) {
            local_gI[ig] = 0.0f;
            local_hI[ig] = 0.0f;
        }
        for (PosInt j = 0; j < nVec[i0+i]; j++) {
            PosIntL ipre = vecID[i0+i][j];
            PosInt tid = ipre%blockSize;

            Float strength = conVec[j];
            Float distance = delayVec[j];
            Float *local_g;
            Size ngType;
            ConductanceShape *cond;
            // TODO direct output
            if (tid > mE) {
                local_g = gI;
                ngType = ngTypeI;
                cond = &condE;
            } else {
                local_g = local_gE;
                ngType = ngTypeE;
                cond = &condI;
            }
            for (k = 0; k < trainDepth; k++) {
                PosInt isp = ipre*trainDepth + (currentTimeSlot + k) % trainDepth;
                Float sInfo = spikeTrain[isp];
                if (sInfo >= 0) {
                    Float nsp = flooring(sInfo);
                    Float tsp = (sInfo - nsp + (trainDepth-j))*dt - distance/speedOfThought;
                    if (tsp >= dt) continue; // this spike has passed
                    else if (tsp < 0) break; // break at the first spike that did not arrive
                    nsp += 1;
                    for (PosInt ig=0; ig<ngType; ig++) {
                        cond->compute_single_input_conductance(local_g, local_h, strength*nsp, dt*(1-tsp), ig);
                    }
                }
                //__syncwarp(); // may not be needed
            }
        }
        // output
        for (PosInt ig=0; ig<ngTypeE; ig++) {
            PosInt gid = ig*maxChunkSize*blockSize + i;
            gE[gid] = local_gE[ig];
            hE[gid] = local_hE[ig];
        }
        for (PosInt ig=0; ig<ngTypeI; ig++) {
            PosInt gid = ig*maxChunkSize*blockSize + i;
            gI[gid] = local_gI[ig];
            hI[gid] = local_hI[ig];
        }
    }
    delete [] local_gE;
    delete [] local_gI;
    delete [] local_hE;
    delete [] local_hI;
}

__global__
sum_G(
        Size* __restrict__ nVec,
        Float* __restrict__ gEt,
        Float* __restrict__ gE,
        Float* __restrict__ gIt,
        Float* __restrict__ gI,
        Float* __restrict__ hEt,
        Float* __restrict__ hE,
        Float* __restrict__ hIt,
        Float* __restrict__ hI)
{
    PosInt id = blockIdx.x*blockDim.x + threadIdx.x;
    if (nVec[id] > 0) {
        gE[id] += gEt[id];
        hE[id] += hEt[id];
        gI[id] += gIt[id];
        hI[id] += hIt[id];
    }
}

