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

__global__ 
void recal_G(Float* __restrict__ g,
                        Float* __restrict__ h,
                        Float* __restrict__ preMat,
                        Float* __restrict__ gactVec,
                        Float* __restrict__ hactVec,
                        Float* __restrict__ g_b1x,
                        Float* __restrict__ h_b1x,
                        Size n, PosInt offset, Size ngType, Size ns, Int m) 
{
    // 2D blockGrid
    // -> D-1 pieces of actVec 
    // -> D-2 pieces of post-synaptic neurons 
    // 1D threadBlock
    extern __shared__ Float actVec[];
    Float *gaV = actVec;
    Float *haV = &(actVec[ngType*ns]);
    PosInt id = blockDim.x*blockIdx.y + threadIdx.x;
    unsigned int ss = ns/m;
    #pragma unroll
    for (int ig=0; ig<ngType; ig++) {
        #pragma unroll
        for (int i=0; i<m; i++) {
            // av = Float[ngType,#(ns),ns]
            // actVec = Float[ngType,n]
            if (threadIdx.x < ss) {
                PosInt sid = ig*ns + (i*ss + threadIdx.x);
                PosInt gid = (ig*n + offset + ns*blockIdx.x) + (i*ss + threadIdx.x);
                gaV[sid] = gactVec[gid];
                haV[sid] = hactVec[gid];
            }
        }
    }
    __syncthreads();
    for (int ig=0; ig<ngType; ig++) {
        Float g_t = 0.0f;
        Float h_t = 0.0f;
        for (int i = 0; i<ns; i++) {
            unsigned sid = ig*ns + i;
            if (gaV[sid] > 0) {
                unsigned pid = (offset + blockIdx.x*ns + i)*n + id;
                Float s = preMat[pid];
                g_t += gaV[sid] * s;
                h_t += haV[sid] * s;
            }
        }
        if (gridDim.x < 32) {
            if (g_t > 0) {
                PosInt gid = ig*n + id;
                atomicAdd(&(g[gid]), g_t);
                atomicAdd(&(h[gid]), h_t);
            }
        } else {
            // b1x = Float[ngType, n/ns(gridDim.x), n]
            PosInt b1xid = ig*n*gridDim.x + blockIdx.x*n + id;
            g_b1x[b1xid] = g_t;
            h_b1x[b1xid] = h_t;
        }
    }
}

__global__ 
void reduce_G(Float* __restrict__ g,
                         Float* __restrict__ h,
                         Float* __restrict__ g_b1x, 
                         Float* __restrict__ h_b1x,
                         Size ngType, int n) 
{ 
    // b1x = Float[ngType, n/ns(gridDim.x), n]
    // n x #(ns)
    extern __shared__ Float blk[];
    Float* g_blk = blk;
    Float* h_blk = &(blk[blockDim.x]);
    for (int ig=0; ig<ngType; ig++) {
        PosInt gid = ig*blockDim.x*gridDim.x + threadIdx.x*gridDim.x + blockIdx.x;
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
            Float g_warp = g_blk[threadIdx.x];
            Float h_warp = h_blk[threadIdx.x];
            for (int offset = 16; offset > 0; offset /= 2) {
                g_warp += __shfl_down_sync(FULL_MASK, g_warp, offset);  
                h_warp += __shfl_down_sync(FULL_MASK, h_warp, offset);  
            }
            if (threadIdx.x == 0) {
                PosInt id = ig*gridDim.x + blockIdx.x;
                g[id] += g_warp;
                h[id] += g_warp;
            }
        }
    }
}

__device__  Float one(LIF* lif, Float dt, Float tRef, PosInt id, Float gE, Float gI, Float tsp[]) {
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

__device__ void LIF::implicit_rk2(Float dt) {
    v = impl_rk2(dt, a0, b0, a1, b1, v0);
}

__device__ void LIF::compute_spike_time(Float dt, Float t0) {
    tsp = comp_spike_time(v, v0, dt, t0);
}

__device__ void LIF::recompute(Float dt, Float t0) {
    Float rB = dt/(tBack-t0) - 1; 
    Float denorm = 2 + a1*dt;
    Float A = (2 - a0*dt)/denorm;
    Float B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
    v = A*v0 + B;
}

__device__ void LIF::recompute_v(Float dt, Float t0) {
    Float rB = dt/(tBack-t0) - 1; 
    Float denorm = 2 + a1*dt;
    Float A = (2 - a0*dt)/denorm;
    Float B = (b0 + b1)*dt/denorm;
    v = recomp_v(A, B, rB);
}

__device__ void LIF::recompute_v0(Float dt, Float t0) {
    Float rB = dt/(tBack-t0) - 1; 
    Float denorm = 2 + a1*dt;
    Float A = (2 - a0*dt)/denorm;
    Float B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
}

__device__ void LIF::set_p0(Float gE, Float gI, Float gL) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

__device__ void LIF::set_p1(Float gE, Float gI, Float gL) {
    a1 = get_a(gE, gI, gL);
    b1 = get_b(gE, gI, gL); 
}

__device__ void LIF::reset_v() {
    v = vL;
}

__device__ 
void evolve_g(ConductanceShape &cond,
              Float* g, 
              Float* h, 
              Float* f,
              Float inputTime[],
              Int nInput, Float dt, unsigned int ig)
{
    cond.decay_conductance(g, h, dt, ig); 
    for (int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, *f, dt-inputTime[i], ig);
    }
}

__global__ 
void compute_V(Float* __restrict__ v,
               Float* __restrict__ gE,
               Float* __restrict__ gI,
               Float* __restrict__ hE,
               Float* __restrict__ hI,
               Float* __restrict__ spikeTrain,
               Size* __restrict__ nSpike,
               Float* __restrict__ tBack,
               Float* __restrict__ sLGN,
               Float* __restrict__ LGN_idx,
               Float* __restrict__ LGN_idy,
               Float* __restrict__ gactVec,
               Float* __restrict__ hactVec,
               curandStateMRG32k3a* __restrict__ stateE,
               curandStateMRG32k3a* __restrict__ stateI,
               Size ngTypeE, Size ngTypeI, Size ngType, ConductanceShape condE, ConductanceShape condI, Float dt, Size networkSize, Size nE, PosIntL seed)
{
    PosInt id = blockIdx.x * blockDim.x + threadIdx.x;
    // if #E neurons comes in warps (size of 32) then there is no branch divergence.
    // TODO: load individual gl, tref
    LIF lif(v[id], tBack[id]);
    Float gL, tRef;
    if (id < nE) {
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
    // Get LGN input
    for (Size iLGN = 0; i<nLGN; i++) {
        Float spikeInfo;
        surf2Dread(&spikeInfo, LGNspikeSurface, x, y);
        if (spikeInfo >= 0.0) {
            // evolve g to t+dt with ff input only
            gE_t = prep_cond(condE, gE_local, hE_local, fE_local, inputTimeE, nInputE, ngTypeE, dt, dt); 
            gI_t = prep_cond(condI, gI_local, hI_local, fI_local, inputTimeI, nInputI, ngTypeI, dt, dt); 
            // update_g(spikeInfo)
            //  get_tsp_nsp(spikeInfo)
        } else {

        }
    }
    /* evolve g to t+dt with ff input only */
    unsigned int gid;
    gE_t = 0.0f;
    #pragma unroll
    for (int ig=0; ig<ngTypeE; ig++) {
        gid = networkSize*ig + id;
        Float g_i = gE[gid];
        Float h_i = hE[gid];
        Float f_i = fE[gid];
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
        Float g_i = gI[gid];
        Float h_i = hI[gid];
        Float f_i = fI[gid];
        evolve_g(condI, &g_i, &h_i, &f_i, inputTimeI, nInputI, dt, ig);
        //__syncwarp();
        gI_t += g_i;
        gI[gid] = g_i;
        hI[gid] = h_i;
        // for learning
        //fI[gid] = f_i;
    }
    lif.set_p1(gE_t, gI_t, gL);
    // step
    Float tsp[MAX_SPIKE_PER_DT];
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
    Float g_end, h_end;
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
