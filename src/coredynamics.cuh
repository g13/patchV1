#ifndef COREDYNAMICS_CUH
#define COREDYNAMICS_CUH

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <curand_kernel.h>
#include <cuda.h>
#include "condShape.h"
#include "LIF_inlines.h"
#include "MACRO.h"
#include <vector>

struct LIF {
    Size spikeCount;
    Float v, v0;
    Float tBack, tsp;
    Float a0, b0;
    Float a1, b1;
    Float denorm;
    __device__ LIF(Float _v0, Float _tBack): v0(_v0), tBack(_tBack) {};
    __device__ virtual void set_p0(Float _gE, Float _gI, Float _gL);
    __device__ virtual void set_p1(Float _gE, Float _gI, Float _gL);
    __device__ virtual void implicit_rk2(Float dt);
    __device__ virtual void compute_spike_time(Float dt, Float t0 = 0.0f);
    __device__ virtual void recompute(Float dt, Float t0=0.0f);
    __device__ virtual void recompute_v0(Float dt, Float t0=0.0f);
    __device__ virtual void recompute_v(Float dt, Float t0=0.0f);
    __device__ virtual void reset_v();
};

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
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt currentTimeSlot, Size trainDepth, Size nearNeighborBlock, Size nE, Size nV1, Float speedOfThought);

__global__ void logRand_init(Float *logRand, Float *lTR, curandStateMRG32k3a *state, PosIntL seed, Size n);

__global__ 
void compute_V_collect_spike(
        Float* __restrict__ v,
        Float* __restrict__ gFF,
        Float* __restrict__ hFF,
        Float** __restrict__ gE,
        Float** __restrict__ gI,
        Float** __restrict__ hE,
        Float** __restrict__ hI,
        Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        Float* __restrict__ tBack,
        Size* __restrict__ nLGN,
        Float* __restrict__ sLGN,
        PosInt* __restrict__ LGN_idx,
        PosInt* __restrict__ LGN_idy,
        PosInt currentTimeSlot, Size trainDepth, Size max_nLGN, Size ngTypeFF, Size ngTypeE, Size ngTypeI, ConductanceShape condFF, ConductanceShape condE, ConductanceShape condI, Float dt, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nV1, PosIntL seed
);

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
        Size ngTypeE, Size ngTypeI
);

void recal_G_vec(
        Float spikeTrain[],
        std::vector<Size> &nVec, std::vector<std::vector<PosInt>> &vecID, std::vector<std::vector<Float>> &conVec, std::vector<std::vector<Float>> &delayVec,
        Float gE[], Float gI[], Float hE[], Float hI[],
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt block_offset, PosInt currentTimeSlot, Size trainDepth, Size nE, Size nV1, Float speedOfThought, Size chunkSize
);

#endif
