#ifndef COREDYNAMICS_H
#define COREDYNAMICS_H

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <curand_kernel.h>
#include <cuda.h>
#include "condShape.h"
#include "LIF_inlines.h"
#include "MACRO.h"

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
void recal_G(
        Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        Float* __restrict__ preMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        Float* __restrict__ delayMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        Float* __restrict__ gE, // [ngTypeE, networkSize]
        Float* __restrict__ gI, // [ngTypeI, networkSize] 
        Float* __restrict__ hE,
        Float* __restrict__ hI,
        PosInt* __restrict__ blockGready,
        ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt block_offset, Size trainDepth, Size nearNeighborBlock, Size networkSize, Size mE, Float speedOfThought);

__global__ void logRand_init(Float *logRand, Float *lTR, curandStateMRG32k3a *state, PosIntL seed);

template <typename T>
__global__ void init(T *array,
                     T value) 
{
    PosInt id = blockIdx.x * blockDim.x + threadIdx.x;
    array[id] = value;
}

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
        PosInt current_slot, Size trainDepth, Size max_nLGN, Size ngTypeE, Size ngTypeI, Size ngType, ConductanceShape condE, ConductanceShape condI, Float dt, Size networkSize, Size mE, PosIntL seed);

#endif
