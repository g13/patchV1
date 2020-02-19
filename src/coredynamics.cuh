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

__global__ 
void recal_G(Float* __restrict__ g,
             Float* __restrict__ h,
             Float* __restrict__ preMat,
             Float* __restrict__ gactVec,
             Float* __restrict__ hactVec,
             Float* __restrict__ g_b1x,
             Float* __restrict__ h_b1x,
             Size n, PosInt offset, Size ngType, Size ns, Int m);

__global__ 
void reduce_G(Float* __restrict__ g,
              Float* __restrict__ h,
              Float* __restrict__ g_b1x,
              Float* __restrict__ h_b1x,
              Size ngType, Int n);

__global__ void logRand_init(Float *logRand, Float *lTR, curandStateMRG32k3a *state, PosIntL seed);

template <typename T>
__global__ void init(T *array,
                     T value) 
{
    PosInt id = blockIdx.x * blockDim.x + threadIdx.x;
    array[id] = value;
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
               Size ngTypeE, Size ngTypeI, Size ngType, ConductanceShape condE, ConductanceShape condI, Float dt, Size networkSize, Size nE, PosIntL seed);

#endif
