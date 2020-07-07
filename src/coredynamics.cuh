#ifndef COREDYNAMICS_CUH
#define COREDYNAMICS_CUH

#include "cuda_runtime.h"
#include <stdio.h>
#include <random>
#include <stdlib.h>
#include <cassert>
#include <curand_kernel.h>
#include <cuda.h>
#include "condShape.h"
#include "LIF_inlines.h"
#include "MACRO.h"
#include <vector>

struct IF {
    Size spikeCount;
    Float v, v0;
    Float a0, b0;
    Float a1, b1;
    Float vR, vThres;
    Float tRef, tBack, tsp;
    Float gL;
    __device__ IF(Float _v0, Float _tBack, Float _vR, Float _vThres, Float _gL, Float _tRef): v0(_v0), tBack(_tBack), vR(_vR), vThres(_vThres), gL(_gL), tRef(_tRef) {
		spikeCount = 0;
	};
    __device__ virtual void rk2(Float dt)=0;
    __device__ virtual void recompute(Float dt, Float t0=0.0f) = 0;
    __device__ virtual void recompute_v0(Float dt, Float t0=0.0f) = 0;
    __device__ virtual void recompute_v(Float dt, Float t0=0.0f) = 0;
    __device__ virtual void rk2_vFixedBefore(Float dt) {}
    __device__ virtual void rk2_vFixedAfter(Float dt) {}
    __device__ virtual void reset0() = 0;

    __device__ virtual void set_p0(Float gE, Float gI);
    __device__ virtual void set_p1(Float gE, Float gI);

    __device__ virtual void compute_spike_time(Float dt, Float t0 = 0.0f);
    __device__ virtual void reset1();
    __device__ virtual void update(Float **var) {};
};

struct LIF: IF {
    Float denorm;
    __device__ LIF(Float _v0, Float _tBack, Float _vR, Float _vThres, Float _gL, Float _tRef): IF(_v0, _tBack, _vR, _vThres, _gL, _tRef) {};
    __device__ void rk2(Float dt);
    __device__ void recompute(Float dt, Float t0=0.0f);
    __device__ void recompute_v0(Float dt, Float t0=0.0f);
    __device__ void recompute_v(Float dt, Float t0=0.0f);
    __device__ 
	__forceinline__
	void reset0() {
		v0 = vR;
	};
};

struct AdEx: IF { //Adaptive Exponential IF
	Float w0, w;
	Float tau_w, a, b;
	Float vT, deltaT;
    __device__ AdEx(Float _w0, Float _tau_w, Float _a, Float _b, Float _v0, Float _tBack, Float _vR, Float _vThres, Float _gL, Float _tRef, Float _vT, Float _deltaT): IF(_v0, _tBack, _vR, _vThres, _gL, _tRef), w0(_w0), tau_w(_tau_w), a(_a), b(_b), vT(_vT), deltaT(_deltaT) {};
    __device__ void rk2(Float dt); 
    __device__ 
	__forceinline__
	void rk2_vFixedBefore(Float dt) {
		Float A = a*(v0-vL) * tau_w;
		w0 = (w0 - A) * exponential(-dt/tau_w) + A;
	}
    __device__ 
	__forceinline__
	void rk2_vFixedAfter(Float dt) {
		Float A = a*(v-vL) * tau_w;
		w = (w0 - A) * exponential(-dt/tau_w) + A;
	}
    __device__ 
	__forceinline__
	void reset0() {
		w0 += b;
		v0 = vR;
	};
    __device__
	__forceinline__
	void reset1() {
		w0 += b;
		v = vR;
	}
    __device__ void update(Float **var) {
		*var[0] = w;
	};
    __device__ void recompute(Float dt, Float t0=0.0f) {}
    __device__ void recompute_v0(Float dt, Float t0=0.0f) {}
    __device__ void recompute_v(Float dt, Float t0=0.0f) {}
};


/*
struct AdEx: LIF {
    Size spikeCount;
    Float v;
    Float vR, vT, deltaT;
    Float tRef, tBack, tsp;
    Float gL;
    Float tau_w, a;
    __device__ AdEx(Float _vR, Float _vT, Float deltaT, Float _tRef, Float _tBack, Float _gL, Float tau_w, Float _a): v0(_v0), tBack(_tBack), vR(_vR), vT(_vT), tRef(_tRef), gL(_gL) {};
    __device__ 
    __forceinline__ {

    }
    __device__ 
    __forceinline__
    void reset_v() {
        v = vR;
    }
};
*/

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
                 Float* __restrict__ tRef,
                 Float* __restrict__ tau_w,
                 Float* __restrict__ d_a,
                 Float* __restrict__ d_b,
                 curandStateMRG32k3a* __restrict__ rGenCond,
                 PosIntL seed, Size networkSize, Size nType, Size SCsplit, Size trainDepth, Float dt, bool iModel);


__global__ void logRand_init(Float *logRand, Float *lTR, int* LGN_idx, int* LGN_idy, curandStateMRG32k3a *state, PosIntL seed, Size n, Size nFF);

void recal_G_vec(
        std::vector<std::vector<std::vector<Float>>> &spikeTrain, std::vector<std::vector<Size>> &trainDepth, std::vector<std::vector<PosInt>> &currentTimeSlot,
        std::vector<Size> &nVec,  std::vector<std::vector<PosInt>> &vecID, std::vector<std::vector<Float>> &conVec, std::vector<std::vector<Float>> &delayVec,
        Float gE[], Float gI[], Float hE[], Float hI[], Float pE[], Float pI[], Size typeAcc[],
        std::default_random_engine *h_rGenCond, Float noisyCondE[], Float noisyCondI[], Float synFailE[], Float synFailI[],
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt block_offset, Size nType, Size nE, Size nV1, Float speedOfThought, Size chunkSize, bool noisyH
);

//template<int ntimesFF, int ntimesE, int ntimesI> extern
__launch_bounds__(1024,1)
__global__ 
void compute_V_collect_spike_learnFF(
        Float* __restrict__ v,
        Float* __restrict__ w, // AdEx
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
        Float* __restrict__ vAvgI, //        post, [                       nblock, nI,        ]
        Float* __restrict__ vLTP_E, //        pre, [nLearnTypeE,    depth, nblock, nE,       2]
        Float* __restrict__ vLTD_E, //       post, [nLearnTypeE,           nblock, nE,       2]
        Float* __restrict__ vTripE, //       post, [nLearnTypeE,           nblock, nE,       2]
        Float* __restrict__ vSTDP_QE,  //  E post, [nLearnTypeQ,           nblock, nE        2]
        Float* __restrict__ vSTDP_QI,  //   I pre, [nLearnTypeQ,    depth, nblock, nI,       2]
        Float* __restrict__ pFF, // LIF
        Float* __restrict__ vR,
        Float* __restrict__ vThres,
        Float* __restrict__ gL,
        Float* __restrict__ tRef,
        Float* __restrict__ vT, // AdEx
        Float* __restrict__ deltaT,
        Float* __restrict__ tau_w,
        Float* __restrict__ a,
        Float* __restrict__ b,
        Size* __restrict__ typeAcc,
        curandStateMRG32k3a* __restrict__ rGenCond,
        Float* __restrict__ noisyCondFF,
        Float* __restrict__ synFailFF,
        PosInt currentTimeSlot, Size trainDepth, Size max_nLGN, Size ngTypeFF, Size ngTypeE, Size ngTypeI, ConductanceShape condFF, ConductanceShape condE, ConductanceShape condI, Float dt, Size maxChunkSize, Size remainChunkSize, PosInt iSizeSplit, Size nChunk, Size nE, Size nI, Size nV1, int learning, int varSlot, Size nType,
        LearnVarShapeFF_E_pre  learnE_pre,  LearnVarShapeFF_I_pre  learnI_pre, 
        LearnVarShapeFF_E_post learnE_post, LearnVarShapeFF_I_post learnI_post, 
        LearnVarShapeE learnE, LearnVarShapeQ learnQ, int iModel, bool noisyH
);

//template<int ntimesE, int ntimesI> extern
__launch_bounds__(1024, 2)
__global__  
void recal_G_mat( // <<< nblock[partial], blockSize >>>
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
        Float* __restrict__ noisyCondE,
        Float* __restrict__ noisyCondI,
        Float* __restrict__ synFailE,
        Float* __restrict__ synFailI,
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt currentTimeSlot, Size trainDepth, Size nearNeighborBlock, Size nE, Size nI, Size nV1, Float speedOfThought, int learning, PosInt block_offset, Size nType,
        LearnVarShapeE lE, LearnVarShapeQ lQ, PosInt iChunk, bool noisyH
);

//template<int ntimesE, int ntimesI> extern
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
        Size ngTypeE, Size ngTypeI
);
#endif
