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
#include "MACRO.h"
#include <vector>

__forceinline__  
__device__ 
Float get_a(Float gE, Float gI, Float gL) {
    return gE + gI + gL;
}

__forceinline__ 
__device__ 
Float get_b(Float gE, Float gI, Float gL) {
    return gE * vE + gI * vI + gL * vL;
}

struct AdEx { //Adaptive Exponential IF
	Size spikeCount;
    Float v, v0;
    Float a0, b0;
    Float a1, b1;
    Float vR, vThres;
    Float tRef, tBack, tsp;
    Float C, gL, depC;
	Float w0, w;
	Float tau_w, a, b;
	Float vT, deltaT;
	Float gapS;
    __device__ 
	__forceinline__
	AdEx(Float _w0, Float _tau_w, Float _a, Float _b, Float _v0, Float _tBack, Float _vR, Float _vThres, Float _gL, Float _C, Float _tRef, Float _vT, Float _deltaT, Float _gapS, Float dep): v0(_v0), tBack(_tBack), vR(_vR), vThres(_vThres), gL(_gL), C(_C), tRef(_tRef), w0(_w0), tau_w(_tau_w), a(_a), b(_b), vT(_vT), deltaT(_deltaT), gapS(_gapS) {
		spikeCount = 0;
		Float targetV = vR + (vT-vR)*dep;
		if (targetV > vT) {
			depC = (gL+a)*(targetV - vL) - gL*deltaT;
			
		} else {
			depC = (gL+a)*(targetV - vL) - gL*deltaT*exponential((targetV - vT)/deltaT);
		}
	};
    __device__ 
	__forceinline__
	void rk2_vFixedBefore(Float dt) {
		Float A = a*(v0 - vL);
		w0 = (w0 - A) * exponential(-dt/tau_w) + A;
	}
    __device__ 
	__forceinline__
	void rk2_vFixedAfter(Float dt) {
		Float A = a*(v - vL);
		w = (w0 - A) * exponential(-dt/tau_w) + A;
	}

	__device__ 
	__forceinline__
	void rk2(Float dt, Float noise) {
		Float dTgL = deltaT*gL;
		Float fk1 = -a0*v0 + b0 + dTgL*exponential((v0-vT)/deltaT) - w0;
		fk1 *= dt;
		fk1 += noise;
		fk1 /= C;
		Float v1 = v0 + fk1;
		v = v0 + fk1/2;

		Float gk1 = a*(v0 - vL) - w0;
		gk1 *= dt/tau_w;
		Float w1 = w0 + gk1; // split add fk1, fk2 to optimize register usage
		w = w0 + gk1/2;

		Float fk2 = -a1*v1 + b1 + dTgL*exponential((v1-vT)/deltaT) - w1;
		Float gk2 = (a*(v1 - vL) - w1)*dt/tau_w;
		w += (gk2*dt/tau_w)/2;
		v += (fk2*dt/C + noise)/2;
	}

	__device__
	__forceinline__
	void compute_spike_time(Float dt, Float t0) {
		//if (v > vPeak) v = vPeak;
		//Float dv0 = v0-vR;
		//Float denorm = logarithm((v-vR)/dv0);
    	//tsp = t0 + logarithm((vThres-vR)/dv0)/denorm*dt;
		//Float eT = exponential(-(vThres - v0));
		//Float eV = exponential(-(v - v0));
		//tsp = t0 + (1-eT)/(1-eV)*dt;
		tsp = t0 + (vThres-v0)/(v-v0)*dt;
	}

	__device__ 
	__forceinline__
	void set_p0(Float gE, Float gI, Float gap) {
	    a0 = get_a(gE, gI, gL) + gapS;
	    b0 = get_b(gE, gI, gL) + depC + gap;
	}
	
	__device__ 
	__forceinline__
	void set_p1(Float gE, Float gI, Float gap) {
	    a1 = get_a(gE, gI, gL) + gapS;
	    b1 = get_b(gE, gI, gL) + depC + gap;
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
    __device__ void update(Float var[], PosInt tid) {
		var[tid] = w;
	};
};

template <typename T>
__device__ void block_collectID(bool exist, T &id, PosInt counter[], PosInt sep) {
}

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
                 PosIntL seed, Size networkSize, Size nType, Size SCsplit, Size trainDepth, Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, Size nE, Size nI, int noDelay, bool iModel
);

__global__ void logRand_init(Float *logRand, Float *lTR, int* LGN_idx, int* LGN_idy, curandStateMRG32k3a *state, cudaSurfaceObject_t LGNspikeSurface, PosIntL seed, Size n, Size nFF);

void recal_G_vec(
        std::vector<std::vector<std::vector<Float>>> &spikeTrain, std::vector<std::vector<Size>> &trainDepth, std::vector<std::vector<PosInt>> &currentTimeSlot, Float og[], Float oh[],
        std::vector<Size> &nVec,  std::vector<std::vector<PosInt>> &vecID, std::vector<std::vector<Float>> &conVec, std::vector<std::vector<Float>> &delayVec,
        Float gE[], Float gI[], Float hE[], Float hI[], Float pE[], Float pI[], Size typeAcc[],
        std::default_random_engine *h_rGenCond, Float synFail[], Float synPerCon[],
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, PosInt block_offset, Size nType, Size nE, Size nI, Size nV1, Float speedOfThought, Size chunkSize, bool noFarDelay, PosInt it, Size neuronPerBlock
);

void recal_Gap_vec(
        std::vector<std::vector<std::vector<Float>>> &gapTrain, std::vector<std::vector<Size>> &gapDepth, std::vector<std::vector<PosInt>> &gap_currentTimeSlot,
        std::vector<Size> &nGapVec, std::vector<std::vector<PosInt>> &gapVecID, std::vector<std::vector<Float>> &gapVec, std::vector<std::vector<Float>> &gapDelayVec,
		std::vector<Float> &vThres, Float gap[], Size typeAcc[], 
        Float dt, PosInt block_offset, Size nType, Size nTypeE, Size nI, Float speedOfThought, Size chunkSize, bool noFarDelay, Size neuronPerBlock
);

//template<int ntimesFF, int ntimesE, int ntimesI> extern
__global__ 
void compute_V_collect_spike_learnFF(
        Float* __restrict__ v,
        Float* __restrict__ dep,
        Float* __restrict__ w, // AdEx
        Float* __restrict__ gapS, // gap junction strength
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
        Float* __restrict__ C,
        Float* __restrict__ tRef,
        Float* __restrict__ tonicDep,
        Float* __restrict__ vT, // AdEx
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
        LearnVarShapeE learnE, LearnVarShapeQ learnQ, Float exp_homeo, int iModel, int noDelay, int applyHomeo, bool symmetricHomeo, bool InhGap
);

__global__ 
void compute_V_collect_spike_learnFF_fast(
        Float* __restrict__ v,
        Float* __restrict__ dep,
        Float* __restrict__ w, // AdEx
        Float* __restrict__ gapS, // gap junction strength
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
        Float* __restrict__ C,
        Float* __restrict__ tRef,
        Float* __restrict__ tonicDep,
        Float* __restrict__ vT, // AdEx
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
        LearnVarShapeE learnE, LearnVarShapeQ learnQ, Float exp_homeo, int iModel, int noDelay, int applyHomeo, bool symmetricHomeo, bool InhGap
);

//template<int ntimesE, int ntimesI> extern
__global__  
void recal_G_mat( // <<< nblock[partial], blockSize >>>
        Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        float* __restrict__ conMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        float* __restrict__ delayMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        float* __restrict__ gapMat, // [nblock, nearNeighborBlock, nI, nI]
        Size* __restrict__ nNeighborBlock,
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
        LearnVarShapeE lE, LearnVarShapeQ lQ, PosInt iChunk, bool InhGap
);

//template<int ntimesE, int ntimesI> extern
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
        Size ngTypeE, Size ngTypeI, PosInt it
);

__global__
void sum_Gap(
        Size* __restrict__ nGapVec, // block_offset accounted for
        Float* __restrict__ gapt,
        Float* __restrict__ gap
);

//template<int ntimesE, int ntimesI> extern
__global__  
void recal_G_mat_nd( // <<< nblock[partial], blockSize >>>
        Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        Float* __restrict__ output_g, // [depth, nblock, blockSize]
        Float* __restrict__ output_h, // [depth, nblock, blockSize]
        float* __restrict__ conMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        float* __restrict__ gapMat, // [nblock, nearNeighborBlock, nI, nI]
        Size* __restrict__ nNeighborBlock,
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
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, Size nearNeighborBlock, Size nE, Size nI, Size nV1, int learning, PosInt block_offset, Size nType, Size nTypeE, Size nTypeI,
        LearnVarShapeE lE, LearnVarShapeQ lQ, PosInt iChunk, PosInt it, bool InhGap
);

__global__  
void recal_G_mat_nd_fast( // <<< nblock[partial], blockSize >>>
        Float* __restrict__ spikeTrain, // [depth, nblock, blockSize]
        PosInt* __restrict__ ipre, // [depth, nblock, nTypeHierarchy]
        Size* __restrict__ npre, // [depth, nblock, nTypeHierarchy]
        Float* __restrict__ output_g, // [depth, nblock, blockSize]
        Float* __restrict__ output_h, // [depth, nblock, blockSize]
        float* __restrict__ conMat, // [nblock, nearNeighborBlock, blockSize, blockSize]
        float* __restrict__ gapMat, // [nblock, nearNeighborBlock, nI, nI]
        Size* __restrict__ nNeighborBlock,
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
        Float dt, ConductanceShape condE, ConductanceShape condI, Size ngTypeE, Size ngTypeI, Size nearNeighborBlock, Size nE, Size nI, Size nV1, int learning, PosInt block_offset, Size nType, Size nTypeE, Size nTypeI, Size nblock,
        LearnVarShapeE lE, LearnVarShapeQ lQ, PosInt iChunk, PosInt it, bool InhGap
);
#endif
