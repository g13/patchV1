#ifndef COREDYNAMICS_H
#define COREDYNAMICS_H

#include <curand_kernel.h>
#include <cassert>
#include "MACRO.h"
#include "CONST.h"
#include "condShape.h"
#include "util/cuda_util.h"
#include "types.h"
#include "rk2.h"

__global__ 
void logRand_init(Float *logRand,
                  curandStateMRG32k3a *state,
                  BigSize seed,
                  Float *lTR,
                  Float dInput,
                  Size offset);

__global__ 
void preMatRandInit(Float* __restrict__ preMat, 
					Float* __restrict__ v, 
					curandStateMRG32k3a* __restrict__ state,
                    Float sEE, Float sIE, Float sEI, Float sII,
                    Size networkSize, Size nE, BigSize seed);

__global__ 
void f_init(Float* __restrict__ f,
            Size networkSize, Size nE, Size ngType,
            Float Ef, Float If);

__launch_bounds__(blockSize, 1)
__global__ 
void compute_V_without_ssc(Float* __restrict__ v,
                           Float* __restrict__ gE,
                           Float* __restrict__ gI,
                           Float* __restrict__ hE,
                           Float* __restrict__ hI,
                           Float* __restrict__ preMat,
                           Float* __restrict__ inputRateE,
                           Float* __restrict__ inputRateI,
                           Int* __restrict__ eventRateE,
                           Int* __restrict__ eventRateI,
                           Float* __restrict__ spikeTrain,
                           Size* __restrict__ nSpike,
                           Float* __restrict__ tBack,
                           Float* __restrict__ fE,
                           Float* __restrict__ fI,
                           Float* __restrict__ leftTimeRateE,
                           Float* __restrict__ leftTimeRateI,
                           Float* __restrict__ lastNegLogRandE,
                           Float* __restrict__ lastNegLogRandI,
                           curandStateMRG32k3a* __restrict__ stateE,
                           curandStateMRG32k3a* __restrict__ stateI,
                           ConductanceShape condE, ConductanceShape condI, Float dt, Size networkSize, Size nE, BigSize seed, Float dInputE, Float dInputI, Float t);

/*
    __global__ void
    __launch_bounds__(blockSize, 1)
    int_V(Float* __restrict__ v,
    	  Float* __restrict__ dVs,
          Float* __restrict__ gE,
          Float* __restrict__ gI,
          Float* __restrict__ hE,
          Float* __restrict__ hI,
          Float* __restrict__ preMat,
          Float* __restrict__ inputRateE,
          Float* __restrict__ inputRateI,
          int* __restrict__ eventRateE,
          int* __restrict__ eventRateI,
          Float* __restrict__ spikeTrain,
          Size* __restrict__ nSpike,
          Float* __restrict__ tBack,
          Float* __restrict__ fE,
          Float* __restrict__ fI,
          Float* __restrict__ leftTimeRateE,
          Float* __restrict__ leftTimeRateI,
          Float* __restrict__ lastNegLogRandE,
          Float* __restrict__ lastNegLogRandI,
          curandStateMRG32k3a* __restrict__ stateE,
          curandStateMRG32k3a* __restrict__ stateI,
          ConductanceShape condE, ConductanceShape condI, Float dt, Size networkSize, Size nE, BigSize seed, Float dInputE, Float dInputI, Float t);

        __global__ void
        __launch_bounds__(blockSize, 1)
        compute_V(Float* __restrict__ v,
                  Float* __restrict__ gE,
                  Float* __restrict__ gI,
                  Float* __restrict__ hE,
                  Float* __restrict__ hI,
                  Float* __restrict__ preMat,
                  Float* __restrict__ inputRateE,
                  Float* __restrict__ inputRateI,
                  int* __restrict__ eventRateE,
                  int* __restrict__ eventRateI,
                  Float* __restrict__ spikeTrain,
                  Size* __restrict__ nSpike,
                  Float* __restrict__ tBack,
                  Float* __restrict__ fE,
                  Float* __restrict__ fI,
                  Float* __restrict__ leftTimeRateE,
                  Float* __restrict__ leftTimeRateI,
                  Float* __restrict__ lastNegLogRandE,
                  Float* __restrict__ lastNegLogRandI,
                  curandStateMRG32k3a* __restrict__ stateE,
                  curandStateMRG32k3a* __restrict__ stateI,
                  ConductanceShape condE, ConductanceShape condI, Float dt, Size networkSize, Size nE, BigSize seed, Float dInputE, Float dInputI, Float t);
*/

#endif
