#ifndef COREDYNAMICS_H
#define COREDYNAMICS_H

#include <curand_kernel.h>
#include <cassert>
#include "DIRECTIVE.h"
#include "CUDA_MACRO.h"
#include "condShape.h"

__global__ void logRand_init(_float *logRand, curandStateMRG32k3a *state, unsigned long long seed, _float *lTR, _float dInput, unsigned int offset);

__global__ void preMatRandInit(_float* __restrict__ preMat, 
						       _float* __restrict__ v, 
						       curandStateMRG32k3a* __restrict__ state,
_float sEE, _float sIE, _float sEI, _float sII, unsigned int networkSize, unsigned int nE, unsigned long long seed);

__global__ void f_init(_float* __restrict__ f, unsigned networkSize, unsigned int nE, unsigned int ngType, _float Ef, _float If);

#if SCHEME < 2
    #ifdef SPIKE_CORRECTION

        __global__ void
        __launch_bounds__(blockSize, 1)
        compute_V(_float* __restrict__ v,
                  _float* __restrict__ gE,
                  _float* __restrict__ gI,
                  _float* __restrict__ hE,
                  _float* __restrict__ hI,
                  _float* __restrict__ preMat,
                  _float* __restrict__ inputRateE,
                  _float* __restrict__ inputRateI,
                  int* __restrict__ eventRateE,
                  int* __restrict__ eventRateI,
                  _float* __restrict__ spikeTrain,
                  unsigned int* __restrict__ nSpike,
                  _float* __restrict__ tBack,
                  _float* __restrict__ fE,
                  _float* __restrict__ fI,
                  _float* __restrict__ leftTimeRateE,
                  _float* __restrict__ leftTimeRateI,
                  _float* __restrict__ lastNegLogRandE,
                  _float* __restrict__ lastNegLogRandI,
                  curandStateMRG32k3a* __restrict__ stateE,
                  curandStateMRG32k3a* __restrict__ stateI,
                  ConductanceShape condE, ConductanceShape condI, _float dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, _float dInputE, _float dInputI, _float t);

    #else

        __global__ void
        __launch_bounds__(blockSize, 1)
        compute_V_without_ssc(_float* __restrict__ v,
                              _float* __restrict__ gE,
                              _float* __restrict__ gI,
                              _float* __restrict__ hE,
                              _float* __restrict__ hI,
                              _float* __restrict__ preMat,
                              _float* __restrict__ inputRateE,
                              _float* __restrict__ inputRateI,
                              int* __restrict__ eventRateE,
                              int* __restrict__ eventRateI,
                              _float* __restrict__ spikeTrain,
                              unsigned int* __restrict__ nSpike,
                              _float* __restrict__ tBack,
                              _float* __restrict__ fE,
                              _float* __restrict__ fI,
                              _float* __restrict__ leftTimeRateE,
                              _float* __restrict__ leftTimeRateI,
                              _float* __restrict__ lastNegLogRandE,
                              _float* __restrict__ lastNegLogRandI,
                              curandStateMRG32k3a* __restrict__ stateE,
                              curandStateMRG32k3a* __restrict__ stateI,
                              ConductanceShape condE, ConductanceShape condI, _float dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, _float dInputE, _float dInputI, _float t);

    #endif
#else


    __global__ void
    __launch_bounds__(blockSize, 1)
    int_V(_float* __restrict__ v,
    	  _float* __restrict__ dVs,
          _float* __restrict__ gE,
          _float* __restrict__ gI,
          _float* __restrict__ hE,
          _float* __restrict__ hI,
          _float* __restrict__ preMat,
          _float* __restrict__ inputRateE,
          _float* __restrict__ inputRateI,
          int* __restrict__ eventRateE,
          int* __restrict__ eventRateI,
          _float* __restrict__ spikeTrain,
          unsigned int* __restrict__ nSpike,
          _float* __restrict__ tBack,
          _float* __restrict__ fE,
          _float* __restrict__ fI,
          _float* __restrict__ leftTimeRateE,
          _float* __restrict__ leftTimeRateI,
          _float* __restrict__ lastNegLogRandE,
          _float* __restrict__ lastNegLogRandI,
          curandStateMRG32k3a* __restrict__ stateE,
          curandStateMRG32k3a* __restrict__ stateI,
          ConductanceShape condE, ConductanceShape condI, _float dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, _float dInputE, _float dInputI, _float t);

#endif

#endif
