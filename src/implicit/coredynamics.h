#ifndef COREDYNAMICS_H
#define COREDYNAMICS_H

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <curand_kernel.h>
#include <cuda.h>
#include "condShape.h"
#include "MACRO.h"
#include "rk2.h"


__global__ void logRand_init(double *logRand, curandStateMRG32k3a *state, unsigned long long seed, double *lTR, double dInput, unsigned int offset);

__global__ void randInit(double* __restrict__ preMat, 
						 double* __restrict__ v, 
						 curandStateMRG32k3a* __restrict__ state,
double sEE, double sIE, double sEI, double sII, unsigned int networkSize, unsigned int nE, unsigned long long seed);

template <typename T>
__global__ void init(T *array, T value) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    array[id] = value;
}

__global__ void f_init(double* __restrict__ f, unsigned networkSize, unsigned int nE, unsigned int ngType, double Ef, double If);

__global__ void compute_V(double* __restrict__ v,
                          double* __restrict__ gE,
                          double* __restrict__ gI,
                          double* __restrict__ hE,
                          double* __restrict__ hI,
                          double* __restrict__ preMat,
                          double* __restrict__ inputRateE,
                          double* __restrict__ inputRateI,
                          int* __restrict__ eventRateE,
                          int* __restrict__ eventRateI,
                          double* __restrict__ spikeTrain,
                          unsigned int* __restrict__ nSpike,
                          double* __restrict__ tBack,
                          double* __restrict__ fE,
                          double* __restrict__ fI,
                          double* __restrict__ leftTimeRateE,
                          double* __restrict__ leftTimeRateI,
                          double* __restrict__ lastNegLogRandE,
                          double* __restrict__ lastNegLogRandI,
                          curandStateMRG32k3a* __restrict__ stateE,
                          curandStateMRG32k3a* __restrict__ stateI,
                          ConductanceShape condE, ConductanceShape condI, double dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, double dInputE, double dInputI);

__global__ void compute_V_without_ssc(double* __restrict__ v,
                                      double* __restrict__ gE,
                                      double* __restrict__ gI,
                                      double* __restrict__ hE,
                                      double* __restrict__ hI,
                                      double* __restrict__ preMat,
                                      double* __restrict__ inputRateE,
                                      double* __restrict__ inputRateI,
                                      int* __restrict__ eventRateE,
                                      int* __restrict__ eventRateI,
                                      double* __restrict__ spikeTrain,
                                      unsigned int* __restrict__ nSpike,
                                      double* __restrict__ tBack,
                                      double* __restrict__ fE,
                                      double* __restrict__ fI,
                                      double* __restrict__ leftTimeRateE,
                                      double* __restrict__ leftTimeRateI,
                                      double* __restrict__ lastNegLogRandE,
                                      double* __restrict__ lastNegLogRandI,
                                      curandStateMRG32k3a* __restrict__ stateE,
                                      curandStateMRG32k3a* __restrict__ stateI,
                                      ConductanceShape condE, ConductanceShape condI, double dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, double dInputE, double dInputI);

#endif
