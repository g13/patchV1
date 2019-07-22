#ifndef COREDYNAMICS_H
#define COREDYNAMICS_H

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <cuda.h>
#include "condShape.h"
#include "LIF_inlines.h"
#include "MACRO.h"

struct Func_RK2 {
    double v, v0;
    // type variable
    double tBack, tsp;
    unsigned int spikeCount;
    __device__ Func_RK2(double _v0, double _tBack) : v0(_v0), tBack(_tBack) {};
    __device__ void runge_kutta_2(double dt);
    __device__ virtual void set_p0(double _gE, double _gI, double _gL) = 0;
    __device__ virtual void set_p1(double _gE, double _gI, double _gL) = 0;
    __device__ virtual double eval0(double _v) = 0;
    __device__ virtual double eval1(double _v) = 0;
    __device__ virtual void reset_v() = 0;
    __device__ virtual void compute_pseudo_v0(double dt) = 0;
    __device__ virtual void compute_v(double dt) = 0;
    __device__ virtual double compute_spike_time(double dt) = 0;
};

struct LIF: Func_RK2 {
    double a1, b1;
    double a0, b0;
    __device__ LIF(double _v0, double _tBack) : Func_RK2(_v0, _tBack) {};
    __device__ virtual void set_p0(double _gE, double _gI, double _gL);
    __device__ virtual void set_p1(double _gE, double _gI, double _gL);
    __device__ virtual double eval0(double _v);
    __device__ virtual double eval1(double _v);
    __device__ virtual void reset_v();
    __device__ virtual void compute_pseudo_v0(double dt);
    __device__ virtual void compute_v(double dt);
    __device__ virtual double compute_spike_time(double dt);
};

__global__ void recal_G(double* __restrict__ g,
                        double* __restrict__ h,
                        double* __restrict__ preMat,
                        double* __restrict__ gactVec,
                        double* __restrict__ hactVec,
                        double* __restrict__ g_b1x,
                        double* __restrict__ h_b1x,
                        unsigned int n, unsigned int offset, unsigned int ngType, unsigned int ns, int m);

__global__ void reduce_G(double* __restrict__ g,
                         double* __restrict__ h,
                         double* __restrict__ g_b1x,
                         double* __restrict__ h_b1x,
                         unsigned int ngType, int n);

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
                          double* __restrict__ a,
                          double* __restrict__ b,
                          double* __restrict__ preMat,
                          double* __restrict__ inputRateE,
                          double* __restrict__ inputRateI,
                          int* __restrict__ eventRateE,
                          int* __restrict__ eventRateI,
                          double* __restrict__ spikeTrain,
                          unsigned int* __restrict__ nSpike,
                          double* __restrict__ tBack,
                          double* __restrict__ gactVec,
                          double* __restrict__ hactVec,
                          double* __restrict__ fE,
                          double* __restrict__ fI,
                          double* __restrict__ leftTimeRateE,
                          double* __restrict__ leftTimeRateI,
                          double* __restrict__ lastNegLogRandE,
                          double* __restrict__ lastNegLogRandI,
                          curandStateMRG32k3a* __restrict__ stateE,
                          curandStateMRG32k3a* __restrict__ stateI,
                          unsigned int ngTypeE, unsigned int ngTypeI, unsigned int ngType, ConductanceShape condE, ConductanceShape condI, double dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, double dInputE, double dInputI);

#endif
