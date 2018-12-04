#ifndef COREDYNAMICS_H
#define COREDYNAMICS_H

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cassert>
#include "condShape.h"
#include "LIF_inlines.h"
#include "MACRO.h"

struct Func_RK2 {
    double v, v0, v_hlf;
    // type variable
    double tBack, tsp;
    bool correctMe;
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

__global__ void compute_dV(double* __restrict__ v0,
                           double* __restrict__ dv,
                           double* __restrict__ gE,
                           double* __restrict__ gI,
                           double* __restrict__ hE,
                           double* __restrict__ hI,
                           double* __restrict__ a0,
                           double* __restrict__ b0,
                           double* __restrict__ a1,
                           double* __restrict__ b1,
                           double* __restrict__ preMat,
                           double* __restrict__ inputRate,
                           int* __restrict__ eventRate,
                           double* __restrict__ spikeTrain,
						   double* __restrict__ tBack,
                           double* __restrict__ gactVec,
                           double* __restrict__ hactVec,
                           double* __restrict__ fE,
                           double* __restrict__ fI,
                           double* __restrict__ leftTimeRate,
                           double* __restrict__ lastNegLogRand,
                           double* __restrict__ v_hlf,
                           curandStateMRG32k3a* __restrict__ state,
                           unsigned int ngTypeE, unsigned int ngTypeI, unsigned int ngType, ConductanceShape condE, ConductanceShape condI, double dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, int nInput, bool it);

__global__ void prepare_cond(double* __restrict__ tBack,
                             double* __restrict__ spikeTrain,
                             double* __restrict__ gactVec,
                             double* __restrict__ hactVec,
                             ConductanceShape cond, double dt, unsigned int ngType, unsigned int offset, unsigned int networkSize);

__global__ void recal_G(double* __restrict__ g,
                        double* __restrict__ h,
                        double* __restrict__ preMat,
                        double* __restrict__ gactVec,
                        double* __restrict__ hactVec,
                        double* __restrict__ g_b1y,
                        double* __restrict__ h_b1y,
                        unsigned int n, unsigned int offset, unsigned int ngType, unsigned int ns, int m);

__global__ void reduce_G(double* __restrict__ g,
                         double* __restrict__ h,
                         double* __restrict__ g_b1y,
                         double* __restrict__ h_b1y,
                         unsigned int ngType, int n);

__global__ void correct_spike(bool*   __restrict__ not_matched,
                              double* __restrict__ spikeTrain,
                              double* __restrict__ v_hlf,
                              double* __restrict__ v0,
                              double* __restrict__ dv,
                              double* __restrict__ a0,
                              double* __restrict__ b0,
                              double* __restrict__ a1,
                              double* __restrict__ b1,
                              double* __restrict__ vnew,
                              double* __restrict__ preMat,
                              unsigned int ngTypeE, unsigned int ngTypeI, ConductanceShape condE, ConductanceShape condI, double dt, unsigned int poolSizeE, unsigned int poolSize);

__global__ void logRand_init(double *logRand, curandStateMRG32k3a *state, unsigned long long seed);

template <typename T>
__global__ void init(T *array, T value) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    array[id] = value;
}

#endif
