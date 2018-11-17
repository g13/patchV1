#ifndef COREDYNAMICS_H
#define COREDYNAMICS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <cuda.h>
#include "MACRO.h"

struct ConductanceShape {
    double riseTime[5], decayTime[5], deltaTau[5];
    __host__ __device__ ConductanceShape() {};
    __host__ __device__ ConductanceShape(double rt[], double dt[], unsigned int ng) {
        for (int i = 0; i < ng; i++) {
            riseTime[i] = rt[i];
            decayTime[i] = dt[i];
            deltaTau[i] = dt[i] - rt[i];
        }
    }
    __host__ __device__ __forceinline__ void compute_single_input_conductance(double *g, double *h, double f, double dt, unsigned int ig) {
        double etr = exp(-dt / riseTime[ig]);
        (*g) += f * decayTime[ig] * (exp(-dt / decayTime[ig]) - etr) / deltaTau[ig];
        (*h) += f * etr;
    }
    __host__ __device__ __forceinline__ void decay_conductance(double *g, double *h, double dt, unsigned int ig) {
        double etr = exp(-dt / riseTime[ig]);
        double etd = exp(-dt / decayTime[ig]);
        (*g) = (*g)*etd + (*h)*decayTime[ig] * (etd - etr) / deltaTau[ig];
        (*h) = (*h)*etr;
    }
};

struct Func_RK2 {
    double v, v0;
    // type variable
    double tBack, tsp;
    __device__ Func_RK2(double _v0, double _tBack) : v0(_v0), tBack(_tBack) {};
    __device__ void runge_kutta_2(double dt);
    __device__ virtual void set_p0(double _gE, double _gI, double _gL) = 0;
    __device__ virtual void set_p1(double _gE, double _gI, double _gL) = 0;
    __device__ virtual double eval0(double _v) = 0;
    __device__ virtual double eval1(double _v) = 0;
    __device__ virtual void reset_v() = 0;
    __device__ virtual void compute_pseudo_v0(double dt) = 0;
    __device__ virtual double compute_spike_time(double dt) = 0;
};

struct LIF : Func_RK2 {
    double a1, b1;
    double a0, b0;
    __device__ LIF(double _v0, double _tBack) : Func_RK2(_v0, _tBack) {};
    __device__ virtual void set_p0(double _gE, double _gI, double _gL);
    __device__ virtual void set_p1(double _gE, double _gI, double _gL);
    __device__ virtual double eval0(double _v);
    __device__ virtual double eval1(double _v);
    __device__ virtual void reset_v();
    __device__ virtual void compute_pseudo_v0(double dt);
    __device__ virtual double compute_spike_time(double dt);
};

__global__ void compute_V(double* __restrict__ v,
                          double* __restrict__ gE,
                          double* __restrict__ gI,
                          double* __restrict__ hE,
                          double* __restrict__ hI,
                          double* __restrict__ a,
                          double* __restrict__ b,
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
                          curandStateMRG32k3a* __restrict__ state,
                          unsigned int ngTypeE, unsigned int ngTypeI, unsigned int ngType, ConductanceShape condE, ConductanceShape condI, double dt, unsigned int networkSize, unsigned int nE, unsigned long long seed, bool it);

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

__global__ void logRand_init(double *logRand, curandStateMRG32k3a *state, unsigned long long seed);

template <typename T>
__global__ void init(T *array, T value) {
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
    array[id] = value;
}

#endif
