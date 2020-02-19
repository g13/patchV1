#ifndef LIF_INLINES_H
#define LIF_INLINES_H
#include "CONST.h"

__forceinline__ __host__ __device__ double get_a(double gE, double gI, double gL) {
    return gE + gI + gL;
}

__forceinline__ __host__ __device__ double get_b(double gE, double gI, double gL) {
    return gE * vE + gI * vI + gL * vL;
}

__forceinline__ __host__ __device__ double impl_rk2(double dt, double a0, double b0, double a1, double b1, double v0) {
    return (2*v0 + (-a0*v0+b0+b1)*dt)/(2+a1*dt);
}

__forceinline__ __host__ __device__ double recomp_v(double A, double B, double rB) {
    return (A*(1+rB)*vL + rB*B)/(A+rB);
}

__forceinline__ __host__ __device__ double recomp_v0(double A, double B, double rB) {
    return ((1+rB)*vL - B)/(A+rB);
}

__forceinline__ __host__ __device__ double comp_spike_time(double v,double v0, double dt, double t0 = 0.0) {
    return t0 + (vT-v0)/(v-v0)*dt;
}
#endif
