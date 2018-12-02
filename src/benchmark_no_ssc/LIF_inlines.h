#ifndef LIF_INLINES_H
#define LIF_INLINES_H
#include "CONST.h"

__forceinline__  __host__ __device__ double eval_LIF(double a, double b, double v) {
    return -a * v + b;
}

__forceinline__ __host__ __device__ double get_a(double gE, double gI, double gL) {
    return gE + gI + gL;
}

__forceinline__ __host__ __device__ double get_b(double gE, double gI, double gL) {
    return gE * vE + gI * vI + gL * vL;
}

__forceinline__ __host__ __device__ double compute_v1(double dt, double a0, double b0, double a1, double b1, double v, double t) {
    double A = 1.0 + (a0*a1*dt - a0 - a1) * dt/2.0f;
    double B = (b0 + b1 - a1*b0*dt) * dt/2.0f;
    return (B*(t-dt) - A*v*dt)/(t-dt-A*t);
}


#endif
