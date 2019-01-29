#ifndef RK2_H
#define RK2_H
#include <cassert>
#include "CONST.h"
#include "stdio.h"

typedef struct Runge_Kutta_2 {
    unsigned int spikeCount;
    bool correctMe;
    double v, v0;
    double tBack, tsp;
    double a0, b0;
    double a1, b1;
    __device__ Runge_Kutta_2(double _v0, double _tBack): v0(_v0), tBack(_tBack) {};
    __device__ virtual void set_p0(double _gE, double _gI, double _gL);
    __device__ virtual void set_p1(double _gE, double _gI, double _gL);
    __device__ virtual void transfer_p1_to_p0();
    __device__ virtual void compute_spike_time(double dt, double t0 = 0.0f);
    __device__ virtual void reset_v();
    __device__ virtual void compute_v(double dt) = 0;
    __device__ virtual void recompute(double dt, double t0=0.0f) = 0;
    __device__ virtual void recompute_v0(double dt, double t0=0.0f) = 0;
    __device__ virtual void recompute_v(double dt, double t0=0.0f) = 0;
} LIF;

struct impl_rk2: Runge_Kutta_2 {
    double denorm;
    __device__ impl_rk2(double _v0, double _tBack): Runge_Kutta_2(_v0, _tBack) {};
    __device__ void compute_v(double dt);
    __device__ void recompute(double dt, double t0=0.0f);
    __device__ void recompute_v0(double dt, double t0=0.0f);
    __device__ void recompute_v(double dt, double t0=0.0f);
};

struct rk2: Runge_Kutta_2 {
    __device__ rk2(double _v0, double _tBack): Runge_Kutta_2(_v0, _tBack) {};
    __device__ double eval0(double _v);
    __device__ double eval1(double _v);
    __device__ void compute_v(double dt);
    __device__ void recompute(double dt, double t0=0.0f);
    __device__ void recompute_v0(double dt, double t0=0.0f);
    __device__ void recompute_v(double dt, double t0=0.0f);
};

typedef struct rangan_int {
    unsigned int spikeCount;
    bool correctMe;
    double v, v0;
    double tBack, tsp;
} rangan;

#endif
