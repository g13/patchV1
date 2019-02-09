#ifndef RK2_H
#define RK2_H
#include "DIRECTIVE.h"
#include "CONST.h"

typedef struct Runge_Kutta_2 {
    unsigned int spikeCount;
    bool correctMe;
    _float v, v0;
    _float tBack, tsp;
    _float a0, b0;
    _float a1, b1;
    __device__ Runge_Kutta_2(_float _v0, _float _tBack): v0(_v0), tBack(_tBack) {};

    __device__ void compute_spike_time(_float dt, _float t0 = 0.0f);
    __device__ void set_p0(_float gE, _float gI, _float gL);
    __device__ void set_p1(_float gE, _float gI, _float gL);
    __device__ void transfer_p1_to_p0();
    __device__ void reset_v();
    __device__ virtual void compute_v(_float dt) = 0;
    __device__ virtual void recompute(_float dt, _float t0=0.0f) = 0;
    __device__ virtual void recompute_v0(_float dt, _float t0=0.0f) = 0;
    __device__ virtual void recompute_v(_float dt, _float t0=0.0f) = 0;
    
} LIF;

#if SCHEME == 0
struct rk2: Runge_Kutta_2 {
    __device__ rk2(_float _v0, _float _tBack): Runge_Kutta_2(_v0, _tBack) {};
    __device__ void rk2::compute_v(_float dt);
    __device__ void rk2::recompute_v(_float dt, _float t0 = 0.0f);
    __device__ void rk2::recompute_v0(_float dt, _float t0 = 0.0f);
};
#endif

#if SCHEME == 1
struct impl_rk2: Runge_Kutta_2 {
    _float denorm;
    __device__ impl_rk2(_float _v0, _float _tBack): Runge_Kutta_2(_v0, _tBack) {};
    __device__ void compute_v(_float dt);
    __device__ void recompute(_float dt, _float t0 = 0.0f);
    __device__ void recompute_v(_float dt, _float t0 = 0.0f);
    __device__ void recompute_v0(_float dt, _float t0 = 0.0f);
};
#endif

#if SCHEME == 2
typedef struct rangan_int: Runge_Kutta_2 {
    _float eG;
    _float dVs0, dVs1;
    __device__ rangan_int(_float _v0, _float _tBack, _float _dVs): Runge_Kutta_2(_v0, _tBack), dVs0(_dVs) {};
    __device__ void set_dVs0(_float dgE, _float dgI);
    __device__ void set_dVs1(_float dgE, _float dgI);
    __device__ void set_G(_float G, _float gL, _float dt);
    __device__ void compute_v(_float dt);
    __device__ void recompute(_float dt, _float t0 = 0.0f);
    __device__ void recompute_v0(_float dt, _float t0 = 0.0f);
    __device__ void recompute_v(_float dt, _float t0 = 0.0f);
} rangan;
#endif

#endif
