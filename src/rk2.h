#ifndef RK2_H
#define RK2_H

#include "DIRECTIVE.h"
#include "CONST.h"
#include "types.h"

__device__
__forceinline__
Float get_a(Float gE, Float gI, Float gL) {
    return gE + gI + gL;
}

__device__
__forceinline__
Float get_b(Float gE, Float gI, Float gL) {
    return gE * vE + gI * vI + gL * vL;
}

__device__
__forceinline__
Float comp_spike_time(Float v,Float v0, Float dt, Float t0 = 0.0) {
    return t0 + (vT-v0)/(v-v0)*dt;
}

__device__
__forceinline__
Float recomp_v0(Float A, Float B, Float rB) {
    return ((1+rB)*vL - B)/(A+rB);
}

__device__ __forceinline__ Float eval_fk(Float a, Float b, Float v) {
    return -a * v + b;
}

typedef struct Runge_Kutta_2 {
    unsigned int spikeCount;
    bool correctMe;
    Float v, v0;
    Float tBack, tsp;
    Float a0, b0;
    Float a1, b1;
    __device__ __forceinline__ Runge_Kutta_2(Float _v0, Float _tBack): v0(_v0), tBack(_tBack) {};

    __device__ __forceinline__ void compute_spike_time(Float dt, Float t0 = 0.0f);
    __device__ __forceinline__ void set_p0(Float gE, Float gI, Float gL);
    __device__ __forceinline__ void set_p1(Float gE, Float gI, Float gL);
    __device__ __forceinline__ void transfer_p1_to_p0();
    __device__ __forceinline__ void reset_v();
    __device__ __forceinline__ virtual void compute_v(Float dt) = 0;
    __device__ __forceinline__ virtual void recompute(Float dt, Float t0=0.0f) = 0;
    __device__ __forceinline__ virtual void recompute_v0(Float dt, Float t0=0.0f) = 0;
    __device__ __forceinline__ virtual void recompute_v(Float dt, Float t0=0.0f) = 0;
    
} LIF;

__device__
__forceinline__
void Runge_Kutta_2::compute_spike_time(Float dt, Float t0) {
    tsp = comp_spike_time(v, v0, dt, t0);
}

__device__
__forceinline__
void Runge_Kutta_2::set_p0(Float gE, Float gI, Float gL) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

__device__
__forceinline__
void Runge_Kutta_2::set_p1(Float gE, Float gI, Float gL) {
    a1 = get_a(gE, gI, gL);
    b1 = get_b(gE, gI, gL); 
}

__device__
__forceinline__
void Runge_Kutta_2::transfer_p1_to_p0() {
    a0 = a1;
    b0 = b1;
}

__device__
__forceinline__
void Runge_Kutta_2::reset_v() {
    v = vL;
}

//#if SCHEME == 1
struct impl_rk2: Runge_Kutta_2 {
    Float denorm;
    __device__ __forceinline__ impl_rk2(Float _v0, Float _tBack): Runge_Kutta_2(_v0, _tBack) {};
    __device__ __forceinline__ void compute_v(Float dt);
    __device__ __forceinline__ void recompute(Float dt, Float t0 = 0.0f);
    __device__ __forceinline__ void recompute_v(Float dt, Float t0 = 0.0f);
    __device__ __forceinline__ void recompute_v0(Float dt, Float t0 = 0.0f);
};

__device__ __forceinline__ void impl_rk2::compute_v(Float dt) {
    v = (2*v0 + (-a0*v0+b0+b1)*dt)/(2+a1*dt);
}

__device__ __forceinline__ void impl_rk2::recompute(Float dt, Float t0) {
    Float rB = dt/(tBack-t0) - 1; 
    Float denorm = 2 + a1*dt;
    Float A = (2 - a0*dt)/denorm;
    Float B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
    v = (A*(1+rB)*vL + rB*B)/(A+rB);
}

__device__ __forceinline__ void impl_rk2::recompute_v(Float dt, Float t0) {
    Float rB = dt/(tBack-t0) - 1; 
    Float denorm = 2 + a1*dt;
    Float A = (2 - a0*dt)/denorm;
    Float B = (b0 + b1)*dt/denorm;
    v = (A*(1+rB)*vL + rB*B)/(A+rB);
}

__device__ __forceinline__ void impl_rk2::recompute_v0(Float dt, Float t0) {
    Float rB = dt/(tBack-t0) - 1; 
    Float denorm = 2 + a1*dt;
    Float A = (2 - a0*dt)/denorm;
    Float B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
}
//#endif
/*
    #if SCHEME == 0
        struct rk2: Runge_Kutta_2 {
            __device__ __forceinline__ rk2(Float _v0, Float _tBack): Runge_Kutta_2(_v0, _tBack) {};
            __device__ __forceinline__ void compute_v(Float dt);
        	__device__ __forceinline__ void recompute(Float dt, Float t0);
            __device__ __forceinline__ void recompute_v(Float dt, Float t0 = 0.0f);
            __device__ __forceinline__ void recompute_v0(Float dt, Float t0 = 0.0f);
        };
        
        __device__ __forceinline__ void rk2::compute_v(Float dt) {
            Float fk0 = eval_fk(a0, b0, v0);
            Float fk1 = eval_fk(a1, b1, v0 + dt*fk0);
            v = v0 + dt*(fk0+fk1)/2.0f;
        }
        
        __device__ __forceinline__ void rk2::recompute(Float dt, Float t0) {
            recompute_v0(dt, t0);
            compute_v(dt);
        }
        
        __device__ __forceinline__ void rk2::recompute_v(Float dt, Float t0) {
            recompute(dt, t0);
        }
        
        __device__ __forceinline__ void rk2::recompute_v0(Float dt, Float t0) {
            Float A = a0*a1*dt - a0 - a1;
            Float B = b0 + b1 - a1*b0*dt;
            Float t = tBack-t0;
            v0 = (2*vL-t*B)/(2+t*A);
        }
    #endif
    
    #if SCHEME == 2
        typedef struct rangan_int: Runge_Kutta_2 {
            Float eG;
            Float dVs0, dVs1;
            __device__ __forceinline__ rangan_int(Float _v0, Float _tBack, Float _dVs): Runge_Kutta_2(_v0, _tBack), dVs0(_dVs) {};
            __device__ __forceinline__ void set_dVs0(Float dgE, Float dgI);
            __device__ __forceinline__ void set_dVs1(Float dgE, Float dgI);
            __device__ __forceinline__ void set_G(Float G, Float gL, Float dt);
            __device__ __forceinline__ void compute_v(Float dt);
            __device__ __forceinline__ void recompute(Float dt, Float t0 = 0.0f);
            __device__ __forceinline__ void recompute_v0(Float dt, Float t0 = 0.0f);
            __device__ __forceinline__ void recompute_v(Float dt, Float t0 = 0.0f);
        } rangan;
        
        __device__ __forceinline__ void rangan_int::set_dVs0(Float dgE, Float dgI) {
            Float da = dgE + dgI;
            Float db = dgE*vE + dgI*vI;
            dVs0 = (db - b0/a0*da)/a0;
        }
        
        __device__ __forceinline__ void rangan_int::set_dVs1(Float dgE, Float dgI) {
            Float da = dgE + dgI;
            Float db = dgE*vE + dgI*vI;
            dVs1 = (db - b1/a1*da)/a1;
        }
        
        __device__ __forceinline__ void rangan_int::set_G(Float G, Float gL, Float dt) {
            eG = exp(-G-gL*dt);
        }
        
        __device__ __forceinline__ void rangan_int::compute_v(Float dt) {
        	v = b1/a1 + eG*(v0 - b0/a0) - (eG*dVs0 + dVs1)*dt/2.0;
        }
        
        __device__ __forceinline__ void rangan_int::recompute(Float dt, Float t0) {
            Float rB = dt/(tBack-t0) - 1;
            Float integral = (eG*dVs0 + dVs1)*dt/2.0;
            v0 = ((1+rB)*vL - b1/a1 + eG*b0/a0 + integral)/(rB+eG);
            v = (1+rB)*vL - rB*v0;
        
        }
        __device__ __forceinline__ void rangan_int::recompute_v0(Float dt, Float t0) {
            Float rB = dt/(tBack-t0) - 1;
            Float integral = (eG*dVs0 + dVs1)*dt/2.0;
            v0 = ((1+rB)*vL - b1/a1 + eG*b0/a0 + integral)/(rB+eG);
        
        }
        __device__ __forceinline__ void rangan_int::recompute_v(Float dt, Float t0) {
            Float rB = dt/(tBack-t0) - 1;
            Float integral = (eG*dVs0 + dVs1)*dt/2.0;
            v = (rB*(b1/a1 - eG*b0/a0 - integral) + eG*(1+rB)*vL)/(rB+eG);
        }
    #endif
*/

#endif
