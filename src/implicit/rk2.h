#ifndef RK2_H
#define RK2_H
#include <cassert>
#include "CONST.h"
#include "DIRECTIVE.h"
#include "stdio.h"

__device__ double get_a(double gE, double gI, double gL) {
    return gE + gI + gL;
}

__device__ double get_b(double gE, double gI, double gL) {
    return gE * vE + gI * vI + gL * vL;
}

__device__ double comp_spike_time(double v,double v0, double dt, double t0 = 0.0) {
    return t0 + (vT-v0)/(v-v0)*dt;
}

__device__ double recomp_v0(double A, double B, double rB) {
    return ((1+rB)*vL - B)/(A+rB);
}

__device__ double eval_fk(double a, double b, double v) {
    return -a * v + b;
}


typedef struct Runge_Kutta_2 {
    unsigned int spikeCount;
    bool correctMe;
    double v, v0;
    double tBack, tsp;
    double a0, b0;
    double a1, b1;
    __device__ Runge_Kutta_2(double _v0, double _tBack): v0(_v0), tBack(_tBack) {};

    __device__ void compute_spike_time(double dt, double t0 = 0.0f) {
        tsp = comp_spike_time(v, v0, dt, t0);
    }
    
    __device__ void set_p0(double gE, double gI, double gL) {
        a0 = get_a(gE, gI, gL);
        b0 = get_b(gE, gI, gL); 
    }
    
    __device__ void set_p1(double gE, double gI, double gL) {
        a1 = get_a(gE, gI, gL);
        b1 = get_b(gE, gI, gL); 
    }
    
    __device__ void transfer_p1_to_p0() {
        a0 = a1;
        b0 = b1;
    }
    
    __device__ void reset_v() {
        v = vL;
    }
    __device__ virtual void compute_v(double dt) = 0;
    __device__ virtual void recompute(double dt, double t0=0.0f) = 0;
    __device__ virtual void recompute_v0(double dt, double t0=0.0f) = 0;
    __device__ virtual void recompute_v(double dt, double t0=0.0f) = 0;
    
} LIF;

#if SCHEME == 0
struct rk2: Runge_Kutta_2 {
    __device__ rk2(double _v0, double _tBack): Runge_Kutta_2(_v0, _tBack) {};

    __device__ void rk2::compute_v(double dt) {
        double fk0 = eval_fk(a0, b0, v0);
        double fk1 = eval_fk(a1, b1, v0 + dt*fk0);
        v = v0 + dt*(fk0+fk1)/2.0f;
    }
    
    __device__ void rk2::recompute(double dt, double t0 = 0.0f) {
        recompute_v0(dt, t0);
        compute_v(dt);
    }
    
    __device__ void rk2::recompute_v(double dt, double t0 = 0.0f) {
        recompute(dt, t0);
    }
    
    __device__ void rk2::recompute_v0(double dt, double t0 = 0.0f) {
        double A = a0*a1*dt - a0 - a1;
        double B = b0 + b1 - a1*b0*dt;
        double t = tBack-t0;
        v0 = (2*vL-t*B)/(2+t*A);
    }
};
#endif

#if SCHEME == 1
struct impl_rk2: Runge_Kutta_2 {
    double denorm;
    __device__ impl_rk2(double _v0, double _tBack): Runge_Kutta_2(_v0, _tBack) {};

    __device__ void compute_v(double dt) {
        v = (2*v0 + (-a0*v0+b0+b1)*dt)/(2+a1*dt);
    }
    
    __device__ void recompute(double dt, double t0 = 0.0f) {
        double rB = dt/(tBack-t0) - 1; 
        double denorm = 2 + a1*dt;
        double A = (2 - a0*dt)/denorm;
        double B = (b0 + b1)*dt/denorm;
        v0 = recomp_v0(A, B, rB);
        v = (A*(1+rB)*vL + rB*B)/(A+rB);
    }
    
    __device__ void recompute_v(double dt, double t0 = 0.0f) {
        double rB = dt/(tBack-t0) - 1; 
        double denorm = 2 + a1*dt;
        double A = (2 - a0*dt)/denorm;
        double B = (b0 + b1)*dt/denorm;
        v = (A*(1+rB)*vL + rB*B)/(A+rB);
    }
    
    __device__ void recompute_v0(double dt, double t0 = 0.0f) {
        double rB = dt/(tBack-t0) - 1; 
        double denorm = 2 + a1*dt;
        double A = (2 - a0*dt)/denorm;
        double B = (b0 + b1)*dt/denorm;
        v0 = recomp_v0(A, B, rB);
    }
};
#endif

#if SCHEME == 2
typedef struct rangan_int: Runge_Kutta_2 {
    double eG;
    double dVs0, dVs1;
    __device__ rangan_int(double _v0, double _tBack, double _dVs): Runge_Kutta_2(_v0, _tBack), dVs0(_dVs) {};
    __device__ void set_dVs0(double dgE, double dgI) {
        double da = dgE + dgI;
        double db = dgE*vE + dgI*vI;
        dVs0 = (db - b0/a0*da)/a0;
    }
    
    __device__ void set_dVs1(double dgE, double dgI) {
        double da = dgE + dgI;
        double db = dgE*vE + dgI*vI;
        dVs1 = (db - b1/a1*da)/a1;
    }
    
    __device__ void set_G(double G, double gL, double dt) {
        eG = exp(-G-gL*dt);
    }
    
    __device__ void compute_v(double dt) {
    	v = b1/a1 + eG*(v0 - b0/a0) - (eG*dVs0 + dVs1)*dt/2.0;
    }
    
    __device__ void recompute(double dt, double t0 = 0.0f) {
        double rB = dt/(tBack-t0) - 1;
        double integral = (eG*dVs0 + dVs1)*dt/2.0;
        v0 = ((1+rB)*vL - b1/a1 + eG*b0/a0 + integral)/(rB+eG);
        v = (1+rB)*vL - rB*v0;
    
    }
    __device__ void recompute_v0(double dt, double t0 = 0.0f) {
        double rB = dt/(tBack-t0) - 1;
        double integral = (eG*dVs0 + dVs1)*dt/2.0;
        v0 = ((1+rB)*vL - b1/a1 + eG*b0/a0 + integral)/(rB+eG);
    
    }
    __device__ void recompute_v(double dt, double t0 = 0.0f) {
        double rB = dt/(tBack-t0) - 1;
        double integral = (eG*dVs0 + dVs1)*dt/2.0;
        v = (rB*(b1/a1 - eG*b0/a0 - integral) + eG*(1+rB)*vL)/(rB+eG);
    }
} rangan;
#endif

#endif
