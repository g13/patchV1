#include "rk2.h"

__forceinline__ __host__ __device__ double get_a(double gE, double gI, double gL) {
    return gE + gI + gL;
}

__forceinline__ __host__ __device__ double get_b(double gE, double gI, double gL) {
    return gE * vE + gI * vI + gL * vL;
}

__forceinline__ __host__ __device__ double comp_spike_time(double v,double v0, double dt, double t0 = 0.0) {
    return t0 + (vT-v0)/(v-v0)*dt;
}

__forceinline__ __host__ __device__ double recomp_v0(double A, double B, double rB) {
    return ((1+rB)*vL - B)/(A+rB);
}

__forceinline__  __host__ __device__ double eval_fk(double a, double b, double v) {
    return -a * v + b;
}

__device__ void Runge_Kutta_2::compute_spike_time(double dt, double t0) {
    tsp = comp_spike_time(v, v0, dt, t0);
}

__device__ void Runge_Kutta_2::set_p0(double gE, double gI, double gL) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

__device__ void Runge_Kutta_2::set_p1(double gE, double gI, double gL) {
    a1 = get_a(gE, gI, gL);
    b1 = get_b(gE, gI, gL); 
}

__device__ void Runge_Kutta_2::transfer_p1_to_p0() {
    a0 = a1;
    b0 = b1;
}

__device__ void Runge_Kutta_2::reset_v() {
    v = vL;
}

__device__ double rk2::eval0(double _v) {
    return eval_fk(a0,b0,_v);
}

__device__ double rk2::eval1(double _v) {
    return eval_fk(a1,b1,_v);
}

__device__ void rk2::compute_v(double dt) {
    double fk0 = eval0(v0);
    double fk1 = eval1(v0 + dt*fk0);
    v = v0 + dt*(fk0+fk1)/2.0f;
}

__device__ void rk2::recompute(double dt, double t0) {
    recompute_v0(dt, t0);
    rk2(dt);
}

__device__ void rk2::recompute_v(double dt, double t0) {
    recompute(dt, t0);
}

__device__ void rk2::recompute_v0(double dt, double t0) {
    double A = (a0*a1*dt - a0 - a1) * dt/2.0f;
    double B = (b0 + b1 - a1*b0*dt) * dt/2.0f;
    double t = tBack-t0;
    v0 = (vL-t*B/(1.0f+t*A);
}

__device__ void impl_rk2::compute_v(double dt) {
    return (2*v0 + (-a0*v0+b0+b1)*dt)/(2+a1*dt);
}

__device__ void impl_rk2::recompute(double dt, double t0) {
    double rB = dt/(tBack-t0) - 1; 
    double denorm = 2 + a1*dt;
    double A = (2 - a0*dt)/denorm;
    double B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
    v = A*v0 + B;
}

__device__ void impl_rk2::recompute_v(double dt, double t0) {
    double rB = dt/(tBack-t0) - 1; 
    double denorm = 2 + a1*dt;
    double A = (2 - a0*dt)/denorm;
    double B = (b0 + b1)*dt/denorm;
    v = (A*(1+rB)*vL + rB*B)/(A+rB);
}

__device__ void impl_rk2::recompute_v0(double dt, double t0) {
    double rB = dt/(tBack-t0) - 1; 
    double denorm = 2 + a1*dt;
    double A = (2 - a0*dt)/denorm;
    double B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
}

__device__ void impl_rk2::set_p0(double gE, double gI, double gL) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

__device__ void impl_rk2::set_p1(double gE, double gI, double gL) {
    a1 = get_a(gE, gI, gL);
    b1 = get_b(gE, gI, gL); 
}

