#include "rk2.h"

__device__ _float get_a(_float gE, _float gI, _float gL) {
    return gE + gI + gL;
}

__device__ _float get_b(_float gE, _float gI, _float gL) {
    return gE * vE + gI * vI + gL * vL;
}

__device__ _float comp_spike_time(_float v,_float v0, _float dt, _float t0 = 0.0) {
    return t0 + (vT-v0)/(v-v0)*dt;
}

__device__ _float recomp_v0(_float A, _float B, _float rB) {
    return ((1+rB)*vL - B)/(A+rB);
}

__device__ _float eval_fk(_float a, _float b, _float v) {
    return -a * v + b;
}

__device__ void Runge_Kutta_2::compute_spike_time(_float dt, _float t0) {
    tsp = comp_spike_time(v, v0, dt, t0);
}

__device__ void Runge_Kutta_2::set_p0(_float gE, _float gI, _float gL) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

__device__ void Runge_Kutta_2::set_p1(_float gE, _float gI, _float gL) {
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

#if SCHEME == 0
__device__ void rk2::compute_v(_float dt) {
    _float fk0 = eval_fk(a0, b0, v0);
    _float fk1 = eval_fk(a1, b1, v0 + dt*fk0);
    v = v0 + dt*(fk0+fk1)/2.0f;
}

__device__ void rk2::recompute(_float dt, _float t0) {
    recompute_v0(dt, t0);
    compute_v(dt);
}

__device__ void rk2::recompute_v(_float dt, _float t0) {
    recompute(dt, t0);
}

__device__ void rk2::recompute_v0(_float dt, _float t0) {
    _float A = a0*a1*dt - a0 - a1;
    _float B = b0 + b1 - a1*b0*dt;
    _float t = tBack-t0;
    v0 = (2*vL-t*B)/(2+t*A);
}
#endif

#if SCHEME == 1
__device__ void impl_rk2::compute_v(_float dt) {
    v = (2*v0 + (-a0*v0+b0+b1)*dt)/(2+a1*dt);
}

__device__ void impl_rk2::recompute(_float dt, _float t0) {
    _float rB = dt/(tBack-t0) - 1; 
    _float denorm = 2 + a1*dt;
    _float A = (2 - a0*dt)/denorm;
    _float B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
    v = (A*(1+rB)*vL + rB*B)/(A+rB);
}

__device__ void impl_rk2::recompute_v(_float dt, _float t0) {
    _float rB = dt/(tBack-t0) - 1; 
    _float denorm = 2 + a1*dt;
    _float A = (2 - a0*dt)/denorm;
    _float B = (b0 + b1)*dt/denorm;
    v = (A*(1+rB)*vL + rB*B)/(A+rB);
}

__device__ void impl_rk2::recompute_v0(_float dt, _float t0) {
    _float rB = dt/(tBack-t0) - 1; 
    _float denorm = 2 + a1*dt;
    _float A = (2 - a0*dt)/denorm;
    _float B = (b0 + b1)*dt/denorm;
    v0 = recomp_v0(A, B, rB);
}
#endif

#if SCHEME == 2
__device__ void rangan_int::set_dVs0(_float dgE, _float dgI) {
    _float da = dgE + dgI;
    _float db = dgE*vE + dgI*vI;
    dVs0 = (db - b0/a0*da)/a0;
}

__device__ void rangan_int::set_dVs1(_float dgE, _float dgI) {
    _float da = dgE + dgI;
    _float db = dgE*vE + dgI*vI;
    dVs1 = (db - b1/a1*da)/a1;
}

__device__ void rangan_int::set_G(_float G, _float gL, _float dt) {
    eG = exp(-G-gL*dt);
}

__device__ void rangan_int::compute_v(_float dt) {
	v = b1/a1 + eG*(v0 - b0/a0) - (eG*dVs0 + dVs1)*dt/2.0;
}

__device__ void rangan_int::recompute(_float dt, _float t0) {
    _float rB = dt/(tBack-t0) - 1;
    _float integral = (eG*dVs0 + dVs1)*dt/2.0;
    v0 = ((1+rB)*vL - b1/a1 + eG*b0/a0 + integral)/(rB+eG);
    v = (1+rB)*vL - rB*v0;

}
__device__ void rangan_int::recompute_v0(_float dt, _float t0) {
    _float rB = dt/(tBack-t0) - 1;
    _float integral = (eG*dVs0 + dVs1)*dt/2.0;
    v0 = ((1+rB)*vL - b1/a1 + eG*b0/a0 + integral)/(rB+eG);

}
__device__ void rangan_int::recompute_v(_float dt, _float t0) {
    _float rB = dt/(tBack-t0) - 1;
    _float integral = (eG*dVs0 + dVs1)*dt/2.0;
    v = (rB*(b1/a1 - eG*b0/a0 - integral) + eG*(1+rB)*vL)/(rB+eG);
}
#endif
