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

__forceinline__ __host__ __device__ double comp_spike_time(double v,double v0, double vT, double dt, double t0 = 0.0) {
    return t0 + (vT-v0)/(v-v0)*dt;
}
#endif

/* remains
struct LIF {
    Size spikeCount;
    Float v, v0;
    Float a0, b0;
    Float a1, b1;
    Float vR, vThres;
    Float tRef, tBack, tsp;
    Float gL, depC;
    __device__ LIF(Float _v0, Float _tBack, Float _vR, Float _vThres, Float _gL, Float _tRef, Float dep): v0(_v0), tBack(_tBack), vR(_vR), vThres(_vThres), gL(_gL), tRef(_tRef) {
		spikeCount = 0;
		Float targetV = vR + (vThres-vR)*dep;
		depC = gL*(targetV - vL);
	};
    __device__ virtual void update(Float **var) {};
    __device__ virtual void rk2_vFixedBefore(Float dt) {}
    __device__ virtual void rk2_vFixedAfter(Float dt) {}

	__device__
	__forceinline__
	virtual void rk2(Float dt, Float noise) {
		if (noise == 0) {
	    	v = impl_rk2(dt, a0, b0, a1, b1, v0);
		} else {
			noise *= square_root(dt)*depC;
			Float fk1 = (-a0*v0 + b0)*dt;
			fk1 += noise; 
			Float v1 = v0 + fk1;
			v = v0 + fk1/2;
			v += ((-a1*v1 + b1)*dt + noise)/2;
		}
	}

    __device__ 
	__forceinline__
	virtual void reset0() {
		v0 = vR;
	}

	__device__ 
	__forceinline__
	virtual void reset1() {
	    v = vR;
	}
	
	__device__
	__forceinline__
	virtual void compute_spike_time(Float dt, Float t0) {
    	tsp = t0 + (vThres-v0)/(v-v0)*dt;
	}
	
	__device__ 
	__forceinline__
	virtual void set_p0(Float gE, Float gI) {
	    a0 = get_a(gE, gI, gL);
	    b0 = get_b(gE, gI, gL) + depC;
	}
	
	__device__ 
	__forceinline__
	virtual void set_p1(Float gE, Float gI) {
	    a1 = get_a(gE, gI, gL);
	    b1 = get_b(gE, gI, gL) + depC;
	}
};
*/
