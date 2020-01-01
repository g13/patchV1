#ifndef COND_SHAPE_H
#define COND_SHAPE_H

#include "../MACRO.h"

struct ConductanceShape {
    _float riseTime[max_ngType], decayTime[max_ngType], dod[max_ngType], coef2[max_ngType];
    __host__ __device__ __forceinline__ ConductanceShape() {};
    __host__ __device__ __forceinline__ ConductanceShape(_float rt[], _float dt[], unsigned int ng);
    __device__ __forceinline__ void compute_single_input_conductance(_float &g, _float &h, _float f, _float dt, unsigned int i);
    __device__ __forceinline__ void decay_conductance(_float &g, _float &h, _float dt, unsigned int i);
    __device__ __forceinline__ void diff_and_int_cond(_float &ig, _float &g, _float &h, _float &dg, _float f, _float dt, unsigned int i);
    __device__ __forceinline__ void diff_and_int_decay(_float &ig, _float &g, _float &h, _float &dg, _float dt, unsigned int i);
};

__host__ __device__ __forceinline__ ConductanceShape::ConductanceShape(_float rt[], _float dt[], unsigned int ng) {
    assert(ng < max_ngType);
    for (unsigned int i = 0; i < ng; i++) {
        riseTime[i] = rt[i];
        decayTime[i] = dt[i];
        dod[i] = dt[i]/ (dt[i] - rt[i]);
        coef2[i] = (rt[i] + dt[i])/(rt[i]*dt[i]*2.0f);
    }
}
__device__ __forceinline__ void ConductanceShape::compute_single_input_conductance(_float &g, _float &h, _float f, _float dt, unsigned int i) {
    _float etr = exponential(-dt / riseTime[i]);
    g += f * dod[i] * (exponential(-dt / decayTime[i]) - etr);
    h += f * etr;
}
__device__ __forceinline__ void ConductanceShape::decay_conductance(_float &g, _float &h, _float dt, unsigned int i) {
    _float etr = exponential(-dt / riseTime[i]);
    _float etd = exponential(-dt / decayTime[i]);
    g = g*etd + h*dod[i]*(etd - etr);
    h = h*etr;
}
__device__ __forceinline__ void ConductanceShape::diff_and_int_cond(_float &ig, _float &g, _float &h, _float &dg, _float f, _float dt, unsigned int i) {
    _float etr = exponential(-dt / riseTime[i]);
    _float etd = exponential(-dt / decayTime[i]);
    _float f_mod = f*dod[i];
	ig += f_mod * (decayTime[i] * (1 - etd) - riseTime[i] * (1 - etr));
	dg += f_mod * (etr / riseTime[i] - etd / decayTime[i]);
    g += f_mod*(etd - etr);
    h += f * etr;
}
__device__ __forceinline__ void ConductanceShape::diff_and_int_decay(_float &ig, _float &g, _float &h, _float &dg, _float dt, unsigned int i) {
    _float etr = exponential(-dt / riseTime[i]);
    _float etd = exponential(-dt / decayTime[i]);
    _float h_mod = h*dod[i];
	ig = decayTime[i] * (1 - etd)*(g + h_mod) - riseTime[i] * (1 - etr)*h_mod;
	dg = etr / riseTime[i] * h_mod - etd / decayTime[i] * (g + h_mod);
    g = g*etd + h_mod*(etd - etr);
    h = h*etr;
}

#endif
