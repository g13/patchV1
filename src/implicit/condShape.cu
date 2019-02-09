#include "condShape.h"

#ifdef SINGLE_PRECISION
	using func = _float(*)(_float);
	__device__ func expp = &expf;
#else
	using func = _float(*)(_float);
	__device__ func expp = &exp;
#endif

__host__ __device__ ConductanceShape::ConductanceShape(_float rt[], _float dt[], unsigned int ng) {
    for (unsigned int i = 0; i < ng; i++) {
        riseTime[i] = rt[i];
        decayTime[i] = dt[i];
        dod[i] = dt[i]/ (dt[i] - rt[i]);
        coef2[i] = (rt[i] + dt[i])/(rt[i]*dt[i]*2.0f);
    }
}
__device__ __forceinline__ void ConductanceShape::compute_single_input_conductance(_float &g, _float &h, _float f, _float dt, unsigned int i) {
    _float etr = expp(-dt / riseTime[i]);
    g += f * dod[i] * (expp(-dt / decayTime[i]) - etr);
    h += f * etr;
}
__device__ __forceinline__ void ConductanceShape::decay_conductance(_float &g, _float &h, _float dt, unsigned int i) {
    _float etr = expp(-dt / riseTime[i]);
    _float etd = expp(-dt / decayTime[i]);
    g = g*etd + h*dod[i]*(etd - etr);
    h = h*etr;
}
__device__ __forceinline__ void ConductanceShape::diff_and_int_cond(_float &ig, _float &g, _float &h, _float &dg, _float f, _float dt, unsigned int i) {
    _float etr = expp(-dt / riseTime[i]);
    _float etd = expp(-dt / decayTime[i]);
    _float f_mod = f*dod[i];
	ig += f_mod * (decayTime[i] * (1 - etd) - riseTime[i] * (1 - etr));
	dg += f_mod * (etr / riseTime[i] - etd / decayTime[i]);
    g += f_mod*(etd - etr);
    h += f * etr;
}
__device__ __forceinline__ void ConductanceShape::diff_and_int_decay(_float &ig, _float &g, _float &h, _float &dg, _float dt, unsigned int i) {
    _float etr = expp(-dt / riseTime[i]);
    _float etd = expp(-dt / decayTime[i]);
    _float h_mod = h*dod[i];
	ig = decayTime[i] * (1 - etd)*(g + h_mod) - riseTime[i] * (1 - etr)*h_mod;
	dg = etr / riseTime[i] * h_mod - etd / decayTime[i] * (g + h_mod);
    g = g*etd + h_mod*(etd - etr);
    h = h*etr;
}