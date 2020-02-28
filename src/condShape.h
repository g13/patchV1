#ifndef COND_SHAPE_H
#define COND_SHAPE_H
#include <cmath>
#include "types.h"

struct ConductanceShape {
    Float riseTime[5], decayTime[5], dod[5], coef2[5];
    __host__ __device__ 
    __forceinline__ ConductanceShape() {};

    __host__ __device__ 
    __forceinline__ ConductanceShape(Float rt[], Float dt[], Size ng) {
        for (PosInt i = 0; i < ng; i++) {
            riseTime[i] = rt[i];
            decayTime[i] = dt[i];
            dod[i] = dt[i]/ (dt[i] - rt[i]);
            coef2[i] = (rt[i] + dt[i])/(rt[i]*dt[i]*2.0);
        }
    }
    __host__ __device__ 
    __forceinline__ void compute_single_input_conductance(Float &g, Float &h, Float f, Float dt, PosInt ig) {
        Float etr = exp(-dt / riseTime[ig]);
        g += f * dod[ig] * (exp(-dt / decayTime[ig]) - etr);
        h += f * etr;
    }
    __host__ __device__ 
    __forceinline__ void decay_conductance(Float &g, Float &h, Float dt, PosInt ig) {
        Float etr = exp(-dt / riseTime[ig]);
        Float etd = exp(-dt / decayTime[ig]);
        g = g*etd + h*dod[ig]*(etd - etr);
        h *= etr;
    }
};

#endif
