#ifndef COND_SHAPE_H
#define COND_SHAPE_H

#include "DIRECTIVE.h"
#include "CONST.h"

struct ConductanceShape {
    _float riseTime[max_ngType], decayTime[max_ngType], dod[max_ngType], coef2[max_ngType];
    __host__ __device__ ConductanceShape() {};
    __host__ __device__ ConductanceShape(_float rt[], _float dt[], unsigned int ng);
    __device__ __forceinline__ void compute_single_input_conductance(_float &g, _float &h, _float f, _float dt, unsigned int i);
    __device__ __forceinline__ void decay_conductance(_float &g, _float &h, _float dt, unsigned int i);
    __device__ __forceinline__ void diff_and_int_cond(_float &ig, _float &g, _float &h, _float &dg, _float f, _float dt, unsigned int i);
    __device__ __forceinline__ void diff_and_int_decay(_float &ig, _float &g, _float &h, _float &dg, _float dt, unsigned int i);
};

#endif
