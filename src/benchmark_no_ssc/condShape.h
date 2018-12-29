#ifndef COND_SHAPE_H
#define COND_SHAPE_H
#include <cmath>

struct ConductanceShape {
    double riseTime[5], decayTime[5], dod[5], coef2[5];
    __host__ __device__ ConductanceShape() {};
    __host__ __device__ ConductanceShape(double rt[], double dt[], unsigned int ng) {
        for (unsigned int i = 0; i < ng; i++) {
            riseTime[i] = rt[i];
            decayTime[i] = dt[i];
            dod[i] = dt[i]/ (dt[i] - rt[i]);
            coef2[i] = (rt[i] + dt[i])/(rt[i]*dt[i]*2.0);
        }
    }
    __host__ __device__ __forceinline__ void compute_single_input_conductance(double *g, double *h, double f, double dt, unsigned int ig) {
        double etr = exp(-dt / riseTime[ig]);
        (*g) += f * dod[ig] * (exp(-dt / decayTime[ig]) - etr);
        (*h) += f * etr;
    }
    __host__ __device__ __forceinline__ void decay_conductance(double *g, double *h, double dt, unsigned int ig) {
        double etr = exp(-dt / riseTime[ig]);
        double etd = exp(-dt / decayTime[ig]);
        (*g) = (*g)*etd + (*h)*dod[ig]*(etd - etr);
        (*h) = (*h)*etr;
    }
};

#endif
