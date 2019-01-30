#ifndef COND_SHAPE_H
#define COND_SHAPE_H
#include <cmath>
#include <cassert>

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
    __host__ __device__ __forceinline__ void compute_single_input_conductance(double &g, double &h, double f, double dt, unsigned int i) {
        double etr = exp(-dt / riseTime[i]);
        g += f * dod[i] * (exp(-dt / decayTime[i]) - etr);
        h += f * etr;
    }
    __host__ __device__ __forceinline__ void decay_conductance(double &g, double &h, double dt, unsigned int i) {
        double etr = exp(-dt / riseTime[i]);
        double etd = exp(-dt / decayTime[i]);
        g = g*etd + h*dod[i]*(etd - etr);
        h = h*etr;
    }
    __host__ __device__ __forceinline__ void diff_and_int_cond(double &ig, double &g, double &h, double &dg, double f, double dt, unsigned i) {
        double etr = exp(-dt / riseTime[i]);
        double etd = exp(-dt / decayTime[i]);
        double f_mod = f*dod[i];
		ig += f_mod * (decayTime[i] * (1 - etd) - riseTime[i] * (1 - etr));
		dg += f_mod * (etr / riseTime[i] - etd / decayTime[i]);
        g += f_mod*(etd - etr);
        h += f * etr;
    }
    __host__ __device__ __forceinline__ void diff_and_int_decay(double &ig, double &g, double &h, double &dg, double dt, unsigned i) {
        double etr = exp(-dt / riseTime[i]);
        double etd = exp(-dt / decayTime[i]);
        double h_mod = h*dod[i];
		ig = decayTime[i] * (1 - etd)*(g + h_mod) - riseTime[i] * (1 - etr)*h_mod;
		dg = etr / riseTime[i] * h_mod - etd / decayTime[i] * (g + h_mod);
        g = g*etd + h_mod*(etd - etr);
        h = h*etr;
    }
};

#endif
