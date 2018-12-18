#ifndef CCOND_SHAPE_H
#define CCOND_SHAPE_H
#include <cmath>

struct cConductanceShape {
    double riseTime[5], decayTime[5], dod[5], coef2[5];
    cConductanceShape() {};
    cConductanceShape(double rt[], double dt[], unsigned int ng) {
        for (int i = 0; i < ng; i++) {
            riseTime[i] = rt[i];
            decayTime[i] = dt[i];
            dod[i] = dt[i]/ (dt[i] - rt[i]);
            coef2[i] = (rt[i] + dt[i])/(rt[i]*dt[i]*2.0);
        }
    }
    inline void compute_single_input_conductance(double *g, double *h, double f, double dt, unsigned int ig) {
        double etr = exp(-dt / riseTime[ig]);
        (*g) += f * dod[ig] * (exp(-dt / decayTime[ig]) - etr);
        (*h) += f * etr;
    }
    inline void decay_conductance(double *g, double *h, double dt, unsigned int ig) {
        double etr = exp(-dt / riseTime[ig]);
        double etd = exp(-dt / decayTime[ig]);
        (*g) = (*g)*etd + (*h)*dod[ig]*(etd - etr);
        (*h) = (*h)*etr;
    }
    
    double dg_approx(double dgt, unsigned int ig) {
        return (1.0 - coef2[ig] * dgt)*dgt / riseTime[ig];
    }
    double dg(double dgt, unsigned int ig) {
        return dod[ig] * (exp(-dgt / decayTime[ig]) - exp(-dgt / riseTime[ig]));
    }
};

#endif
