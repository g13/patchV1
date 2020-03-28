#ifndef COND_SHAPE_H
#define COND_SHAPE_H
#include <cmath>
#include <vector>
#include "types.h"
#include "DIRECTIVE.h"
#include "CONST.h"

struct ConductanceShape {
    Float riseTime[max_ngType], decayTime[max_ngType], dod[max_ngType], coef2[max_ngType];

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
        Float etr = exponential(-dt / riseTime[ig]);
        g += f * dod[ig] * (exponential(-dt / decayTime[ig]) - etr);
        h += f * etr;
    }
    __host__ __device__ 
    __forceinline__ void decay_conductance(Float &g, Float &h, Float dt, PosInt ig) {
        Float etr = exponential(-dt / riseTime[ig]);
        Float etd = exponential(-dt / decayTime[ig]);
        g = g*etd + h*dod[ig]*(etd - etr);
        h *= etr;
    }
};

struct preOnlyVarShape { // for LGN to V1
    Float preTau[sum_nLearnType]; // for tau_LTP, "r1" in Jen's paper
    __host__ __device__ 
    __forceinline__ preOnlyVarShape(Float tau[], Size n) {
        assert(n<sum_nLearnType);
        for (PosInt i=0; i<n; i++) {
            preTau[i] = tau[i];
        }
    }
    __host__ __device__
    __forceinline__ void decay(Float &preVar, Float dt, PosInt i) {
        preVar *= exponential(-dt/preTau[i]);
    }
};

struct corticalVarShapeI {
    Size nTau, nA;
    Float tau[4*max_nTypeI]; // tau_Q, tau_LTD, tau_trip, tau_avg
    Float A_LTP[max_nTypeI]; // A_LGN
    Float A_ratio[max_nTypeI];

    __host__ __device__ 
    __forceinline__ corticalVarShapeI() {};

    __host__ __device__ 
    __forceinline__ corticalVarShapeI(Float tau_Q[], Float tau_LTD[], Float tau_trip[], Float tau_avg[], Float A_LGN[], Size n): nTau(4), nA(1) {
        assert(n<max_nLearnTypeI);
        for (PosInt i=0; i<n; i++) {
            tau[nTau*i+0] = tau_Q[i];
            tau[nTau*i+1] = tau_LTD[i];
            tau[nTau*i+2] = tau_trip[i];
            tau[nTau*i+3] = tau_avg[i];
            A_LTP[i] = A_LGN[i];
            A_ratio[i] = tau[nTau*i+2]*A_LTP[i]/(tau[nTau*i+1]*tau_avg[i]*tau_avg[i]); // * tau_LTP * mean firing rate^2 / target firing rate = A_LTD
        }
    }

    __host__ __device__
    __forceinline__ void decay(Float &var, Float dt, PosInt i) {
        var *= exponential(-dt/tau[i]);
    }
                                                                                                                    // shapeI only have LGN
    __host__ __device__                                                                            // type   // source (0, LGN; 1, V1), 
    __forceinline__ void trip_LTD(Float &f, Float var, Float dt, Float tau_LTP, Float fr_ratio, PosInt i, PosInt j, Float min) { // dt = t_preSpike - t_postSpike
        var *= exponential(-dt/tau[nTau*i+1]);
        Float A_LTD = A_ratio[nA*i+j]*A_LTP[nA*i+j]*fr_ratio*tau_LTP;
        f -= var*A_LTD;
        if (f<min) f = min;
    }

    __host__ __device__
    __forceinline__ void trip_LTP(Float &f, Float postVar, Float preVar, PosInt i, PosInt j, Float max) {
        f += postVar*preVar*A_LTP[nA*i+j];
        if (f<max) f = max;
    }
};

struct corticalVarShapeE {
    Size nTau, nA;
    Float tau[5*max_nLearnTypeE]; // tau_Q, tau_LTD, tau_trip, tau_LTP, tau_avg
    Float A_LTP[3*max_nLearnTypeE]; // A_LGN, A_V1, A_Q
    Float A_ratio[2*max_nLearnTypeE];

    __host__ __device__ 
    __forceinline__ corticalVarShapeE(Float tau_Q[], Float tau_LTD[], Float tau_trip[], Float tau_LTP[], Float tau_avg[], Float A_LGN[], Float A_V1[], Float A_Q[], Size n): nTau(5), nA(3) {
        assert(n<max_nLearnTypeE);
        for (PosInt i=0; i<n; i++) {
            tau[nTau*i+0] = tau_Q[i];
            tau[nTau*i+1] = tau_LTD[i];
            tau[nTau*i+2] = tau_trip[i];
            tau[nTau*i+3] = tau_LTP[i];
            tau[nTau*i+4] = tau_avg[i];
            A_LTP[nA*i+0] = A_LGN[i];
            A_LTP[nA*i+1] = A_V1[i];
            A_LTP[nA*i+2] = A_Q[i];
            A_ratio[(nA-1)*i+0] = tau[nTau*i+2]*A_LTP[nA*i+0]/(tau[nTau*i+1]*tau_avg[i]*tau_avg[i]); // * tau_LTP * mean firing rate^2 / target firing rate = A_LTD for LGN
            A_ratio[(nA-1)*i+1] = tau[nTau*i+2]*A_LTP[nA*i+1]/(tau[nTau*i+1]*tau_avg[i]*tau_avg[i]); // for V1
        }
    }

    __host__ __device__
    __forceinline__ void decay(Float &var, Float dt, PosInt i) {
        var *= exponential(-dt/tau[i]);
    }
                                                                                                                    // shapeI only have LGN
    __host__ __device__                                                                            // type   // source (0, LGN; 1, V1), 
    __forceinline__ void trip_LTD(Float &f, Float var, Float dt, Float tau_LTP, Float fr_ratio, PosInt i, PosInt j, Float min) { // dt = t_preSpike - t_postSpike
        var *= exponential(-dt/tau[nTau*i+1]);
        Float A_LTD = A_ratio[nA*i+j]*A_LTP[nA*i+j]*fr_ratio*tau_LTP;
        f -= var*A_LTD;
        if (f<min) f = min;
    }

    __host__ __device__
    __forceinline__ void trip_LTP(Float &f, Float postVar, Float preVar, PosInt i, PosInt j, Float max) {
        f += postVar*preVar*A_LTP[nA*i+j];
        if (f<max) f = max;
    }

    __host__ __device__
    __forceinline__ void stdp_pre(Float &f, Float target_fr, Float postVar, Float preVar, Float dt, Float min, Float max, PosInt i) {
        postVar *= exponential(-dt/tau[nTau*i+0]);
        f += (postVar - 2*target_fr*tau[nTau*i+0])*A_LTP[nA*i+2];
        if (f>max) f=max;
        else if (f<min) f=min;
    }

    __host__ __device__
    __forceinline__ void stdp_post(Float &f, Float var, PosInt i, Float max) {
        f += var*A_LTP[nA*i+2];
        if (f>max) f=max;
    }
};

#endif
