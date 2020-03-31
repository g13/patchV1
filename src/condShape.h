#ifndef COND_SHAPE_H
#define COND_SHAPE_H
#include <cmath>
#include <iostream>
#include <string>
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

__host__ __device__
__forceinline__ void decay(Float &var, Float tau, Float dt) {
    var *= exponential(-dt/tau);
}

__host__ __device__
__forceinline__ void trip_LTD(Float &f, Float var, Float dt, // dt = t_preSpike - t_postSpike
                              Float tau_LTP, Float A_LTP,
                              Float A_ratio, Float fr_ratio,
                              Float min) 
{ 
    var *= exponential(-dt/tau_LTP);
    Float A_LTD = A_ratio*A_LTP*fr_ratio*tau_LTP;
    f -= var*A_LTD;
    if (f<min) f = min;
}

__host__ __device__
__forceinline__ void trip_LTP(Float &f, Float postVar, Float preVar, Float A_LTP, Float max) {
    f += postVar*preVar*A_LTP;
    if (f<max) f = max;
}

__host__ __device__
__forceinline__ void stdp_pre(Float &f, Float target_fr, Float postVar, Float preVar, Float tauQ, Float A_Q, Float dt, Float min, Float max) {
    postVar *= exponential(-dt/tauQ);
    f += (postVar - 2*target_fr*tauQ)*A_Q;
    if (f>max) f=max;
    else if (f<min) f=min;
}

__host__ __device__
__forceinline__ void stdp_post(Float &f, Float var, Float A_Q, Float max) {
    f += var*A_Q;
    if (f>max) f=max;
}

struct LearnVarShapeFF_E_pre { // for LGN to V1
    Size n;
    Float tauLTP[max_nLearnTypeFF_E]; // for tau_LTP, "r1" in Jen's paper
};
struct LearnVarShapeFF_I_pre { // for LGN to V1
    Size n;
    Float tauLTP[max_nLearnTypeFF_I]; // for tau_LTP, "r1" in Jen's paper
};

template<class T>
void learnFF_pre(T &l, Float tau[], Size n) {
    l.n = n;
    for (PosInt i=0; i<n; i++) {
        l.tauLTP[i] = tau[i];
    }
}

template<class T>
void printFF_pre(T &l, int EI) {
    std::cout << "FF -> ";
    if (EI) {
        std::cout << "E" ;
    } else {
        std::cout << "I";
    }
    std::cout << "(pre) learning time scales\n";
    std::cout << "#     LTP\n";
    for (PosInt i=0; i<l.n; i++) {
        std::cout << i << ",    " << l.tauLTP[i] << "\n";
    }
}

struct LearnVarShapeFF_E_post {
    Size n; // types of FF->E
    Float tau[3*max_nLearnTypeFF_E]; // tau_LTD, tau_trip, tau_avg
    Float A_LTP[max_nLearnTypeFF_E]; // A_LGN
    Float A_ratio[max_nLearnTypeFF_E];
};
struct LearnVarShapeFF_I_post {
    Size n; // types of FF->I
    Float tau[3*max_nLearnTypeFF_I]; // tau_LTD, tau_trip, tau_avg
    Float A_LTP[max_nLearnTypeFF_I]; // A_LGN
    Float A_ratio[max_nLearnTypeFF_I];
};

template<class T>
void learnFF_post(T &l, Float tauLTD[], Float tau_trip[], Float tau_avg[], Float A_LTP[], Size n) {
    l.n = n;
    for (PosInt i=0; i<n; i++) {
        l.tau[3*i+0] = tauLTD[i];
        l.tau[3*i+1] = tau_trip[i];
        l.tau[3*i+2] = tau_avg[i];
        l.A_LTP[i] = A_LTP[i];
        l.A_ratio[i] = tau_trip[i]*A_LTP[i]/(tauLTD[i]*(tau_avg[i]*tau_avg[i])); // * tau_LTP * filtered spike avg^2 / target firing rate = A_LTD
    }
}
template<class T>
void printFF_post(T &l, int EI) {
    std::cout << "FF -> ";
    if (EI) {
        std::cout << "E" ;
    } else {
        std::cout << "I";
    }
    std::cout <<"(post) learning time scales\n";
    std::cout << "#     LTD     trip       sp avg     rLTP\n";
    for (PosInt i=0; i<l.n; i++) {
        std::cout << i << ":    " << l.tau[3*i+0] << ",   " << l.tau[3*i+1] << ",   " << l.tau[3*i+2] << ",   " << l.A_LTP[i] << "\n";
    }
}

struct LearnVarShapeE { // types of E->E
    Size n; 
    Float tau[4*max_nLearnTypeE]; // tau_LTP, tau_LTD, tau_trip, tau_avg
    Float A_LTP[max_nLearnTypeE]; // A_V1
    Float A_ratio[max_nLearnTypeE];
};
void learnE(LearnVarShapeE &l, Float tauLTP[], Float tauLTD[], Float tau_trip[], Float tau_avg[], Float A_LTP[], Size n);
void printE(LearnVarShapeE &l);

struct LearnVarShapeQ {
    Size n; // types of I->E
    Float tau[2*max_nLearnTypeQ];
    Float A_LTP[max_nLearnTypeQ]; // A_Q
    Float A_LTD[max_nLearnTypeQ]; // A_Q
};
void learnQ(LearnVarShapeQ &l, Float tauQ[], Float A_Q[], Size n);
void printQ(LearnVarShapeQ &l); 
#endif
