#ifndef FUNC_H
#define FUNC_H
#include <cmath>
#include <iostream>
#include <iomanip>
#include "cuda_precision.h"



// struct for member variables

typedef struct Remain {

    float_point g0;
    Remain(float_point _g0 = 0.0): g0(_g0) {};

} Rem;

typedef struct RemainAlpha: Remain {

    float_point h0;
    RemainAlpha(float_point _g0 = 0.0, float_point _h0 = 0.0): Remain(_g0), h0(_h0) {};

} RemAlpha;

typedef struct CondParam {

    float_point f; // strength
    float_point tau_decay; // decay time constant
    float_point e; //reversal potential
    CondParam(float_point _f, float_point _tau_decay, float_point _e): f(_f), tau_decay(_tau_decay), e(_e) {};

} CondP;

typedef struct CondAlphaParam: CondParam {

    float_point tau_rise;
    CondAlphaParam(float_point _f, float_point _tau_rise, float_point _tau_decay, float_point _e): CondParam(_f, _tau_decay, _e), tau_rise(_tau_rise) {};

} CondAlphaP;

// 1D real Functor
typedef struct Functor1 {

    Functor1(){};

    virtual float_point eval(float_point t) = 0;
    virtual void display() = 0;

} Func1;

typedef struct Conductance {

    int cond_id;
    float_point dg, g, ig;  // cond evals at t, derivative(g), g, integral(g)
    float_point dg_rem, g_rem, ig_rem;  // remaining cond evals at t
    float_point* condInt; // for storage on the integral limits: \int(a,b) = \int(0,b)-\int(0,a)
    float_point condRemInt; // condInt for remaining conductance from last step
    float_point g_dt; // takeaway for next dt
    Conductance();

    virtual void book_keeping(float_point *tIn, int n, float_point dt);
    virtual void eval(float_point t);
    virtual void eval_rem(float_point t);
    virtual void advance(float_point *dtIn, int n, float_point dt);

    virtual void set_dtTempVar(float_point t) = 0;
    virtual float_point eval_ig(float_point t) = 0;
    virtual float_point eval_ig_rem(float_point t) = 0;
    virtual float_point eval_g(float_point t) = 0;
    virtual float_point eval_g_rem(float_point t) = 0;
    virtual float_point eval_dg(float_point t) = 0;
    virtual float_point eval_dg_rem(float_point t) = 0;
    virtual float_point get_e() = 0;
    virtual float_point get_last_g() = 0;
    virtual void evolve_init() {}; // evolve extra variables if have any
    virtual void evolve_dt(float_point t) {}; // evolve extra variables if have any
    virtual void evolve_dt_rem(float_point t) {}; 
    virtual void next_rem() = 0; // for Remain types
    virtual void reset(Rem* _r) = 0; // for Remain types
    virtual void display() = 0;
    ~Conductance();

} Cond;

// conductance shapes
typedef struct Conductance_Exp: Conductance {

    Rem r;
    CondP p;
    float_point etd; // reusable temp variables during dt
    Conductance_Exp(CondP _p, Rem _r);
    
    virtual inline void set_dtTempVar(float_point t);
    virtual float_point eval_ig(float_point t);
    virtual float_point eval_ig_rem(float_point t);
    virtual float_point eval_g(float_point t);
    virtual float_point eval_g_rem(float_point t);
    virtual float_point eval_dg(float_point t);
    virtual float_point eval_dg_rem(float_point t);
    virtual float_point get_e();
    virtual float_point get_last_g();
    virtual void next_rem(); // for Remain types
    virtual void reset(Rem* _r); // for Remain types
    virtual void display();

} CondExp;

typedef struct Conductance_Alpha: Conductance {
   
    RemAlpha r;
    CondAlphaP p;
    float_point dtau, f_rconst, rconst; // const during simulation
    float_point h0_rconst; // const during dt 
    float_point etr, etd; // reusable temp variables during dt
    float_point h_dt;
    Conductance_Alpha(CondAlphaP _p, RemAlpha _r);

    virtual inline void set_dtTempVar(float_point t);
    virtual float_point eval_ig(float_point t);
    virtual float_point eval_ig_rem(float_point t);
    virtual float_point eval_g(float_point t);
    virtual float_point eval_g_rem(float_point t);
    virtual float_point eval_dg(float_point t);
    virtual float_point eval_dg_rem(float_point t);
    virtual float_point get_e();
    virtual float_point get_last_g();
    virtual void evolve_init(); // evolve extra variables if have any
    virtual void evolve_dt(float_point t);
    virtual void evolve_dt_rem(float_point t);
    virtual void next_rem(); // for Remain types
    virtual void reset(Rem* _r); // for Remain types
    virtual void display();


} CondAlpha;

// Conductance wrappers for Func1
typedef struct Functor1_wrapper_Condutance_dg: Functor1 {
    Cond *cond;
    Functor1_wrapper_Condutance_dg(Cond *_cond): cond(_cond) {};
    virtual float_point eval(float_point t) {
        cond->set_dtTempVar(t);
        return cond->eval_dg(t);
    }
    virtual void display() { 
        cond->display();
    };
} Func1Cond_dg;

typedef struct Functor1_wrapper_Condutance_dg_rem: Functor1 {
    Cond *cond;
    Functor1_wrapper_Condutance_dg_rem(Cond *_cond): cond(_cond) {};
    virtual float_point eval(float_point t) {
        cond->set_dtTempVar(t);
        return cond->eval_dg_rem(t);
    }
    virtual void display() { 
        cond->display();
    };
} Func1Cond_dgr;

typedef struct Functor1_wrapper_Condutance_g: Functor1 {
    Cond *cond;
    Functor1_wrapper_Condutance_g(Cond *_cond): cond(_cond) {};
    virtual float_point eval(float_point t) {
        cond->set_dtTempVar(t);
        return cond->eval_g(t);
    }
    virtual void display() { 
        cond->display();
    };
} Func1Cond_g;

typedef struct Functor1_wrapper_Condutance_g_rem: Functor1 {
    Cond *cond;
    Functor1_wrapper_Condutance_g_rem(Cond *_cond): cond(_cond) {};
    virtual float_point eval(float_point t) {
        cond->set_dtTempVar(t);
        return cond->eval_g_rem(t);
    }
    virtual void display() { 
        cond->display();
    };
} Func1Cond_gr;

/* Functions to be Evaluated by Romberg Integration (qromb) */

// integrand for the decay of Vs from Rangan_Cai 2007, one for each neuron if parallel
typedef struct Rangan_Cai_Integrand: Functor1 {

    int ntype; // # types of conductances 
    Cond **cond; // conductance shapes [shape,rem]
    int *n;  // # inputs
    float_point dt; // step size
    float_point **tIn; // time of inputs
    float_point **dtIn; // dt - time of inputs
    float_point gL, eL; // constant leaky conductance, and reversal potential
    float_point integrated_factor;  // \int_t0^t1{exp(-Gs)}
    // temp vars
    float_point dGs, dGsEs;
    float_point Gs, GsEs;
    float_point GsInt, integrating_factor, dVs;

    Rangan_Cai_Integrand(int _ntype, Cond **_cond, float_point _gL, float_point _eL);
    void step(float_point _dt, float_point **_tIn, int *_n);
    void finish_step();

    virtual float_point eval(float_point t);
    virtual void display();
    void reset_dt(int *n, float_point _dt);
    void book_keeping();
    void book_lookup();
    ~Rangan_Cai_Integrand();

} mNYU_vsDecayInt;


/* transcendental equation from modified RK4 method in Rangan_Cai 2007
typedef struct Shelley_Tao_RK4_spikeTime: Functor1 {

} RK4_tsp;
*/

#endif
