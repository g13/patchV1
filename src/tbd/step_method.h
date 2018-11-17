#ifdef STEP_METHOD_H
#ifdef STEP_METHOD_H
#include "func.h"
#include "cuda_precision.h"


// basic inteface of step methods for interaction between neurons through spikes
struct Step_Method {
    //initialize
    float_point tsp;
    float_point v0;
    float_point t;
    //flag
    int verbose;

    Step_Method(float_point _v0, float_point t0, int _verbose = 0): v0(_v0), t(_t0), verbose(_verbose) {}
    // stepping function, default value is binded to base at compile time
    virtual step(float_point dt, int _verbose = 0){};
};
typedef struct Step_Method step_method;

// direct integration method from Rangan_Cai 2007
struct Rangan_Cai: Step_Method{
    float_point intGs;
    // step one dt of simulation
    void step(float_point dt, int verbose);
    // evaluate \int_{t_{i}}^{t_{i+1}}{G_{s}(t)dt}
    void get_intGs();
    // evaluate Rangan_Cai_Integrand from func.h
    void get_vsDecayInt();
    // complete current step
}
typedef struct Rangan_Cai mNYU; //method from NYU model

// modified RK2 from Shelley_Tao 2001
struct Shelley_Tao_RK2: Step_Method{

}
typedef struct Shelley_Tao_RK mRK2; //modified RK2

/*
// modified RK4 from Shelley_Tao 2001
struct Shelley_Tao_RK4: Step_Method{

}
typedef struct Shelley_Tao_RK4 mRK4; //modified RK4
*/
#endif
