#ifndef INPUT_ARGS_CA1_H
#define INPUT_ARGS_CA1
#include "input_args.h"

struct input_args_CA1 : input_args {
    string libFile;
    double vThres;
    double vRest;
    double vTol;
    double rLinear;
    double vBuffer;
    double dendClampRatio;
    double ignoreT;
    double tRef;
    double trans;
    double trans0;
    double rBiLinear;
    int kVStyle;
    int afterSpikeBehavior;
    unsigned int vInit;
    input_args_CA1();
    using input_args::read;
};

typedef struct input_args_CA1 InputArgsCA1;
#endif
