#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>
#include <ctime>
#include <cmath>
#include <fenv.h>
#include <boost/program_options.hpp>

#include "connect.h"
#include "../types.h"
#include "../util/po.h" // custom validator for reading vector in the configuration file
#include "../util/util.h"

#define nFunc 2

__device__
Float ODpref(Float post, Float pre) { // ocular dominance
    Float p;
    if (post*pre < 0) {
        p = 0.0;
    } else {
        p = 1.0;
    }
    return p; 
}
__device__ pFeature p_OD = ODpref;

__device__
Float OPpref(Float post, Float pre) { // orientation preference
    Float dp = pre-post;
    Float sig = M_PI/12; // set spread here
    Float A = square_root(2*M_PI);
    Float p = exponential(-dp*dp/(2*sig*sig))/(A*sig);
    return p; 
}
__device__ pFeature p_OP = OPpref;

__device__
Float pass(Float post, Float pre) {
    return 1.0;
}
__device__ pFeature p_pass = pass;

__device__ __constant__ pFeature pref[nFunc];

void initializePreferenceFunctions(Size nFeature) {
    pFeature h_pref[nFunc];
    if (nFunc != nFeature) {
        std::cout << "number of preference functions = " << nFunc << ", is inconsistent with the number of the features: " << nFeature << "\n";
        assert(nFunc == nFeature);
    }
    checkCudaErrors(cudaMemcpyFromSymbol(&h_pref[0], p_OD, sizeof(pFeature), 0, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_pref[1], p_pass, sizeof(pFeature), 0, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpyToSymbol(pref, h_pref, nFunc*sizeof(pFeature), 0, cudaMemcpyHostToDevice));
}

/* TODO:
template <typename T, typename I>
void check_statistics(T* array, I n, T &max, T &min, T &mean, T &std) {

}
*/
