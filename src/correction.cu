#include "correction.h"

// initialize Potential Firing Squad
__device__ void determine_PFS() {

}

__global__ void aim_and_eliminate_PFS() {

}

__host__ __device__ double dV(ConductanceShape cond, double dt, double dgt, unsigned int ig, double v_hlf, double v_R) {
    double dg = cond.dg_approx(dgt, ig);
    return -dg*(v_hlf - v_R)*dt/2.0;
}

#endif
