#include "coredynamics.cuh"
#include <fstream>
#include <cassert>
#include <ctime>
#include <cmath>
#include "cpu.h"

extern __constant__ double vE[1], vI[1], vT[1];
extern __constant__ double gL_E[1], gL_I[1];
extern __constant__ double tRef_E[1], tRef_I[1];
extern __constant__ unsigned int nE[1];