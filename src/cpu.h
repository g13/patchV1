#include <chrono>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "DIRECTIVE.h"
#include "CONST.h"
#include "condShape_cpu.h"
#include <cassert>
/* === RAND === #include "curand.h" */

using namespace std::chrono;

/* === RAND === Uncomment all RANDs if curand can ensure cpu and gpu generate the same rands
    int set_input_time(double *inputTime, double dt, double rate, double &leftTimeRate, double &lastNegLogRand, curandGenerator_t &randGen);
*/

struct cpu_LIF {
    double v, v0, v_hlf;
    // type variable
    double tBack, tsp;
    bool correctMe;
    unsigned int spikeCount;
    double a1, b1;
    double a0, b0;
    cpu_LIF(double _v0): v(_v0) {
        tBack = -1.0f;
    }; 
    void runge_kutta_2(double dt);
    void set_p0(double _gE, double _gI, double _gL);
    void set_p1(double _gE, double _gI, double _gL);
    double eval0(double _v);
    double eval1(double _v);
    void reset_v();
    void compute_pseudo_v0(double dt, double pdt0);
	void compute_v(double dt, double pd0);
    void compute_spike_time(double dt, double pdt0);
};

void cpu_version(int networkSize, /* === RAND === flatRate */double dInput, unsigned int nstep, double dt, unsigned int nE, double preMat0[], double vinit[], double firstInput[], /* === RAND === unsigned long long seed, */ double ffsE, double ffsI, std::string theme, double inputRate);
