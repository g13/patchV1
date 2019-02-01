#include <chrono>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "DIRECTIVE.h"
#include <random>
#include "CONST.h"
#include "condShape.h"
#include <cassert>
/* === try when cuRAND's cpu result is consistent with device result === #include "curand.h" */

using namespace std::chrono;

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

void cpu_version(int networkSize, double dInputE, double dInputI, unsigned int nstep, double dt, unsigned int nE, double preMat0[], double vinit[], double firstInputE[], double firstInputI[], unsigned long long seed, double EffsE, double IffsE, double EffsI, double IffsI, std::string theme, double inputRateE, double inputRateI, ConductanceShape &condE, ConductanceShape &condI);
