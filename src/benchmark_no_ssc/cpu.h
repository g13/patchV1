#include <chrono>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "LIF_inlines.h"
#include "DIRECTIVE.h"
#include "CONST.h"
#include "condShape.h"
/* === RAND === #include "curand.h" */

using namespace std::chrono;

/* === RAND === Uncomment all RANDs if curand can ensure cpu and gpu generate the same rands
    int set_input_time(double *inputTime, double dt, double rate, double &leftTimeRate, double &lastNegLogRand, curandGenerator_t &randGen) {
        int i = 0;
        double tau, dTau, negLogRand;
        tau = (lastNegLogRand - leftTimeRate)/rate;
        if (tau > dt) {
            leftTimeRate += (dt * rate);
            return i;
        } else do {
            inputTime[i] = tau;
            curandGenerateUniformDouble(randGen, &negLogRand, 1);
            negLogRand = -log(negLogRand);        
            dTau = negLogRand/rate;
            tau += dTau;
            i++;
            if (i == MAX_FFINPUT_PER_DT) {
                printf("exceeding max input per dt %i\n", 10);
                break;
            }
        } while (tau <= dt);
        lastNegLogRand = negLogRand;
        leftTimeRate = (dt - tau + dTau) * rate;
        return i;
    }
*/

void evolve_g(ConductanceShape &cond, double *g, double *h, double *f, double *inputTime, int nInput, double dt, unsigned int ig) {
    cond.decay_conductance(g, h, dt, ig); 
    for (int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, *f, dt-inputTime[i], ig);
    }
}

struct cpu_LIF {
    double v, v0;
    // type variable
    double tBack, tsp;
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
    void compute_pseudo_v0(double dt);
	void compute_v(double dt);
    double compute_spike_time(double dt);
};

double cpu_step(cpu_LIF* lif, double dt, double tRef, unsigned int id, double gE) {
    lif->tsp = -1.0f;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            lif->compute_pseudo_v0(dt);
            lif->tBack = -1.0f;
        }
        lif->runge_kutta_2(dt);
        while (lif->v > vT && lif->tBack < 0.0f) {
            // crossed threshold
            if (lif->v > vE) {
                printf("#%i exc conductance is too high %f\n", id, gE);
            }

            lif->tsp = lif->compute_spike_time(dt); 
            lif->tBack = lif->tsp + tRef;
            //printf("neuron #%i fired initially\n", id);
            //assert(lif->tBack > 0);
            if (lif->tBack < dt) {
                // refractory period ended during dt
                lif->compute_v(dt);
                lif->tBack = -1.0f;
                if (lif->v > vT) {
                    printf("multiple spike in one time step, only the last spike is counted, refractory period = %f ms, dt = %f\n", tRef, dt);
                    //assert(lif->v <= vT);
                }
            }
        }
    } 
    if (lif->tBack >= dt) {
        // during refractory period
        lif->reset_v(); 
        lif->tBack -= dt;
    } 
    return lif->tsp;
}

void cpu_LIF::runge_kutta_2(double dt) {
    double fk0 = eval0(v0);
    double fk1 = eval1(v0 + dt*fk0);
    v = v0 + dt*(fk0+fk1)/2.0f;
}

double cpu_LIF::compute_spike_time(double dt) {
    return (vT-v0)/(v-v0)*dt;
}

void cpu_LIF::compute_v(double dt) {
    v = compute_v1(dt, a0, b0, a1, b1, vL, tBack);
}

void cpu_LIF::compute_pseudo_v0(double dt) {
    v0 = (vL-tBack*(b0 + b1 - a1*b0*dt)/2.0f)/(1.0f+tBack*(-a0 - a1 + a1*a0*dt)/2.0f);
}

void cpu_LIF::set_p0(double gE, double gI, double gL ) {
    a0 = get_a(gE, gI, gL);
    b0 = get_b(gE, gI, gL); 
}

void cpu_LIF::set_p1(double gE, double gI, double gL) {
    a1 = get_a(gE, gI, gL);
    b1 = get_b(gE, gI, gL); 
}

double cpu_LIF::eval0(double _v) {
    return eval_LIF(a0,b0,_v);
}
double cpu_LIF::eval1(double _v) {
    return eval_LIF(a1,b1,_v);
}

void cpu_LIF::reset_v() {
    v = vL;
}


void cpu_version(int networkSize, /* === RAND === flatRate */int _nInput, int nskip, unsigned int nstep, double dt, unsigned int nE, double s, /* === RAND === unsigned long long seed, */ double ffsE, double ffsI) {
    unsigned int ngTypeE = 2;
    unsigned int ngTypeI = 1;
    double gL, tRef;
    double *v = new double[networkSize];
    double *gE = new double[networkSize*ngTypeE];
    double *gI = new double[networkSize*ngTypeI];
    double *hE = new double[networkSize*ngTypeE];
    double *hI = new double[networkSize*ngTypeI];
    double *fE = new double[networkSize*ngTypeE];
    double *fI = new double[networkSize*ngTypeI];
    double *spikeTrain = new double[networkSize];
    double *preMat = new double[networkSize*networkSize];
    std::ofstream v_file, spike_file, gE_file, gI_file;
    v_file.open("v_CPU.bin", std::ios::out|std::ios::binary);
    spike_file.open("s_CPU.bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_CPU.bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_CPU.bin", std::ios::out|std::ios::binary);
    double *inputTime = new double[networkSize*_nInput];
    /* === RAND === 
        curandGenerator_t *randGen = new curandGenerator_t[networkSize];
        int *nInput = new int[networkSize];
    */
	/* not used if not RAND */
    double *logRand = new double[networkSize];
    double *lTR = new double[networkSize];
    double riseTimeE[2] = {1.0f, 5.0f}; // ms
    double riseTimeI[1] = {1.0f};
    double decayTimeE[2] = {3.0f, 80.0f};
    double decayTimeI[1] = {5.0f};
    ConductanceShape condE(riseTimeE, decayTimeE, ngTypeE);
    ConductanceShape condI(riseTimeI, decayTimeI, ngTypeI);

    printf("cpu version started\n");
	/* === RAND ===
		high_resolution_clock::time_point rStart = timeNow();
		for (unsigned int i = 0; i < networkSize; i++) {
			curandCreateGeneratorHost(&randGen[i], CURAND_RNG_PSEUDO_MRG32K3A);
			curandSetPseudoRandomGeneratorSeed(randGen[i], seed + i);
			curandGenerateUniformDouble(randGen[i], &(logRand[i]), 1);
			logRand[i] = -log(logRand[i]);
			lTR[i] = 0.0f;
			printf("\r initializing curand Host generators %3.1f%%", 100.0f*float(i + 1) / networkSize);
		}
		double rTime = static_cast<double>(duration_cast<microseconds>(timeNow() - rStart).count());
		printf(" cost: %fms\n", rTime/1000.0f);
	*/
    cpu_LIF **lif = new cpu_LIF*[networkSize];
	high_resolution_clock::time_point iStart = timeNow();
	for (unsigned int i=0; i<networkSize; i++) {
		v[i] = 0.0f;
		lif[i] = new cpu_LIF(v[i]);
		for (int j = 0; j < networkSize; j++) {
			preMat[i*networkSize + j] = s;
		}
	}
	double iTime = static_cast<double>(duration_cast<microseconds>(timeNow() - iStart).count());
	printf("initialization cost: %fms\n", iTime/1000.f);

    for (int ig = 0; ig < ngTypeE; ig++) {
        for (unsigned int i = 0; i < networkSize; i++) {
            unsigned int gid = ig*networkSize + i;
            gE[gid] = 0.0f;
            hE[gid] = 0.0f;
            fE[gid] = ffsE;
        }
    }
    for (int ig = 0; ig < ngTypeI; ig++) {
        for (unsigned int i = 0; i < networkSize; i++) {
            unsigned int gid = ig*networkSize + i;
            gI[gid] = 0.0f;
            hI[gid] = 0.0f;
            fI[gid] = ffsI;
        }
    }
    //printf("cpu initialized\n");
    high_resolution_clock::time_point start = timeNow();
    v_file.write((char*)v, networkSize * sizeof(double));
    gE_file.write((char*)gE, networkSize * ngTypeE * sizeof(double));
    gI_file.write((char*)gI, networkSize * ngTypeI * sizeof(double));
	/* === RAND ===
		int inputEvents = 0;
	*/
    int outputEvents = 0;
    bool InhFired = false;
    double vTime = 0.0f;
	double gTime = 0.0f;
    int iskip = 1;
    int nInput = _nInput;
    for (unsigned int istep=0; istep<nstep; istep++) {
		if (iskip % nskip == 0) {
			nInput = _nInput;
		}
		else {
			nInput = 0;
		}
        high_resolution_clock::time_point vStart = timeNow();
        for (unsigned int i=0; i<networkSize; i++) {
            lif[i]->v0 = lif[i]->v;
            if (i<nE) {
                gL = gL_E;
                tRef = tRef_E;
            } else {
                gL = gL_I;
                tRef = tRef_I;
            }
            // get a0, b0
            double gE_t, gI_t;
            gE_t = 0.0f;
            #pragma unroll
            for (int ig = 0; ig < ngTypeE; ig++) {
                int gid = networkSize*ig + i;
                gE_t += gE[gid];
            }
            gI_t = 0.0f;
            #pragma unroll
            for (int ig = 0; ig < ngTypeI; ig++) {
                int gid = networkSize*ig + i;
                gI_t += gI[gid];
            }
            lif[i]->set_p0(gE_t, gI_t, gL);
			/* === RAND === #ifdef TEST_WITH_MANUAL_FFINPUT */
                #pragma unroll
                for (int iInput = 0; iInput < nInput; iInput++) {
                    inputTime[i*_nInput + iInput] = (iInput + double(i)/networkSize)*dt/_nInput;
                }
                // not used if not RAND
                logRand[i] = 1.0f;
                lTR[i] = 0.0f;
			/* === RAND === 
				#else
					nInput[i] = set_input_time(&(inputTime[i*MAX_FFINPUT_PER_DT]), dt, flatRate, lTR[i], logRand[i], randGen[i]);
					inputEvents += nInput[i];
				#endif
			*/
            /* evolve g to t+dt with ff input only */
            gE_t = 0.0f;
            #pragma unroll
            for (int ig=0; ig<ngTypeE; ig++) {
                unsigned int gid = networkSize*ig + i;
                evolve_g(condE, &(gE[gid]), &(hE[gid]), &(fE[gid]), &(inputTime[i*_nInput]), nInput, dt, ig);
                gE_t += gE[gid];
            }
            gI_t = 0.0f; 
            // no feed-forward inhibitory input (setting nInput = 0)
            #pragma unroll
            for (int ig=0; ig<ngTypeI; ig++) {
                unsigned int gid = networkSize*ig + i;
                evolve_g(condI, &(gI[gid]), &(hI[gid]), &(fI[gid]), &(inputTime[i*_nInput]), 0, dt, ig);
                gI_t += gI[gid];
            }
            lif[i]->set_p1(gE_t, gI_t, gL);
            spikeTrain[i] = cpu_step(lif[i], dt, tRef, i, gE_t);
            v[i] = lif[i]->v;
        }
        vTime += static_cast<double>(duration_cast<microseconds>(timeNow()-vStart).count());
        high_resolution_clock::time_point gStart = timeNow();
        for (unsigned int i=0; i<networkSize; i++) {
            double g_end, h_end;
            #ifndef NAIVE
            if (spikeTrain[i] > 0.0f) {
            #endif
                outputEvents ++;
                if (i < nE) {
                    //printf("exc-%i fired\n", i);
                    #pragma unroll
                    for (int ig=0; ig<ngTypeE; ig++) {
                        g_end = 0.0;
                        h_end = 0.0;
                        #ifdef NAIVE
                        if (spikeTrain[i] > 0.0f) {
                        #endif
                            condE.compute_single_input_conductance(&g_end, &h_end, 1.0f, dt-lif[i]->tsp, ig);
                        #ifdef NAIVE
                        }
                        #endif
                        for (int ii = 0; ii < networkSize; ii++) {
                            int gid = networkSize*ig+ii;
                            gE[gid] += g_end * preMat[i*networkSize + ii];
                            hE[gid] += h_end * preMat[i*networkSize + ii];
                        }
                    }
                } else {
                    InhFired = true;
                    #pragma unroll
                    for (int ig=0; ig<ngTypeI; ig++) {
                        g_end = 0.0;
                        h_end = 0.0;
                        #ifdef NAIVE
                        if (spikeTrain[i] > 0.0f) {
                        #endif
                            condI.compute_single_input_conductance(&g_end, &h_end, 1.0f, dt-lif[i]->tsp, ig);
                        #ifdef NAIVE
                        }
                        #endif
                        for (int ii = 0; ii < networkSize; ii++) {
                            int gid = networkSize*ig+ii;
                            gI[gid] += g_end * preMat[i*networkSize + ii];
                            hI[gid] += h_end * preMat[i*networkSize + ii];
                        }
                    }
                }
            #ifndef NAIVE
            }
            #endif
        }
        gTime += static_cast<double>(duration_cast<microseconds>(timeNow()-gStart).count());
        v_file.write((char*)v, networkSize * sizeof(double));
        spike_file.write((char*)spikeTrain, networkSize * sizeof(int));
        gE_file.write((char*)gE, networkSize * ngTypeE * sizeof(double));
        gI_file.write((char*)gI, networkSize * ngTypeI * sizeof(double));
        printf("\r stepping %3.1f%%", 100.0f*float(istep+1)/nstep);
        iskip++;
		//printf("gE0 = %f, v = %f \n", gE[0], v[0]);
    }
    printf("\n");
    if (InhFired) {
        printf("    inh fired\n");
    }
    printf("input events rate %fkHz\n", /* === RAND === float(inputEvents)/(dt*nstep*networkSize)*/ float(_nInput)/(dt*nskip));
    printf("output events rate %fHz\n", float(outputEvents)*1000.0f/(dt*nstep*networkSize));
    auto cpuTime = duration_cast<microseconds>(timeNow()-start).count();
    printf("cpu version time cost: %3.1fms\n", static_cast<double>(cpuTime)/1000.0f);
    printf("compute_V cost: %fms\n", vTime/1000.0f);
    printf("recal_G cost: %fms\n", gTime/1000.0f);
    /* Cleanup */
    printf("Cleaning up\n");
    if (v_file.is_open()) v_file.close();
    if (spike_file.is_open()) spike_file.close();
    if (gE_file.is_open()) gE_file.close();
    if (gI_file.is_open()) gI_file.close();
    delete []v;
    delete []gE;
    delete []gI;
    delete []hE;
    delete []hI;
    delete []fE;
    delete []fI;
    delete []preMat;
    delete []spikeTrain;
    /* === RAND === delete []randGen; */
    for (unsigned int i=0; i<networkSize; i++) {
        delete []lif[i];
    }
    delete []lif;

    delete []inputTime;
	/* === RAND === 
		delete []nInput;
    */
	delete []logRand;
	delete []lTR;
    printf("Memories freed\n");
}
