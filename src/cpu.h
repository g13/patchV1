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
    double v, v0, v_hlf;
    // type variable
    double tBack, tsp;
    bool correctMe;
    int spikeCount;
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
    lif->tsp = dt;
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
                lif->v = vE;
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

double cpu_dab(cpu_LIF* lif, double dt, double tRef, unsigned int id, double gE) {
    lif->tsp = dt;
    lif->spikeCount = 0;
    // not in refractory period
    if (lif->tBack < dt) {
        // return from refractory period
        if (lif->tBack > 0.0f) {
            lif->compute_pseudo_v0(dt);
            lif->tBack = -1.0f;
        }

        lif->runge_kutta_2(dt);
        if (lif->v > vT) {
            // crossed threshold

            if (lif->v > vE) {
                printf("#%i exc conductance is too high %f, v = %f\n", id, gE_t, lif->v);
                lif->v = vE;
            }
            lif->tsp = lif->compute_spike_time(dt); 
            // dabbing not commiting, doest not reset v or recored tBack, TBD by spike correction.
        }
    } else {
        // during refractory period
        lif->reset_v(); 
        lif->tBack -= dt;
        lif->correctMe = false;
    } 
    return lif->tsp;
}

void cpu_LIF::runge_kutta_2(double dt) {
    double fk0 = eval0(v0);
    v_hlf = v0 + dt*fk0;
    double fk1 = eval1(v_hlf);
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
    runge_kutta_2(dt);
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

unsigned int find1stAfter(double spikeTrain[], unsigned int n, double min) {
    // in case no spike i=0
    unsigned int i = 0;
    for (int j=0; j<n; j++) {
        if (spikeTrain[j] < 0.0f) continue;
        if (spikeTrain[j] < min) {
            min = spikeTrain[j];
            i = j;
        }
    }
    return i;
}

// Spike-spike correction
void cpu_ssc(cpu_LIF* lif, double v[], double gE0[], double gI0[], double hE0[], double hI0[], double gE1[], double gI1[], double hE1[], double hI1[], double preMat[], unsigned int networkSize, ConductanceShape condE, ConductanceShape condI, int ngTypeE, int ngTypeI, double inputTime[], int nInput, double wSpikeTrain[], double spikeTrain[], unsigned int idTrain[], unsigned int nE, double dt) {
    unsigned int n = 0;
    for (i=0; i<networkSize; i++) {
        if (lif[i]->tsp < dt) {
            wSpikeTrain[n] = lif[i]->tsp;
            idTrain[n] = i;
            n++;
        }
    }
    double ddt = dt; // varying dt
    double *g_end, *h_end;
    if (ngTypeE > ngTypeI) {
        g_end = new double[ngTypeE];
        h_end = new double[ngTypeE];
    } else {
        g_end = new double[ngTypeI];
        h_end = new double[ngTypeI];
    }
    while (n>0) {
        unsigned int head = find1stAfter(wSpikeTrain, n, dt);
        // the first spike is accurate and set to output spikeTrain
        unsigned int i = idTrain[head];
        spikeTrain[i] = wSpikeTrain[head];
        ConductanceShape *cond;
        int ngType;
        double *g, *h;
        double tRef;
        if (i < nE) {
            cond = &condE;
            ngType = ngTypeE;
            g = gE1;
            tRef = tRef_E;
        } else {
            cond = &condI;
            ngType = ngTypeI;
            g = gI1;
            tRef = tRef_I;
        }
        lif[i]->spikeCount++;
        if (lif[i]->spikeCount > 1) {
            printf("multiple spikes in one dt, consider smaller time step than %fms\n", dt);
        }
        lif[i]->tBack = lif[i]->tsp + tRef;
        if (lif[i]->tBack < dt) {
            lif[i]->compute_v(dt);
        } else {
            lif[i]->reset_v();
            lif[i]->tBack -= dt;
            lif[i]->correctMe = false;
        }
        v[i] = lif[i]->v;
        // rerun rk2 to fill-up the wSpikeTrain
        ii = 0;
        pdt = lif[i]->tsp;
        ddt = dt - pdt;
        for (int ig = 0; ig<ngType; ig++) {
            cond->compute_single_input_conductance(&(g_end[ig]), &(h_end[ig]), 1.0f, ddt, ig)
        }
        for (int j=0; j<networkSize; j++) {
            // new cortical input
            for (int ig=0; ig<ngType; ig++) {
                unsigned int gid = networkSize*ig + j;
                g[gid] += g_end[ig] * preMat[i*networkSize + j];
                h[gid] += h_end[ig] * preMat[i*networkSize + j];
            }
            // if not in refractory period
            if (lif[j]->correctMe) {
                if (j<nE) {
                    gL = gL_E;
                    tRef = tRef_E;
                } else {
                    gL = gL_I;
                    tRef = tRef_I;
                }
                double gE_t, gI_t;
                gE_t = 0.0f;
                int iInput = 0;
                while (inputTime[j*nInput+iInput] < pdt && iInput < nInput) {
                    iInput++;
                }
                #pragma unroll
                for (int ig=0; ig<ngTypeE; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    evolve_g(condE, &(gE0[gid]), &(hE0[gid]), &(fE[gid]), &(inputTime[j*nInput]), iInput, pdt, ig);
                    gE_t += gE0[gid];
                }
                // no feed-forward inhibitory input (setting nInput = 0)
                gI_t = 0.0f; 
                #pragma unroll
                for (int ig=0; ig<ngTypeI; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    evolve_g(condI, &(gI0[gid]), &(hI0[gid]), &(fI[gid]), &(inputTime[j*nInput]), 0, pdt, ig);
                    gI_t += gI0[gid];
                }
                lif[j]->set_p0(gE_t, gI_t, gL);
                
                gE_t = 0.0f;
                #pragma unroll
                for (int ig = 0; ig < ngTypeE; ig++) {
                    int gid = networkSize*ig + j;
                    gE_t += gE1[gid];
                }
                gI_t = 0.0f;
                #pragma unroll
                for (int ig = 0; ig < ngTypeI; ig++) {
                    int gid = networkSize*ig + j;
                    gI_t += gI1[gid];
                }
                lif[j]->set_p1(gE_t, gI_t, gL);
                // dab
                cpu_dab(lif[j], ddt, tRef, j, gE_t);
                // put back to [0,dt]
                lif[j]->tsp += pdt;
                if (lif[j]->tsp < dt) {
                    idTrain[ii] = j;
                    wSpikeTrain[ii] = lif[j]->tsp;
                    ii++;
                }
            }
        }
        n--;
    }
    delete []g_end;
    delete []h_end;
}

void cpu_version(int networkSize, /* === RAND === flatRate */int nInput, unsigned int nstep, double dt, unsigned int nE, double s, /* === RAND === unsigned long long seed, */ double ffsE, double ffsI) {
    unsigned int ngTypeE = 2;
    unsigned int ngTypeI = 1;
    double gL, tRef;
    double *v1 = new double[networkSize];
    double *v2 = new double[networkSize];
    double *v_current, *v_old;
    double *gE0 = new double[networkSize*ngTypeE];
    double *gI0 = new double[networkSize*ngTypeI];
    double *gE1 = new double[networkSize*ngTypeE];
    double *gI1 = new double[networkSize*ngTypeI];
    double *gE_current, *gI_current, *gE_old, *gI_old;
    double *hE0 = new double[networkSize*ngTypeE];
    double *hI0 = new double[networkSize*ngTypeI];
    double *hE1 = new double[networkSize*ngTypeE];
    double *hI1 = new double[networkSize*ngTypeI];
    double *hE_current, *hI_current, *hE_old, *hI_old;
    double *fE = new double[networkSize*ngTypeE];
    double *fI = new double[networkSize*ngTypeI];
    double *dv = new double[networkSize];
    double *v_hlf = new double[networkSize];
    double *a0 = new double[networkSize];
    double *b0 = new double[networkSize];
    double *a1 = new double[networkSize];
    double *b1 = new double[networkSize];
    double *spikeTrain = new double[networkSize];
    double *preMat = new double[networkSize*networkSize];
    double *wSpikeTrain = new double[networkSize];
    unsigned int *idTrain = new unsigned int[networkSize];
    std::ofstream v_file, spike_file, gE_file, gI_file;
    v_file.open("v_CPU.bin", std::ios::out|std::ios::binary);
    spike_file.open("s_CPU.bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_CPU.bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_CPU.bin", std::ios::out|std::ios::binary);
    double *inputTime = new double[networkSize*nInput];
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
        v1[i] = 0.0f;
        v2[i] = 0.0f;
        lif[i] = new cpu_LIF(v1[i]);
        lTR[i] = 0.0f;
        for (int j=0; j<networkSize; j++) {
            preMat[i*networkSize+j] = s;
        }
    }
	double iTime = static_cast<double>(duration_cast<microseconds>(timeNow() - iStart).count());
	printf("initialization cost: %fms\n", iTime/1000.f);

    for (int ig = 0; ig < ngTypeE; ig++) {
        for (unsigned int i = 0; i < networkSize; i++) {
            unsigned int gid = ig*networkSize + i;
            gE0[gid] = 0.0f;
            hE0[gid] = 0.0f;
            fE[gid] = ffsE;
        }
    }
    for (int ig = 0; ig < ngTypeI; ig++) {
        for (unsigned int i = 0; i < networkSize; i++) {
            unsigned int gid = ig*networkSize + i;
            gI0[gid] = 0.0f;
            hI0[gid] = 0.0f;
            fI[gid] = ffsI;
        }
    }
    //printf("cpu initialized\n");
    high_resolution_clock::time_point start = timeNow();
    v_file.write((char*)v1, networkSize * sizeof(double));
    gE_file.write((char*)gE0, networkSize * ngTypeE * sizeof(double));
    gI_file.write((char*)gI0, networkSize * ngTypeI * sizeof(double));
	/* === RAND ===
		int inputEvents = 0;
	*/
    int outputEvents = 0;
    bool InhFired = false;
    double vTime = 0.0f;
    double gTime = 0.0f;
    for (unsigned int istep=0; istep<nstep; istep++) {
        if (istep%2 == 0) {
            v_current = v2;
            v_old = v1;
            gE_current = gE1;
            hE_current = hE1;
            gE_old = gE0;
            hE_old = hE0;
            gI_current = gI1;
            hI_current = hI1;
            gI_old = gI0;
            hI_old = hI0;
        } else {
            v_current = v1;
            v_old = v2;
            gE_current = gE0;
            hE_current = hE0;
            gE_old = gE1;
            hE_old = hE1;
            gI_current = gI0;
            hI_current = hI0;
            gI_old = gI1;
            hI_old = hI1;
        }

        high_resolution_clock::time_point vStart = timeNow();
        bool spiked = false;
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
                gE_t += gE_old[gid];
                gE_current[gid] = gE_old[gid];
                hE_current[gid] = hE_old[gid];
            }
            gI_t = 0.0f;
            #pragma unroll
            for (int ig = 0; ig < ngTypeI; ig++) {
                int gid = networkSize*ig + i;
                gI_t += gI_old[gid];
                gI_current[gid] = gI_old[gid];
                hI_current[gid] = hI_old[gid];
            }
            lif[i]->set_p0(gE_t, gI_t, gL);
			/* === RAND === #ifdef TEST_WITH_MANUAL_FFINPUT */
                #pragma unroll
                for (int iInput = 0; iInput < nInput; iInput++) {
                    inputTime[iInput*nInput] = (iInput + double(i)/networkSize)*dt/nInput;
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
                evolve_g(condE, &(gE_current[gid]), &(hE_current[gid]), &(fE[gid]), &(inputTime[i*nInput]), nInput, dt, ig);
                gE_t += gE_current[gid];
            }
            gI_t = 0.0f; 
            // no feed-forward inhibitory input (setting nInput = 0)
            #pragma unroll
            for (int ig=0; ig<ngTypeI; ig++) {
                unsigned int gid = networkSize*ig + i;
                evolve_g(condI, &(gI_current[gid]), &(hI_current[gid]), &(fI[gid]), &(inputTime[i*nInput]), 0, dt, ig);
                gI_t += gI_current[gid];
            }
            lif[i]->set_p1(gE_t, gI_t, gL);
            // rk2 step
            spikeTrain[i] = cpu_dab(lif[i], dt, tRef, /*the last 2 args are for deugging*/ i, gE_t);
            if (spikeTrain[i] < dt) spiked = true;
            if (lif[i]->v < vI) {
                printf("#%i inh conductance is too high %f, v = %f\n", i, gI_t, lif->v);
                lif[i]->v = vI;
            }   
            v_current[i] = lif[i]->v;
        }
        vTime += static_cast<double>(duration_cast<microseconds>(timeNow()-vStart).count());
            // spike-spike correction
        if (spiked) {
            high_resolution_clock::time_point sStart = timeNow();
            #ifdef TEST_SCC
                cpu_ssc(lif, v_current, gE_old, gI_old, hE_old, hI_old, gE_current, gI_current, preMat, networkSize, condE, condI, ngTypeE, ngTypeI, inputTime, nInput, wSpikeTrain, spikeTrain, idTrain, nE, dt); 
            #endif
            sTime += static_cast<double>(duration_cast<microseconds>(timeNow()-vStart).count());
        }
        /* recal_G
        high_resolution_clock::time_point gStart = timeNow();
        for (unsigned int i=0; i<networkSize; i++) {
            double g_end, h_end;
            #ifndef NAIVE
            if (spikeTrain[i] < dt) {
            #endif
                outputEvents ++;
                if (i < nE) {
                    //printf("exc-%i fired\n", i);
                    #pragma unroll
                    for (int ig=0; ig<ngTypeE; ig++) {
                        #ifdef NAIVE
                        g_end = 0.0;
                        h_end = 0.0;
                        if (spikeTrain[i] < dt) {
                        #endif
                            condE.compute_single_input_conductance(&g_end, &h_end, 1.0f, dt-lif[i]->tsp, ig);
                        #ifdef NAIVE
                        }
                        #endif
                        for (int ii = 0; ii < networkSize; ii++) {
                            int gid = networkSize*ig+ii;
                            gE_current[gid] += g_end * preMat[i*networkSize + ii];
                            hE_current[gid] += h_end * preMat[i*networkSize + ii];
                        }
                    }
                } else {
                    InhFired = true;
                    #pragma unroll
                    for (int ig=0; ig<ngTypeI; ig++) {
                        #ifdef NAIVE
                        g_end = 0.0;
                        h_end = 0.0;
                        if (spikeTrain[i] < dt) {
                        #endif
                            condI.compute_single_input_conductance(&g_end, &h_end, 1.0f, dt-lif[i]->tsp, ig);
                        #ifdef NAIVE
                        }
                        #endif
                        for (int ii = 0; ii < networkSize; ii++) {
                            int gid = networkSize*ig+ii;
                            gI_current[gid] += g_end * preMat[i*networkSize + ii];
                            hI_current[gid] += h_end * preMat[i*networkSize + ii];
                        }
                    }
                }
            #ifndef NAIVE
            }
            #endif
        }
        gTime += static_cast<double>(duration_cast<microseconds>(timeNow()-gStart).count());
        */
        v_file.write((char*)v_current, networkSize * sizeof(double));
        spike_file.write((char*)spikeTrain, networkSize * sizeof(int));
        gE_file.write((char*)gE_current, networkSize * ngTypeE * sizeof(double));
        gI_file.write((char*)gI_current, networkSize * ngTypeI * sizeof(double));
        printf("\r stepping %3.1f%%", 100.0f*float(istep+1)/nstep);
    }
    printf("\n");
    if (InhFired) {
        printf("    inh fired\n");
    }
    printf("input events rate %fkHz\n", /* === RAND === float(inputEvents)/(dt*nstep*networkSize)*/ float(nInput/dt));
    printf("output events rate %fHz\n", float(outputEvents)*1000.0f/(dt*nstep*networkSize));
    auto cpuTime = duration_cast<microseconds>(timeNow()-start).count();
    printf("cpu version time cost: %3.1fms\n", static_cast<double>(cpuTime)/1000.0f);
    printf("compute_V cost: %fms\n", vTime/1000.0f);
    printf("recal_G cost: %fms\n", gTime/1000.0f);
    #ifdef TEST_SCC
    printf("correct_spike cost: %fms\n", sTime/1000.0f);
    #endif
    /* Cleanup */
    printf("Cleaning up\n");
    if (v_file.is_open()) v_file.close();
    if (spike_file.is_open()) spike_file.close();
    if (gE_file.is_open()) gE_file.close();
    if (gI_file.is_open()) gI_file.close();
    delete []v1;
    delete []v2;
    delete []gE0;
    delete []gI0;
    delete []gE1;
    delete []gI1;
    delete []hE0;
    delete []hI0;
    delete []hE1;
    delete []hI1;
    delete []fE;
    delete []fI;
    delete []preMat;
    delete []spikeTrain;
    delete []wSpikeTrain;
    delete []idTrain;
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
