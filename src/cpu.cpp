#include "cpu.h"
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

void evolve_g(cConductanceShape &cond, double *g, double *h, double *f, double *inputTime, int nInput, double dt, double dt0, unsigned int ig) {
    // dt0: with respect to the time frame of inputTime (0)
    // dt: with respect to time frame of actual integration 
    cond.decay_conductance(g, h, dt, ig);  
    for (int i=0; i<nInput; i++) {
        cond.compute_single_input_conductance(g, h, *f, dt0-inputTime[i], ig);
    }
}

double cpu_dab(cpu_LIF* lif, double pdt0, double dt, double pdt, double dt0, double tRef, unsigned int id, double gE, double gI) {
    // dt0: with respect to the original start of time step (0) 
    // dt: with respect to time frame of actual integration 
    // time --->
    // |      |              |           |
    // 0    pdt0 <---dt---->pdt        dt0
    // tsp is set to be relative to t=0 (of the current step instead of the beginning of the simulation)
    lif->tsp = dt0;
    lif->correctMe = true;
    // not in refractory period
    if (lif->tBack < pdt) {
        // return from refractory period
        if (lif->tBack > pdt0) {
            lif->compute_pseudo_v0(dt, pdt0);
        }
        lif->runge_kutta_2(dt);
        if (lif->v > vT) {
            // crossed threshold
            lif->compute_spike_time(dt, pdt0); 
            // dabbing not commiting, doest not reset v or recored tBack, TBD by spike correction.
        } 
    } else {
        lif->reset_v(); 
        if (lif->tBack >= dt0) {
            // during refractory period
            lif->correctMe = false;
        }
    } 
    return lif->tsp;
}

double cpu_simple(cpu_LIF* lif, double pdt0, double dt, double pdt, double dt0, double tRef, unsigned int id) {
    // dt0: with respect to the original start of time step (0) 
    // dt: with respect to time frame of actual integration 
    // time --->
    // |      |              |           |
    // 0    pdt0 <---dt---->pdt        dt0
    // tsp is set to be relative to t=0 (of the current step instead of the beginning of the simulation)
    lif->tsp = dt0;
    lif->correctMe = true;
    // not in refractory period
    if (lif->tBack < pdt) {
        // return from refractory period
        if (lif->tBack > pdt0) {
            lif->compute_pseudo_v0(dt, pdt0);
            lif->tBack = -1.0f;
        }
        lif->runge_kutta_2(dt);
        while (lif->v > vT) {
            // crossed threshold
            lif->compute_spike_time(dt, pdt0); 
            lif->spikeCount++;
            lif->tBack = lif->tsp + tRef;
            if (lif->tBack < pdt) {
                lif->compute_pseudo_v0(dt, pdt0);
				lif->runge_kutta_2(dt);
                lif->tBack = -1.0f;
            } else {
                lif->reset_v();
            }
            // dabbing not commiting, doest not reset v or recored tBack, TBD by spike correction.
        } 
    } else {
        lif->reset_v(); 
        if (lif->tBack >= dt0) {
            // during refractory period
            lif->correctMe = false;
        }
    } 
    return lif->tsp;
}

double eval_LIF(double a, double b, double v) {
	return -a * v + b;
}

double get_a(double gE, double gI, double gL) {
	return gE + gI + gL;
}

double get_b(double gE, double gI, double gL) {
	return gE * vE + gI * vI + gL * vL;
}

double compute_v1(double dt, double a0, double b0, double a1, double b1, double v, double t) {
	double A = 1.0 + (a0*a1*dt - a0 - a1) * dt / 2.0f;
	double B = (b0 + b1 - a1 * b0*dt) * dt / 2.0f;
	return (B*(t - dt) - A * v*dt) / (t - dt - A * t);
}


void cpu_LIF::runge_kutta_2(double dt) {
    double fk0 = eval0(v0);
    v_hlf = v0 + dt*fk0;
    double fk1 = eval1(v_hlf);
    v = v0 + dt*(fk0+fk1)/2.0f;
}

void cpu_LIF::compute_spike_time(double dt, double pdt0) {
    tsp = pdt0 + (vT-v0)/(v-v0)*dt;
}

void cpu_LIF::compute_v(double dt, double pdt0) {
    v = compute_v1(dt, a0, b0, a1, b1, vL, tBack-pdt0);
}

void cpu_LIF::compute_pseudo_v0(double dt, double pdt0) {
    v0 = (vL-(tBack-pdt0)*(b0 + b1 - a1*b0*dt)/2.0f)/(1.0f+(tBack-pdt0)*(-a0 - a1 + a1*a0*dt)/2.0f);
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

#ifdef RECLAIM
// Spike-spike correction
unsigned int cpu_ssc(cpu_LIF* lif[], double v[], double gE0[], double gI0[], double hE0[], double hI0[], double gE1[], double gI1[], double hE1[], double hI1[], double fE[], double fI[], double preMat[], unsigned int networkSize, cConductanceShape condE, cConductanceShape condI, int ngTypeE, int ngTypeI, double inputTime[], int nInput[], double spikeTrain[], unsigned int nE, double dt) {
    static unsigned int c2_n = 0;
    double *v0 = new double[networkSize];
    double *a1 = new double[networkSize];
    double *b1 = new double[networkSize];
    unsigned int n = 0;
    double minTsp = dt;
    unsigned int imin;
    for (unsigned int i=0; i<networkSize; i++) {
        // get the first spike, candidate to output spikeTrain
        if (dt - lif[i]->tsp > EPS) {
            if (lif[i]->tsp < minTsp) {
                minTsp = lif[i]->tsp;
                imin = i;
            }
            n++;
        }
        // collect v0 in case of reclaim
        v0[i] = lif[i]->v0;
		// reset the spikeTrain for refill REMEMBER THIS BUG!
		spikeTrain[i] = dt;
    }
    double *g_end, *h_end;
    if (ngTypeE > ngTypeI) {
        g_end = new double[ngTypeE];
        h_end = new double[ngTypeE];
    } else {
        g_end = new double[ngTypeI];
        h_end = new double[ngTypeI];
    }
    int ngType;
    double *g, *h;
    cConductanceShape *cond;
    double pdt0 = 0.0f; // old pdt
    int* iInput = new int[networkSize];
    int* lastInput = new int[networkSize]();
    double* gE = new double[networkSize*ngTypeE];
    double* hE = new double[networkSize*ngTypeE];
    double* gI = new double[networkSize*ngTypeI];
    double* hI = new double[networkSize*ngTypeI];
    unsigned int r = 0;
    unsigned int last_i;
    unsigned int corrected_n = 0;
    while (n>0) {
        //if (r>2*networkSize) {
        //    printf("fucked\n");
        //}
        // candidate index = i
        double pdt = minTsp;
        unsigned int i = imin;
        double dpdt = pdt - pdt0;
        //printf("pdt - pdt0  = %e, %i, %f, %f, %i, %i, %i\n", pdt - pdt0, i, gE0[i], gI0[i], corrected_n, r, n);
        // |      |             |             |
        // 0     pdt0<--dpdt-->pdt<---ddt---->dt
        // prepare the last cortical input if exists
        if (corrected_n > 0) {
            if (last_i < nE) {
                cond = &condE;
                ngType = ngTypeE;
            } else {
                cond = &condI;
                ngType = ngTypeI;
            }
            for (int ig = 0; ig<ngType; ig++) {
                g_end[ig] = 0.0f;
                h_end[ig] = 0.0f;
                cond->compute_single_input_conductance(&(g_end[ig]), &(h_end[ig]), 1.0f, dpdt, ig);
            }
        }
        // restep to fill-up the wSpikeTrain
        // candidate will be cancelled if another spike reclaimed before the candidate
        double xdt = pdt;
        double dxdt = dpdt;
        unsigned int ihalf = 0;
        bool passed = false;
        do {
            unsigned int in;
            do {
                in = 0;
                for (unsigned int j=0; j<networkSize; j++) {
                    if (lif[j]->correctMe) {
                        double gL, tRef;
                        if (j<nE) {
                            gL = gL_E;
                            tRef = tRef_E;
                        } else {
                            gL = gL_I;
                            tRef = tRef_I;
                        }
                        if (i!=j) {
                        // excluding neurons in refractory period, nothing new here except in advancing the conductance putatively and set_p0 that wont be used by the candidate, neuron i 
                            double gE_t = 0.0f;
                            #pragma unroll
                            for (int ig=0; ig<ngTypeE; ig++) {
                                unsigned int gid = networkSize*ig + j;
                                gE_t += gE0[gid];
                            }
                            double gI_t = 0.0f;
                            #pragma unroll
                            for (int ig=0; ig<ngTypeI; ig++) {
                                unsigned int gid = networkSize*ig + j;
                                gI_t += gI0[gid];
                            }
                            lif[j]->set_p0(gE_t, gI_t, gL);
                        }
                        // evolve g h and lastInput to xdt (if no spike is reclaimed then update g0 and h0 and lastInput)
                        // determine ext input range [pdt0, xdt]
                        iInput[j] = lastInput[j];
                        if  (iInput[j] < nInput[j]) {
                            while (inputTime[j*MAX_FFINPUT_PER_DT+iInput[j]] < xdt) {
                                iInput[j]++;
                                if (iInput[j] == nInput[j]) break;
                            }
                        }
                        double gE_t = 0.0f;
                        #pragma unroll
                        for (int ig=0; ig<ngTypeE; ig++) {
                            unsigned int gid = networkSize*ig + j;
                            gE[gid] = gE0[gid];
                            hE[gid] = hE0[gid];
                            evolve_g(condE, &(gE[gid]), &(hE[gid]), &(fE[gid]), &(inputTime[j*MAX_FFINPUT_PER_DT+lastInput[j]]), iInput[j]-lastInput[j], dxdt, xdt, ig);
                            if (corrected_n > 0) {
                                if (last_i < nE) {
                                    gE[gid] += g_end[ig] * preMat[last_i*networkSize + j];
                                    hE[gid] += h_end[ig] * preMat[last_i*networkSize + j];
                                }
                            }
                            gE_t += gE[gid];
                        }
                        // no feed-forward inhibitory input (setting nInput = 0)
                        double gI_t = 0.0f; 
                        #pragma unroll
                        for (int ig=0; ig<ngTypeI; ig++) {
                            unsigned int gid = networkSize*ig + j;
                            gI[gid] = gI0[gid];
                            hI[gid] = hI0[gid];
                            evolve_g(condI, &(gI[gid]), &(hI[gid]), &(fI[gid]), inputTime, 0, dxdt, xdt, ig);
                            if (corrected_n > 0) {
                                if (last_i >= nE) {
                                    gI[gid] += g_end[ig] * preMat[last_i*networkSize + j];
                                    hI[gid] += h_end[ig] * preMat[last_i*networkSize + j];
                                }
                            }
                            gI_t += gI[gid];
                        }
                        lif[j]->set_p1(gE_t, gI_t, gL);
                        if (i!=j) {
                            // exclude the candidate itself for updating voltage or tsp
                            lif[j]->v0 = v0[j];
                            cpu_dab(lif[j], pdt0, dxdt, xdt, dt, tRef, j, gE_t, gI_t); 
                            if (lif[j]->tsp < 0.0f) {
                                printf("#%i v = %f, %f -> %f, tB = %f\n", j, v0[j], lif[j]->v0, lif[j]->v, lif[j]->tBack);
                                assert(lif[j]->tsp >= 0.0f);
                            }
                            if (dt - lif[j]->tsp > EPS ) {
                                // ignore round off error
                                in++;
                                xdt = lif[j]->tsp;
                                dxdt = xdt - pdt0;
                                i = j;
                                passed = false;
                                break;
                            } 
                        }
                    }
                }
                if (!passed && in == 0) {
                    pdt = xdt;
                    for (unsigned int j=0; j<networkSize; j++) {
                        if (lif[j]->correctMe) {
                            a1[j] = lif[j]->a1;    
                            b1[j] = lif[j]->b1;    
                        }
                    }
                }
            } while (in > 0);
            if (!passed) {
                dxdt = (xdt-pdt0)/2;
                xdt = pdt0 + dxdt;
                passed = true;
            } 
#ifdef DEBUG 
            printf("%i halfed, pdt0 = %e, xdt = %e\n", ihalf, pdt0, xdt);
#endif
            ihalf++;
        } while (!passed);
        // if indeed no spike comes before neuron i with the smaller pdt interpolation, commit into spikeTrain
        corrected_n++;
        last_i = i;
        spikeTrain[i] = pdt;
        minTsp = dt;
        lif[i]->spikeCount++;
        double tRef;
        if (i < nE) {
			tRef = tRef_E;
        } else {
            tRef = tRef_I;
        }
        lif[i]->tBack = lif[i]->tsp + tRef;
        if (lif[i]->tBack > dt) {
            lif[i]->reset_v();
            lif[i]->correctMe = false;
            v[i] = lif[i]->v;
        }
#ifdef DEBUG
        printf("#%i v=%f, spiked at %f -> %f\n", i, lif[i]->v, lif[i]->tsp, lif[i]->tBack);
        if (lif[i]->spikeCount > 1) {
            printf("    %i times in one dt, %f", lif[i]->spikeCount, dt);
        }
#endif
        // set pdt0 for next
        pdt0 = pdt;
        // move on to next pdt period
        double ddt = dt - pdt;
        // prepare the conductance changes caused by the confirmed spike from neuron i
        if (i < nE) {
            cond = &condE;
            ngType = ngTypeE;
            g = gE1;
			h = hE1;
        } else {
            cond = &condI;
            ngType = ngTypeI;
            g = gI1;
			h = hI1;
        }
        for (int ig = 0; ig<ngType; ig++) {
            g_end[ig] = 0.0f;
            h_end[ig] = 0.0f;
            cond->compute_single_input_conductance(&(g_end[ig]), &(h_end[ig]), 1.0f, ddt, ig);
        }
        //printf("#%i, confirmed g = %f, h = %f\n", i, g_end[0], h_end[0]);
        // update v, tsp and conductance for the time interval of [pdt, dt]
        n = 0;
        for (unsigned int j=0; j<networkSize; j++) {
            // new cortical input
            for (int ig=0; ig<ngType; ig++) {
                unsigned int gid = networkSize*ig + j;
                g[gid] += g_end[ig] * preMat[i*networkSize + j];
                h[gid] += h_end[ig] * preMat[i*networkSize + j];
            }
            if (lif[j]->correctMe) {
                // commit g0 h0
                for (int ig=0; ig<ngTypeE; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gE0[gid] = gE[gid];
                    hE0[gid] = hE[gid];
                }
                for (int ig=0; ig<ngTypeI; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gI0[gid] = gI[gid];
                    hI0[gid] = hI[gid];
                }
                // commit lastInput
                lastInput[j] = iInput[j];
                double gL, tRef;
                if (j<nE) {
                    gL = gL_E;
                    tRef = tRef_E;
                } else {
                    gL = gL_I;
                    tRef = tRef_I;
                }
                // if not in refractory period
                lif[j]->v0 = lif[j]->v; // v0 is irrelevant if the neuron is coming back from spike, it will be reset in cpu_dab
                lif[j]->a0 = a1[j];
                lif[j]->b0 = b1[j];

                double gE_t = 0.0f;
                #pragma unroll
                for (int ig = 0; ig < ngTypeE; ig++) {
                    int gid = networkSize*ig + j;
                    gE_t += gE1[gid];
                }
                double gI_t = 0.0f;
                #pragma unroll
                for (int ig = 0; ig < ngTypeI; ig++) {
                    int gid = networkSize*ig + j;
                    gI_t += gI1[gid];
                }
                lif[j]->set_p1(gE_t, gI_t, gL);
                // dab
                cpu_dab(lif[j], pdt, ddt, dt, dt, tRef, j, gE_t, gI_t);
                // put back to [0,dt]
                if (lif[j]->tsp < 0.0f) {
                    printf("%i v = %f -> %f, tB = %f\n", j, lif[j]->v0, lif[j]->v, lif[j]->tBack);
                    assert(lif[j]->tsp >= 0.0f);
                }
                if (dt - lif[j]->tsp > EPS) { // equiv lif->v > vT
                    // ignore round off error
                    n++;
                    if (lif[j]->tsp < minTsp) {
                        imin = j;
                        minTsp = lif[j]->tsp;
                    }
                } else {
                    //update volts
                    v[j] = lif[j]->v;
#ifdef DEBUG
                    if (lif[j]->v > vT) {
                        printf("%i v = %f -> %f, tB = %f\n", j, lif[j]->v0, lif[j]->v, lif[j]->tBack);
                        assert(lif[j]->v <= vT);
                    }
#endif
                }
                // commits v0 for next advancement in 
                v0[j] = lif[j]->v0;
            }
        }
        r++;
#ifdef DEBUG 
        printf("%i corrected %i-%i, pdt0 = %e, dpdt = %e\n", r, corrected_n, i, pdt0, dpdt);
#endif
    }
    delete []g_end;
    delete []h_end;
    delete []iInput;
    delete []lastInput;
    delete []gE;
    delete []gI;
    delete []hE;
    delete []hI;
    delete []v0;
    delete []a1;
    delete []b1;
    //if (corrected_n > 1) {
    //    c2_n++;
    //    printf("%u corrected, more-than-2 total %u\n", corrected_n, c2_n);

    //}
    return corrected_n;
}
#else
unsigned int cpu_ssc(cpu_LIF* lif[], double v[], double gE0[], double gI0[], double hE0[], double hI0[], double gE1[], double gI1[], double hE1[], double hI1[], double fE[], double fI[], double preMat[], unsigned int networkSize, cConductanceShape condE, cConductanceShape condI, int ngTypeE, int ngTypeI, double inputTime[], int nInput[], double spikeTrain[], unsigned int nE, double dt) {
    //static unsigned int c2_n = 0;
    unsigned int in = 0;
    double* wSpikeTrain = new double[networkSize];
    double* v0 = new double[networkSize];
    unsigned int *idTrain = new unsigned int[networkSize];
    unsigned int *ns = new unsigned int[networkSize];
    double minTsp = dt;
    unsigned int imin;
    for (unsigned int i=0; i<networkSize; i++) {
        // get the first spike, candidate to output spikeTrain
        if (lif[i]->tsp < minTsp) {
            minTsp = lif[i]->tsp;
            imin = i;
        }
        // collect v0 in case of reclaim
        v0[i] = lif[i]->v0;
		// reset the spikeTrain for refill REMEMBER THIS BUG!
		spikeTrain[i] = dt;
    }
    double g_end, h_end; 
    double *gE_end = new double[ngTypeE];
    double *hE_end = new double[ngTypeE];
    double *gI_end = new double[ngTypeI];
    double *hI_end = new double[ngTypeI];
    double pdt0 = 0.0f; // old pdt
    int* iInput = new int[networkSize];
    int* lastInput = new int[networkSize]();
    unsigned int r = 0;
    unsigned int corrected_n = in;
    bool lastI, lastE;
    bool still_have_spikes = true;
    while (still_have_spikes) {
        //if (r>2*networkSize) {
        //    printf("fucked\n");
        //}
        // candidate index = i
        double pdt = minTsp;
        unsigned int i = imin;
        double dpdt = pdt - pdt0;
        //printf("pdt - pdt0  = %e, %i, %f, %f, %i, %i, %i\n", pdt - pdt0, i, gE0[i], gI0[i], corrected_n, r, n);
        // |      |             |             |
        // 0     pdt0<--dpdt-->pdt<---ddt---->dt
        // prepare the last cortical input if exists
        if (in > 0) {
            if (lastE) {
                for (int ig = 0; ig<ngTypeE; ig++) {
                    g_end = 0.0f;
                    h_end = 0.0f;
                    condE.compute_single_input_conductance(&g_end, &h_end, 1.0f, dpdt, ig);
                    gE_end[ig] = g_end;
                    hE_end[ig] = h_end;
                }
#ifdef DEBUG
                printf("gE_end = %f, %f\n", gE_end[0], gE_end[1]);
#endif
            }
            if (lastI) {
                for (int ig = 0; ig<ngTypeI; ig++) {
                    g_end = 0.0f;
                    h_end = 0.0f;
                    condI.compute_single_input_conductance(&g_end, &h_end, 1.0f, dpdt, ig);
                    gI_end[ig] = g_end;
                    hI_end[ig] = h_end;
                }
#ifdef DEBUG
                printf("gI_end = %f\n", gI_end[0]);
#endif
            }
        }
        for (unsigned int j=0; j<networkSize; j++) {
            if (lif[j]->correctMe) {
                double gL;
                if (j<nE) {
                    gL = gL_E;
                } else {
                    gL = gL_I;
                }
                if (i!=j) {
                // excluding neurons in refractory period, nothing new here except in advancing the conductance putatively and set_p0 that wont be used by the candidate, neuron i 
                    double gE_t = 0.0f;
                    #pragma unroll
                    for (int ig=0; ig<ngTypeE; ig++) {
                        unsigned int gid = networkSize*ig + j;
                        gE_t += gE0[gid];
                    }
                    double gI_t = 0.0f;
                    #pragma unroll
                    for (int ig=0; ig<ngTypeI; ig++) {
                        unsigned int gid = networkSize*ig + j;
                        gI_t += gI0[gid];
                    }
                    lif[j]->set_p0(gE_t, gI_t, gL);
                }
                // evolve g h and lastInput to xdt (if no spike is reclaimed then update g0 and h0 and lastInput)
                // determine ext input range [pdt0, pdt]
                iInput[j] = lastInput[j];
                if  (iInput[j] < nInput[j]) {
                    while (inputTime[j*MAX_FFINPUT_PER_DT+iInput[j]] < pdt) {
                        iInput[j]++;
                        if (iInput[j] == nInput[j]) break;
                    }
                }
                for (int ig=0; ig<ngTypeE; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    evolve_g(condE, &(gE0[gid]), &(hE0[gid]), &(fE[gid]), &(inputTime[j*MAX_FFINPUT_PER_DT+lastInput[j]]), iInput[j]-lastInput[j], dpdt, pdt, ig);
                    if (gE0[gid] < 0) {
                        printf("    gE#%i = %f, In: %i->%i <= %i\n", j, gE0[gid], lastInput[j], iInput[j], nInput[j]);
                    }
                }
                // no feed-forward inhibitory input (setting nInput = 0)
                for (int ig=0; ig<ngTypeI; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    evolve_g(condI, &(gI0[gid]), &(hI0[gid]), &(fI[gid]), inputTime, 0, dpdt, pdt, ig);
                    if (gI0[gid] < 0) {
                        printf("    gI#%i = %f, In: %i->%i <= %i\n", j, gI0[gid], lastInput[j], iInput[j], nInput[j]);
                    }
                }
                lastInput[j] = iInput[j];
                // add latest cortical inputs
                for (unsigned int ini = 0; ini < in; ini++) {
                    unsigned int id = idTrain[ini];
                    if (id < nE) {
                        for (int ig=0; ig<ngTypeE; ig++) {
                            unsigned int gid = networkSize*ig + j;
                            gE0[gid] += gE_end[ig] * ns[ini] * preMat[id*networkSize + j];
                            hE0[gid] += hE_end[ig] * ns[ini] * preMat[id*networkSize + j];
                            if (gE0[gid] < 0) {
                                printf("    %u gE#%u %f+=%fx%ix%f\n", id, j, gE0[gid], gE_end[ig], ns[ini], preMat[id*networkSize+j]);
                            }
                        }
                    } else {
                        for (int ig=0; ig<ngTypeI; ig++) {
                            unsigned int gid = networkSize*ig + j;
                            gI0[gid] += gI_end[ig] * ns[ini] * preMat[id*networkSize + j];
                            hI0[gid] += hI_end[ig] * ns[ini] * preMat[id*networkSize + j];
                            if (gI0[gid] < 0) {
                                printf("    %u gI#%i %f+=%fx%ix%f\n", id, j, gI0[gid], gI_end[ig], ns[ini], preMat[id*networkSize+j]);
                            }
                        }
                    }
                }
                double gE_t = 0.0f;
                for (int ig=0; ig<ngTypeE; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gE_t += gE0[gid];
                }
                double gI_t = 0.0f; 
                for (int ig=0; ig<ngTypeI; ig++) {
                    unsigned int gid = networkSize*ig + j;
                    gI_t += gI0[gid];
                }
                lif[j]->set_p1(gE_t, gI_t, gL);
            }
        }
        in = 0;
        for (unsigned int j = 0; j<networkSize; j++) {
            if (lif[j]->correctMe && i!=j) {
                double tRef;
                if (j<nE) {
                    tRef = tRef_E;
                } else {
                    tRef = tRef_I;
                }
                // exclude the candidate itself for updating voltage or tsp
                lif[j]->v0 = v0[j];
                unsigned int ns0 = lif[j]->spikeCount;
                cpu_simple(lif[j], pdt0, dpdt, pdt, dt, tRef, j); 
                if (lif[j]->tsp < 0.0f) {
                    printf("v = %f -> %f, tB = %f\n", lif[j]->v0, lif[j]->v, lif[j]->tBack);
                    assert(lif[j]->tsp >= 0.0f);
                }
                if (dt - lif[j]->tsp > EPS ) {
                    // ignore round off error
                    ns[in] = lif[j]->spikeCount - ns0;
                    wSpikeTrain[in] = pdt;
                    assert(lif[j]->tsp < pdt);
                    idTrain[in] = j;
                    in++;
                } 
            }
        }
        // if indeed no spike comes before neuron i with the smaller pdt interpolation, commit into spikeTrain
        wSpikeTrain[in]  = pdt;
        idTrain[in] = i;
        ns[in] = 1;
        in++;
#ifdef DEBUG
        printf("ssc round %i, based on #%i:\n", r, i);
#endif
        double tRef;
        if (i < nE) {
			tRef = tRef_E;
        } else {
            tRef = tRef_I;
        }
        lif[i]->spikeCount++;
        lif[i]->reset_v();
        lastI = false;
        lastE = false;
        for (unsigned int j=0; j<in; j++) {
            unsigned int id = idTrain[j];
            spikeTrain[id] = pdt;
            lif[id]->tBack = pdt + tRef;
            if (lif[id]->tBack > dt) {
                lif[id]->correctMe = false;
                v[id] = lif[id]->v;
            }
            if (id<nE) lastE = true;
            else lastI = true;
#ifdef DEBUG
            printf("    #%i v=%f, spiked at %f -> %f, %i times\n", id, lif[id]->v, lif[id]->tsp, lif[id]->tBack, ns[j]);
            printf("        a = %f -> %f, b = %f -> %f\n", lif[id]->a0, lif[id]->a1, lif[id]->b0, lif[id]->b1);
#endif
        // set pdt0 for next
        }
        corrected_n += in;
        minTsp = dt;
        pdt0 = pdt;
        // move on to next pdt period
        double ddt = dt - pdt;
        // prepare the conductance changes caused by the confirmed spike from neuron i
        if (lastE) {
            for (int ig = 0; ig<ngTypeE; ig++) {
                g_end = 0.0f;
                h_end = 0.0f;
                condE.compute_single_input_conductance(&g_end, &h_end, 1.0f, ddt, ig);
                gE_end[ig] = g_end;
                hE_end[ig] = h_end;
            }
        }
        if (lastI) {
            for (int ig = 0; ig<ngTypeI; ig++) {
                g_end = 0.0f;
                h_end = 0.0f;
                condI.compute_single_input_conductance(&g_end, &h_end, 1.0f, ddt, ig);
                gI_end[ig] = g_end;
                hI_end[ig] = h_end;
            }
        }
        //printf("#%i, confirmed g = %f, h = %f\n", i, g_end[0], h_end[0]);
        // update v, tsp and conductance for the time interval of [pdt, dt]
        still_have_spikes = false;
        for (unsigned int j=0; j<networkSize; j++) {
            // new cortical input
            for (unsigned int ini = 0; ini < in; ini++) {
                unsigned int id = idTrain[ini];
                if (id < nE) {
                    for (int ig=0; ig<ngTypeE; ig++) {
                        unsigned int gid = networkSize*ig + j;
                        gE1[gid] += gE_end[ig] * ns[ini] * preMat[id*networkSize + j];
                        hE1[gid] += hE_end[ig] * ns[ini] * preMat[id*networkSize + j];
                    }
                } else {
                    for (int ig=0; ig<ngTypeI; ig++) {
                        unsigned int gid = networkSize*ig + j;
                        gI1[gid] += gI_end[ig] * ns[ini] * preMat[id*networkSize + j];
                        hI1[gid] += hI_end[ig] * ns[ini] * preMat[id*networkSize + j];
                    }
                }
            }
            if (lif[j]->correctMe) {
                // if not in refractory period
                double gL, tRef;
                if (j<nE) {
                    gL = gL_E;
                    tRef = tRef_E;
                } else {
                    gL = gL_I;
                    tRef = tRef_I;
                }
                // for next pdt0-pdt period if exist
                v0[j] = lif[j]->v;
                lif[j]->v0 = lif[j]->v; // v0 is irrelevant if the neuron is coming back from spike, it will be reset in cpu_dab
                lif[j]->a0 = lif[j]->a1;
                lif[j]->b0 = lif[j]->b1;

                double gE_t = 0.0f;
                #pragma unroll
                for (int ig = 0; ig < ngTypeE; ig++) {
                    int gid = networkSize*ig + j;
                    gE_t += gE1[gid];
                }
                double gI_t = 0.0f;
                #pragma unroll
                for (int ig = 0; ig < ngTypeI; ig++) {
                    int gid = networkSize*ig + j;
                    gI_t += gI1[gid];
                }
                lif[j]->set_p1(gE_t, gI_t, gL);
                // dab
                cpu_dab(lif[j], pdt, ddt, dt, dt, tRef, j, gE_t, gI_t);
                // put back to [0,dt]
                if (lif[j]->tsp < 0.0f) {
                    printf("%i v = %f -> %f, tB = %f\n", j, lif[j]->v0, lif[j]->v, lif[j]->tBack);
                    assert(lif[j]->tsp >= 0.0f);
                }
                if (dt - lif[j]->tsp > EPS) { // equiv lif->v > vT
                    // ignore round off error
                    if (lif[j]->tsp < minTsp) {
                        minTsp = lif[j]->tsp;
                        imin = j;
                    }
                    still_have_spikes = true;
                } else {
                    //update volts
                    v[j] = lif[j]->v;
#ifdef DEBUG
                    if (lif[j]->v > vT) {
                        printf("%i v = %f -> %f, tB = %f\n", j, lif[j]->v0, lif[j]->v, lif[j]->tBack);
                        assert(lif[j]->v <= vT);
                    }
#endif
                }
            }
        }
        r++;
#ifdef DEBUG 
        printf("%i corrected %i-%i, pdt0 = %e, dpdt = %e\n", r, corrected_n, i, pdt0, dpdt);
#endif
    }
    delete []gE_end;
    delete []hE_end;
    delete []gI_end;
    delete []hI_end;
    delete []wSpikeTrain;
    delete []idTrain;
    delete []v0;
    delete []ns;
    delete []iInput;
    delete []lastInput;
    for (unsigned int j=0; j<networkSize; j++) {
        if (lif[j]->v > vT) {
            printf("#%i, v =  %f, %f -> %f, tB = %f\n", j, v0[j], lif[j]->v0, lif[j]->v, lif[j]->tBack);
            printf("a = %f->%f, b = %f->%f\n", lif[j]->a0, lif[j]->a1, lif[j]->b0, lif[j]->b1);
            assert(lif[j]->v <= vT);
        }
    }
    return corrected_n;
}
#endif

void cpu_version(int networkSize, /* === RAND === flatRate */double dInput, unsigned int nstep, double dt, unsigned int nE, double preMat0[], double vinit[], double firstInput[], /* === RAND === unsigned long long seed, */ double EffsE, double IffsE, double EffsI, double IffsI, std::string theme, double inputRate) {
    unsigned int ngTypeE = 2;
    unsigned int ngTypeI = 1;
    double gL, tRef;
    double *v = new double[networkSize];
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
    double *spikeTrain = new double[networkSize];
    unsigned int *nSpike = new unsigned int[networkSize];
    double *preMat = new double[networkSize*networkSize];
    double exc_input_ratio = 0.0f;
    double gEavgE = 0.0f;
    double gIavgE = 0.0f;
    double gEavgI = 0.0f;
    double gIavgI = 0.0f;
    std::ofstream p_file, v_file, spike_file, nSpike_file, gE_file, gI_file;
#ifdef RECLAIM
    p_file.open("rp_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    v_file.open("rv_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    spike_file.open("rs_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    nSpike_file.open("rn_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    gE_file.open("rgE_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    gI_file.open("rgI_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
#else
    p_file.open("p_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    v_file.open("v_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    spike_file.open("s_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    nSpike_file.open("n_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    gE_file.open("gE_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
    gI_file.open("gI_CPU" + theme + ".bin", std::ios::out|std::ios::binary);
#endif

    unsigned int nI = networkSize - nE;
    p_file.write((char*)&nE, sizeof(unsigned int));
    p_file.write((char*)&nI, sizeof(unsigned int));
    p_file.write((char*)&ngTypeE, sizeof(unsigned int));
    p_file.write((char*)&ngTypeI, sizeof(unsigned int));
    double dtmp = vL;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = vT;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = vE;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = vI;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = gL_E;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = gL_I;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = tRef_E;
    p_file.write((char*)&dtmp, sizeof(double));
    dtmp = tRef_I;
    p_file.write((char*)&dtmp, sizeof(double));
    p_file.write((char*)&nstep, sizeof(unsigned int));
    p_file.write((char*)&dt, sizeof(double));
    p_file.write((char*)&inputRate, sizeof(double));

    double *inputTime = new double[networkSize*MAX_FFINPUT_PER_DT];
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
    cConductanceShape condE(riseTimeE, decayTimeE, ngTypeE);
    cConductanceShape condI(riseTimeI, decayTimeI, ngTypeI);

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
        v[i] = vinit[i];
        lif[i] = new cpu_LIF(v[i]);
        lTR[i] = firstInput[i];
        for (int j=0; j<networkSize; j++) {
            preMat[i*networkSize+j] = preMat0[i*networkSize+j];
        }
    }
	double iTime = static_cast<double>(duration_cast<microseconds>(timeNow() - iStart).count());
	printf("initialization cost: %fms\n", iTime/1000.f);

    for (int ig = 0; ig < ngTypeE; ig++) {
        for (unsigned int i = 0; i < networkSize; i++) {
            unsigned int gid = ig*networkSize + i;
            gE0[gid] = 0.0f;
            hE0[gid] = 0.0f;
            if (i<nE) {
                fE[gid] = EffsE;
            } else {
                fE[gid] = IffsE;
            }
        }
    }
    for (int ig = 0; ig < ngTypeI; ig++) {
        for (unsigned int i = 0; i < networkSize; i++) {
            unsigned int gid = ig*networkSize + i;
            gI0[gid] = 0.0f;
            hI0[gid] = 0.0f;
            if (i<nE) {
                fI[gid] = EffsI;
            } else {
                fI[gid] = IffsI;
            }
        }
    }
    //printf("cpu initialized\n");
    high_resolution_clock::time_point start = timeNow();
    v_file.write((char*)v, networkSize * sizeof(double));
    gE_file.write((char*)gE0, networkSize * ngTypeE * sizeof(double));
    gI_file.write((char*)gI0, networkSize * ngTypeI * sizeof(double));
	int inputEvents = 0;
    unsigned int spikesE = 0;
    unsigned int spikesI = 0;
    double vTime = 0.0f;
	double wTime = 0.0f;
    double sTime = 0.0f;
    int *nInput = new int[networkSize];
	for (unsigned int istep = 0; istep < nstep; istep++) {
		if (istep % 2 == 0) {
			gE_current = gE1;
			hE_current = hE1;
			gE_old = gE0;
			hE_old = hE0;
			gI_current = gI1;
			hI_current = hI1;
			gI_old = gI0;
			hI_old = hI0;
		} else {
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
            lif[i]->spikeCount = 0;
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
                nInput[i] = 0;
                if (lTR[i] < dt) {
                    inputTime[i*MAX_FFINPUT_PER_DT] = lTR[i];
                    nInput[i]++;
                    double tmp = lTR[i] + dInput;
                    while (tmp < dt){
                        inputTime[i*MAX_FFINPUT_PER_DT + nInput[i]] = tmp;
                        nInput[i]++;
                        tmp += dInput;
                    }
                    lTR[i] = tmp - dt;
                } else {
                    lTR[i] -= dt;
                }
				inputEvents += nInput[i];
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
                evolve_g(condE, &(gE_current[gid]), &(hE_current[gid]), &(fE[gid]), &(inputTime[i*MAX_FFINPUT_PER_DT]), nInput[i], dt, dt, ig);
                gE_t += gE_current[gid];
            }
            gI_t = 0.0f; 
            // no feed-forward inhibitory input (setting nInput = 0)
            #pragma unroll
            for (int ig=0; ig<ngTypeI; ig++) {
                unsigned int gid = networkSize*ig + i;
                evolve_g(condI, &(gI_current[gid]), &(hI_current[gid]), &(fI[gid]), inputTime, 0, dt, dt, ig);
                gI_t += gI_current[gid];
            }
            lif[i]->set_p1(gE_t, gI_t, gL);
            // rk2 step
            double v0old = lif[i]->v0;
            spikeTrain[i] = cpu_dab(lif[i], 0, dt, dt, dt, tRef, i, gE_t, gI_t);
            if (lif[i]->v < vI) {
                lif[i]->v = vI;
            }
            if (lif[i]->tsp < 0.0f) {
                printf("#%i, v =  %f, %f -> %f, tB = %f\n", i, v0old, lif[i]->v0, lif[i]->v, lif[i]->tBack);
                printf("a = %f->%f, b = %f->%f\n", lif[i]->a0, lif[i]->a1, lif[i]->b0, lif[i]->b1);
                assert(lif[i]->tsp >= 0.0f);
            }
            if (dt - lif[i]->tsp > EPS) spiked = true;
            v[i] = lif[i]->v;
        }
        vTime += static_cast<double>(duration_cast<microseconds>(timeNow()-vStart).count());
            // spike-spike correction
        if (spiked) {
            //printf("spiked:\n");
            high_resolution_clock::time_point sStart = timeNow();
            unsigned int nsp = cpu_ssc(lif, v, gE_old, gI_old, hE_old, hI_old, gE_current, gI_current, hE_current, hI_current, fE, fI, preMat, networkSize, condE, condI, ngTypeE, ngTypeI, inputTime, nInput, spikeTrain, nE, dt);
            sTime += static_cast<double>(duration_cast<microseconds>(timeNow()-vStart).count());
#ifdef DEBUG
            printf("%u spikes during dt\n", nsp);
#endif
        } 
        for (unsigned int i=0; i<networkSize; i++) {
            if (i < nE) {
                spikesE += nSpike[i]; 
            } else {
                spikesI += nSpike[i]; 
            }
            nSpike[i] = lif[i]->spikeCount;
            lif[i]->tBack -= dt;
        }
		high_resolution_clock::time_point wStart = timeNow();
        v_file.write((char*)v, networkSize * sizeof(double));
        spike_file.write((char*)spikeTrain, networkSize * sizeof(double));
        nSpike_file.write((char*)nSpike, networkSize * sizeof(unsigned int));
        gE_file.write((char*)gE_current, networkSize * ngTypeE * sizeof(double));
        gI_file.write((char*)gI_current, networkSize * ngTypeI * sizeof(double));
		wTime += static_cast<double>(duration_cast<microseconds>(timeNow() - wStart).count());
#ifdef DEBUG
        printf("stepping %3.1f%%, t = %f\n", 100.0f*float(istep+1)/nstep, istep*dt);
#else
        printf("\r stepping %3.1f%%, t = %f", 100.0f*float(istep+1)/nstep, istep*dt);

#endif
        double ir = 0.0f;
        for (unsigned int i=0; i<nE; i++) {
            double sEi = 0.0f;
            for (unsigned int j=0; j<networkSize; j++) {
                sEi += preMat[i*networkSize + j] * nSpike[i];
            }
            ir += sEi;
        }
        exc_input_ratio += ir/networkSize;
		//printf("gE0 = %f, v = %f \n", gE_current[0], v[0]);
        for (unsigned int j=0; j<networkSize; j++) {
            if (j<nE) {
                for (unsigned int ig=0; ig<ngTypeE; ig++) {
                    gEavgE += gE_current[ig*networkSize + j];
                }
                for (unsigned int ig=0; ig<ngTypeI; ig++) {
                    gIavgE += gI_current[ig*networkSize + j];
                }
            } else {
                for (unsigned int ig=0; ig<ngTypeE; ig++) {
                    gEavgI += gE_current[ig*networkSize + j];
                }
                for (unsigned int ig=0; ig<ngTypeI; ig++) {
                    gIavgI += gI_current[ig*networkSize + j];
                }
            }
        }
    }
    printf("\n");
    printf("input events rate %ekHz\n", float(inputEvents)/(dt*nstep*networkSize));
    printf("exc firing rate = %eHz\n", float(spikesE)/(dt*nstep*nE)*1000.0);
    printf("inh firing rate = %eHz\n", float(spikesI)/(dt*nstep*nI)*1000.0);
    auto cpuTime = duration_cast<microseconds>(timeNow()-start).count();
    printf("cpu version time cost: %3.1fms\n", static_cast<double>(cpuTime)/1000.0f);
    printf("compute_V cost: %fms\n", vTime/1000.0f);
    printf("correct_spike cost: %fms\n", sTime/1000.0f);
	printf("writing data to disk cost: %fms\n", wTime/1000.0f);
    printf("input ratio recurrent:feedforward = %f\n", exc_input_ratio/((EffsE*nE+IffsE*nI)/networkSize*dt*nstep/dInput));
    printf("           exc,        inh\n");
    printf("avg gE = %e, %e\n", gEavgE/nstep/nE, gEavgI/nstep/nI);
    printf("avg gI = %e, %e\n", gIavgE/nstep/nE, gIavgI/nstep/nI);
    /* Cleanup */
    printf("Cleaning up\n");
    int nTimer = 2;
    p_file.write((char*)&nTimer, sizeof(int));
    p_file.write((char*)&vTime, sizeof(double));
    p_file.write((char*)&sTime, sizeof(double));
    
    if (p_file.is_open()) p_file.close();
    if (v_file.is_open()) v_file.close();
    if (spike_file.is_open()) spike_file.close();
    if (nSpike_file.is_open()) nSpike_file.close();
    if (gE_file.is_open()) gE_file.close();
    if (gI_file.is_open()) gI_file.close();
    delete []v;
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
    delete []nSpike;
    /* === RAND === delete []randGen; */
    for (unsigned int i=0; i<networkSize; i++) {
        delete []lif[i];
    }
    delete []lif;

    delete []inputTime;
	delete []logRand;
	delete []lTR;
    delete []nInput;
    printf("Memories freed\n");
}
